"""Tests for the compliance gate.

The tests that matter here are not "does a rule fire". They are the ones that
would catch a gate that *looks* compliant: an action reaching an executor
without adjudication, a deferral that fires while still blocked, a rule that
goes quiet on the case it exists for, or house policy wearing a regulator's
authority.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import pickle

import pytest

from rebound.compliance import (
    DEFAULT_RULES,
    _APPROVAL,
    AfaCeiling,
    ApprovedAction,
    Basis,
    ComplianceGate,
    ComplianceViolation,
    ContactCap,
    Decision,
    ExecutionWindow,
    MandateMustBeAlive,
    PreDebitNotificationRequired,
    QuietHours,
    Request,
    RetryCap,
    Ruling,
    SpendBudget,
    TerminalStop,
    Verdict,
)
from rebound.regulation import (
    QUIET_HOURS_END,
    QUIET_HOURS_START,
    REGULATORY_SOURCES,
    UPI_EXECUTION_WINDOWS,
    next_contactable_moment,
    next_execution_window_open,
    within_upi_execution_window,
)
from rebound.taxonomy import (
    CUSTOMER_FACING_ACTIONS,
    DEBIT_ACTIONS,
    Action,
    Disposition,
    Rail,
)

NOTICE = dt.datetime(2026, 3, 1, 8, 0)


def make_request(**overrides) -> Request:
    """A request that every default rule permits, unless overridden.

    Starting from a clean request matters: a fixture that starts blocked makes
    every test pass for the wrong reason.
    """
    base = dict(
        episode_id="EP_1",
        customer_id="CUST_1",
        rail=Rail.ENACH,
        disposition=Disposition.RETRY_TIMING,
        mandate_alive=True,
        cycle_amount_paise=500_00,
        ceiling_paise=1_000_00,
        valid_until=dt.date(2027, 1, 1),
        attempts=0,
        debit_attempts=0,
        contacts_made=0,
        spent_paise=0,
        notification_sent_at=NOTICE,
        action=Action.RETRY_SAME_RAIL,
        at=dt.datetime(2026, 3, 3, 14, 0),
    )
    base.update(overrides)
    return Request(**base)  # type: ignore[arg-type]


# ==========================================================================
# The baseline the other tests depend on
# ==========================================================================


def test_a_clean_request_is_allowed():
    """If this fails, every 'the gate denied it' test below is vacuous."""
    decision = ComplianceGate().adjudicate(make_request())
    assert decision.verdict is Verdict.ALLOW, decision.explain()
    assert decision.approval is not None
    assert decision.binding is None


# ==========================================================================
# Non-bypassability
# ==========================================================================


def test_an_approved_action_cannot_be_built_without_the_gate():
    """The whole design rests on this.

    If an executor's input can be forged, the gate is a convention rather than
    a control — and a convention is what fails silently when someone adds a
    call site next year and forgets a line.
    """
    with pytest.raises(PermissionError):
        ApprovedAction("EP_1", Action.RETRY_SAME_RAIL, dt.datetime(2026, 3, 3), object())

    with pytest.raises(PermissionError):
        ApprovedAction("EP_1", Action.RETRY_SAME_RAIL, dt.datetime(2026, 3, 3), None)


def test_a_denied_action_yields_no_approval():
    decision = ComplianceGate().adjudicate(make_request(mandate_alive=False))
    assert decision.verdict is Verdict.DENY
    assert decision.approval is None
    assert not decision.allowed


def test_require_raises_rather_than_returning_nothing():
    """A call site that has already decided to act must be stopped loudly.

    Returning ``None`` there invites ``if approval:`` being omitted.
    """
    gate = ComplianceGate()
    with pytest.raises(ComplianceViolation) as excinfo:
        gate.require(make_request(mandate_alive=False))
    assert "MANDATE_ALIVE" in str(excinfo.value)

    approval = gate.require(make_request())
    assert isinstance(approval, ApprovedAction)


def test_an_approval_cannot_be_edited_into_a_different_action():
    """``dataclasses.replace`` was a live forgery route and is the reason the
    token is init-only.

    When the token was a stored field, ``replace`` copied it through with
    everything else — so an approved retry could be edited into an approved
    voice call using nothing but the public API and no private names. Init-only
    means there is no token on the instance to copy, and no key to read off an
    approval someone already holds.
    """
    approval = ComplianceGate().require(make_request())

    with pytest.raises((PermissionError, TypeError, ValueError)):
        dataclasses.replace(approval, action=Action.VOICE_CALL)

    # An InitVar with a default leaves `token` on the class, so it is still
    # readable — but it reads as the default, never the key. That is the
    # property that matters: an approval someone already holds carries nothing
    # they can use to mint another one.
    assert getattr(approval, "token", None) is not _APPROVAL
    assert "token" not in getattr(ApprovedAction, "__slots__", ())

    # Copying an approval unchanged is fine — it is still the same permission.
    assert copy.deepcopy(approval).action is approval.action
    assert pickle.loads(pickle.dumps(approval)).action is approval.action


def test_the_token_is_the_only_key():
    """Documents the escape hatch honestly rather than pretending there is none.

    Python has no true private state. Someone importing ``_APPROVAL`` can forge
    an approval — but that is a deliberate act visible in a diff, which is a
    different risk from a forgotten call.
    """
    forged = ApprovedAction("EP_X", Action.VOICE_CALL, dt.datetime(2026, 3, 3), _APPROVAL)
    assert forged.episode_id == "EP_X"


# ==========================================================================
# Regulatory rules
# ==========================================================================


def test_a_dead_mandate_cannot_be_debited():
    decision = ComplianceGate().adjudicate(make_request(mandate_alive=False))
    assert decision.binding is not None
    assert decision.binding.rule_id == "REG.MANDATE_ALIVE"
    assert decision.binding.basis is Basis.REGULATORY


def test_an_expired_mandate_cannot_be_debited():
    ruling = MandateMustBeAlive().check(
        make_request(valid_until=dt.date(2026, 1, 1))
    )
    assert ruling is not None and ruling.verdict is Verdict.DENY


def test_a_debit_without_notice_is_denied_but_immature_notice_is_deferred():
    """The distinction the gate exists to make.

    No notice at all is a denial: the fix is to send one, which is a different
    action, not a later version of this one. Notice sent but not matured is a
    deferral, because waiting genuinely resolves it.
    """
    rule = PreDebitNotificationRequired()

    missing = rule.check(make_request(notification_sent_at=None))
    assert missing is not None and missing.verdict is Verdict.DENY

    at = dt.datetime(2026, 3, 1, 20, 0)
    immature = rule.check(make_request(notification_sent_at=NOTICE, at=at))
    assert immature is not None and immature.verdict is Verdict.DEFER
    assert immature.earliest_allowed_at == NOTICE + dt.timedelta(hours=24)


def test_notice_matures_exactly_at_the_boundary():
    """Off-by-one here is a compliance failure in the permissive direction."""
    rule = PreDebitNotificationRequired()
    matures = NOTICE + dt.timedelta(hours=24)

    assert rule.check(make_request(at=matures - dt.timedelta(seconds=1))) is not None
    assert rule.check(make_request(at=matures)) is None


def test_an_amount_over_the_afa_ceiling_is_denied():
    over = AfaCeiling().check(make_request(cycle_amount_paise=20_000_00))
    assert over is not None and over.verdict is Verdict.DENY
    assert AfaCeiling().check(make_request(cycle_amount_paise=15_000_00)) is None


def test_the_afa_rule_reads_the_cycle_amount_not_the_mandate_ceiling():
    """A mandate registered with headroom is not the thing being debited.

    Reading ``ceiling_paise`` here would deny ordinary small debits under large
    mandates — a rule that fires on the wrong field is worse than no rule,
    because it looks like diligence.
    """
    ruling = AfaCeiling().check(
        make_request(cycle_amount_paise=500_00, ceiling_paise=50_000_00)
    )
    assert ruling is None


def test_upi_executions_outside_the_window_are_deferred_not_denied():
    """A gate that can only say no turns every timing rule into a lost recovery."""
    at = dt.datetime(2026, 3, 3, 11, 0)
    assert not within_upi_execution_window(at)

    ruling = ExecutionWindow().check(make_request(rail=Rail.UPI_AUTOPAY, at=at))
    assert ruling is not None
    assert ruling.verdict is Verdict.DEFER
    assert ruling.earliest_allowed_at is not None
    assert within_upi_execution_window(ruling.earliest_allowed_at)


def test_the_execution_window_binds_only_upi():
    at = dt.datetime(2026, 3, 3, 11, 0)
    assert ExecutionWindow().check(make_request(rail=Rail.ENACH, at=at)) is None


def test_the_retry_cap_denies_beyond_the_scheme_limit():
    from rebound.regulation import MAX_EXECUTIONS_PER_CYCLE, MAX_RETRIES_PER_CYCLE

    cap = MAX_EXECUTIONS_PER_CYCLE + MAX_RETRIES_PER_CYCLE
    assert RetryCap().check(make_request(debit_attempts=cap - 1)) is None
    capped = RetryCap().check(make_request(debit_attempts=cap))
    assert capped is not None and capped.verdict is Verdict.DENY


def test_the_retry_cap_counts_presentations_not_nudges():
    """Three SMS nudges must not exhaust a presentation cap.

    ``Ledger.attempts`` increments on every action, so reading it here denied
    debits that had never been presented - and stamped the denial REGULATORY,
    telling a merchant a regulator forbade something no regulator had seen.
    Mislabelling a house effect as law is the exact failure this module's
    Basis split exists to prevent, so it mattered more than the lost retries.
    """
    nudged = make_request(attempts=99, debit_attempts=0)
    assert RetryCap().check(nudged) is None

    presented = make_request(attempts=99, debit_attempts=4)
    ruling = RetryCap().check(presented)
    assert ruling is not None and ruling.verdict is Verdict.DENY


def test_an_alt_rail_retry_is_judged_on_the_rail_it_presents_on():
    """``RETRY_ALT_RAIL`` presents on a different rail by definition.

    Keying the window rule on the episode's original rail got both directions
    wrong: an eNACH episode hopping to UPI - the most common hop, since UPI is
    first in ALT_RAILS for both other rails - escaped NPCI's windows entirely,
    while a UPI episode hopping to card was deferred for a constraint cards do
    not have. Unsafe one way, lost recovery the other.
    """
    from rebound.taxonomy import ALT_RAILS

    assert ALT_RAILS[Rail.ENACH][0] is Rail.UPI_AUTOPAY
    outside = dt.datetime(2026, 3, 3, 11, 30)

    hop_to_upi = make_request(
        rail=Rail.ENACH, action=Action.RETRY_ALT_RAIL, at=outside
    )
    ruling = ExecutionWindow().check(hop_to_upi)
    assert ruling is not None and ruling.verdict is Verdict.DEFER

    hop_off_upi = make_request(
        rail=Rail.UPI_AUTOPAY, action=Action.RETRY_ALT_RAIL, at=outside
    )
    assert hop_off_upi.presenting_rail is Rail.CARD_ON_FILE
    assert ExecutionWindow().check(hop_off_upi) is None


def test_the_gate_rejects_timezone_aware_times():
    """The window constants are IST wall-clock with no tzinfo, so an aware
    datetime compares against naive boundaries and either crashes or silently
    shifts. Rejected at the boundary rather than half-handled inside."""
    with pytest.raises(ValueError):
        make_request(at=dt.datetime(2026, 3, 3, 14, 0, tzinfo=dt.UTC))


def test_regulatory_rules_ignore_actions_that_are_not_debits():
    """A retry cap that also capped SMS would silently double as a contact cap,
    and the audit trail would cite a scheme rule for a house decision."""
    request = make_request(
        action=Action.NUDGE_SMS, attempts=99, debit_attempts=99, mandate_alive=False
    )
    for rule in (MandateMustBeAlive(), PreDebitNotificationRequired(), AfaCeiling(), RetryCap()):
        assert rule.check(request) is None, rule.rule_id


# ==========================================================================
# Policy rules
# ==========================================================================


def test_quiet_hours_defer_customer_contact():
    at = dt.datetime(2026, 3, 3, 2, 30)
    ruling = QuietHours().check(make_request(action=Action.VOICE_CALL, at=at))
    assert ruling is not None
    assert ruling.verdict is Verdict.DEFER
    assert ruling.basis is Basis.POLICY
    assert ruling.earliest_allowed_at == dt.datetime(2026, 3, 3, 9, 0)


def test_the_two_contact_sets_stay_in_step():
    """The gate's contact set and the simulator's must not drift apart silently.

    They answer adjacent questions - the simulator's is "what causes contact
    fatigue", the gate's is "what reaches the customer" - and today they differ
    by exactly one member: the pre-debit notification, which reaches the
    customer but is a compliance artifact rather than collection pressure.

    Asserted rather than assumed. An earlier version of this test claimed a
    wider gap than actually existed, which is how a false premise gets written
    into a docstring and then believed.
    """
    from rebound.sim.world import CONTACT_ACTIONS

    assert CUSTOMER_FACING_ACTIONS - CONTACT_ACTIONS == {
        Action.SEND_PRE_DEBIT_NOTIFICATION
    }
    assert CONTACT_ACTIONS - CUSTOMER_FACING_ACTIONS == set()


def test_quiet_hours_cover_a_collect_link():
    """Not a nudge, still a message arriving on a phone at 03:00."""
    at = dt.datetime(2026, 3, 3, 3, 0)
    ruling = QuietHours().check(make_request(action=Action.SEND_COLLECT_LINK, at=at))
    assert ruling is not None and ruling.verdict is Verdict.DEFER


def test_the_mandatory_notification_is_exempt_from_quiet_hours():
    """A courtesy rule must not cause a regulatory failure.

    The pre-debit notification has its own 24h timing requirement. Blocking it
    overnight means a debit due at 09:00 can never be noticed in time.
    """
    at = dt.datetime(2026, 3, 3, 3, 0)
    ruling = QuietHours().check(
        make_request(action=Action.SEND_PRE_DEBIT_NOTIFICATION, at=at)
    )
    assert ruling is None


def test_quiet_hours_do_not_block_a_debit():
    """Nobody is disturbed by a debit presenting itself at 03:00."""
    at = dt.datetime(2026, 3, 3, 3, 0)
    assert QuietHours().check(make_request(action=Action.RETRY_SAME_RAIL, at=at)) is None


def test_the_contact_cap_denies_further_contact():
    rule = ContactCap(max_contacts=3)
    assert rule.check(make_request(action=Action.NUDGE_SMS, contacts_made=2)) is None
    capped = rule.check(make_request(action=Action.NUDGE_SMS, contacts_made=3))
    assert capped is not None and capped.verdict is Verdict.DENY
    assert capped.basis is Basis.POLICY


def test_the_spend_budget_scales_with_the_amount_being_recovered():
    """A flat rupee cap would be absurd across a 200-rupee subscription and a
    20,000-rupee EMI. The budget is a fraction of what is being recovered."""
    rule = SpendBudget(fraction_of_amount=0.25)
    assert rule.check(make_request(cycle_amount_paise=1_000_00, spent_paise=200_00)) is None
    assert rule.check(make_request(cycle_amount_paise=100_00, spent_paise=200_00)) is not None


def test_stopping_is_never_denied():
    """Whatever else is true, the gate must always permit giving up.

    A gate that can deny STOP can trap an episode in a state where no action is
    permitted and it cannot be closed either.
    """
    trapped = make_request(
        action=Action.STOP,
        disposition=Disposition.TERMINAL,
        mandate_alive=False,
        attempts=99,
        debit_attempts=99,
        contacts_made=99,
        spent_paise=10_000_00,
        cycle_amount_paise=100_00,
        notification_sent_at=None,
        at=dt.datetime(2026, 3, 3, 3, 0),
    )
    decision = ComplianceGate().adjudicate(trapped)
    assert decision.allowed, decision.explain()


def test_a_terminal_disposition_permits_nothing_but_stopping():
    rule = TerminalStop()
    for action in (Action.RETRY_SAME_RAIL, Action.NUDGE_SMS, Action.REQUEST_REMANDATE):
        ruling = rule.check(make_request(action=action, disposition=Disposition.TERMINAL))
        assert ruling is not None and ruling.verdict is Verdict.DENY
    assert rule.check(make_request(action=Action.STOP, disposition=Disposition.TERMINAL)) is None


# ==========================================================================
# Adjudication
# ==========================================================================


def test_every_rule_runs_even_after_one_denies():
    """A merchant asking why nothing happened deserves every reason at once,
    not one per failed attempt."""
    request = make_request(
        mandate_alive=False,
        attempts=99,
        debit_attempts=99,
        cycle_amount_paise=20_000_00,
        notification_sent_at=None,
    )
    decision = ComplianceGate().adjudicate(request)
    fired = {r.rule_id for r in decision.rulings}
    assert {
        "REG.MANDATE_ALIVE",
        "REG.PRE_DEBIT_NOTICE",
        "REG.AFA_CEILING",
        "REG.RETRY_CAP",
    } <= fired


def test_deny_outranks_defer():
    """Otherwise a deferred action would be scheduled and then refused."""
    request = make_request(
        rail=Rail.UPI_AUTOPAY,
        at=dt.datetime(2026, 3, 3, 11, 0),
        mandate_alive=False,
    )
    decision = ComplianceGate().adjudicate(request)
    assert decision.verdict is Verdict.DENY
    assert any(r.verdict is Verdict.DEFER for r in decision.rulings)


def test_the_deferral_time_satisfies_every_deferring_rule():
    """Taking the earliest deferral would fire an action that is still blocked.

    Two constraints at once: a UPI execution outside its window, and a
    pre-debit notice that has not matured. The answer has to satisfy both.
    """
    sent = dt.datetime(2026, 3, 3, 5, 0)
    request = make_request(
        rail=Rail.UPI_AUTOPAY,
        notification_sent_at=sent,
        at=dt.datetime(2026, 3, 3, 11, 0),
    )
    decision = ComplianceGate().adjudicate(request)
    assert decision.verdict is Verdict.DEFER

    when = decision.earliest_allowed_at
    assert when is not None
    assert when >= sent + dt.timedelta(hours=24)
    assert within_upi_execution_window(when)

    # Not simply the max of the two proposals. 18:00 satisfies the notice but
    # lands in the 17:00-21:30 dead zone, so the gate re-adjudicates until the
    # candidate stops moving.
    settled = ComplianceGate().adjudicate(_moved(request, when))
    assert settled.allowed, settled.explain()


def _moved(request: Request, at: dt.datetime) -> Request:
    import dataclasses as _dc

    return _dc.replace(request, at=at)


def test_a_deferral_lands_on_a_time_the_gate_will_still_honour():
    """The property that justifies having DEFER at all.

    Taking the latest proposed time assumes every constraint means "permitted
    from T onward". The execution window is not that shape - it is a set of
    intervals, and clearing the notice requirement can drop you inside a closed
    one. This case took three hops to settle: 11:00 to 18:00 to 21:30.
    """
    request = make_request(
        rail=Rail.UPI_AUTOPAY,
        notification_sent_at=dt.datetime(2026, 3, 9, 18, 0),
        at=dt.datetime(2026, 3, 10, 11, 0),
    )
    decision = ComplianceGate().adjudicate(request)
    assert decision.verdict is Verdict.DEFER

    when = decision.earliest_allowed_at
    assert when is not None
    assert within_upi_execution_window(when)
    assert ComplianceGate().adjudicate(_moved(request, when)).allowed


def test_a_denied_action_carries_no_reschedule_time():
    """A DENY that also collected a deferral used to publish that deferral's
    time, so a caller keying off the field rather than the verdict would
    reschedule a permanently dead action forever."""
    request = make_request(
        rail=Rail.UPI_AUTOPAY,
        at=dt.datetime(2026, 3, 3, 11, 0),
        cycle_amount_paise=20_000_00,
    )
    decision = ComplianceGate().adjudicate(request)
    assert decision.verdict is Verdict.DENY
    assert decision.earliest_allowed_at is None

    gate = ComplianceGate()
    gate.adjudicate(request)
    assert gate.audit_trail()[0]["earliest_allowed_at"] is None


def test_the_explanation_cites_every_rule_at_the_deciding_severity():
    """With two simultaneous deferrals the old one-line record named whichever
    rule sat earlier in the list and never mentioned the other."""
    request = make_request(
        rail=Rail.UPI_AUTOPAY,
        notification_sent_at=dt.datetime(2026, 3, 9, 18, 0),
        at=dt.datetime(2026, 3, 10, 11, 0),
    )
    decision = ComplianceGate().adjudicate(request)
    assert len(decision.governing) == 2
    text = decision.explain()
    assert "REG.PRE_DEBIT_NOTICE" in text
    assert "REG.EXECUTION_WINDOW" in text


def test_a_deferral_must_say_until_when():
    """A deferral with no time is a denial wearing a softer word."""
    with pytest.raises(ValueError):
        Ruling("X", Basis.POLICY, Verdict.DEFER, "later")

    with pytest.raises(ValueError):
        Ruling(
            "X",
            Basis.POLICY,
            Verdict.DENY,
            "no",
            earliest_allowed_at=dt.datetime(2026, 3, 3),
        )


def test_permitted_actions_returns_only_allowed_ones():
    gate = ComplianceGate()
    base = make_request(disposition=Disposition.TERMINAL)
    at = base.at
    allowed = gate.permitted_actions(
        [(a, at) for a in (Action.RETRY_SAME_RAIL, Action.NUDGE_SMS, Action.STOP)],
        base,
    )
    assert allowed == (Action.STOP,)


def test_the_audit_trail_records_every_decision_with_its_reason():
    gate = ComplianceGate()
    gate.adjudicate(make_request())
    gate.adjudicate(make_request(mandate_alive=False))

    trail = gate.audit_trail()
    assert len(trail) == 2
    assert trail[0]["verdict"] == "allow"
    assert trail[1]["verdict"] == "deny"
    assert trail[1]["binding_rule"] == "REG.MANDATE_ALIVE"
    assert trail[1]["basis"] == "regulatory"
    assert trail[1]["reason"]


def test_the_explanation_names_the_rule_and_its_basis():
    decision = ComplianceGate().adjudicate(
        make_request(action=Action.NUDGE_SMS, contacts_made=9)
    )
    text = decision.explain()
    assert "POL.CONTACT_CAP" in text
    assert "policy" in text


# ==========================================================================
# Law versus house policy
# ==========================================================================


def test_every_rule_declares_a_basis():
    for rule in DEFAULT_RULES:
        assert rule.basis in (Basis.REGULATORY, Basis.POLICY), rule.rule_id


def test_rule_ids_are_prefixed_by_basis_and_unique():
    """The prefix is what makes a scan of an audit trail readable, and the
    uniqueness is what makes a citation resolvable."""
    ids = [rule.rule_id for rule in DEFAULT_RULES]
    assert len(ids) == len(set(ids))
    for rule in DEFAULT_RULES:
        expected = "REG." if rule.basis is Basis.REGULATORY else "POL."
        assert rule.rule_id.startswith(expected), rule.rule_id


def test_contact_limits_are_never_labelled_regulatory():
    """Calling a self-imposed contact cap 'compliance' claims a regulator's
    authority for a product decision, and makes it unarguable by
    misattribution. It is ours; it is labelled ours."""
    for rule in (QuietHours(), ContactCap(), SpendBudget()):
        assert rule.basis is Basis.POLICY


def test_the_gate_reports_which_regulatory_claims_are_unverified():
    """An audit trail implying a diligence that was never done is worse than
    no audit trail. Every constant is currently unconfirmed and says so."""
    unverified = ComplianceGate().unverified_rules()

    # Rule ids, not constant names. It returned constant names before, so a
    # caller joining this against ``Decision.binding.rule_id`` got an empty set
    # and silently reported zero unverified rules - the precise failure the
    # method exists to prevent.
    assert "REG.PRE_DEBIT_NOTICE" in unverified
    assert "REG.AFA_CEILING" in unverified
    assert "REG.EXECUTION_WINDOW" in unverified

    rule_ids = {rule.rule_id for rule in DEFAULT_RULES}
    assert set(unverified) <= rule_ids


def test_a_probe_does_not_enter_the_audit_trail():
    """The trail answers "why did nothing happen for this customer". Filling it
    with hypotheticals the sequencer asked itself makes it unreadable - one
    ``permitted_actions`` call over eleven candidates wrote eleven rows, none
    of which corresponded to anything that happened."""
    gate = ComplianceGate()
    gate.permitted_actions(
        [(a, make_request().at) for a in (Action.STOP, Action.NUDGE_SMS)],
        make_request(),
    )
    assert gate.audit_trail() == ()

    gate.adjudicate(make_request())
    assert len(gate.audit_trail()) == 1


def test_every_regulatory_constant_has_a_source():
    """A new constant cannot be added without declaring where it came from."""
    import rebound.regulation as reg

    documented = set(REGULATORY_SOURCES)
    constants = {
        name
        for name in dir(reg)
        if name.isupper()
        and not name.startswith("_")
        and "QUIET" not in name
        and name != "REGULATORY_SOURCES"
    }
    assert constants == documented, f"undocumented: {constants - documented}"


# ==========================================================================
# Window arithmetic
# ==========================================================================


def test_next_execution_window_open_is_the_earliest_valid_moment():
    """Probed across a whole day, because window arithmetic is where
    off-by-ones live and every one of them is a compliance failure."""
    day = dt.datetime(2026, 3, 3)
    for minutes in range(0, 24 * 60, 5):
        at = day + dt.timedelta(minutes=minutes)
        opens = next_execution_window_open(at)
        assert opens >= at
        assert within_upi_execution_window(opens), opens
        if within_upi_execution_window(at):
            assert opens == at


def test_next_contactable_moment_handles_the_midnight_wrap():
    """Quiet hours run 21:00 to 09:00, so the period wraps midnight — the exact
    shape that produces bugs."""
    day = dt.datetime(2026, 3, 3)
    for minutes in range(0, 24 * 60, 5):
        at = day + dt.timedelta(minutes=minutes)
        resumes = next_contactable_moment(at)
        assert resumes >= at
        clock = resumes.time()
        assert QUIET_HOURS_END <= clock < QUIET_HOURS_START, resumes


def test_the_windows_are_sorted_and_non_overlapping():
    ordered = sorted(UPI_EXECUTION_WINDOWS)
    assert list(UPI_EXECUTION_WINDOWS) == ordered
    for (_, end), (start, _) in zip(ordered, ordered[1:]):
        assert end < start
