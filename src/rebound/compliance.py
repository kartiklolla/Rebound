"""The compliance gate: the agent proposes, the gate disposes.

Every action a *customer* sees passes through here. ``MessageBrief.build``
accepts an ``ApprovedAction`` and nothing else, and one cannot be constructed by
any ordinary route outside this module — direct construction raises, and
``dataclasses.replace`` has no token to pass. A caller that wants to skip the
gate has to import a private sentinel and pass it explicitly, which nobody does
by accident and which shows up in a diff.

Two limits on that, both found by an outside evaluator and both worth stating
where the claim is made rather than in a footnote:

*The scope is narrower than "every action".* ``World.apply`` takes a bare
``Action`` and ``evaluate_policy`` takes a plain ``Decision``, so the simulator's
execution boundary has no approval in it — deliberately, since the baselines are
meant to be ungated, but it means the guarantee covers what a customer receives
and not what is presented to a rail.

*And it is not absolute.* ``object.__new__`` plus ``__setattr__``, subclassing,
harvesting ``_APPROVAL``, and unpickling all mint one, and the same route alters
an existing one in place. The threat model here is an author who forgets a line
next year, and against that the design holds; against someone deliberately
reaching around it, it does not, and this module should not be read as claiming
otherwise.

Why it is built this way
------------------------
The obvious design is a ``check_compliance(action) -> bool`` that the sequencer
calls before acting. That design fails in one specific way: it is correct only
as long as every future call site remembers to call it. The failure is silent,
it is a single missing line, and the tests still pass because the tests call the
checker directly. This is the same shape of defect as the audit's finding that
``decision_index`` was *documented* as an identifier and never added to the
identifier set — a comment, or a convention, is not a control.

So the type system carries it: there is no accidental path from "I would like to
send this" to "a customer received it" that skips ``adjudicate``.

Three verdicts, not two
-----------------------
A gate that can only say yes or no turns every timing rule into a lost recovery.
UPI executions are refused outside NPCI's windows — but "denied" tells a
sequencer to abandon an action that becomes perfectly legal in four hours.
``DEFER`` carries ``earliest_allowed_at``, so the sequencer can schedule instead
of giving up. Most of the value in this module is in that distinction.

Law and house policy are not the same thing
-------------------------------------------
Each rule declares a ``Basis``. ``REGULATORY`` rules encode external
requirements; violating one is a compliance incident. ``POLICY`` rules encode
our own restraint — contact caps, spend budgets, quiet hours — where violating
one is merely bad behaviour that a merchant could legitimately configure
differently.

Keeping them apart matters for honesty in both directions. A merchant must be
able to see which constraints they may tune and which they may not. And this
system must not claim regulatory cover for choices that are actually its own
preferences: labelling a self-imposed contact cap as "compliance" is a way of
making a product decision unarguable by misattributing it to a regulator.

The gate reports the evidentiary status of the regulatory rules it applies. All
of them currently rest on secondary sources — see ``rebound.regulation``. An
audit trail that implied otherwise would be worse than useless.

Deliberately not an LLM
-----------------------
This is the clearest case in the system for *not* using a model. A compliance
decision has to be deterministic, reproducible on replay, explainable by
citation to the rule that fired, and identical for two customers in identical
circumstances. A language model is none of those things. The gate is a few
hundred lines of boolean logic because a few hundred lines of boolean logic is
the correct tool, and reaching for a model here would be the kind of judgment
error the whole system is meant to demonstrate the absence of.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import InitVar, dataclass, field, fields
from enum import StrEnum
from typing import Protocol

from rebound.regulation import (
    AFA_EXEMPT_CEILING_PAISE,
    MAX_EXECUTIONS_PER_CYCLE,
    MAX_RETRIES_PER_CYCLE,
    PRE_DEBIT_NOTIFICATION_HOURS,
    QUIET_HOURS_END,
    QUIET_HOURS_START,
    REGULATORY_SOURCES,
    next_contactable_moment,
    next_execution_window_open,
)
from rebound.taxonomy import (
    ALT_RAILS,
    CUSTOMER_FACING_ACTIONS,
    DEBIT_ACTIONS,
    Action,
    Disposition,
    Rail,
)


class Basis(StrEnum):
    """Where a rule's authority comes from."""

    REGULATORY = "regulatory"
    """External: a regulator or scheme rule. Not ours to relax."""

    POLICY = "policy"
    """Ours: restraint we chose, and answerable to us rather than a regulator.

    Not a synonym for "safe to switch off". ``TerminalStop`` is policy because
    refusing to spend money on a dead account is a judgement about futility
    rather than a rule anyone imposed on us — but a merchant who disabled it
    would burn gateway fees on closed accounts, which is their money and their
    call. The label says who owns the rule, not how harmless it is to drop.
    """


class Verdict(StrEnum):
    """What the gate says about a proposed action."""

    ALLOW = "allow"
    DEFER = "defer"
    """Legal, but not now. Carries the moment it becomes permissible."""

    DENY = "deny"
    """Not permissible, and waiting will not change that within this episode."""


#: Actions no house-policy rule may block.
#:
#: ``STOP`` because a gate that can deny it can trap an episode with no legal
#: action and no way to close.
#:
#: ``SEND_PRE_DEBIT_NOTIFICATION`` because it is a compliance artifact with its
#: own timing requirement, not collection pressure. ``QuietHours`` exempted it
#: from the start; ``ContactCap`` and ``SpendBudget`` did not, so a spent budget
#: could deny the very notice a lawful debit depends on. Three sibling rules
#: reasoning inconsistently about the same case is why the exemption is one
#: named set rather than a condition repeated in each rule.
_ALWAYS_PERMITTED: frozenset[Action] = frozenset(
    {Action.STOP, Action.SEND_PRE_DEBIT_NOTIFICATION}
)

#: Severity order. A DENY anywhere beats a DEFER anywhere beats ALLOW.
_SEVERITY: dict[Verdict, int] = {Verdict.ALLOW: 0, Verdict.DEFER: 1, Verdict.DENY: 2}


@dataclass(frozen=True, slots=True)
class Ruling:
    """One rule's opinion about one proposed action."""

    rule_id: str
    basis: Basis
    verdict: Verdict
    reason: str
    earliest_allowed_at: dt.datetime | None = None

    def __post_init__(self) -> None:
        if self.verdict is Verdict.DEFER and self.earliest_allowed_at is None:
            raise ValueError(
                f"{self.rule_id} deferred without saying until when. A deferral "
                f"with no time is a denial wearing a softer word."
            )
        if self.verdict is not Verdict.DEFER and self.earliest_allowed_at is not None:
            raise ValueError(
                f"{self.rule_id} gave a time on a {self.verdict} ruling"
            )


@dataclass(frozen=True, slots=True)
class Request:
    """A proposed action, in the form the gate can adjudicate.

    Carries only what a merchant's own systems would know at decision time —
    the same discipline ``EpisodeView`` enforces on the model. The gate does
    not get privileged information either.
    """

    episode_id: str
    customer_id: str
    rail: Rail
    disposition: Disposition
    mandate_alive: bool
    cycle_amount_paise: int
    ceiling_paise: int
    valid_until: dt.date
    attempts: int
    """Every action taken on this episode so far, contacts included.

    No rule reads this — ``RetryCap`` binds on ``debit_attempts`` and the
    contact rules on ``contacts_made``. It is carried because ``_as_dict``
    walks ``fields()`` into the audit record, so every adjudication is stored
    with the full state it was decided against. A reviewer reconstructing why
    a decision went the way it did needs the numbers the rules did *not* use
    as much as the ones they did.
    """

    debit_attempts: int
    """Presentations against the mandate only.

    Separate from ``attempts`` because they are different numbers and the
    scheme retry cap binds on this one. ``attempts`` comes from the ledger,
    which increments on every action — so three SMS nudges used to exhaust the
    NPCI *presentation* cap, and the gate denied a debit that had never been
    presented while citing a regulator as the reason.
    """

    contacts_made: int
    spent_paise: int
    notification_sent_at: dt.datetime | None
    action: Action
    at: dt.datetime

    contact_suppressed: bool = False
    """The customer asked us to stop contacting them about this payment.

    Defaulted rather than required, and placed last, so that adding an
    enforcement mechanism did not force a rewrite of every construction site —
    including the fixtures whose job is to start from a request every rule
    permits. A request that does not mention suppression is not suppressed.
    """

    def __post_init__(self) -> None:
        if self.at.tzinfo is not None:
            raise ValueError(
                "Request.at must be naive local time. The window constants are "
                "IST-local wall-clock times with no tzinfo, so an aware "
                "datetime silently compares against naive boundaries. Convert "
                "at the edge of the system, not here."
            )

    @property
    def presenting_rail(self) -> Rail:
        """The rail this action would actually present on.

        Not always ``self.rail``. ``RETRY_ALT_RAIL`` presents on a *different*
        rail by definition, and keying the execution-window rule on the
        episode's original rail got both directions wrong: an eNACH episode
        hopping to UPI escaped NPCI's windows entirely, while a UPI episode
        hopping to card was deferred for a constraint that does not apply to
        cards. The first fails unsafe, the second loses recoveries.
        """
        if self.action is Action.RETRY_ALT_RAIL:
            alternatives = ALT_RAILS.get(self.rail, ())
            return alternatives[0] if alternatives else self.rail
        return self.rail

    @classmethod
    def from_view(cls, view: object, action: Action, at: dt.datetime) -> Request:
        """Build a request from anything with the ``EpisodeView`` shape.

        Structurally typed rather than importing ``EpisodeView``, so the gate
        does not depend on the simulator. In production the same call site
        would pass a real episode record.
        """
        return cls(
            episode_id=getattr(view, "episode_id"),
            customer_id=getattr(view, "customer_id"),
            rail=getattr(view, "rail"),
            disposition=getattr(view, "disposition"),
            mandate_alive=getattr(view, "mandate_alive"),
            cycle_amount_paise=getattr(view, "cycle_amount_paise"),
            ceiling_paise=getattr(view, "ceiling_paise"),
            valid_until=getattr(view, "valid_until"),
            attempts=getattr(view, "attempts"),
            debit_attempts=sum(
                1
                for outcome in getattr(view, "history", ())
                if outcome.action in DEBIT_ACTIONS
            ),
            contacts_made=getattr(view, "contacts_made"),
            contact_suppressed=getattr(view, "contact_suppressed", False),
            spent_paise=getattr(view, "spent_paise"),
            notification_sent_at=getattr(view, "notification_sent_at"),
            action=action,
            at=at,
        )


class Rule(Protocol):
    """A single constraint.

    Returns ``None`` when the rule has no opinion — most rules are silent on
    most actions — or a ``Ruling`` when it does.
    """

    rule_id: str
    basis: Basis
    source_key: str
    """Key into ``REGULATORY_SOURCES``. Empty for policy rules and for
    regulatory rules that rest on structure rather than a cited constant."""

    def check(self, request: Request) -> Ruling | None: ...


# ==========================================================================
# Regulatory rules
# ==========================================================================


@dataclass(frozen=True, slots=True)
class MandateMustBeAlive:
    """A revoked, cancelled or expired mandate cannot be debited.

    Structural, not probabilistic: this is true by definition of the rail, and
    it is the rule that stops the fixed ladder from spending three more gateway
    fees asking a dead mandate to reconsider.
    """

    rule_id: str = "REG.MANDATE_ALIVE"
    basis: Basis = Basis.REGULATORY
    source_key: str = ""
    """No constant behind this one. That a cancelled mandate cannot be
    debited is true by definition of the rail, not by citation."""

    def check(self, request: Request) -> Ruling | None:
        if request.action not in DEBIT_ACTIONS:
            return None
        if not request.mandate_alive:
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DENY,
                "the mandate did not survive this failure; it cannot be presented",
            )
        if request.at.date() > request.valid_until:
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DENY,
                f"the mandate expired on {request.valid_until}",
            )
        return None


@dataclass(frozen=True, slots=True)
class PreDebitNotificationRequired:
    """A debit needs notice given at least 24 hours earlier.

    Deferred rather than denied when notice has been sent but has not matured,
    and denied when no notice exists at all — because in that case the fix is to
    send one, which is a different action, not a later version of this one.

    Scope: the notice attaches to the **cycle**, not to each presentation
    within it. A recovery episode exists because a scheduled debit already went
    out and failed, so the cycle's notice question is settled — unless the
    failure says otherwise, which is precisely what a ``MERCHANT_FIX``
    disposition means. ``CARD_PRE_DEBIT_NOTIFICATION_MISSING`` and its siblings
    are the codes where the notice is the outstanding blocker, and there the
    rule binds hard.

    **This rule was written wrong first, and the correction needs recording
    because of how it was found.** The original applied to every debit
    unconditionally. That banned every retry in every recovery episode — none
    of them carry a notification timestamp — and the sequencer's recovery rate
    came out at 0.0136 against the naive ladder's 0.4561, with
    ``retry_same_rail`` never once selected.

    Loosening a compliance rule because it produced an unflattering number is
    how compliance logic rots, so the argument has to stand without that
    result. It does: reading the requirement as per-presentation would mean a
    merchant must give 24 hours' notice before each retry of a debit the
    customer was already told about, which would make same-day retry impossible
    for everyone in the market and is not how the rails behave. The simulator
    encodes the same reading independently at ``world.py:831``. The measurement
    is what prompted the re-read; it is not the justification.
    """

    hours: int = PRE_DEBIT_NOTIFICATION_HOURS
    rule_id: str = "REG.PRE_DEBIT_NOTICE"
    basis: Basis = Basis.REGULATORY
    source_key: str = "PRE_DEBIT_NOTIFICATION_HOURS"

    def check(self, request: Request) -> Ruling | None:
        if request.action not in DEBIT_ACTIONS:
            return None
        if request.disposition is not Disposition.MERCHANT_FIX:
            return None
        sent = request.notification_sent_at
        if sent is None:
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DENY,
                f"no pre-debit notification on record; {self.hours}h notice is "
                f"required before presenting",
            )
        matures = sent + dt.timedelta(hours=self.hours)
        if request.at < matures:
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DEFER,
                f"notice sent {sent:%Y-%m-%d %H:%M} matures at "
                f"{matures:%Y-%m-%d %H:%M}",
                earliest_allowed_at=matures,
            )
        return None


@dataclass(frozen=True, slots=True)
class AfaCeiling:
    """Above the AFA-exempt ceiling a recurring debit needs authentication.

    The gate cannot supply that authentication, so it denies rather than
    defers: the resolution is a mandate amendment or a customer-authenticated
    payment, both of which are other actions.

    Compares against the *cycle amount*, not the mandate ceiling. A mandate
    registered with headroom is not the thing being debited.
    """

    ceiling_paise: int = AFA_EXEMPT_CEILING_PAISE
    rule_id: str = "REG.AFA_CEILING"
    basis: Basis = Basis.REGULATORY
    source_key: str = "AFA_EXEMPT_CEILING_PAISE"

    def check(self, request: Request) -> Ruling | None:
        if request.action not in DEBIT_ACTIONS:
            return None
        # Applied to all three rails, deliberately and not by oversight. An
        # outside reviewer flagged it as "a card rule applied to eNACH and UPI"
        # and was explicit about not having checked the underlying regulation;
        # RBI's e-mandate framework covers recurring e-mandates on cards, on
        # accounts, and on UPI, so the wider scope is the defensible reading.
        #
        # It is recorded here rather than silently narrowed because narrowing
        # it would fail in the dangerous direction — permitting an unauthorised
        # debit — while the wide reading fails in the safe one, and the measured
        # incidence is 3 of 6,898 episodes. The primary-source check is on
        # regulation.py's unverified list, where AFA_EXEMPT_CEILING_PAISE is
        # marked REPORTED rather than CONFIRMED; that is the thing to resolve,
        # not this branch.
        if request.cycle_amount_paise > self.ceiling_paise:
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DENY,
                f"cycle amount {request.cycle_amount_paise / 100:,.0f} exceeds the "
                f"AFA-exempt ceiling {self.ceiling_paise / 100:,.0f}; the debit "
                f"needs an additional factor this gate cannot supply",
            )
        return None


@dataclass(frozen=True, slots=True)
class ExecutionWindow:
    """UPI Autopay executions may only be presented inside NPCI's windows.

    The rule that most justifies having ``DEFER`` at all. Outside a window the
    rail refuses the presentation, but the same retry four hours later is
    ordinary and legal.
    """

    rule_id: str = "REG.EXECUTION_WINDOW"
    basis: Basis = Basis.REGULATORY
    source_key: str = "UPI_EXECUTION_WINDOWS"

    def check(self, request: Request) -> Ruling | None:
        if request.action not in DEBIT_ACTIONS:
            return None
        if request.presenting_rail is not Rail.UPI_AUTOPAY:
            return None
        opens = next_execution_window_open(request.at)
        if opens > request.at:
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DEFER,
                f"outside the UPI Autopay execution windows; next opens "
                f"{opens:%Y-%m-%d %H:%M}",
                earliest_allowed_at=opens,
            )
        return None


@dataclass(frozen=True, slots=True)
class RetryCap:
    """One presentation per cycle plus a bounded number of retries.

    This is the constraint that gives the whole project its shape. With
    unlimited retries the optimal policy is to retry forever and the model is
    decoration. The cap is what makes *which* attempts to spend a decision with
    value.
    """

    max_presentations: int = MAX_EXECUTIONS_PER_CYCLE + MAX_RETRIES_PER_CYCLE
    """Total presentations the cycle allows, the original debit included."""

    rule_id: str = "REG.RETRY_CAP"
    basis: Basis = Basis.REGULATORY
    source_key: str = "MAX_RETRIES_PER_CYCLE"

    def check(self, request: Request) -> Ruling | None:
        if request.action not in DEBIT_ACTIONS:
            return None
        # ``debit_attempts`` counts presentations *within the recovery
        # episode*, and the episode exists because a scheduled debit already
        # went out and failed — which is stated explicitly in
        # ``PreDebitNotificationRequired`` a few classes above. Comparing an
        # in-episode count against a cycle-wide cap therefore granted one
        # presentation too many: four in-episode retries plus the original is
        # five against a cap of four, and the denial read "4 presentations
        # already made against a cap of 4" while standing at five. Measured on
        # a full run, 61 of 321 episodes (19%) reached that fifth presentation.
        #
        # Two rules in this file disagreed about whether presentation #1 had
        # happened, and the one citing a regulator was the one that was wrong.
        presentations = MAX_EXECUTIONS_PER_CYCLE + request.debit_attempts
        if presentations >= self.max_presentations:
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DENY,
                f"{presentations} presentations already made against a "
                f"cap of {self.max_presentations} for this cycle "
                f"({MAX_EXECUTIONS_PER_CYCLE} original plus "
                f"{MAX_RETRIES_PER_CYCLE} retries)",
            )
        return None


# ==========================================================================
# Policy rules
# ==========================================================================


@dataclass(frozen=True, slots=True)
class ContactSuppressed:
    """The customer asked us to stop, so we stop.

    Policy rather than regulatory, because no regulator imposed it — but it is
    the one policy rule that is not ours to relax, and a merchant who disabled
    it would be breaking a promise this system made in writing.

    It exists because the promise had no mechanism. ``portal.answer`` returned
    "we will not message you about this payment again" and nothing anywhere
    could stop the next nudge: no flag on the episode, no rule here, no check
    in the sequencer. An unenforceable undertaking on a system whose pitch is
    contact discipline is the one a compliance-minded reader finds first.

    ``STOP`` and the pre-debit notice are exempt through ``_ALWAYS_PERMITTED``:
    a notice is a disclosure the following debit depends on, not collection
    pressure, and a gate that can deny ``STOP`` can trap an episode open.
    """

    rule_id: str = "POL.CONTACT_SUPPRESSED"
    basis: Basis = Basis.POLICY
    source_key: str = ""

    def check(self, request: Request) -> Ruling | None:
        if request.action in _ALWAYS_PERMITTED:
            return None
        if not request.contact_suppressed:
            return None
        if request.action not in CUSTOMER_FACING_ACTIONS:
            return None
        return Ruling(
            self.rule_id,
            self.basis,
            Verdict.DENY,
            "the customer asked not to be contacted about this payment",
        )


@dataclass(frozen=True, slots=True)
class QuietHours:
    """No collection pressure overnight.

    House policy, labelled as house policy. India has no single statutory
    quiet-hours rule binding a merchant's own transactional messaging the way
    this system uses it, and calling this "compliance" would be claiming a
    regulator's authority for our own preference.

    The mandatory pre-debit notification is exempt. It is a compliance artifact
    with its own timing requirement, and blocking it overnight would mean a
    debit due at 09:00 could never be noticed in time — a self-imposed courtesy
    rule causing a regulatory failure.
    """

    start: dt.time = QUIET_HOURS_START
    end: dt.time = QUIET_HOURS_END
    rule_id: str = "POL.QUIET_HOURS"
    basis: Basis = Basis.POLICY
    source_key: str = ""

    def check(self, request: Request) -> Ruling | None:
        if request.action not in CUSTOMER_FACING_ACTIONS:
            return None
        if request.action is Action.SEND_PRE_DEBIT_NOTIFICATION:
            return None
        resumes = next_contactable_moment(request.at)
        if resumes > request.at:
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DEFER,
                f"quiet hours {self.start:%H:%M}-{self.end:%H:%M}; contact "
                f"resumes {resumes:%Y-%m-%d %H:%M}",
                earliest_allowed_at=resumes,
            )
        return None


@dataclass(frozen=True, slots=True)
class ContactCap:
    """A ceiling on how many times one failure may be raised with a customer.

    The measured justification is in the baselines: ``aggressive_contact``
    recovers less than the naive ladder *and* pushes revocation from 6.00% to
    14.54%, destroying more value than abandoning every failed debit. Contact
    is not free and the model cannot be trusted to discover that on its own,
    because the immediate label rewards it for trying.
    """

    max_contacts: int = 3
    rule_id: str = "POL.CONTACT_CAP"
    basis: Basis = Basis.POLICY
    source_key: str = ""

    def check(self, request: Request) -> Ruling | None:
        if request.action not in CUSTOMER_FACING_ACTIONS:
            return None
        if request.action is Action.SEND_PRE_DEBIT_NOTIFICATION:
            return None
        if request.contacts_made >= self.max_contacts:
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DENY,
                f"{request.contacts_made} contacts already made against a cap of "
                f"{self.max_contacts} for this episode",
            )
        return None


@dataclass(frozen=True, slots=True)
class SpendBudget:
    """Stop spending when recovery costs more than it returns.

    Expressed as a fraction of the amount being recovered rather than a flat
    rupee figure, because the sensible spend on a 200-rupee subscription and a
    20,000-rupee loan EMI are not the same number.
    """

    fraction_of_amount: float = 0.25
    rule_id: str = "POL.SPEND_BUDGET"
    basis: Basis = Basis.POLICY
    source_key: str = ""

    def check(self, request: Request) -> Ruling | None:
        if request.action in _ALWAYS_PERMITTED:
            return None
        budget = int(request.cycle_amount_paise * self.fraction_of_amount)
        if budget == 0:
            # Integer truncation, not an exhausted budget. Below four paise a
            # quarter of the amount rounds to nothing, so the ordinary test
            # `spent >= budget` read `0 >= 0` and denied every action before a
            # single paise had been spent — reporting "budget exhausted" for an
            # episode that had never been worked. Denying is still right, since
            # no recovery action costs less than the amount at stake; saying
            # why is the part that was wrong.
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DENY,
                f"a cycle amount of {request.cycle_amount_paise} paise cannot "
                f"justify any recovery spend at "
                f"{self.fraction_of_amount:.0%} of the amount",
            )
        if request.spent_paise >= budget:
            return Ruling(
                self.rule_id,
                self.basis,
                Verdict.DENY,
                f"spent {request.spent_paise / 100:,.2f} against a budget of "
                f"{budget / 100:,.2f} ({self.fraction_of_amount:.0%} of the "
                f"amount being recovered)",
            )
        return None


@dataclass(frozen=True, slots=True)
class TerminalStop:
    """Nothing but STOP on a terminal disposition.

    A closed account does not reopen because it was asked politely. Every
    action here has a cost and a probability of zero, and the taxonomy already
    knows which failures these are — this rule just refuses to let the
    sequencer's optimism override it.
    """

    rule_id: str = "POL.TERMINAL_STOP"
    basis: Basis = Basis.POLICY
    source_key: str = ""

    def check(self, request: Request) -> Ruling | None:
        if request.disposition is not Disposition.TERMINAL:
            return None
        if request.action in _ALWAYS_PERMITTED:
            return None
        return Ruling(
            self.rule_id,
            self.basis,
            Verdict.DENY,
            "the failure is terminal; no action but stopping can succeed",
        )


#: The rules applied when a gate is built without an explicit set.
DEFAULT_RULES: tuple[Rule, ...] = (
    MandateMustBeAlive(),
    PreDebitNotificationRequired(),
    AfaCeiling(),
    ExecutionWindow(),
    RetryCap(),
    QuietHours(),
    ContactSuppressed(),
    ContactCap(),
    SpendBudget(),
    TerminalStop(),
)


# ==========================================================================
# Approval
# ==========================================================================

#: Held by this module alone. An ``ApprovedAction`` cannot be built without it.
_APPROVAL = object()


@dataclass(frozen=True, slots=True)
class ApprovedAction:
    """Permission to take one action, at one moment, on one episode.

    The only object an executor accepts. Constructing one requires ``_APPROVAL``,
    which is module-private, so the only way to *mint* one is through
    ``ComplianceGate.adjudicate``.

    Copying is a different matter and is deliberately allowed: ``copy``,
    ``deepcopy`` and ``pickle`` all bypass ``__init__`` and reproduce an
    approval unchanged. That is correct — a copy of a permission is the same
    permission. What is blocked is *altering* one, which is why the token is
    init-only rather than stored.

    This is not paranoia about malicious code. It is about the ordinary failure
    where someone adds a call site next year and forgets the check — a single
    missing line, no error, tests still green. Here that mistake does not
    compile in the only sense Python has: the executor has nothing to accept.
    """

    episode_id: str
    action: Action
    at: dt.datetime
    token: InitVar[object] = None
    """Init-only, deliberately. An earlier version stored it as a field, and
    ``dataclasses.replace`` then forged approvals freely: it copies every field
    through, including the valid token, so an approved retry could be turned
    into an approved voice call. Init-only means ``replace`` has no token to
    pass and fails, and an existing approval carries no key to steal."""

    def __post_init__(self, token: object) -> None:
        if token is not _APPROVAL:
            raise PermissionError(
                "ApprovedAction cannot be constructed directly. Every action "
                "must be adjudicated by ComplianceGate.adjudicate."
            )


class ComplianceViolation(Exception):
    """Raised when an action is required to be permitted and is not."""


@dataclass(frozen=True, slots=True)
class Decision:
    """The gate's answer, with everything needed to defend it later."""

    request: Request
    verdict: Verdict
    rulings: tuple[Ruling, ...]
    binding: Ruling | None
    """The ruling that determined the verdict. ``None`` when nothing objected."""

    approval: ApprovedAction | None

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW

    earliest_allowed_at: dt.datetime | None = None
    """When a deferred action becomes permissible. ``None`` unless DEFER.

    Resolved by the gate, not derived here, and the difference is a bug this
    module shipped with. The old version took the ``max`` of every deferral,
    which is only correct if each constraint means "permitted from T onward".
    ``ExecutionWindow`` is not that shape — it is a set of intervals, and
    leaving one does not put you inside the next. A UPI retry blocked at 11:00
    by both an immature notice (matures 18:00) and the execution window (opens
    13:00) got 18:00, which lands in the 17:00-21:30 dead zone. The gate handed
    the sequencer a time it would itself refuse.

    It was also populated on DENY decisions, so a permanently denied action
    carried a reschedule time and a caller keying off the field rather than the
    verdict would retry it forever.
    """

    @property
    def regulatory_rulings(self) -> tuple[Ruling, ...]:
        return tuple(r for r in self.rulings if r.basis is Basis.REGULATORY)

    @property
    def governing(self) -> tuple[Ruling, ...]:
        """Every ruling at the deciding severity, not just the first one.

        ``max`` returns the earliest maximum, so with two simultaneous
        deferrals the "binding" rule was always whichever sat earlier in the
        rule list — and the audit line never mentioned the constraint that
        actually governed the outcome.
        """
        if self.binding is None:
            return ()
        top = _SEVERITY[self.binding.verdict]
        return tuple(r for r in self.rulings if _SEVERITY[r.verdict] == top)

    def explain(self) -> str:
        """One line, quotable into an audit record or a merchant-facing report."""
        if self.binding is None:
            return f"{self.request.action} allowed; no rule objected"
        cited = "; ".join(
            f"{r.rule_id} [{r.basis}]: {r.reason}" for r in self.governing
        )
        when = (
            f" (earliest {self.earliest_allowed_at:%Y-%m-%d %H:%M})"
            if self.verdict is Verdict.DEFER and self.earliest_allowed_at
            else ""
        )
        return f"{self.request.action} {self.verdict} by {cited}{when}"


@dataclass
class ComplianceGate:
    """Applies every rule to every proposed action.

    **All rules run, always.** No short-circuit on the first denial, which costs
    a negligible amount of work and buys a complete record: a merchant asking
    "why did nothing happen for this customer" gets every reason at once rather
    than peeling them off one failed attempt at a time.
    """

    rules: Sequence[Rule] = DEFAULT_RULES
    audit: list[Decision] = field(default_factory=list)

    max_deferral_hops: int = 8
    """How many times a deferral may be re-adjudicated before giving up.

    Each hop moves the candidate time strictly forward, and there are three
    execution windows a day, so convergence is fast in practice. The bound
    exists so a pathological rule set produces a denial rather than a hang.
    """

    def _rulings_for(self, request: Request) -> tuple[Ruling, ...]:
        return tuple(
            ruling
            for ruling in (rule.check(request) for rule in self.rules)
            if ruling is not None
        )

    def _settle_deferral(self, request: Request) -> tuple[dt.datetime | None, tuple[Ruling, ...]]:
        """Find a moment that satisfies every constraint at once.

        Iterating is the right shape and, on the current taxonomy, it never
        actually iterates. Taking the latest proposed time assumes each
        constraint means "permitted from T onward", and the execution window
        does not: satisfying the notice requirement can land inside a window
        that is closed. So the candidate is re-adjudicated until the rules stop
        moving it.

        **How often the second hop happens: never.** An exhaustive sweep of
        148,608 requests — every failure code, every action ``legal_actions``
        admits for it, 48 times of day, every notice age and attempt state —
        found no case where two rules deferred at once. Hop counts came back
        ``{1: 24924}``, and ``max_deferral_hops`` was never approached.

        The two rules that could collide cannot co-occur.
        ``PreDebitNotificationRequired`` binds only on ``MERCHANT_FIX``, whose
        sole taxonomy member is a card failure; ``ExecutionWindow`` binds only
        when the presenting rail is UPI, which for a card episode requires
        ``RETRY_ALT_RAIL`` — and ``legal_actions(MERCHANT_FIX)`` excludes it.
        ``QuietHours`` and ``ExecutionWindow`` are disjoint by construction,
        since no action is both a contact and a debit.

        Forcing a request past the taxonomy reproduces the documented
        11:00 / 18:00 / 13:00 case exactly and settles it to 21:30, so the
        arithmetic is right. It stays, because the collision becomes reachable
        the moment a second ``MERCHANT_FIX`` code is added on another rail, and
        because a gate that hands back a time it would itself refuse is the
        worst failure this class has. But it is defensive code rather than
        load-bearing code, and earlier versions of this docstring — and of
        PROGRESS D21 — presented it as the latter.

        Returns ``(None, rulings)`` when no reachable moment clears every rule.
        """
        at = request.at
        rulings = self._rulings_for(request)
        for _ in range(self.max_deferral_hops):
            proposed = [
                r.earliest_allowed_at
                for r in rulings
                if r.earliest_allowed_at is not None
            ]
            if any(r.verdict is Verdict.DENY for r in rulings):
                return None, rulings
            if not proposed:
                return at, rulings
            nxt = max(proposed)
            if nxt <= at:
                return at, rulings
            at = nxt
            rulings = self._rulings_for(_at_time(request, at))
        return None, rulings

    def adjudicate(self, request: Request, record: bool = True) -> Decision:
        """Apply every rule and decide.

        ``record=False`` adjudicates without writing to the audit trail, for
        hypothetical probes. The trail is what a merchant reads to answer "why
        did nothing happen for this customer", and filling it with counterfactual
        questions the sequencer asked itself makes that record useless — one
        call to ``permitted_actions`` over eleven candidates was writing eleven
        rows, none of which corresponded to anything that happened.
        """
        rulings = self._rulings_for(request)
        binding = max(rulings, key=lambda r: _SEVERITY[r.verdict], default=None)
        verdict = binding.verdict if binding is not None else Verdict.ALLOW

        earliest: dt.datetime | None = None
        if verdict is Verdict.DEFER:
            earliest, _ = self._settle_deferral(request)
            if earliest is None:
                verdict = Verdict.DENY
                binding = Ruling(
                    "GATE.UNREACHABLE",
                    Basis.POLICY,
                    Verdict.DENY,
                    "no moment within the deferral horizon satisfies every "
                    "constraint at once",
                )
                rulings = rulings + (binding,)

        decision = Decision(
            request=request,
            verdict=verdict,
            rulings=rulings,
            binding=binding,
            earliest_allowed_at=earliest,
            approval=(
                ApprovedAction(
                    request.episode_id, request.action, request.at, _APPROVAL
                )
                if verdict is Verdict.ALLOW
                else None
            ),
        )
        if record:
            self.audit.append(decision)
        return decision

    def require(self, request: Request) -> ApprovedAction:
        """Adjudicate and raise unless permitted.

        For call sites that have already decided an action must happen. Raising
        beats returning ``None`` there, because a ``None`` gets ignored.
        """
        decision = self.adjudicate(request)
        if decision.approval is None:
            raise ComplianceViolation(decision.explain())
        return decision.approval

    def permitted_actions(
        self, request_for: Iterable[tuple[Action, dt.datetime]], base: Request
    ) -> tuple[Action, ...]:
        """Which of a set of candidate actions the gate would allow right now.

        A convenience for a caller that would rather ask than propose. The
        shipped sequencer does *not* use it — it calls ``adjudicate`` per
        candidate, because it needs the deferral time and the binding ruling
        for its own audit trail, and this returns neither. The docstring used
        to call this "the sequencer's real entry point", which was true of a
        design that was never built.
        """
        allowed = []
        for action, at in request_for:
            probe = Request(**{**_as_dict(base), "action": action, "at": at})
            if self.adjudicate(probe, record=False).allowed:
                allowed.append(action)
        return tuple(allowed)

    # -- reporting ---------------------------------------------------------

    def unverified_rules(self) -> tuple[str, ...]:
        """Regulatory rules resting on sources that have not been confirmed.

        Reported rather than hidden. Every one of them is currently unverified,
        and an audit trail that did not say so would be claiming a diligence
        that was never done.
        """
        unconfirmed = {
            key for key, src in REGULATORY_SOURCES.items() if not src.is_confirmed
        }
        return tuple(
            sorted(
                rule.rule_id
                for rule in self.rules
                if getattr(rule, "source_key", "") in unconfirmed
                and getattr(rule, "source_key", "")
            )
        )

    def audit_trail(self) -> tuple[dict[str, object], ...]:
        """Every decision made, flattened for logging or a report."""
        return tuple(
            {
                "episode_id": d.request.episode_id,
                "customer_id": d.request.customer_id,
                "at": d.request.at,
                "action": str(d.request.action),
                "verdict": str(d.verdict),
                "binding_rule": d.binding.rule_id if d.binding else "",
                "basis": str(d.binding.basis) if d.binding else "",
                "reason": d.binding.reason if d.binding else "",
                "earliest_allowed_at": d.earliest_allowed_at,
                "rules_fired": len(d.rulings),
            }
            for d in self.audit
        )


def _as_dict(request: Request) -> dict[str, object]:
    """Field values by name.

    ``dataclasses.fields`` rather than ``__slots__``: the two agree today, but
    ``__slots__`` is an implementation detail of ``slots=True`` and would
    silently pick up any non-field slot added later.
    """
    return {f.name: getattr(request, f.name) for f in fields(request)}


def _at_time(request: Request, at: dt.datetime) -> Request:
    return Request(**{**_as_dict(request), "at": at})
