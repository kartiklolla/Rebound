"""Tests for the sequencer.

Most of these run against a stub pricer rather than a fitted model. That is
deliberate: the decision logic — the stopping rule, the marginal arithmetic, the
gate interaction — is what can be wrong in ways a metric will not show, and
testing it against real models would make every case a minute long and tempt
nobody to write the awkward ones.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from rebound.compliance import ComplianceGate, Request, Verdict
from rebound.economics import revocation_cost_paise
from rebound.sequencer import (
    UNSERVABLE_COLUMNS,
    ActionPricer,
    Candidate,
    Sequencer,
    _observable,
)
from rebound.model import COLLECTING_ACTIONS
from rebound.sim.world import World
from rebound.taxonomy import Action, Rail

NOW = dt.datetime(2026, 4, 6, 14, 0)
DEADLINE = NOW + dt.timedelta(days=20)


class StubPricer(ActionPricer):
    """A pricer with opinions we control.

    ``recover`` and ``revoke`` map an action name to a probability. The STOP row
    the sequencer appends is priced like any other, which is what lets a test
    set the counterfactual explicitly.
    """

    def __init__(self, recover: dict[str, float], revoke: dict[str, float]):
        self.recover = recover
        self.revoke = revoke
        self.calls = 0
        # No real specs to compare, so the shared-encoding fast path is off and
        # the sequencer falls back to asking each head separately.
        self._shares_encoding = False
        self.passive_revocation_rate = 0.07
        # A flat one-further-contact curve: enough to exercise the externality
        # path without the test's expectations depending on a fitted estimate.
        self.future_contacts = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}

    def encode(self, frame):
        return None

    def frame(self, episode, candidates):
        self.calls += 1
        return pd.DataFrame(
            [{**_observable(episode, at), "action": str(a)} for a, at in candidates]
        )

    def _lookup(self, frame, table, default):
        return np.array([table.get(a, default) for a in frame["action"]])

    def _wrap(self, frame):
        return frame

    def predict_downstream(self, frame):
        return self._lookup(frame, self.recover, 0.1)

    def predict_revoke(self, frame):
        return self._lookup(frame, self.revoke, 0.05)

    @property
    def heads(self):
        pricer = self

        class _Heads:
            @staticmethod
            def predict_downstream(frame):
                return pricer.predict_downstream(frame)

            @staticmethod
            def predict_immediate(frame):
                # Later is always better, so a test can tell whether the timing
                # head was consulted at all.
                return np.linspace(0.1, 0.9, len(frame))

        return _Heads()

    @property
    def revocation(self):
        pricer = self

        class _Rev:
            @staticmethod
            def predict_proba(frame):
                return pricer.predict_revoke(frame)

        return _Rev()


@pytest.fixture(scope="module")
def episode():
    """A real EpisodeView, so the feature builder is exercised for real."""
    world = World(seed=4242)
    customers = world.sample_customers(20)
    mandates = world.sample_mandates(
        customers, dt.date(2024, 6, 1), dt.date(2026, 1, 1)
    )
    mandate = next(m for m in mandates if m.rail is Rail.ENACH)
    customer = next(c for c in customers if c.customer_id == mandate.customer_id)
    ep = world.open_episode(
        episode_id="EP_TEST_1",
        mandate=mandate,
        customer=customer,
        failure_code="NACH_INSUFFICIENT_FUNDS",
        failed_at=NOW - dt.timedelta(hours=2),
        cycles_elapsed=3,
    )
    return ep.view()


def sequencer(recover, revoke, **kwargs) -> Sequencer:
    return Sequencer(pricer=StubPricer(recover, revoke), **kwargs)


# ==========================================================================
# The marginal arithmetic
# ==========================================================================


def _candidate(**overrides) -> Candidate:
    base = dict(
        action=Action.RETRY_SAME_RAIL,
        at=NOW,
        p_recover=0.0,
        p_revoke=0.0,
        value_paise=1_000_00,
        cost_paise=0,
        revocation_cost_paise=revocation_cost_paise(1_000_00),
        passive_revocation_rate=0.07,
    )
    base.update(overrides)
    return Candidate(**base)  # type: ignore[arg-type]


def test_the_expected_value_matches_its_closed_form():
    """The three exact cases the derivation pins down.

    Stopping leaves the episode to one passive draw, worth `-h*D`. Acting has
    three exclusive outcomes: it recovers (`+A`, and no passive draw, because
    `close_episode` returns early on a resolved episode); it revokes at the
    action (`-D`, also no passive draw); or neither, and the passive draw
    happens anyway. Differencing gives

        EV = p_R*(A + h*D) - p_V*D*(1-h) - cost

    An earlier version subtracted a model-predicted baseline for a STOP row
    instead. That is a different marginalisation and a wrong one: it applied the
    `h*D` credit unconditionally, paying it to actions with `p_recover = 0`.
    """
    A = 1_000_00
    D = revocation_cost_paise(A)
    h = 0.07

    # Nothing can happen: both branches face the same draw, so only cost differs.
    assert _candidate(cost_paise=500).expected_value_paise == pytest.approx(-500)

    # Certain recovery is worth the amount plus the churn it prevents.
    assert _candidate(p_recover=1.0).expected_value_paise == pytest.approx(A + h * D)

    # Certain revocation costs the mandate, less the churn that was coming anyway.
    assert _candidate(p_revoke=1.0).expected_value_paise == pytest.approx(-D * (1 - h))


def test_the_revocation_charge_is_discounted_by_the_passive_rate():
    """An action that causes churn is charged for bringing it forward, not for
    all of it. The counterfactual is not a customer who stays forever — an
    unrecovered episode faced the passive draw regardless."""
    c = _candidate(p_revoke=1.0)
    assert c.revocation_charge_paise == pytest.approx(
        c.revocation_cost_paise * (1 - 0.07)
    )
    assert c.revocation_charge_paise < c.revocation_cost_paise


def test_a_recovery_is_worth_more_than_the_amount_collected():
    """Measured: P(revoke | recovered) = 0.0000 across 12,524 recovered
    episodes, against 0.0952 unrecovered. Crediting a recovery with only the
    amount collected ignores more than half of what it is worth, and the EV did
    exactly that through four revisions of this arithmetic."""
    from rebound.economics import LTV_HORIZON_CYCLES

    c = _candidate()
    assert c.recovery_value_paise == pytest.approx(
        1_000_00 * (1 + 0.07 * LTV_HORIZON_CYCLES)
    )
    assert c.recovery_value_paise > c.value_paise

    # With no passive churn in the world it collapses to the amount, so the
    # term is additive rather than a rescaling of something already correct.
    assert _candidate(passive_revocation_rate=0.0).recovery_value_paise == 1_000_00


def test_no_stop_rows_are_scored(episode):
    """The stop counterfactual is closed-form, so predicting it was estimating
    two known constants — `P(recover | stop)` is exactly 0 because a stop ends
    the episode, and `revoked` is structurally 0 on a stop row."""
    seq = sequencer(
        {a: 0.3 for a in _all_action_names()},
        {a: 0.05 for a in _all_action_names()},
    )
    captured: dict[str, object] = {}
    original = seq.pricer.frame

    def spy(ep, candidates):
        captured["candidates"] = candidates
        return original(ep, candidates)

    seq.pricer.frame = spy  # type: ignore[method-assign]
    seq.decide(episode, NOW, DEADLINE)
    assert Action.STOP not in {a for a, _ in captured["candidates"]}


# ==========================================================================
# Stopping
# ==========================================================================


def test_it_stops_when_nothing_beats_doing_nothing(episode):
    """The stopping rule, which is the whole reason to compute an EV at all."""
    seq = sequencer(recover={}, revoke={})
    seq.pricer.recover = {a: 0.02 for a in _all_action_names()}
    seq.pricer.revoke = {a: 0.5 for a in _all_action_names()}
    seq.pricer.recover["stop"] = 0.02
    seq.pricer.revoke["stop"] = 0.05

    decision = seq.decide(episode, NOW, DEADLINE)
    assert decision is not None
    assert decision.action is Action.STOP
    assert "non-positive expected value" in decision.reason


def test_it_acts_when_something_clearly_beats_doing_nothing(episode):
    seq = sequencer(recover={}, revoke={})
    seq.pricer.recover = {a: 0.02 for a in _all_action_names()}
    seq.pricer.revoke = {a: 0.05 for a in _all_action_names()}
    seq.pricer.recover["retry_same_rail"] = 0.70

    decision = seq.decide(episode, NOW, DEADLINE)
    assert decision is not None
    assert decision.action is Action.RETRY_SAME_RAIL


def test_stopping_is_recorded_with_a_reason(episode):
    """A merchant asking 'why did you give up on this customer' is asking about
    exactly these rows. Returning None would end the episode just as well and
    leave nothing to answer with."""
    seq = sequencer(
        {a: 0.0 for a in _all_action_names()},
        {a: 0.9 for a in _all_action_names()} | {"stop": 0.0},
    )
    seq.decide(episode, NOW, DEADLINE)
    assert seq.trail
    last = seq.trail[-1]
    assert last["chosen"] == "stop"
    assert last["considered"]


def _all_action_names() -> list[str]:
    return [str(a) for a in Action]


# ==========================================================================
# The gate is upstream, and has the last word
# ==========================================================================


def test_a_gate_denial_removes_the_action_entirely(episode):
    """Priced-then-filtered would let an illegal action win and then silently
    fall through to something else. Filtered-then-priced is the only order that
    makes the audit trail mean anything."""
    seq = sequencer(
        {a: 0.9 for a in _all_action_names()},
        {a: 0.0 for a in _all_action_names()},
    )
    decision = seq.decide(episode, NOW, DEADLINE)
    assert decision is not None

    if decision.action is not Action.STOP:
        request = Request.from_view(episode, decision.action, decision.at)
        assert ComplianceGate().adjudicate(request).verdict is Verdict.ALLOW


def test_every_returned_decision_is_one_the_gate_permits(episode):
    """Swept across a range of stub opinions, because the failure mode is a
    rare combination rather than a wrong branch."""
    for lift in (0.1, 0.5, 0.9):
        seq = sequencer(
            {a: lift for a in _all_action_names()} | {"stop": 0.0},
            {a: 0.0 for a in _all_action_names()},
        )
        decision = seq.decide(episode, NOW, DEADLINE)
        assert decision is not None
        if decision.action is Action.STOP:
            continue
        verdict = ComplianceGate().adjudicate(
            Request.from_view(episode, decision.action, decision.at)
        ).verdict
        assert verdict is Verdict.ALLOW, f"{decision.action} at {decision.at}"


def test_the_decision_is_never_after_the_deadline(episode):
    seq = sequencer(
        {a: 0.9 for a in _all_action_names()} | {"stop": 0.0},
        {a: 0.0 for a in _all_action_names()},
    )
    tight = NOW + dt.timedelta(hours=3)
    decision = seq.decide(episode, NOW, tight)
    assert decision is not None
    assert decision.at <= tight


# ==========================================================================
# Efficiency and the harness contract
# ==========================================================================


def test_one_frame_per_decision(episode):
    """Five sklearn calls per decision on three-row frames is what got the
    first version killed by the harness timeout at 4,883 of 6,898 episodes.
    The fix was structural, so it gets a structural test."""
    seq = sequencer(
        {a: 0.3 for a in _all_action_names()},
        {a: 0.05 for a in _all_action_names()},
    )
    seq.decide(episode, NOW, DEADLINE)
    assert seq.pricer.calls == 1


def test_reset_clears_state_between_runs(episode):
    # An acting decision, not a stop: only the chosen action is adjudicated
    # with record=True, so a run that stops leaves the gate's trail empty --
    # correct, since nothing was ever authorised.
    seq = sequencer(
        {a: 0.02 for a in _all_action_names()} | {"retry_same_rail": 0.8},
        {a: 0.05 for a in _all_action_names()},
    )
    decision = seq.decide(episode, NOW, DEADLINE)
    assert decision is not None and decision.action is not Action.STOP
    assert seq.trail and seq.gate.audit

    seq.reset()
    assert not seq.trail
    assert not seq.gate.audit


def test_deciding_twice_gives_the_same_answer(episode):
    seq = sequencer(
        {a: 0.3 for a in _all_action_names()},
        {a: 0.05 for a in _all_action_names()},
    )
    first = seq.decide(episode, NOW, DEADLINE)
    seq.reset()
    second = seq.decide(episode, NOW, DEADLINE)
    assert first == second


# ==========================================================================
# Train/serve agreement
# ==========================================================================


def test_the_served_feature_row_has_no_unservable_columns(episode):
    """The four ``cust_prior_*`` columns are cross-episode history an
    ``EpisodeView`` cannot supply. Filling them with zeros at inference is
    silent train/serve skew — the model scored on one distribution and deployed
    on another, with nothing raising."""
    row = _observable(episode, NOW)
    assert not set(row) & set(UNSERVABLE_COLUMNS)


def test_the_served_row_equals_the_training_row_value_for_value(episode):
    """The check that actually catches skew.

    A subset assertion on column *names* — which is what this test used to be —
    passes if the serving path silently drops half its features, and says
    nothing at all about whether a column that exists is computed the same way.
    A field that trains on ``ledger.attempts`` and serves on
    ``contacts_made`` has matching names and shifts every prediction.

    So this compares values, on a real episode, against the training-time
    builder itself.
    """
    from rebound.sim.dataset import _CustomerHistory, _observable_features

    world = World(seed=4242)
    customers = world.sample_customers(20)
    mandates = world.sample_mandates(
        customers, dt.date(2024, 6, 1), dt.date(2026, 1, 1)
    )
    mandate = next(m for m in mandates if m.rail is Rail.ENACH)
    customer = next(c for c in customers if c.customer_id == mandate.customer_id)
    live = world.open_episode(
        episode_id="EP_TEST_1",
        mandate=mandate,
        customer=customer,
        failure_code="NACH_INSUFFICIENT_FUNDS",
        failed_at=NOW - dt.timedelta(hours=2),
        cycles_elapsed=3,
    )

    at = NOW + dt.timedelta(hours=30)
    trained = _observable_features(
        live, _CustomerHistory(), Action.RETRY_SAME_RAIL, at, step=0
    )
    served = {**_observable(live.view(), at), "action": str(Action.RETRY_SAME_RAIL)}

    shared = set(served) & set(trained)
    assert len(shared) >= 20, "too few overlapping columns to be a real check"

    differences = {
        column: (trained[column], served[column])
        for column in sorted(shared)
        if trained[column] != served[column]
    }
    assert not differences, f"train/serve value mismatch: {differences}"

    # Everything the training builder produced and the serving path did not
    # must be a column we consciously declared unservable.
    missing = set(trained) - set(served) - set(UNSERVABLE_COLUMNS)
    identifiers = {
        "episode_id",
        "decision_index",
        "customer_id",
        "mandate_id",
        "failed_at",
        "decided_at",
    }
    assert not (missing - identifiers), missing - identifiers


# ==========================================================================
# What the review found
# ==========================================================================


def test_one_trail_row_per_decision(episode):
    """The row was appended before the stopping check and again inside it, so
    the trail reported actions that were never taken — 620 collect links where
    9 reached the world."""
    seq = sequencer(
        {a: 0.0 for a in _all_action_names()},
        {a: 0.9 for a in _all_action_names()} | {"stop": 0.0},
    )
    seq.decide(episode, NOW, DEADLINE)
    assert len(seq.trail) == 1

    seq.reset()
    seq.pricer.recover = {a: 0.02 for a in _all_action_names()}
    seq.pricer.recover["retry_same_rail"] = 0.9
    seq.pricer.revoke = {a: 0.05 for a in _all_action_names()}
    seq.decide(episode, NOW, DEADLINE)
    assert len(seq.trail) == 1


def test_stopping_reaches_the_compliance_audit_trail(episode):
    """The trail exists so a merchant can ask why nothing happened. Stopping is
    the decision that *is* nothing happening, and it was the one decision the
    gate never recorded."""
    seq = sequencer(
        {a: 0.0 for a in _all_action_names()},
        {a: 0.9 for a in _all_action_names()} | {"stop": 0.0},
    )
    decision = seq.decide(episode, NOW, DEADLINE)
    assert decision is not None and decision.action is Action.STOP

    trail = seq.gate.audit_trail()
    assert any(row["action"] == "stop" for row in trail), trail


def test_candidate_order_does_not_depend_on_set_iteration(episode):
    """``legal_actions`` returns a frozenset and StrEnum hashing is salted by
    PYTHONHASHSEED. That order reached ``max`` over expected values, and
    RETRY_SAME_RAIL and RETRY_ALT_RAIL share cost, value and revocation cost —
    so exact ties broke by list position and net value moved 2.2% between
    interpreters. A same-process rerun cannot see it; the ordering can."""
    seq = sequencer(
        {a: 0.3 for a in _all_action_names()},
        {a: 0.05 for a in _all_action_names()},
    )
    pairs = seq._permitted(episode, NOW, DEADLINE)
    actions = [str(a) for a, _ in pairs]
    assert actions == sorted(actions), actions


# ==========================================================================
# The production path
# ==========================================================================
#
# Everything above runs against StubPricer, which sets `_shares_encoding =
# False` — so it exercises the *fallback*. Review found the shared-encoding
# path, `predict_proba_prepared`, `_same_spec`, `fit_for_serving` and the
# no-contact configuration had zero coverage between them. A test suite that
# only tests the branch production does not take is worse than none, because it
# reports green over the untested code.


@pytest.fixture(scope="module")
def fitted():
    """Real fitted models. Small, but the genuine article."""
    from rebound.eval.splits import all_splits
    from rebound.sequencer import fit_for_serving
    from rebound.sim.dataset import GenerationConfig, generate_log

    log = generate_log(
        GenerationConfig(
            n_customers=250,
            start=dt.date(2025, 1, 1),
            end=dt.date(2025, 9, 30),
            seed=606,
        )
    )
    return all_splits(log)["time"], fit_for_serving(
        all_splits(log)["time"].train, max_iter=30
    )


def test_the_revocation_head_is_fitted_on_the_per_action_label(fitted):
    """`episode_revoked` is smeared across every row of an episode and inverts
    the ordering — it reads `stop` as the most dangerous action because the
    behavioural policy stops on episodes already lost."""
    from rebound.model import TARGET_ACTION_REVOKED

    _, pricer = fitted
    assert pricer.revocation.target == TARGET_ACTION_REVOKED


def test_prepared_prediction_matches_the_ordinary_one(fitted):
    """The optimisation must be invisible in the output, or it is a bug that
    happens to be fast."""
    split, pricer = fitted
    frame = split.test.drop(columns=list(UNSERVABLE_COLUMNS), errors="ignore").head(60)
    model = pricer.heads.downstream

    np.testing.assert_array_equal(
        model.predict_proba(frame),
        model.predict_proba_prepared(model.spec_.transform(frame)),
    )


def test_the_shared_encoding_is_used_and_changes_nothing(fitted):
    """Both heads read one encoded matrix. If they ever stop agreeing on how a
    frame encodes, sharing it would predict on mis-ordered columns and return
    confident nonsense with nothing raising."""
    split, pricer = fitted
    assert pricer._shares_encoding, "production path is not active"

    frame = split.test.drop(columns=list(UNSERVABLE_COLUMNS), errors="ignore").head(60)
    shared = pricer.encode(frame)
    assert shared is not None

    np.testing.assert_array_equal(
        pricer.heads.downstream.predict_proba_prepared(shared),
        pricer.heads.downstream.predict_proba(frame),
    )
    np.testing.assert_array_equal(
        pricer.revocation.predict_proba_prepared(shared),
        pricer.revocation.predict_proba(frame),
    )


def test_sharing_is_refused_when_the_two_specs_disagree(fitted):
    """The guard, not the happy path. Column order and category levels both
    have to match, and a missing spec must not be treated as a match."""
    from rebound.sequencer import _same_spec

    _, pricer = fitted
    spec = pricer.heads.downstream.spec_

    assert _same_spec(spec, pricer.revocation.spec_)
    assert not _same_spec(spec, None)
    assert not _same_spec(None, spec)

    import dataclasses

    reversed_columns = dataclasses.replace(
        spec, columns=tuple(reversed(spec.columns))
    )
    assert not _same_spec(spec, reversed_columns)

    first = next(iter(spec.categories))
    trimmed = dict(spec.categories)
    trimmed[first] = tuple(spec.categories[first][:-1])
    assert not _same_spec(spec, dataclasses.replace(spec, categories=trimmed))


def test_the_no_contact_configuration_never_contacts(fitted, episode):
    """Shipped as a second configuration and reported beside the first, so it
    has to actually do what its name says."""
    from rebound.compliance import DEFAULT_RULES, ComplianceGate, ContactCap
    from rebound.taxonomy import CUSTOMER_FACING_ACTIONS

    _, pricer = fitted
    rules = tuple(
        ContactCap(max_contacts=0) if isinstance(r, ContactCap) else r
        for r in DEFAULT_RULES
    )
    seq = Sequencer(pricer=pricer, gate=ComplianceGate(rules=rules))

    decision = seq.decide(episode, NOW, DEADLINE)
    assert decision is not None

    contactable = CUSTOMER_FACING_ACTIONS - {Action.SEND_PRE_DEBIT_NOTIFICATION}
    assert decision.action not in contactable

    permitted = {a for a, _ in seq._permitted(episode, NOW, DEADLINE)}
    assert not (permitted & contactable), permitted & contactable


# ==========================================================================
# What a recovery is worth
# ==========================================================================


def test_the_passive_rate_excludes_revocations_an_action_caused():
    """The double-count guard, and the reason the credited rate is 0.0700
    rather than 0.0952.

    26.6% of revocations in unrecovered episodes happened *at an action*, and
    the EV already charges for those through `marginal_revocation`. Counting
    them here as well would inflate every recovery by about 16%.
    """
    from rebound.sequencer import estimate_passive_revocation_rate

    frame = pd.DataFrame(
        [
            # Recovered: excluded from the denominator entirely.
            {"episode_id": "A", "episode_recovered": 1, "episode_revoked": 0, "revoked": 0},
            # Unrecovered, churned on its own -> counts.
            {"episode_id": "B", "episode_recovered": 0, "episode_revoked": 1, "revoked": 0},
            # Unrecovered, revoked AT an action -> must NOT count.
            {"episode_id": "C", "episode_recovered": 0, "episode_revoked": 1, "revoked": 0},
            {"episode_id": "C", "episode_recovered": 0, "episode_revoked": 1, "revoked": 1},
            # Unrecovered, survived.
            {"episode_id": "D", "episode_recovered": 0, "episode_revoked": 0, "revoked": 0},
        ]
    )
    # Three unrecovered episodes (B, C, D); only B is passive churn.
    assert estimate_passive_revocation_rate(frame) == pytest.approx(1 / 3)


def test_the_passive_rate_is_estimated_not_hardcoded(fitted):
    """A constant measured once and pasted into the source goes stale in
    silence, and this one multiplies a 12-cycle horizon."""
    split, pricer = fitted
    from rebound.sequencer import estimate_passive_revocation_rate

    assert pricer.passive_revocation_rate == pytest.approx(
        estimate_passive_revocation_rate(split.train)
    )
    assert 0.0 <= pricer.passive_revocation_rate < 0.5


# ==========================================================================
# The fatigue externality
# ==========================================================================


def test_a_contact_is_charged_for_the_contacts_that_come_after_it():
    """The defect a one-step expected value cannot see.

    `prior_contacts` is identical across every candidate at a decision point,
    so the head prices today's fatigue level correctly and the cost of *adding*
    a level is structurally invisible. Correcting the recovery credit alone —
    arithmetically right — moved contacts from 0.74 to 1.00 per episode and net
    value from -73,367 to -159,979, because a more accurate myopic objective
    buys more of exactly what makes `aggressive_contact` worst on the board.
    """
    c = _candidate(
        action=Action.VOICE_CALL,
        p_recover=0.3,
        fatigue_delta=0.01,
        future_contacts=1.5,
    )
    assert c.fatigue_externality_paise == pytest.approx(
        0.01 * 1.5 * c.revocation_charge_paise
    )

    import dataclasses as _dc

    unpriced = _dc.replace(c, fatigue_delta=0.0, future_contacts=0.0)
    assert c.expected_value_paise < unpriced.expected_value_paise


def test_an_action_that_creates_no_fatigue_is_charged_nothing():
    """A retry does not reach the customer, and the mandated pre-debit notice
    never draws the hazard in `world.apply` — so neither raises the fatigue
    level for anything that follows."""
    from rebound.sequencer import FATIGUE_ACTIONS

    assert str(Action.RETRY_SAME_RAIL) not in FATIGUE_ACTIONS
    assert str(Action.SEND_PRE_DEBIT_NOTIFICATION) not in FATIGUE_ACTIONS
    assert str(Action.VOICE_CALL) in FATIGUE_ACTIONS

    assert _candidate().fatigue_externality_paise == 0.0


def test_the_externality_uses_a_shadow_row_one_fatigue_level_up(episode):
    """The increment comes from the head's own prediction at
    `prior_contacts + 1`, not from a chosen coefficient."""
    seq = sequencer(
        {a: 0.3 for a in _all_action_names()},
        {a: 0.05 for a in _all_action_names()},
    )
    captured: dict[str, pd.DataFrame] = {}
    original = seq.pricer.frame

    def spy(ep, candidates):
        built = original(ep, candidates)
        captured["frame"] = built
        return built

    seq.pricer.frame = spy  # type: ignore[method-assign]
    seq.decide(episode, NOW, DEADLINE)

    built = captured["frame"]
    contact_rows = built[built["action"].isin(FATIGUE_ACTIONS_STR)]
    if len(contact_rows):
        base = episode.contacts_made
        assert (contact_rows["prior_contacts"] == base).all()


def test_the_expected_further_contacts_curve_falls_with_fatigue(fitted):
    """Measured from the log, not bounded by the contact cap. The cap-derived
    bound is wrong against this data: at k=2 it says zero further contacts and
    the log shows 0.396, because the behavioural policy has no gate binding it.
    """
    split, pricer = fitted
    curve = pricer.future_contacts
    assert curve, "no curve estimated"

    values = [curve[k] for k in sorted(curve)]
    assert all(a >= b for a, b in zip(values, values[1:])), curve
    assert all(v >= 0 for v in values)


def test_a_negative_fatigue_increment_is_not_paid_out():
    """The head's fatigue slope is a coarse step function, so an increment can
    come back slightly negative from tree quantisation. A contact making later
    contacts *safer* is not something this data supports, and paying for it
    would be the mirror of the revocation-credit bug that had 30.7% of
    candidates being rewarded for churn they did not prevent."""
    c = _candidate(
        action=Action.VOICE_CALL, fatigue_delta=0.0, future_contacts=1.5
    )
    assert c.fatigue_externality_paise == 0.0


FATIGUE_ACTIONS_STR = {
    "nudge_sms",
    "nudge_whatsapp",
    "nudge_email",
    "voice_call",
    "send_collect_link",
    "request_remandate",
    "request_mandate_amendment",
}


# ==========================================================================
# The fitted-Q policy
# ==========================================================================


@pytest.fixture(scope="module")
def q_policy(fitted):
    from rebound.fqi import fit_fitted_q
    from rebound.sequencer import QSequencer

    split, pricer = fitted
    q = fit_fitted_q(split.train, sweeps=3)
    return QSequencer(q=q, pricer=pricer), q


def test_the_q_policy_stops_when_stopping_is_worth_more(q_policy, episode):
    """`Q(s, STOP)` competes on the same scale as every other action, so
    stopping needs no threshold. The EV sequencer needed a rule, a marginal
    baseline and a clamp to express the same idea.

    The earlier version asserted only that STOP appeared in `considered`, which
    `decide` appends unconditionally — it would have passed with the stopping
    logic deleted. This drives the value function instead: with every action
    priced below STOP, the policy must actually choose to stop.
    """
    import numpy as np

    policy, _ = q_policy
    policy.reset()

    real = policy.q.predict

    def stop_dominates(frame):
        return np.where(
            frame["action"].astype(str) == str(Action.STOP), 0.0, -1_000_000.0
        )

    policy.q.predict = stop_dominates  # type: ignore[method-assign]
    try:
        decision = policy.decide(episode, NOW, DEADLINE)
    finally:
        policy.q.predict = real  # type: ignore[method-assign]

    assert decision is not None
    assert decision.action is Action.STOP, decision


def test_the_q_policy_acts_when_acting_is_worth_more(q_policy, episode):
    """The mirror. If the policy stopped regardless of the values, the test
    above would pass for the wrong reason."""
    import numpy as np

    policy, _ = q_policy
    policy.reset()
    real = policy.q.predict

    def retry_dominates(frame):
        return np.where(
            frame["action"].astype(str) == str(Action.RETRY_SAME_RAIL),
            0.0,
            -1_000_000.0,
        )

    policy.q.predict = retry_dominates  # type: ignore[method-assign]
    try:
        decision = policy.decide(episode, NOW, DEADLINE)
    finally:
        policy.q.predict = real  # type: ignore[method-assign]

    assert decision is not None
    assert decision.action is Action.RETRY_SAME_RAIL, decision


def test_the_timing_head_picks_the_moment_not_q(q_policy, episode):
    """Q is fitted on logged transitions where timing was never randomised, so
    `days_since_failure` is entangled with how many attempts already failed. It
    learns "later decisions are worse" rather than "waiting refills the
    account", and without a timing head it acts immediately on nearly
    everything — 82.7% of decisions at zero delay against the log's 79-hour
    median. That is `immediate_retry` rediscovered.

    The earlier version asserted only structural identities of `_choose_times`
    and passed with `predict_immediate` replaced by a constant. This one changes
    what the timing head *says* and requires the chosen moment to follow.
    """
    import numpy as np

    policy, _ = q_policy
    pairs = policy._expand(episode, NOW, DEADLINE)
    collecting = [(a, t) for a, t in pairs if str(a) in COLLECTING_ACTIONS]
    if len({t for _, t in collecting}) < 2:
        pytest.skip("no timing choice available for this episode")

    heads = policy.pricer.heads
    real = heads.predict_immediate

    def prefer(index: int):
        def scorer(frame):
            scores = np.zeros(len(frame))
            scores[index % len(frame)] = 1.0
            return scores

        return scorer

    chosen = []
    for index in (0, -1):
        heads.predict_immediate = prefer(index)  # type: ignore[method-assign]
        try:
            picked = policy._choose_times(episode, pairs)
        finally:
            heads.predict_immediate = real  # type: ignore[method-assign]
        chosen.append({a: t for a, t in picked if str(a) in COLLECTING_ACTIONS})

    assert chosen[0] != chosen[1], (
        "the chosen moment did not follow the timing head's ranking"
    )


def test_the_q_policy_never_returns_an_action_the_gate_refuses(q_policy, episode):
    """Q ranks; compliance disposes. A refused winner falls through to stopping
    rather than silently taking the runner-up, because the recorded reason has
    to match the action actually taken."""
    policy, _ = q_policy
    decision = policy.decide(episode, NOW, DEADLINE)
    assert decision is not None

    # The policy's OWN gate, not a fresh one. A fresh ComplianceGate carries no
    # audit state and is strictly more permissive, so re-adjudicating with one
    # could pass while the policy's gate would have refused.
    verdict = policy.gate.adjudicate(
        Request.from_view(episode, decision.action, decision.at)
    ).verdict
    assert verdict is Verdict.ALLOW


def test_the_q_policy_records_what_it_considered(q_policy, episode):
    policy, _ = q_policy
    policy.reset()
    policy.decide(episode, NOW, DEADLINE)

    assert len(policy.trail) == 1
    row = policy.trail[-1]
    assert row["episode_id"] == episode.episode_id
    assert row["considered"]
    assert isinstance(row["q_paise"], float)
    assert policy.gate.audit, "the chosen action must reach the compliance trail"
