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


def test_expected_value_is_measured_against_stopping():
    """Absolute probabilities price the wrong comparison.

    Customers revoke without being contacted — the measured floor is 8.78%
    under a policy that does nothing — so charging an action the full
    revocation probability bills it for churn that was already coming. At a
    12-cycle horizon that term runs an order of magnitude above the recovery
    term, so the double-count decides the answer rather than nudging it.
    """
    amount = 1_000_00
    destroyed = revocation_cost_paise(amount)

    candidate = Candidate(
        action=Action.RETRY_SAME_RAIL,
        at=NOW,
        p_recover=0.40,
        p_revoke=0.07,
        value_paise=amount,
        cost_paise=0,
        revocation_cost_paise=destroyed,
        baseline_recover=0.02,
        baseline_revoke=0.065,
    )

    absolute = 0.40 * amount - 0.07 * destroyed
    assert absolute < 0, "the absolute form should look unprofitable here"
    assert candidate.expected_value_paise > 0, (
        "the marginal form should find this worth doing: it lifts recovery 38 "
        "points for half a point of extra revocation risk"
    )


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


def test_an_action_is_never_credited_for_preventing_revocation():
    """Unclamped, the marginal term pays out for a model artifact.

    30.7% of candidates came back with negative marginal revocation, and at a
    12-cycle horizon the observed minimum of -0.0659 hands an action +0.79x the
    amount — enough to make a voice call profitable on any episode. The credit
    is not a finding: in the log ``stop`` has the *highest* revocation rate
    because the behavioural policy stops on episodes already lost.
    """
    amount = 1_000_00
    candidate = Candidate(
        action=Action.VOICE_CALL,
        at=NOW,
        p_recover=0.10,
        p_revoke=0.01,
        value_paise=amount,
        cost_paise=0,
        revocation_cost_paise=revocation_cost_paise(amount),
        baseline_recover=0.10,
        baseline_revoke=0.0759,
    )
    assert candidate.p_revoke - candidate.baseline_revoke < 0
    assert candidate.marginal_revocation == 0.0
    assert candidate.expected_value_paise <= 0, (
        "an action with no recovery lift must not be profitable on a "
        "revocation credit alone"
    )


def test_the_do_nothing_baseline_is_priced_at_the_candidate_s_own_time(episode):
    """A single baseline at ``now`` confounds the action effect with a time shift.

    84% of candidates are scheduled at some other moment, median 48 hours away,
    and seven features derive from the timestamp — so the difference being
    maximised was the action effect plus a two-to-seven-day shift, which is the
    quantity the timing head is separately choosing.
    """
    seq = sequencer(
        {a: 0.3 for a in _all_action_names()},
        {a: 0.05 for a in _all_action_names()},
    )
    captured: dict[str, pd.DataFrame] = {}
    original = seq.pricer.frame

    def spy(ep, candidates):
        built = original(ep, candidates)
        captured["frame"] = built
        captured["candidates"] = candidates
        return built

    seq.pricer.frame = spy  # type: ignore[method-assign]
    seq.decide(episode, NOW, DEADLINE)

    candidates = captured["candidates"]
    stop_times = {at for a, at in candidates if a is Action.STOP}
    other_times = {at for a, at in candidates if a is not Action.STOP}
    assert other_times <= stop_times, (
        f"candidate times with no baseline priced at the same moment: "
        f"{other_times - stop_times}"
    )


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
