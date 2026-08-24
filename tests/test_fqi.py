"""Tests for fitted Q-iteration.

The load-bearing one is the reward reconciliation. Everything downstream — the
value function, the policy, the reported net value — is measuring whatever the
reward says it is measuring, so if the reward and the harness's ledger disagree,
every number is quietly about a different quantity and nothing else in this file
would notice.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from rebound.economics import LTV_HORIZON_CYCLES
from rebound.eval.splits import all_splits
from rebound.fqi import FittedQ, build_transitions, fit_fitted_q
from rebound.sim.dataset import GenerationConfig, generate_log
from rebound.taxonomy import Action, legal_actions

CONFIG = GenerationConfig(
    n_customers=300,
    start=dt.date(2025, 1, 1),
    end=dt.date(2025, 9, 30),
    seed=515,
)


@pytest.fixture(scope="module")
def log():
    return generate_log(CONFIG)


@pytest.fixture(scope="module")
def transitions(log):
    return build_transitions(log)


# ==========================================================================
# The reward
# ==========================================================================


def test_rewards_reconcile_with_the_episode_ledger(log, transitions):
    """Every episode's rewards must sum to the net the harness scores.

    This is the join between what the policy optimises and what the evaluation
    reports. If it drifts, the Q function is maximising a quantity nobody
    measures and the comparison against the baselines becomes meaningless
    without anything failing.
    """
    per_episode = (
        pd.Series(transitions.reward)
        .groupby(transitions.frame["episode_id"].to_numpy(), observed=True)
        .sum()
    )
    truth = transitions.frame.groupby("episode_id", observed=True)[
        "episode_net_paise"
    ].max()

    aligned = per_episode.reindex(truth.index)
    np.testing.assert_allclose(aligned.to_numpy(), truth.to_numpy(), atol=1e-6)


def test_passive_churn_is_charged_to_the_last_decision(log):
    """`close_episode` settles an abandoned episode outside the decision log, so
    passive revocation is never a row of its own — 73% of all revocations have
    no action to blame. Dropping it would make giving up look free, which is
    precisely the cost the stopping decision has to weigh.

    The earlier version of this test re-implemented the arithmetic and asserted
    against its own copy. It never touched `transitions.reward`, and passed
    with the charge attached to the *first* decision instead of the last.
    """
    transitions = build_transitions(log)
    frame = transitions.frame
    reward = transitions.reward

    row_destroyed = (
        frame["revoked"].astype(int) * frame["amount_paise"] * LTV_HORIZON_CYCLES
    ).to_numpy()
    plain = (
        frame["recovered_paise"].to_numpy()
        - frame["cost_paise"].to_numpy()
        - row_destroyed
    )
    # Whatever the reward carries beyond the per-row ledger is the passive charge.
    extra = plain - reward

    passive_episodes = frame.loc[extra > 0, "episode_id"].unique()
    assert len(passive_episodes) > 0, "no passive churn in this log to test"

    for episode_id in passive_episodes[:200]:
        rows = np.flatnonzero(frame["episode_id"].to_numpy() == episode_id)
        charged = np.flatnonzero(extra[rows] > 0)
        assert charged.tolist() == [len(rows) - 1], (
            f"{episode_id}: passive charge on row(s) {charged.tolist()} "
            f"of {len(rows)}, expected only the last"
        )


def test_truncated_episodes_are_distinguished_from_real_endings(log):
    """The generator stops at `max_ladder_steps`, so a last row that neither
    collected, nor lost the mandate, nor chose to stop was cut off rather than
    concluded — 53% of last rows. Treating those as evidence that acting ends
    badly conflates "the ladder ran out of budget" with "the episode was lost".
    """
    transitions = build_transitions(log)
    frame = transitions.frame

    assert transitions.truncated.sum() > 0
    assert not (transitions.truncated & ~transitions.terminal).any(), (
        "a truncated row must also be a last row"
    )

    cut = frame[transitions.truncated]
    assert not cut["succeeded"].astype(bool).any()
    assert not cut["revoked"].astype(bool).any()
    assert (cut["action"].astype(str) != str(Action.STOP)).all()

    ended = frame[transitions.terminal & ~transitions.truncated]
    concluded = (
        ended["succeeded"].astype(bool)
        | ended["revoked"].astype(bool)
        | (ended["action"].astype(str) == str(Action.STOP))
    )
    assert concluded.all()


# ==========================================================================
# The transition structure
# ==========================================================================


def test_successors_stay_inside_their_own_episode(transitions):
    """A successor from a different episode would let value flow between
    unrelated customers, and the resulting Q would look fine."""
    frame = transitions.frame
    live = ~transitions.terminal
    successor = transitions.next_row[live]

    episodes = frame["episode_id"].to_numpy()
    assert (episodes[successor] == episodes[live]).all()

    steps = frame["decision_index"].to_numpy()
    assert (steps[successor] - steps[live] == 1).all()


def test_terminals_are_exactly_the_last_decision_of_each_episode(transitions):
    frame = transitions.frame
    assert transitions.terminal.sum() == frame["episode_id"].nunique()
    assert ((transitions.next_row == -1) == transitions.terminal).all()


def test_a_log_missing_its_structure_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        build_transitions(pd.DataFrame({"episode_id": ["A"], "action": ["stop"]}))


# ==========================================================================
# The value function
# ==========================================================================


@pytest.fixture(scope="module")
def fitted(log):
    return fit_fitted_q(all_splits(log)["time"].train, sweeps=4)


def test_the_continuation_never_considers_an_illegal_action(log):
    """Taking `max` over every action lets the value function assume a
    continuation the taxonomy forbids — a nudge to a closed account — and that
    optimism propagates backwards into every earlier decision."""
    q = FittedQ(sweeps=2)
    transitions = build_transitions(all_splits(log)["time"].train)
    q.spec_ = None
    q.action_levels_ = tuple(sorted({str(a) for a in Action}))

    allowed = q._legal_mask(transitions)
    codes = transitions.failure_code[
        transitions.next_row[~transitions.terminal]
    ]

    for action, mask in allowed.items():
        for i in np.flatnonzero(mask)[:40]:
            assert Action(action) in legal_actions(codes[i])
        for i in np.flatnonzero(~mask)[:40]:
            assert Action(action) not in legal_actions(codes[i])


def test_stopping_recovers_the_passive_churn_rate(fitted, log):
    """`Q(s, STOP)` should equal minus the churn rate times the LTV at risk.

    **Not an independent confirmation of the closed-form EV, and an earlier
    version of this test claimed it was.** On every nonzero stop row the logged
    reward is exactly `-1.0 x amount x LTV_HORIZON_CYCLES`, so `Q(s, STOP)` is
    definitionally that same closed form with the churn rate *estimated* from
    the data rather than supplied. One computation, not two witnesses.

    What it does check is that the estimate is a plausible probability and that
    stopping is never valued as free.
    """
    test = build_transitions(all_splits(log)["time"].test).frame
    q_stop = fitted.predict(test.assign(action=str(Action.STOP))).mean()
    at_risk = test["amount_paise"].mean() * LTV_HORIZON_CYCLES

    implied_rate = -q_stop / at_risk
    assert 0.01 < implied_rate < 0.25, (
        f"Q(stop) implies a churn rate of {implied_rate:.4f}, which is not a "
        f"plausible passive revocation probability"
    )


def test_backward_induction_moves_the_values(log):
    """The test the suite did not have.

    The whole 480-test suite passed with backward induction disabled, so every
    mutation inside the recursion was untested by construction. This asserts the
    recursion does something a one-step reward model cannot: the continuation
    term has to move Q by an amount comparable to the spread between actions,
    or it is not carrying information the policy could act on.

    **It does not assert the direction, and the first version of it did.** That
    version expected induction to *raise* the value of a nudge — a nudge
    collects nothing, so only a continuation can justify it, which is the
    original "never contact anyone" bug restated. Measured, induction lowers it
    (-631 to -973 rupees). The cause is real and is recorded on
    ``Transitions.truncated``: 53% of last rows are the generator's step budget
    running out, each carrying a full twelve-cycle churn charge and no
    continuation, so induction faithfully propagates "episodes end badly"
    backwards. Asserting the direction I expected would have meant deleting a
    true finding to keep a comfortable test.
    """
    train = all_splits(log)["time"].train
    test = build_transitions(all_splits(log)["time"].test).frame
    probe = test.assign(action=str(Action.NUDGE_SMS))

    one_step = fit_fitted_q(train, sweeps=1)
    with_lookahead = fit_fitted_q(train, sweeps=4)

    moved = abs(
        with_lookahead.predict(probe).mean() - one_step.predict(probe).mean()
    )

    # The spread the policy actually discriminates on, for scale.
    spread = abs(
        with_lookahead.predict(test.assign(action=str(Action.RETRY_SAME_RAIL))).mean()
        - with_lookahead.predict(test.assign(action=str(Action.STOP))).mean()
    )

    assert spread > 0
    assert moved > 0.1 * spread, (
        f"induction moved Q by {moved / 100:,.0f} rupees against an "
        f"action spread of {spread / 100:,.0f}; the continuation term is "
        f"contributing nothing the policy could act on"
    )


def test_retrying_is_valued_above_calling(fitted, log):
    """A retry costs a gateway fee and carries no revocation hazard at all in
    the world; a voice call is the most intrusive thing available.

    Weak on its own — the one-step reward already separates these by 75,534
    paise, so this passes without any induction at all. It is kept as a guard
    against the *sign* inverting, which is what the episode-level revocation
    label did, and paired with the ordering check below so that a Q collapsing
    to a constant plus noise is caught.
    """
    test = build_transitions(all_splits(log)["time"].test).frame
    values = {
        action: fitted.predict(test.assign(action=str(action))).mean()
        for action in (
            Action.RETRY_SAME_RAIL,
            Action.SEND_COLLECT_LINK,
            Action.VOICE_CALL,
            Action.STOP,
        )
    }
    assert values[Action.RETRY_SAME_RAIL] > values[Action.VOICE_CALL]

    # A constant-plus-noise Q would not reproduce the domain ordering.
    assert values[Action.RETRY_SAME_RAIL] > values[Action.SEND_COLLECT_LINK]
    assert values[Action.SEND_COLLECT_LINK] > values[Action.STOP]


def test_the_ensemble_is_split_on_whole_episodes(fitted, log):
    """Splitting rows would put a decision in one half and its own successor in
    the other, so each model would be evaluated on continuations it had trained
    against — the double-Q construction would then be double-counting its own
    estimate rather than checking it."""
    frame = build_transitions(all_splits(log)["time"].train).frame
    parts = fitted._partition(frame)

    episodes = frame["episode_id"].to_numpy()
    seen = [set(episodes[rows]) for rows in parts]
    for i, left in enumerate(seen):
        for right in seen[i + 1 :]:
            assert not (left & right)
    assert sum(len(rows) for rows in parts) == len(frame)


def test_pessimism_actually_lowers_something(fitted, log):
    """`min <= mean` is an arithmetic identity and the earlier version of this
    test asserted only that. The question worth asking is whether the ensemble
    disagrees at all — if one member were uniformly lower, `min` would just
    return that member and the ensemble would be decorative."""
    test = build_transitions(all_splits(log)["time"].test).frame.head(1500)
    probe = test.assign(action=str(Action.SEND_COLLECT_LINK))

    fitted.pessimistic = True
    low = fitted.predict(probe)
    fitted.pessimistic = False
    mean = fitted.predict(probe)
    fitted.pessimistic = True

    assert (low <= mean + 1e-9).all()
    strictly_lower = (low < mean - 1e-9).mean()
    assert strictly_lower > 0.5, (
        f"pessimism bites on only {strictly_lower:.1%} of rows; the ensemble "
        f"members are not disagreeing and `min` is decorative"
    )


def test_an_unfitted_model_refuses_to_predict():
    with pytest.raises(RuntimeError, match="not fitted"):
        FittedQ().predict(pd.DataFrame({"action": ["stop"]}))
