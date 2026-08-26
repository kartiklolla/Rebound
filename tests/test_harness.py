"""Tests for the rollout harness and the baseline policies."""

from __future__ import annotations

import datetime as dt

import pytest

from rebound.eval.baselines import (
    AggressiveContact,
    DispositionAwareRules,
    FixedLadder,
    NoRecovery,
    default_baselines,
)
from rebound.eval.harness import build_eval_batch, evaluate_all, evaluate_policy
from rebound.policy import Decision, Policy
from rebound.sim.world import World
from rebound.taxonomy import Action, is_terminal


@pytest.fixture(scope="module")
def setup():
    world = World(seed=1234)
    customers = world.sample_customers(900)
    mandates = world.sample_mandates(
        customers, dt.date(2024, 1, 1), dt.date(2025, 6, 30)
    )
    world.calibrate(
        customers,
        mandates,
        [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(14)],
    )
    batch = build_eval_batch(
        world, customers, mandates, dt.date(2026, 1, 1), dt.date(2026, 2, 28)
    )
    return world, batch


# ==========================================================================
# The harness
# ==========================================================================


def test_batch_is_non_trivial(setup):
    _, batch = setup
    assert len(batch) > 200
    assert len({spec.failure_code for spec in batch}) > 5


def test_evaluation_is_reproducible(setup):
    world, batch = setup
    first = evaluate_policy(world, FixedLadder(), batch, seed=77)
    second = evaluate_policy(world, FixedLadder(), batch, seed=77)
    assert first.report.net_paise == second.report.net_paise
    assert first.report.recovery_rate == second.report.recovery_rate


def test_common_random_numbers_pair_the_policies(setup):
    """Two runs of the same policy at the same seed must agree exactly, which
    is what makes a difference between two *different* policies attributable to
    the policy rather than to luck."""
    world, batch = setup
    a = evaluate_policy(world, NoRecovery(), batch, seed=5)
    b = evaluate_policy(world, NoRecovery(), batch, seed=5)
    assert a.report.revocation_rate == b.report.revocation_rate


def test_empty_batch_is_rejected(setup):
    world, _ = setup
    with pytest.raises(ValueError, match="empty batch"):
        evaluate_policy(world, FixedLadder(), [])


def test_a_policy_that_never_advances_time_cannot_hang_the_harness(setup):
    """A candidate policy with a scheduling bug should score badly, not loop
    forever. The harness nudges non-advancing actions forward rather than
    trusting every policy to be correct."""
    world, batch = setup

    class Stuck(Policy):
        name = "stuck"

        def decide(self, episode, now, deadline):
            return Decision(action=Action.RETRY_SAME_RAIL, at=now, reason="stuck")

    result = evaluate_policy(world, Stuck(), batch[:50], max_steps=4)
    assert result.report.episodes == 50


def test_audit_trail_records_a_reason_for_every_action(setup):
    """The track requires an audit trail. A log of what without why cannot be
    reviewed by the team that has to answer for it."""
    world, batch = setup
    result = evaluate_policy(world, DispositionAwareRules(), batch[:300])
    frame = result.audit_frame()
    assert len(frame) > 0
    assert (frame["reason"].str.len() > 0).all()
    assert set(frame.columns) >= {"episode_id", "action", "reason", "detail"}


def test_exception_list_explains_every_unrecovered_episode(setup):
    world, batch = setup
    result = evaluate_policy(world, DispositionAwareRules(), batch[:300])
    exceptions = result.exception_frame()
    unrecovered = round(
        (1 - result.report.recovery_rate) * result.report.episodes
    )
    assert len(exceptions) == pytest.approx(unrecovered, abs=1)
    assert (exceptions["reason"].str.len() > 0).all()


# ==========================================================================
# Baseline behaviour
# ==========================================================================


def test_no_recovery_spends_nothing_but_still_loses_customers(setup):
    """Giving up is not free. If abandoning a debt carried no churn risk, every
    comparison in the project would be rigged toward doing nothing."""
    world, batch = setup
    result = evaluate_policy(world, NoRecovery(), batch, seed=42)
    assert result.report.spent_paise == 0
    assert result.report.recovery_rate == 0.0
    assert result.report.revocation_rate > 0.0
    assert result.report.destroyed_paise > 0


def test_recovering_payments_reduces_churn(setup):
    """The central economic claim of the project.

    A policy that recovers payments must lose fewer customers than one that
    abandons them. If this ever inverts, the argument for doing recovery at all
    has collapsed.
    """
    world, batch = setup
    idle = evaluate_policy(world, NoRecovery(), batch, seed=42).report
    working = evaluate_policy(world, FixedLadder(), batch, seed=42).report
    assert working.revocation_rate < idle.revocation_rate
    assert working.net_paise > idle.net_paise


def test_aggressive_contact_recovers_less_and_destroys_more(setup):
    """Included precisely because it wins on the metric merchants watch.

    It contacts relentlessly, burns the recovery window, and churns customers.
    Its role is to demonstrate that recovery rate is the wrong headline.
    """
    world, batch = setup
    aggressive = evaluate_policy(world, AggressiveContact(), batch, seed=42).report
    ladder = evaluate_policy(world, FixedLadder(), batch, seed=42).report
    assert aggressive.revocation_rate > ladder.revocation_rate
    assert aggressive.net_paise < ladder.net_paise
    assert aggressive.contacts_per_episode > ladder.contacts_per_episode


def test_disposition_rules_stop_immediately_on_terminal_failures(setup):
    """The cheapest win available. Paying to confirm that a revoked mandate is
    still revoked is pure waste, and it is what the naive ladder does."""
    world, batch = setup
    terminal = [spec for spec in batch if is_terminal(spec.failure_code)][:200]
    assert terminal, "no terminal failures in the batch to test against"

    rules = evaluate_policy(world, DispositionAwareRules(), terminal, seed=9)
    ladder = evaluate_policy(world, FixedLadder(), terminal, seed=9)
    assert rules.report.spent_paise < ladder.report.spent_paise
    assert rules.report.attempts_per_episode < ladder.report.attempts_per_episode


def test_no_policy_recovers_money_from_a_terminal_failure(setup):
    """A structural guarantee, not a statistical one. If any policy ever
    collects on a revoked mandate, the world has a hole in it."""
    world, batch = setup
    terminal = [spec for spec in batch if is_terminal(spec.failure_code)][:150]
    for policy in default_baselines():
        result = evaluate_policy(world, policy, terminal, seed=3)
        assert result.report.recovered_paise == 0, (
            f"{policy.name} recovered money from a terminal failure"
        )


def test_fixed_ladder_wastes_attempts_on_dead_mandates(setup):
    """The waste this project exists to remove, demonstrated rather than
    asserted."""
    world, batch = setup
    terminal = [spec for spec in batch if is_terminal(spec.failure_code)][:150]
    ladder = evaluate_policy(world, FixedLadder(), terminal, seed=3).report
    assert ladder.attempts_per_episode > 1.0
    assert ladder.spent_paise > 0
    assert ladder.recovered_paise == 0


def test_every_baseline_runs_and_reports(setup):
    world, batch = setup
    results = evaluate_all(world, default_baselines(), batch[:400])
    assert set(results) == {policy.name for policy in default_baselines()}
    for name, result in results.items():
        assert result.report.episodes == 400, name


def test_upi_retries_are_scheduled_inside_the_execution_window(setup):
    """Baselines that know about the windows must respect them. A retry into a
    closed window is a guaranteed decline that still costs a fee."""
    world, batch = setup
    upi = [spec for spec in batch if str(spec.mandate.rail) == "upi_autopay"][:250]
    result = evaluate_policy(world, DispositionAwareRules(), upi, seed=8)
    frame = result.audit_frame()
    retries = frame[frame["action"] == str(Action.RETRY_SAME_RAIL)]
    if len(retries):
        outside = retries["detail"].str.contains("execution window").sum()
        assert outside == 0, (
            f"{outside} retries were presented into a closed UPI window"
        )


# ==========================================================================
# The random stream
# ==========================================================================
#
# Two defects lived here and neither was visible to any tamper check, because
# neither involved tampering. Every draw came from one sequential stream, so
# *which* uniform met a given action depended on how many draws had happened
# first — that is, on everything else the policy had chosen to do.
#
# It made common random numbers not common: a policy sending one extra email
# met a different settlement outcome on 15.75% of episodes than one sending
# none, worth 177k-319k rupees per thousand episodes of alignment noise against
# a policy gap of 316k, and on one batch it reversed two baselines outright.
#
# And it was exploitable: the per-episode seed was `seed + index` and the
# episode id was `EV_{index:08d}`, so a policy could read its own stream index,
# reconstruct the uniforms, and burn cheap actions to slide its retry onto a
# favourable one. A demonstration lifted recovery 0.505 -> 0.754 and net value
# 7.4x out of the same single retry.


class _Burner(Policy):
    """k emails, then one retry. k is the knob the exploit turned."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"burn_{k}"

    def reset(self) -> None:
        pass

    def decide(self, view, now, deadline):
        if view.steps_taken < self.k:
            return Decision(
                Action.NUDGE_EMAIL, now + dt.timedelta(minutes=5), "burn"
            )
        if view.steps_taken == self.k:
            return Decision(
                Action.RETRY_SAME_RAIL, now + dt.timedelta(hours=2), "retry"
            )
        return None


def test_a_draw_depends_on_its_ordinal_and_nothing_else(setup):
    """The exact property the fix rests on.

    The uniform met by the k-th debit presentation is a function of the
    episode and k alone. Anything else the policy did — however much of it —
    cannot move it, because the stream is addressed rather than consumed.
    """
    from rebound.sim.world import _DRAW_OUTCOME, _DRAW_SETTLEMENT, EpisodeEntropy

    entropy = EpisodeEntropy(20260825)
    first = entropy.uniform(_DRAW_OUTCOME, 0)

    # Asking for every other draw, in any order, any number of times.
    for ordinal in range(12):
        entropy.uniform(_DRAW_OUTCOME, ordinal)
        entropy.uniform(_DRAW_SETTLEMENT, ordinal)
    for ordinal in reversed(range(12)):
        entropy.uniform(_DRAW_OUTCOME, ordinal)

    assert entropy.uniform(_DRAW_OUTCOME, 0) == first
    # And distinct addresses are genuinely distinct draws, or the whole scheme
    # would be a constant.
    assert len({entropy.uniform(_DRAW_OUTCOME, k) for k in range(12)}) == 12
    assert entropy.uniform(_DRAW_SETTLEMENT, 0) != first


def test_the_settlement_draw_ignores_everything_the_policy_did(setup):
    """Common random numbers, tested where they were actually broken.

    Tested on the draw itself rather than on a revocation rate. A rate cannot
    isolate this: taking more retries resolves more episodes, which removes
    them from settlement altogether, so the rate moves for an honest reason and
    a stream artefact would hide inside it.

    Two episodes, same entropy, same customer, same everything — except one has
    a long history behind it. Under the old shared stream the settled outcome
    depended on how many draws that history had consumed.
    """
    from rebound.sim.world import ActionOutcome, EpisodeEntropy

    world, batch = setup
    spec = batch[0]

    def settle(history_length: int):
        episode = world.open_episode(
            episode_id="EP_PROBE",
            mandate=spec.mandate,
            customer=spec.customer,
            failure_code=spec.failure_code,
            failed_at=spec.failed_at,
            cycles_elapsed=spec.cycles_elapsed,
            entropy=EpisodeEntropy(20260825),
        )
        for _ in range(history_length):
            episode.history.append(
                ActionOutcome(
                    action=Action.NUDGE_EMAIL,
                    at=spec.failed_at,
                    succeeded=False,
                    recovered_paise=0,
                    cost_paise=200,
                    revoked=False,
                    destroyed_paise=0,
                    detail="probe",
                )
            )
        outcome = world.close_episode(episode)
        return outcome is not None

    settled = {n: settle(n) for n in range(8)}
    assert len(set(settled.values())) == 1, settled


def test_an_episode_id_does_not_reveal_its_stream_index(setup):
    """The id was `EV_{index:08d}` and the per-episode seed was `seed + index`.

    A policy reads the id off its own view, so the index of the random stream
    it was about to face was handed to it directly, and the arithmetic to turn
    one into the other was a documented constant.
    """
    from rebound.eval.harness import _episode_id

    ids = [_episode_id(4242, i) for i in range(200)]

    for i, episode_id in enumerate(ids):
        assert episode_id != f"EV_{i:08d}"
        suffix = episode_id.removeprefix("EV_")
        # Not the index in any base, and not adjacent to it.
        assert suffix != str(i)
        if suffix.isdigit():
            assert abs(int(suffix) - i) > 1000

    # A hash is not monotonic in its input; a counter is.
    assert ids != sorted(ids)
    assert len(set(ids)) == len(ids), "ids collided"

    # Reproducible from the seed, or nothing in this project replays.
    assert ids == [_episode_id(4242, i) for i in range(200)]
    # And the seed actually participates.
    assert ids[0] != _episode_id(4243, 0)


def test_the_same_seed_reproduces_the_same_report(setup):
    world, batch = setup
    episodes = batch[:300]
    first = evaluate_policy(world, DispositionAwareRules(), episodes, seed=11).report
    second = evaluate_policy(world, DispositionAwareRules(), episodes, seed=11).report
    assert first == second


def test_a_collect_link_recovers_nothing_from_a_closed_account(setup):
    """Structural, not statistical — which it was not before.

    `_apply_collect_link` was the one path that recovered money without
    consulting `mandate_alive`, so it disagreed with the taxonomy, which makes
    every action on a TERMINAL failure illegal. The guarantee held only because
    the old draw ordering never happened to pay a link on those episodes; the
    moment the draws were re-addressed, `aggressive_contact` collected Rs 394
    from a dead account.
    """
    world, batch = setup
    terminal = [spec for spec in batch if is_terminal(spec.failure_code)][:150]
    assert terminal, "no terminal failures in the batch"
    for seed in (3, 17, 4242, 20260825):
        result = evaluate_policy(world, AggressiveContact(), terminal, seed=seed)
        assert result.report.recovered_paise == 0, (
            f"collected from a terminal failure on seed {seed}"
        )
