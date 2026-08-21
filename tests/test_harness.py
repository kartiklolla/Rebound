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
