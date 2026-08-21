"""Tests for the historical log.

The load-bearing tests here are the leakage guards. A generator that quietly
leaks a latent produces a model with excellent held-out metrics and no value
whatsoever, and nothing about the failure looks like a failure.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rebound.sim.dataset import (
    EXPLORATION_WEIGHTS,
    FORBIDDEN_COLUMNS,
    GenerationConfig,
    coverage_report,
    generate_log,
)
from rebound.taxonomy import Action

CONFIG = GenerationConfig(
    n_customers=250,
    start=dt.date(2025, 1, 1),
    end=dt.date(2025, 10, 31),
    seed=555,
)


@pytest.fixture(scope="module")
def log():
    return generate_log(CONFIG)


# ==========================================================================
# Leakage
# ==========================================================================


def test_no_latent_columns_reach_the_log(log):
    """The central guard on Claim A.

    If a customer latent appears in the training data, the model's held-out
    metrics measure its ability to read an answer key rather than to learn
    anything, and every downstream number in the README becomes fiction.
    """
    leaked = FORBIDDEN_COLUMNS & set(log.columns)
    assert not leaked, f"latent columns leaked into the log: {leaked}"


def test_customer_history_is_strictly_backward_looking(log):
    """Prior-history features must only ever see the customer's past.

    Accumulating them chronologically is the defence; this asserts the defence
    actually holds. A single misordered update would let each row see its own
    outcome, which inflates every metric and looks like a very good model.
    """
    for customer_id, rows in log.groupby("customer_id"):
        ordered = rows.sort_values(["decided_at", "decision_index"])
        failures = ordered["cust_prior_failures"].to_numpy()
        assert (failures[1:] >= failures[:-1]).all(), (
            f"cust_prior_failures decreases over time for {customer_id}"
        )
        assert failures[0] == 0, (
            f"{customer_id}'s first ever decision already has prior failures"
        )


def test_prior_rates_are_zero_before_any_history(log):
    fresh = log[log["cust_prior_failures"] == 0]
    assert (fresh["cust_prior_recovery_rate"] == 0).all()
    assert (fresh["cust_prior_recoveries"] == 0).all()


def test_prior_recoveries_never_exceed_prior_failures(log):
    assert (log["cust_prior_recoveries"] <= log["cust_prior_failures"]).all()


# ==========================================================================
# Structural consistency
# ==========================================================================


def test_decisions_come_after_the_failure(log):
    assert (log["decided_at"] > log["failed_at"]).all()


def test_decision_indices_are_contiguous_within_an_episode(log):
    for episode_id, rows in log.groupby("episode_id"):
        indices = sorted(rows["decision_index"])
        assert indices == list(range(len(indices))), (
            f"{episode_id} has non-contiguous decision indices: {indices}"
        )


def test_days_since_failure_is_positive_and_increasing(log):
    for episode_id, rows in log.groupby("episode_id"):
        ordered = rows.sort_values("decision_index")["days_since_failure"].to_numpy()
        assert (ordered > 0).all()
        assert (ordered[1:] >= ordered[:-1]).all(), episode_id


def test_episodes_never_continue_past_resolution(log):
    """Acting on a resolved episode would bill a customer who has already
    paid. The world raises on it; this checks the ladder never tries."""
    for episode_id, rows in log.groupby("episode_id"):
        ordered = rows.sort_values("decision_index")
        succeeded = ordered["succeeded"].to_numpy()
        if succeeded.any():
            assert succeeded.argmax() == len(succeeded) - 1, (
                f"{episode_id} kept acting after recovering"
            )


def test_stop_is_always_the_final_decision(log):
    for episode_id, rows in log.groupby("episode_id"):
        ordered = rows.sort_values("decision_index")
        actions = list(ordered["action"])
        if str(Action.STOP) in actions:
            assert actions[-1] == str(Action.STOP), episode_id


# ==========================================================================
# Labels
# ==========================================================================


def test_downstream_label_matches_what_happened_in_the_episode(log):
    for episode_id, rows in log.groupby("episode_id"):
        recovered = bool(rows["succeeded"].any())
        assert rows["episode_recovered"].nunique() == 1
        assert bool(rows["episode_recovered"].iloc[0]) == recovered, episode_id


def test_downstream_label_rescues_enabling_actions(log):
    """The reason the downstream label exists.

    Under the immediate label a nudge scores exactly zero, because nudges
    unblock customers rather than collect money. A model trained on that learns
    to never contact anyone. If this ever regresses, the dataset has quietly
    gone back to teaching that lesson.
    """
    nudges = log[log["action"].isin([str(Action.NUDGE_SMS), str(Action.NUDGE_WHATSAPP)])]
    assert len(nudges) > 100, "not enough nudge rows to judge"
    assert nudges["succeeded"].mean() == 0.0, (
        "a nudge should never directly collect money"
    )
    assert nudges["episode_recovered"].mean() > 0.05, (
        "nudges show no downstream value at all — the credit-assignment fix "
        "has regressed and the model will learn to never nudge"
    )


def test_stopped_episodes_never_recover(log):
    stopped = log[log["episode_stopped"]]
    if len(stopped):
        assert not stopped["episode_recovered"].any()


def test_revocation_destroys_value_in_the_ledger(log):
    revoked = log[log["episode_revoked"]]
    if len(revoked):
        assert (revoked["episode_destroyed_paise"] > 0).all()
        assert (revoked["episode_net_paise"] < 0).all(), (
            "a revoked episode must show a net loss, or the policy will learn "
            "that churning customers is free"
        )


# ==========================================================================
# The behavioural policy
# ==========================================================================


def test_exploration_weights_form_a_distribution():
    total = sum(EXPLORATION_WEIGHTS.values())
    assert total == pytest.approx(1.0, abs=1e-9)
    assert all(w > 0 for w in EXPLORATION_WEIGHTS.values())


def test_every_action_appears_in_the_log(log):
    """Coverage. An action the behavioural policy never took is an action the
    model has no evidence about, and any later claim regarding it is
    extrapolation dressed as a prediction."""
    seen = set(log["action"])
    expected = {str(a) for a in EXPLORATION_WEIGHTS}
    assert expected <= seen, f"never explored: {expected - seen}"


def test_propensities_are_recorded_and_valid(log):
    assert (log["propensity"] > 0).all()
    assert (log["propensity"] <= 1).all()


def test_the_ladder_takes_futile_actions(log):
    """Deliberate, not a defect.

    Real merchants retry revoked mandates. If the behavioural policy were too
    sensible to do that, the log would contain no examples of futility and the
    model could not learn to avoid it — losing the single largest source of
    wasted spend.
    """
    dead_retries = log[
        (log["action"] == str(Action.RETRY_SAME_RAIL)) & (~log["mandate_alive"])
    ]
    assert len(dead_retries) > 50, (
        "the log contains almost no doomed retries, so the model has nothing "
        "to learn futility from"
    )
    assert dead_retries["succeeded"].mean() < 0.02


def test_timing_actually_varies(log):
    """A merchant who always retried at exactly +1/+3/+7 days would generate a
    log from which nothing about timing could be inferred."""
    assert log["days_since_failure"].std() > 1.0
    assert log["decision_hour"].nunique() > 12


# ==========================================================================
# Determinism and reporting
# ==========================================================================


def test_generation_is_reproducible():
    a = generate_log(CONFIG)
    b = generate_log(CONFIG)
    assert len(a) == len(b)
    assert a["episode_id"].tolist() == b["episode_id"].tolist()
    assert a["succeeded"].tolist() == b["succeeded"].tolist()


def test_revoked_customers_stop_generating_episodes(log):
    """Once a mandate is revoked it stops billing. Continuing to present
    against it would manufacture failures no merchant would ever see."""
    for customer_id, rows in log.groupby("customer_id"):
        ordered = rows.sort_values(["decided_at", "decision_index"])
        revoked_at = ordered["revoked"].to_numpy().argmax() if ordered["revoked"].any() else None
        if revoked_at is None:
            continue
        after = ordered.iloc[revoked_at + 1 :]
        assert after["episode_id"].nunique() <= 1, (
            f"{customer_id} started new episodes after revoking"
        )


def test_coverage_report_is_populated(log):
    report = coverage_report(log)
    assert len(report) > 0
    assert {"rows", "immediate_rate", "downstream_rate", "mean_propensity"} <= set(
        report.columns
    )
    assert report["rows"].sum() == len(log)
