"""Tests for the metric implementations.

Metrics are tested against cases with known answers. A miscalculated metric is
worse than no metric: it produces a number that gets written into a README and
believed.
"""

from __future__ import annotations

import numpy as np
import pytest

from rebound.eval.metrics import (
    PolicyReport,
    classification_report,
    lift_over_baseline,
    policy_comparison,
    reliability_table,
    value_preserved,
)


def _report(name: str, **overrides) -> PolicyReport:
    defaults = dict(
        policy=name,
        episodes=1000,
        recovery_rate=0.3,
        revocation_rate=0.05,
        recovered_paise=1_000_000,
        spent_paise=50_000,
        destroyed_paise=200_000,
        net_paise=750_000,
        attempts_per_episode=2.0,
        contacts_per_episode=1.0,
    )
    defaults.update(overrides)
    return PolicyReport(**defaults)


# ==========================================================================
# Classification metrics
# ==========================================================================


def test_perfect_predictions_score_perfectly():
    y = np.array([0, 0, 1, 1, 0, 1])
    report = classification_report(y, y.astype(float))
    assert report.roc_auc == pytest.approx(1.0)
    assert report.pr_auc == pytest.approx(1.0)
    assert report.brier == pytest.approx(0.0)


def test_inverted_predictions_score_at_the_floor():
    y = np.array([0, 0, 1, 1, 0, 1])
    report = classification_report(y, 1.0 - y.astype(float))
    assert report.roc_auc == pytest.approx(0.0)


def test_constant_predictions_have_no_discrimination():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    report = classification_report(y, np.full(500, 0.5))
    assert report.roc_auc == pytest.approx(0.5, abs=0.02)


def test_base_rate_is_the_observed_positive_rate():
    y = np.array([1, 0, 0, 0])
    assert classification_report(y, np.full(4, 0.25)).base_rate == pytest.approx(0.25)


def test_a_well_calibrated_model_has_slope_near_one():
    rng = np.random.default_rng(7)
    probs = rng.uniform(0.05, 0.95, 20_000)
    outcomes = (rng.uniform(size=20_000) < probs).astype(int)
    report = classification_report(outcomes, probs)
    assert report.calibration_slope == pytest.approx(1.0, abs=0.1)
    assert report.expected_calibration_error < 0.02


def test_an_overconfident_model_has_slope_below_one():
    """The failure mode that matters most here.

    Overconfidence turns a good ranker into a policy that overspends on cases
    it was too sure about — and it is invisible in AUC, which is exactly why
    calibration is reported alongside it rather than instead of it.
    """
    rng = np.random.default_rng(11)
    truth = rng.uniform(0.2, 0.8, 20_000)
    outcomes = (rng.uniform(size=20_000) < truth).astype(int)
    overconfident = np.clip((truth - 0.5) * 2.5 + 0.5, 0.001, 0.999)
    report = classification_report(outcomes, overconfident)
    assert report.calibration_slope < 0.8
    assert report.expected_calibration_error > 0.02


def test_precision_at_capacity_beats_the_base_rate_for_a_good_model():
    rng = np.random.default_rng(3)
    probs = rng.uniform(0, 1, 5000)
    outcomes = (rng.uniform(size=5000) < probs).astype(int)
    report = classification_report(outcomes, probs, capacity=0.1)
    assert report.precision_at_capacity > report.base_rate
    assert report.lift_at_capacity > 1.0


def test_probabilities_outside_the_unit_interval_are_rejected():
    y = np.array([0, 1])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        classification_report(y, np.array([0.5, 1.5]))


def test_non_finite_probabilities_are_rejected():
    y = np.array([0, 1])
    with pytest.raises(ValueError, match="non-finite"):
        classification_report(y, np.array([0.5, np.nan]))


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="length"):
        classification_report(np.array([0, 1]), np.array([0.5]))


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        classification_report(np.array([]), np.array([]))


def test_single_class_input_does_not_crash():
    """Real slices are sometimes all-negative. The metrics that are undefined
    should be NaN, not an exception that kills an evaluation run."""
    y = np.zeros(50)
    report = classification_report(y, np.full(50, 0.2))
    assert np.isnan(report.roc_auc)
    assert report.brier == pytest.approx(0.04)


def test_reliability_table_bins_sum_to_the_input():
    rng = np.random.default_rng(5)
    probs = rng.uniform(0, 1, 1000)
    outcomes = (rng.uniform(size=1000) < probs).astype(int)
    table = reliability_table(outcomes, probs, bins=10)
    assert table["n"].sum() == 1000
    assert len(table) == 10


# ==========================================================================
# Policy metrics
# ==========================================================================


def test_per_1000_normalisation_scales_with_batch_size():
    """Reports must be comparable across batches of different sizes, or a
    policy evaluated on a bigger month looks better for no reason."""
    small = _report("p", episodes=500, recovered_paise=500_000)
    large = _report("p", episodes=5000, recovered_paise=5_000_000)
    assert small.recovered_rupees_per_1000 == pytest.approx(
        large.recovered_rupees_per_1000
    )


def test_true_cost_includes_destroyed_value():
    """The difference between the two cost ratios is the price of the churn a
    policy caused — invisible in every metric a merchant normally watches."""
    report = _report("p", recovered_paise=1000, spent_paise=100, destroyed_paise=900)
    assert report.cost_per_rupee_recovered == pytest.approx(0.1)
    assert report.true_cost_per_rupee_recovered == pytest.approx(1.0)
    assert report.true_cost_per_rupee_recovered > report.cost_per_rupee_recovered


def test_cost_ratios_are_infinite_when_nothing_is_recovered():
    report = _report("floor", recovered_paise=0)
    assert report.cost_per_rupee_recovered == float("inf")
    assert report.true_cost_per_rupee_recovered == float("inf")


def test_comparison_is_ordered_by_net_not_by_recovery():
    """A policy that recovers more while destroying more must not appear to
    win. This ordering is the whole argument against recovery rate as a
    headline."""
    greedy = _report("greedy", recovery_rate=0.5, net_paise=100_000)
    careful = _report("careful", recovery_rate=0.3, net_paise=900_000)
    frame = policy_comparison([greedy, careful])
    assert list(frame["policy"]) == ["careful", "greedy"]


def test_value_preserved_is_measured_against_the_floor():
    floor = _report("no_recovery", net_paise=-1_000_000)
    better = _report("candidate", net_paise=-400_000)
    frame = value_preserved([floor, better])
    row = frame[frame["policy"] == "candidate"].iloc[0]
    assert row["value_preserved_rs_per_1000"] == pytest.approx(6000.0)


def test_value_preserved_requires_the_floor_policy():
    with pytest.raises(ValueError, match="floor policy"):
        value_preserved([_report("only_one")])


def test_lift_is_relative_to_the_baseline():
    baseline = _report("base", net_paise=1_000_000)
    candidate = _report("candidate", net_paise=1_500_000)
    lift = lift_over_baseline(candidate, baseline)
    assert lift["net_lift"] == pytest.approx(0.5)


def test_lift_handles_a_zero_baseline_without_dividing_by_zero():
    baseline = _report("base", net_paise=0, recovery_rate=0.0)
    candidate = _report("candidate", net_paise=5_000)
    lift = lift_over_baseline(candidate, baseline)
    assert np.isnan(lift["net_lift"])
