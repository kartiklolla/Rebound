"""Tests for the recovery-probability model.

The tests that matter here are not "does it fit". They are the ones that would
catch a model whose numbers look good for the wrong reason: leakage through the
feature set, calibration fitted on data the base model already saw, or a lift
measured against a baseline weak enough to guarantee it.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from rebound.eval.metrics import classification_report, slice_report
from rebound.eval.splits import all_splits
from rebound.model import (
    TARGET,
    FailureCodePrior,
    FeatureSpec,
    GlobalPrior,
    RecoveryModel,
)
from rebound.sim.dataset import (
    FORBIDDEN_COLUMNS,
    OUTCOME_COLUMNS,
    GenerationConfig,
    generate_log,
)

CONFIG = GenerationConfig(
    n_customers=900,
    start=dt.date(2025, 1, 1),
    end=dt.date(2026, 3, 31),
    seed=4321,
)


@pytest.fixture(scope="module")
def log():
    return generate_log(CONFIG)


@pytest.fixture(scope="module")
def split(log):
    return all_splits(log)["time"]


@pytest.fixture(scope="module")
def fitted(split):
    return RecoveryModel(max_iter=120).fit(split.train)


# ==========================================================================
# Feature space
# ==========================================================================


def test_feature_spec_excludes_outcomes_and_latents(log):
    """The leakage guard, at the point where it actually matters.

    ``episode_net_paise`` predicts the label at accuracy 1.000 because it is
    the label restated. If it reaches the feature space, every metric below is
    meaningless while looking excellent.
    """
    spec = FeatureSpec.fit(log)
    columns = set(spec.columns)
    assert not columns & OUTCOME_COLUMNS
    assert not columns & FORBIDDEN_COLUMNS
    assert TARGET not in columns
    assert "episode_net_paise" not in columns
    assert len(columns) > 15


def test_feature_spec_pins_category_levels(log):
    """Levels are fitted once and reused.

    Re-deriving categories per frame means a test set missing one failure code
    shifts every category index, and the model silently reads the wrong column
    values — a corruption that produces plausible numbers rather than an error.
    """
    spec = FeatureSpec.fit(log)
    assert "failure_code" in spec.categorical
    assert spec.categories["failure_code"]

    subset = log[log["failure_code"] != log["failure_code"].iloc[0]]
    transformed = spec.transform(subset)
    assert list(transformed["failure_code"].cat.categories) == list(
        spec.categories["failure_code"]
    )


def test_unseen_categories_become_missing_not_a_new_code(log):
    spec = FeatureSpec.fit(log)
    altered = log.head(50).copy()
    altered["failure_code"] = "CODE_THE_RAIL_ADDED_LAST_TUESDAY"
    transformed = spec.transform(altered)
    assert transformed["failure_code"].isna().all()


def test_transform_rejects_a_frame_missing_fitted_features(log):
    spec = FeatureSpec.fit(log)
    with pytest.raises(ValueError, match="missing fitted features"):
        spec.transform(log.drop(columns=[spec.columns[0]]))


def test_booleans_are_numeric_after_transform(log):
    spec = FeatureSpec.fit(log)
    transformed = spec.transform(log.head(100))
    for column in spec.boolean:
        assert transformed[column].dtype.kind in "iu"


# ==========================================================================
# Fitting and calibration
# ==========================================================================


def test_predictions_are_probabilities(fitted, split):
    probs = fitted.predict_proba(split.test)
    assert len(probs) == len(split.test)
    assert np.isfinite(probs).all()
    assert (probs >= 0).all() and (probs <= 1).all()


def test_calibration_slice_is_disjoint_and_later(split):
    """Calibrating on data the base model already fitted produces a model that
    looks beautifully calibrated in-sample and is overconfident on everything
    else — the exact failure calibration exists to prevent."""
    model = RecoveryModel(max_iter=60, calibration_fraction=0.25).fit(split.train)
    assert model.fit_rows_ + model.calibration_rows_ == len(split.train)
    assert model.calibration_rows_ == pytest.approx(len(split.train) * 0.25, rel=0.02)

    ordered = split.train.sort_values("failed_at")
    boundary = ordered.iloc[model.fit_rows_ - 1]["failed_at"]
    calibration_start = ordered.iloc[model.fit_rows_]["failed_at"]
    assert calibration_start >= boundary


def test_calibration_is_chosen_by_measurement_not_assumption(fitted):
    """Calibration is routinely described as free. It is not.

    Isotonic regression overfits thin data, and a booster trained with log-loss
    on a modest dataset is often already well calibrated — in which case any
    correction makes it worse. Both were observed while building this.

    So all three candidates are scored, including doing nothing, and the winner
    is whichever measured best on held-out data. This asserts that process
    happened rather than asserting an outcome, because which candidate wins is
    a property of the dataset and will legitimately differ by scale.
    """
    assert set(fitted.calibration_scores_) == {"none", "sigmoid", "isotonic"}
    assert fitted.calibration_method_used_ in {"none", "sigmoid", "isotonic"}
    best = min(fitted.calibration_scores_, key=fitted.calibration_scores_.get)
    assert fitted.calibration_method_used_ == best


def test_declining_to_calibrate_still_produces_predictions(split):
    """When 'none' wins, predict_proba must fall through to the raw model
    rather than returning nothing or raising."""
    model = RecoveryModel(max_iter=60).fit(split.train)
    model.calibrated_ = None
    model.calibration_method_used_ = "none"
    probs = model.predict_proba(split.test)
    np.testing.assert_allclose(
        probs, model.predict_proba_uncalibrated(split.test)
    )


def test_calibration_selection_uses_data_the_base_model_never_saw(split):
    """Selecting on the base model's own training data would pick whichever
    candidate overfits hardest."""
    model = RecoveryModel(max_iter=60, calibration_fraction=0.25).fit(split.train)
    ordered = split.train.sort_values("failed_at")
    calibration_part = ordered.iloc[model.fit_rows_ :]
    assert len(calibration_part) == model.calibration_rows_
    assert model.calibration_rows_ > 0


def test_an_explicit_method_overrides_the_selection(split):
    model = RecoveryModel(max_iter=60, calibration_method="sigmoid").fit(split.train)
    assert model.calibration_method_used_ == "sigmoid"
    assert model.calibrated_ is not None


def test_calibration_costs_little_discrimination(fitted, split):
    """If calibration destroyed ranking power it would not be worth the trade,
    and the honest move would be to report the uncalibrated model instead."""
    truth = split.test[TARGET].astype(int)
    raw = classification_report(truth, fitted.predict_proba_uncalibrated(split.test))
    calibrated = classification_report(truth, fitted.predict_proba(split.test))
    assert calibrated.pr_auc > raw.pr_auc - 0.03


def test_fitting_is_reproducible(split):
    a = RecoveryModel(max_iter=60, seed=7).fit(split.train).predict_proba(split.test)
    b = RecoveryModel(max_iter=60, seed=7).fit(split.train).predict_proba(split.test)
    np.testing.assert_allclose(a, b)


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        RecoveryModel().predict_proba(None)


def test_single_class_training_data_is_rejected(split):
    positives = split.train[split.train[TARGET]]
    with pytest.raises(ValueError, match="single class"):
        RecoveryModel().fit(positives)


def test_impossible_calibration_fraction_is_rejected():
    for bad in (0.0, 0.6, 1.0, -0.1):
        with pytest.raises(ValueError, match="calibration_fraction"):
            RecoveryModel(calibration_fraction=bad)


# ==========================================================================
# Does it actually beat anything
# ==========================================================================


def test_model_beats_the_base_rate(fitted, split):
    truth = split.test[TARGET].astype(int)
    floor = GlobalPrior().fit(split.train).predict_proba(split.test)
    model = fitted.predict_proba(split.test)
    assert classification_report(truth, model).pr_auc > classification_report(
        truth, floor
    ).pr_auc


def test_model_beats_the_failure_code_prior(fitted, split):
    """The baseline that makes the lift honest.

    A merchant with a pivot table has the failure-code prior. If a gradient
    booster barely beats a group mean, the machinery is not carrying its weight
    and the honest answer is to say so rather than to report the raw AUC.
    """
    truth = split.test[TARGET].astype(int)
    prior = FailureCodePrior().fit(split.train).predict_proba(split.test)
    model = fitted.predict_proba(split.test)
    prior_score = classification_report(truth, prior).pr_auc
    model_score = classification_report(truth, model).pr_auc
    assert model_score > prior_score, (
        f"model PR-AUC {model_score:.4f} does not beat the failure-code prior "
        f"{prior_score:.4f}; the taxonomy is doing all the work"
    )


def test_global_prior_predicts_the_training_base_rate(split):
    prior = GlobalPrior().fit(split.train)
    probs = prior.predict_proba(split.test)
    assert probs.min() == probs.max()
    assert probs[0] == pytest.approx(split.train[TARGET].mean())


def test_failure_code_prior_falls_back_for_unseen_combinations(split):
    """A group mean over a handful of rows is noise wearing a decimal point,
    so thin cells fall back to the global rate."""
    prior = FailureCodePrior(min_rows=30).fit(split.train)
    unseen = split.test.head(20).copy()
    unseen["failure_code"] = "NOT_A_REAL_CODE"
    probs = prior.predict_proba(unseen)
    assert np.allclose(probs, prior.fallback_)


# ==========================================================================
# Per-slice honesty
# ==========================================================================


def test_slice_report_covers_every_disposition(fitted, split):
    """Every expensive mistake in this project was invisible in the aggregate
    and obvious per slice. The model gets the same treatment."""
    test = split.test.reset_index(drop=True)
    report = slice_report(
        test,
        test[TARGET].astype(int).to_numpy(),
        fitted.predict_proba(test),
        by="disposition",
    )
    assert set(report["disposition"]) == set(test["disposition"].unique())
    assert report["n"].sum() == len(test)


def test_slice_report_flags_thin_slices_rather_than_hiding_them(fitted, split):
    """Dropping small slices hides exactly the cells where the model is least
    trustworthy, which is the opposite of what the table is for."""
    test = split.test.reset_index(drop=True)
    report = slice_report(
        test,
        test[TARGET].astype(int).to_numpy(),
        fitted.predict_proba(test),
        by="disposition",
        min_rows=10_000,
    )
    assert report["thin"].any()
    assert report.loc[report["thin"], "pr_auc"].isna().all()
    assert report.loc[report["thin"], "n"].gt(0).all()


def test_the_model_is_not_uniformly_good_across_dispositions(fitted, split):
    """Guards the headline against a comfortable misreading.

    A respectable overall PR-AUC here is largely the model knowing which
    failure codes are hopeless — which the taxonomy already encodes. Within-slice
    discrimination, the part that would actually change a decision, is weaker.
    If this test ever starts failing because every slice is strong, the claim in
    the README should be upgraded; until then it must stay honest.
    """
    test = split.test.reset_index(drop=True)
    report = slice_report(
        test,
        test[TARGET].astype(int).to_numpy(),
        fitted.predict_proba(test),
        by="disposition",
    )
    scored = report.dropna(subset=["roc_auc"])
    assert len(scored) >= 4
    assert scored["roc_auc"].min() < scored["roc_auc"].max() - 0.05


# ==========================================================================
# Feature importance
# ==========================================================================


def test_feature_importance_is_ranked_and_covers_the_feature_set(fitted, split):
    importance = fitted.feature_importance(split.test, n_repeats=2, sample=1500)
    assert set(importance["feature"]) == set(fitted.spec_.columns)
    assert importance["importance"].is_monotonic_decreasing


def test_failure_code_is_the_dominant_feature(fitted, split):
    """Expected, and worth asserting so the README's framing stays true: the
    taxonomy carries most of the signal, and the model's job in the sequencer
    is the timing and action choice on top of it."""
    importance = fitted.feature_importance(split.test, n_repeats=2, sample=1500)
    assert importance.iloc[0]["feature"] == "failure_code"
