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
    else — the exact failure calibration exists to prevent.

    This test used to be worthless. It re-sorted the frame by ``failed_at`` and
    asserted that the row after the cut was not earlier than the row before it
    — that is, it asserted a sorted column was sorted, which holds at every
    index regardless of what the model did. It passed while the two slices
    shared an episode and overlapped by 27 days of ``decided_at``.

    It now looks up the rows the model actually fitted on.
    """
    model = RecoveryModel(max_iter=60, calibration_fraction=0.25).fit(split.train)
    assert (
        model.fit_rows_ + model.calibration_rows_ + model.dropped_rows_
        == len(split.train)
    )
    assert model.dropped_rows_ < len(split.train) * 0.15, (
        f"the embargo dropped {model.dropped_rows_} of {len(split.train)} rows; "
        f"a clean boundary is not worth that much of the training set"
    )
    # Not an exact share any more: the embargo removes straddling episodes from
    # both sides, so calibration_fraction is a target rather than a guarantee.
    # A band is the honest assertion — the point of the test is disjointness.
    share = model.calibration_rows_ / len(split.train)
    assert 0.15 < share < 0.30, f"calibration slice is {share:.1%} of train"

    fit_part = split.train.loc[model.fit_index_]
    calibration_part = split.train.loc[model.calibration_index_]

    shared = set(fit_part["episode_id"]) & set(calibration_part["episode_id"])
    assert not shared, (
        f"{len(shared)} episodes straddle the inner cut; their calibration-slice "
        f"labels are readable from their fit-slice rows"
    )
    assert fit_part["decided_at"].max() < calibration_part["decided_at"].min(), (
        "the calibration slice is not strictly later by decision time. Ordering "
        "by failed_at sorts by episode start, so a long episode's later "
        "decisions land on the wrong side of the cut."
    )


def test_the_inner_split_holds_out_customers_when_the_regime_is_cold_start(split):
    """Regression: the inner split must hold out the same thing the outer one does.

    The model always cut the inner split temporally, whatever it was being
    scored against. Under a customer-based outer split that meant 3,533 of
    3,712 calibration-slice customers were also in the base model's fit slice,
    so the calibrator was selected on people the booster had memorised and then
    applied to strangers. It chose isotonic on a selection ECE of 0.0074
    against 0.0153 for no calibration — and on the held-out test set isotonic
    was worse on every metric, ECE included, costing 0.0131 of PR-AUC.

    A confident, reproducible selection drawn from the wrong distribution.
    """
    model = RecoveryModel(max_iter=60, calibration_fraction=0.25).fit(
        split.train, order_by=None, group_by="customer_id"
    )
    fit_customers = set(split.train.loc[model.fit_index_]["customer_id"])
    calibration_customers = set(
        split.train.loc[model.calibration_index_]["customer_id"]
    )
    assert not (fit_customers & calibration_customers), (
        "the calibrator was selected on customers the base model had already "
        "seen, which is not the condition it will be scored under"
    )
    selection_customers = set(split.train.loc[model.selection_index_]["customer_id"])
    assert not (fit_customers & selection_customers)


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
    candidate overfits hardest.

    The previous version of this test re-derived the calibration slice with the
    same arithmetic the model used and asserted the two lengths matched. It
    never touched the base model, and would have passed unchanged if selection
    had been run on the training rows.
    """
    model = RecoveryModel(max_iter=60, calibration_fraction=0.25).fit(split.train)
    assert len(model.selection_index_) > 0

    fit_episodes = set(split.train.loc[model.fit_index_]["episode_id"])
    selection = split.train.loc[model.selection_index_]

    assert not (set(selection["episode_id"]) & fit_episodes), (
        "the calibration method was chosen on episodes the base model trained "
        "on, so the winner is whichever candidate overfits hardest"
    )
    assert set(model.selection_index_) <= set(model.calibration_index_)


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
    """Covers every feature, and ranks a known-worthless one near the bottom.

    Asserting ``is_monotonic_decreasing`` on the returned series, as this test
    used to, checks nothing: ``feature_importance`` sorts descending before
    returning, so it holds even if every importance is identical noise. The
    injected random column is the real check — permutation importance that is
    working puts it at the bottom.
    """
    importance = fitted.feature_importance(split.test, n_repeats=2, sample=1500)
    assert set(importance["feature"]) == set(fitted.spec_.columns)

    rng = np.random.default_rng(11)
    noisy_train = split.train.assign(pure_noise=rng.normal(size=len(split.train)))
    noisy_test = split.test.assign(pure_noise=rng.normal(size=len(split.test)))
    model = RecoveryModel(max_iter=60).fit(noisy_train)
    ranked = model.feature_importance(noisy_test, n_repeats=2, sample=1500)

    position = ranked.reset_index(drop=True).index[
        ranked.reset_index(drop=True)["feature"] == "pure_noise"
    ][0]
    assert position > len(ranked) * 0.5, (
        f"a column of pure noise ranked {position + 1} of {len(ranked)}; "
        f"permutation importance is not measuring what it claims to"
    )


def test_failure_code_is_the_dominant_feature(fitted, split):
    """Expected, and worth asserting so the README's framing stays true: the
    taxonomy carries most of the signal, and the model's job in the sequencer
    is the timing and action choice on top of it."""
    importance = fitted.feature_importance(split.test, n_repeats=2, sample=1500)
    assert importance.iloc[0]["feature"] == "failure_code"


# ==========================================================================
# Two heads
# ==========================================================================


@pytest.fixture(scope="module")
def heads(split):
    from rebound.model import TwoHeadedModel

    return TwoHeadedModel(max_iter=120).fit(split.train)


def test_both_heads_are_fitted(heads):
    from rebound.model import TARGET_DOWNSTREAM, TARGET_IMMEDIATE

    assert heads.immediate.target == TARGET_IMMEDIATE
    assert heads.downstream.target == TARGET_DOWNSTREAM
    assert heads.immediate.calibrated_ is not None or heads.immediate.base_ is not None
    assert heads.downstream.base_ is not None


def test_the_timing_head_trains_only_on_collecting_actions(split):
    """A nudge is structurally incapable of collecting, so its immediate label
    is always 0. Training the timing head on those rows teaches it that nudges
    never work — true, irrelevant, and it swamps the base rate of the rows that
    matter."""
    from rebound.model import COLLECTING_ACTIONS, TwoHeadedModel

    collecting = TwoHeadedModel.collecting_rows(split.train)
    assert set(collecting["action"]) <= COLLECTING_ACTIONS
    assert len(collecting) < len(split.train)
    assert collecting["succeeded"].mean() > 0, (
        "collecting actions must sometimes collect, or the label is degenerate"
    )


def test_nudges_are_excluded_and_would_have_a_degenerate_label(split):
    nudges = split.train[split.train["action"].str.startswith("nudge")]
    assert len(nudges) > 50
    assert nudges["succeeded"].mean() == 0.0


def test_the_two_heads_disagree(heads, split):
    """If they produced the same numbers there would be no reason for two."""
    a = heads.predict_immediate(split.test)
    b = heads.predict_downstream(split.test)
    assert np.abs(a - b).mean() > 0.05


def test_the_timing_head_beats_the_action_head_at_timing(heads, split):
    """The measurement that justifies the whole architecture.

    Choosing *when* to present a retry and choosing *which action* to take are
    different questions. A single model trained on the downstream label answers
    the second well and the first poorly, because the downstream label
    aggregates the whole episode and washes out the timing of any one decision.

    So on the immediate label, over rows that can actually collect, the head
    trained for it must win. If it ever stops winning, the second head is dead
    weight and should be deleted.
    """
    from rebound.model import TARGET_IMMEDIATE, TwoHeadedModel

    collecting = TwoHeadedModel.collecting_rows(split.test)
    truth = collecting[TARGET_IMMEDIATE].astype(int)

    timing = classification_report(truth, heads.predict_immediate(collecting))
    action = classification_report(truth, heads.predict_downstream(collecting))

    assert timing.pr_auc > action.pr_auc, (
        f"timing head PR-AUC {timing.pr_auc:.4f} does not beat the action head's "
        f"{action.pr_auc:.4f} on the immediate label; the split is not earning "
        f"its complexity"
    )


def test_the_action_head_beats_the_timing_head_at_action_choice(heads, split):
    """The mirror. Each head must win on its own question, or one of them is
    simply worse rather than different.

    Scored on collecting rows only, and that is not a detail. The previous
    version scored both heads on *all* test rows. The timing head's
    ``FeatureSpec`` pins ``action`` to the three collecting actions, so every
    other action maps to NaN — 46% of test rows at full scale. It compared a
    model that could see the action against one that could not, on half the
    data. The conclusion happened to be true; the test did not establish it.

    Matched rows and matched label, so the only difference left is the label
    each head was trained on.
    """
    from rebound.model import TARGET_DOWNSTREAM, TwoHeadedModel

    collecting = TwoHeadedModel.collecting_rows(split.test)
    truth = collecting[TARGET_DOWNSTREAM].astype(int)

    action = classification_report(truth, heads.predict_downstream(collecting))
    timing = classification_report(truth, heads.predict_immediate(collecting))

    assert action.pr_auc > timing.pr_auc, (
        f"action head PR-AUC {action.pr_auc:.4f} does not beat the timing "
        f"head's {timing.pr_auc:.4f} on the downstream label"
    )


def test_too_few_collecting_rows_is_rejected(split):
    from rebound.model import TwoHeadedModel

    nudges = split.train[split.train["action"].str.startswith("nudge")]
    with pytest.raises(ValueError, match="collecting-action rows"):
        TwoHeadedModel().fit(nudges)


def test_the_removed_salary_proxies_stay_removed(log):
    """Both were built, measured, and deleted — the failure-day version because
    it was a duplicate of billing_day, the recovery-day version because it made
    the model measurably worse. Re-adding either needs new evidence, not a
    fresh round of the same reasoning."""
    banned = {
        "cust_prior_mean_failure_day",
        "cust_prior_recovery_day_mean",
        "cust_days_from_recovery_day",
        "cust_days_since_last_success",
    }
    assert not banned & set(log.columns), (
        f"a removed salary proxy is back: {banned & set(log.columns)}"
    )
