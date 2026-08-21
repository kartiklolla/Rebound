"""The recovery-probability model.

What it predicts
----------------
``P(the episode is ultimately recovered | observable state, action taken)``.

Not "did this action collect money" — that label scores every nudge at exactly
zero, because nudges unblock customers rather than collect from them. The target
is the return from the decision point, which is the quantity a sequencer needs
in order to choose between actions.

Why calibration is not an afterthought here
-------------------------------------------
The output is multiplied by a rupee amount to make a decision. A model that
ranks episodes perfectly and is systematically overconfident will lose money
with an excellent AUC, because "0.8" that really means 0.5 licenses spend that
was never justified.

So the base model is fitted on one slice and the calibrator on a **later,
disjoint** slice. Calibrating on data the base model has already fitted produces
a model that looks beautifully calibrated in-sample and is overconfident on
everything it has not seen — the exact failure the calibration was meant to
prevent, now wearing a reassuring number.

What it is not allowed to see
-----------------------------
Features come from :func:`rebound.sim.dataset.feature_columns`, an allowlist by
subtraction. Identifiers, outcome-derived columns and simulator latents are all
excluded. That matters more than it sounds: ``episode_net_paise`` predicts the
label at accuracy 1.000 against a base rate of 0.269, because it is the label
restated. See ``docs/SECURITY_REVIEW.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance

from rebound.sim.dataset import feature_columns

TARGET_DOWNSTREAM = "episode_recovered"
"""Did the episode ultimately recover. Answers *which action*."""

TARGET_IMMEDIATE = "succeeded"
"""Did this action collect the money right now. Answers *when*."""

TARGET_REVOKED = "episode_revoked"
"""Did the customer revoke the mandate. Answers *what it costs to be wrong*.

Predicting recovery is only half of an expected value. The measured baseline
result is that ``aggressive_contact`` recovers less than a naive ladder while
pushing revocation from 6.00% to 14.54%, destroying more value than abandoning
every failed debit — so a sequencer that can price recovery but not revocation
optimises the exact quantity that produced the worst policy in the table, and
does it while looking locally correct at every step.
"""

TARGET = TARGET_DOWNSTREAM

#: Actions that can actually collect money, and therefore the only rows on
#: which an immediate-success label means anything.
#:
#: A nudge is structurally incapable of collecting, so its immediate label is
#: always 0. Training the timing head on those rows teaches it that nudges
#: never work — true, irrelevant, and it swamps the base rate of the rows that
#: matter.
COLLECTING_ACTIONS: frozenset[str] = frozenset(
    {"retry_same_rail", "retry_alt_rail", "send_collect_link"}
)


def _split_holdout(
    frame: pd.DataFrame,
    *,
    group_by: str | None,
    order_by: str | None,
    fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut a frame into an earlier part and a held-out part, group-atomically.

    Two properties, and the split is worthless without both.

    **No group straddles the cut.** Rows are assigned by group, never
    individually. An episode with rows on both sides makes its held-out label
    readable from its training rows.

    **The cut matches the regime.** With ``order_by``, the holdout is the
    future and groups straddling the boundary are **dropped**. With
    ``order_by=None``, whole groups are held out at random, which is the
    cold-start condition, and nothing needs dropping.

    The embargo is not optional and cannot be replaced by cleverer ranking.
    Rank groups by their first timestamp and a long-running group reaches
    forward past the boundary; rank by their last and it reaches backward. A
    group that spans the cut has to go somewhere, and wherever it goes it
    carries rows from the other side. Measured here: episode-atomic but
    unembargoed, the two slices shared no episode and still overlapped by 26
    days of ``decided_at``. Dropping the straddlers is what the outer splitter
    already does, so this applies the same standard the project enforces on
    itself elsewhere.

    Falls back to a plain row cut when there is no usable group column or only
    one group, so a small hand-built test frame still works.
    """
    usable = group_by is not None and group_by in frame.columns
    ordered_by = order_by if order_by and order_by in frame.columns else None

    if usable and frame[group_by].nunique() > 1:
        grouped = frame.groupby(group_by, observed=True)

        if ordered_by:
            ends = grouped[ordered_by].max()
            starts = grouped[ordered_by].min()
            rank = ends.sort_values()
            sizes = grouped.size().reindex(rank.index)
            target = len(frame) * (1.0 - fraction)
            n_early = min(max(int((sizes.cumsum() <= target).sum()), 1), len(rank) - 1)

            boundary = rank.iloc[n_early - 1]
            early_groups = set(ends.index[ends <= boundary])
            late_groups = set(starts.index[starts > boundary])

            keys = frame[group_by]
            early = frame[keys.isin(early_groups)].sort_values(ordered_by)
            late = frame[keys.isin(late_groups)].sort_values(ordered_by)
            if len(early) and len(late):
                return early, late
            # Degenerate boundary (every group straddles). Fall through.
        else:
            keys_index = pd.Index(frame[group_by].unique())
            order = np.random.default_rng(seed).permutation(len(keys_index))
            rank = pd.Series(order, index=keys_index).sort_values()
            sizes = grouped.size().reindex(rank.index)
            target = len(frame) * (1.0 - fraction)
            n_early = min(max(int((sizes.cumsum() <= target).sum()), 1), len(rank) - 1)

            in_early = frame[group_by].isin(set(rank.index[:n_early]))
            return frame[in_early], frame[~in_early]

    ordered = frame.sort_values(ordered_by) if ordered_by else frame
    cut = int(len(ordered) * (1.0 - fraction))
    return ordered.iloc[:cut], ordered.iloc[cut:]


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10
) -> float:
    """Local ECE, to keep model selection independent of the reporting layer.

    Deliberately not imported from ``rebound.eval.metrics``. The model must not
    depend on the module that scores it, or tuning the model and moving the
    goalposts become the same edit.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.sum()
    return float(total / len(y_true))


# ==========================================================================
# Features
# ==========================================================================


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """The columns a model uses, and the category levels it was fitted on.

    Carrying the levels matters. If categories were re-derived per frame, a test
    set missing one failure code would silently shift every category index and
    the model would read entirely the wrong column values — a corruption that
    produces plausible-looking numbers rather than an error.
    """

    columns: tuple[str, ...]
    categorical: tuple[str, ...]
    boolean: tuple[str, ...]
    categories: dict[str, tuple[str, ...]]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> FeatureSpec:
        columns = feature_columns(frame)
        if not columns:
            raise ValueError("no selectable feature columns in this frame")

        categorical, boolean = [], []
        for column in columns:
            values = frame[column]
            if pd.api.types.is_bool_dtype(values):
                boolean.append(column)
            elif not pd.api.types.is_numeric_dtype(values):
                categorical.append(column)

        return cls(
            columns=tuple(columns),
            categorical=tuple(categorical),
            boolean=tuple(boolean),
            categories={
                column: tuple(sorted(frame[column].dropna().astype(str).unique()))
                for column in categorical
            },
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Project a frame onto the fitted feature space.

        Unseen category levels become NaN rather than a new code. The gradient
        booster handles missing values natively, and "a code I have never seen"
        is genuinely closer to missing than to any particular known code.
        """
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise ValueError(f"frame is missing fitted features: {sorted(missing)}")

        out = frame.loc[:, list(self.columns)].copy()
        for column in self.categorical:
            dtype = pd.CategoricalDtype(categories=self.categories[column])
            values = out[column].astype(str)
            # Blank unseen levels before the cast rather than during it: casting
            # with out-of-category values in place is deprecated, and silently
            # inventing a level would be worse than treating it as missing.
            out[column] = values.where(values.isin(dtype.categories)).astype(dtype)
        for column in self.boolean:
            out[column] = out[column].astype("int8")
        return out


# ==========================================================================
# Baselines
# ==========================================================================


class GlobalPrior:
    """Predicts the training base rate for everything.

    The floor. Any model that cannot beat this has learned nothing, and it is
    worth having on the table because PR-AUC against an unfamiliar base rate is
    otherwise hard to read.
    """

    name = "global_prior"

    def __init__(self) -> None:
        self.rate_ = 0.0

    def fit(self, frame: pd.DataFrame) -> GlobalPrior:
        self.rate_ = float(frame[TARGET].mean())
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.rate_)


class FailureCodePrior:
    """Predicts the historical recovery rate of the failure code and action.

    The strong baseline for Claim A, and the one that makes the model's lift
    honest. A merchant with a pivot table has this. If a gradient booster
    barely beats a group mean, the extra machinery is not carrying its weight
    and the right answer is to say so.
    """

    name = "failure_code_prior"

    def __init__(self, min_rows: int = 30) -> None:
        self.min_rows = min_rows
        self.rates_: dict[tuple[str, str], float] = {}
        self.fallback_ = 0.0

    def fit(self, frame: pd.DataFrame) -> FailureCodePrior:
        self.fallback_ = float(frame[TARGET].mean())
        grouped = frame.groupby(["failure_code", "action"], observed=True)[TARGET]
        stats = grouped.agg(["mean", "size"])
        # Thin cells fall back to the global rate. A group mean over eight rows
        # is noise wearing a decimal point.
        self.rates_ = {
            key: float(row["mean"])
            for key, row in stats.iterrows()
            if row["size"] >= self.min_rows
        }
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        keys = zip(frame["failure_code"], frame["action"])
        return np.array([self.rates_.get(key, self.fallback_) for key in keys])


# ==========================================================================
# The model
# ==========================================================================


class RecoveryModel:
    """Calibrated gradient-boosted estimate of downstream recovery probability."""

    name = "rebound_model"

    def __init__(
        self,
        target: str = TARGET_DOWNSTREAM,
        seed: int = 20260821,
        calibration_fraction: float = 0.25,
        calibration_method: str = "auto",
        max_iter: int = 300,
        learning_rate: float = 0.06,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
    ) -> None:
        if not 0.05 <= calibration_fraction <= 0.5:
            raise ValueError(
                f"calibration_fraction must be in [0.05, 0.5], got "
                f"{calibration_fraction}"
            )
        self.target = target
        self.seed = seed
        self.calibration_fraction = calibration_fraction
        self.calibration_method = calibration_method
        self.calibration_method_used_ = ""
        self.calibration_scores_: dict[str, float] = {}
        self.params = dict(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            categorical_features="from_dtype",
            random_state=seed,
        )
        self.spec_: FeatureSpec | None = None
        self.calibrated_ = None
        self.base_ = None
        self.fit_rows_ = 0
        self.calibration_rows_ = 0
        self._order_by: str | None = "decided_at"
        self._group_by: str | None = "episode_id"
        self.fit_index_: pd.Index = pd.Index([])
        self.calibration_index_: pd.Index = pd.Index([])
        self.selection_index_: pd.Index = pd.Index([])
        self.dropped_rows_ = 0

    def fit(
        self,
        frame: pd.DataFrame,
        order_by: str | None = "decided_at",
        group_by: str | None = "episode_id",
    ) -> RecoveryModel:
        """Fit the booster, then fit a calibrator on a disjoint held-out slice.

        The inner split has to match the **generalisation regime being claimed**,
        and getting that wrong is not a cosmetic error. This model previously
        always cut the inner split temporally. Under a time-based outer split
        that is right. Under a *customer*-based outer split it silently was not:
        3,533 of 3,712 calibration-slice customers also appeared in the base
        model's fit slice, so the calibrator was chosen on people the booster had
        memorised and then applied to strangers. It picked isotonic on a
        selection score of 0.0074 against 0.0153 for no calibration — and on the
        held-out test set isotonic was worse on *every* metric, ECE included
        (0.0102 against 0.0082), costing 0.0131 of PR-AUC. The selection was
        confident, reproducible, and drawn from the wrong distribution.

        So ``group_by`` and ``order_by`` are now passed by the caller to describe
        the regime, and both the inner split and the inner-inner selection split
        respect them:

        - **time split** → ``group_by="episode_id"``, ``order_by="decided_at"``.
          Fitted on history, calibrated on the most recent history, run on the
          future.
        - **customer split** → ``group_by="customer_id"``, ``order_by=None``.
          The calibrator is selected on customers the booster has never seen,
          which is the condition it will actually be scored under.

        ``group_by`` also makes the cut **group-atomic**, which the row cut was
        not. Ordering by ``failed_at`` sorted by episode *start*, so a long
        episode's later decisions landed on the far side of the cut: the fit and
        calibration slices shared an episode and overlapped by 27 days of
        ``decided_at``. Small in effect, but it meant the inner split failed the
        very cleanliness rule ``assert_split_is_clean`` enforces on the outer
        one. Groups are now ranked by their *last* timestamp, so a group cannot
        reach across its own boundary.
        """
        if self.target not in frame.columns:
            raise ValueError(f"frame has no {self.target!r} column")
        if frame[self.target].nunique() < 2:
            raise ValueError(
                f"{self.target} has a single class in this frame; there is nothing "
                f"to learn and every metric would be undefined"
            )

        self._order_by = order_by
        self._group_by = group_by
        fit_part, calibration_part = _split_holdout(
            frame,
            group_by=group_by,
            order_by=order_by,
            fraction=self.calibration_fraction,
            seed=self.seed,
        )

        if calibration_part[self.target].nunique() < 2:
            raise ValueError(
                "the calibration slice has a single outcome class. A calibrator "
                "fitted on it would map every probability to a constant."
            )

        self.spec_ = FeatureSpec.fit(fit_part)
        self.fit_rows_ = len(fit_part)
        self.calibration_rows_ = len(calibration_part)
        # Record the actual partition, not the recipe for it. The tests that
        # check disjointness previously re-derived the slices with the same
        # arithmetic the model used, so they asserted that a sorted column was
        # sorted and passed while the real slices shared an episode. Reporting
        # the row labels lets a test look up what was really fitted on.
        self.fit_index_ = fit_part.index
        self.calibration_index_ = calibration_part.index
        # Episodes straddling the inner boundary are embargoed, so the two
        # slices need not account for every row. Reported rather than silent.
        self.dropped_rows_ = len(frame) - len(fit_part) - len(calibration_part)

        self.base_ = HistGradientBoostingClassifier(**self.params)
        self.base_.fit(
            self.spec_.transform(fit_part), fit_part[self.target].astype(int)
        )

        self._fit_calibrator(calibration_part)
        return self

    def _fit_calibrator(self, calibration_part: pd.DataFrame) -> None:
        """Fit a calibrator, or decline to, based on measurement.

        Calibration is usually described as free. It is not. Isotonic
        regression is non-parametric and overfits on thin data; and a gradient
        booster trained with log-loss on a modest dataset is often already
        well calibrated, in which case *any* correction makes it worse.

        Both were observed here. On ~3,300 calibration rows the raw model had
        ECE 0.0232 and slope 0.970, and applying isotonic pushed it to 0.0317
        while also costing 2 points of PR-AUC. On ~26,000 rows the raw model
        was overconfident at ECE 0.0205 and isotonic improved it to 0.0079.

        So the method is chosen by measurement rather than by assumption. The
        calibration slice is split again, candidates are fitted on the first
        half and scored on the second, and **the raw model is one of the
        candidates**. If nothing beats it, nothing is applied.

        Selecting on a slice the base model never saw is what makes this
        legitimate rather than a way of picking the flattering number.
        """
        assert self.spec_ is not None and self.base_ is not None
        y = calibration_part[self.target].astype(int)

        if self.calibration_method != "auto":
            self.calibration_method_used_ = self.calibration_method
            self.calibrated_ = CalibratedClassifierCV(
                FrozenEstimator(self.base_), method=self.calibration_method
            ).fit(self.spec_.transform(calibration_part), y)
            return

        inner_fit, inner_eval = _split_holdout(
            calibration_part,
            group_by=self._group_by,
            order_by=self._order_by,
            fraction=0.5,
            seed=self.seed + 1,
        )
        self.selection_index_ = inner_eval.index
        eval_y = inner_eval[self.target].astype(int).to_numpy()

        candidates: dict[str, float] = {
            "none": _expected_calibration_error(
                eval_y, self.predict_proba_uncalibrated(inner_eval)
            )
        }
        fitted: dict[str, CalibratedClassifierCV] = {}

        if inner_fit[self.target].nunique() > 1 and len(inner_eval) > 0:
            for method in ("sigmoid", "isotonic"):
                try:
                    calibrator = CalibratedClassifierCV(
                        FrozenEstimator(self.base_), method=method
                    ).fit(
                        self.spec_.transform(inner_fit),
                        inner_fit[self.target].astype(int),
                    )
                except ValueError:
                    continue
                probs = calibrator.predict_proba(self.spec_.transform(inner_eval))[:, 1]
                candidates[method] = _expected_calibration_error(eval_y, probs)
                fitted[method] = calibrator

        self.calibration_scores_ = dict(candidates)
        best = min(candidates, key=candidates.get)  # type: ignore[arg-type]
        self.calibration_method_used_ = best

        if best == "none":
            self.calibrated_ = None
            return

        # Refit the winner on the whole calibration slice — the selection is
        # already made, so there is no reason to leave half the data unused.
        self.calibrated_ = CalibratedClassifierCV(
            FrozenEstimator(self.base_), method=best
        ).fit(self.spec_.transform(calibration_part), y)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Calibrated probability, or the raw model if calibration did not help.

        ``calibration_method_used_`` records which, and ``calibration_scores_``
        records what each candidate measured. Both are reported rather than
        left as an implementation detail — "we calibrated" and "we checked
        whether calibrating helped" are different claims.
        """
        if self.spec_ is None:
            raise RuntimeError("model is not fitted")
        return self.predict_proba_prepared(self.spec_.transform(frame))

    def predict_proba_prepared(self, prepared: pd.DataFrame) -> np.ndarray:
        """Predict from an already-encoded matrix.

        Exists because encoding costs as much as predicting. On a 15-row frame
        ``spec_.transform`` takes 6.9ms against 6.6ms for the calibrated
        predict, and the whole path is 14.6ms whether the frame holds 1 row or
        1,000 — the cost is per *call*, not per row, so it scales with the
        number of decisions a policy makes rather than with the data.

        Two models with identical specs can therefore share one encoding.
        ``ActionPricer`` does exactly that, and checks the specs match before
        it does: feeding a matrix encoded under one spec to a model fitted
        under another would predict on silently mis-ordered columns and return
        confident nonsense.
        """
        if self.base_ is None:
            raise RuntimeError("model is not fitted")
        if self.calibrated_ is None:
            return self.base_.predict_proba(prepared)[:, 1]
        return self.calibrated_.predict_proba(prepared)[:, 1]

    def predict_proba_uncalibrated(self, frame: pd.DataFrame) -> np.ndarray:
        """The raw booster output, for measuring what calibration actually did.

        Reported alongside the calibrated version. If the two score the same,
        the calibration step is ceremony and should be described as such.
        """
        if self.base_ is None or self.spec_ is None:
            raise RuntimeError("model is not fitted")
        return self.base_.predict_proba(self.spec_.transform(frame))[:, 1]

    def feature_importance(
        self, frame: pd.DataFrame, n_repeats: int = 5, sample: int = 4000
    ) -> pd.DataFrame:
        """Permutation importance on held-out data.

        Permutation rather than a split-count heuristic: it measures what the
        model's accuracy actually depends on, which is the question worth asking
        of a feature set that was deliberately restricted.
        """
        if self.spec_ is None or self.base_ is None:
            raise RuntimeError("model is not fitted")

        subset = (
            frame.sample(sample, random_state=self.seed)
            if len(frame) > sample
            else frame
        )
        result = permutation_importance(
            self.base_,
            self.spec_.transform(subset),
            subset[self.target].astype(int),
            n_repeats=n_repeats,
            random_state=self.seed,
            scoring="average_precision",
        )
        return (
            pd.DataFrame(
                {
                    "feature": self.spec_.columns,
                    "importance": result.importances_mean,
                    "std": result.importances_std,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ==========================================================================
# Two heads
# ==========================================================================


class TwoHeadedModel:
    """Two questions, two labels, two models.

    The sequencer asks two things that look similar and are not:

    **"When should I present this retry?"** The answer depends almost entirely
    on whether the account will have money on that date, which is driven by the
    customer's salary cycle. Measured on retry actions against insufficient
    funds, immediate success runs 0.65 within a few days of payday and 0.22
    three weeks later — a threefold spread.

    **"Which action should I take at all?"** The answer depends on the failure's
    disposition, what has already been tried, and how much contact the customer
    has absorbed. Timing barely enters into it.

    A single model trained on the downstream label answers the second question
    well and the first one not at all. That was measured rather than assumed:
    on the downstream label, adding a *perfect oracle view* of the hidden
    salary day moved PR-AUC from 0.7520 to 0.7870, while on the immediate label
    the same oracle moved it from 0.6362 to 0.7332. The downstream label
    aggregates the whole episode, so later actions wash out the timing of any
    single decision — it dilutes exactly the signal a timing decision runs on.

    Both labels are correct. Neither is sufficient. So there are two heads:

    ``immediate``
        Trained on ``succeeded`` over collecting actions only. Used to choose
        *when*, and to price a candidate retry moment.
    ``downstream``
        Trained on ``episode_recovered`` over everything. Used to choose
        *which action*, and to price the whole remaining episode.
    """

    name = "rebound_two_headed"

    def __init__(self, seed: int = 20260821, **model_kwargs) -> None:
        self.seed = seed
        self.model_kwargs = model_kwargs
        self.immediate = RecoveryModel(
            target=TARGET_IMMEDIATE, seed=seed, **model_kwargs
        )
        self.downstream = RecoveryModel(
            target=TARGET_DOWNSTREAM, seed=seed, **model_kwargs
        )
        self.immediate_rows_ = 0

    @staticmethod
    def collecting_rows(frame: pd.DataFrame) -> pd.DataFrame:
        """Rows where an immediate-success label carries information."""
        return frame[frame["action"].isin(COLLECTING_ACTIONS)]

    def fit(
        self,
        frame: pd.DataFrame,
        order_by: str | None = "decided_at",
        group_by: str | None = "episode_id",
    ):
        """Fit both heads under the same generalisation regime.

        ``order_by``/``group_by`` are forwarded unchanged: whichever regime the
        outer split is testing, both heads must have their calibrators selected
        under it. See ``RecoveryModel.fit`` for why a mismatch is not cosmetic.
        """
        collecting = self.collecting_rows(frame)
        if len(collecting) < 200:
            raise ValueError(
                f"only {len(collecting)} collecting-action rows; the timing "
                f"head has nothing to learn from"
            )
        self.immediate_rows_ = len(collecting)
        self.immediate.fit(collecting, order_by=order_by, group_by=group_by)
        self.downstream.fit(frame, order_by=order_by, group_by=group_by)
        return self

    def predict_immediate(self, frame: pd.DataFrame) -> np.ndarray:
        """P(this action collects the money now). For choosing *when*."""
        return self.immediate.predict_proba(frame)

    def predict_downstream(self, frame: pd.DataFrame) -> np.ndarray:
        """P(the episode ultimately recovers). For choosing *which action*."""
        return self.downstream.predict_proba(frame)
