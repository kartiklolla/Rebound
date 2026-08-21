"""Metrics for both claims.

Claim A metrics judge a probability estimate. Claim B metrics judge money.
They are kept apart because conflating them is how a project ends up reporting
an AUC as though it were a business result.

On the choice of headline metric
--------------------------------
ROC-AUC is reported but is not the headline. It answers "can the model rank a
recovering episode above a non-recovering one," which is not the question — the
policy does not rank episodes against each other, it decides what to do about
one episode at a time, using the probability as a number.

So the metrics that lead are **PR-AUC** (the positive class is the minority and
that is what we care about finding) and **calibration** (a 0.3 has to mean 0.3,
because it gets multiplied by a rupee amount to make a decision). A model that
ranks perfectly and is systematically overconfident will lose money with an
excellent AUC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from rebound.economics import RUPEE


# ==========================================================================
# Claim A — probability estimates
# ==========================================================================


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    n: int
    base_rate: float
    pr_auc: float
    roc_auc: float
    brier: float
    calibration_slope: float
    calibration_intercept: float
    expected_calibration_error: float
    max_calibration_error: float
    capacity: float
    """The review capacity the precision/recall/lift figures below are cut at.

    Carried on the report rather than baked into the field names. An earlier
    version named these ``precision_at_10pct`` while honouring whatever
    ``capacity`` was passed, so calling it with ``capacity=0.5`` produced a
    field labelled ``at_10pct`` holding precision at 50%. For a project whose
    entire thesis is honest metrics that is the purest available own goal.
    """

    precision_at_capacity: float
    recall_at_capacity: float
    lift_at_capacity: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @property
    def capacity_label(self) -> str:
        """Human-readable cut, e.g. ``"precision@10%"``. Safe for headers."""
        return f"{self.capacity:.0%}"


def classification_report(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    capacity: float = 0.10,
    bins: int = 10,
) -> ClassificationReport:
    """Score a set of probability estimates.

    ``capacity`` is the share of cases a team could actually action — the
    precision and recall at that cut are what an ops lead cares about, where a
    threshold-free aggregate is not.
    """
    if not 0.0 < capacity <= 1.0:
        raise ValueError(
            f"capacity must be in (0, 1], got {capacity}. Values outside that "
            f"range silently collapsed to a single case in an earlier version."
        )
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")

    y_true, y_prob = _validate_probabilities(y_true, y_prob)
    base_rate = float(y_true.mean())
    single_class = len(np.unique(y_true)) < 2

    slope, intercept = _calibration_line(y_true, y_prob)
    ece, mce = _calibration_error(y_true, y_prob, bins)
    precision, recall = _precision_recall_at_capacity(y_true, y_prob, capacity)

    return ClassificationReport(
        n=len(y_true),
        base_rate=base_rate,
        pr_auc=float(average_precision_score(y_true, y_prob))
        if not single_class
        else float("nan"),
        roc_auc=float(roc_auc_score(y_true, y_prob))
        if not single_class
        else float("nan"),
        brier=float(brier_score_loss(y_true, y_prob)),
        calibration_slope=slope,
        calibration_intercept=intercept,
        expected_calibration_error=ece,
        max_calibration_error=mce,
        capacity=capacity,
        precision_at_capacity=precision,
        recall_at_capacity=recall,
        lift_at_capacity=precision / base_rate if base_rate > 0 else float("nan"),
    )


def _validate_probabilities(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Shared input gate for everything that scores probabilities.

    Shared on purpose. ``classification_report`` rejected non-finite values
    while ``reliability_table`` accepted them and bucketed NaN into the *top*
    calibration bin — inflating exactly the high-probability bin the policy
    spends most of its money in, and manufacturing an observed rate for it.
    Two functions disagreeing about the same input is how a bad number gets
    quoted from whichever one happened to be called.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true and y_prob differ in length: {len(y_true)} vs {len(y_prob)}"
        )
    if len(y_true) == 0:
        raise ValueError("cannot score an empty set")
    if not np.isfinite(y_prob).all():
        raise ValueError(
            "y_prob contains non-finite values (NaN or inf). These do not sort "
            "or bin meaningfully and would land in the highest-probability bin."
        )
    if (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob contains values outside [0, 1]")
    if not np.isin(y_true, (0.0, 1.0)).all():
        raise ValueError("y_true must contain only 0 and 1")
    return y_true, y_prob


def _calibration_line(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of observed frequency against predicted probability.

    Slope 1, intercept 0 is perfect. Slope below 1 means overconfidence — the
    model's extremes are more extreme than reality — which is the failure mode
    that turns a good ranker into a policy that overspends on cases it was too
    sure about.
    """
    if np.ptp(y_prob) < 1e-12:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(y_prob, y_true, 1)
    return float(slope), float(intercept)


def _calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, bins: int
) -> tuple[float, float]:
    """Expected and maximum calibration error over equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)

    total = 0.0
    worst = 0.0
    for b in range(bins):
        mask = index == b
        if not mask.any():
            continue
        gap = abs(y_true[mask].mean() - y_prob[mask].mean())
        total += gap * mask.sum()
        worst = max(worst, gap)
    return float(total / len(y_true)), float(worst)


def _precision_recall_at_capacity(
    y_true: np.ndarray, y_prob: np.ndarray, capacity: float
) -> tuple[float, float]:
    """Precision and recall over the top ``capacity`` share by predicted score."""
    k = max(1, int(round(len(y_prob) * capacity)))
    top = np.argsort(-y_prob)[:k]
    selected = y_true[top]
    positives = y_true.sum()
    precision = float(selected.mean())
    recall = float(selected.sum() / positives) if positives else float("nan")
    return precision, recall


def reliability_table(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    bins: int = 10,
) -> pd.DataFrame:
    """Predicted vs observed frequency per bin — the calibration plot, as data.

    Worth reading directly rather than only through a summary statistic: a
    model can post a respectable ECE while being badly wrong in exactly the
    high-probability bin the policy spends most of its money in.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")
    y_true, y_prob = _validate_probabilities(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(y_prob, edges[1:-1]), 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = index == b
        rows.append(
            {
                "bin_low": edges[b],
                "bin_high": edges[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()) if mask.any() else np.nan,
                "observed_rate": float(y_true[mask].mean()) if mask.any() else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table["gap"] = table["observed_rate"] - table["mean_predicted"]
    return table


# ==========================================================================
# Claim B — money
# ==========================================================================


@dataclass(frozen=True, slots=True)
class PolicyReport:
    """What a policy did to a batch of failed debits."""

    policy: str
    episodes: int

    recovery_rate: float
    revocation_rate: float

    recovered_paise: int
    spent_paise: int
    destroyed_paise: int
    net_paise: int

    attempts_per_episode: float
    contacts_per_episode: float

    @property
    def recovered_rupees_per_1000(self) -> float:
        return self.recovered_paise / RUPEE / self.episodes * 1000

    @property
    def net_rupees_per_1000(self) -> float:
        """The number that actually matters.

        Recovered, less what was spent chasing it, less the lifetime value
        destroyed by customers who cancelled because of the chasing. Reporting
        recovery alone is the flattering version — it makes "call everyone
        twice a day" look like a triumph.
        """
        return self.net_paise / RUPEE / self.episodes * 1000

    @property
    def cost_per_rupee_recovered(self) -> float:
        if self.recovered_paise == 0:
            return float("inf")
        return self.spent_paise / self.recovered_paise

    @property
    def true_cost_per_rupee_recovered(self) -> float:
        """Spend *and* destroyed value, per rupee recovered.

        The difference between this and ``cost_per_rupee_recovered`` is the
        cost of the churn a policy caused, which is invisible in every metric
        a merchant normally looks at.
        """
        if self.recovered_paise == 0:
            return float("inf")
        return (self.spent_paise + self.destroyed_paise) / self.recovered_paise

    def to_row(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "episodes": self.episodes,
            "recovery_rate": round(self.recovery_rate, 4),
            "revocation_rate": round(self.revocation_rate, 4),
            "recovered_rs_per_1000": round(self.recovered_rupees_per_1000, 1),
            "net_rs_per_1000": round(self.net_rupees_per_1000, 1),
            "cost_per_rs_recovered": round(self.cost_per_rupee_recovered, 4),
            "true_cost_per_rs_recovered": round(
                self.true_cost_per_rupee_recovered, 4
            ),
            "attempts_per_episode": round(self.attempts_per_episode, 2),
            "contacts_per_episode": round(self.contacts_per_episode, 2),
        }


def policy_comparison(reports: list[PolicyReport]) -> pd.DataFrame:
    """Side-by-side policy table, sorted by the metric that matters."""
    if not reports:
        raise ValueError("cannot build a comparison from zero reports")
    names = [report.policy for report in reports]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"duplicate policy names in comparison: {sorted(duplicates)}. "
            f"Rows would be indistinguishable and the floor lookup in "
            f"value_preserved() would silently pick whichever came first."
        )
    frame = pd.DataFrame([report.to_row() for report in reports])
    return frame.sort_values("net_rs_per_1000", ascending=False).reset_index(
        drop=True
    )


def value_preserved(
    reports: list[PolicyReport], floor: str = "no_recovery"
) -> pd.DataFrame:
    """Policy table expressed as value preserved against abandoning the debt.

    Every policy's raw ``net`` is negative, and correctly so: a failed cycle
    puts twelve cycles of lifetime value at risk while offering one cycle of
    revenue as the upside. Failures destroy value; the only question is how
    much of it a policy saves.

    Reported this way because "-447,375" invites the reading "this policy loses
    money", when what it means is "this policy loses much less than doing
    nothing". The floor policy is the denominator that makes any other number
    interpretable.
    """
    frame = policy_comparison(reports)
    floor_rows = frame[frame["policy"] == floor]
    if floor_rows.empty:
        raise ValueError(
            f"no report for floor policy {floor!r}; include it in the "
            f"comparison or the numbers have no reference point"
        )
    floor_net = float(floor_rows["net_rs_per_1000"].iloc[0])
    frame["value_preserved_rs_per_1000"] = (
        frame["net_rs_per_1000"] - floor_net
    ).round(1)
    return frame


def lift_over_baseline(
    candidate: PolicyReport, baseline: PolicyReport
) -> dict[str, float]:
    """Relative improvement over a baseline.

    This — not the absolute rupee figure — is the reportable form of Claim B.
    Absolute recovery is a property of the simulated world; the comparison
    between two policies run against the same world under common random numbers
    is the part that carries information.
    """

    if baseline.episodes == 0 or candidate.episodes == 0:
        raise ValueError(
            "cannot compute lift against a report with zero episodes"
        )

    def relative(new: float, old: float) -> float:
        # NaN, not an exception. Against the documented floor policy
        # (`no_recovery`) the recovery rate and attempt count are zero *by
        # construction*, so two of these four fields are legitimately
        # undefined — that is a property of the comparison, not a bug, and it
        # should print as NaN rather than abort the run.
        if old == 0:
            return float("nan")
        return (new - old) / abs(old)

    return {
        "net_lift": relative(
            candidate.net_rupees_per_1000, baseline.net_rupees_per_1000
        ),
        "recovery_rate_lift": relative(
            candidate.recovery_rate, baseline.recovery_rate
        ),
        "revocation_rate_change": relative(
            candidate.revocation_rate, baseline.revocation_rate
        ),
        "attempts_change": relative(
            candidate.attempts_per_episode, baseline.attempts_per_episode
        ),
    }
