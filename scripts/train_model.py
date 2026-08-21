"""Train the recovery-probability model and report it on both splits.

Run with::

    uv run python scripts/train_model.py

Reports both splits side by side and does not pick a winner between them. They
answer different questions, and a gap is a finding rather than something to
explain away.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from rebound.eval.metrics import (
    classification_report,
    reliability_table,
    slice_report,
)
from rebound.eval.splits import all_splits, assert_split_is_clean, split_report
from rebound.model import (
    TARGET_DOWNSTREAM,
    TARGET_IMMEDIATE,
    FailureCodePrior,
    GlobalPrior,
    TwoHeadedModel,
)
from rebound.sim.dataset import GenerationConfig, feature_columns, generate_log

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

CONFIG = GenerationConfig(
    n_customers=6_000,
    start=dt.date(2025, 1, 1),
    end=dt.date(2026, 6, 30),
    seed=20260821,
)


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> None:
    banner("GENERATING HISTORICAL LOG")
    log = generate_log(CONFIG)
    print(
        f"{len(log):,} decision points · {log['episode_id'].nunique():,} episodes "
        f"· {log['customer_id'].nunique():,} customers"
    )
    print(f"base rate ({TARGET_DOWNSTREAM}): {log[TARGET_DOWNSTREAM].mean():.4f}")
    print(f"selectable features: {len(feature_columns(log))}")

    banner("SPLITS")
    splits = all_splits(log)
    for split in splits.values():
        assert_split_is_clean(split)
    print(
        split_report(splits)[
            [
                "split",
                "train_rows",
                "test_rows",
                "train_customers",
                "test_customers",
                "dropped_rows",
            ]
        ].to_string(index=False)
    )
    print("\nboth splits verified clean")

    results: dict[str, dict[str, object]] = {}

    for name, split in splits.items():
        banner(f"{name.upper()} SPLIT")

        heads = TwoHeadedModel().fit(split.train)
        model = heads.downstream
        print(
            f"fitted on {model.fit_rows_:,} rows, calibrated on a disjoint "
            f"later {model.calibration_rows_:,}"
        )
        print(
            f"calibration chosen by measurement: {model.calibration_method_used_} "
            f"(held-out ECE "
            + ", ".join(
                f"{k}={v:.4f}" for k, v in sorted(model.calibration_scores_.items())
            )
            + ")"
        )

        scored = {
            "global_prior": GlobalPrior().fit(split.train).predict_proba(split.test),
            "failure_code_prior": FailureCodePrior()
            .fit(split.train)
            .predict_proba(split.test),
            "rebound (uncalibrated)": model.predict_proba_uncalibrated(split.test),
            "rebound": model.predict_proba(split.test),
        }

        truth = split.test[TARGET_DOWNSTREAM].astype(int)
        rows = []
        for label, probs in scored.items():
            report = classification_report(truth, probs, capacity=0.10)
            rows.append(
                {
                    "model": label,
                    "pr_auc": round(report.pr_auc, 4),
                    "roc_auc": round(report.roc_auc, 4),
                    "brier": round(report.brier, 4),
                    "calib_slope": round(report.calibration_slope, 3),
                    "ece": round(report.expected_calibration_error, 4),
                    f"prec@{report.capacity_label}": round(
                        report.precision_at_capacity, 4
                    ),
                    f"lift@{report.capacity_label}": round(
                        report.lift_at_capacity, 2
                    ),
                }
            )
        table = pd.DataFrame(rows)
        print(f"\nbase rate: {truth.mean():.4f}   n = {len(truth):,}")
        print(table.to_string(index=False))

        results[name] = {"model": model, "heads": heads, "probs": scored["rebound"], "split": split}

        imm_test = TwoHeadedModel.collecting_rows(split.test)
        imm_truth = imm_test[TARGET_IMMEDIATE].astype(int)
        imm = classification_report(imm_truth, heads.predict_immediate(imm_test))
        print(
            f"\nTIMING HEAD (target={TARGET_IMMEDIATE}, collecting actions only, "
            f"n={len(imm_test):,}, base rate {imm_truth.mean():.4f})"
        )
        print(
            f"  pr_auc={imm.pr_auc:.4f}  roc_auc={imm.roc_auc:.4f}  "
            f"ece={imm.expected_calibration_error:.4f}  "
            f"prec@{imm.capacity_label}={imm.precision_at_capacity:.4f}"
        )
        print(
            f"  trained on {heads.immediate_rows_:,} collecting rows; "
            f"calibration chose {heads.immediate.calibration_method_used_}"
        )

        print("\nreliability (calibrated):")
        rel = reliability_table(truth, scored["rebound"], bins=10)
        print(
            rel[rel["n"] > 0][
                ["bin_low", "bin_high", "n", "mean_predicted", "observed_rate", "gap"]
            ]
            .round(4)
            .to_string(index=False)
        )

        print("\nby disposition — no aggregate gets believed on its own:")
        print(
            slice_report(
                split.test.reset_index(drop=True),
                truth.to_numpy(),
                scored["rebound"],
                by="disposition",
            )
            .round(4)
            .to_string(index=False)
        )

    banner("WHAT THE MODEL LEANS ON (time split, permutation importance)")
    time_result = results["time"]
    importance = time_result["model"].feature_importance(  # type: ignore[union-attr]
        time_result["split"].test  # type: ignore[index]
    )
    print(importance.head(12).round(5).to_string(index=False))

    banner("TIME VS CUSTOMER SPLIT")
    for name, payload in results.items():
        split = payload["split"]
        report = classification_report(
            split.test[TARGET_DOWNSTREAM].astype(int), payload["probs"]  # type: ignore[arg-type]
        )
        print(
            f"  {name:9s} pr_auc={report.pr_auc:.4f}  roc_auc={report.roc_auc:.4f}  "
            f"ece={report.expected_calibration_error:.4f}  n={report.n:,}"
        )
    print(
        "\nA model strong on time and weak on customer has memorised people "
        "rather than learned structure, and will fail on every new signup."
    )


if __name__ == "__main__":
    main()
