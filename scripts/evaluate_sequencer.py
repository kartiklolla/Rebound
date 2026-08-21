"""Claim B: the sequencer against the baselines, scored by identical code.

Run with::

    uv run python scripts/evaluate_sequencer.py            # full, ~7 minutes
    uv run python scripts/evaluate_sequencer.py --quick     # ~40 seconds

Reports the sequencer against the production-standard fixed ladder under
identical simulator conditions. Absolute rupees are simulator-dependent and are
not a real-world forecast — see the README's honest-scope section.

On ``--quick``
--------------
The full run costs about seven minutes, and nearly all of it is per-decision
model inference: a ``predict_proba`` call costs ~14.6ms whether the frame holds
1 row or 1,000, because the cost is sklearn's encode-and-dispatch overhead
rather than tree traversal. It therefore scales with the number of *decisions*,
not the amount of data, and a policy that decides twice per episode over 6,898
episodes pays it ~14,000 times.

``--quick`` runs the same code on a tenth of the customers, in about 40 seconds.
It is for iterating and for reviewers checking *behaviour* rather than
reproducing a number.

Do not read the ranking off it. The broad conclusions survive — the sequencer
sits behind the fixed ladder, ``aggressive_contact`` is worst, doing nothing is
close to worst — but policies that are close together do reorder: at reduced
scale ``disposition_rules`` comes out ahead of ``fixed_ladder``, and at full
scale it comes out behind. Checked rather than assumed, because "the ordering
holds" is exactly the sort of convenient claim that goes into a docstring
unverified.

**Every figure quoted elsewhere in this repo comes from the full run.**
"""

from __future__ import annotations

import datetime as dt
import sys

import pandas as pd

from rebound.eval.baselines import default_baselines
from rebound.eval.harness import build_eval_batch, evaluate_all
from rebound.eval.splits import all_splits
from rebound.sequencer import UNSERVABLE_COLUMNS, Sequencer, fit_for_serving
from rebound.sim.dataset import GenerationConfig, generate_log
from rebound.sim.world import World

pd.set_option("display.width", 200)

QUICK = "--quick" in sys.argv

LOG_CONFIG = GenerationConfig(
    n_customers=600 if QUICK else 6000,
    start=dt.date(2025, 1, 1),
    end=dt.date(2025, 12, 31) if QUICK else dt.date(2026, 3, 31),
    seed=20260821,
)

#: Evaluation batch size, kept in step with the log config above.
EVAL_CUSTOMERS = 400 if QUICK else 4000
EVAL_END = dt.date(2026, 5, 15) if QUICK else dt.date(2026, 6, 30)
FIT_ITERATIONS = 80 if QUICK else 200


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def main() -> None:
    if QUICK:
        banner(
            "QUICK MODE — REDUCED SCALE\n"
            "These figures are NOT the reported ones. Policy ordering holds; "
            "magnitudes do not.\nRun without --quick to reproduce the README."
        )
    banner("FITTING THE SEQUENCER'S MODELS")
    log = generate_log(LOG_CONFIG)
    train = all_splits(log)["time"].train
    print(f"training on {len(train):,} rows")
    print(f"dropped as unservable at decision time: {', '.join(UNSERVABLE_COLUMNS)}")

    pricer = fit_for_serving(train, max_iter=FIT_ITERATIONS)
    print(f"servable features: {len(pricer.heads.downstream.spec_.columns)}")
    print(
        f"revocation head calibrated by {pricer.revocation.calibration_method_used_}; "
        f"base rate {train['episode_revoked'].mean():.4f}"
    )

    banner("BUILDING THE EVALUATION BATCH")
    # A separate world from the one that produced the training log. A policy
    # scored on the episodes its model was fitted from is scored on its own
    # training set.
    world = World(seed=99001)
    customers = world.sample_customers(EVAL_CUSTOMERS)
    mandates = world.sample_mandates(
        customers, dt.date(2024, 6, 1), dt.date(2026, 3, 31)
    )
    world.calibrate(
        customers,
        mandates,
        [dt.date(2026, 4, 1) + dt.timedelta(days=i) for i in range(14)],
    )
    batch = build_eval_batch(
        world, customers, mandates, dt.date(2026, 4, 1), EVAL_END
    )
    print(f"{len(batch):,} failed debits")

    banner("CLAIM B — POLICIES UNDER IDENTICAL CONDITIONS")
    policies = list(default_baselines()) + [Sequencer(pricer=pricer)]

    # The default 120s budget was written for rule baselines that decide in
    # microseconds. A model-driven policy is legitimately slower, and the
    # honest response is to give it a budget it can finish in — not to let it
    # die and report the wreckage.
    results = evaluate_all(world, policies, batch, timeout_seconds=3600.0)

    # A crashed policy scores as an all-zero report. Zero revocation and zero
    # contacts then sort to the *top* of a table ordered by net value, and a
    # lift computed against a negative baseline turns "did nothing at all" into
    # +100%. This script printed exactly that for a sequencer that timed out
    # after 2,001 of 6,898 episodes. A failed run is not a result and is not
    # tabulated as one.
    failed = {name: r.error for name, r in results.items() if r.error}
    if failed:
        banner("RUN FAILED — NO NUMBERS ARE REPORTED")
        for name, error in failed.items():
            print(f"  {name}: {error}")
        raise SystemExit(
            "a policy did not complete the batch; a partial run cannot be "
            "compared against complete ones"
        )

    rows = []
    for name, result in results.items():
        report = result.report
        rows.append(
            {
                "policy": name,
                "recovery_rate": round(report.recovery_rate, 4),
                "revocation_rate": round(report.revocation_rate, 4),
                "contacts_per_episode": round(report.contacts_per_episode, 2),
                "net_per_1000": round(report.net_rupees_per_1000),
                "recovered_per_1000": round(report.recovered_rupees_per_1000),
            }
        )
    table = pd.DataFrame(rows).sort_values("net_per_1000", ascending=False)
    print(table.to_string(index=False))
    if QUICK:
        print("\n  (quick mode — reduced scale, not the reported figures)")

    ladder = table.loc[table["policy"] == "fixed_ladder"]
    seq = table.loc[table["policy"] == "rebound_sequencer"]
    if not ladder.empty and not seq.empty:
        base = float(ladder["net_per_1000"].iloc[0])
        ours = float(seq["net_per_1000"].iloc[0])
        banner("AGAINST THE PRODUCTION-STANDARD LADDER")
        print(f"  fixed_ladder      {base:>12,.0f}")
        print(f"  rebound_sequencer {ours:>12,.0f}")
        print(f"  difference        {ours - base:>12,.0f}")
        # No percentage. Both figures can be negative, and a ratio of two
        # negatives reads as a gain when the second is worse than the first.
        print(
            "\n  "
            + (
                "the sequencer is ahead"
                if ours > base
                else "the sequencer is BEHIND the naive ladder"
            )
        )


if __name__ == "__main__":
    main()
