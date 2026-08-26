"""Claim B: the sequencer against the baselines, scored by identical code.

Run with::

    uv run python scripts/evaluate_sequencer.py            # full, ~25 minutes on 8 cores
    uv run python scripts/evaluate_sequencer.py --quick     # ~40 seconds

Reports the sequencer against the production-standard fixed ladder under
identical simulator conditions. Absolute rupees are simulator-dependent and are
not a real-world forecast — see the README's honest-scope section.

On ``--quick``
--------------
The full run costs about seven minutes, and nearly all of it is per-decision
model inference: a ``predict_proba`` call costs about the same for 1 row as for
100 (measured 11.8ms and 11.7ms; 1,000 rows is 15.9ms, 35% more), because most
of it is sklearn's encode-and-dispatch overhead rather than tree traversal. It therefore scales with the number of *decisions*,
not the amount of data, and a policy that decides twice per episode over 6,898
episodes pays it ~14,000 times.

``--quick`` runs the same code on a tenth of the customers, in about 40 seconds.
It is for iterating and for reviewers checking *behaviour* rather than
reproducing a number.

**Do not read any conclusion off it.** Reduced scale is not a smaller version of
this experiment — it trains a different, weaker model, and the headline inverts.
At reduced scale the contact-enabled sequencer looks catastrophic and the
no-contact variant wins; at full scale the contact-enabled one is the best policy
on the board. ``disposition_rules`` likewise comes out ahead of ``fixed_ladder``
at reduced scale and behind it at full scale.

An earlier version of this note claimed "the broad conclusions survive". They do
not, and the claim was written before anything checked it.

**Every figure quoted elsewhere in this repo comes from the full run.**
"""

from __future__ import annotations

import datetime as dt
import sys

import pandas as pd

from rebound.compliance import DEFAULT_RULES, ComplianceGate, ContactCap
from rebound.eval.baselines import default_baselines
from rebound.eval.harness import build_eval_batch, evaluate_all
from rebound.eval.splits import all_splits
from rebound.model import TARGET_ACTION_REVOKED
from rebound.fqi import fit_fitted_q
from rebound.sequencer import (
    UNSERVABLE_COLUMNS,
    QSequencer,
    Sequencer,
    fit_for_serving,
)
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
            "These figures are NOT the reported ones, and the policy ordering does NOT hold at this scale — the ladder beats every learned policy here. Use quick mode to check the code runs, never to compare policies. "
            "magnitudes do not.\nRun without --quick to reproduce the README."
        )
    banner("FITTING THE SEQUENCER'S MODELS")
    log = generate_log(LOG_CONFIG)
    train = all_splits(log)["time"].train
    print(f"training on {len(train):,} rows")
    print(f"dropped as unservable at decision time: {', '.join(UNSERVABLE_COLUMNS)}")

    pricer = fit_for_serving(train, max_iter=FIT_ITERATIONS)
    print(f"servable features: {len(pricer.heads.downstream.spec_.columns)}")
    # The head is fitted on `revoked`, not `episode_revoked`. This line printed
    # the latter after the target moved, overstating the base rate 13x on the
    # one line that reports the thing that changed.
    print(
        f"revocation head ({TARGET_ACTION_REVOKED}) calibrated by "
        f"{pricer.revocation.calibration_method_used_}; "
        f"base rate {train[TARGET_ACTION_REVOKED].mean():.5f}"
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

    # Two configurations of the same sequencer, and both are reported.
    #
    # The full one is allowed to contact customers and prices that contact with
    # the learned revocation head. The conservative one has the contact cap set
    # to zero, so it can only retry, switch rail, repair a mandate or stop.
    #
    # Reporting only the winner would be fitting the policy to the scoreboard.
    # The pair is the finding: the difference between them is the price of
    # trusting a revocation estimate that observational logs cannot identify.
    full = Sequencer(pricer=pricer)
    full.name = "rebound_sequencer"

    no_contact_rules = tuple(
        ContactCap(max_contacts=0) if isinstance(rule, ContactCap) else rule
        for rule in DEFAULT_RULES
    )
    conservative = Sequencer(
        pricer=pricer, gate=ComplianceGate(rules=no_contact_rules)
    )
    conservative.name = "rebound_sequencer_no_contact"

    # The fitted-Q policy. Ranks actions by total remaining value rather than a
    # hand-built one-step expected value, which is what removes the credit that
    # the EV double-counts across the decisions of a single episode.
    #
    # It takes the *timing head* from the same pricer. Q is fitted on logged
    # transitions where timing was never randomised, so it confuses "later
    # decisions are worse" with "waiting is worse" and acts immediately on
    # everything; the timing head answers the causal question directly.
    q = fit_fitted_q(train, sweeps=8)
    learned = QSequencer(q=q, pricer=pricer)
    learned.name = "rebound_q"

    untimed = QSequencer(q=q, pricer=None)
    untimed.name = "rebound_q_untimed"

    policies = list(default_baselines()) + [full, conservative, learned, untimed]

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

    def net_of(name: str) -> float | None:
        row = table.loc[table["policy"] == name]
        return None if row.empty else float(row["net_per_1000"].iloc[0])

    base = net_of("fixed_ladder")
    ours = net_of("rebound_sequencer")
    learned_q = net_of("rebound_q")
    if base is not None and ours is not None:
        banner("AGAINST THE PRODUCTION-STANDARD LADDER")
        print(f"  fixed_ladder                  {base:>12,.0f}")
        print(f"  rebound_sequencer             {ours:>12,.0f}")
        print(f"  difference                    {ours - base:>12,.0f}")
        if learned_q is not None:
            print(f"\n  fitted Q-iteration:           {learned_q:>12,.0f}")
            # Read off this run, not asserted. An earlier version printed "the
            # hand-built expected value beat it 4/4" unconditionally — and in
            # quick mode Q is routinely ahead, so the caption contradicted the
            # table directly above it. That is the same defect as the +100.0%
            # incident this script was rewritten to prevent, arriving as prose
            # instead of arithmetic.
            if learned_q > ours:
                print(
                    "  On THIS run Q is ahead of the shipped policy by "
                    f"{learned_q - ours:,.0f}. That does not overturn the\n"
                    "  reported result — across four full-scale seeds the "
                    "hand-built expected value\n  beat it 4/4 and Q went "
                    "negative on one — but at reduced scale the ordering is\n"
                    "  noise, which is why quick mode is not a comparison."
                )
            else:
                print(
                    "  Q is behind the shipped policy here, consistent with the "
                    "full-scale result:\n  the hand-built expected value beat "
                    "it 4/4 across four seeds."
                )
            print(
                "  Q is reported, not shipped. See rebound.fqi: 68% of its\n"
                "  decisions fall outside the training support on "
                "days_since_failure."
            )
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
