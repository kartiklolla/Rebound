"""Claim B on seeds that policy selection never saw.

Committed because it was not, and that was the defect. PROGRESS cited
``runs/holdout.py`` as the provenance of the headline number while the file
lived outside the repository, so the one figure this project leads with was the
one an outside reader could not reproduce - held to a looser standard than the
claims it withdraws elsewhere for exactly that reason.

Roughly ten policy variants have been compared on seeds 12345/24680/31415/
55555/99001. Ten looks at a metric whose seed standard deviation is ~96,000 is
ten chances to get lucky, and nothing in the project protected against that.

These five seeds have not been used for any selection decision. Whatever they
say is the reported Claim B number.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from rebound.eval.baselines import default_baselines
from rebound.eval.harness import build_eval_batch, evaluate_all
from rebound.eval.splits import all_splits
from rebound.sequencer import Sequencer, fit_for_serving
from rebound.sim.dataset import GenerationConfig, generate_log
from rebound.sim.world import World

#: Fixed, and fixed before they were run. Changing this tuple after seeing a
#: result would turn the held-out protocol back into the selection it replaced.
HOLDOUT_SEEDS = (777001, 8675309, 20260905, 424242, 1000003)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("holdout.csv"))
parser.add_argument(
    "--customers", type=int, default=6000,
    help="training log size; the shipped number is 6000",
)
args = parser.parse_args()

log = generate_log(GenerationConfig(n_customers=args.customers, start=dt.date(2025, 1, 1),
                                    end=dt.date(2026, 3, 31), seed=20260821))
train = all_splits(log)["time"].train
pricer = fit_for_serving(train, max_iter=200)
print("pricer fitted", flush=True)

rows = []
for seed in HOLDOUT_SEEDS:
    w = World(seed=seed)
    cs = w.sample_customers(4000)
    ms = w.sample_mandates(cs, dt.date(2024, 6, 1), dt.date(2026, 3, 31))
    w.calibrate(cs, ms, [dt.date(2026, 4, 1) + dt.timedelta(days=i) for i in range(14)])
    batch = build_eval_batch(w, cs, ms, dt.date(2026, 4, 1), dt.date(2026, 6, 30))

    seq = Sequencer(pricer=pricer)
    seq.name = "rebound_sequencer"
    res = evaluate_all(w, list(default_baselines()) + [seq], batch,
                       timeout_seconds=3600.0)
    bad = {n: r.error for n, r in res.items() if r.error}
    if bad:
        print(f"seed {seed} FAILED {bad}", flush=True)
        continue

    base = res["fixed_ladder"].report.net_rupees_per_1000
    for name, r in res.items():
        rp = r.report
        rows.append({"seed": seed, "policy": name,
                     "gap_vs_ladder": round(rp.net_rupees_per_1000 - base),
                     "net": round(rp.net_rupees_per_1000),
                     "recovery": round(rp.recovery_rate, 4),
                     "revocation": round(rp.revocation_rate, 4),
                     "contacts": round(rp.contacts_per_episode, 2)})
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"--- seed {seed} (n={len(batch)}, ladder {base:,.0f}) ---", flush=True)
    print(pd.DataFrame([r for r in rows if r["seed"] == seed])
          .sort_values("net", ascending=False).to_string(index=False), flush=True)

d = pd.DataFrame(rows)
print("\n=== HELD-OUT SEEDS: gap over fixed_ladder ===", flush=True)
print(d.pivot(index="seed", columns="policy", values="gap_vs_ladder").to_string(), flush=True)
print("\n=== SUMMARY ===", flush=True)
print(d.groupby("policy")[["gap_vs_ladder", "net", "recovery", "revocation", "contacts"]]
      .agg(["mean", "std", "min", "max"]).round(3).to_string(), flush=True)
print("\nDONE", flush=True)
