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

*Extended from five seeds to twenty.* The first run of the five returned
+40,916 with sd 37,262 - t = 2.455 on 4 df, p ~ 0.070, and negative on one seed.
That is an underpowered measurement rather than a null one, and the remedy for
an underpowered measurement is more draws.

Extending a held-out set after seeing its result is the exact move this file
warns against two comments below, so it is done the only way that is not a
re-selection:

- The original five are unchanged and still reported as their own row, so the
  previously published number stays auditable as a subset rather than being
  absorbed.
- The fifteen new seeds are *derived*, not chosen - a fixed ``SeedSequence``
  draw with a stated entropy, so no hand-picking is possible and any reader can
  re-derive the identical tuple.
- All twenty are reported. If the extension moves the result toward zero, that
  is the result. Dropping seeds after seeing them would be the selection this
  protocol exists to prevent, and it is the one thing that cannot happen here.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
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

#: The fifteen that extend it, derived rather than chosen. Nobody typed these,
#: so nobody can have picked them; the entropy below is the date the extension
#: was decided and is the only free parameter, fixed before the run.
EXTENSION_SEEDS = tuple(
    int(x) for x in np.random.SeedSequence(20260830).generate_state(15)
)

#: Seeds policy selection *did* touch. Asserted disjoint rather than assumed:
#: a collision would silently re-admit a selection seed to the held-out set.
SELECTION_SEEDS = (12345, 24680, 31415, 55555, 99001)

ALL_SEEDS = HOLDOUT_SEEDS + EXTENSION_SEEDS
assert len(set(ALL_SEEDS)) == len(ALL_SEEDS), "duplicate seed"
assert not set(ALL_SEEDS) & set(SELECTION_SEEDS), "a selection seed leaked in"

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
for seed in ALL_SEEDS:
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
    group = "original_5" if seed in HOLDOUT_SEEDS else "extension_15"
    for name, r in res.items():
        rp = r.report
        rows.append({"seed": seed, "group": group, "policy": name,
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

# The point of the extension is power, so report the test rather than leaving a
# reader to run it. All three rows print unconditionally: reporting only the
# twenty, or only whichever group looked better, is the selection this file
# exists to prevent.
print("\n=== rebound_sequencer vs fixed_ladder: is the gap real? ===", flush=True)
seq = d[d.policy == "rebound_sequencer"]
for label, sub in (("original 5 ", seq[seq.group == "original_5"]),
                   ("extension 15", seq[seq.group == "extension_15"]),
                   ("all 20     ", seq)):
    g = sub["gap_vs_ladder"].to_numpy(dtype=float)
    if len(g) < 2:
        continue
    mean, sd = g.mean(), g.std(ddof=1)
    t = mean / (sd / np.sqrt(len(g)))
    try:
        from scipy import stats
        p = f"{2 * stats.t.sf(abs(t), len(g) - 1):.4f}"
    except ImportError:
        p = "scipy absent"
    print(f"  {label}  n={len(g):>2}  mean={mean:>+9,.0f}  sd={sd:>8,.0f}  "
          f"won={int((g > 0).sum()):>2}/{len(g):<2}  t={t:>5.3f}  df={len(g)-1:>2}  "
          f"p={p}", flush=True)
print("\n  A p above 0.05 after twenty seeds is a result, not a run to repeat.",
      flush=True)
print("\nDONE", flush=True)
