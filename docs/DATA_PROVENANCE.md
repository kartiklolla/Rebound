# Data provenance

Where the numbers come from, what is real, and what is not.

## The short, unflattering version

The data is **entirely synthetic**. No production traffic, no real customers, no
real mandates.

Of the 26 behavioural parameters governing the simulated world:

| Provenance | Count | Share |
|------------|-------|-------|
| `ANCHORED` — traceable to a published figure | 1 | 4% |
| `DERIVED` — computed from published figures, reasoning recorded | 4 | 15% |
| `ASSUMED` — judgement, no source | 21 | 81% |

Four fifths of the parameters are assumptions. That is the honest number and it
is stated here rather than left to be discovered. What follows is the argument
for why the results are still worth something, and — more importantly — exactly
which results are *not*.

Every parameter's provenance is machine-checked: `tests/test_params.py` fails if
any parameter lacks an entry, and fails if anything claims `ANCHORED` or
`DERIVED` without citing a source.

## What the assumptions are, and why it is survivable

The assumed parameters are almost all **shapes**, not **levels**:

- How fast nudge efficacy decays with repeated contact
- How concentrated the salary effect is around payday
- The relative intrusiveness of a phone call versus an SMS
- The dispersion of outage durations across banks

The levels that matter most — the marginal failure rates the whole dataset is
scaled to — are anchored or derived from published NPCI figures.

This matters because of how the claims are scoped:

**Claim B (the policy) is a relative comparison.** Rebound and every baseline
run against the *same* world with the *same* parameters. If the fatigue decay
constant is wrong by 30%, every policy is affected by it, and the ranking
between them is largely preserved. The comparison is robust to these parameters
being wrong in *magnitude*.

It is **not** robust to them being wrong in *sign*. Where that risk is real, the
parameter says so in its provenance note, and `contact_fatigue_decay` — the one
Claim B is most sensitive to — gets an explicit sensitivity sweep in the
evaluation rather than a single point estimate.

**Claim A (the model) does not depend on the parameters being right at all.**
It is a supervised-learning result: the model is shown observable history and
has to predict recovery. Whether the real salary effect is 3× or 6× does not
change whether a model can learn a 5× effect from data that contains one. What
the parameters buy is that the learning problem *resembles* the real one.

## Published anchors

| Anchor | Value used | Source |
|--------|-----------|--------|
| eNACH debit failure rate | 0.31 (volume terms) | NPCI NACH data reported by Business Standard: ~31% of presentations failed by volume, ~24.9% by value |
| UPI Autopay mandate execution success | derived → 0.55 failure | NPCI data in trade press: success fell from ~50% (Jan 2024) to ~30% (Nov 2025) while volumes grew ~10× |
| UPI Autopay revocation volume | tunes churn-intent shape | Reported NPCI figures: >20 million mandates revoked per month, attributed largely to insufficient balances |
| UPI execution windows | before 10:00, 13:00–17:00, after 21:30 | NPCI guidelines on UPI/API usage, confining recurring executions to off-peak slots |
| Pre-debit notification | 24 hours | RBI e-mandate framework |
| Executions per cycle | 1 attempt + up to 3 retries | NPCI mandate execution rules |

### On the UPI failure rate

The published figure implies roughly a 70% failure rate. The simulator uses
**0.55**, deliberately lower.

The reason: the headline counts executions blocked by NPCI execution windows as
failures, and this world models execution-window blocking as a *separate*
mechanism that fires on top. Using 0.70 would count those failures twice. The
choice is recorded as `DERIVED` rather than `ANCHORED` precisely because a
judgement was applied to a published number.

### Sources disagree on execution windows

Some reporting describes a single post-21:30 window; more detailed coverage
describes three windows, consistent with 10:00–13:00 being the closed peak. The
three-window version is encoded, and the conflict is flagged in `PROGRESS.md`
under items to confirm against the primary NPCI circular before the README
asserts anything about it.

**Nothing regulatory in this repository should be treated as legal guidance.**
It is a modelling substrate that needs primary-source confirmation.

## Calibration

Failure rates are fitted empirically rather than solved in closed form. Drawing
failures compounds — a debit that already failed for insufficient funds cannot
also fail for another reason — and deterministic structural failures (expiry,
ceiling breaches, closed execution windows) stack on top of the stochastic ones.
Measuring the realised rate and adjusting absorbs all of those interactions
without enumerating them.

Achieved against target, at n≈45k/19k/19k presentations, seed 20260821:

| Rail | Target | Achieved |
|------|--------|----------|
| UPI Autopay | 0.550 | 0.5522 |
| eNACH | 0.310 | 0.3130 |
| Card on file | 0.280 | 0.2830 |

The calibration report is produced by a final verification pass that measures
without adjusting, so it describes the state the world is actually left in.

## What the model is allowed to see

This is the leakage control, and it is the part that makes Claim A meaningful.

**Hidden from the model** — the customer latents:

- salary day
- balance health
- engagement
- churn intent
- preferred channel
- every parameter in `rebound.sim.params`

**Visible to the model** — what a merchant would genuinely have:

- failure code, rail, timestamp, amount
- mandate age, billing day, cycle index
- the customer's own observable history: prior failures, prior actions taken and
  their outcomes

The model must *infer* that recovery is likelier just after payday. It is never
told when payday is. That inference is the learning problem.

Two structural defences back this up:

1. `rebound.taxonomy` encodes which actions are **legal**, never how likely they
   are to **work**. `test_taxonomy_encodes_no_probabilities` fails if a numeric
   field is ever added to it.
2. `rebound.economics` owns **costs**; `rebound.sim` owns **probabilities**.
   Cost-if-revoked is the merchant's own business knowledge and the policy may
   use it. Probability-of-revocation is the answer key and must be learned.

## The label, and what it does and does not mean

Each row in the log is one decision point, and carries two labels.

`succeeded` — did **this action** recover the money. Causally clean, and on its own
badly misleading: a nudge never collects money, it unblocks a customer so that a
later retry collects. Under this label alone, every nudge, notification and mandate
repair scores exactly zero.

`episode_recovered` — did the episode **ultimately** recover. This is the return from
the decision point, and it is what `rebound.model` is trained to predict, because it
is the quantity a sequencer needs in order to choose between actions.

**The caveat.** That return is realised under the *behavioural* policy — the randomised
ladder. It answers "what happened when a merchant took this action and then carried on
behaving like that merchant," not "what would happen under an optimal continuation."
Those differ, and the difference is a real limitation on Claim B.

Logged propensities are what make it measurable rather than rhetorical. Every row
records the probability with which the behavioural policy selected that action, and
`coverage_report()` shows the (disposition × action) cells where the log is thin. In
those cells, any claim is extrapolation rather than evidence, and they are reported
alongside the metrics rather than left for a reader to discover.

## Limitations

Stated plainly, because they are the first things a careful reader will look
for.

- **A simulator cannot validate a simulator.** Absolute rupees recovered is a
  property of this world, not a forecast about any merchant. It is reported as
  such and never as a business projection.
- **Off-policy evaluation.** The training log is generated by a randomised
  retry ladder. Where that behavioural policy never explored, the model has no
  data, and its estimates there rest on extrapolation.
- **No real seasonality beyond the salary cycle.** No festivals, no quarter-end,
  no monsoon effects on rural banking. Real Indian payment data has all of them.
- **Banks are anonymised.** `BANK_01`–`BANK_12` carry invented outage
  propensities. Attaching fabricated reliability figures to a named real
  institution would be publishing a claim about that organisation.
- **Customers do not learn.** In reality, a customer nudged every month starts
  ignoring nudges permanently, not just within one episode. Fatigue here resets
  at the episode boundary.

## Reproducing it

Every figure is regenerable from a seed and a parameter set:

```bash
uv run pytest
```

The default seed is `20260821`. Changing `WorldParams` changes the world;
changing the seed changes the sample.
