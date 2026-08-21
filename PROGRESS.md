# Progress

**Project:** Rebound — a recovery orchestrator for failed recurring debits
**Track:** Razorpay AI Buildathon, Track 03 — AI Revenue Recovery
**Deadline:** 2026-09-05 · **Target submit:** 2026-09-03 (2 days of buffer, deliberately)
**Started:** 2026-08-21

---

## Status at a glance

| Part | Scope | State |
|------|-------|-------|
| 01 | Project scaffold | ✅ done |
| 02 | Failure taxonomy (domain model) | ✅ done — 27 codes, 3 rails, 144 tests |
| 03 | Synthetic mandate-failure generator | 🟡 in progress |
| 04 | Metric harness + baseline policies | ⬜ not started |
| 05 | Recovery-probability model + calibration | ⬜ not started |
| 06 | Compliance gate (non-bypassable) | ⬜ not started |
| 07 | Sequencer / agent policy | ⬜ not started |
| 08 | LLM comms layer (Hinglish/multilingual) | ⬜ not started |
| 09 | Batch runner + demo dashboard | ⬜ not started |
| 10 | README, metrics writeup, provenance table | ⬜ not started |

Legend: ⬜ not started · 🟡 in progress · ✅ done

---

## What this is

Recurring debits in India fail constantly — UPI Autopay mandates, eNACH bank debits,
card-on-file subscriptions. Most processors respond with a fixed retry ladder: try
again in 1 day, 3 days, 7 days, then give up. That ladder ignores *why* the debit
failed, ignores *when* the customer actually has money, and burns gateway fees and
customer patience on retries that were never going to succeed.

Rebound classifies the failure, predicts recovery probability as a function of action
and timing, and runs a bounded recovery workflow under a compliance layer it cannot
bypass — reporting money recovered, cost per rupee recovered, and an honest list of
what it gave up on.

---

## The two claims (kept separate on purpose)

A panel's first attack on any simulator-based project is "you evaluated your policy on
your own simulator, so of course it wins." That is a fair attack. So the claims are
split and scoped differently:

**Claim A — the model.** The recovery-probability model is trained on a disjoint time
window and never sees the generator's parameters. It has to *learn* effects (salary-day
seasonality, bank-specific downtime, failure-code priors) from data alone. Judged on
honest held-out ML metrics: PR-AUC, calibration, Brier score, on a time-based split.

**Claim B — the policy.** Judged only as *relative lift over the production-standard
fixed retry ladder*, evaluated on the same simulator under identical conditions.
Absolute ₹ recovered is simulator-dependent and is reported as such, never as a
real-world forecast.

This split is the honest version. Stated up front in the README, not buried.

---

## Day plan

| Days | Dates | Work |
|------|-------|------|
| 1–2 | Aug 21–22 | Taxonomy, generator, **metric harness first**, baselines |
| 3–5 | Aug 23–25 | Recovery model, calibration, time-split eval, beat baselines |
| 6–8 | Aug 26–28 | Compliance gate, stopping rules, sequencer, audit log |
| 9–10 | Aug 29–30 | LLM comms layer, root-cause narratives |
| 11–12 | Aug 31–Sep 1 | Batch runner, dashboard, exception list |
| 13 | Sep 2 | README: metrics, cost curve, provenance, "where we didn't use AI" |
| 14 | Sep 3 | 5-minute pitch video · **submit** |
| 15 | Sep 4–5 | Buffer |

---

## Decisions log

Short records of choices that would otherwise be re-litigated later, or that a panel
is likely to ask about.

### D1 — Metric harness is built before the model (2026-08-21)
If the harness is written after the model, it gets unconsciously shaped to flatter
whatever the model already does. Locking the harness first is what makes the numbers
worth reporting. Baselines land in the same commit as the harness, before any learned
model exists.

### D2 — `sklearn.HistGradientBoostingClassifier` over LightGBM/XGBoost (2026-08-21)
Same algorithm family, comparable accuracy on tabular data of this size, but zero extra
dependencies and native support for `CalibratedClassifierCV`. Calibration matters more
than the last point of AUC here, because the policy makes rupee decisions off these
probabilities — they need to *be* probabilities, not just rank correctly.

### D3 — Time-based split, never random (2026-08-21)
Recovery behaviour is temporal (salary cycles, bank downtime windows, mandate ageing).
A random split leaks the future into the past and inflates every metric. Train on an
early window, test on a strictly later one.

### D4 — The taxonomy encodes structure, never probabilities (2026-08-21)
`rebound.taxonomy` says which actions are *legal* for a failure, never how likely they
are to work. Structural facts (a cancelled mandate cannot be debited) are true by
definition of the rail. Recovery rates are the generator's opinion — if they were also
written into the taxonomy, the model would read the generator's answer key through a
side channel and Claim A's held-out metrics would be worthless.

Enforced mechanically by `test_taxonomy_encodes_no_probabilities`, which fails if any
numeric field is ever added to `FailureMode`. Adding one requires deleting that test:
a deliberate speed bump, not an obstacle to route around.

### D5 — Unknown failure codes raise instead of defaulting (2026-08-21)
An unmapped code means the rail changed under us. Silently bucketing it as `TERMINAL`
would write off recoverable money with no signal that anything happened. `UnknownFailureCode`
is fatal by design.

---

## Metrics

Filled in as they land. Empty cells are honest — they mean not yet measured.

### Claim A — recovery-probability model (held-out, time-split)

| Metric | Value |
|--------|-------|
| PR-AUC | — |
| ROC-AUC | — |
| Brier score | — |
| Calibration slope | — |
| Baseline (failure-code prior only) PR-AUC | — |

### Claim B — policy vs baselines (held-out batch)

| Policy | Recovery rate | ₹ recovered / 1000 | Cost per ₹ recovered | Attempts used |
|--------|---------------|--------------------|-----------------------|---------------|
| Fixed 3-retry ladder (production standard) | — | — | — | — |
| Immediate retry | — | — | — | — |
| Rebound (ours) | — | — | — | — |

---

## To verify from primary sources before the README claims it

These are encoded in the compliance layer and will be checked by anyone who works in
this domain. Currently based on general knowledge and **must** be confirmed against
RBI / NPCI circulars, with citations in the README.

- [ ] RBI e-mandate pre-debit notification window (currently assuming 24h)
- [ ] AFA (additional factor of authentication) threshold for recurring card debits
- [ ] NPCI retry caps per mandate per cycle for UPI Autopay
- [ ] NACH presentation/re-presentation limits
- [ ] Any restriction on debit attempt timing (quiet hours, business days)

---

## Open questions

- Whether to anchor generator failure-code distributions to published NPCI aggregate
  decline statistics (strongly preferred for credibility — synthetic data anchored to
  real published aggregates is a different credibility class from invented numbers).
