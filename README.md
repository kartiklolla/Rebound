# Rebound

**A recovery orchestrator for failed recurring debits on Indian payment rails.**

Razorpay AI Buildathon · Track 03, AI Revenue Recovery

---

> 🚧 **In development.** Metrics tables in this README are filled in as they are
> measured. Empty means not yet measured, never "measured and omitted."
> See [PROGRESS.md](PROGRESS.md) for current state.

---

## The problem

Recurring debits in India fail constantly. A UPI Autopay mandate hits an account on the
28th when the salary lands on the 1st. An eNACH debit bounces during a bank's
maintenance window. A tokenised card expires. A customer revokes a mandate and nobody
notices for three cycles.

The standard response is a **fixed retry ladder** — retry at +1 day, +3 days, +7 days,
then write it off. That ladder is wrong in both directions at once:

- It **retries what can never succeed.** A revoked mandate will not start working
  because you asked it three more times. Every one of those attempts costs a gateway
  fee and a notification the customer reads as harassment.
- It **gives up on what would have succeeded.** An insufficient-funds failure on the
  28th has a very different outlook from the same failure on the 2nd. The ladder
  cannot see the difference, because it never looks at *why* the debit failed.

## What Rebound does

1. **Classifies the failure** into a disposition that determines which actions are even
   legal — retryable-with-timing, transient, needs-customer-action, needs-mandate-repair,
   or terminal.
2. **Predicts recovery probability** as a function of the action *and its timing*, from
   a calibrated model that learned the effects from data.
3. **Sequences a bounded recovery workflow** under a cost budget — retry, switch rail,
   nudge, send a collect link, request a re-mandate, or stop.
4. **Cannot bypass the compliance layer.** The agent proposes; the rule gate disposes.
5. **Reports honestly** — money recovered, cost per rupee recovered, and the exceptions
   it could not resolve, with reasons.

## Honest scope

This is evaluated against a **simulator**, not production traffic, and the README treats
that as a limitation to state rather than hide. The claims are split accordingly:

- **Claim A (the model)** is a real supervised-learning result on a time-based held-out
  split. The model never sees the generator's parameters and has to learn its effects
  from data.
- **Claim B (the policy)** is reported only as *relative lift over the production-standard
  fixed ladder*, under identical simulator conditions. Absolute rupees recovered are
  simulator-dependent and are not a real-world forecast.

## Metrics

### Claim A — recovery-probability models

162,743 decision points, 43,657 episodes, 5,974 customers. Both splits verified clean
before scoring.

Two heads, because choosing *when* to present a retry and choosing *which action* to
take are different questions with different labels. Immediate retry success runs 0.65
within a few days of the customer's payday and 0.22 three weeks later — but against the
episode-level label, even a perfect view of that latent adds almost nothing, because the
episode label washes out the timing of any single decision.

| | Time split | Customer split |
|---|---:|---:|
| **Action head** (`episode_recovered`) | | |
| PR-AUC | 0.6074 | 0.6325 |
| ROC-AUC | 0.9137 | 0.8941 |
| ECE | 0.0107 | 0.0102 |
| Precision @ 10% capacity | 0.6405 | 0.7034 |
| failure-code prior (baseline) | 0.5483 | 0.5596 |
| **Timing head** (`succeeded`, collecting actions) | | |
| PR-AUC | 0.5869 | 0.6232 |
| ROC-AUC | **0.9427** | **0.9187** |
| ECE | 0.0061 | 0.0085 |

**The caveat belongs next to the number, not below it.** The action head's ROC-AUC of 0.914 is
mostly the model separating hopeless failure dispositions from live ones — which the
failure taxonomy already encodes. Within disposition, where the decisions are actually
hard, discrimination is 0.71–0.77, and on `merchant_fix` it is 0.63. Lift over the
failure-code prior is +0.059 PR-AUC: real, but modest.

Two salary-cycle proxy features were built, measured, and deleted — one was a duplicate
of `billing_day`, the other correlated 0.20 with the hidden latent and still made the
model worse. An oracle given the true latent reaches 0.7205 against our 0.6275, so
roughly 80% of the timing signal remains unclaimed. Full per-slice breakdown and the
measurements behind both deletions are in [PROGRESS.md](PROGRESS.md).

### Claim B — policy vs baselines

Model-driven sequencer not yet built. Baselines measured, floor established:
recovering payments cuts revocation from 8.78% (abandon everything) to 6.00% (retry
ladder), while chasing relentlessly pushes it to 14.54% and destroys more value than
doing nothing.

## Where we deliberately did not use an LLM

Retry timing and action selection are calibrated-probability problems, not language
problems. An LLM asked to choose a retry moment would be uncalibrated, non-reproducible
across runs, orders of magnitude more expensive per decision, and unauditable — and the
output here gets multiplied by a rupee amount to make a spend decision, so the number has
to *be* a probability.

Calibration itself is treated the same way: `none`, `sigmoid` and `isotonic` are fitted
on one slice, scored on a later disjoint one, and the winner is whichever measured best.
It chose differently on the two splits, so a fixed choice would have been wrong on one of
them.

An LLM is used where language is genuinely the problem — customer communication and the
merchant-facing root-cause narrative.

## Adversarial review

The simulator, harness and policy interface were red-teamed black-box before the model
was built. Thirteen findings, five critical, all fixed, each with a regression test that
reproduces the original attack. See [docs/SECURITY_REVIEW.md](docs/SECURITY_REVIEW.md).

## Running it

```bash
uv sync
uv run pytest
```
