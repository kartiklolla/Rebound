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
| PR-AUC | 0.6213 | 0.6578 |
| ROC-AUC | 0.9186 | 0.8954 |
| ECE | 0.0108 | 0.0035 |
| Precision @ 10% capacity | 0.6479 | 0.7199 |
| failure-code prior (baseline) | 0.5483 | 0.5596 |
| **Timing head** (`succeeded`, collecting actions) | | |
| PR-AUC | 0.5901 | 0.6098 |
| ROC-AUC | 0.9424 | 0.9182 |
| ECE | 0.0074 | 0.0090 |

**The caveat belongs next to the number, not below it.** The action head's ROC-AUC of 0.919
is mostly the model separating hopeless failure dispositions from live ones — which the
failure taxonomy already encodes. Within disposition, where the decisions are actually
hard, discrimination falls to 0.72–0.86 on the time split, and `merchant_fix` on the
customer split is 0.64. Lift over the failure-code prior is +0.073 PR-AUC: real, but
modest.

**Two heads, compared honestly.** The heads cannot be ranked by reading their headline
numbers against each other — those are computed on different rows, against different
labels, at different base rates. The comparison worth making holds the rows fixed and
swaps only the label, and `scripts/train_model.py` prints it rather than this README
asserting it:

| Same rows (collecting actions), time split | Timing head | Action head |
|---|---:|---:|
| ROC on `succeeded` | **0.9424** | 0.9132 |
| ROC on `episode_recovered` | 0.8708 | **0.9201** |

Each head wins its own question and loses the other. That is what justifies keeping both.
Had either won on both, the other would be dead weight and should be deleted.

Two salary-cycle proxy features were built, measured, and deleted — one was a duplicate
of `billing_day`, the other correlated 0.20 with the hidden latent and still made the
model worse. An oracle handed the true latent reached 0.7205 against 0.6275 for the model
as it stood at that time, so most of the timing signal is still unclaimed. That gap has
not been re-measured since the fixes below, and the old figures are not carried forward as
though they were current. Full per-slice breakdown and the measurements behind both
deletions are in [PROGRESS.md](PROGRESS.md).

### What a second audit found

The model layer was audited independently after it was built, by an agent with no access
to the reasoning that produced it. Six defects, four in code rather than prose, all fixed
and all with regression tests:

- **The calibrator was selected under the wrong generalisation regime.** The inner
  calibration split was always temporal, even when the outer split held out whole
  customers — so on the customer split the calibrator was chosen on people the booster had
  already memorised, then applied to strangers. It confidently picked isotonic, which was
  worse on *every* held-out metric and cost 0.013 PR-AUC. The inner split now mirrors the
  outer one. Customer-split ECE went 0.0079 → 0.0035 and calibration stopped costing any
  discrimination.
- **Two features were byte-identical.** `decision_index` was documented as an identifier
  but never added to the identifier set, so it reached the model as a twin of
  `prior_attempts`. Permutation importance shuffles one column at a time, so each twin
  masked the other and the published ranking was measured wrong.
- **Two tests asserted things that could not fail.** One checked that a sorted column was
  sorted; another re-derived a slice using the same arithmetic it was meant to be
  checking. Both passed while the defect they were named for was present in the data.
- **The two-head comparison was not like-for-like** — corrected above.
- **Excluding `customer_id` does not prevent customer memorisation.** The tuple
  `(bank, billing_day, amount, ceiling, rail)` is unique across all 5,974 customers, and
  no allowlist can drop it without dropping five legitimate features. Quantified in
  `eval/splits.py`, and it is the reason the customer split is reported at all.

The four code defects moved the headline numbers *up*, which is the part worth noticing:
every one of them had been making the model quietly worse while the tests reported green.

### Claim B — policy vs baselines

Model-driven sequencer not yet built. Baselines measured, floor established:
recovering payments cuts revocation from 8.78% (abandon everything) to 6.00% (retry
ladder), while chasing relentlessly pushes it to 14.54% and destroys more value than
doing nothing.

## The compliance gate

The agent proposes; the gate disposes. Not by convention — structurally. An
executor accepts only an `ApprovedAction`, and one cannot be minted outside
`rebound.compliance`, so there is no path from "I'd like to retry this" to "a
retry happened" that skips adjudication. The alternative design, a
`check_compliance()` call before acting, is correct only as long as every future
call site remembers one line, and that failure is silent.

**Three verdicts, not two.** A gate that only says yes or no turns every timing
rule into a lost recovery — "denied" tells a sequencer to abandon a retry that is
perfectly legal in four hours. `DEFER` carries the moment it becomes permissible,
and that moment is re-adjudicated until it settles, because constraints here are
intervals rather than deadlines: clearing the 24-hour notice requirement can drop
you inside a closed NPCI window.

**Law and house policy are labelled separately.** Five regulatory rules (mandate
alive, pre-debit notice, AFA ceiling, execution windows, presentation cap) and
four policy ones (quiet hours, contact cap, spend budget, terminal stop). A
merchant must be able to see which constraints they may tune, and this system
must not claim regulatory cover for its own preferences — labelling a
self-imposed contact cap "compliance" makes a product decision unarguable by
misattributing it to a regulator.

**The regulatory constants are unverified, and the gate says so.**
`unverified_rules()` names every rule resting on secondary reporting rather than
a primary source, and the NPCI window conflict is recorded as disputed with a
note on which direction the encoded reading errs. An audit trail implying a
diligence that was never done is worse than no audit trail. Confirming these
against the circulars is tracked as open work, not quietly assumed.

## Where we deliberately did not use an LLM

Compliance is the clearest case. A gate decision has to be deterministic,
reproducible on replay, explainable by citation to the rule that fired, and
identical for two customers in identical circumstances. A language model is none
of those. The gate is boolean logic because boolean logic is the correct tool.

Retry timing and action selection are calibrated-probability problems, not language
problems. An LLM asked to choose a retry moment would be uncalibrated, non-reproducible
across runs, orders of magnitude more expensive per decision, and unauditable — and the
output here gets multiplied by a rupee amount to make a spend decision, so the number has
to *be* a probability.

Calibration itself is treated the same way: `none`, `sigmoid` and `isotonic` are fitted
on one slice, scored on a disjoint held-out one, and the winner is whichever measured
best. It chooses differently for the two heads — sigmoid for the action head, isotonic for
the timing head, on both splits — so a single fixed choice would have been wrong for one
of them.

The audit found this measurement was worth less than it looked. The slice it selected on
did not match the condition the model was scored under, and a selection made on the wrong
distribution is confident and reproducible and still wrong. Choosing by measurement is
only better than choosing by assumption if the measurement is taken under the right
regime.

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
