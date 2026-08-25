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
- **Claim B (the policy)** is reported against the production-standard fixed ladder under
  identical simulator conditions and paired random draws. Absolute rupees are
  simulator-dependent and are not a real-world forecast — read the *ordering* of the
  policies, not the magnitudes.

  Reported as a **difference, never a ratio.** Several policies score negative net value
  here, and a ratio of two negatives reads as a gain when the second is worse than the
  first. That is not hypothetical: it is exactly how this project's evaluation script
  once reported +100.0% for a policy that had crashed.

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

### What review found in the evaluation itself

The sequencer's first reported result was **+100.0% over the fixed ladder**, for
a policy that had crashed. It exceeded the harness's 120-second budget after
2,001 of 6,898 episodes; `evaluate_all` isolates a failed policy and substitutes
an all-zero report; zero revocation and zero contacts then sorted that row to the
top of a table ordered by net value; and a lift computed against a negative
baseline turned "did nothing at all" into a gain. The failure notice printed one
line above the table that contradicted it.

No single piece of that was a bug. Sorting by net value, isolating crashes so one
bad policy does not destroy a half-hour run, and expressing lift as a ratio are
all reasonable. They composed into a fabricated headline. The script now refuses
to tabulate an incomplete run at all, and reports a difference rather than a
ratio.

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

**Measured on five world seeds that policy selection never touched.** Full
scale, ~6,900 failed debits per seed, paired random draws.

| Policy | Recovery | Revocation | Contacts/ep | Net ₹/1000 | Gap vs ladder |
|---|---:|---:|---:|---:|---:|
| **`rebound_sequencer`** | **0.540** | 0.052 | 0.95 | −97,278 | **+56,787** |
| `fixed_ladder` | 0.458 | 0.048 | 0.00 | −154,065 | — |
| `immediate_retry` | 0.407 | 0.052 | 0.00 | −271,976 | −117,911 |
| `disposition_rules` | 0.428 | 0.061 | 0.90 | −369,653 | −215,588 |
| `no_recovery` | 0.000 | 0.088 | 0.00 | −1,274,866 | −1,120,801 |
| `aggressive_contact` | 0.306 | 0.133 | 3.62 | −1,516,667 | −1,362,603 |

Gap over the ladder, per seed: **+11,992 / +38,466 / +59,435 / +73,236 /
+100,804**. Mean **+56,787**, sd 33,754, **positive on 5 of 5**.

**Every net figure is negative, the ladder's included.** "Ahead" means less bad,
not profitable. These are simulator rupees — read the ordering, not the
magnitudes.

### Why these seeds and not the earlier ones

Roughly ten policy variants were compared during development on a fixed set of
four seeds. Ten looks at a metric whose seed standard deviation is ~96,000 on
that set is ten chances to get lucky, and nothing in the project protected
against that — every other selection decision here is split-protected, and
policy selection was not.

So Claim B was re-measured on five seeds selection had never seen. It cost the
headline most of its size:

| | mean gap | min | seeds won |
|---|---:|---:|---:|
| Selection seeds | +164,807 | +76,248 | 4/4 |
| **Held-out seeds** | **+56,787** | **+11,992** | **5/5** |

**The selection-set figure was inflated about 2.9×.** The direction survived and
the magnitude did not, which is the outcome this protocol exists to detect. The
held-out number is the one reported.

### What Claim B is not

Four things the number above does not support, stated because each was
initially believed here and had to be withdrawn.

**Not a claim that fitted Q-iteration works here.** It was built properly —
ledger-derived rewards reconciling exactly with `episode_net_paise`, backward
induction, double-Q on episode-atomic partitions, pessimistic combination — and
it **lost**. Across four full-scale seeds the hand-built expected value beat it
4/4, and Q went *negative* on one, losing to a three-line retry ladder.

The diagnosis is the useful part. `EXPLORATION_WEIGHTS` randomised *actions*;
nothing randomised *timing*. The log has a hard floor at `days_since_failure =
1.0` — zero of 88,178 rows below it — while the candidate grid starts at zero
delay, so **68% of that policy's decisions fall outside the training support**.
Fitted-Q is the correct method applied to a log built to answer a different
question, and the fix is a generator that randomises timing, not a better
regressor. See `rebound.fqi`.

**Not a claim that contact is profitable.** Two configurations of the same
sequencer are shipped and both are reported: `rebound_sequencer`, and
`rebound_sequencer_no_contact` with the contact cap set to zero. On the one
configuration a committed script reproduces — `scripts/evaluate_sequencer.py`
at its default world seed — enabling contact is worth **+6,757**, a gap over the
ladder of +29,680 with contact against +22,923 without. That is a fifth of one
seed standard deviation and settles nothing. The distance between the two is the
price of trusting a revocation estimate that observational logs cannot identify,
and reporting only the winner would be fitting the policy to the scoreboard.

An earlier version of this section put that difference at +51,550 across five
seeds, winning 4 of 5, p≈0.056. Those figures came from a multi-seed run with no
committed script and cannot be reproduced, so they are withdrawn rather than
carried forward.

**Not converged in training scale.** How much training log the policy's models
are fitted on moves the headline further than the headline itself. At a tenth
scale, `scripts/evaluate_sequencer.py --quick` puts the sequencer **158,186 behind** the
ladder where the full run puts it 29,680 ahead, inverts the ordering of the
contact-enabled and no-contact configurations, and lifts `disposition_rules`
above `fixed_ladder` where at full scale it is well below. A reduced-scale run is
not a small version of this experiment — it is a different, weaker model.

An earlier version of this section reported a 1,200 / 3,000 / 6,000 training-log
sweep. Its contact-enabled row no longer reproduces — the 6,000-customer cell
read +99,516 where that configuration now measures +29,680 — and the sweep has
no committed script, so it is withdrawn too.

**Not free of pricing error.** The policy under-prices a voice call by about 3×
relative to the world's compounding contact fatigue: the world's revocation
hazard compounds at `1.45 ** contacts` while the per-action label is flat in
`prior_contacts`, so three voice calls are priced at 0.0225 where the world
charges 0.0710. And the expected value credits every decision in an episode with
the whole episode's recovery — the defect fitted-Q was built to remove. Both are
open and quantified.

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

## The comms layer: the model writes, it does not decide

This is the one place a language model is clearly the right tool, and the one place where
giving it the obvious amount of authority would be a mistake.

By the time a drafter runs, everything carrying money or legal exposure is already fixed:
**whether** to contact by the sequencer's expected value, **when** by the gate's deferral
arithmetic, the amount and rail by the record, and — the one that matters most — **what
the customer is told to do**, derived from the failure's disposition by `comms.ask_for`.
What is left is a language problem: say this, to this person, in Hinglish, inside 134
characters.

The instruction is the boundary worth defending. A model that picks its own ask will
eventually tell a customer whose balance was short to replace a card that is fine. They
will replace it, the next debit will fail identically, and the contact is spent. So `Ask`
is a closed set of seven derived by a function with no judgement in it, and a check
confirms the message carries its own instruction and no other.

**Generate → verify → repair once → fall back.** Every draft is checked against the brief
by 13 deterministic checks. Nothing that fails is sent. A failure is fed back for one
repair attempt; a second failure falls back to a template proven by test to pass every
check for every combination of instruction, channel and language. That fallback is what
makes the model safe to use at all — without it the failure mode is either sending
something unverified or sending nothing.

The verifier calls no model. A model grading a model shares its failure modes and cannot
be cross-examined later by whoever has to decide whether a message should have gone out.

### Segment arithmetic, which turned out to be a language decision

One character outside GSM-7 re-encodes an entire SMS as UCS-2 and cuts the per-segment
budget from 153 characters to 67. Rendered from the shipped templates:

| Language | Encoding | SMS segments |
|---|---|---:|
| English | GSM-7 | 1 |
| Hinglish | GSM-7 | 1 |
| Hindi | UCS-2 | 2 |

The same reminder costs twice as much to send in Devanagari. That is a large part of why
Indian merchants send Hinglish — and Hinglish, whose spelling is not standardised, is
exactly what a template library is worst at and a model is good at. It is also why the
templates write `Rs.` and never `₹`: the rupee sign is not in GSM-7, so one of them
halves the message.

Three languages ship: English, Hindi, Hinglish. Not eight. A language ships when a native
speaker has read its fallback templates, because the fallback is what goes out on the
worst day, and an unreviewed fallback is a guaranteed send of text nobody checked.

### What the verifier catches, and what it does not

`scripts/evaluate_comms.py` runs a corpus of bad drafts through the checks. **28 of 28
probes are caught by the check written for them** — wrong amounts, invented reference
numbers, phishing hosts, leaked internal failure codes, credential solicitation, threats,
unsigned messages, wrong instructions, a pre-debit notice missing its disclosures, an
undeliverable message.

A count like that is only worth its weakest probe. An earlier version reported the link
check at 3/3 while `vahan-secure.ru/pay` was being cleared to send, because the one
scheme-less probe happened to sit inside the detector's blind spot. The link probes now
sit on the boundary: an unlisted top-level domain, a bare IP address, a lookalike dot, and
a `upi://` deep link to an attacker's VPA.

That number alone would be worthless: the corpus and the checks have the same author, so
a high catch rate measures that author's imagination. What makes it evidence is the other
half of the report — **five probes the checks provably do not catch**, printed as
prominently as the wins:

- **A polite threat.** Coercion with no word from the lexicon.
- **A false causal claim.** "Your bank declined this payment" when the cause may have been
  our own missing pre-debit notice. The brief withholds the failure code on purpose, so no
  check here has anything to contradict it with.
- **A social-engineering setup.** "Our agent will call you shortly." Asks for no
  credential, so the lexicon has nothing to match.
- **Fluent but ungrammatical Hinglish.** A marker count cannot measure grammar.
- **Correct but cruel.** Every fact true; no check reads tone.

Three of the five are tone and intent, exactly where a deterministic check has nothing to
compare against. The checks are labelled by tier — *exact*, *bounded*, *lexical* — so a
clean run on the lexical ones is not read as a guarantee.

### Status of the live path

The verifier, the templates and the desk are exercised by 333 tests, including 27
mutations of the source that the suite catches. Ten of those tests build nothing by hand:
they sample a population, take the debits that actually failed, adjudicate through a real
compliance gate, and compose from the resulting approval. That is how the two worst
defects in this layer were found. **The live API path has not been run** —
there is no API key in this environment, so every number above is measured against the
verifier, the templates and a stub client. To exercise it:

```bash
ANTHROPIC_API_KEY=... uv run python scripts/evaluate_comms.py --model
```

## Adversarial review

The simulator, harness and policy interface were red-teamed black-box before the model
was built. Thirteen findings, five critical, all fixed, each with a regression test that
reproduces the original attack. See [docs/SECURITY_REVIEW.md](docs/SECURITY_REVIEW.md).

## Running it

```bash
uv sync
uv run pytest
```
