# Rebound

**A recovery orchestrator for failed recurring debits on Indian payment rails.**

Razorpay AI Buildathon · Track 03, AI Revenue Recovery

```bash
uv sync
uv run python scripts/demo.py     # ~1 min, no API key, no network
```

---

## What broke, and how we caught it

Most of the engineering here went into finding out that our own results were
wrong. Each line links to the section with the full account.

1. **Our evaluation reported +100.0% for a policy that had crashed** — four
   reasonable design choices composing into a fabricated headline. → [the +100% incident](#the-1000-incident)
2. **A policy could read its own random stream out of its episode ID**, lifting
   recovery 0.505 → 0.754 without tripping a single integrity check — because it
   never tampered with anything, it read. → [why measured twice](#why-these-seeds-and-why-they-were-measured-twice)
3. **Common random numbers were not common.** Two policies met provably identical
   churn hazards and different outcomes on 15.75% of episodes. → [why measured twice](#why-these-seeds-and-why-they-were-measured-twice)
4. **The calibrator was chosen under the wrong generalisation regime** and
   confidently picked the option that was worse on *every* held-out metric. → [second audit](#what-a-second-audit-found)
5. **Two features were byte-identical**, so each masked the other under
   permutation importance and the published ranking was measured wrong. → [second audit](#what-a-second-audit-found)
6. **A phishing URL was cleared to send** while the link check reported 3/3,
   because the one scheme-less probe sat in the detector's blind spot. → [the verifier](#what-the-verifier-catches-and-what-it-does-not)
7. **Every stale figure in this README had drifted in the flattering direction** —
   found by an outside reviewer recomputing them. → [Claim A](#claim-a--recovery-probability-models)
8. **Claim B has been measured three times. Each time it got smaller**, and the
   third measurement is not statistically significant. → [Claim B](#claim-b--policy-vs-baselines)

Four further claims were withdrawn rather than carried forward. They are named in
[what Claim B is not](#what-claim-b-is-not).

---

## The problem

Recurring debits in India fail constantly. A UPI Autopay mandate hits an account on the
28th when the salary lands on the 1st. An eNACH debit bounces during a bank's maintenance
window. A tokenised card expires. A customer revokes a mandate and nobody notices for
three cycles.

The standard response is a **fixed retry ladder** — +1 day, +3 days, +7 days, then write
it off. That ladder is wrong in both directions at once:

- It **retries what can never succeed.** A revoked mandate will not start working because
  you asked it three more times. Every attempt costs a gateway fee and a notification the
  customer reads as harassment.
- It **gives up on what would have succeeded.** An insufficient-funds failure on the 28th
  has a very different outlook from the same failure on the 2nd. The ladder cannot see the
  difference, because it never looks at *why* the debit failed.

## What Rebound does

1. **Classifies the failure** into a disposition that determines which actions are even
   legal — retryable-with-timing, transient, needs-customer-action, needs-mandate-repair,
   or terminal.
2. **Predicts recovery probability** as a function of the action *and its timing*, from a
   calibrated model that learned the effects from data.
3. **Sequences a bounded recovery workflow** under a cost budget — retry, switch rail,
   nudge, send a collect link, request a re-mandate, or stop.
4. **Cannot bypass the compliance layer.** The agent proposes; the rule gate disposes.
5. **Reports honestly** — money recovered, cost per rupee recovered, and the exceptions it
   could not resolve, with reasons.

## Honest scope

This is evaluated against a **simulator**, not production traffic. The claims are split
accordingly:

- **Claim A (the model)** is a real supervised-learning result on a time-based held-out
  split. The model never sees the generator's parameters and has to learn its effects from
  data.
- **Claim B (the policy)** is reported against the production-standard fixed ladder under
  identical simulator conditions and paired random draws. Absolute rupees are
  simulator-dependent — read the *ordering* of the policies, not the magnitudes.

Claim B is reported as a **difference, never a ratio.** Several policies score negative net
value here, and a ratio of two negatives reads as a gain when the second is worse than the
first — which is [exactly how](#the-1000-incident) this project once reported +100.0% for a
policy that had crashed.

---

## Claim A — recovery-probability models

163,129 decision points, 43,789 episodes, 5,981 customers. Both splits verified clean
before scoring. Reproduce with `uv run python scripts/train_model.py`.

Two heads, because choosing *when* to present a retry and choosing *which action* to take
are different questions with different labels. Immediate retry success runs 0.65 within a
few days of the customer's payday and 0.22 three weeks later — but against the
episode-level label, even a perfect view of that latent adds almost nothing, because the
episode label washes out the timing of any single decision.

| | Time split | Customer split |
|---|---:|---:|
| **Action head** (`episode_recovered`) | | |
| PR-AUC | 0.6045 | 0.6693 |
| ROC-AUC | 0.9182 | 0.9045 |
| ECE | 0.0257 | 0.0068 |
| Precision @ 10% capacity | 0.6298 | 0.7248 |
| failure-code prior (baseline) | 0.5406 | 0.5595 |
| **Timing head** (`succeeded`, collecting actions) | | |
| PR-AUC | 0.5778 | 0.6243 |
| ROC-AUC | 0.9397 | 0.9215 |
| ECE | 0.0216 | 0.0074 |

**Calibration on the time split is the weak number**, and it got worse rather than better:
held-out ECE 0.0257 against 0.0068 on the customer split. The selection slice picked
sigmoid on an inner ECE of 0.0101 and the held-out figure is 2.5× that — the calibrator
generalises across customers and not across time. The policy multiplies these probabilities
by rupee amounts, so this is the figure to be least comfortable with.

**The caveat belongs next to the number.** The action head's ROC-AUC of 0.918 is mostly the
model separating hopeless failure dispositions from live ones — which the failure taxonomy
already encodes. Within disposition, where the decisions are actually hard, discrimination
falls to **0.72–0.82** on the time split (`terminal` is the outlier at 0.97, and it is the
easy one). On the customer split **`merchant_fix` is 0.55 on 462 rows — essentially
chance.** Lift over the failure-code prior is **+0.064** PR-AUC on time, +0.110 on customer:
real, but modest.

Those four figures read 0.72–0.86, 0.64 and +0.073 until an outside reviewer recomputed
them. Every one had drifted in the flattering direction, in the paragraph whose entire job
is to name this model's weakness — the headline table had been regenerated and the prose
around it had not.

**Two heads, compared honestly.** The heads cannot be ranked by reading their headline numbers
against each other — different rows, different labels, different base rates. The comparison
worth making holds the rows fixed and swaps only the label:

| Same rows (collecting actions), time split | Timing head | Action head |
|---|---:|---:|
| PR on `succeeded` | **0.5778** | 0.4511 |
| ROC on `succeeded` | **0.9397** | 0.9042 |
| PR on `episode_recovered` | 0.6437 | **0.6526** |
| ROC on `episode_recovered` | 0.9001 | **0.9164** |

Each head wins its own question and loses the other — which is what justifies keeping both.
The PR margin on the downstream label is under 0.01, thin enough that a reduced-scale fixture
inverts it, so the unit test asserts ROC and this table is where the PR comparison is
established.

Two salary-cycle proxy features were built, measured, and deleted — one duplicated
`billing_day`, the other correlated 0.20 with the hidden latent and still made the model worse.
An oracle handed the true latent reached 0.7205 against 0.6275 for the model as it stood then,
so most of the timing signal is still unclaimed. That gap has not been re-measured since the
fixes below. Full per-slice breakdown in [PROGRESS.md](PROGRESS.md).

### What a second audit found

The model layer was audited independently after it was built, by an agent with no access to
the reasoning that produced it. Six defects, four in code, all fixed with regression tests:

- **The calibrator was selected under the wrong generalisation regime.** The inner
  calibration split was always temporal, even when the outer split held out whole customers
  — so on the customer split the calibrator was chosen on people the booster had already
  memorised, then applied to strangers. It confidently picked isotonic, which was worse on
  *every* held-out metric and cost 0.013 PR-AUC. The inner split now mirrors the outer one;
  customer-split ECE went 0.0079 → 0.0035.
- **Two features were byte-identical.** `decision_index` was documented as an identifier but
  never added to the identifier set, so it reached the model as a twin of `prior_attempts`.
  Permutation importance shuffles one column at a time, so each twin masked the other and
  the published ranking was measured wrong.
- **Two tests asserted things that could not fail.** One checked that a sorted column was
  sorted; another re-derived a slice using the same arithmetic it was meant to be checking.
  Both passed while the defect they were named for was present in the data.
- **The two-head comparison was not like-for-like** — corrected above.
- **Excluding `customer_id` does not prevent customer memorisation.** The tuple
  `(bank, billing_day, amount, ceiling, rail)` is unique across all 5,981 customers, and no
  allowlist can drop it without dropping five legitimate features. Quantified in
  `eval/splits.py`, and it is why the customer split is reported at all.

The four code defects moved the headline numbers *up*. Every one had been making the model
quietly worse while the tests reported green.

---

## Claim B — policy vs baselines

**Measured on five world seeds that policy selection never touched**, and re-measured from
scratch after two independent sweeps found defects in the harness itself. Full scale, ~6,900
failed debits per seed, paired random draws. Reproduce with
`uv run python scripts/holdout.py`.

| Policy | Recovery | Revocation | Contacts/ep | Net ₹/1000 | Gap vs ladder |
|---|---:|---:|---:|---:|---:|
| **`rebound_sequencer`** | **0.511** | 0.049 | 0.98 | −54,696 | **+40,916** |
| `fixed_ladder` | 0.460 | 0.047 | 0.00 | −95,613 | — |
| `immediate_retry` | 0.408 | 0.052 | 0.00 | −221,097 | −125,485 |
| `disposition_rules` | 0.431 | 0.056 | 0.90 | −233,064 | −137,452 |
| `aggressive_contact` | 0.274 | 0.101 | 3.67 | −1,114,588 | −1,018,976 |
| `no_recovery` | 0.000 | 0.087 | 0.00 | −1,221,633 | −1,126,020 |

Gap over the ladder, per seed: **−9,596 / +19,282 / +42,038 / +74,125 / +78,733**.
Mean **+40,916**, sd 37,262, **positive on 4 of 5**.

**This is not statistically significant.** t = 2.455 on 4 df, p ≈ 0.070. The sequencer loses
to the fixed ladder on one of the five held-out seeds. The direction is consistent and the
effect is not resolved by five seeds — that is the honest description.

**Every net figure in this table is negative, the ladder's included.** "Ahead" means less
bad, not profitable. These are simulator rupees — read the ordering, not the magnitudes.

### Why these seeds, and why they were measured twice

Roughly ten policy variants were compared during development on a fixed set of four seeds. Ten
looks at a metric whose seed standard deviation is ~96,000 is ten chances to get lucky, and
policy selection was the one decision here that was not split-protected. So Claim B was
re-measured on five seeds selection had never seen.

Then the harness those seeds ran through turned out to have two defects, both found by review
rather than by any test here:

- **Common random numbers were not common.** Every draw came from one sequential stream, so
  the passive-churn draw that settles an episode depended on how many draws the policy had
  already made. Two policies met provably identical churn hazards and different outcomes on
  15.75% of episodes — worth 177,000–319,000 ₹/1000 of pure alignment noise against a policy
  gap of 316,000, and on one batch it reversed two baselines outright.
- **A policy could read its own random stream.** The per-episode seed was `seed + index` and
  the episode id was `EV_{index:08d}`, so the index was handed to the policy in its own
  view. A demonstration policy that reconstructed the uniforms and burned 2-paise emails to
  align its single retry lifted recovery from 0.505 to 0.754 and net value 7.4×, without
  tripping a single integrity check — because it never tampered with anything.

So the numbers above are the third measurement, not the first:

| | mean gap | min | seeds won |
|---|---:|---:|---:|
| Selection seeds (4) | +164,807 | +76,248 | 4/4 |
| Held-out seeds, old harness | +56,787 | +11,992 | 5/5 |
| **Held-out seeds, fixed harness** | **+40,916** | **−9,596** | **4/5** |

Each re-measurement cost the headline size and cost it certainty. The selection-set figure
was inflated about 2.9×; correcting the harness took another 28% off what remained and
turned one seed negative. The direction has survived all three; nothing else has.

### The +100.0% incident

The sequencer's first reported result was **+100.0% over the fixed ladder**, for a policy
that had crashed. It exceeded the harness's 120-second budget after 2,001 of 6,898 episodes;
`evaluate_all` isolates a failed policy and substitutes an all-zero report; zero revocation
and zero contacts then sorted that row to the top of a table ordered by net value; and a
lift computed against a negative baseline turned "did nothing at all" into a gain. The
failure notice printed one line above the table that contradicted it.

No single piece of that was a bug. Sorting by net value, isolating crashes so one bad policy
does not destroy a half-hour run, and expressing lift as a ratio are all reasonable. They
composed into a fabricated headline. The script now refuses to tabulate an incomplete run at
all, and reports a difference rather than a ratio.

### What Claim B is not

Four things the number above does not support, stated because each was initially believed
here and had to be withdrawn.

**Not a claim that fitted Q-iteration works here.** It was built properly — ledger-derived
rewards reconciling exactly with `episode_net_paise`, backward induction, double-Q on
episode-atomic partitions, pessimistic combination — and it **lost**. Across four full-scale
seeds the hand-built expected value beat it 4/4, and Q went *negative* on one, losing to a
three-line retry ladder.

The diagnosis is the useful part. `EXPLORATION_WEIGHTS` randomised *actions*; nothing
randomised *timing*. The log has a hard floor at `days_since_failure = 1.0` — zero of 88,178
rows below it — while the candidate grid starts at zero delay, so **68% of that policy's
decisions fall outside the training support**. Fitted-Q is the correct method applied to a
log built to answer a different question, and the fix is a generator that randomises timing,
not a better regressor. See `rebound.fqi`.

**Not a claim that contact is profitable.** Two configurations are shipped and both reported:
`rebound_sequencer`, and `rebound_sequencer_no_contact` with the contact cap at zero. On the
one configuration a committed script reproduces — `scripts/evaluate_sequencer.py` at its
default world seed — enabling contact is worth **+6,757**: a gap over the ladder of +29,680
with contact against +22,923 without. That is a fifth of one seed standard deviation and
settles nothing. The distance between the two is the price of trusting a revocation estimate
that observational logs cannot identify, and reporting only the winner would be fitting the
policy to the scoreboard.

*Withdrawn:* an earlier version put that difference at +51,550 across five seeds, winning 4
of 5, p≈0.056. Those figures came from a multi-seed run with no committed script and cannot
be reproduced.

**Not converged in training scale.** How much training log the policy's models are fitted on
moves the headline further than the headline itself. At a tenth scale,
`scripts/evaluate_sequencer.py --quick` puts the sequencer **304,955 behind** the ladder,
puts fitted Q **124,644 ahead** of the shipped policy, and inverts the contact-enabled and
no-contact configurations. At full scale the sequencer leads the ladder and Q loses 4/4. A
reduced-scale run is not a small version of this experiment — it is a different, weaker
model, and its ordering carries no information.

That script used to print fixed captions — *"policy ordering holds; magnitudes do not"*,
*"the hand-built expected value beat it 4/4"* — above tables where neither was true. Both now
read the run they are printed under. Same defect as the +100.0% incident, arriving as prose
instead of arithmetic; `scripts/demo.py` had the matching hole and now names the winner when
a baseline beats us.

*Withdrawn:* an earlier version reported a 1,200 / 3,000 / 6,000 training-log sweep. Its
contact-enabled row no longer reproduces — the 6,000-customer cell read +99,516 where that
configuration now measures +29,680 — and the sweep has no committed script.

**Not free of pricing error.** The policy under-prices a voice call by about 3× relative to
the world's compounding contact fatigue: the world's revocation hazard compounds at
`1.45 ** contacts` while the per-action label is flat in `prior_contacts`, so three voice
calls are priced at 0.0225 where the world charges 0.0710. And the expected value credits
every decision in an episode with the whole episode's recovery — the defect fitted-Q was
built to remove. Both are open and quantified.

---

## The compliance gate

The agent proposes; the gate disposes. `ComplianceGate.adjudicate` is the only way to mint
an `ApprovedAction` through any ordinary route: direct construction raises, and
`dataclasses.replace` — which copies every field through, and so once turned an approved
retry into an approved voice call — has no token to pass, because the token is init-only.
The alternative design, a `check_compliance()` call before acting, is correct only as long
as every future call site remembers one line, and that failure is silent.

**Two limits, both found by an outside evaluator.** An `ApprovedAction` *can* be minted by
`object.__new__` and `__setattr__`, by subclassing, by harvesting the module-private token, or
by unpickling. The threat model is an author who forgets a line next year, not sabotage, and
against that the design holds — but "cannot be minted outside `rebound.compliance`" was stated
absolutely and is not true absolutely.

And the guarantee is **narrower than the execution boundary**. `World.apply` takes a bare
`Action` and `evaluate_policy` takes a plain `Decision`, so every Claim B number is produced by
a path with no approval object in it. The only consumer of an `ApprovedAction` today is the
comms layer: a real structural guarantee about what a customer *receives*, and not the same as
one about what gets presented to a rail.

**Three verdicts, not two.** A gate that only says yes or no turns every timing rule into a
lost recovery — "denied" tells a sequencer to abandon a retry that is perfectly legal in four
hours. `DEFER` carries the moment it becomes permissible, and that moment is re-adjudicated
until it settles: constraints here are intervals, not deadlines, and clearing the 24-hour
notice requirement can drop you inside a closed NPCI window.

**Law and house policy are labelled separately.** Five regulatory rules (mandate alive,
pre-debit notice, AFA ceiling, execution windows, presentation cap) and four policy ones (quiet
hours, contact cap, spend budget, terminal stop). Labelling a self-imposed contact cap
"compliance" makes a product decision unarguable by misattributing it to a regulator.

**The regulatory constants are unverified, and the gate says so.** `unverified_rules()` names
every rule resting on secondary reporting rather than a primary source, and the NPCI window
conflict is recorded as disputed with a note on which direction the encoded reading errs.
Confirming these against the circulars is tracked as open work, not quietly assumed.

---

## Where we deliberately did not use an LLM

**Compliance.** A gate decision has to be deterministic, reproducible on replay, explainable
by citation to the rule that fired, and identical for two customers in identical
circumstances. A language model is none of those.

**Retry timing and action selection** are calibrated-probability problems, not language
problems. The output gets multiplied by a rupee amount to make a spend decision, so the
number has to *be* a probability — and an LLM's is uncalibrated, non-reproducible across
runs, and unauditable.

An LLM is used in **exactly one place**: writing the words of a customer message. An earlier
draft of this section also claimed a "merchant-facing root-cause narrative" — that feature
does not exist, and claiming a second use of a model inside the section arguing for restraint
was the most self-defeating sentence in the document.

## The comms layer: the model writes, it does not decide

By the time a drafter runs, everything carrying money or legal exposure is already fixed:
**whether** to contact by the sequencer's expected value, **when** by the gate's deferral
arithmetic, the amount and rail by the record, and — the one that matters most — **what the
customer is told to do**, derived from the failure's disposition by `comms.ask_for`. What is
left is a language problem: say this, to this person, in Hinglish, in one GSM-7 segment.

The instruction is the boundary worth defending. A model that picks its own ask will eventually
tell a customer whose balance was short to replace a card that is fine. They will replace it,
the next debit will fail identically, and the contact is spent. So `Ask` is a closed set of
seven derived by a function with no judgement in it.

**Generate → verify → repair once → fall back.** Every draft is checked against the brief by 13
deterministic checks. Nothing that fails is sent. A failure is fed back for one repair attempt;
a second failure falls back to a template proven by test to pass every check for every
combination of instruction, channel and language. That fallback is what makes the model safe to
use at all — without it the failure mode is either sending something unverified or sending
nothing.

The verifier calls no model. A model grading a model shares its failure modes and cannot be
cross-examined later by whoever has to decide whether a message should have gone out.

### Segment arithmetic, which turned out to be a language decision

One character outside GSM-7 re-encodes an entire SMS as UCS-2 and cuts the per-segment budget
from 153 characters to 67. Rendered from the shipped templates:

| Language | Encoding | SMS segments |
|---|---|---:|
| English | GSM-7 | 1 |
| Hinglish | GSM-7 | 1 |
| Hindi | UCS-2 | 2 |

The same reminder costs twice as much to send in Devanagari. That is a large part of why
Indian merchants send Hinglish — and Hinglish, whose spelling is not standardised, is exactly
what a template library is worst at and a model is good at. It is also why the templates write
`Rs.` and never `₹`: the rupee sign is not in GSM-7, so one of them halves the message.

Three languages ship: English, Hindi, Hinglish. Not eight. A language ships when a native
speaker has read its fallback templates, because the fallback is what goes out on the worst
day, and an unreviewed fallback is a guaranteed send of text nobody checked.

### What the verifier catches, and what it does not

`scripts/evaluate_comms.py` runs a corpus of bad drafts through the checks. **28 of 28 probes
are caught by the check written for them** — wrong amounts, invented reference numbers,
phishing hosts, leaked internal failure codes, credential solicitation, threats, unsigned
messages, wrong instructions, a pre-debit notice missing its disclosures, an undeliverable
message.

A count like that is only worth its weakest probe. An earlier version reported the link check
at 3/3 while `vahan-secure.ru/pay` was being cleared to send, because the one scheme-less probe
sat inside the detector's blind spot. The link probes now sit on the boundary: an unlisted
top-level domain, a bare IP address, a lookalike dot, and a `upi://` deep link to an attacker's
VPA.

And the corpus and the checks have the same author, so a high catch rate measures that author's
imagination. What makes it evidence is the other half of the report — **five probes the checks
provably do not catch**, printed as prominently as the wins:

- **A polite threat.** Coercion with no word from the lexicon.
- **A false causal claim.** "Your bank declined this payment" when the cause may have been our
  own missing pre-debit notice. The brief withholds the failure code on purpose, so no check
  here has anything to contradict it with.
- **A social-engineering setup.** "Our agent will call you shortly." Asks for no credential,
  so the lexicon has nothing to match.
- **Fluent but ungrammatical Hinglish.** A marker count cannot measure grammar.
- **Correct but cruel.** Every fact true; no check reads tone.

Three of the five are tone and intent, exactly where a deterministic check has nothing to
compare against. The checks are labelled by tier — *exact*, *bounded*, *lexical* — so a clean
run on the lexical ones is not read as a guarantee.

### Status of the live path

The verifier, the templates and the desk are exercised by 340 tests. Ten of those build
nothing by hand: they sample a population, take the debits that actually failed, adjudicate
through a real compliance gate, and compose from the resulting approval. That is how the two
worst defects in this layer were found.

**The live API path has not been run.** There is no API key in this environment, so every
number above is measured against the verifier, the templates and a stub client. The request
shape *is* verified against the installed SDK — a signature test asserts every argument we
send exists on the real client, which is how a `temperature` parameter that does not exist on
`anthropic` 1.0.0 was caught before it could raise on a first live call. What remains
unmeasured is behavioural: what fraction of drafts pass first time, and which checks a real
model trips.

```bash
ANTHROPIC_API_KEY=... uv run python scripts/evaluate_comms.py --model --verbose
```

30 briefs — three languages × ten ask/channel/rail combinations — at up to two calls each.

## Adversarial review

The simulator, harness and policy interface were red-teamed black-box before the model was
built. Thirteen findings, five critical, all fixed, each with a regression test that
reproduces the original attack. See [docs/SECURITY_REVIEW.md](docs/SECURITY_REVIEW.md).

---

## Running it

```bash
uv sync
uv run python scripts/demo.py
```

About a minute, no API key, no network. It walks one batch of failed debits all the way
through: the taxonomy deciding which actions are legal, the gate answering allow / defer / deny
with its reasons, three messages composed in English, Hinglish and Hindi with their SMS segment
costs, six hostile drafts being refused, the full policy comparison, and the exception list
with a reason per episode. Everything in it is the shipped code, running against the same
harness and the same gate as the tables above.

There was no such command until an outside evaluator pointed out that a reader could run the
test suite and three analysis scripts and never once see the system do its job. It found a bug
on its first run: the templates still signed with the full legal entity after the sender check
had moved off it, so a realistic merchant name pushed a Hindi SMS past its UCS-2 budget and the
fallback failed — meaning nothing would have been sent at all.

### The dashboard

```bash
uv run python scripts/dashboard.py --seed 20260904
```

Three views. **Live run** replays a recorded rollout action by action at 1× / 4× / 16×, with
recovered, spent, revocations and net accumulating from the audit trail — a replay, not live
computation, and labelled as one. **Requests** lists customer requests; opening one shows the
four stages (classify → price → adjudicate → act), the rules that fired, and the expected
value decomposed line by line rather than asserted. **Batch** carries the policy table,
outcome breakdown and exception reasons.

Nothing in that page recomputes a decision — a second implementation in JavaScript would be a
second thing to keep in agreement.

**Pass the seed.** The builder's default does not contain every case worth showing;
`--seed 20260904` does, and the build prints a coverage report naming which cases the chosen
seed actually contains. The strongest one — a profitable action overruled by a regulatory rule
— is data-dependent, and a run that lacks it will not say so unless you read that report.

### Everything else

```bash
uv run pytest                              # 846 collected: 843 pass, 3 skipped, ~45s
./scripts/mutation_check.sh                # break the source 16 ways, confirm the suite notices
uv run python scripts/evaluate_comms.py    # verifier red team, 28/28 caught + 5 known holes
uv run python scripts/holdout.py           # Claim B, five held-out seeds (slow)
uv run python scripts/train_model.py       # Claim A, both splits
uv run python scripts/evaluate_sequencer.py --quick   # reduced scale; ordering carries no information
```

Current state and open work: [PROGRESS.md](PROGRESS.md).
