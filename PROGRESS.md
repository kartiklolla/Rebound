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
| 03 | World simulator + economics + provenance | ✅ done — 234 tests, rates calibrated to anchors |
| 04 | Historical dataset (exploration ladder) | ✅ done — 255 tests, ~53k rows / ~14k episodes |
| 05 | Metric harness + baselines + **both splits** | ✅ done — 311 tests |
| 05a | **Adversarial review + hardening** | ✅ done — 13 findings fixed, 352 tests |
| 06 | Recovery-probability model + calibration | ✅ done |
| 07 | Two-headed model (timing + action) | ✅ done — 385 tests |
| 07 | Compliance gate (non-bypassable) | ⬜ not started |
| 08 | Sequencer / agent policy | ⬜ not started |
| 09 | LLM comms layer (Hinglish/multilingual) | ⬜ not started |
| 10 | Batch runner + demo dashboard | ⬜ not started |
| 11 | README, metrics writeup, provenance table | ⬜ not started |

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

### D19 — Calibration is chosen by measurement, with "none" as a candidate (2026-08-21)
Calibration is routinely treated as free. It is not, and both failure modes showed up
here within an hour of each other.

Isotonic regression is non-parametric and overfits thin data. And a booster trained with
log-loss on a modest dataset is often *already* well calibrated, in which case any
correction makes it worse — measured on a small run: raw ECE 0.0232 and slope 0.970,
isotonic pushed it to 0.0317 *and* cost 2 points of PR-AUC.

So all three candidates — `none`, `sigmoid`, `isotonic` — are fitted on the first half of
the calibration slice, scored on the second, and the winner refitted on the whole slice.
Selecting on data the base model never saw is what makes it legitimate rather than a way
of picking the flattering number.

It earned its place immediately. At full scale it chose **sigmoid** over the isotonic I
had hardcoded (held-out ECE: sigmoid 0.0141, isotonic 0.0151, none 0.0207), which on
test gave ECE 0.0205 → 0.0117 with PR-AUC **unchanged** at 0.6025. The hardcoded isotonic
had been costing 0.0075 PR-AUC for worse calibration. On the customer split, isotonic won
instead — so a fixed choice would have been wrong on one split or the other.

`calibration_scores_` records every candidate. "We calibrated" and "we checked whether
calibrating helped" are different claims.

### D16 — The policy is untrusted, and the harness proves it (2026-08-21)
A black-box red team broke the harness five ways. All variants of one mistake: the
report was read off the object handed to the untrusted component, and the audit trail
that would have caught it was built and never compared.

Two defences now. **Structural** — policies get a frozen `EpisodeView`, never the live
episode, with no path back to mutable state. **Reconciliation** — every reported figure
is rebuilt from outcomes the harness itself observed, and `_Observed.verify` re-derives
episode state after every step, raising `IntegrityError` on disagreement.

The threat model is not sabotage. The policy is the component still being written, and
the realistic failure is an author believing a subtly wrong policy's numbers. Full
write-up in `docs/SECURITY_REVIEW.md`.

### D17 — `EpisodeView` also closes a latent leak the reviewer could not see (2026-08-21)
`Episode.customer` handed policies the whole `Customer` — `salary_day`,
`balance_health`, `engagement`, `churn_intent`, `preferred_channel`. The learned policy
could have read churn intent directly and posted meaningless numbers.

Invisible from black box, because the baselines happen not to use those fields. Found by
asking what a component *could* reach rather than what it currently does.

### D18 — Features are an allowlist, not a denylist (2026-08-21)
`FORBIDDEN_COLUMNS` guards latents, so `df.drop(columns=[label] + FORBIDDEN_COLUMNS)`
leaves `episode_net_paise` in — which predicts the label at accuracy **1.000** against a
base rate of 0.269, because it is the label restated.

`feature_columns()` now subtracts identifiers, outcomes, metadata and latents. A denylist
fails open; this fails closed. Backed by a threshold-sweep probe that fails on any
selectable feature above 97% accuracy, so it catches the next one whatever it is called.

### D12 — Customers churn on their own, not only when contacted (2026-08-21)
The first version of the world made merchant contact the *only* cause of revocation. So
`fixed_ladder`, which never contacts, showed a revocation rate of exactly 0.0000 and won
the comparison by construction. The optimal policy in that world is to never speak to
anyone.

Added a passive revocation hazard: an unresolved failed payment is itself a churn
trigger, independent of what the merchant does. This inverts the economics into
something true — *recovering a payment prevents churn* — and turns contact into a real
trade-off (marginal added risk against removing the passive risk) rather than a rigged
one.

Effect: floor revocation 8.78%, ladder 6.00%, aggressive contact 14.54%.

### D13 — Common random numbers across policies (2026-08-21)
Each episode gets a random stream seeded from its index, so every policy meets the same
customer under the same luck. With revocation events at a few percent, unpaired draws
mean a large share of any measured difference is just which policy drew better dice.

### D14 — Value is reported against the do-nothing floor (2026-08-21)
Raw net is negative for every policy, correctly: a failed cycle risks twelve cycles of
LTV against one cycle of upside. Reporting "−447,375" invites the reading "this policy
loses money" when it means "this policy loses much less than doing nothing". The floor
policy is the denominator that makes any other number interpretable.

### D15 — A strong hand-written baseline is included on purpose (2026-08-21)
`disposition_rules` reads the taxonomy, stops on terminal codes, notifies on merchant
defects, nudges before retrying when the customer is the blocker, and aims retries at
the start of the month. Roughly what a careful engineer builds in a week without ML.

Beating `fixed_ladder` proves little — it retries revoked mandates. If the learned
sequencer cannot beat careful rules, this problem did not need a model, and that is far
better to discover in week one than in front of a panel.

### D9 — Rows are labelled with the downstream episode outcome (2026-08-21)
The obvious label — "did *this* action recover the money" — is causally clean and badly
wrong. A nudge never collects; it unblocks a customer so a later retry collects. Under
the direct label every nudge, notification and mandate repair scores exactly 0.000, and
a model trained on it learns to never contact anyone.

Each decision point therefore carries the *return from that point*: whether the episode
ultimately recovered. Measured effect of the fix — nudges go from 0.000 immediate to
0.15 downstream overall, and 0.36–0.39 on insufficient-funds failures.

`succeeded` is retained alongside it. The contrast between immediate and downstream
effect is precisely what separates a collecting action from an enabling one.

**Caveat carried into the README:** this return is realised under the *behavioural*
policy. It answers "what happened when a merchant did this and then carried on behaving
like that merchant," not "what happens under an optimal continuation." Logged
propensities make the gap measurable rather than rhetorical.

### D10 — Mandate lifetime value is memoryless (2026-08-21)
`revocation_cost_paise` originally computed `horizon - cycles_elapsed`, implying a
mandate twelve cycles old had no value left and could be churned for free. Exactly
backwards: a customer who has paid reliably for a year is *more* likely to keep paying.
A policy optimising against that would aim its riskiest actions at the most loyal
customers on the book.

Now a flat expected remaining life, independent of tenure.

### D11 — Recovery is bounded by the billing cycle (2026-08-21)
Episodes may run at most 28 days. A merchant stops chasing last month's payment when
this month's comes due. Without the bound, episodes overlapped in wall-clock time,
which made per-customer history features arrive stale and described a merchant nobody
would recognise.

### D6 — Failure rates calibrated empirically, not in closed form (2026-08-21)
The salary curve decides *who* fails and *when*; it does not produce the right
*overall* rate on its own. Reconciling them in closed form is tractable but wrong in
practice — failure draws compound, and deterministic structural failures (expiry,
ceiling breaches, closed execution windows) stack on top. Measuring the realised rate
and adjusting absorbs every interaction without enumerating them.

Achieved vs target at n≈45k/19k/19k: UPI 0.5522 / 0.550, eNACH 0.3130 / 0.310, card
0.2830 / 0.280.

### D7 — Every world parameter carries a provenance tag (2026-08-21)
`ANCHORED` (published figure), `DERIVED` (computed from one), or `ASSUMED` (judgement,
no source). Machine-checked: tests fail if a parameter has no entry, or claims
ANCHORED/DERIVED without citing a source.

Current split is **1 anchored / 4 derived / 21 assumed** — 81% assumptions. Published
in `docs/DATA_PROVENANCE.md` as the opening paragraph rather than buried, because it is
the first number a panel will try to extract and volunteering it is worth more than
defending it.

The defence is scoping, not denial: the assumed parameters are mostly *shapes* (decay
rates, dispersions), the anchored ones are the *levels* (marginal failure rates), and
Claim B is a relative comparison run against the same world for every policy.

### D8 — Banks are anonymised (2026-08-21)
`BANK_01`–`BANK_12` rather than real bank names. The simulator assigns each an outage
propensity, and attaching invented reliability figures to a named real institution
would be publishing a claim about that organisation. The model learns exactly as much
from an opaque id.

### D5 — Unknown failure codes raise instead of defaulting (2026-08-21)
An unmapped code means the rail changed under us. Silently bucketing it as `TERMINAL`
would write off recoverable money with no signal that anything happened. `UnknownFailureCode`
is fatal by design.

---

## Evaluation splits

Both are built and both are reported. They answer different questions and a model can
pass one while failing the other.

**Time-based split** — train on an earlier window, test on a strictly later one. Tests
generalisation *forward in time*, which is the deployment condition: a model trained on
history and run on next month. Catches temporal leakage and drift. This is the primary
split for the headline numbers.

**Customer-based split** — disjoint customer sets, whole customers held out. Tests
generalisation *to people never seen before*, which is the cold-start condition and the
one that catches a model quietly memorising per-customer history features rather than
learning transferable structure. Given that `cust_prior_*` features exist specifically
to let the model infer latent traits, this is the split that keeps them honest.

A gap between the two is informative, not embarrassing: time-split strong and
customer-split weak means the model is leaning on customer memorisation and will fail on
new signups.

---

## Metrics

Filled in as they land. Empty cells are honest — they mean not yet measured.

### Claim A — recovery-probability model

162,743 decision points · 43,657 episodes · 5,974 customers · 32 selectable features.
Seed 20260821. Both splits verified clean before scoring.

| Metric | Time split | Customer split |
|--------|-----------:|---------------:|
| n (test) | 51,322 | 49,116 |
| base rate | 0.1507 | 0.2161 |
| **PR-AUC** | **0.6025** | **0.6363** |
| ROC-AUC | 0.9143 | 0.8950 |
| Brier | 0.0788 | 0.1055 |
| Calibration slope | 0.972 | 1.038 |
| ECE | 0.0117 | 0.0095 |
| Precision @ 10% capacity | 0.6374 | 0.7119 |
| Lift @ 10% capacity | 4.23× | 3.29× |
| — global prior PR-AUC | 0.1507 | 0.2161 |
| — failure-code prior PR-AUC | 0.5483 | 0.5596 |

**Lift over the strong baseline is modest: +0.054 PR-AUC on time (+9.9% relative),
+0.077 on customer (+13.7%).** The failure-code prior — a pivot table any merchant
already has — gets most of the way. Reported this way round because the honest question
is not "is the model good" but "is the model worth the machinery over a group mean."

PR-AUC is not comparable across the two splits, since the base rates differ. Lift over
base rate is: **4.23× time, 3.29× customer**, and ROC agrees (0.9143 vs 0.8950). So there
is a real but small generalisation gap to unseen customers — in the feared direction,
but nowhere near the size that would indicate memorisation.

### Where the model is actually weak (per-disposition, time split)

| Disposition | n | base rate | PR-AUC | ROC-AUC |
|-------------|--:|----------:|-------:|--------:|
| mandate_repair | 31,172 | 0.0168 | 0.1294 | 0.7694 |
| retry_timing | 8,172 | 0.4634 | 0.6553 | 0.7216 |
| retry_transient | 7,778 | 0.4060 | 0.5921 | 0.7113 |
| customer_action | 2,037 | 0.0717 | 0.2997 | 0.7721 |
| terminal | 1,871 | 0.0572 | 0.4115 | 0.8637 |
| merchant_fix | 292 | 0.0308 | 0.0657 | 0.6302 |

**This table is the important one, and it qualifies the headline.** Aggregate ROC-AUC of
0.914 is largely the model separating hopeless dispositions from live ones — which
`failure_code` already encodes and the taxonomy already knew. *Within* slice, where the
decisions are actually hard, discrimination is 0.71–0.77. `merchant_fix` at 0.63 is
barely better than a coin flip.

The README must not claim 0.914 without this caveat attached.

### What the model leans on (permutation importance, time split)

| Feature | Importance |
|---------|-----------:|
| failure_code | 0.380 |
| action | 0.114 |
| decision_index | 0.039 |
| within_upi_window | 0.038 |
| decision_day_of_month | 0.018 |
| cust_prior_contacts | 0.015 |

**Investigated and resolved — see the two-headed model below.** The salary signal is
real and large; it was being measured against the wrong label.

### Claim B — policy vs baselines

Batch of 4,703 failed debits, seed 20260821, common random numbers across policies.
"Value preserved" is measured against abandoning the debt, because every raw net figure
is negative — a failed cycle risks twelve cycles of lifetime value while offering one
cycle of revenue as the upside. Failures destroy value; the only question is how much a
policy saves.

| Policy | Recovery | Revocation | Value preserved ₹/1000 | True cost per ₹ | Contacts/ep |
|--------|----------|-----------|------------------------|-----------------|-------------|
| `fixed_ladder` (production standard) | 0.3264 | 0.0600 | **938,202** | 2.05 | 0.00 |
| `immediate_retry` | 0.2824 | 0.0655 | 803,482 | 2.58 | 0.00 |
| `disposition_rules` (best hand-written) | 0.3255 | 0.0763 | 777,732 | 2.43 | 1.18 |
| `no_recovery` (floor) | 0.0000 | 0.0878 | 0 | ∞ | 0.00 |
| `aggressive_contact` | 0.2222 | 0.1454 | **−346,088** | 6.87 | 3.70 |
| **Rebound (learned)** | — | — | — | — | — |

Three things this table already establishes, before any model exists:

1. **Recovery prevents churn.** Doing nothing revokes 8.78% of mandates; retrying pulls
   that to 6.00%. That is the actual argument for doing recovery at all.
2. **Chasing harder is worse than doing nothing.** `aggressive_contact` recovers *less*
   than the ladder (it burns the window on nudges) and churns 14.54%, ending below the
   floor. Recovery rate as a headline metric would rank it respectably.
3. **The best hand-written rules currently lose to the naive ladder.** `disposition_rules`
   stops on terminal codes and spends less per rupee, but its 1.18 contacts per episode
   drive revocation from 6.00% to 7.63%, and at twelve cycles of LTV per revocation that
   costs more than the savings.

Point 3 is the opening the model has to exploit, and it is a narrow one: contact
*selectively*, only where churn risk is low and nudge value is high. A rule cannot
express that. A calibrated per-customer estimate can. If the learned policy cannot beat
`fixed_ladder`, the honest conclusion is that this problem did not need a model.

### The two-headed model

The salary-cycle question from the single-model work turned out to be a question about
the *label*, not the feature set.

Immediate retry success against insufficient funds, by days since the customer's actual
payday: **0.6505** at days 2–5, falling to **0.2156** at days 24–31. A threefold spread,
present in the data the whole time.

The ablation explains why the first model could not see it:

| Target | no timing features | current | + oracle `days_since_salary` |
|--------|-------------------:|--------:|----------------------------:|
| `succeeded` (immediate) | 0.5870 | 0.6362 | **0.7332** |
| `episode_recovered` (downstream) | 0.7415 | 0.7520 | 0.7870 |

Against the downstream label, even a *perfect* view of the hidden latent adds almost
nothing. The downstream label aggregates the whole episode, so later actions wash out
the timing of any single decision — it dilutes exactly the signal a timing decision runs
on. Both labels are correct; neither is sufficient alone.

So there are two heads. **Timing** (`succeeded`, collecting actions only) prices *when*.
**Action** (`episode_recovered`, all rows) prices *which*.

| | Time split | Customer split |
|---|---:|---:|
| **Action head** PR-AUC | 0.6074 | 0.6325 |
| ROC-AUC | 0.9137 | 0.8941 |
| ECE | 0.0107 | 0.0102 |
| **Timing head** PR-AUC | 0.5869 | 0.6232 |
| ROC-AUC | **0.9427** | **0.9187** |
| ECE | 0.0061 | 0.0085 |
| (timing head n / base rate) | 27,708 / 0.1114 | 26,391 / 0.1605 |

The timing head out-discriminates the action head on ROC (0.9427 vs 0.9137) and is
better calibrated, which is the point: it is answering a question the action head was
never able to.

### Two salary proxies built, measured, and deleted

Recorded because the second was a good idea that did not work, and deleting the evidence
would leave someone to rebuild it.

**`cust_prior_mean_failure_day`** — the average day-of-month the customer previously
*failed* on. Every failure for a mandate lands on its fixed billing day, so it correlated
**1.0000 with `billing_day`** and 0.0226 with the true salary day. A duplicate column
wearing an explanation. Removing it alone lifted the action head from 0.6025 to 0.6074.

**`cust_prior_recovery_day_mean`** and friends — the average day the customer previously
*recovered* on. Successes only happen when the account has funds, so they should cluster
near payday, and they do: correlation with the latent rose to **0.1972**, genuinely
independent of `billing_day` (−0.04). It still made the model worse:

| Feature set | PR-AUC (immediate, collecting actions) |
|---|---:|
| neither block | **0.6275** |
| day-proxy only | 0.6237 |
| both | 0.6210 |
| recency only | 0.6164 |

A 0.20 correlation is weak enough that the tree spends splits on it and overfits, and it
is missing for ~37% of rows. Not noise — seed variance within a configuration is 0.0000.
Correct reasoning, measurably negative result, feature deleted.

**An oracle handed the true latent reaches 0.7205, so ~80% of the timing signal remains
unclaimed.** Stated as a limitation rather than papered over.

### Splits actually produced

| Split | Train rows | Test rows | Train customers | Test customers | Dropped |
|-------|-----------|-----------|-----------------|----------------|---------|
| time | 9,014 | 4,362 | 750 | 521 | 706 |
| customer | 10,177 | 3,905 | 562 | 228 | 0 |

The 706 dropped rows are episodes straddling the time cut. Dropped rather than assigned:
putting them in train leaks post-cut information backwards, putting them in test scores
the model on episodes it trained on.

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
