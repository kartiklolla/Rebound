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
| 06 | Recovery-probability model + calibration | 🟡 in progress |
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

| Metric | Time split | Customer split |
|--------|-----------|----------------|
| PR-AUC | — | — |
| ROC-AUC | — | — |
| Brier score | — | — |
| Calibration slope | — | — |
| Baseline (failure-code prior only) PR-AUC | — | — |

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
