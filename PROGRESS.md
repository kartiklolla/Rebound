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
| 07a | **Independent audit of the model layer + fixes** | ✅ done — 6 findings fixed, 387 tests |
| 08 | Compliance gate (non-bypassable) | ✅ done — 9 rules, reviewed, 434 tests |
| 09 | Sequencer / agent policy | ✅ done — held-out seeds, see Claim B |
| 09a | Fitted Q-iteration | ✅ built, measured, **not shipped** — diagnosed |
| 10 | LLM comms layer (Hinglish/multilingual) | ✅ done — 13 checks, 28/28 red team, 5 holes documented, 333 tests, 27/27 mutations, live path unrun |
| 11 | Batch runner + demo dashboard | ⬜ not started |
| 12 | README, metrics writeup, provenance table | 🟡 in progress — README and provenance doc written, final pass pending |

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
had hardcoded, and the hardcoded choice had been costing PR-AUC for worse calibration.

**Then the audit found the measurement was taken under the wrong regime.** The inner
split was always temporal, whatever the outer split was holding out. On the customer
split that meant 3,533 of 3,712 calibration-slice customers were also in the base model's
fit slice, so the calibrator was selected on people the booster had memorised and applied
to strangers. It picked isotonic on a selection ECE of 0.0074 against 0.0153 for none -
and on test isotonic was worse on *every* metric, ECE included (0.0102 vs 0.0082), while
costing 0.0131 PR-AUC. Confident, reproducible, drawn from the wrong distribution.

The inner split now mirrors the outer one: episode-grouped and temporal for the time
split, customer-disjoint and unordered for the customer split. Post-fix, held-out ECE is
sigmoid 0.0102 / isotonic 0.0128 / none 0.0191 on time and sigmoid 0.0179 / none 0.0184 /
isotonic 0.0186 on customer. Sigmoid now wins on both for the action head and isotonic on
both for the timing head - so the choice varies by *head*, not by split, and calibration
no longer costs any discrimination (PR-AUC identical calibrated and raw on both splits).

The lesson is not "measure instead of assume." It is that a measurement taken under the
wrong regime is worth less than an assumption you knew was an assumption.

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

Current split is **1 anchored / 5 derived / 21 assumed** of 27 — 78% assumptions. Published
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

### D20 - The model layer was audited by an agent that had not built it (2026-08-21)

After the two heads were finished, the whole model layer went to an independent agent
with the source, no access to the reasoning behind it, and an instruction to find things
that were wrong rather than to confirm things that were right.

It found six defects. Four were in code, and all four had been making the model quietly
worse while 385 tests reported green:

1. **The calibrator was selected under the wrong generalisation regime.** The single most
   valuable finding. Detailed in the calibration section above.
2. **`decision_index` and `prior_attempts` were byte-identical** in all 162,743 rows.
   `decision_index` was even *commented* as an identifier at its construction site - it
   had simply never been added to `IDENTIFIER_COLUMNS`, and the allowlist happily passed
   it through. A comment is not a control.
3. **Two tests could not fail.** One asserted that a sorted column was sorted. The other
   re-derived a slice with the same arithmetic it was meant to be checking. Both were
   named for properties that were, at the time, actually violated in the data.
4. **The two-head comparison was not like-for-like.**

Plus two reporting defects: every metric table in this file was a full run stale, and the
claim that excluding `customer_id` prevents customer memorisation was wrong - the tuple
`(bank, billing_day, amount, ceiling, rail)` is unique across all 5,974 customers.

Three things worth keeping from this:

**The tests were the problem, not the safety net.** Two of them were named for exactly the
defects that were present and passed anyway. A test that re-derives the value it is
checking, using the code under test, always passes. Writing the assertion against
*recorded state* - what the model actually did - rather than against a recomputation is
what made them bite.

**Fixing all four moved the headline numbers up** (time PR-AUC 0.6074 to 0.6213, customer
0.6325 to 0.6578, customer ECE 0.0102 to 0.0035). Defects that make the numbers worse are
the hard ones to find, because nothing looks wrong.

**I wrote a docstring citing a test I had not written yet.** While fixing finding 2 I
described `test_no_two_features_are_identical` as the guard preventing recurrence, and it
did not exist. Caught on re-read, written, and it passes - but it is the same failure mode
as the two tautological tests: describing a check rather than having one.

### D21 - The gate cannot be forgotten, and it can say "not yet" (2026-08-21)

The obvious build is `check_compliance(action) -> bool` called before acting.
That is correct exactly as long as every future call site remembers one line. The
failure is silent, it is a single omission, and the tests still pass because the
tests call the checker directly - the same shape as `decision_index` being
*documented* as an identifier and never added to the identifier set.

So the type carries it. An executor accepts only an `ApprovedAction`, which
cannot be minted outside `rebound.compliance`. Copying one is allowed (a copy of
a permission is the same permission); *altering* one is not, which is why the
token is init-only. `dataclasses.replace` was a live forgery route until the
reviewer found it: it copied the valid token through with every other field, so
an approved retry could be edited into an approved voice call using nothing but
the public API.

**Three verdicts, not two.** A gate that only says yes or no turns every timing
rule into a lost recovery: "denied" tells the sequencer to abandon a retry that
is legal in four hours. `DEFER` carries the moment it becomes permissible.

**Law and house policy are labelled separately.** Five REGULATORY rules, four
POLICY. The point is honesty in both directions - a merchant must see which
constraints they may tune, and this system must not claim regulatory cover for
its own preferences. Calling a self-imposed contact cap "compliance" makes a
product decision unarguable by misattributing it to a regulator.

**Regulatory constants moved out of `sim/params.py` into `rebound.regulation`.**
A compliance gate importing from the simulator has its dependency backwards. More
importantly the two kinds of error differ: a wrong simulator parameter shifts a
measured number, a wrong regulatory constant makes the system non-compliant
*while reporting that it complied* - which no amount of testing against our own
simulator can catch, because the simulator would be wrong in the same direction.
All five constants are unverified; `unverified_rules()` reports which rules rest
on them rather than letting an audit trail imply diligence that never happened.

### D22 - What the reviewer found in the gate (2026-08-21)

Reviewed by an independent agent while it was being written rather than after.
Ten findings. The three that mattered:

**The deferral time was a time the gate would itself refuse.** `max()` over every
deferral is only correct if each constraint means "permitted from T onward". The
execution window is not that shape - it is a set of intervals, and leaving one
does not put you inside the next. A UPI retry blocked at 11:00 by both an
immature notice (matures 18:00) and the window (opens 13:00) was told 18:00,
which lands in the 17:00-21:30 dead zone. Since `DEFER`'s entire justification is
that the time it carries is actionable, this broke the module's central claim. It
now re-adjudicates until the candidate stops moving; that case takes three hops.

**The retry cap counted nudges as debit presentations.** `Ledger.attempts`
increments on every action, so three SMS nudges exhausted the NPCI *presentation*
cap - and the denial was stamped REGULATORY, telling a merchant a regulator
forbade a debit that had never been presented. Mislabelling a house effect as law
is precisely what the Basis split exists to prevent, so this was worse than the
lost retries. `Request` now carries `debit_attempts` separately.

**`RETRY_ALT_RAIL` was judged on the wrong rail.** It presents on a *different*
rail by definition, and UPI is first in `ALT_RAILS` for both other rails - so the
most common hop, eNACH to UPI, escaped NPCI's windows entirely, while UPI to card
was deferred for a constraint cards do not have. Unsafe one way, lost recovery the
other.

Also fixed: probes from `permitted_actions` were writing ~11 counterfactual rows
per step into the audit trail a merchant reads to find out why nothing happened;
DENY decisions published a reschedule time; `explain()` named only the first of
two tied deferrals and never mentioned the rule that actually governed;
`unverified_rules()` returned constant names while claiming to return rule ids, so
a caller joining it against `binding.rule_id` silently reported zero unverified
rules; and `ContactCap`/`SpendBudget` could deny the mandatory pre-debit
notification, which `QuietHours` had been careful to exempt - three sibling rules
reasoning inconsistently about the same case, now one named set.

Two findings were downgraded rather than fixed. Timezone-aware datetimes crashed
the window arithmetic; unreachable today since the simulator is naive throughout,
but `Request` now rejects aware times at the boundary rather than half-handling
them. And `TerminalStop` being POLICY was called a mislabelling - it is not, but
the `Basis` docstring overclaimed by saying policy rules are ones a merchant "may
legitimately configure differently". The label says who owns a rule, not how
harmless it is to drop.

### D23 - The sequencer works, and loses (2026-08-21)

Built, reviewed, measured. It is not a win, and the number is reported as it is.

| Policy | Recovery | Revocation | Contacts/ep | Net Rs/1000 |
|---|---:|---:|---:|---:|
| fixed_ladder | 0.5155 | 0.0465 | 0.00 | **+12,287** |
| immediate_retry | 0.4516 | 0.0484 | 0.00 | -80,274 |
| **rebound_sequencer** | **0.6279** | 0.0581 | 1.09 | -126,859 |
| disposition_rules | 0.4632 | 0.0601 | 0.80 | -174,909 |
| no_recovery | 0.0000 | 0.0950 | 0.00 | -1,085,275 |
| aggressive_contact | 0.3585 | 0.1415 | 3.56 | -1,176,290 |

**It recovers more than anything else and still destroys value.** It over-contacts,
and the mechanism is identified.

**The revocation head's marginal estimate has the wrong sign.** Across every
action, `p_revoke(action) - p_revoke(stop)` comes out negative: the model says
contacting a customer makes them *less* likely to revoke. In the training log
`stop` carries the highest observed revocation rate (0.1038) and
`retry_same_rail` the lowest (0.0582) - because the behavioural policy stops on
episodes that are already lost, and a stopped episode never gets the chance to
recover. "Stop causes revocation" is selection. The sequencer read it causally.

This is **the label bug from D19 in a new costume**: an episode-level label
attributed to a single decision. Every row in an episode carries the same
`episode_revoked`, so what the head learned is "episodes where action *a*
appeared revoke at rate X", dominated by which episodes get which actions rather
than by what the action does. Identifying the causal quantity needs a per-action
revocation label or interventional data. Neither can be produced by tuning.

The floor at zero in `Candidate.marginal_revocation` is a guard against a
known-bad estimate, not a fix for it - unclamped, 30.7% of candidates were
*credited* for reducing revocation, and at a 12-cycle horizon a delta of -0.0659
pays +0.79x the amount, enough to make a voice call profitable on any episode.

Lowering the contact cap to 1 would turn the number green. That is fitting the
policy to the scoreboard while the mechanism stays broken, so it was not done.

### D24 - What the reviewer found in the sequencer (2026-08-21)

Ten findings. The first is the one that matters.

**The evaluation script reported +100% for a policy that crashed.** The
sequencer exceeded the harness's default 120s budget after 2,001 of 6,898
episodes. `evaluate_all` isolates a failed policy and substitutes an all-zero
report - so zero revocation and zero contacts sorted it to the *top* of a table
ordered by net value, and a lift computed as `(ours - base)/abs(base)` against a
negative baseline turned "did nothing at all" into a +100.0% gain. The failure
line printed one row above the table contradicting it.

Three ordinary decisions compounded into a fabricated result: sort by net value,
divide by a negative baseline, and isolate crashes so a run completes. Each is
defensible alone. The script now refuses to tabulate an incomplete run at all,
and prints a difference rather than a ratio.

Also fixed:

- **The do-nothing baseline was priced at the wrong time.** One STOP row at
  `now`, while 84% of candidates were scheduled elsewhere - median 48 hours
  away - and seven features derive from the timestamp. The maximised quantity
  was the action effect confounded with a two-to-seven-day shift, which is
  precisely what the timing head is separately choosing. Now one baseline per
  distinct candidate time.
- **Non-determinism across processes.** `legal_actions` returns a frozenset and
  StrEnum hashing is salted by `PYTHONHASHSEED`; that order reached `max` over
  expected values, and since `RETRY_SAME_RAIL` and `RETRY_ALT_RAIL` share cost,
  value and revocation cost, exact ties broke by list position. Net value moved
  2.2% between interpreters running identical code on an identical batch. A
  same-process rerun test cannot see this.
- **The trail double-wrote and named actions never taken** - `send_collect_link`
  620 times where 9 reached the world, in the record a merchant reads to find
  out what happened. One row per decision now, written after the branches
  resolve; trail rows and gate-audit rows now match exactly (1,372 = 1,372).
- **STOP never reached the compliance audit trail** - 497 of 3,129 decisions
  absent from the record whose stated purpose is answering "why did nothing
  happen for this customer".
- A third sklearn call per decision whose output was discarded, and a dead
  fallback branch that would have returned a refused time if reached.

Two carried forward rather than fixed: the harness bumps execution to `now + 1
minute` when a decision is scheduled at or before `now`, so an action approved
exactly on a window boundary executes one minute outside it; and
`fit_for_serving` claims dropping the unservable columns "costs some accuracy
which is reported", while nothing measures that gap.

The review also confirmed **zero train/serve skew** across 665 real decision
points, every field value-compared against the training-time builder - and the
test that was supposed to guarantee that was a subset check on column *names*
that would have passed if the serving path dropped half its features. It now
compares values.

### D25 - The EV double-counts across decisions, and fixing it properly did not help (2026-08-24)

`p_recover` is the downstream head - P(episode recovers | this action) - so it
already contains everything that happens *after* the action. In a three-decision
episode all three decisions are credited with the same recovery: the nudge is
credited for the retry that follows it, then the retry is credited again.

That is why correcting the recovery coefficient to its true 1.84x made the policy
**worse** (net -73,367 to -159,979, contacts 0.74 to 1.00). It amplified a term
already counted several times. No coefficient fixes it.

The closed-form derivation is correct arithmetic for a decision that **ends the
episode**. Episodes do not end. There is no term in it for "and then I decide
again", and that missing term is the whole problem.

### D26 - Fitted Q-iteration: the right method, the wrong data (2026-08-24)

Built it properly: rewards from the ledger (reconciling exactly with
`episode_net_paise`), backward induction, double-Q on episode-atomic partitions,
pessimistic combination.

**It lost.** Across four full-scale seeds the hand-built expected value beat it
4/4 - mean gap over the ladder +164,807 against +100,876 - and Q went *negative*
on one seed, losing to a three-line retry ladder.

| seed | rebound_q | rebound_sequencer |
|---|---:|---:|
| 12345 | +283,783 | **+297,047** |
| 24680 | **-42,116** | +117,474 |
| 31415 | +113,240 | **+168,457** |
| 55555 | +48,598 | **+76,248** |

The diagnosis is specific and is the useful part. `EXPLORATION_WEIGHTS`
randomised *actions*; nothing randomised *timing*. The behavioural log has a hard
floor at `days_since_failure = 1.0` - **zero of 88,178 rows below it** - while
the candidate grid starts at zero delay. So **68% of the Q policy's decisions
fall outside the training support**, in a region where the booster's leftmost bin
makes the feature constant. Without a timing head it schedules 82.7% of decisions
at zero delay and becomes `immediate_retry` with extra machinery.

Two more the data cannot support: **53% of last rows are the generator's step
budget running out**, not an ending, each carrying a full twelve-cycle churn
charge with no continuation - so induction propagates "episodes end badly"
backwards. And the training log allows five decisions where the harness allows
eight, so 3.1% of rollout decisions sit at `prior_actions` values with zero
training rows.

Fitted-Q is not a detour. It is the correct method applied to a log built to
answer a different question, and the fix is a generator that randomises timing -
not a better regressor.

### D27 - Three failures of my own worth recording (2026-08-24)

**I reported a headline from a batch too small to resolve it.** +67,914 on ~1,700
episodes, where one extra revoked episode moves the figure by ~45,000. The gap I
quoted was inside the noise, and at full scale it inverted.

**I claimed a circular result as independent corroboration, twice.** That fitted
Q learned `Q(stop) = -992` against the closed form's -1,002 "from a completely
independent route". On every nonzero stop row the logged reward *is* exactly
`-1.0 x amount x LTV_HORIZON_CYCLES`, so `Q(s,STOP)` is definitionally that same
closed form with the rate estimated rather than supplied. One computation
presented as two witnesses, offered as validation of precisely the thing it could
not validate.

**My tests did not test the algorithm.** The entire 480-test suite passed with
backward induction disabled. Six of fifteen new tests could not fail at all - one
passed with the timing head replaced by a constant, another with passive churn
attached to the *first* decision instead of the last. Now verified by mutation:
seven mutations, each caught.

One test deliberately does not assert the direction I expected.
`test_backward_induction_moves_the_values` was first written expecting induction
to *raise* the value of a nudge, since a nudge collects nothing and only a
continuation can justify it. Measured, induction lowers it, because of the
truncation charge above. Asserting the expected direction would have meant
deleting a true finding to keep a comfortable test.

### D28 - Ten looks at one metric, and the held-out protocol (2026-08-24)

Roughly ten policy variants have now been compared on the same evaluation:
absolute EV, marginal-vs-stop, baseline-per-time, clamped, closed-form,
plus-externality, naive Q, double-Q, pessimistic Q, timing-hybrid Q. Each time
the better-scoring one was kept.

The seed standard deviation on that metric is ~96,000. Ten looks is ten chances
to get lucky, and nothing in the project protected against it - every other
selection decision here is split-protected, and policy selection was not.

So Claim B is re-measured on five seeds that selection never touched
(`runs/holdout.py`). Whatever they say is the number that gets published.

### D29 - The model writes, it does not decide (2026-08-24)

The comms layer is the one place in this system where a language model is
clearly the right tool, and the one place where handing it the obvious amount
of authority would be a mistake.

By the time a drafter is called, everything carrying money or legal exposure is
already fixed by code that was measured or by a regulation that was cited:
*whether* to contact by the sequencer's expected value, *when* by the gate's
deferral arithmetic, the amount and rail by the record, and - the one that
matters most - *what the customer is told to do*, derived from the failure's
disposition in `comms.ask_for`. What is left is a language problem: say this,
to this person, in Hinglish, inside one GSM-7 segment.

(That line said "inside 134 characters" until an outside evaluator pointed out that 134 is *Hindi's* two-segment UCS-2 budget. Hinglish is GSM-7 and gets 306; the shipped Hinglish templates already run to 140. The rhetorical flourish had quietly attributed one language's constraint to another, in the section that is otherwise exact.)

The instruction is the boundary worth defending. A model that picks its own ask
will eventually tell a customer whose balance was short to replace a card that
is fine. They will replace it, the next debit will fail identically, and the
contact is gone. So `Ask` is a closed set of seven, derived by a function with
no judgement in it, and `verify.AskIsHonoured` checks that the message carries
its own instruction and no other.

Three places a model is deliberately **not** used:

- **Classification.** The taxonomy is a lookup on rail return codes. A model
  would be slower, non-deterministic and worse at it.
- **Verification.** A model grading a model shares its failure modes and cannot
  be cross-examined later by anyone deciding whether a message should have gone
  out. Every check is an exact comparison against the brief.
- **The fallback.** Templates, proven by test to pass every check for every
  combination of instruction, channel and language. This is what makes the
  model safe to use at all: without a fallback the failure mode is either
  sending something unverified or sending nothing.

**A measured result that fell out of the encoding rules.** One character outside
GSM-7 re-encodes an entire SMS as UCS-2 and cuts the per-segment budget from 153
characters to 67. Rendered from the templates, every English and Hinglish
reminder fits **one** segment; every Hindi one needs **two**. The same message
costs twice as much to send in Devanagari - a large part of why Indian merchants
send Hinglish, and why Hinglish is the register that justifies a model rather
than a translation table.

Three languages ship: English, Hindi, Hinglish. Not eight. A language ships when
a native speaker has read its fallback templates, because the fallback is what
goes out on the worst day, and an unreviewed fallback is a guaranteed send of
text nobody checked. Everything else in the layer is already language-agnostic;
`REVIEWED_LANGUAGES` is the gate.

### D30 - The verifier's holes are part of the deliverable (2026-08-24)

`scripts/evaluate_comms.py` runs a corpus of bad drafts through the checks. It
catches 22 of 22 probes with the check written for each.

That number alone would be worthless. The corpus and the checks have the same
author, so a high catch rate measures that author's imagination. What makes it
evidence is the other half of the report: five probes marked `MISSED`, each a
real harm these checks provably do not catch, printed as prominently as the
wins.

- **A polite threat.** Coercion with no word from the lexicon.
- **A false causal claim.** "Your bank declined this payment" when the cause may
  have been our own missing pre-debit notice. The brief withholds the failure
  code on purpose, so no check here has anything to contradict it with.
- **A social-engineering setup.** "Our agent will call you shortly." Asks for no
  credential, so the lexicon has nothing to match.
- **Fluent but ungrammatical Hinglish.** A marker count cannot measure grammar.
- **Correct but cruel.** Every fact true; no check reads tone.

Three of the five are tone and intent, exactly where a deterministic check has
nothing to compare against. That is the honest limit of this layer. The checks
are labelled by tier in the module docstring - *exact*, *bounded*, *lexical* -
so nobody reads a clean run on the lexical ones as a guarantee.

### D31 - What the reviewer found in the comms layer (2026-08-24)

Nine findings on the first cut. The first is the one that matters.

**A scheme-less URL passed every check.** `_URL` required `https?://` or `www.`,
so `evil.example.com/pay` matched nothing - including on voice scripts, where
links are supposed to be barred outright. Indian transactional SMS routinely
carries a bare host to save characters, so it is the form a model imitating the
register actually writes. A verifier that only sees the well-formed half of a
threat is not a verifier. Now a TLD allow-list rather than `\w+\.\w+`, which
would have matched "Rs.1,299" and every sentence break.

**The amount check reported a pre-debit notice's own mandate reference as a
wrong amount.** `_AMOUNT_SUFFIXED` accepted a number followed by "Rs." with no
left boundary, so `mandate UMRN0012345678 Rs.1,299` parsed `0012345678` as a sum
of money. It fired on the one message legally required to carry that reference.
The suffix form now accepts only spelled-out units; nobody writes "1299 Rs" in
this register anyway.

**The Hinglish check rejected half of natural Hinglish.** The marker list held
the words a template author reaches for, not the postpositions and verb tails
that actually make a sentence Hinglish - "Rs.1,299 ka payment fail ho gaya,
balance daal dijiye" scored zero. Hinglish is the register named as the reason a
model earns its place, so a false-reject rate there does not merely annoy: it
inflates the fallback headline in the flattering direction.

Also fixed: two fact tables that had to agree and did not, so quoting the true
date a debit failed was reported as a fabrication (now one table, derived);
identifier comparison that tested string formatting rather than truth, so the
real support number written as `1800-267-0001` was an invention; three of seven
instructions with no contradiction entry, so "update your card, and keep
sufficient balance" passed; and an emoji range list standing in for the property
it actually cared about, which is GSM-7 encodability - it missed `!! (tm) (c)`
and their neighbours, and a curly apostrophe is not an emoji and costs exactly
as much.

**Then mutation testing found what the review did not.** Eighteen mutations, seventeen
caught on the first pass. The survivor: raising `_IDENTIFIER_DIGITS` from 5 to
50 left all 202 tests green, because every case exercised only one of the
check's two branches. They are not redundant - the token branch tests substring containment,
so it accepts any *fragment* of a true fact, while the digit branch tests exact
membership and rejects one. `180026` and `2670001` are pieces of the real support
number, and a number that is nearly right is worse than one invented: it looks
right enough to dial. (That reasoning was correct at the time and has since
been overtaken — see D32, where making the token branch exact subsumed the
digit branch entirely and it was deleted.)

**Then two more, found by refusing to build any fixture by hand.**
`tests/test_comms_integration.py` samples a population, takes the debits that
actually failed, adjudicates through a real gate, and composes from the
resulting approval and the episode's own view. Both defects it found were
invisible to every hand-built test.

*The approval was forgeable.* `MessageBrief.build` structurally typed both its
arguments. That is right for the episode record — it comes from whatever system
holds it — and wrong for the approval, whose entire purpose is to be evidence
the gate was consulted. Any object with three attributes produced a brief, so
"there is no path to a message the gate did not permit" rested on callers
choosing not to. `build` now requires a real `ApprovedAction`. Notably my own
test asserted the wrong exception type, which is how the hole stayed hidden
inside a test written to find exactly this.

*Compliance and structural legality are different questions, and only one was
asked.* The gate has no rule about dispositions, so it approves
`REQUEST_REMANDATE` on a `RETRY_TIMING` episode — an action the taxonomy calls
meaningless there. Composed, that becomes "set up your autopay mandate again"
sent to a customer whose balance was briefly short. Nothing downstream can
fault it: the message is internally consistent, quotes the right amount and
honours the instruction it was given. The instruction was wrong, and the
instruction is the one thing this layer exists to get right. The sequencer does
restrict candidates by disposition, so the shipped path never hit it — which is
the argument for checking in the brief rather than against it, since a
guarantee that rests on one caller remembering is not a guarantee.

**And a bug that hid behind a convenient fixture.** Two checks contradicted each
other and neither review nor mutation could see it, because every test used the
mandate reference `UMRN2024HDFC0009911` — invented for readability. The
simulator issues `MND_0000001`, which matches the SCREAMING_SNAKE shape
`NoInternalCodes` looks for, while `PreDebitDisclosure` *requires* a notice to
quote its mandate reference. With the real format, every pre-debit notice was
blocked outright — fallback included, so `sent is None` and nothing went out at
all. `NoInternalCodes` now exempts strings that are the brief's own facts, and
`TestAgainstRealIdentifiers` renders every instruction and language against the
formats the simulator actually issues. A fixture chosen for legibility is a
fixture chosen to avoid the collisions real data has.

**And a bug neither found, that only the SDK could tell me.** `AnthropicDrafter`
sent `temperature=0.4` with a paragraph in its docstring justifying the value.
`messages.create` on this SDK has no such parameter. The first live call would
have raised `TypeError`, every message would have fallen back to a template, and
the report would have shown a model in use. The stub client accepts any keyword,
which is precisely why a stub cannot catch it; a test now checks the kwargs
against the real signature without making a network call. The same look turned up
native structured output, which replaced asking for JSON in the prompt and
parsing the reply hopefully.

**The live path has not been run.** There is no API key in this environment, so
everything above is measured against the verifier, the templates and a stub
client. `--model` is written and unexercised, and stays labelled that way until
it has been run.

### D32 - The link check had the wrong polarity (2026-08-25)

A second independent review found fifteen defects. One of them is the worst
thing found anywhere in this project.

**`vahan-secure.ru/pay` was cleared to send.** The link check detected URLs
against an allow-list of twenty-one top-level domains, so anything off the list
matched nothing, drew no finding, and came back from the desk as a message to
deliver. It reproduced on voice scripts too, where links are supposed to be
barred absolutely. Also invisible: bare IPv4, any two-label host that was not
lowercase, a host with a port, U+2024 in place of the dot, and
`upi://pay?pa=someone@psp` — a tappable payment intent, prefilled with the
right amount, straight to whoever wrote it, needing no domain name at all.

The red-team corpus reported that check at 3/3 the whole time. Its one
scheme-less probe sat comfortably inside the allow-list, so the number measured
nothing. **A probe that does not press on a boundary is a probe that measures
nothing**, and "22 of 22 caught" was worth exactly as much as its weakest
probe.

The polarity was wrong. An enumeration of bad shapes cannot work, because the
attacker picks the shape. Anything token-shaped that could resolve, be tapped
or be typed is now suspect by default, and three things clear it: the exact
link on the brief, an ordinary number, and a short list of abbreviations.

**That reverses a decision made one day earlier, for a reason worth recording.**
D31 describes "fixing" a false positive where `balance.In case of trouble`
matched as a host. Under the new rule it matches again, deliberately. The
original fix was optimising the reported fallback rate rather than the harm,
and the two are not comparable: a false positive costs one repair round trip,
with the offending token named in the feedback so the model fixes the space; a
false negative costs a customer their money. Fail closed. The filter was also
load-bearing in the wrong direction — it accepted any host that was not
lowercase and had fewer than three labels, which is most of the phishing
surface.

**Three more checks could not fire.** `"prosecut"` sat in a lexicon matched with
a trailing word boundary, so it could only match the string "prosecut", which
is not a word: "we will prosecute you for this outstanding amount" passed the
entire verifier. So did `blacklisted`, `defaulters` and `seized`. Stems are now
a separate table from whole words, because "court" as a prefix matches
"courtesy" and a polite message would be rejected for good manners.

The pre-debit disclosure accepted the bare day-of-month as a date — a one or
two character substring tested against the whole message — so the `5` in
`Rs.1,599` discharged a disclosure about 5 September and the debit behind that
notice went out unnotified. A day now counts only with a month beside it. The
check also claimed in its own docstring to require the amount and did not.

**`compose` was not total, and `fell_back` was drafter-controlled.** A raising
fallback propagated, which is exactly the condition the guarantee exists for —
`render_template` raises `KeyError` the moment a rail or language is added
without a table entry. And `fell_back` was `sent.produced_by == "template"`, a
string the drafter writes: running the desk with `TemplateDrafter` as the
*primary* drafter, which is the no-model baseline, reported 100% fallback, and
any drafter could launder its own output as the safe path by claiming the name.
Whether the fallback ran is a fact about control flow and is now the loop's to
state. Nothing type-checked the returned draft either, so a duck-typed object
whose `rendered()` returned different text on the second call was verified as
one message and handed back as another.

**Nine of thirty evaluation briefs were shapes the sequencer cannot emit** —
channel set independently of the action it arrives on — so part of the measured
fallback rate was measured on messages that do not exist.

**Then the mutation suite lied to me.** Four mutations reported as survivors
were patterns that no longer matched the rewritten source: `perl` changes
nothing and exits 0, so a stale mutation looks like a hole in the tests rather
than a hole in the runner. It now refuses to score a mutation that did not
change the file. A fifth was worse — `_COERCION_STEMS = () or (...)` evaluates
to the right operand, a semantic no-op that *did* change the file, so the diff
guard could not see it. And one genuine survivor was a test of mine that set
`body="x"` on the object it was probing with, which fails the length check on
its own, so the guard under test was never reached.

Twenty-seven mutations, twenty-seven caught. 818 tests, 333 in this layer. The
red team is 28 of 28 with five documented holes — and the link check now has
seven probes rather than three, sitting on the boundary rather than well inside
it.

### D33 - Two sweeps, and the harness was the problem (2026-08-25)

A white-box bug hunt over everything built before the comms layer, and a
black-box evaluation that was denied the source and given only what a judge
sees. Fifteen findings each. The two worst were in the same place, and it is the
place D1 says was built first precisely so it could not be bent to flatter the
model. It was not bent. It was porous, in a way nothing tested for.

**A policy could read the simulator's random stream out of its own episode id.**
The per-episode seed was `seed + index` and the id was `EV_{index:08d}`, so the
index of the stream a policy was about to face was handed to it. A demonstration
policy that reconstructed the uniforms and burned 2-paise emails to align its
single retry took recovery from 0.505 to 0.754 and net value up 7.4x. No
integrity check fired, because nothing was tampered with. It read.

**And common random numbers were not common.** The passive-churn draw settling
an episode came from the same stream the policy had been consuming, so the
settlement outcome depended on how many draws the policy had made. The hazard
itself was provably identical across policies (max difference 0.0) and outcomes
still differed on 15.75% of episodes. Across the baselines that was 177,000 to
319,000 rupees per thousand episodes of pure alignment noise against a policy
gap of 316,000 - and on a second batch it reversed the ordering of two
baselines outright.

Both have one cause: every draw came from one sequential stream, so *which*
uniform met an action depended on how many draws preceded it. Draws are now
addressed by purpose and by ordinal within their own kind, so the third debit
presentation always meets the same uniform whatever else happened. Emails cannot
move it, because emails are not debit presentations. The id is hashed so the
index is not free. What that does *not* claim is written into `EpisodeEntropy`:
reproducibility from a published seed and secrecy from someone holding that seed
cannot both hold, and this is a measurement harness, not a sandbox. The exploit
is dead because it is useless, not because it is hard.

**Re-seeding immediately exposed a hole.** `test_no_policy_recovers_money_from_
a_terminal_failure` calls itself "a structural guarantee, not a statistical
one". It was statistical: `_apply_collect_link` was the only path that recovered
money without consulting `mandate_alive`, and it passed only because the old
draw ordering never happened to pay a link on those 150 episodes. Re-addressed,
`aggressive_contact` collected Rs 394 from a closed account. Second time in this
project that a guarantee turned out to hold for one alignment of the dice.

**Four more that changed measured numbers.** The retry cap counted in-episode
presentations against a cycle-wide limit, permitting a fifth against a cap of
four on 19% of episodes while the denial text said "4 against a cap of 4" - two
rules in one file disagreeing about whether the debit that opened the episode
had happened, and the one citing a regulator was wrong. Alt-rail retries were
priced on the rail they left, all three wrong, UPI-to-card under-charged 6x on
the most common rail. The world never re-checked mandate expiry once an episode
opened, so the ladder collected Rs 10,060 the gate calls uncollectable and the
gate-bound sequencer was denied - policies compared under different rules. And
81% of `disposition_rules`' "we drove them away" exception lines blamed contact
for passive churn, reading a flag that was already recorded correctly and
ignored.

**The headline number had no reproduction path.** PROGRESS cited
`runs/holdout.py` for Claim B; the file lived outside the repository the whole
time, because run artefacts were moved there when the scratchpad kept being
wiped. The recorded output matches every published figure exactly, so the number
was real - and unverifiable by anyone else, which is the standard this document
uses to withdraw other claims. It is now `scripts/holdout.py`.

**And two guarantees were stated more absolutely than they hold.** An
`ApprovedAction` can be minted by `object.__new__`, by subclassing, by
harvesting the module-private token, and by unpickling, and altered in place the
same way; the design defeats every accidental route, which is the stated threat
model, but "cannot be minted outside `rebound.compliance`" is not true
absolutely. More importantly, *no executor in the package accepts one* -
`World.apply` takes a bare `Action`, so every Claim B number comes through a
path with no approval in it. The structural guarantee is real for the comms
layer and does not reach the execution boundary. Both now say so.

**The drafter could rewrite the brief it was verified against.** `MessageBrief`
is frozen against assignment but not against `object.__setattr__`, and the live
object was handed over - so a drafter could move `brief.link` to a host it
controlled, quote it, and have the verifier confirm the match. `vahan-secure.ru`
came back cleared with `fell_back=False`. The check was not evaded; the ground
truth was moved. Structurally the same defect the harness had, and fixed the
same way: the drafter gets a copy, the original is what the draft is checked
against.

Also fixed: `_split_holdout` silently abandoning group atomicity on a degenerate
boundary (verified not to fire on either shipped regime, which is why it now
raises); `slice_report` reporting the whole frame per slice on a non-unique
index, in the function whose docstring says every expensive mistake here was
invisible in the aggregate; `_null_report` scoring a crashed policy at zero and
sorting it to the top; `regulation.py`'s window helpers stripping tzinfo;
`SpendBudget` denying everything below four paise through integer truncation and
calling it an exhausted budget.

**The mutation runner lied to me twice.** Four "survivors" were stale patterns
that no longer matched the rewritten source - `perl` changes nothing and exits
0, so a dead mutation reads as a hole in the tests. It now refuses to score a
mutation that did not change the file. A fifth was worse: `_COERCION_STEMS = ()
or (...)` evaluates to the right operand, so the file changed and the behaviour
did not. And one genuine survivor was a test of mine that set `body="x"` on the
object it was probing with, which fails the length check on its own, so the
guard under test was never reached.

**Every number in this document is now pending re-measurement.** Claim A and
Claim B were both produced by code that has since changed in ways that move
them. They are being re-run; whatever comes back is what gets published,
including if it is worse.

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

162,743 decision points · 43,657 episodes · 5,974 customers · 30 selectable features.
Seed 20260821. Both splits verified clean before scoring. Every number below is from a
single run of `scripts/train_model.py`; nothing here is carried over by hand.

| Metric | Time split | Customer split |
|--------|-----------:|---------------:|
| n (test) | 51,322 | 49,116 |
| base rate | 0.1507 | 0.2161 |
| **PR-AUC** | **0.6213** | **0.6578** |
| ROC-AUC | 0.9186 | 0.8954 |
| Brier | 0.0771 | 0.1047 |
| Calibration slope | 0.948 | 0.998 |
| ECE | 0.0108 | 0.0035 |
| Precision @ 10% capacity | 0.6479 | 0.7199 |
| Lift @ 10% capacity | 4.30x | 3.33x |
| - global prior PR-AUC | 0.1507 | 0.2161 |
| - failure-code prior PR-AUC | 0.5483 | 0.5596 |

**Lift over the strong baseline is modest: +0.073 PR-AUC on time (+13.3% relative),
+0.098 on customer (+17.5%).** The failure-code prior - a pivot table any merchant
already has - gets most of the way. Reported this way round because the honest question
is not "is the model good" but "is the model worth the machinery over a group mean."

PR-AUC is not comparable across the two splits, since the base rates differ. Lift over
base rate is: **4.30x time, 3.33x customer**, and ROC agrees (0.9186 vs 0.8954). So there
is a real but small generalisation gap to unseen customers - in the feared direction,
but nowhere near the size that would indicate memorisation.

### Where the model is actually weak (per-disposition, time split)

| Disposition | n | base rate | PR-AUC | ROC-AUC |
|-------------|--:|----------:|-------:|--------:|
| mandate_repair | 31,172 | 0.0168 | 0.1576 | 0.8009 |
| retry_timing | 8,172 | 0.4634 | 0.6791 | 0.7380 |
| retry_transient | 7,778 | 0.4060 | 0.6060 | 0.7225 |
| customer_action | 2,037 | 0.0717 | 0.3182 | 0.7564 |
| terminal | 1,871 | 0.0572 | 0.4331 | 0.8634 |
| merchant_fix | 292 | 0.0308 | 0.1047 | 0.7793 |

**This table is the important one, and it qualifies the headline.** Aggregate ROC-AUC of
0.919 is largely the model separating hopeless dispositions from live ones - which
`failure_code` already encodes and the taxonomy already knew. *Within* slice, where the
decisions are actually hard, discrimination falls to 0.72-0.86. On the customer split
`merchant_fix` is 0.64, on 370 rows.

The README must not claim 0.914 without this caveat attached.

### What the model leans on (permutation importance, time split)

| Feature | Importance | std |
|---------|-----------:|----:|
| failure_code | 0.39800 | 0.00725 |
| action | 0.12265 | 0.00718 |
| within_upi_window | 0.04409 | 0.00249 |
| prior_actions | 0.03961 | 0.00682 |
| decision_day_of_month | 0.02563 | 0.00390 |
| rail | 0.01436 | 0.00302 |
| days_since_failure | 0.01293 | 0.00154 |
| cust_prior_contacts | 0.01234 | 0.00675 |

The previous version of this table listed `decision_index` at 0.039 as a distinct signal.
It was byte-identical to `prior_attempts`, which the table never mentioned; the two masked
each other under permutation. Merged into `prior_actions` it measures 0.0396 - so the
masking cost less than expected, and the damage was mostly to the honesty of the ranking.

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

The ablation explains why the first model could not see it. **These four cells and the
proxy table below were measured on the pre-audit model, in a one-off script that was not
committed.** They are kept because the finding they support is the reason the architecture
has two heads, but they are not comparable with the current numbers above and no
reproduction path exists for them. Regenerating them behind a committed script is an open
item, listed below.

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
| **Action head** PR-AUC | 0.6213 | 0.6578 |
| ROC-AUC | 0.9186 | 0.8954 |
| ECE | 0.0108 | 0.0035 |
| **Timing head** PR-AUC | 0.5901 | 0.6098 |
| ROC-AUC | 0.9424 | 0.9182 |
| ECE | 0.0074 | 0.0090 |
| (timing head n / base rate) | 27,708 / 0.1114 | 26,391 / 0.1605 |

**These two columns do not compare to each other, and the audit was right to say so.**
The earlier claim here - "the timing head out-discriminates the action head, 0.9427 vs
0.9137" - put a number computed on 27,708 collecting rows against `succeeded` next to one
computed on 51,322 rows against `episode_recovered`. Different rows, different label,
different base rate. It was not a comparison.

The comparison that means something holds the rows fixed and swaps only the label. The
script prints it now, so it cannot drift:

| Same rows (collecting), time split | Timing head | Action head |
|---|---:|---:|
| PR-AUC on `succeeded` | **0.5901** | 0.5024 |
| ROC on `succeeded` | **0.9424** | 0.9132 |
| PR-AUC on `episode_recovered` | 0.6238 | **0.6769** |
| ROC on `episode_recovered` | 0.8708 | **0.9201** |

Each head wins its own question and loses the other. The real gap on the timing question
is 0.0292 ROC, wider than the 0.0287 the bad comparison happened to show - so the
incorrect version was understating the case for the architecture, not inflating it.

### Two salary proxies built, measured, and deleted

Recorded because the second was a good idea that did not work, and deleting the evidence
would leave someone to rebuild it.

**`cust_prior_mean_failure_day`** — the average day-of-month the customer previously
*failed* on. Every failure for a mandate lands on its fixed billing day, so it correlated
**1.0000 with `billing_day`** and 0.0226 with the true salary day. A duplicate column
wearing an explanation. Removing it alone moved the action head from 0.6025 to 0.6074 on
the time split - and from 0.6363 to 0.6325 on the customer split, which the first draft of
this entry did not mention. Quoting only the split that improved is the kind of reporting
this document exists to prevent.

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

**An oracle handed the true latent reached 0.7205 against 0.6275, so most of the timing
signal remains unclaimed.** Stated as a limitation rather than papered over - and, per the
note above, measured before the audit fixes and not re-run since.

### Splits actually produced

| Split | Train rows | Test rows | Train customers | Test customers | Dropped |
|-------|-----------|-----------|-----------------|----------------|---------|
| time | 106,199 | 51,322 | 5,859 | 3,783 | 5,222 |
| customer | 113,627 | 49,116 | 4,167 | 1,807 | 0 |

The 5,222 dropped rows are episodes straddling the time cut. Dropped rather than assigned:
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

- Regenerate the ablation, oracle, salary-proxy and Claim B tables behind a committed
  script. They are currently hand-typed from one-off runs with no reproduction path, and
  the audit was right that a hand-typed number is a number that can drift. The ablation
  cells are also pre-audit and no longer comparable with the current model.
- Re-measure the oracle ceiling against the fixed model, so the "most of the timing signal
  is unclaimed" limitation is quantified against something current.

- Whether to anchor generator failure-code distributions to published NPCI aggregate
  decline statistics (strongly preferred for credibility — synthetic data anchored to
  real published aggregates is a different credibility class from invented numbers).
