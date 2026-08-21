# Adversarial review

An independent black-box red team was run against parts 01–05 before any model
was built. The reviewer had no access to the source — runtime introspection and
behaviour only — and was asked to break the system by any available path.

It found thirteen issues, five of them critical. All are fixed, and every one
has a regression test in `tests/test_integrity.py` that reproduces the original
attack and asserts it now fails.

This document exists because the findings are more interesting than the fixes.

## Why a payments library gets red-teamed at all

There is no server here and no attacker. The threat model is narrower and more
uncomfortable than that.

**The untrusted component is the policy** — and the policy is *me*. It is the
part still being written, the part that will become model-driven in part 08, and
the part whose numbers decide whether this project claims a result. The
realistic failure is not sabotage. It is an author writing a subtly wrong policy
and believing the number it produces.

A project whose entire pitch is honest metrics has to be able to answer "how do
you know your own harness isn't lying to you?" The answer should not be "I was
careful."

## The finding that mattered

**The harness reported its numbers by reading the object it had just handed to
the untrusted component, and never reconciled that against the audit trail it
was building in parallel.**

`Ledger` was correctly frozen. But `Episode.ledger` was a rebindable field, so:

```python
def decide(self, episode, now, deadline):
    episode.ledger = Ledger(10_000_000, 0, 0, 0)
    episode.resolved = True
    return None
```

produced `recovery_rate = 1.0`, `net = +410,000,000`, and **zero audit rows**.
The honest floor on the same batch was `net = −4,211,136`.

Every headline metric — `recovery_rate`, `revocation_rate`, `net_paise`,
`contacts_per_episode` — was a direct read of a policy-writable field. The
reviewer demonstrated five variants:

| Attack | Effect |
|--------|--------|
| Rebind `ledger` | Arbitrary money, empty audit trail |
| Set `resolved = True` | 100% recovery rate at zero cost |
| Rewrite `failure_code` | Collected money from **revoked mandates** |
| Reset `contacts_made` | Churn cost cut 4×, intrusiveness under-reported 7.7× |
| Call `world.apply` directly | 82.7 attempts/episode past a cap of 3, empty audit trail |

The last one is the most dangerous, because it does not produce an absurd
number. It produces a *plausible winning* number attached to an audit trail a
reviewer would read as "this policy took no actions."

### Why it was fatal rather than merely a bug

The evidence that would have caught all five was already being built. `AuditEntry`
carries `recovered_paise` and `cost_paise` per event. It was simply never
compared to the report it was supposed to substantiate.

`Ledger`'s own docstring promised that "every rupee in the final report can be
traced to the action that produced it." That was not true, and nothing checked.
Building the evidence and not checking it against the claim is worse than not
building it, because it looks like assurance.

### The fix

Two independent defences, because either alone has failed before.

**Structural.** Policies now receive a frozen `EpisodeView`, never the live
episode. There is no path back to mutable state — history is handed over as a
tuple of frozen outcomes.

**Reconciliation.** Every figure in the report is rebuilt from the outcomes the
harness itself observed returning from `world.apply`. On top of that, `_Observed.verify`
re-derives the episode's own state from those observations after every single
step and raises `IntegrityError` on any disagreement. If a future change reopens
a tampering path, the run fails loudly instead of reporting a good number.

Verified end to end: `report.recovered_paise == audit["recovered_paise"].sum()`.

## The finding the reviewer missed, and I found while fixing it

`Episode.customer` handed the policy the entire `Customer` object — including
`salary_day`, `balance_health`, `engagement`, `churn_intent` and
`preferred_channel`.

Those are the simulator's answer key. A learned policy could have read churn
intent directly, posted spectacular numbers, and been completely worthless. The
`EpisodeView` closes this too: it exposes an opaque `customer_id` and nothing
else about the person.

Worth recording that the black-box reviewer could not have found this. It is
invisible from the outside — the baselines happen not to use those fields, so
nothing misbehaves. It only shows up when you ask *what could this component
reach* rather than *what does it currently do*.

## The finding that was not an attack

The most dangerous item on the list was a mistake waiting to be made in good
faith, two days later, by me.

`FORBIDDEN_COLUMNS` guards the *simulator's latents*. So the natural way to
build a feature matrix is:

```python
X = df.drop(columns=[label] + list(FORBIDDEN_COLUMNS))
```

That leaves `episode_net_paise` in. Best single-threshold accuracy against
`episode_recovered`, base rate 0.269:

| Column | Accuracy | Was it forbidden? |
|--------|---------:|-------------------|
| `episode_net_paise` | **1.0000** | no |
| `recovered_paise` | 0.8329 | no |
| `succeeded` | 0.8329 | no |
| `episode_steps` | 0.7898 | no |
| `episode_spent_paise` | 0.7427 | no |

`episode_net_paise` *is* the label, arithmetically restated. I would have
trained a model in part 06, seen a held-out AUC near 1.0, and been delighted.

**Fix:** `feature_columns()` is now an allowlist by subtraction — everything that
is not an identifier, an outcome, behavioural metadata, or a latent. Deliberately
not a denylist: a denylist fails open, so the day someone adds a column and
forgets to blacklist it, it silently becomes a feature. This fails closed.

Backed by a *probe* rather than a naming check: `test_no_selectable_feature_trivially_determines_the_label`
sweeps thresholds over every selectable feature and fails if any exceeds 97%
accuracy. That catches the next one too, whatever it gets called.

## Everything else

| # | Finding | Fix |
|---|---------|-----|
| 2 | One negative `cycle_amount_paise` flipped every policy's net positive; `no_recovery` reported as earning ₹120 crore | Validation at `Mandate.__post_init__`, `revocation_cost_paise`, and all three `Ledger` mutators |
| 3 | Cloned rows with renamed ids passed `assert_split_is_clean`; label recoverable by exact join at accuracy 1.000 | Feature-twin detection with a tolerance, on content rather than identity |
| 3b | The verifier chose *which* checks to run from the free-text `Split.name`; `"whatever"` skipped both | Typed `SplitKind` enum |
| 8 | `precision_at_10pct` held precision at whatever `capacity` was passed | Renamed `*_at_capacity`, `capacity` carried on the report, range validated |
| 9 | `reliability_table` binned NaN into the **top** calibration bin and fabricated an observed rate; `classification_report` rejected the same input | Shared `_validate_probabilities` gate |
| 10 | `lift_over_baseline` returned NaN on 2 of 4 fields against the documented floor policy | Documented as correct-by-construction; zero-episode reports now raise |
| 12 | One crashing policy destroyed an entire comparison run; no timeout | Per-policy isolation in `evaluate_all`, wall-clock timeout. `IntegrityError` is never isolated |
| 13 | Bare `KeyError`/`TypeError` on malformed decisions; tz-aware datetimes; single-class test sets; `bins=0`; duplicate policy names | Validation with messages that name the caller's error |

## What held

Worth recording, because it is most of the codebase and the reviewer attacked it
properly.

**Reproducibility discipline was called "the best-executed part of the
codebase."** Identical results across fresh worlds at the same seed, across
reordered policy lists, and after another policy burned 500 RNG draws per
decision. No cross-run contamination.

Also held: `Ledger` frozen against in-place mutation; `world.apply` refusing
closed episodes; the recovery-window deadline; backwards time-travel clamped;
`max_steps` bounding the loop; `classification_report`'s input gate; correct sign
handling on negative nets; `test_fraction` validation; the stable customer hash;
and `assert_split_is_clean` correctly catching shared episodes, empty sides,
temporal reversal and shared customers.

## What I take from it

The trust boundary was drawn one level too shallow. The *fields* were protected
and the *object graph* was not, and every one of the five critical findings is a
variant of that single mistake.

The pattern across this whole build is consistent, and it is not about security:
**the expensive errors were modelling and assurance errors, not coding errors.**
The label that taught the model never to contact anyone, the LTV horizon that
made loyal customers free to churn, the world where only merchants caused
revocation, and the harness that trusted its own subject — none were bugs. In
every case the code did exactly what was written. All four would have produced
confident, well-tested, entirely wrong numbers.

Tests do not catch that class of mistake. Adversarial review and per-slice
output inspection do.
