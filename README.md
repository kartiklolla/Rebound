# Rebound

**A recovery orchestrator for failed recurring debits on Indian payment rails.**

Razorpay AI Buildathon · Track 03, AI Revenue Recovery

---

> 🚧 **In development.** Metrics tables in this README are filled in as they are
> measured. Empty means not yet measured, never "measured and omitted."
> See [PROGRESS.md](PROGRESS.md) for current state and [JOURNAL.md](JOURNAL.md) for
> what broke along the way.

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

Not yet measured. See [PROGRESS.md](PROGRESS.md).

## Where we deliberately did not use an LLM

Not yet written — see [PROGRESS.md](PROGRESS.md). (Summary: retry timing and rail
selection are calibrated-probability problems, not language problems.)

## Running it

```bash
uv sync
uv run pytest
```
