"""Rollout harness: run a policy against a batch of failed debits and price it.

Trust model
-----------
**The policy is untrusted.** In deployment it is the component under active
development — eventually model-driven — so the harness must not be foolable by
a buggy or hostile one. That is not paranoia about malice; the realistic failure
is an author who writes a subtly wrong policy and then believes its numbers.

Two independent defences, because either alone has failed before:

**Structural.** A policy is handed a frozen
:class:`~rebound.sim.world.EpisodeView`, never the live episode. It cannot set
``resolved``, rebind the ledger, rewrite the failure code, reset the contact
counter, or reach the simulator's customer latents. An earlier version passed
the live object and every one of those was demonstrated against it.

**Reconciliation.** Every figure in the report is rebuilt from the outcomes the
harness itself observed coming back out of ``world.apply`` — never read off the
episode. On top of that, ``_verify`` re-derives the episode's own state from
those observations after every step and raises :class:`IntegrityError` on any
disagreement. If a future change reopens a tampering path, the run fails loudly
instead of reporting a good number.

The reconciliation exists because the audit trail was already being built and
never compared to the report it was supposed to substantiate. Building the
evidence and not checking it against the claim is worse than not building it.

Common random numbers
---------------------
Every policy faces the same batch, and each episode gets its own seeded stream
derived from its index, so policy A and policy B meet the same customer under
the same luck. With revocation at a few percent, unpaired draws mean much of any
measured difference is just which policy drew better dice.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from rebound.economics import Ledger
from rebound.eval.metrics import PolicyReport
from rebound.policy import Decision, Policy
from rebound.sim.world import (
    CONTACT_ACTIONS,
    ActionOutcome,
    Customer,
    Episode,
    Mandate,
    World,
)
from rebound.taxonomy import Action

DEFAULT_RECOVERY_WINDOW_DAYS = 28
DEFAULT_MAX_STEPS = 8
DEFAULT_TIMEOUT_SECONDS = 120.0


class IntegrityError(RuntimeError):
    """Raised when an episode's state disagrees with the harness's own record.

    Always a defect, never a data condition. Either the world changed the
    episode in a way the harness did not observe, or something mutated it
    outside the rollout. Both mean the reported numbers cannot be trusted, so
    the run stops rather than producing a figure nobody can stand behind.
    """


class PolicyFailed(RuntimeError):
    """Raised when a policy misbehaves badly enough to abandon its run."""


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """A failed debit to be worked, independent of who works it."""

    mandate: Mandate
    customer: Customer
    failure_code: str
    failed_at: dt.datetime
    cycles_elapsed: int


@dataclass(slots=True)
class AuditEntry:
    """One line of the audit trail.

    Carries the policy's stated reason alongside the outcome, so a reviewer can
    see not just what was done but what the system believed when it did it.
    """

    episode_id: str
    customer_id: str
    step: int
    at: dt.datetime
    action: str
    reason: str
    succeeded: bool
    revoked: bool
    cost_paise: int
    recovered_paise: int
    detail: str


@dataclass(slots=True)
class RolloutResult:
    report: PolicyReport
    audit: list[AuditEntry] = field(default_factory=list)
    exceptions: list[dict] = field(default_factory=list)
    """Episodes that ended without recovery, and why — the business exception
    list the track asks for. Not error handling; see ``error`` for that."""

    error: str | None = None
    """Set when the policy crashed and its run was abandoned."""

    @property
    def failed(self) -> bool:
        return self.error is not None

    def audit_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(entry) for entry in self.audit])

    def exception_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.exceptions)


# ==========================================================================
# Integrity
# ==========================================================================


@dataclass(slots=True)
class _Observed:
    """The harness's own record of one episode, independent of the episode.

    Every reported figure is derived from here. The episode object is treated
    as a suggestion; this is the evidence.
    """

    failure_code: str
    outcomes: list[ActionOutcome] = field(default_factory=list)

    @property
    def ledger(self) -> Ledger:
        ledger = Ledger()
        for outcome in self.outcomes:
            ledger = ledger.plus_cost(
                outcome.cost_paise, counts_as_attempt=outcome.action is not Action.STOP
            )
            if outcome.recovered_paise:
                ledger = ledger.plus_recovery(outcome.recovered_paise)
            if outcome.destroyed_paise:
                ledger = ledger.plus_destruction(outcome.destroyed_paise)
        return ledger

    @property
    def recovered(self) -> bool:
        return any(outcome.succeeded for outcome in self.outcomes)

    @property
    def revoked(self) -> bool:
        return any(outcome.revoked for outcome in self.outcomes)

    @property
    def contacts(self) -> int:
        return sum(1 for o in self.outcomes if o.action in CONTACT_ACTIONS)

    def verify(self, episode: Episode) -> None:
        """Re-derive the episode's state from observation and insist they agree.

        Defence in depth. The read-only view should already make every one of
        these impossible; this catches the case where a future change hands out
        a live reference again, and turns a silently inflated metric into a
        crash.
        """
        if episode.failure_code != self.failure_code:
            raise IntegrityError(
                f"{episode.episode_id}: failure code changed from "
                f"{self.failure_code!r} to {episode.failure_code!r} mid-episode. "
                f"The failure code is a world-owned fact."
            )
        if len(episode.history) != len(self.outcomes):
            raise IntegrityError(
                f"{episode.episode_id}: episode has {len(episode.history)} "
                f"recorded outcomes but the harness observed "
                f"{len(self.outcomes)}. Something acted on this episode outside "
                f"the rollout, so the audit trail is not a complete record."
            )
        if episode.ledger != self.ledger:
            raise IntegrityError(
                f"{episode.episode_id}: episode ledger {episode.ledger} does "
                f"not reconcile with the harness's record {self.ledger}."
            )
        if episode.resolved != self.recovered:
            raise IntegrityError(
                f"{episode.episode_id}: resolved={episode.resolved} but no "
                f"observed outcome recovered money."
            )
        if episode.revoked != self.revoked:
            raise IntegrityError(
                f"{episode.episode_id}: revoked={episode.revoked} disagrees "
                f"with the observed outcomes."
            )
        if episode.contacts_made != self.contacts:
            raise IntegrityError(
                f"{episode.episode_id}: contacts_made={episode.contacts_made} "
                f"but {self.contacts} contact actions were observed. Resetting "
                f"this counter suppresses contact fatigue and revocation risk."
            )


# ==========================================================================
# Batch construction
# ==========================================================================


def build_eval_batch(
    world: World,
    customers: list[Customer],
    mandates: list[Mandate],
    start: dt.date,
    end: dt.date,
    max_episodes: int | None = None,
    seed: int = 909,
) -> list[EpisodeSpec]:
    """Collect the failed debits a merchant would face over a period.

    Uses a dedicated stream so that building the batch neither depends on nor
    disturbs whatever a policy run does afterwards. The batch is fixed before
    any policy sees it.
    """
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    rng = np.random.default_rng(seed)
    by_id = {c.customer_id: c for c in customers}
    by_billing_day: dict[int, list[Mandate]] = {}
    for mandate in mandates:
        by_billing_day.setdefault(mandate.billing_day, []).append(mandate)

    specs: list[EpisodeSpec] = []
    day = start
    while day <= end:
        for mandate in by_billing_day.get(day.day, ()):
            if day < mandate.registered_on:
                continue
            customer = by_id.get(mandate.customer_id)
            if customer is None:
                continue
            at = dt.datetime.combine(
                day, dt.time(int(rng.integers(0, 24)), int(rng.integers(0, 60)))
            )
            code = world.sample_failure(mandate, customer, at, rng=rng)
            if code is None:
                continue
            specs.append(
                EpisodeSpec(
                    mandate=mandate,
                    customer=customer,
                    failure_code=code,
                    failed_at=at,
                    cycles_elapsed=max(
                        0,
                        (day.year - mandate.registered_on.year) * 12
                        + (day.month - mandate.registered_on.month),
                    ),
                )
            )
            if max_episodes and len(specs) >= max_episodes:
                return specs
        day += dt.timedelta(days=1)
    return specs


# ==========================================================================
# Rollout
# ==========================================================================


def evaluate_policy(
    world: World,
    policy: Policy,
    batch: list[EpisodeSpec],
    seed: int = 31337,
    recovery_window_days: int = DEFAULT_RECOVERY_WINDOW_DAYS,
    max_steps: int = DEFAULT_MAX_STEPS,
    collect_audit: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> RolloutResult:
    """Run one policy over the whole batch and price what it did.

    Every reported figure comes from the harness's own observations, not from
    the episode objects the policy participated in.
    """
    if not batch:
        raise ValueError("cannot evaluate a policy on an empty batch")

    policy.reset()
    started = time.monotonic()

    total = Ledger()
    recovered = revoked = contacts = 0
    audit: list[AuditEntry] = []
    exceptions: list[dict] = []

    for index, spec in enumerate(batch):
        if time.monotonic() - started > timeout_seconds:
            raise PolicyFailed(
                f"{policy.name} exceeded {timeout_seconds}s after {index} of "
                f"{len(batch)} episodes. A policy with an unbounded decide() "
                f"would otherwise hang the whole comparison."
            )

        # Common random numbers: this episode's stream depends only on its
        # index, so every policy meets it under identical conditions.
        world.rng = np.random.default_rng(seed + index)

        episode = world.open_episode(
            episode_id=f"EV_{index:08d}",
            mandate=spec.mandate,
            customer=spec.customer,
            failure_code=spec.failure_code,
            failed_at=spec.failed_at,
            cycles_elapsed=spec.cycles_elapsed,
        )
        observed = _Observed(failure_code=spec.failure_code)
        deadline = spec.failed_at + dt.timedelta(days=recovery_window_days)

        _work_episode(
            world,
            policy,
            episode,
            observed,
            deadline,
            max_steps,
            audit if collect_audit else None,
        )

        # Settle passive churn. An abandoned payment carries its own revocation
        # risk, so giving up is never actually free.
        settlement = world.close_episode(episode)
        if settlement is not None:
            observed.outcomes.append(settlement)
        observed.verify(episode)

        total = total + observed.ledger
        recovered += observed.recovered
        revoked += observed.revoked
        contacts += observed.contacts

        if not observed.recovered:
            exceptions.append(_exception_row(episode, observed))

    report = PolicyReport(
        policy=policy.name,
        episodes=len(batch),
        recovery_rate=recovered / len(batch),
        revocation_rate=revoked / len(batch),
        recovered_paise=total.recovered_paise,
        spent_paise=total.spent_paise,
        destroyed_paise=total.destroyed_paise,
        net_paise=total.net_paise,
        attempts_per_episode=total.attempts / len(batch),
        contacts_per_episode=contacts / len(batch),
    )
    return RolloutResult(report=report, audit=audit, exceptions=exceptions)


def _work_episode(
    world: World,
    policy: Policy,
    episode: Episode,
    observed: _Observed,
    deadline: dt.datetime,
    max_steps: int,
    audit: list[AuditEntry] | None,
) -> None:
    now = episode.failed_at

    for step in range(max_steps):
        if episode.closed:
            return

        # The policy sees a frozen projection, never the live episode.
        decision = policy.decide(episode.view(), now, deadline)
        if decision is None:
            return

        _validate_decision(decision)
        at = _schedule(decision, now, deadline)
        if at is None:
            return

        outcome = world.apply(episode, decision.action, at)
        observed.outcomes.append(outcome)
        observed.verify(episode)
        now = at

        if audit is not None:
            audit.append(
                AuditEntry(
                    episode_id=episode.episode_id,
                    customer_id=episode.customer.customer_id,
                    step=step,
                    at=at,
                    action=str(decision.action),
                    reason=decision.reason or "",
                    succeeded=outcome.succeeded,
                    revoked=outcome.revoked,
                    cost_paise=outcome.cost_paise,
                    recovered_paise=outcome.recovered_paise,
                    detail=outcome.detail,
                )
            )

        if decision.action is Action.STOP:
            return


def _validate_decision(decision: Decision) -> None:
    """Reject malformed decisions with a message that names the policy's error.

    Without this, a policy returning a string action or a date instead of a
    datetime surfaces as a bare ``KeyError`` or ``TypeError`` from somewhere
    deep in the world, which is a miserable thing to debug.
    """
    if not isinstance(decision, Decision):
        raise PolicyFailed(
            f"decide() must return a Decision or None, got {type(decision).__name__}"
        )
    if not isinstance(decision.action, Action):
        raise PolicyFailed(
            f"Decision.action must be a taxonomy Action, got "
            f"{decision.action!r} ({type(decision.action).__name__})"
        )
    if not isinstance(decision.at, dt.datetime):
        raise PolicyFailed(
            f"Decision.at must be a datetime, got "
            f"{decision.at!r} ({type(decision.at).__name__})"
        )
    if decision.at.tzinfo is not None:
        raise PolicyFailed(
            "Decision.at must be timezone-naive; the whole simulation runs in "
            "one implicit local timezone and mixing the two raises deep inside "
            "the world on an unrelated comparison"
        )


def _schedule(
    decision: Decision, now: dt.datetime, deadline: dt.datetime
) -> dt.datetime | None:
    """Enforce that time moves forward and stays inside the recovery window.

    A policy that schedules at or before ``now`` would loop without advancing.
    The harness nudges it forward rather than trusting every policy to be
    correct — a bug in a candidate should show up as a poor score, not a hang.
    """
    at = decision.at
    if at <= now:
        at = now + dt.timedelta(minutes=1)
    if at > deadline:
        return None
    return at


def _exception_row(episode: Episode, observed: _Observed) -> dict:
    """Why one episode was not recovered.

    The track asks for an honest exception list, and this is it: not a count of
    failures but a per-episode reason, which is the difference between a number
    a merchant can act on and one they can only feel bad about.
    """
    if observed.revoked:
        reason = (
            "customer revoked the mandate after being contacted"
            if observed.contacts
            else "customer revoked on their own once the payment went unrecovered"
        )
    elif episode.stopped:
        reason = "policy stopped deliberately"
    elif observed.outcomes:
        reason = f"exhausted the recovery window: {observed.outcomes[-1].detail}"
    else:
        reason = "policy took no action"

    ledger = observed.ledger
    return {
        "episode_id": episode.episode_id,
        "customer_id": episode.customer.customer_id,
        "failure_code": observed.failure_code,
        "disposition": str(episode.disposition),
        "rail": str(episode.mandate.rail),
        "amount_paise": episode.mandate.cycle_amount_paise,
        "attempts": ledger.attempts,
        "contacts": observed.contacts,
        "spent_paise": ledger.spent_paise,
        "destroyed_paise": ledger.destroyed_paise,
        "reason": reason,
    }


def evaluate_all(
    world: World,
    policies: list[Policy],
    batch: list[EpisodeSpec],
    seed: int = 31337,
    fail_fast: bool = False,
    **kwargs,
) -> dict[str, RolloutResult]:
    """Run every policy over the same batch under the same paired draws.

    A crashing policy is isolated by default: its result carries an ``error``
    and the rest of the comparison still completes. Given that the policy is
    the component under development, losing an entire comparison run to one
    exception is a real cost, and a half-hour rollout is an expensive thing to
    throw away over a typo.

    ``IntegrityError`` is never isolated. It means the numbers cannot be
    trusted, and continuing would produce a table containing a figure nobody
    can stand behind.
    """
    results: dict[str, RolloutResult] = {}
    for policy in policies:
        try:
            results[policy.name] = evaluate_policy(
                world, policy, batch, seed=seed, **kwargs
            )
        except IntegrityError:
            raise
        except Exception as error:  # noqa: BLE001 — isolating untrusted plugins
            if fail_fast:
                raise
            results[policy.name] = RolloutResult(
                report=_null_report(policy.name, len(batch)),
                error=f"{type(error).__name__}: {error}",
            )
    return results


def _null_report(name: str, episodes: int) -> PolicyReport:
    return PolicyReport(
        policy=name,
        episodes=episodes,
        recovery_rate=0.0,
        revocation_rate=0.0,
        recovered_paise=0,
        spent_paise=0,
        destroyed_paise=0,
        net_paise=0,
        attempts_per_episode=0.0,
        contacts_per_episode=0.0,
    )
