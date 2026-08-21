"""Rollout harness: run a policy against a batch of failed debits and price it.

Common random numbers
---------------------
Every policy faces the same batch, and each episode is given its own seeded
random stream derived from its index. Policy A and policy B therefore meet the
same customer under the same luck.

This matters more than it sounds. Without it, part of the measured difference
between two policies is just which one drew better dice, and with revocation
events at a couple of percent it takes a large batch before that noise stops
dominating. Pairing the draws removes most of it, so a reported lift is a real
difference rather than a lucky sample.

The pairing is not perfect — once two policies take different actions their
streams diverge — but it is aligned where it matters most, at the start of each
episode where the failure and the early outcomes are decided.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from rebound.economics import Ledger
from rebound.eval.metrics import PolicyReport
from rebound.policy import Decision, Policy
from rebound.sim.world import Customer, Episode, Mandate, World
from rebound.taxonomy import Action

DEFAULT_RECOVERY_WINDOW_DAYS = 28
DEFAULT_MAX_STEPS = 8


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
    """Episodes that ended without recovery, and why — the honest exception list."""

    def audit_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(entry) for entry in self.audit])

    def exception_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.exceptions)


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

    Uses a dedicated stream so that building the batch does not depend on, or
    disturb, whatever a policy run does afterwards. The batch is fixed before
    any policy sees it.
    """
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
            customer = by_id[mandate.customer_id]
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


def evaluate_policy(
    world: World,
    policy: Policy,
    batch: list[EpisodeSpec],
    seed: int = 31337,
    recovery_window_days: int = DEFAULT_RECOVERY_WINDOW_DAYS,
    max_steps: int = DEFAULT_MAX_STEPS,
    collect_audit: bool = True,
) -> RolloutResult:
    """Run one policy over the whole batch and price what it did."""
    if not batch:
        raise ValueError("cannot evaluate a policy on an empty batch")

    policy.reset()
    total = Ledger()
    recovered = revoked = 0
    contacts = 0
    audit: list[AuditEntry] = []
    exceptions: list[dict] = []

    for index, spec in enumerate(batch):
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
        deadline = spec.failed_at + dt.timedelta(days=recovery_window_days)
        _work_episode(
            world, policy, episode, deadline, max_steps, audit if collect_audit else None
        )
        # Settle passive churn. An abandoned payment carries its own revocation
        # risk, so giving up is never actually free.
        world.close_episode(episode)

        total = total + episode.ledger
        recovered += episode.resolved
        revoked += episode.revoked
        contacts += episode.contacts_made

        if not episode.resolved:
            exceptions.append(_exception_row(episode))

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
    deadline: dt.datetime,
    max_steps: int,
    audit: list[AuditEntry] | None,
) -> None:
    now = episode.failed_at

    for step in range(max_steps):
        if episode.closed:
            return
        decision = policy.decide(episode, now, deadline)
        if decision is None:
            return

        at = _validate(decision, now, deadline)
        if at is None:
            return

        outcome = world.apply(episode, decision.action, at)
        now = at

        if audit is not None:
            audit.append(
                AuditEntry(
                    episode_id=episode.episode_id,
                    customer_id=episode.customer.customer_id,
                    step=step,
                    at=at,
                    action=str(decision.action),
                    reason=decision.reason,
                    succeeded=outcome.succeeded,
                    revoked=outcome.revoked,
                    cost_paise=outcome.cost_paise,
                    recovered_paise=outcome.recovered_paise,
                    detail=outcome.detail,
                )
            )

        if decision.action is Action.STOP:
            return


def _validate(
    decision: Decision, now: dt.datetime, deadline: dt.datetime
) -> dt.datetime | None:
    """Enforce that time moves forward and stays inside the recovery window.

    A policy that schedules an action at or before ``now`` would loop forever
    without advancing. Rather than trust every policy to get this right, the
    harness nudges the action forward by a minute — a bug in a candidate policy
    should show up as a poor score, not as a hung evaluation.
    """
    at = decision.at
    if at <= now:
        at = now + dt.timedelta(minutes=1)
    if at > deadline:
        return None
    return at


def _exception_row(episode: Episode) -> dict:
    """Why one episode was not recovered.

    The track asks for an honest exception list, and this is it: not a count of
    failures but a per-episode reason, which is the difference between a number
    a merchant can act on and one they can only feel bad about.
    """
    if episode.revoked:
        contacted = episode.contacts_made > 0
        reason = (
            "customer revoked the mandate after being contacted"
            if contacted
            else "customer revoked on their own once the payment went unrecovered"
        )
    elif episode.stopped:
        reason = "policy stopped deliberately"
    elif episode.history:
        reason = f"exhausted the recovery window: {episode.history[-1].detail}"
    else:
        reason = "policy took no action"

    return {
        "episode_id": episode.episode_id,
        "customer_id": episode.customer.customer_id,
        "failure_code": episode.failure_code,
        "disposition": str(episode.disposition),
        "rail": str(episode.mandate.rail),
        "amount_paise": episode.mandate.cycle_amount_paise,
        "attempts": episode.ledger.attempts,
        "contacts": episode.contacts_made,
        "spent_paise": episode.ledger.spent_paise,
        "destroyed_paise": episode.ledger.destroyed_paise,
        "reason": reason,
    }


def evaluate_all(
    world: World,
    policies: list[Policy],
    batch: list[EpisodeSpec],
    seed: int = 31337,
    **kwargs,
) -> dict[str, RolloutResult]:
    """Run every policy over the same batch under the same paired draws."""
    return {
        policy.name: evaluate_policy(world, policy, batch, seed=seed, **kwargs)
        for policy in policies
    }
