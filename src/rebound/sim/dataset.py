"""Generation of the historical recovery log — the training data.

What this simulates
-------------------
A merchant who has been running recurring billing for eighteen months with a
crude, semi-random retry ladder, and kept records. That is the data a real
merchant would hand over on day one: not the output of a good policy, but the
output of whatever they were doing before.

Three properties of that log matter more than its size.

**It must contain bad decisions.** The behavioural ladder retries revoked
mandates, calls customers who only needed an SMS, and re-presents into closed
execution windows. If it only did sensible things, the model would never see a
futile retry and could not learn that futility — which is the single most
valuable thing it can know, since avoiding doomed attempts is where a large
share of the savings comes from. Real merchants make these mistakes constantly,
so modelling them is realistic and load-bearing at the same time.

**It must vary its timing.** A merchant who always retried at exactly +1/+3/+7
days generates a log from which nothing about timing can be inferred, because
timing never varied. The ladder jitters both the delay and the hour of day.

**Its features must be strictly backward-looking.** Per-customer history is
accumulated in chronological order and read before it is updated, so a row can
only ever see that customer's past. The subtle prize here is
``cust_prior_mean_failure_day``: the average day-of-month this customer has
previously failed on. It is entirely observable to a merchant, and it is a
proxy for the customer's hidden salary day. Inferring the latent from history
is the learning problem this dataset is built to pose.

Logged propensities
-------------------
Every row records the probability with which the behavioural policy chose that
action. This makes the log usable for off-policy estimation rather than only
supervised learning, and it makes the coverage limits measurable instead of
rhetorical: where propensity is near zero, any later claim about that action is
extrapolation.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rebound.sim.world import (
    Customer,
    Episode,
    Mandate,
    World,
    within_upi_execution_window,
)
from rebound.taxonomy import Action, get_mode

# --------------------------------------------------------------------------
# The behavioural policy
# --------------------------------------------------------------------------

#: What the merchant was doing before Rebound.
#:
#: Deliberately *not* conditioned on the failure code. Merchants routinely
#: retry without parsing why the debit failed, which is both the realistic
#: behaviour and the reason the log contains the futile attempts the model
#: needs in order to learn futility. Weighted toward retrying because retrying
#: is cheap and feels free.
EXPLORATION_WEIGHTS: dict[Action, float] = {
    Action.RETRY_SAME_RAIL: 0.42,
    Action.NUDGE_SMS: 0.13,
    Action.NUDGE_WHATSAPP: 0.11,
    Action.NUDGE_EMAIL: 0.08,
    Action.SEND_COLLECT_LINK: 0.07,
    Action.RETRY_ALT_RAIL: 0.05,
    Action.SEND_PRE_DEBIT_NOTIFICATION: 0.04,
    Action.REQUEST_REMANDATE: 0.03,
    Action.REQUEST_MANDATE_AMENDMENT: 0.03,
    Action.VOICE_CALL: 0.02,
    Action.STOP: 0.02,
}

#: Delay in days before the next action, as a mixture.
#:
#: Centred on the familiar +1/+3/+7 ladder, because that is what merchants
#: actually do, but jittered and with a long tail. Without the jitter the log
#: would contain no information about timing at all.
_LADDER_DAYS: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 14)
_LADDER_WEIGHTS: tuple[float, ...] = (0.26, 0.16, 0.20, 0.12, 0.14, 0.07, 0.05)


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Everything that determines the generated log."""

    n_customers: int = 6_000
    start: dt.date = dt.date(2025, 1, 1)
    end: dt.date = dt.date(2026, 6, 30)
    registration_start: dt.date = dt.date(2024, 1, 1)
    registration_end: dt.date = dt.date(2025, 6, 30)
    max_ladder_steps: int = 5
    recovery_window_days: int = 28
    """How long recovery on one cycle may run before it is abandoned.

    Bounded by the billing period, because a merchant stops chasing last
    month's payment when this month's comes due. Without the bound, episodes
    overlap in wall-clock time — January's recovery is still limping along when
    February bills — which makes per-customer history features arrive stale and
    makes the log describe a merchant nobody would recognise.
    """

    seed: int = 20260821

    @property
    def span_days(self) -> int:
        return (self.end - self.start).days


@dataclass(slots=True)
class _CustomerHistory:
    """Running, strictly backward-looking statistics for one customer.

    Read before update, always. The ordering is the entire leakage defence for
    every ``cust_prior_*`` feature, so it is done in one place rather than
    scattered through the generation loop.
    """

    failures: int = 0
    recoveries: int = 0
    contacts: int = 0
    failure_day_total: int = 0
    revoked: bool = False

    @property
    def recovery_rate(self) -> float:
        return self.recoveries / self.failures if self.failures else 0.0

    @property
    def mean_failure_day(self) -> float:
        """Average day-of-month this customer has previously failed on.

        Observable to any merchant, and a usable proxy for the customer's
        hidden salary day. The model is never told the salary day; this is one
        of the few routes by which it can be inferred.
        """
        return self.failure_day_total / self.failures if self.failures else 0.0


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def generate_log(
    config: GenerationConfig = GenerationConfig(),
    world: World | None = None,
) -> pd.DataFrame:
    """Generate the historical recovery log.

    One row per decision point: a single action taken on an open episode, the
    observable state at the moment it was taken, and what happened.
    """
    world = world or World(seed=config.seed)
    rng = np.random.default_rng(config.seed + 2)

    customers = world.sample_customers(config.n_customers)
    mandates = world.sample_mandates(
        customers, config.registration_start, config.registration_end
    )
    calibration_dates = [
        config.start + dt.timedelta(days=i) for i in range(0, 28)
    ]
    world.calibrate(customers, mandates, calibration_dates)

    by_id = {c.customer_id: c for c in customers}
    history: dict[str, _CustomerHistory] = {
        c.customer_id: _CustomerHistory() for c in customers
    }

    rows: list[dict] = []
    episode_counter = 0

    # Indexed by billing day so each date touches only the mandates that
    # actually bill on it. The naive nested loop is days x mandates, which is
    # millions of iterations spent almost entirely on a date comparison.
    by_billing_day: dict[int, list[Mandate]] = {}
    for mandate in mandates:
        by_billing_day.setdefault(mandate.billing_day, []).append(mandate)

    for billing_date in _billing_dates(config):
        for mandate in by_billing_day.get(billing_date.day, ()):
            if billing_date < mandate.registered_on:
                continue

            customer = by_id[mandate.customer_id]
            record = history[customer.customer_id]
            if record.revoked:
                # A revoked mandate stops billing entirely. Continuing to
                # present against it would manufacture failures that no real
                # merchant would ever see.
                continue

            presented_at = dt.datetime.combine(
                billing_date, _billing_time(rng)
            )
            code = world.sample_failure(mandate, customer, presented_at)
            if code is None:
                continue

            episode_counter += 1
            episode = world.open_episode(
                episode_id=f"EP_{episode_counter:08d}",
                mandate=mandate,
                customer=customer,
                failure_code=code,
                failed_at=presented_at,
                cycles_elapsed=_cycles_elapsed(mandate, billing_date),
            )

            episode_rows = _run_behavioural_ladder(
                world,
                episode,
                record,
                rng,
                config.max_ladder_steps,
                deadline=presented_at + dt.timedelta(days=config.recovery_window_days),
            )
            # Settle passive churn before labelling, so the training data
            # reflects the same dynamics the policy will be evaluated against.
            # A model trained on a world where giving up is free would
            # systematically under-value recovery.
            world.close_episode(episode)
            rows.extend(_attach_outcome_labels(episode_rows, episode))

            # Update history only after the episode is fully recorded.
            record.failures += 1
            record.failure_day_total += billing_date.day
            record.contacts += episode.contacts_made
            if episode.resolved:
                record.recoveries += 1
            if episode.revoked:
                record.revoked = True

    frame = pd.DataFrame(rows)
    return frame.sort_values(["decided_at", "episode_id"]).reset_index(drop=True)


def _billing_dates(config: GenerationConfig) -> list[dt.date]:
    return [
        config.start + dt.timedelta(days=i) for i in range(config.span_days + 1)
    ]


def _billing_time(rng: np.random.Generator) -> dt.time:
    """Time of day a scheduled debit is presented.

    Spread across the clock rather than pinned to one hour, so that closed
    UPI execution windows are hit naturally instead of being injected.
    """
    return dt.time(int(rng.integers(0, 24)), int(rng.integers(0, 60)))


def _cycles_elapsed(mandate: Mandate, on: dt.date) -> int:
    return max(0, (on.year - mandate.registered_on.year) * 12
               + (on.month - mandate.registered_on.month))


def _run_behavioural_ladder(
    world: World,
    episode: Episode,
    record: _CustomerHistory,
    rng: np.random.Generator,
    max_steps: int,
    deadline: dt.datetime,
) -> list[dict]:
    """Work one episode with the crude randomised ladder, recording each step.

    Abandons the episode when the next action would fall past ``deadline`` —
    the point at which the next billing cycle takes over and nobody is still
    chasing the old one.
    """
    actions = list(EXPLORATION_WEIGHTS)
    weights = np.array([EXPLORATION_WEIGHTS[a] for a in actions], dtype=float)
    weights /= weights.sum()

    rows: list[dict] = []
    at = episode.failed_at

    for step in range(max_steps):
        if episode.closed:
            break

        index = int(rng.choice(len(actions), p=weights))
        action = actions[index]
        propensity = float(weights[index])

        delay_days = int(
            rng.choice(_LADDER_DAYS, p=np.array(_LADDER_WEIGHTS) / sum(_LADDER_WEIGHTS))
        )
        at = at + dt.timedelta(
            days=delay_days,
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60)),
        )
        if at > deadline:
            # The recovery window closed. The merchant moves on to the next
            # cycle; this one is simply dropped rather than explicitly stopped.
            episode.stopped = True
            break

        features = _observable_features(episode, record, action, at, step)
        outcome = world.apply(episode, action, at)

        features.update(
            {
                "propensity": propensity,
                "succeeded": bool(outcome.succeeded),
                "revoked": bool(outcome.revoked),
                "recovered_paise": int(outcome.recovered_paise),
                "cost_paise": int(outcome.cost_paise),
            }
        )
        rows.append(features)

        if action is Action.STOP:
            break

    return rows


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def _attach_outcome_labels(rows: list[dict], episode: Episode) -> list[dict]:
    """Attach episode-level outcomes to every decision point in the episode.

    The credit-assignment fix, and the most consequential modelling decision in
    the dataset.

    The obvious label — "did *this* action recover the money" — is clean,
    causally unambiguous, and badly wrong. A nudge never recovers money
    directly; it unblocks the customer so that a *later* retry collects. Under
    the direct label every nudge, every pre-debit notification, every mandate
    repair scores exactly zero, and a model trained on it learns to never
    contact anyone. That would be a confident, well-tested conclusion that the
    main lever merchants actually have is worthless.

    So each decision point also carries the **return from that point**: whether
    the episode ultimately recovered, and whether it ultimately ended in
    revocation. That is the quantity a sequencer needs in order to choose an
    action, and it is what ``rebound.model`` is trained to predict.

    The honest caveat, carried into the README rather than left implicit: this
    return is realised *under the behavioural policy*. It answers "what
    happened when a merchant took this action and then carried on behaving like
    that merchant," not "what would happen under an optimal continuation."
    Logged propensities are what make that gap measurable rather than
    rhetorical.

    ``succeeded`` is kept alongside it. The contrast between immediate and
    downstream effect is exactly what distinguishes a collecting action from an
    enabling one, and both are worth being able to see.
    """
    for row in rows:
        row["episode_recovered"] = bool(episode.resolved)
        row["episode_revoked"] = bool(episode.revoked)
        row["episode_stopped"] = bool(episode.stopped)
        row["episode_steps"] = len(rows)
        row["episode_net_paise"] = int(episode.ledger.net_paise)
        row["episode_spent_paise"] = int(episode.ledger.spent_paise)
        row["episode_destroyed_paise"] = int(episode.ledger.destroyed_paise)
        row["steps_remaining"] = len(rows) - row["decision_index"] - 1
    return rows


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

#: Latent customer attributes that must never reach the log.
#:
#: Named explicitly so the leakage test can assert on them rather than trusting
#: that nobody added a convenient column during a late-night debugging session.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "salary_day",
        "balance_health",
        "engagement",
        "churn_intent",
        "preferred_channel",
        "days_since_salary",
        "funds_probability",
        "nudge_efficacy",
        "revocation_hazard",
        "outage_ends_at",
    }
)

#: Columns that identify a row rather than describe it.
#:
#: Useful for grouping and splitting, never as model input — a customer id is a
#: perfect memorisation handle and nothing else.
IDENTIFIER_COLUMNS: frozenset[str] = frozenset(
    {"episode_id", "customer_id", "mandate_id", "failed_at", "decided_at"}
)

#: The label, and everything computed from the episode's outcome.
#:
#: This set is the fix for the most dangerous gap found in review, and it was
#: not an attack — it was the mistake about to be made in good faith.
#:
#: ``FORBIDDEN_COLUMNS`` guards the *simulator's* latents, so the natural way to
#: build a feature matrix is ``df.drop(columns=[label] + list(FORBIDDEN_COLUMNS))``.
#: That leaves ``episode_net_paise`` in — and it predicts ``episode_recovered``
#: at accuracy **1.000** against a base rate of 0.269, because it *is* the label
#: arithmetically restated. Also leaking: ``recovered_paise`` (0.833),
#: ``succeeded`` (0.833), ``episode_steps`` (0.790), ``episode_spent_paise``
#: (0.743). The result would have been a model with a near-perfect held-out
#: score and no predictive content whatsoever.
OUTCOME_COLUMNS: frozenset[str] = frozenset(
    {
        "succeeded",
        "revoked",
        "recovered_paise",
        "cost_paise",
        "episode_recovered",
        "episode_revoked",
        "episode_stopped",
        "episode_steps",
        "episode_net_paise",
        "episode_spent_paise",
        "episode_destroyed_paise",
        "steps_remaining",
    }
)

#: Behavioural-policy metadata: real, but not a feature.
#:
#: Propensity describes how the *historical* policy chose, which a deployed
#: model would never have at decision time.
METADATA_COLUMNS: frozenset[str] = frozenset({"propensity"})


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The columns a model may legitimately train on.

    An **allowlist by subtraction**: everything that is not an identifier, an
    outcome, behavioural metadata, or a latent. Deliberately not a denylist —
    a denylist fails open, so the day someone adds a column and forgets to
    blacklist it, it silently becomes a feature. This fails closed: a new
    column must be classified before it can be used.
    """
    excluded = (
        IDENTIFIER_COLUMNS | OUTCOME_COLUMNS | METADATA_COLUMNS | FORBIDDEN_COLUMNS
    )
    return [column for column in frame.columns if column not in excluded]


def _observable_features(
    episode: Episode,
    record: _CustomerHistory,
    action: Action,
    at: dt.datetime,
    step: int,
) -> dict:
    """Everything a merchant would genuinely know at this decision point.

    Nothing here touches a customer latent. The closest thing to one is
    ``cust_prior_mean_failure_day``, which is computed from the merchant's own
    billing records and is exactly the kind of inference the model is meant to
    have to make.
    """
    mandate = episode.mandate
    mode = get_mode(episode.failure_code)

    return {
        # -- identifiers (for grouping and splitting, never features) ------
        "episode_id": episode.episode_id,
        "decision_index": step,
        "customer_id": episode.customer.customer_id,
        "mandate_id": mandate.mandate_id,
        "bank": episode.customer.bank,
        # -- timestamps -----------------------------------------------------
        "failed_at": episode.failed_at,
        "decided_at": at,
        # -- failure context ------------------------------------------------
        "rail": str(mandate.rail),
        "failure_code": episode.failure_code,
        "disposition": str(mode.disposition),
        "mandate_alive": mode.mandate_alive,
        "needs_customer_action": mode.needs_customer_action,
        "is_ambiguous_code": mode.ambiguous,
        # -- amounts --------------------------------------------------------
        "amount_paise": mandate.cycle_amount_paise,
        "ceiling_paise": mandate.ceiling_paise,
        "amount_to_ceiling": mandate.cycle_amount_paise / mandate.ceiling_paise,
        # -- mandate --------------------------------------------------------
        "mandate_age_days": (at.date() - mandate.registered_on).days,
        "days_to_expiry": (mandate.valid_until - at.date()).days,
        "billing_day": mandate.billing_day,
        "cycles_elapsed": episode.cycles_elapsed,
        # -- decision timing ------------------------------------------------
        "days_since_failure": (at - episode.failed_at).total_seconds() / 86400.0,
        "decision_day_of_month": at.day,
        "decision_hour": at.hour,
        "decision_weekday": at.weekday(),
        "within_upi_window": within_upi_execution_window(at),
        # -- episode state so far -------------------------------------------
        "prior_attempts": episode.ledger.attempts,
        "prior_contacts": episode.contacts_made,
        "notification_sent": episode.notification_sent_at is not None,
        "customer_unblocked": episode.customer_unblocked,
        "mandate_repaired": episode.mandate_repaired,
        "spent_so_far_paise": episode.ledger.spent_paise,
        # -- customer history, strictly prior --------------------------------
        "cust_prior_failures": record.failures,
        "cust_prior_recoveries": record.recoveries,
        "cust_prior_recovery_rate": record.recovery_rate,
        "cust_prior_mean_failure_day": record.mean_failure_day,
        "cust_prior_contacts": record.contacts,
        # -- the action -----------------------------------------------------
        "action": str(action),
    }


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def write_log(frame: pd.DataFrame, path: str) -> None:
    frame.to_parquet(path, index=False)


def read_log(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def coverage_report(frame: pd.DataFrame) -> pd.DataFrame:
    """How much of the (action × disposition) space the log actually covers.

    The honest counterpart to logged propensities. Cells with few rows are
    where any later claim is extrapolation rather than evidence, and this is
    the table that says which cells those are.
    """
    counts = (
        frame.groupby(["disposition", "action"], observed=True)
        .agg(
            rows=("succeeded", "size"),
            immediate_rate=("succeeded", "mean"),
            downstream_rate=("episode_recovered", "mean"),
            mean_propensity=("propensity", "mean"),
        )
        .reset_index()
    )
    return counts.sort_values(["disposition", "rows"], ascending=[True, False])
