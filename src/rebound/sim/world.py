"""The synthetic world: customer latents, failure generation, outcome sampling.

Reading guide
-------------
Three ideas carry most of the weight here.

**Latents the model cannot see.** Each customer has a salary day, a balance
health, an engagement level, a preferred channel and a churn intent. None are
observable. The model sees only what a merchant would actually have — failure
codes, timestamps, amounts, and what happened last time. Everything useful has
to be inferred from that, which is what makes Claim A a learning problem rather
than a lookup.

**Insufficient funds is generated from the salary curve, then calibrated.**
An NSF failure is not drawn from a flat rate; it is drawn from whether the
account is funded on that date, which depends on distance from the customer's
salary day. A calibration step then scales the draw so the *marginal* failure
rate matches the published NPCI anchor while the *conditional* structure
survives. The headline number stays honest and the signal stays learnable.

**Nudging can make things worse.** Every customer-facing action carries a
revocation hazard proportional to its intrusiveness and the customer's latent
churn intent, growing with each prior contact — while its efficacy decays with
each prior contact. Chasing harder works less and costs more, simultaneously.
Without this pairing, "contact everyone constantly" is optimal and the whole
problem is trivial.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field, replace

import numpy as np

from rebound.economics import (
    CHANNEL_INTRUSIVENESS,
    Ledger,
    attempt_cost_paise,
    revocation_cost_paise,
)
from rebound.sim.params import (
    DEFAULT_PARAMS,
    FAILURE_MIX,
    PRE_DEBIT_NOTIFICATION_HOURS,
    UPI_EXECUTION_WINDOWS,
    WorldParams,
)
from rebound.taxonomy import (
    Action,
    Disposition,
    Rail,
    disposition_of,
    get_mode,
)

#: Anonymised banks.
#:
#: Deliberately not real bank names. The simulator assigns each one an outage
#: propensity and a technical-decline tendency, and attaching invented
#: reliability figures to a named real institution would be publishing a claim
#: about that organisation. The model learns exactly as much from an opaque id.
BANKS: tuple[str, ...] = tuple(f"BANK_{i:02d}" for i in range(1, 13))

NUDGE_ACTIONS: frozenset[Action] = frozenset(
    {
        Action.NUDGE_SMS,
        Action.NUDGE_WHATSAPP,
        Action.NUDGE_EMAIL,
        Action.VOICE_CALL,
        Action.SEND_COLLECT_LINK,
    }
)

PREFERRED_CHANNELS: tuple[Action, ...] = (
    Action.NUDGE_SMS,
    Action.NUDGE_WHATSAPP,
    Action.NUDGE_EMAIL,
    Action.VOICE_CALL,
)

_RETRY_ACTIONS: frozenset[Action] = frozenset(
    {Action.RETRY_SAME_RAIL, Action.RETRY_ALT_RAIL}
)

_REPAIR_ACTIONS: frozenset[Action] = frozenset(
    {Action.REQUEST_REMANDATE, Action.REQUEST_MANDATE_AMENDMENT}
)


# ==========================================================================
# Entities
# ==========================================================================


@dataclass(frozen=True, slots=True)
class Customer:
    """A customer, including latents the model is never shown."""

    customer_id: str
    bank: str

    # -- latent -----------------------------------------------------------
    salary_day: int
    balance_health: float
    engagement: float
    churn_intent: float
    preferred_channel: Action
    has_alt_rail: bool


@dataclass(frozen=True, slots=True)
class Mandate:
    mandate_id: str
    customer_id: str
    rail: Rail
    cycle_amount_paise: int
    ceiling_paise: int
    billing_day: int
    registered_on: dt.date
    valid_until: dt.date


@dataclass(slots=True)
class Episode:
    """One failed debit, and the state of the attempt to recover it.

    Mutable, because it is a running conversation with a customer rather than a
    value. Everything that mutates is recorded in ``history`` so the audit trail
    can be reconstructed exactly.
    """

    episode_id: str
    mandate: Mandate
    customer: Customer
    failure_code: str
    failed_at: dt.datetime
    cycles_elapsed: int

    contacts_made: int = 0
    customer_unblocked: bool = False
    """Set when the customer clears whatever was blocking the debit."""

    notification_sent_at: dt.datetime | None = None
    outage_ends_at: dt.datetime | None = None
    mandate_repaired: bool = False

    resolved: bool = False
    revoked: bool = False
    stopped: bool = False
    ledger: Ledger = field(default_factory=Ledger)
    history: list[ActionOutcome] = field(default_factory=list)

    @property
    def closed(self) -> bool:
        return self.resolved or self.revoked or self.stopped

    @property
    def disposition(self) -> Disposition:
        return disposition_of(self.failure_code)


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """What happened when one action was taken. One row of the audit trail."""

    action: Action
    at: dt.datetime
    succeeded: bool
    recovered_paise: int
    cost_paise: int
    revoked: bool
    destroyed_paise: int
    detail: str


# ==========================================================================
# Time helpers
# ==========================================================================


def within_upi_execution_window(at: dt.datetime) -> bool:
    """Whether a UPI Autopay execution may be presented at this moment.

    Outside these windows the presentation is refused by the rail. Not a
    heuristic — a hard constraint, and the reason retry timing here is a
    compliance question and not only an optimisation.
    """
    clock = at.time()
    return any(start <= clock <= end for start, end in UPI_EXECUTION_WINDOWS)


def _days_since_salary(salary_day: int, on: dt.date) -> int:
    """Days since the customer's most recent salary credit.

    Wraps at the month boundary: a salary day of 28 and a date of the 2nd is
    a few days after payday, not twenty-six days before it.
    """
    if on.day >= salary_day:
        return on.day - salary_day
    previous_month_length = (on.replace(day=1) - dt.timedelta(days=1)).day
    return on.day + previous_month_length - salary_day


# ==========================================================================
# World
# ==========================================================================


class World:
    """Sampler for failures and recovery outcomes.

    Holds the only source of randomness in the system. Seeded explicitly so
    that every number in the README can be regenerated from a seed and a
    parameter set.
    """

    def __init__(
        self,
        params: WorldParams = DEFAULT_PARAMS,
        seed: int = 20260821,
    ) -> None:
        self.params = params
        self.rng = np.random.default_rng(seed)
        self._calibration_rng = np.random.default_rng(seed + 1)
        self._nsf_scale: dict[Rail, float] = {}
        self._other_scale: dict[Rail, float] = {}
        self.calibration_report: dict[Rail, dict[str, float]] = {}
        self._bank_outage_factor: dict[str, float] = {
            bank: float(self.rng.lognormal(0.0, params.bank_outage_dispersion))
            for bank in BANKS
        }

    # -- population -------------------------------------------------------

    def sample_customers(self, n: int) -> list[Customer]:
        p = self.params
        rng = self.rng

        early = rng.random(n) < p.salary_day_early_share
        salary_days = np.where(
            early,
            rng.integers(1, 8, size=n),
            rng.integers(8, 29, size=n),
        )
        return [
            Customer(
                customer_id=f"CUST_{i:07d}",
                bank=str(rng.choice(BANKS)),
                salary_day=int(salary_days[i]),
                balance_health=float(
                    rng.beta(p.balance_health_alpha, p.balance_health_beta)
                ),
                engagement=float(rng.beta(p.engagement_alpha, p.engagement_beta)),
                churn_intent=float(
                    rng.beta(p.churn_intent_alpha, p.churn_intent_beta)
                ),
                preferred_channel=PREFERRED_CHANNELS[
                    int(rng.integers(0, len(PREFERRED_CHANNELS)))
                ],
                has_alt_rail=bool(rng.random() < p.alt_rail_availability),
            )
            for i in range(n)
        ]

    def sample_mandates(
        self,
        customers: list[Customer],
        registered_from: dt.date,
        registered_to: dt.date,
    ) -> list[Mandate]:
        p = self.params
        rng = self.rng
        span_days = max(1, (registered_to - registered_from).days)

        rail_choices = (Rail.UPI_AUTOPAY, Rail.ENACH, Rail.CARD_ON_FILE)
        rail_weights = (0.52, 0.24, 0.24)

        mandates: list[Mandate] = []
        for i, customer in enumerate(customers):
            amount = int(
                np.clip(
                    rng.lognormal(
                        math.log(p.mean_cycle_amount_paise),
                        p.cycle_amount_log_sigma,
                    ),
                    5_000,
                    50_000_00,
                )
            )
            # Ceilings are usually set with headroom, but not always — a
            # too-tight ceiling is how a price rise turns into a permanent
            # AMOUNT_EXCEEDS_MANDATE failure every single cycle.
            ceiling = int(amount * float(rng.choice([1.0, 1.25, 2.0], p=[0.12, 0.33, 0.55])))
            registered = registered_from + dt.timedelta(
                days=int(rng.integers(0, span_days))
            )
            mandates.append(
                Mandate(
                    mandate_id=f"MND_{i:07d}",
                    customer_id=customer.customer_id,
                    rail=Rail(rng.choice(rail_choices, p=rail_weights)),
                    cycle_amount_paise=amount,
                    ceiling_paise=ceiling,
                    billing_day=int(rng.integers(1, 29)),
                    registered_on=registered,
                    valid_until=registered + dt.timedelta(days=int(rng.integers(180, 1100))),
                )
            )
        return mandates

    # -- the salary-day curve --------------------------------------------

    def funds_probability(self, customer: Customer, on: dt.date) -> float:
        """Probability the account is funded on this date.

        Flat and high for a few days after salary credit, then decaying
        exponentially toward a trough, scaled by the customer's latent balance
        health. This curve is the single most valuable thing in the dataset and
        the model is given no direct view of it.
        """
        p = self.params
        elapsed = _days_since_salary(customer.salary_day, on)
        overshoot = max(0, elapsed - p.funds_peak_window_days)
        decay = math.exp(-overshoot / p.funds_decay_days)
        base = p.funds_trough_probability + (
            p.funds_peak_probability - p.funds_trough_probability
        ) * decay
        scaled = base * (0.55 + customer.balance_health)
        return float(np.clip(scaled, 0.02, 0.97))

    def calibrate(
        self,
        customers: list[Customer],
        mandates: list[Mandate],
        sample_dates: list[dt.date],
        rounds: int = 6,
    ) -> None:
        """Fit per-rail scales so marginal failure rates match published anchors.

        The salary curve determines *who* fails and *when*. It does not, on its
        own, produce the right *overall* rate. Without reconciling the two, the
        choice is between a realistic headline number and a learnable
        conditional structure; this keeps both, which is the whole argument for
        the dataset being worth training on.

        Fitted **empirically** rather than solved in closed form. The closed
        form is tractable in principle and wrong in practice: failure draws
        compound (a debit that already failed NSF cannot also fail for another
        reason), and deterministic structural failures — expiry, ceiling
        breaches, closed execution windows — stack on top of both. Measuring
        the realised rate and adjusting absorbs every one of those interactions
        without needing to enumerate them.

        Runs on a dedicated RNG stream so calibration does not perturb the
        sampling sequence used for the actual dataset.
        """
        p = self.params
        rail_rates = {
            Rail.UPI_AUTOPAY: p.upi_failure_rate,
            Rail.ENACH: p.enach_failure_rate,
            Rail.CARD_ON_FILE: p.card_failure_rate,
        }
        by_customer = {c.customer_id: c for c in customers}
        pairs_by_rail: dict[Rail, list[tuple[Mandate, Customer]]] = {
            rail: [] for rail in Rail
        }
        for mandate in mandates:
            customer = by_customer.get(mandate.customer_id)
            if customer is not None:
                pairs_by_rail[mandate.rail].append((mandate, customer))

        self._nsf_scale = dict.fromkeys(Rail, 1.0)
        self._other_scale = dict.fromkeys(Rail, 1.0)

        for rail, target_rate in rail_rates.items():
            pairs = pairs_by_rail[rail]
            if not pairs:
                continue
            nsf_share = self._nsf_share(rail)
            target_nsf = target_rate * nsf_share
            target_other = target_rate - target_nsf

            for _ in range(rounds):
                nsf_rate, other_rate = self._measure(pairs, sample_dates)
                # Structural failures are deterministic and cannot be scaled
                # away, so if they already exceed the target the honest move is
                # to stop pushing rather than drive a scale negative.
                self._nsf_scale[rail] *= target_nsf / max(nsf_rate, 1e-6)
                if other_rate > 0:
                    self._other_scale[rail] = max(
                        0.0,
                        self._other_scale[rail] * target_other / max(other_rate, 1e-6),
                    )

            # Verification pass: measure once more without adjusting, so the
            # report describes the state the world is actually left in.
            # Reporting the last in-loop measurement instead would be one
            # adjustment behind — a number that was never true of the finished
            # simulator, published as though it were.
            nsf_rate, other_rate = self._measure(pairs, sample_dates)
            self.calibration_report[rail] = {
                "target_failure_rate": target_rate,
                "achieved_failure_rate": nsf_rate + other_rate,
                "target_nsf_rate": target_nsf,
                "achieved_nsf_rate": nsf_rate,
                "nsf_scale": self._nsf_scale[rail],
                "other_scale": self._other_scale[rail],
                "sample_size": len(pairs) * len(sample_dates),
            }

    def _measure(
        self,
        pairs: list[tuple[Mandate, Customer]],
        sample_dates: list[dt.date],
    ) -> tuple[float, float]:
        """Realised (NSF rate, non-NSF rate) over a sample, on the calibration RNG."""
        nsf_hits = other_hits = total = 0
        for mandate, customer in pairs:
            for on in sample_dates:
                at = dt.datetime.combine(on, dt.time(9, 0))
                code = self.sample_failure(
                    mandate, customer, at, rng=self._calibration_rng
                )
                total += 1
                if code is None:
                    continue
                if "INSUFFICIENT_FUNDS" in code:
                    nsf_hits += 1
                else:
                    other_hits += 1
        return nsf_hits / total, other_hits / total

    @staticmethod
    def _nsf_share(rail: Rail) -> float:
        mix = FAILURE_MIX[rail]
        total = sum(mix.values())
        nsf = sum(w for code, w in mix.items() if "INSUFFICIENT_FUNDS" in code)
        return nsf / total

    # -- failure generation ----------------------------------------------

    def sample_failure(
        self,
        mandate: Mandate,
        customer: Customer,
        at: dt.datetime,
        notification_sent: bool = True,
        rng: np.random.Generator | None = None,
    ) -> str | None:
        """Failure code for a debit presented at ``at``, or ``None`` if it succeeded.

        Order matters. Structural failures are checked first because they are
        deterministic given the mandate's state — a debit above its ceiling
        fails every time, not probabilistically. Only then does the
        balance-driven draw happen. This gives the model a mix of crisp
        learnable rules and genuinely stochastic signal, which is what real
        failure data looks like.
        """
        if not self._nsf_scale:
            raise RuntimeError(
                "World.calibrate() must be called before sampling failures, "
                "or marginal failure rates will not match their anchors."
            )

        rng = rng if rng is not None else self.rng

        # -- deterministic, structural -----------------------------------
        if at.date() > mandate.valid_until:
            return self._expiry_code(mandate.rail)

        if mandate.cycle_amount_paise > mandate.ceiling_paise:
            return self._ceiling_code(mandate.rail)

        if mandate.rail is Rail.CARD_ON_FILE and not notification_sent:
            return "CARD_PRE_DEBIT_NOTIFICATION_MISSING"

        if mandate.rail is Rail.UPI_AUTOPAY and not within_upi_execution_window(at):
            # Refused by the rail, not declined by the bank. Presents to the
            # merchant as an ordinary failure, which is precisely why merchants
            # misdiagnose it and retry straight back into the same closed window.
            return "UPI_PSP_UNAVAILABLE"

        # -- balance-driven ----------------------------------------------
        shortage = 1.0 - self.funds_probability(customer, at.date())
        if rng.random() < shortage * self._nsf_scale[mandate.rail]:
            return self._nsf_code(mandate.rail)

        # -- everything else ---------------------------------------------
        mix = FAILURE_MIX[mandate.rail]
        non_nsf = {
            code: w for code, w in mix.items() if "INSUFFICIENT_FUNDS" not in code
        }
        rail_rate = {
            Rail.UPI_AUTOPAY: self.params.upi_failure_rate,
            Rail.ENACH: self.params.enach_failure_rate,
            Rail.CARD_ON_FILE: self.params.card_failure_rate,
        }[mandate.rail]
        other_rate = (
            rail_rate
            * (1.0 - self._nsf_share(mandate.rail))
            * self._other_scale[mandate.rail]
        )
        if rng.random() >= other_rate:
            return None

        codes = list(non_nsf)
        weights = np.array([non_nsf[c] for c in codes], dtype=float)
        weights /= weights.sum()
        return str(rng.choice(codes, p=weights))

    @staticmethod
    def _nsf_code(rail: Rail) -> str:
        return {
            Rail.UPI_AUTOPAY: "UPI_INSUFFICIENT_FUNDS",
            Rail.ENACH: "NACH_INSUFFICIENT_FUNDS",
            Rail.CARD_ON_FILE: "CARD_INSUFFICIENT_FUNDS",
        }[rail]

    @staticmethod
    def _ceiling_code(rail: Rail) -> str:
        return {
            Rail.UPI_AUTOPAY: "UPI_AMOUNT_EXCEEDS_MANDATE",
            Rail.ENACH: "NACH_AMOUNT_EXCEEDS_MANDATE",
            Rail.CARD_ON_FILE: "CARD_AFA_REQUIRED",
        }[rail]

    @staticmethod
    def _expiry_code(rail: Rail) -> str:
        return {
            Rail.UPI_AUTOPAY: "UPI_MANDATE_EXPIRED",
            Rail.ENACH: "NACH_MANDATE_NOT_REGISTERED",
            Rail.CARD_ON_FILE: "CARD_TOKEN_INVALID",
        }[rail]

    # -- behavioural response --------------------------------------------

    def nudge_efficacy(self, episode: Episode, action: Action) -> float:
        """Probability the customer acts on this contact.

        Decays multiplicatively with every prior contact in the episode. The
        fourth message is not as good as the first, and modelling it as though
        it were is how a policy learns to spam.
        """
        p = self.params
        customer = episode.customer
        efficacy = p.nudge_base_efficacy * customer.engagement
        if action is customer.preferred_channel:
            efficacy *= p.channel_match_multiplier
        if action is Action.VOICE_CALL:
            efficacy *= p.voice_efficacy_multiplier
        efficacy *= p.contact_fatigue_decay**episode.contacts_made
        return float(np.clip(efficacy, 0.0, 0.95))

    def revocation_hazard(self, episode: Episode, action: Action) -> float:
        """Probability this contact causes the customer to revoke outright.

        Grows with prior contacts, exactly as efficacy decays. That pairing is
        the heart of the problem: the marginal nudge is always less likely to
        work and more likely to destroy the mandate than the one before it.
        """
        p = self.params
        intrusiveness = CHANNEL_INTRUSIVENESS.get(action, 0.0)
        if intrusiveness == 0.0:
            return 0.0
        mean_churn_intent = p.churn_intent_alpha / (
            p.churn_intent_alpha + p.churn_intent_beta
        )
        relative_intent = episode.customer.churn_intent / mean_churn_intent
        hazard = (
            p.revocation_base_rate
            * intrusiveness
            * relative_intent
            * p.revocation_fatigue_growth**episode.contacts_made
        )
        return float(np.clip(hazard, 0.0, 0.5))

    # -- the step function ------------------------------------------------

    def apply(self, episode: Episode, action: Action, at: dt.datetime) -> ActionOutcome:
        """Take one action on an open episode and return what happened.

        Mutates ``episode`` and appends to its history. This is the only place
        an episode's state changes.
        """
        if episode.closed:
            raise RuntimeError(
                f"episode {episode.episode_id} is already closed; "
                f"the sequencer should not be acting on it"
            )

        cost = attempt_cost_paise(action, episode.mandate.rail)

        if action is Action.STOP:
            episode.stopped = True
            return self._record(
                episode, action, at, False, 0, 0, False, 0, "gave up",
                counts_as_attempt=False,
            )

        # Contact-driven actions carry revocation risk before anything else.
        if action in NUDGE_ACTIONS or action in _REPAIR_ACTIONS:
            if self.rng.random() < self.revocation_hazard(episode, action):
                destroyed = revocation_cost_paise(
                    episode.mandate.cycle_amount_paise
                )
                episode.revoked = True
                episode.contacts_made += 1
                return self._record(
                    episode,
                    action,
                    at,
                    False,
                    0,
                    cost,
                    True,
                    destroyed,
                    "customer revoked the mandate after being contacted",
                )

        if action in _RETRY_ACTIONS:
            return self._apply_retry(episode, action, at, cost)
        if action is Action.SEND_PRE_DEBIT_NOTIFICATION:
            episode.notification_sent_at = at
            return self._record(
                episode, action, at, False, 0, cost, False, 0,
                "pre-debit notification sent; debit presentable after the window",
            )
        if action is Action.SEND_COLLECT_LINK:
            return self._apply_collect_link(episode, action, at, cost)
        if action in _REPAIR_ACTIONS:
            return self._apply_repair(episode, action, at, cost)
        if action in NUDGE_ACTIONS:
            return self._apply_nudge(episode, action, at, cost)

        raise ValueError(f"no outcome model for action {action!r}")

    # -- action handlers --------------------------------------------------

    def _apply_retry(
        self, episode: Episode, action: Action, at: dt.datetime, cost: int
    ) -> ActionOutcome:
        mandate = episode.mandate
        rail = mandate.rail
        if action is Action.RETRY_ALT_RAIL:
            if not episode.customer.has_alt_rail:
                return self._record(
                    episode, action, at, False, 0, cost, False, 0,
                    "no mandate registered on an alternate rail",
                )
            rail = next(
                r for r in (Rail.UPI_AUTOPAY, Rail.CARD_ON_FILE, Rail.ENACH)
                if r is not mandate.rail
            )

        if rail is Rail.UPI_AUTOPAY and not within_upi_execution_window(at):
            return self._record(
                episode, action, at, False, 0, cost, False, 0,
                "presented outside the permitted UPI execution window",
            )

        mode = get_mode(episode.failure_code)
        if not mode.mandate_alive and not episode.mandate_repaired:
            return self._record(
                episode, action, at, False, 0, cost, False, 0,
                "mandate is not live; presentation cannot succeed",
            )

        disposition = episode.disposition
        if disposition is Disposition.MERCHANT_FIX:
            sent = episode.notification_sent_at
            elapsed_ok = sent is not None and (
                at - sent >= dt.timedelta(hours=PRE_DEBIT_NOTIFICATION_HOURS)
            )
            if not elapsed_ok:
                return self._record(
                    episode, action, at, False, 0, cost, False, 0,
                    "pre-debit notification not yet sent or window not elapsed",
                )
        elif disposition is Disposition.RETRY_TRANSIENT:
            if episode.outage_ends_at is None:
                episode.outage_ends_at = self._sample_outage_end(episode)
            if at < episode.outage_ends_at:
                return self._record(
                    episode, action, at, False, 0, cost, False, 0,
                    "upstream outage has not cleared",
                )
        elif disposition is Disposition.CUSTOMER_ACTION and not episode.customer_unblocked:
            return self._record(
                episode, action, at, False, 0, cost, False, 0,
                "customer has not yet cleared the block",
            )
        elif disposition is Disposition.MANDATE_REPAIR and not episode.mandate_repaired:
            return self._record(
                episode, action, at, False, 0, cost, False, 0,
                "mandate still defective; re-presentation fails identically",
            )

        funds_p = self.funds_probability(episode.customer, at.date())
        if episode.customer_unblocked and disposition is Disposition.RETRY_TIMING:
            # A customer who responded to a nudge about a failed payment
            # generally responds by topping up. Without this, nudging does
            # nothing for insufficient funds — the largest failure class on
            # every rail — and the evaluation would conclude that the main
            # lever merchants actually have is worthless.
            funds_p += (1.0 - funds_p) * self.params.nudge_top_up_lift

        if self.rng.random() >= funds_p:
            return self._record(
                episode, action, at, False, 0, cost, False, 0,
                "account still not funded",
            )

        episode.resolved = True
        return self._record(
            episode, action, at, True, mandate.cycle_amount_paise, cost, False, 0,
            "debit succeeded on re-presentation",
        )

    def _apply_nudge(
        self, episode: Episode, action: Action, at: dt.datetime, cost: int
    ) -> ActionOutcome:
        acted = self.rng.random() < self.nudge_efficacy(episode, action)
        episode.contacts_made += 1
        if acted:
            episode.customer_unblocked = True
            return self._record(
                episode, action, at, False, 0, cost, False, 0,
                "customer acknowledged; the block is cleared for the next attempt",
            )
        return self._record(
            episode, action, at, False, 0, cost, False, 0, "no response"
        )

    def _apply_collect_link(
        self, episode: Episode, action: Action, at: dt.datetime, cost: int
    ) -> ActionOutcome:
        """A one-time payment link, which sidesteps the mandate entirely.

        The only action that recovers money from a structurally dead mandate,
        and therefore the only thing standing between a MANDATE_REPAIR failure
        and a total write-off.
        """
        paid = self.rng.random() < self.nudge_efficacy(episode, action) * (
            0.55 + 0.45 * self.funds_probability(episode.customer, at.date())
        )
        episode.contacts_made += 1
        if paid:
            episode.resolved = True
            return self._record(
                episode, action, at, True, episode.mandate.cycle_amount_paise,
                cost, False, 0, "customer paid via collect link",
            )
        return self._record(
            episode, action, at, False, 0, cost, False, 0, "collect link not paid"
        )

    def _apply_repair(
        self, episode: Episode, action: Action, at: dt.datetime, cost: int
    ) -> ActionOutcome:
        # Re-mandating is slower and more effortful than tapping a link, so it
        # converts at a fraction of ordinary nudge efficacy.
        repaired = self.rng.random() < self.nudge_efficacy(episode, action) * 0.7
        episode.contacts_made += 1
        if repaired:
            episode.mandate_repaired = True
            return self._record(
                episode, action, at, False, 0, cost, False, 0,
                "mandate repaired; the debit is presentable again",
            )
        return self._record(
            episode, action, at, False, 0, cost, False, 0,
            "customer did not complete the mandate repair",
        )

    def _sample_outage_end(self, episode: Episode) -> dt.datetime:
        hours = (
            self.params.outage_mean_hours
            * self._bank_outage_factor[episode.customer.bank]
        )
        drawn = float(self.rng.exponential(max(0.25, hours)))
        return episode.failed_at + dt.timedelta(hours=drawn)

    # -- bookkeeping ------------------------------------------------------

    def _record(
        self,
        episode: Episode,
        action: Action,
        at: dt.datetime,
        succeeded: bool,
        recovered: int,
        cost: int,
        revoked: bool,
        destroyed: int,
        detail: str,
        counts_as_attempt: bool = True,
    ) -> ActionOutcome:
        ledger = episode.ledger.plus_cost(cost, counts_as_attempt)
        if recovered:
            ledger = ledger.plus_recovery(recovered)
        if destroyed:
            ledger = ledger.plus_destruction(destroyed)
        episode.ledger = ledger

        outcome = ActionOutcome(
            action=action,
            at=at,
            succeeded=succeeded,
            recovered_paise=recovered,
            cost_paise=cost,
            revoked=revoked,
            destroyed_paise=destroyed,
            detail=detail,
        )
        episode.history.append(outcome)
        return outcome

    # -- convenience ------------------------------------------------------

    def open_episode(
        self,
        episode_id: str,
        mandate: Mandate,
        customer: Customer,
        failure_code: str,
        failed_at: dt.datetime,
        cycles_elapsed: int,
    ) -> Episode:
        return Episode(
            episode_id=episode_id,
            mandate=mandate,
            customer=customer,
            failure_code=failure_code,
            failed_at=failed_at,
            cycles_elapsed=cycles_elapsed,
        )


def with_amount(mandate: Mandate, cycle_amount_paise: int) -> Mandate:
    """A copy of the mandate at a different cycle amount.

    Used to model price changes, which is how a healthy mandate crosses its own
    ceiling or the AFA-exempt threshold and starts failing every cycle.
    """
    return replace(mandate, cycle_amount_paise=cycle_amount_paise)
