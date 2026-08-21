"""What each recovery action costs, in paise.

Boundary note
-------------
This module owns **costs**. It does not own **probabilities**.

The distinction matters and is easy to blur. "A revoked mandate destroys the
remaining lifetime value of the subscription" is the merchant's own business
knowledge — the policy is entitled to reason with it. "A voice call has a 4%
chance of causing a revocation" is the simulator's answer key, and a policy that
reads it is cheating.

So: cost-if-revoked lives here. Probability-of-revocation lives in
``rebound.sim`` and must be learned by ``rebound.model`` from observed data.

Money is integer paise throughout. Recovery economics turn on differences of a
few rupees across millions of attempts, and float drift at that scale produces
totals that quietly disagree with themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

from rebound.taxonomy import Action, Rail

RUPEE = 100
"""Paise per rupee."""


# --------------------------------------------------------------------------
# Per-attempt rail costs
# --------------------------------------------------------------------------

#: Cost of presenting one debit attempt, by rail. A failed presentation costs
#: the same as a successful one — which is the entire reason retrying a dead
#: mandate is expensive rather than merely useless.
#:
#: Indicative processing costs, not published tariffs; every merchant's
#: negotiated rate differs. What the policy is actually sensitive to is the
#: *ordering* (NACH presentation costs meaningfully more than a UPI execution),
#: and that ordering is stable across merchants.
RAIL_ATTEMPT_COST_PAISE: dict[Rail, int] = {
    Rail.UPI_AUTOPAY: 50,
    Rail.CARD_ON_FILE: 300,
    Rail.ENACH: 500,
}


# --------------------------------------------------------------------------
# Per-action costs
# --------------------------------------------------------------------------

#: Direct cost of each non-retry action. Retry costs are rail-dependent and
#: come from ``RAIL_ATTEMPT_COST_PAISE`` instead.
ACTION_COST_PAISE: dict[Action, int] = {
    Action.SEND_PRE_DEBIT_NOTIFICATION: 20,
    Action.NUDGE_EMAIL: 2,
    Action.NUDGE_SMS: 20,
    Action.NUDGE_WHATSAPP: 55,
    Action.SEND_COLLECT_LINK: 55,
    Action.VOICE_CALL: 350,
    Action.REQUEST_MANDATE_AMENDMENT: 200,
    Action.REQUEST_REMANDATE: 200,
    Action.STOP: 0,
}


#: How intrusive each customer-facing action feels, on a 0–1 scale.
#:
#: Not a cost in rupees — it is the input to revocation risk. An email is
#: ignorable; a phone call at dinner is an event. The simulator turns this into
#: revocation probability; the policy only sees the ordering.
CHANNEL_INTRUSIVENESS: dict[Action, float] = {
    Action.NUDGE_EMAIL: 0.10,
    Action.SEND_PRE_DEBIT_NOTIFICATION: 0.10,
    Action.NUDGE_SMS: 0.30,
    Action.NUDGE_WHATSAPP: 0.45,
    Action.SEND_COLLECT_LINK: 0.45,
    Action.REQUEST_MANDATE_AMENDMENT: 0.55,
    Action.REQUEST_REMANDATE: 0.60,
    Action.VOICE_CALL: 1.00,
}


#: Expected remaining life of a surviving mandate, in billing cycles.
#:
#: A revoked monthly subscription does not cost one cycle's revenue, it costs
#: every cycle that would have followed. Twelve is a deliberately conservative
#: stand-in for an indefinite subscription — long enough that revocation
#: dominates the cost matrix (which is true), short enough that the number is
#: defensible rather than rhetorical.
#:
#: Read as *remaining* life, not total life. See ``revocation_cost_paise``.
LTV_HORIZON_CYCLES = 12


def attempt_cost_paise(action: Action, rail: Rail) -> int:
    """Direct cost of taking ``action`` on a mandate presented over ``rail``.

    Retries are priced by rail; everything else by channel.
    """
    if action in (Action.RETRY_SAME_RAIL, Action.RETRY_ALT_RAIL):
        return RAIL_ATTEMPT_COST_PAISE[rail]
    return ACTION_COST_PAISE[action]


def revocation_cost_paise(
    cycle_amount_paise: int,
    horizon: int = LTV_HORIZON_CYCLES,
) -> int:
    """Value destroyed when a mandate is revoked.

    The revenue of the cycles that would have followed, undiscounted.

    The horizon is **memoryless**: a mandate's remaining life does not tick
    down from its registration date. An earlier version of this function
    computed ``horizon - cycles_elapsed``, which implied that a mandate twelve
    cycles old had no value left and could be churned for free. That is exactly
    backwards — a customer who has paid reliably for a year is more likely to
    keep paying, not less — and a policy optimising against it would learn to
    aim its riskiest actions at the most loyal customers on the book.

    Undiscounted on purpose. A discount rate would be a second assumption
    layered on the horizon assumption, and it moves the answer far less than
    the horizon does.

    Raises on a non-positive amount rather than returning a negative cost. A
    single negative amount — one bad row in a merchant's export — otherwise
    propagates into ``destroyed_paise`` as a *credit*, which flips every
    policy's net positive and reports the do-nothing floor as having earned
    money. Found by red-teaming; see docs/SECURITY_REVIEW.md.
    """
    if cycle_amount_paise <= 0:
        raise ValueError(
            f"cycle_amount_paise must be positive, got {cycle_amount_paise}. "
            f"A non-positive amount turns destroyed value into a credit and "
            f"silently inverts every reported net."
        )
    return cycle_amount_paise * max(0, horizon)


def _reject_negative(kind: str, paise: int) -> None:
    """Guard every ledger entry against negative money.

    A negative cost is a rebate, a negative recovery is a refund, and a
    negative destruction is value created out of nothing. None of them are
    things this system can legitimately book, and all three silently improve
    whatever total they land in — which makes them the highest-leverage way to
    make the reported numbers lie.
    """
    if paise < 0:
        raise ValueError(
            f"{kind} must be non-negative, got {paise}. Negative entries "
            f"improve the totals they land in and cannot be legitimate here."
        )


@dataclass(frozen=True, slots=True)
class Ledger:
    """Running economics of one recovery episode.

    Kept as an explicit object rather than loose totals so that every rupee in
    the final report can be traced to the action that produced it. The audit
    trail is a track requirement, and reconstructing costs afterwards from
    aggregates is how numbers stop reconciling.
    """

    recovered_paise: int = 0
    spent_paise: int = 0
    destroyed_paise: int = 0
    """Lifetime value lost to revocations we caused."""

    attempts: int = 0

    def plus_cost(self, paise: int, counts_as_attempt: bool = True) -> Ledger:
        """Book a cost.

        ``counts_as_attempt=False`` for actions that spend nothing and reach
        nobody — deciding to stop, most importantly. Counting a decision to
        give up as an attempt inflates every per-attempt metric in the
        evaluation, and inflates it *most* for the policies that correctly give
        up early, which is exactly backwards.
        """
        _reject_negative("cost", paise)
        return Ledger(
            recovered_paise=self.recovered_paise,
            spent_paise=self.spent_paise + paise,
            destroyed_paise=self.destroyed_paise,
            attempts=self.attempts + (1 if counts_as_attempt else 0),
        )

    def plus_recovery(self, paise: int) -> Ledger:
        _reject_negative("recovery", paise)
        return Ledger(
            recovered_paise=self.recovered_paise + paise,
            spent_paise=self.spent_paise,
            destroyed_paise=self.destroyed_paise,
            attempts=self.attempts,
        )

    def plus_destruction(self, paise: int) -> Ledger:
        _reject_negative("destruction", paise)
        return Ledger(
            recovered_paise=self.recovered_paise,
            spent_paise=self.spent_paise,
            destroyed_paise=self.destroyed_paise + paise,
            attempts=self.attempts,
        )

    @property
    def net_paise(self) -> int:
        """Money recovered, less what it cost and less what it destroyed.

        This is the number the policy optimises and the number the README
        reports. Reporting ``recovered_paise`` alone is the flattering version:
        it makes "call everyone twice a day" look like a triumph.
        """
        return self.recovered_paise - self.spent_paise - self.destroyed_paise

    def __add__(self, other: Ledger) -> Ledger:
        return Ledger(
            recovered_paise=self.recovered_paise + other.recovered_paise,
            spent_paise=self.spent_paise + other.spent_paise,
            destroyed_paise=self.destroyed_paise + other.destroyed_paise,
            attempts=self.attempts + other.attempts,
        )


EMPTY_LEDGER = Ledger()
