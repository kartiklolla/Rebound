"""Baseline policies.

Four of these, spanning from "do nothing" to "chase relentlessly", plus one
that is genuinely good.

``DispositionAwareRules`` is the baseline that matters. It is not a strawman —
it reads the failure taxonomy, stops on terminal codes, sends the notification
on merchant-side defects, nudges before retrying when the customer is the
blocker, and times retries toward the start of the month. A competent engineer
with a week and no machine learning would write approximately this.

Including it is the point. Beating ``FixedLadder`` proves very little, since
that policy retries revoked mandates. If the learned sequencer cannot beat a
careful set of hand-written rules, the honest conclusion is that this problem
did not need a model, and it is far better to discover that in week one than to
find out in front of a panel.
"""

from __future__ import annotations

import datetime as dt

from rebound.policy import Decision, Policy
from rebound.sim.params import MAX_RETRIES_PER_CYCLE, PRE_DEBIT_NOTIFICATION_HOURS
from rebound.sim.world import EpisodeView, within_upi_execution_window
from rebound.taxonomy import Action, Disposition, Rail, get_mode


def _next_upi_window(at: dt.datetime) -> dt.datetime:
    """Advance to the next moment a UPI execution may legally be presented.

    Steps in fifteen-minute increments rather than solving the window algebra.
    The windows are a handful of intervals in a day, so the loop terminates
    almost immediately, and it stays correct if the windows are ever changed.
    """
    probe = at
    for _ in range(4 * 24 * 2):
        if within_upi_execution_window(probe):
            return probe
        probe += dt.timedelta(minutes=15)
    return at


class NoRecovery(Policy):
    """Do nothing. The floor.

    Establishes what the failures are worth if simply abandoned, which is the
    only way to read any other policy's numbers. Without it, "recovered ₹X" has
    no denominator.
    """

    name = "no_recovery"
    description = "Abandon every failed debit immediately."

    def decide(self, episode, now, deadline):
        return None


class FixedLadder(Policy):
    """Retry at +1, +3, +7 days. The production standard, and the thing to beat.

    Deliberately blind to the failure code, because that is what the real
    version is. It will cheerfully re-present a revoked mandate three times.
    """

    name = "fixed_ladder"
    description = "Retry at +1/+3/+7 days regardless of why the debit failed."

    def __init__(self, delays_days: tuple[int, ...] = (1, 3, 7)) -> None:
        self.delays_days = delays_days

    def decide(self, episode, now, deadline):
        step = episode.attempts
        if step >= len(self.delays_days):
            return None
        at = episode.failed_at + dt.timedelta(days=self.delays_days[step])
        if at <= now:
            at = now + dt.timedelta(hours=1)
        if episode.rail is Rail.UPI_AUTOPAY:
            at = _next_upi_window(at)
        if at > deadline:
            return None
        return Decision(
            action=Action.RETRY_SAME_RAIL,
            at=at,
            reason=f"fixed ladder step {step + 1} of {len(self.delays_days)}",
        )


class ImmediateRetry(Policy):
    """Retry hard and fast, up to the per-cycle cap.

    The instinct that a failed payment should be retried straight away. Useful
    as a baseline because it is wrong in a specific, measurable way: it burns
    the cycle's permitted attempts within hours of a failure, before anything
    about the customer's situation has had a chance to change.
    """

    name = "immediate_retry"
    description = "Retry every 6 hours until the per-cycle attempt cap is reached."

    def __init__(self, gap_hours: int = 6, max_attempts: int = MAX_RETRIES_PER_CYCLE):
        self.gap_hours = gap_hours
        self.max_attempts = max_attempts

    def decide(self, episode, now, deadline):
        if episode.attempts >= self.max_attempts:
            return None
        at = now + dt.timedelta(hours=self.gap_hours)
        if episode.rail is Rail.UPI_AUTOPAY:
            at = _next_upi_window(at)
        if at > deadline:
            return None
        return Decision(
            action=Action.RETRY_SAME_RAIL,
            at=at,
            reason=f"immediate retry {episode.attempts + 1}",
        )


class AggressiveContact(Policy):
    """Contact on every channel, then retry. Maximise recovery rate at any cost.

    Included precisely because it wins on the metric merchants usually watch.
    It recovers more debits than anything else here and destroys value doing
    it, because every contact carries revocation risk and it never stops.

    Its role in the results table is to demonstrate that recovery rate is the
    wrong headline, and that a policy optimising it will happily burn the book.
    """

    name = "aggressive_contact"
    description = "Nudge on every channel, escalate to voice, then retry. No restraint."

    _LADDER: tuple[Action, ...] = (
        Action.NUDGE_SMS,
        Action.NUDGE_WHATSAPP,
        Action.VOICE_CALL,
        Action.RETRY_SAME_RAIL,
        Action.SEND_COLLECT_LINK,
    )

    def decide(self, episode, now, deadline):
        step = episode.steps_taken
        if step >= len(self._LADDER):
            return None
        action = self._LADDER[step]
        at = now + dt.timedelta(days=1)
        if action is Action.RETRY_SAME_RAIL and episode.rail is Rail.UPI_AUTOPAY:
            at = _next_upi_window(at)
        if at > deadline:
            return None
        return Decision(action=action, at=at, reason=f"aggressive step {step + 1}")


class DispositionAwareRules(Policy):
    """Hand-written rules that actually read the failure code. The real baseline.

    What a careful engineer would build in a week without machine learning:

    * stop immediately on terminal codes rather than paying to confirm they are
      terminal
    * send the pre-debit notification when the failure was our own compliance
      defect, wait the mandated window, re-present
    * nudge before retrying when the customer is the blocker, because
      re-presenting against an unchanged state cannot work
    * ask for a collect link when the mandate is structurally dead but the
      customer is not
    * on insufficient funds, aim the retry at the start of the month rather
      than a fixed number of days later

    The last rule is the interesting one. It encodes the salary-cycle intuition
    as a hard rule for everybody. The learned model's advantage, if it has one,
    should come from knowing that different customers are paid on different
    days — which is precisely what a rule cannot express.
    """

    name = "disposition_rules"
    description = "Hand-written rules driven by the failure taxonomy."

    def __init__(self, max_steps: int = 4) -> None:
        self.max_steps = max_steps

    def decide(self, episode, now, deadline):
        if episode.steps_taken >= self.max_steps:
            return None

        mode = get_mode(episode.failure_code)
        disposition = mode.disposition

        if disposition is Disposition.TERMINAL:
            return Decision(
                action=Action.STOP,
                at=now,
                reason=f"{episode.failure_code} is terminal; further spend cannot recover",
            )

        decision = self._decide_by_disposition(episode, now, disposition)
        if decision is None:
            return None
        if decision.at > deadline:
            return None
        return decision

    def _decide_by_disposition(
        self, episode: EpisodeView, now: dt.datetime, disposition: Disposition
    ) -> Decision | None:
        if disposition is Disposition.MERCHANT_FIX:
            return self._merchant_fix(episode, now)
        if disposition is Disposition.RETRY_TRANSIENT:
            return self._transient(episode, now)
        if disposition is Disposition.CUSTOMER_ACTION:
            return self._customer_action(episode, now)
        if disposition is Disposition.MANDATE_REPAIR:
            return self._mandate_repair(episode, now)
        return self._retry_timing(episode, now)

    def _merchant_fix(self, episode: EpisodeView, now: dt.datetime) -> Decision | None:
        if episode.notification_sent_at is None:
            return Decision(
                action=Action.SEND_PRE_DEBIT_NOTIFICATION,
                at=now + dt.timedelta(hours=1),
                reason="our own notification defect caused this; send it",
            )
        ready = episode.notification_sent_at + dt.timedelta(
            hours=PRE_DEBIT_NOTIFICATION_HOURS + 1
        )
        # No UPI window adjustment here, unlike every sibling handler. The only
        # MERCHANT_FIX code in the taxonomy is a card failure, so this branch is
        # only ever reached on card_on_file and a UPI check could not fire. It
        # was there, and an exhaustive sweep found it unreachable; leaving dead
        # code that looks load-bearing is how a reader learns to distrust the
        # rest. If a MERCHANT_FIX code is ever added on UPI, the assertion below
        # is what will say so.
        assert episode.rail is not Rail.UPI_AUTOPAY, (
            "a MERCHANT_FIX failure on UPI now exists; this handler needs the "
            "execution-window adjustment its siblings have"
        )
        return Decision(
            action=Action.RETRY_SAME_RAIL,
            at=max(now, ready),
            reason="notification window elapsed; re-present",
        )

    def _transient(self, episode: EpisodeView, now: dt.datetime) -> Decision | None:
        if episode.attempts >= 2:
            return None
        at = now + dt.timedelta(hours=12)
        if episode.rail is Rail.UPI_AUTOPAY:
            at = _next_upi_window(at)
        return Decision(
            action=Action.RETRY_SAME_RAIL,
            at=at,
            reason="upstream outage; wait it out and re-present",
        )

    def _customer_action(self, episode: EpisodeView, now: dt.datetime) -> Decision | None:
        if not episode.customer_unblocked:
            if episode.contacts_made >= 2:
                return Decision(
                    action=Action.STOP,
                    at=now,
                    reason="customer has not responded to two contacts",
                )
            action = (
                Action.NUDGE_WHATSAPP
                if episode.contacts_made == 0
                else Action.NUDGE_SMS
            )
            return Decision(
                action=action,
                at=now + dt.timedelta(days=1),
                reason="only the customer can clear this; retrying cannot help",
            )
        at = now + dt.timedelta(hours=6)
        if episode.rail is Rail.UPI_AUTOPAY:
            at = _next_upi_window(at)
        return Decision(
            action=Action.RETRY_SAME_RAIL,
            at=at,
            reason="customer responded; re-present",
        )

    def _mandate_repair(self, episode: EpisodeView, now: dt.datetime) -> Decision | None:
        if episode.contacts_made >= 2:
            return Decision(
                action=Action.STOP,
                at=now,
                reason="mandate is unrepaired after two attempts",
            )
        if episode.contacts_made == 0:
            return Decision(
                action=Action.SEND_COLLECT_LINK,
                at=now + dt.timedelta(days=1),
                reason="mandate is dead but the customer is not; collect directly",
            )
        return Decision(
            action=Action.REQUEST_REMANDATE,
            at=now + dt.timedelta(days=2),
            reason="restore the mandate so future cycles do not fail identically",
        )

    def _retry_timing(self, episode: EpisodeView, now: dt.datetime) -> Decision | None:
        """Insufficient funds: aim at the start of the month, not a fixed delay.

        A single population-level rule, because a rule cannot know that this
        particular customer is paid on the 22nd. That limitation is exactly
        where a learned model should earn its place.
        """
        if episode.attempts >= 2:
            if episode.contacts_made == 0:
                return Decision(
                    action=Action.NUDGE_SMS,
                    at=now + dt.timedelta(days=1),
                    reason="two retries failed; ask the customer to top up",
                )
            return None

        at = _start_of_next_month(now) if now.day > 7 else now + dt.timedelta(days=2)
        if episode.rail is Rail.UPI_AUTOPAY:
            at = _next_upi_window(at)
        return Decision(
            action=Action.RETRY_SAME_RAIL,
            at=at,
            reason="balances recover around salary credit; wait for it",
        )


def _start_of_next_month(at: dt.datetime) -> dt.datetime:
    year = at.year + (at.month // 12)
    month = at.month % 12 + 1
    return dt.datetime(year, month, 1, 9, 0)


def default_baselines() -> list[Policy]:
    """Every baseline, in the order they belong in the results table."""
    return [
        NoRecovery(),
        FixedLadder(),
        ImmediateRetry(),
        AggressiveContact(),
        DispositionAwareRules(),
    ]
