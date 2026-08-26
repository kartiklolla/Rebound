"""External constraints on recurring debits: rail rules and regulator rules.

Separated from ``rebound.sim.params`` deliberately. These constants used to sit
at the top of the simulator's parameter file, under a heading that already
admitted they were a different kind of thing — "facts about the world, not
modelling choices". Leaving them there meant the compliance gate, which is
production logic, would have to import from the simulator to find out what the
law is. That dependency runs the wrong way: delete the simulator and the law
does not change.

The distinction the rest of the system relies on:

**A simulator parameter is an opinion.** ``upi_marginal_failure_rate`` is the
generator's guess. Get it wrong and the measured numbers shift.

**A regulatory constant is a claim about the world.** ``PRE_DEBIT_NOTIFICATION_HOURS``
is either what the RBI e-mandate framework requires or it is wrong. Get it wrong
and the system does something non-compliant while reporting that it complied.
The second failure is worse and is not detectable by any amount of testing
against our own simulator, because the simulator would be wrong in the same
direction.

Verification status
-------------------
**Every constant in this module is currently unverified against a primary
source.** They come from secondary reporting and are recorded here as claims to
be confirmed, not as established fact. ``REGULATORY_SOURCES`` carries the status
of each one, and the compliance gate reports the status of every rule it applies
so that no audit trail can imply a confirmation that was never obtained.

This is the honest position given where the project is, and it is a better one
than asserting a number confidently because it appeared in a blog post. The
work of confirming them against the circulars is tracked in PROGRESS.md.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum

# ==========================================================================
# Provenance of the constants themselves
# ==========================================================================


class SourceStatus(StrEnum):
    """How well a regulatory claim is evidenced."""

    CONFIRMED = "confirmed"
    """Checked against the primary circular or scheme rulebook."""

    REPORTED = "reported"
    """From secondary sources that agree with each other. Not yet confirmed."""

    DISPUTED = "disputed"
    """Secondary sources disagree. The encoded value is a choice between them."""


@dataclass(frozen=True, slots=True)
class RegulatorySource:
    """Where a constant came from and how much weight it can carry."""

    name: str
    authority: str
    status: SourceStatus
    note: str

    @property
    def is_confirmed(self) -> bool:
        return self.status is SourceStatus.CONFIRMED


# ==========================================================================
# The constants
# ==========================================================================

#: Windows in which a UPI Autopay mandate execution may be presented.
#:
#: NPCI restricts recurring executions to defined windows so that mandate
#: traffic does not collide with the morning peak. Presenting outside them is
#: not a bad idea that might work anyway — it is refused by the rail. This is
#: what makes retry *timing* a compliance question in this domain rather than
#: purely an optimisation problem.
UPI_EXECUTION_WINDOWS: tuple[tuple[dt.time, dt.time], ...] = (
    (dt.time(0, 0), dt.time(10, 0)),
    (dt.time(13, 0), dt.time(17, 0)),
    (dt.time(21, 30), dt.time(23, 59, 59)),
)

#: Hours of advance notice required before debiting under the RBI e-mandate
#: framework. Missing it blocks the debit outright.
PRE_DEBIT_NOTIFICATION_HOURS = 24

#: Executions permitted per mandate per billing cycle: one presentation plus
#: retries. The cap is the reason "just retry more" is not an available
#: strategy, and therefore the reason choosing *which* attempts to spend has
#: value at all.
MAX_EXECUTIONS_PER_CYCLE = 1
MAX_RETRIES_PER_CYCLE = 3

#: Ceiling below which a recurring debit needs no additional factor of
#: authentication. Above it the customer must authenticate every cycle, which
#: is why a price rise across this line converts a healthy mandate into a
#: recurring AFA failure.
AFA_EXEMPT_CEILING_PAISE = 15_000 * 100

#: Local hours during which the system will not contact a customer.
#:
#: Not a regulatory constant, and deliberately kept in this module next to the
#: ones that are, so the contrast is visible at the point of use. India has no
#: single statutory quiet-hours rule binding on a merchant's own transactional
#: messaging in the way this system uses it; TRAI's commercial-communication
#: regulations govern a related but distinct surface. This window is house
#: policy chosen to be defensible, and the gate labels it as such rather than
#: dressing restraint up as compliance.
QUIET_HOURS_START = dt.time(21, 0)
QUIET_HOURS_END = dt.time(9, 0)


#: The evidentiary status of each claim above. Machine-checked by
#: ``test_every_regulatory_constant_has_a_source``, so a new constant cannot be
#: added without declaring where it came from.
REGULATORY_SOURCES: dict[str, RegulatorySource] = {
    "UPI_EXECUTION_WINDOWS": RegulatorySource(
        name="NPCI UPI Autopay execution windows",
        authority="NPCI",
        status=SourceStatus.DISPUTED,
        note=(
            "Sources disagree. Some describe a single post-21:30 window; more "
            "detailed coverage describes the three encoded here, consistent "
            "with peak hours being 10:00-13:00. The three-window version is "
            "encoded because it is the stricter reading in the middle of the "
            "day and the more permissive one overall; if it is wrong, the "
            "gate defers executions it need not defer, which is the safe "
            "direction to be wrong in."
        ),
    ),
    "PRE_DEBIT_NOTIFICATION_HOURS": RegulatorySource(
        name="RBI framework for processing e-mandates on recurring transactions",
        authority="RBI",
        status=SourceStatus.REPORTED,
        note="24 hours is consistently reported. Not yet read in the circular.",
    ),
    "MAX_EXECUTIONS_PER_CYCLE": RegulatorySource(
        name="NACH/UPI Autopay presentation limits",
        authority="NPCI",
        status=SourceStatus.REPORTED,
        note="One original presentation per cycle.",
    ),
    "MAX_RETRIES_PER_CYCLE": RegulatorySource(
        name="NACH/UPI Autopay retry limits",
        authority="NPCI",
        status=SourceStatus.REPORTED,
        note=(
            "Up to three retries after the original presentation. Scheme "
            "rules and sponsor-bank policy can both bind here and the "
            "stricter applies, so this is an upper bound rather than an "
            "entitlement."
        ),
    ),
    "AFA_EXEMPT_CEILING_PAISE": RegulatorySource(
        name="RBI AFA exemption ceiling for recurring transactions",
        authority="RBI",
        status=SourceStatus.REPORTED,
        note=(
            "Revised upward more than once, and category-specific carve-outs "
            "exist. The current figure is reported as Rs 15,000. A stale value "
            "here fails in the dangerous direction: the gate would allow an "
            "un-authenticated debit above the real ceiling."
        ),
    ),
}


# ==========================================================================
# Window arithmetic
# ==========================================================================


def within_upi_execution_window(at: dt.datetime) -> bool:
    """Whether a UPI Autopay execution may be presented at this moment.

    Outside these windows the presentation is refused by the rail. Not a
    heuristic — a hard constraint, and the reason retry timing here is a
    compliance question and not only an optimisation.
    """
    clock = at.time()
    return any(start <= clock <= end for start, end in UPI_EXECUTION_WINDOWS)


def next_execution_window_open(at: dt.datetime) -> dt.datetime:
    """The earliest moment at or after ``at`` when an execution may be presented.

    Returns ``at`` unchanged if it is already inside a window, so callers can
    use this unconditionally.

    The gate needs this because refusing an out-of-window retry is only half an
    answer. "Denied" tells a sequencer to give up on an action that is perfectly
    legal four hours from now; "not before 13:00" tells it to wait. A compliance
    layer that can only say no turns every timing rule into a lost recovery.
    """
    _require_naive(at, "next_execution_window_open")
    if within_upi_execution_window(at):
        return at

    clock = at.time()
    for start, _ in UPI_EXECUTION_WINDOWS:
        if clock < start:
            return dt.datetime.combine(at.date(), start)

    # Past the last window today; the first window tomorrow is the answer.
    first_start = UPI_EXECUTION_WINDOWS[0][0]
    return dt.datetime.combine(at.date() + dt.timedelta(days=1), first_start)


def _require_naive(at: dt.datetime, where: str) -> None:
    """Refuse an aware datetime rather than quietly discarding its offset.

    Every window constant here is an IST-local wall-clock time with no tzinfo.
    ``dt.datetime.combine(at.date(), start)`` drops the offset, so an aware
    input produced a naive output — a "next window open" that could not even be
    compared against the input it was derived from without raising, and that
    silently answered in a different frame of reference from the question.

    This module is documented as production logic independent of the simulator,
    and these are its public entry points. ``Request.__post_init__`` and the
    harness both reject aware datetimes before they get here, so nothing
    shipped reaches this — which is exactly why it should raise rather than
    keep working by accident.
    """
    if at.tzinfo is not None:
        raise ValueError(
            f"{where} takes naive IST wall-clock time; got {at!r}. The window "
            "constants carry no tzinfo, so an aware datetime would be compared "
            "against naive boundaries. Convert at the edge of the system."
        )


def next_contactable_moment(at: dt.datetime) -> dt.datetime:
    """The earliest moment at or after ``at`` outside quiet hours."""
    _require_naive(at, "next_contactable_moment")
    clock = at.time()
    if QUIET_HOURS_END <= clock < QUIET_HOURS_START:
        return at
    if clock < QUIET_HOURS_END:
        return dt.datetime.combine(at.date(), QUIET_HOURS_END)
    return dt.datetime.combine(at.date() + dt.timedelta(days=1), QUIET_HOURS_END)
