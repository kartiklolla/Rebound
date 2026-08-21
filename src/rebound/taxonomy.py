"""Failure taxonomy for recurring debits on Indian payment rails.

This module is the domain model the rest of the system branches off. A failed
recurring debit is not one thing: a revoked mandate and an insufficient-funds
decline share a status code and nothing else. Each failure mode maps to a
*disposition*, and the disposition determines which actions are even legal.

Deliberate constraint
---------------------
This module encodes **structure only** — which actions are legal for a given
failure, whether the mandate survived, whether the customer has to do something.
It encodes **no recovery probabilities**. Those are learned from data by
``rebound.model``.

The reason is evaluation integrity. The synthetic generator has opinions about
how often each failure mode recovers. If those opinions were also written here,
the model would be reading the generator's answer key through a side channel and
its held-out metrics would mean nothing. Structural facts (a cancelled mandate
cannot be debited) are safe to encode because they are true by definition of the
rail, not by assumption of the generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Rail(StrEnum):
    """The payment rail a recurring debit was presented on."""

    UPI_AUTOPAY = "upi_autopay"
    ENACH = "enach"
    CARD_ON_FILE = "card_on_file"


class Disposition(StrEnum):
    """What class of response a failure admits.

    This is the single most important abstraction in the system. Everything the
    sequencer is allowed to do is gated on it.
    """

    RETRY_TIMING = "retry_timing"
    """Retryable, and *when* dominates the outcome. The account is real, the
    mandate is live, the money simply was not there. Recovery is a question of
    waiting for the right moment rather than trying harder."""

    RETRY_TRANSIENT = "retry_transient"
    """A technical failure on someone else's infrastructure. Nothing about the
    customer changed. Retry once the window passes; do not escalate to the
    customer, who did nothing wrong and will not understand the message."""

    CUSTOMER_ACTION = "customer_action"
    """The debit cannot succeed until the customer does something — approve an
    authentication prompt, update an expired card. Retrying without a nudge
    just burns attempts against an unchanged state."""

    MANDATE_REPAIR = "mandate_repair"
    """The mandate itself is the problem: wrong amount ceiling, lapsed validity,
    registration defect. Re-presenting the same debit against the same mandate
    fails identically every time. Needs amendment or re-registration."""

    MERCHANT_FIX = "merchant_fix"
    """Our own compliance or data defect caused this — most commonly a missing
    pre-debit notification. Correctable without involving the customer, and the
    cheapest recovery in the system. Also the one worth alerting on, because a
    spike here is a merchant-side bug, not customer behaviour."""

    TERMINAL = "terminal"
    """Dead. The account is closed, the mandate is revoked, debits are barred.
    Further attempts cost money and goodwill and cannot succeed. Recognising
    these early is where a large share of the savings actually comes from."""


class Action(StrEnum):
    """Everything the sequencer can do about a failed debit."""

    RETRY_SAME_RAIL = "retry_same_rail"
    RETRY_ALT_RAIL = "retry_alt_rail"
    SEND_PRE_DEBIT_NOTIFICATION = "send_pre_debit_notification"
    NUDGE_SMS = "nudge_sms"
    NUDGE_WHATSAPP = "nudge_whatsapp"
    NUDGE_EMAIL = "nudge_email"
    SEND_COLLECT_LINK = "send_collect_link"
    VOICE_CALL = "voice_call"
    REQUEST_MANDATE_AMENDMENT = "request_mandate_amendment"
    REQUEST_REMANDATE = "request_remandate"
    STOP = "stop"


#: Actions that present a debit against the mandate.
#:
#: These are the ones the rails' execution windows, retry caps and
#: authentication ceilings bind on. Everything else in ``Action`` is a message
#: or a request, and is governed by different rules.
DEBIT_ACTIONS: frozenset[Action] = frozenset(
    {Action.RETRY_SAME_RAIL, Action.RETRY_ALT_RAIL}
)

#: Actions that put something in front of the customer.
#:
#: Lives here rather than being read from ``world.CONTACT_ACTIONS`` because the
#: compliance gate must not depend on the simulator: delete the simulator and
#: quiet hours still apply. The two sets answer adjacent questions —
#: ``world.CONTACT_ACTIONS`` is "what causes contact fatigue and revocation
#: risk", this is "what arrives on a person's phone" — and today they differ by
#: exactly one member, ``SEND_PRE_DEBIT_NOTIFICATION``, which reaches the
#: customer but is a compliance artifact rather than collection pressure.
#:
#: That near-identity is asserted by ``test_the_two_contact_sets_stay_in_step``
#: so the two cannot drift apart silently. If a future action belongs in one
#: and not the other, that test is where the decision gets made explicitly.
CUSTOMER_FACING_ACTIONS: frozenset[Action] = frozenset(
    {
        Action.NUDGE_SMS,
        Action.NUDGE_WHATSAPP,
        Action.NUDGE_EMAIL,
        Action.VOICE_CALL,
        Action.SEND_COLLECT_LINK,
        Action.SEND_PRE_DEBIT_NOTIFICATION,
        Action.REQUEST_REMANDATE,
        Action.REQUEST_MANDATE_AMENDMENT,
    }
)


@dataclass(frozen=True, slots=True)
class FailureMode:
    """A single failure reason and its structural consequences."""

    code: str
    rail: Rail
    label: str
    disposition: Disposition
    mandate_alive: bool
    """Whether the mandate survives this failure and can be presented again."""

    needs_customer_action: bool
    """Whether the state blocking the debit can only be cleared by the customer."""

    ambiguous: bool = False
    """Issuer catch-all codes that hide their true cause. See ``AMBIGUOUS_CODES``."""

    note: str = ""


# --------------------------------------------------------------------------
# UPI Autopay
# --------------------------------------------------------------------------

_UPI: tuple[FailureMode, ...] = (
    FailureMode(
        code="UPI_INSUFFICIENT_FUNDS",
        rail=Rail.UPI_AUTOPAY,
        label="Insufficient balance in payer account",
        disposition=Disposition.RETRY_TIMING,
        mandate_alive=True,
        needs_customer_action=False,
        note="The dominant failure mode by volume on every rail.",
    ),
    FailureMode(
        code="UPI_MANDATE_REVOKED",
        rail=Rail.UPI_AUTOPAY,
        label="Mandate revoked by payer",
        disposition=Disposition.TERMINAL,
        mandate_alive=False,
        needs_customer_action=False,
        note="Customer cancelled in their UPI app. Often invisible to the "
        "merchant until the next cycle fails.",
    ),
    FailureMode(
        code="UPI_MANDATE_PAUSED",
        rail=Rail.UPI_AUTOPAY,
        label="Mandate paused by payer",
        disposition=Disposition.CUSTOMER_ACTION,
        mandate_alive=True,
        needs_customer_action=True,
        note="Distinct from revoked — the mandate still exists and the customer "
        "can resume it, so this is worth a nudge where revoked is not.",
    ),
    FailureMode(
        code="UPI_MANDATE_EXPIRED",
        rail=Rail.UPI_AUTOPAY,
        label="Mandate validity period ended",
        disposition=Disposition.MANDATE_REPAIR,
        mandate_alive=False,
        needs_customer_action=True,
    ),
    FailureMode(
        code="UPI_AMOUNT_EXCEEDS_MANDATE",
        rail=Rail.UPI_AUTOPAY,
        label="Debit amount above mandate ceiling",
        disposition=Disposition.MANDATE_REPAIR,
        mandate_alive=True,
        needs_customer_action=True,
        note="Typically a price increase applied without amending the mandate. "
        "Every future cycle fails identically until amended.",
    ),
    FailureMode(
        code="UPI_AFA_NOT_COMPLETED",
        rail=Rail.UPI_AUTOPAY,
        label="Payer did not complete authentication",
        disposition=Disposition.CUSTOMER_ACTION,
        mandate_alive=True,
        needs_customer_action=True,
    ),
    FailureMode(
        code="UPI_PSP_UNAVAILABLE",
        rail=Rail.UPI_AUTOPAY,
        label="Payer PSP or handle unavailable",
        disposition=Disposition.RETRY_TRANSIENT,
        mandate_alive=True,
        needs_customer_action=False,
    ),
    FailureMode(
        code="UPI_ACCOUNT_BLOCKED",
        rail=Rail.UPI_AUTOPAY,
        label="Payer account blocked or frozen",
        disposition=Disposition.TERMINAL,
        mandate_alive=False,
        needs_customer_action=False,
    ),
    FailureMode(
        code="UPI_RETRY_LIMIT_EXCEEDED",
        rail=Rail.UPI_AUTOPAY,
        label="Permitted debit attempts exhausted for this cycle",
        disposition=Disposition.TERMINAL,
        mandate_alive=True,
        needs_customer_action=False,
        note="Terminal for the current cycle only, not for the mandate. A "
        "system that cannot tell those apart writes off live customers.",
    ),
)


# --------------------------------------------------------------------------
# eNACH / NACH
# --------------------------------------------------------------------------

_ENACH: tuple[FailureMode, ...] = (
    FailureMode(
        code="NACH_INSUFFICIENT_FUNDS",
        rail=Rail.ENACH,
        label="Insufficient funds in account",
        disposition=Disposition.RETRY_TIMING,
        mandate_alive=True,
        needs_customer_action=False,
    ),
    FailureMode(
        code="NACH_MANDATE_CANCELLED",
        rail=Rail.ENACH,
        label="Mandate cancelled by customer or bank",
        disposition=Disposition.TERMINAL,
        mandate_alive=False,
        needs_customer_action=False,
    ),
    FailureMode(
        code="NACH_MANDATE_NOT_REGISTERED",
        rail=Rail.ENACH,
        label="No active mandate found at destination bank",
        disposition=Disposition.MANDATE_REPAIR,
        mandate_alive=False,
        needs_customer_action=True,
        note="Usually a registration that silently failed upstream. Presents as "
        "a payment failure but is really an onboarding defect.",
    ),
    FailureMode(
        code="NACH_ACCOUNT_CLOSED",
        rail=Rail.ENACH,
        label="Destination account closed",
        disposition=Disposition.TERMINAL,
        mandate_alive=False,
        needs_customer_action=False,
    ),
    FailureMode(
        code="NACH_ACCOUNT_DORMANT",
        rail=Rail.ENACH,
        label="Account dormant or inoperative",
        disposition=Disposition.CUSTOMER_ACTION,
        mandate_alive=True,
        needs_customer_action=True,
        note="Recoverable but only the customer can reactivate, and only in "
        "branch or app. Long tail; rarely worth many attempts.",
    ),
    FailureMode(
        code="NACH_AMOUNT_EXCEEDS_MANDATE",
        rail=Rail.ENACH,
        label="Debit amount above registered mandate limit",
        disposition=Disposition.MANDATE_REPAIR,
        mandate_alive=True,
        needs_customer_action=True,
    ),
    FailureMode(
        code="NACH_MANDATE_DEFECT",
        rail=Rail.ENACH,
        label="Signature or mandate detail mismatch",
        disposition=Disposition.MANDATE_REPAIR,
        mandate_alive=False,
        needs_customer_action=True,
    ),
    FailureMode(
        code="NACH_DEBIT_NOT_PERMITTED",
        rail=Rail.ENACH,
        label="Account type does not permit direct debit",
        disposition=Disposition.TERMINAL,
        mandate_alive=False,
        needs_customer_action=False,
    ),
    FailureMode(
        code="NACH_TECHNICAL_DECLINE",
        rail=Rail.ENACH,
        label="Destination bank technical failure",
        disposition=Disposition.RETRY_TRANSIENT,
        mandate_alive=True,
        needs_customer_action=False,
    ),
)


# --------------------------------------------------------------------------
# Card on file (recurring)
# --------------------------------------------------------------------------

_CARD: tuple[FailureMode, ...] = (
    FailureMode(
        code="CARD_INSUFFICIENT_FUNDS",
        rail=Rail.CARD_ON_FILE,
        label="Insufficient funds or credit limit reached",
        disposition=Disposition.RETRY_TIMING,
        mandate_alive=True,
        needs_customer_action=False,
    ),
    FailureMode(
        code="CARD_DO_NOT_HONOUR",
        rail=Rail.CARD_ON_FILE,
        label="Issuer declined without a specific reason",
        disposition=Disposition.RETRY_TIMING,
        mandate_alive=True,
        needs_customer_action=False,
        ambiguous=True,
        note="The issuer's catch-all. Could be balance, could be an issuer risk "
        "rule, could be a velocity check. Unknowable a priori — which is "
        "exactly why a learned model beats a hand-written ladder here.",
    ),
    FailureMode(
        code="CARD_EXPIRED",
        rail=Rail.CARD_ON_FILE,
        label="Card past expiry date",
        disposition=Disposition.CUSTOMER_ACTION,
        mandate_alive=True,
        needs_customer_action=True,
    ),
    FailureMode(
        code="CARD_TOKEN_INVALID",
        rail=Rail.CARD_ON_FILE,
        label="Network token invalid or de-registered",
        disposition=Disposition.MANDATE_REPAIR,
        mandate_alive=False,
        needs_customer_action=True,
    ),
    FailureMode(
        code="CARD_AFA_REQUIRED",
        rail=Rail.CARD_ON_FILE,
        label="Additional factor of authentication required for this debit",
        disposition=Disposition.CUSTOMER_ACTION,
        mandate_alive=True,
        needs_customer_action=True,
        note="Applies above the regulatory threshold for recurring card debits. "
        "Amount-dependent, so the same customer succeeds one month and "
        "fails the next after a price change.",
    ),
    FailureMode(
        code="CARD_PRE_DEBIT_NOTIFICATION_MISSING",
        rail=Rail.CARD_ON_FILE,
        label="Mandatory pre-debit notification not sent",
        disposition=Disposition.MERCHANT_FIX,
        mandate_alive=True,
        needs_customer_action=False,
        note="Self-inflicted. Send the notification, wait out the window, "
        "re-present. Cheapest recovery in the system and a spike here is a "
        "merchant-side bug worth paging someone about.",
    ),
    FailureMode(
        code="CARD_BLOCKED",
        rail=Rail.CARD_ON_FILE,
        label="Card blocked, stolen, or restricted by issuer",
        disposition=Disposition.TERMINAL,
        mandate_alive=False,
        needs_customer_action=False,
    ),
    FailureMode(
        code="CARD_ISSUER_UNAVAILABLE",
        rail=Rail.CARD_ON_FILE,
        label="Issuer switch unavailable",
        disposition=Disposition.RETRY_TRANSIENT,
        mandate_alive=True,
        needs_customer_action=False,
    ),
    FailureMode(
        code="CARD_STANDING_INSTRUCTION_CANCELLED",
        rail=Rail.CARD_ON_FILE,
        label="Standing instruction cancelled by cardholder",
        disposition=Disposition.TERMINAL,
        mandate_alive=False,
        needs_customer_action=False,
    ),
)


FAILURE_MODES: dict[str, FailureMode] = {
    mode.code: mode for mode in (*_UPI, *_ENACH, *_CARD)
}

AMBIGUOUS_CODES: frozenset[str] = frozenset(
    code for code, mode in FAILURE_MODES.items() if mode.ambiguous
)


# --------------------------------------------------------------------------
# Disposition → legal actions
# --------------------------------------------------------------------------

_DISPOSITION_ACTIONS: dict[Disposition, frozenset[Action]] = {
    Disposition.RETRY_TIMING: frozenset(
        {
            Action.RETRY_SAME_RAIL,
            Action.RETRY_ALT_RAIL,
            Action.NUDGE_SMS,
            Action.NUDGE_WHATSAPP,
            Action.NUDGE_EMAIL,
            Action.SEND_COLLECT_LINK,
            Action.VOICE_CALL,
        }
    ),
    Disposition.RETRY_TRANSIENT: frozenset(
        {
            Action.RETRY_SAME_RAIL,
            Action.RETRY_ALT_RAIL,
        }
    ),
    Disposition.CUSTOMER_ACTION: frozenset(
        {
            Action.NUDGE_SMS,
            Action.NUDGE_WHATSAPP,
            Action.NUDGE_EMAIL,
            Action.SEND_COLLECT_LINK,
            Action.VOICE_CALL,
            Action.RETRY_SAME_RAIL,
        }
    ),
    Disposition.MANDATE_REPAIR: frozenset(
        {
            Action.REQUEST_MANDATE_AMENDMENT,
            Action.REQUEST_REMANDATE,
            Action.SEND_COLLECT_LINK,
            Action.NUDGE_SMS,
            Action.NUDGE_WHATSAPP,
            Action.NUDGE_EMAIL,
        }
    ),
    Disposition.MERCHANT_FIX: frozenset(
        {
            Action.SEND_PRE_DEBIT_NOTIFICATION,
            Action.RETRY_SAME_RAIL,
        }
    ),
    Disposition.TERMINAL: frozenset(),
}


#: Rails a failed debit can fall back to. Ordered by how often the fallback is
#: actually available for a typical Indian consumer, most available first.
ALT_RAILS: dict[Rail, tuple[Rail, ...]] = {
    Rail.ENACH: (Rail.UPI_AUTOPAY, Rail.CARD_ON_FILE),
    Rail.CARD_ON_FILE: (Rail.UPI_AUTOPAY, Rail.ENACH),
    Rail.UPI_AUTOPAY: (Rail.CARD_ON_FILE, Rail.ENACH),
}


class UnknownFailureCode(KeyError):
    """Raised for a failure code absent from the taxonomy.

    Deliberately fatal rather than silently defaulting. An unmapped code means
    the rail changed under us, and quietly bucketing it as ``TERMINAL`` would
    write off recoverable money without anyone noticing.
    """


def get_mode(code: str) -> FailureMode:
    """Look up a failure mode, raising on anything unmapped."""
    try:
        return FAILURE_MODES[code]
    except KeyError:
        raise UnknownFailureCode(
            f"{code!r} is not in the taxonomy. Add it to rebound.taxonomy "
            f"rather than letting it fall through to a default disposition."
        ) from None


def disposition_of(code: str) -> Disposition:
    """Disposition for a failure code."""
    return get_mode(code).disposition


def legal_actions(code: str) -> frozenset[Action]:
    """Actions structurally permitted for a failure code.

    Structural legality only — this says a nudge is *meaningful* for a paused
    mandate and *meaningless* for a closed account. It says nothing about
    whether the nudge is compliant right now (``rebound.compliance``), nor
    whether it is likely to work (``rebound.model``).

    ``STOP`` is always included: giving up is legal for every failure, and
    for terminal ones it is the only legal action.
    """
    return _DISPOSITION_ACTIONS[disposition_of(code)] | {Action.STOP}


def is_terminal(code: str) -> bool:
    """Whether a failure admits no recovery action other than stopping."""
    return disposition_of(code) is Disposition.TERMINAL


def codes_for_rail(rail: Rail) -> tuple[str, ...]:
    """Every failure code that can occur on a rail, in declaration order."""
    return tuple(
        code for code, mode in FAILURE_MODES.items() if mode.rail is rail
    )
