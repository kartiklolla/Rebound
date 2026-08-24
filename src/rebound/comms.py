"""What we say to a customer whose debit failed, and what we are allowed to say.

The sequencer decides *whether* to contact someone, the compliance gate decides
*when*, and this module decides *what words*. That division is the whole point.
By the time anything here runs, every quantity that carries money or legal
exposure is already fixed by code that was measured or by a regulation that was
cited: the amount comes from the mandate, the channel from the chosen action,
the moment from the gate's deferral arithmetic, the instruction from the
failure's disposition. What is left over is a language problem — say this, to
this person, in Hinglish, inside 134 characters — and that is the one part of
the system where a template library is genuinely worse than a language model.

So a drafter is handed a :class:`MessageBrief` and returns prose. It does not
choose the amount, the ask, the channel or the time; those are already on the
brief. Whatever it returns is then checked against the brief by
:mod:`rebound.verify`, which is deterministic and does not call a model,
because a model checking a model is not an independent check.

Nothing here imports the simulator. A brief is built from an approved action
and an episode-shaped record, which in production would be a real one.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from enum import StrEnum

from .compliance import ApprovedAction
from .taxonomy import Action, Disposition, Rail

__all__ = [
    "Ask",
    "Channel",
    "ChannelSpec",
    "CHANNEL_SPECS",
    "CHANNEL_FOR_ACTION",
    "Draft",
    "Language",
    "MerchantProfile",
    "MessageBrief",
    "REVIEWED_LANGUAGES",
    "ask_for",
    "format_rupees",
    "gsm7_encodable",
    "segments_for",
]


# ==========================================================================
# Channels
# ==========================================================================


class Channel(StrEnum):
    """The medium a message goes out on. Distinct from the action."""

    SMS = "sms"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    VOICE = "voice"


#: Which channel each customer-facing action uses.
#:
#: ``SEND_COLLECT_LINK`` and the two mandate-repair requests do not name a
#: medium in :class:`~rebound.taxonomy.Action`, because the sequencer chooses
#: between them on expected value and the medium is an implementation detail of
#: delivery. They are all delivered over WhatsApp here: a collect link and a
#: re-mandate request both need a tappable URL and a message that survives
#: longer than an SMS inbox, and WhatsApp Business is how Indian merchants
#: actually send both.
CHANNEL_FOR_ACTION: dict[Action, Channel] = {
    Action.NUDGE_SMS: Channel.SMS,
    Action.NUDGE_WHATSAPP: Channel.WHATSAPP,
    Action.NUDGE_EMAIL: Channel.EMAIL,
    Action.VOICE_CALL: Channel.VOICE,
    Action.SEND_COLLECT_LINK: Channel.WHATSAPP,
    Action.REQUEST_MANDATE_AMENDMENT: Channel.WHATSAPP,
    Action.REQUEST_REMANDATE: Channel.WHATSAPP,
    Action.SEND_PRE_DEBIT_NOTIFICATION: Channel.SMS,
}


#: The GSM 03.38 basic character set.
#:
#: An SMS whose every character is in here (or in the extension table below)
#: is sent as GSM-7 and fits 160 characters in one segment. One character
#: outside it forces the whole message to UCS-2 and the segment drops to 70.
#:
#: This is why Indian transactional SMS says "Rs." and not "₹". The rupee sign
#: is not in this table, so a single ₹ costs more than half the message.
_GSM7_BASIC = frozenset(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?"
    "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)

#: GSM 03.38 extension table. Each of these occupies two septets, not one.
_GSM7_EXTENDED = frozenset("^{}\\[~]|€")


def gsm7_encodable(text: str) -> bool:
    """Whether ``text`` survives GSM-7 encoding without forcing UCS-2."""
    return all(ch in _GSM7_BASIC or ch in _GSM7_EXTENDED for ch in text)


def _septets(text: str) -> int:
    return sum(2 if ch in _GSM7_EXTENDED else 1 for ch in text)


def segments_for(text: str) -> tuple[int, int, str]:
    """How many SMS segments ``text`` costs.

    Returns ``(segments, units, encoding)`` where *units* is septets for GSM-7
    and UTF-16 code units for UCS-2 — the two things carriers actually count.

    Concatenated messages lose room to the segmentation header, so a two-part
    GSM-7 message carries 153 per part rather than 160, and a two-part UCS-2
    message carries 67 rather than 70. Getting this wrong in the safe
    direction still costs money at scale; getting it wrong in the unsafe
    direction truncates the amount off the end of a payment reminder.
    """
    if gsm7_encodable(text):
        units = _septets(text)
        if units <= 160:
            return 1, units, "GSM-7"
        return -(-units // 153), units, "GSM-7"
    # Surrogate pairs count as two UTF-16 units, which is what the carrier bills.
    units = sum(2 if ord(ch) > 0xFFFF else 1 for ch in text)
    if units <= 70:
        return 1, units, "UCS-2"
    return -(-units // 67), units, "UCS-2"


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """What a channel will physically carry."""

    channel: Channel
    max_chars: int
    """A hard cap on the rendered body. For SMS this is advisory — the real
    constraint is ``max_segments``, which depends on the encoding — but a cap
    still stops a model from returning an essay."""

    max_segments: int | None = None
    """SMS only. ``None`` on channels that are not billed by segment."""

    links_allowed: bool = True
    subject_required: bool = False
    emoji_allowed: bool = True


CHANNEL_SPECS: dict[Channel, ChannelSpec] = {
    # Two segments. One is tight enough that a Devanagari message cannot carry
    # an amount and a date; three is a message nobody reads and three times the
    # cost of one.
    Channel.SMS: ChannelSpec(
        Channel.SMS,
        max_chars=306,
        max_segments=2,
        links_allowed=True,
        emoji_allowed=False,
    ),
    Channel.WHATSAPP: ChannelSpec(
        Channel.WHATSAPP, max_chars=1024, links_allowed=True, emoji_allowed=True
    ),
    Channel.EMAIL: ChannelSpec(
        Channel.EMAIL,
        max_chars=2000,
        links_allowed=True,
        subject_required=True,
        emoji_allowed=False,
    ),
    # A voice script is read aloud by an IVR or an agent. A URL read out as
    # characters is unusable, so links are barred rather than merely discouraged.
    Channel.VOICE: ChannelSpec(
        Channel.VOICE, max_chars=600, links_allowed=False, emoji_allowed=False
    ),
}


# ==========================================================================
# Languages
# ==========================================================================


class Language(StrEnum):
    """The language a message is written in.

    ``HINGLISH`` is Hindi in Latin script, not a mixture of two languages'
    scripts. It is the register most Indian transactional SMS actually uses and
    it is the single hardest thing here to template, because the spelling is
    not standardised — which is exactly why a model earns its place.
    """

    EN = "en"
    HI = "hi"
    HINGLISH = "hinglish"


#: Languages whose fallback templates have been read by someone who speaks them.
#:
#: The fallback is what goes out when a generation fails verification, so an
#: unreviewed fallback is worse than no language at all: it is a guaranteed
#: send of text nobody checked. Adding a language is therefore not a matter of
#: adding an enum member — it is a matter of a native speaker signing off on
#: the templates in :data:`TEMPLATES`. Everything else in this module is
#: already language-agnostic.
REVIEWED_LANGUAGES: frozenset[Language] = frozenset(
    {Language.EN, Language.HI, Language.HINGLISH}
)

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def devanagari_ratio(text: str) -> float:
    """Share of the *letters* in ``text`` written in Devanagari.

    Letters only. Counting over all characters makes the ratio depend on how
    many digits the amount happens to have, so the same sentence about ₹99 and
    ₹14,999 scores differently.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if _DEVANAGARI.match(ch)) / len(letters)


# ==========================================================================
# The ask
# ==========================================================================


class Ask(StrEnum):
    """The one thing the message tells the customer to do.

    Derived from the disposition, never chosen by a drafter. This is the
    boundary that matters most: a model that picks its own ask will eventually
    tell a customer whose balance was short to replace a card that is fine,
    and the customer will do it, and the debit will still fail.
    """

    NOTHING = "nothing"
    """Informational. We are retrying; the customer has no task."""

    KEEP_BALANCE = "keep_balance"
    """Have funds in the account before the next presentation."""

    APPROVE_IN_APP = "approve_in_app"
    """Approve the pending mandate prompt in the UPI app."""

    UPDATE_CARD = "update_card"
    """The saved card is expired or blocked and must be replaced."""

    PAY_NOW_VIA_LINK = "pay_now_via_link"
    """Settle this cycle directly, outside the mandate."""

    AMEND_MANDATE = "amend_mandate"
    """The mandate's ceiling or validity no longer covers the debit."""

    REAUTHORISE_MANDATE = "reauthorise_mandate"
    """The mandate is unusable and a fresh one is needed."""


def ask_for(action: Action, disposition: Disposition, rail: Rail) -> Ask:
    """The instruction a given action carries, as a function of nothing else.

    Deliberately total and deliberately boring. Every branch is a fact about
    Indian rails rather than a judgement call, which is what makes it safe to
    keep out of the model's hands.
    """
    if action is Action.SEND_COLLECT_LINK:
        return Ask.PAY_NOW_VIA_LINK
    if action is Action.REQUEST_MANDATE_AMENDMENT:
        return Ask.AMEND_MANDATE
    if action is Action.REQUEST_REMANDATE:
        return Ask.REAUTHORISE_MANDATE
    if action is Action.SEND_PRE_DEBIT_NOTIFICATION:
        # A pre-debit notice announces a debit that is going to happen. It is
        # not a request, and adding one turns a regulatory notice into dunning.
        return Ask.NOTHING
    if disposition is Disposition.RETRY_TIMING:
        return Ask.KEEP_BALANCE
    if disposition is Disposition.CUSTOMER_ACTION:
        # UPI and eNACH block on an authorisation the customer holds; a card
        # blocks on the card itself. Sending the wrong one of these two is the
        # most common way a recovery message wastes a contact.
        if rail is Rail.CARD_ON_FILE:
            return Ask.UPDATE_CARD
        return Ask.APPROVE_IN_APP
    if disposition is Disposition.MANDATE_REPAIR:
        return Ask.AMEND_MANDATE
    return Ask.NOTHING


# ==========================================================================
# The brief
# ==========================================================================


@dataclass(frozen=True, slots=True)
class MerchantProfile:
    """The sender. Every fact here is quotable into a message."""

    name: str
    support_number: str
    link_host: str
    """The only host a link in any message may point at."""

    sender_id: str = ""
    """The six-character TRAI header an Indian transactional SMS is sent under."""


_DIGIT_RUN = re.compile(r"\d+")


def format_rupees(paise: int) -> str:
    """Rupees in Indian digit grouping, without a currency symbol.

    ``1,50,000`` rather than ``150,000``. The symbol is left to the caller
    because on SMS it must be the ASCII ``Rs.`` — see :data:`_GSM7_BASIC`.
    """
    if paise < 0:
        raise ValueError("amounts are not negative")
    whole, rem = divmod(paise, 100)
    digits = str(whole)
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        digits = ",".join([*parts, tail])
    return digits if rem == 0 else f"{digits}.{rem:02d}"


@dataclass(frozen=True, slots=True)
class MessageBrief:
    """Everything a drafter is allowed to know, and nothing else.

    A drafter sees this and only this. It carries no customer name, no
    balance, no history and no failure code — the first because we do not
    model one, the last three because a customer told "your account has failed
    three times" hears an accusation, and because a raw rail return code in a
    consumer message is a support ticket.

    The brief is also the verifier's reference. Every check in
    :mod:`rebound.verify` is a statement about a draft *relative to this
    object*, which is why it has to carry the permitted numerals and the
    permitted link explicitly rather than leaving them implicit in prose.
    """

    episode_id: str
    customer_id: str

    action: Action
    channel: Channel
    language: Language

    merchant: MerchantProfile
    amount_paise: int
    mandate_reference: str
    bank: str
    rail: Rail
    disposition: Disposition
    ask: Ask

    reference_date: dt.date
    """The day the message goes out, from the caller's clock rather than the
    module's. Deliberately has no default: ``date.today()`` would make a brief
    unreproducible, and a brief that cannot be rebuilt cannot be audited."""

    retry_on: dt.date | None = None
    """When the next presentation happens. Present for a pre-debit notice,
    where it is a disclosure requirement, and for a timing nudge, where it is
    the only reason the customer would act. ``None`` when we do not know."""

    link: str | None = None
    """The single URL this message may contain. ``None`` means no URL at all."""

    @property
    def amount_rupees(self) -> str:
        return format_rupees(self.amount_paise)

    @property
    def currency_prefix(self) -> str:
        """``Rs.`` on SMS, ``₹`` elsewhere.

        Not cosmetic. A rupee sign is outside GSM-7, so one of them halves an
        SMS from 160 characters to 70 and doubles what it costs to send.
        """
        return "Rs." if self.channel is Channel.SMS else "₹"

    @property
    def amount_text(self) -> str:
        return f"{self.currency_prefix}{self.amount_rupees}"

    @property
    def spec(self) -> ChannelSpec:
        return CHANNEL_SPECS[self.channel]

    @property
    def quotable_facts(self) -> tuple[str, ...]:
        """Every string a draft may legitimately contain a form of.

        One table, read by both halves of the fabrication check. It used to be
        two — a list of permitted digit runs and a separate list of permitted
        tokens — and they disagreed: ``episode_id`` was in one, the reference
        date was in neither, and quoting the true date a debit failed was
        reported as a fabricated identifier. Two tables that must agree and are
        maintained separately will not agree.

        Dates appear in several renderings because a drafter picking a
        different one is choosing a format, not inventing a fact.
        """
        facts: list[str] = [
            self.amount_rupees,
            self.amount_rupees.replace(",", ""),
            str(self.amount_paise // 100),
            self.mandate_reference,
            self.merchant.support_number,
            self.merchant.name,
            self.merchant.sender_id,
            self.merchant.link_host,
            self.bank,
            self.episode_id,
        ]
        for day in (self.retry_on, self.reference_date):
            if day is None:
                continue
            facts += [
                day.isoformat(),
                day.strftime("%d/%m/%Y"),
                day.strftime("%d-%m-%Y"),
                day.strftime("%d.%m.%Y"),
                day.strftime("%d %b %Y"),
                day.strftime("%d %B %Y"),
                day.strftime("%d %b"),
                day.strftime("%d %B"),
                str(day.year),
                str(day.day),
            ]
        if self.link:
            facts.append(self.link)
        return tuple(fact for fact in facts if fact)

    @property
    def permitted_numerals(self) -> frozenset[str]:
        """Every digit run a draft may legitimately contain.

        Derived from :attr:`quotable_facts` rather than listed again, so a
        fact added to the brief is permitted in both halves of the check or
        neither.
        """
        return frozenset(
            run for fact in self.quotable_facts for run in _DIGIT_RUN.findall(fact)
        )

    @classmethod
    def build(
        cls,
        approved: ApprovedAction,
        view: object,
        *,
        merchant: MerchantProfile,
        language: Language,
        retry_on: dt.date | None = None,
        link: str | None = None,
    ) -> MessageBrief:
        """Build a brief from an approval and an episode-shaped record.

        Takes an :class:`~rebound.compliance.ApprovedAction` rather than a bare
        action for the same reason the executor does: there is then no way to
        compose a message for something the gate did not permit. The approval
        cannot be forged and cannot be edited, so a brief carrying one is
        evidence that the send was adjudicated.

        Structurally typed on ``view`` so this module does not depend on the
        simulator, matching :meth:`rebound.compliance.Request.from_view`.
        """
        action = approved.action
        if action not in CHANNEL_FOR_ACTION:
            raise ValueError(
                f"{action} does not put anything in front of a customer; "
                "there is no message to compose"
            )
        if language not in REVIEWED_LANGUAGES:
            raise ValueError(
                f"{language} has no reviewed fallback templates. A language "
                "without a reviewed fallback is a guaranteed send of unchecked "
                "text the first time a generation fails."
            )
        if getattr(view, "episode_id") != approved.episode_id:
            raise ValueError(
                "approval is for a different episode than the record given"
            )
        rail: Rail = getattr(view, "rail")
        disposition: Disposition = getattr(view, "disposition")
        channel = CHANNEL_FOR_ACTION[action]
        if link is not None and not CHANNEL_SPECS[channel].links_allowed:
            raise ValueError(f"{channel} messages cannot carry a link")
        ask = ask_for(action, disposition, rail)
        if ask is Ask.PAY_NOW_VIA_LINK and not link:
            raise ValueError(
                "a collect-link message with no link asks the customer to pay "
                "and gives them nowhere to do it"
            )
        return cls(
            episode_id=approved.episode_id,
            customer_id=getattr(view, "customer_id"),
            action=action,
            channel=channel,
            language=language,
            merchant=merchant,
            amount_paise=getattr(view, "cycle_amount_paise"),
            mandate_reference=getattr(view, "mandate_id"),
            bank=getattr(view, "bank"),
            rail=rail,
            disposition=disposition,
            ask=ask,
            reference_date=approved.at.date(),
            retry_on=retry_on,
            link=link,
        )


# ==========================================================================
# A finished message
# ==========================================================================


@dataclass(frozen=True, slots=True)
class Draft:
    """One candidate message. Not yet cleared to send."""

    body: str
    language: Language
    produced_by: str
    """``template`` or ``model:<id>``. Kept on the draft rather than inferred
    later because the fallback rate is a headline number and it has to come
    from the record of what actually happened, not from a reconstruction."""

    subject: str | None = None

    def rendered(self) -> str:
        """Subject and body together, which is what the checks run over.

        An email whose subject says ₹4,999 and whose body says ₹499 passes
        every check applied to the body alone.
        """
        return f"{self.subject}\n{self.body}" if self.subject else self.body
