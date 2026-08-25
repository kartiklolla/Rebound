"""Deterministic checks between a generated message and the facts it came from.

Nothing in this module calls a language model, and that is the design rather
than an omission. A model asked to grade another model's output shares its
failure modes, agrees with itself under pressure, and cannot be cross-examined
afterwards by anyone deciding whether a message should have gone out. Every
check here is an exact comparison against :class:`~rebound.comms.MessageBrief`
and can be re-run years later on the stored pair.

The checks are not equally strong and pretending otherwise would be the
dangerous part. Three tiers, honestly labelled:

*Exact.* Amounts, links, invented identifiers, length, script, subject. These
are decidable. A draft that passes them cannot be misquoting a number.

*Bounded.* The instruction check knows the finite set of things we ever ask a
customer to do, so it can confirm the right one is present and the wrong ones
are absent. It cannot confirm the sentence around them is coherent.

*Lexical.* Coercion and credential-solicitation are word lists. They catch the
obvious cases and a sufficiently polite threat walks straight through. They are
here because the obvious cases are the ones that actually occur, and they are
labelled :attr:`Category.SAFETY` so nobody mistakes them for a guarantee.

Every finding blocks the send. There is no severity ladder, because a
"warning" that still goes out to a customer is not a warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from .comms import (
    Ask,
    Channel,
    Draft,
    Language,
    MessageBrief,
    devanagari_ratio,
    gsm7_encodable,
    segments_for,
)
from .taxonomy import FAILURE_MODES

__all__ = [
    "Category",
    "Check",
    "CHECKS",
    "Finding",
    "verify",
    "feedback_for",
]


class Category(StrEnum):
    """What kind of thing went wrong. Used to report failures, never to excuse one."""

    FIDELITY = "fidelity"
    """The draft says something the brief does not support — a wrong amount, an
    invented reference, a link we do not own."""

    SAFETY = "safety"
    """The draft would harm the recipient if sent: it solicits a credential, or
    it threatens."""

    INSTRUCTION = "instruction"
    """The draft tells the customer to do the wrong thing, or nothing at all."""

    FORMAT = "format"
    """The draft will not render or deliver correctly on the chosen channel."""


@dataclass(frozen=True, slots=True)
class Finding:
    """One reason a draft cannot be sent."""

    check_id: str
    category: Category
    detail: str
    """Written to be readable by the model on a repair pass *and* by a human in
    an audit. One string serving both is deliberate: a repair instruction the
    auditor cannot read is a repair nobody can review."""


class Check(Protocol):
    check_id: str
    category: Category

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None: ...


def _finding(check: Check, detail: str) -> Finding:
    return Finding(check_id=check.check_id, category=check.category, detail=detail)


# ==========================================================================
# Exact checks
# ==========================================================================

# --------------------------------------------------------------------------
# Anything that could function as a link
# --------------------------------------------------------------------------
#
# This is written the opposite way round from how it started, and the reversal
# is the single most important correction in this module.
#
# The first version detected URLs against an allow-list of twenty-one
# top-level domains. Everything off that list was invisible, so
# ``vahan-secure.ru/pay`` matched nothing, drew no finding, and was returned by
# the desk as cleared to send — on a voice script too, where links are supposed
# to be barred absolutely. The red-team corpus reported the link check at 3/3
# and said nothing about it, because its one scheme-less probe happened to sit
# comfortably inside the allow-list. An enumeration of the bad shapes cannot
# work: the attacker picks the shape.
#
# So the polarity is inverted. Anything token-shaped that could plausibly
# resolve, be tapped, or be typed into a browser is *suspect by default*, and
# only three things clear it: the exact link on the brief, an ordinary number
# or amount, and a short list of English abbreviations.
#
# That deliberately rejects "keep sufficient balance.In case of trouble" — a
# missing space, not a link. An earlier version of this module treated that
# false positive as the thing to optimise away, which was the wrong call and
# was made for the wrong reason: it was reasoning about the reported fallback
# rate rather than about harm. The two errors are not comparable. A false
# positive costs one repair round trip, and the repair pass is told exactly
# which token offended, so the model fixes the typo and the message goes out.
# A false negative costs a customer their money. Fail closed.

#: Characters that are not U+002E but that resolvers and humans read as a dot.
_CONFUSABLE_DOTS = str.maketrans(
    {"․": ".", "．": ".", "。": ".", "۔": ".", "‧": "."}
)

#: Punctuation that ends a sentence rather than belonging to the token.
_TRAILING_PUNCTUATION = ".,;:!?)]}\"'»…"

#: A dot immediately followed by a letter. This is what distinguishes a host
#: from money: ``vahan.in`` has one, ``Rs.1,299`` and ``05.09.2026`` do not.
_DOT_THEN_LETTER = re.compile(r"\.[^\W\d_]", re.UNICODE)

#: Four numeric labels. A bare address needs no TLD at all.
_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}(?:[:/].*)?$")

#: A host:port, which carries no dot-then-letter when the host is numeric.
_HOST_PORT = re.compile(r":\d{2,5}(?:/|$)")

#: A plain quantity — an amount, a decimal, a numbered reference.
_JUST_A_NUMBER = re.compile(
    r"^(?:₹|rs\.?|inr|₨)?[\d,]+(?:\.\d+)?(?:/-)?$", re.IGNORECASE
)

#: English abbreviations that carry an internal dot and are not hosts.
_ABBREVIATIONS = frozenset(
    {"a.m.", "p.m.", "e.g.", "i.e.", "no.", "vs.", "etc.", "pvt.", "ltd.", "co."}
)


def _tokens(text: str) -> list[str]:
    return text.translate(_CONFUSABLE_DOTS).split()


def _is_link_shaped(token: str) -> bool:
    """Whether ``token`` could resolve, be tapped, or be typed into a browser.

    Deliberately broad. ``upi://pay?pa=someone@psp`` is the case that matters
    most in this domain and has no TLD at all: it is a tappable payment intent
    that sends money straight to whoever wrote it.
    """
    if token.casefold() in _ABBREVIATIONS:
        return False
    stripped = token.strip(_TRAILING_PUNCTUATION)
    if not stripped:
        return False
    if "://" in stripped or stripped.startswith("//"):
        return True
    if _JUST_A_NUMBER.match(stripped):
        return False
    if "@" in stripped:
        return True
    if _IPV4.match(stripped) or _HOST_PORT.search(stripped):
        return True
    return _DOT_THEN_LETTER.search(stripped) is not None


def _permitted_tokens(brief: MessageBrief) -> frozenset[str]:
    """Bare forms of every whitespace-separated word in the brief's own facts.

    Whole tokens, never substrings. Substring containment would exempt
    ``vahan.in`` because it sits inside the permitted ``pay.vahan.in``, and a
    bare parent domain is a different destination from the link we issued.
    """
    return frozenset(
        _bare(word)
        for fact in brief.quotable_facts
        for word in fact.split()
        if _bare(word)
    )


def _find_urls(text: str) -> list[str]:
    """Every token in ``text`` that could function as a link.

    Returned without trailing sentence punctuation, so the caller compares a
    destination rather than a destination plus a full stop.
    """
    return [
        token.strip(_TRAILING_PUNCTUATION)
        for token in _tokens(text)
        if _is_link_shaped(token)
    ]


def _strip_urls(text: str) -> str:
    """``text`` with its link-shaped tokens blanked out.

    Used by the checks that must not read a URL as prose — the script ratio
    would otherwise call a fully Devanagari message 44% Latin because of the
    path on a payment link.
    """
    return " ".join(
        "" if _is_link_shaped(token) else token for token in _tokens(text)
    )

#: Money written with a symbol in front of it.
#:
#: The leading lookbehind stops "hou**rs.**500" being read as a sum of money.
_AMOUNT_PREFIXED = re.compile(
    r"(?<![A-Za-z0-9])(?:₹|Rs\.?|INR|रु\.?|₨)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
    re.IGNORECASE,
)

#: Money written with the unit spelled out after it.
#:
#: Only spelled-out units. An earlier version also accepted a number followed
#: by "Rs." or "₹", which reads the *first* number in "mandate UMRN0012345678
#: Rs.1,299" as a sum of money and reports the pre-debit notice's own legally
#: required reference as a wrong amount. Nobody writes "1299 Rs" in this
#: register anyway; the symbol goes in front.
_AMOUNT_SUFFIXED = re.compile(
    r"(?<![A-Za-z0-9,./-])([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*"
    r"(?:rupees|rupaye|rupaiye|रुपये|रुपए)",
    re.IGNORECASE,
)


_NOT_ALNUM = re.compile(r"[^a-z0-9]+")


def _bare(text: str) -> str:
    """``text`` reduced to lowercase letters and digits.

    Comparing raw strings made the fabrication check a test of formatting
    rather than of truth: a drafter that rendered the real support number
    ``18002670001`` as the readable ``1800-267-0001`` was reported as
    inventing it, while storing it hyphenated on the brief made the same draft
    pass. Dates had the identical problem across ``/``, ``-`` and ``.``.
    """
    return _NOT_ALNUM.sub("", text.casefold())


def _to_paise(literal: str) -> int | None:
    cleaned = literal.replace(",", "")
    if "." in cleaned:
        whole, _, frac = cleaned.partition(".")
        frac = (frac + "00")[:2]
    else:
        whole, frac = cleaned, "00"
    if not whole.isdigit():
        return None
    return int(whole) * 100 + int(frac)


@dataclass(frozen=True, slots=True)
class AmountIsExact:
    """Every sum of money named must be the sum actually owed.

    Both directions matter. A draft that quotes no amount is not a payment
    reminder, and a draft that quotes two different amounts has invented one of
    them. Exactly one distinct value is permitted because this system never
    adds a late fee, so there is no second legitimate figure a message could
    carry.
    """

    check_id: str = "amount_is_exact"
    category: Category = Category.FIDELITY

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        text = draft.rendered()
        literals = _AMOUNT_PREFIXED.findall(text) + _AMOUNT_SUFFIXED.findall(text)
        if not literals:
            return _finding(
                self,
                f"names no amount; must state {brief.amount_text} exactly once",
            )
        seen = {_to_paise(literal) for literal in literals}
        wrong = {paise for paise in seen if paise != brief.amount_paise}
        if wrong:
            quoted = ", ".join(
                "unparseable" if paise is None else f"{paise / 100:.2f}"
                for paise in sorted(wrong, key=lambda p: (p is None, p or 0))
            )
            return _finding(
                self,
                f"quotes {quoted} but the amount due is "
                f"{brief.amount_rupees} — use {brief.amount_text} and no other figure",
            )
        return None


#: A token long enough to be an identifier rather than a passing number.
#:
#: Five characters. Four would flag years and clock times; six would let a
#: five-digit invented reference through. Five sits above everything a message
#: says in passing — a date, an hour, a day of the month — and below every real
#: identifier on these rails, where a UMRN and a customer-care number are both
#: far longer.
_ALNUM_TOKEN = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9/_-]{4,}\b")


@dataclass(frozen=True, slots=True)
class NoFabricatedIdentifiers:
    """Reference numbers in the message must be reference numbers we hold.

    The specific failure this exists for: a model asked for a payment reminder
    fills the shape of one, and the shape includes a transaction reference. If
    it does not have one it will write a plausible one. The customer then
    quotes it to a support agent who cannot find it, which is worse than
    omitting it, and to the customer looks exactly like a phishing message.
    """

    check_id: str = "no_fabricated_identifiers"
    category: Category = Category.FIDELITY

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        text = _strip_urls(draft.rendered())
        facts = _permitted_tokens(brief)

        # One rule, not two. There used to be a separate pass over bare digit
        # runs, because the token pass compared by substring containment and so
        # accepted any *fragment* of a true fact — a run like 180026, a piece
        # of the real support number, needed exact set membership to be caught.
        #
        # Moving the token pass to exact matching subsumed it entirely. Over
        # 400,000 randomly generated strings there was no input the digit pass
        # rejected and the token pass accepted, while the token pass rejects
        # strictly more: HDFC0009911, the tail of the real mandate reference,
        # is a fabrication that the digit pass waved through because every
        # digit run inside it is genuinely ours.
        for token in _ALNUM_TOKEN.findall(text):
            if not any(ch.isdigit() for ch in token):
                continue
            # Whole tokens, not substrings. Substring containment accepted any
            # *fragment* of a true fact, so ``HDFC0009911`` — the tail of the
            # real reference UMRN2024HDFC0009911 — passed both branches: the
            # digit run inside it is genuinely one of ours. A customer quotes
            # that to support, support cannot find it, and a nearly-right
            # reference is worse than an obviously invented one.
            if _bare(token) not in facts:
                return _finding(
                    self,
                    f"contains the reference {token!r}, which is not one of "
                    "ours — quote only the mandate reference "
                    f"{brief.mandate_reference}",
                )
        return None


_SCHEME = re.compile(r"^(?:https?://)?(?:www\.)?", re.IGNORECASE)


def _canonical_url(url: str) -> str:
    """A URL without its scheme, ``www.`` or trailing sentence punctuation."""
    return _SCHEME.sub("", url.strip()).rstrip(".,;:)!?").casefold()


@dataclass(frozen=True, slots=True)
class LinksAreOurs:
    """A URL must be the one URL on the brief, or there must be no URL.

    Barred outright on channels that cannot carry one, which for a voice script
    means barred because it would be read aloud character by character.
    """

    check_id: str = "links_are_ours"
    category: Category = Category.FIDELITY

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        found = _find_urls(draft.rendered())

        # Our own link, and any dotted token that is a word from the brief's
        # own facts, are the only things that clear. Compared without the
        # scheme and without trailing punctuation, because a message that
        # writes our link as a bare host is writing the same link — while a
        # *different* host is caught either way.
        ours = _canonical_url(brief.link) if brief.link else None
        permitted = _permitted_tokens(brief)
        stray = [
            token
            for token in found
            if _canonical_url(token) != ours
            and _bare(token.strip(_TRAILING_PUNCTUATION)) not in permitted
        ]

        if not stray:
            if brief.ask is Ask.PAY_NOW_VIA_LINK and not any(
                _canonical_url(token) == ours for token in found
            ):
                return _finding(
                    self, f"must contain the payment link {brief.link}"
                )
            return None

        if not brief.spec.links_allowed:
            return _finding(
                self,
                f"{brief.channel} messages carry no links; remove {stray[0]!r}",
            )
        if brief.link is None:
            return _finding(
                self,
                f"contains {stray[0]!r}, which could be tapped or typed as a "
                "link, and this message has no link to send",
            )
        return _finding(
            self,
            f"contains {stray[0]!r}, which is not ours — the only permitted "
            f"URL is {brief.link}. If that was a missing space after a full "
            "stop, add the space",
        )


@dataclass(frozen=True, slots=True)
class WithinChannelBudget:
    """The message must fit what the channel will actually deliver.

    On SMS this is not a style rule. A message that spills into a third segment
    costs three times a one-segment send, and Devanagari drops the per-segment
    budget from 153 characters to 67 because the whole message is re-encoded as
    UCS-2. A Hindi reminder therefore has a third of the room an English one
    has, which is the constraint that makes this worth generating rather than
    translating.
    """

    check_id: str = "within_channel_budget"
    category: Category = Category.FORMAT

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        spec = brief.spec
        body = draft.body.strip()
        if len(body) < 20:
            return _finding(self, "is too short to be a message")
        if len(draft.rendered()) > spec.max_chars:
            return _finding(
                self,
                f"is {len(draft.rendered())} characters; the limit on "
                f"{brief.channel} is {spec.max_chars}",
            )
        if spec.max_segments is not None:
            segments, units, encoding = segments_for(body)
            if segments > spec.max_segments:
                return _finding(
                    self,
                    f"needs {segments} SMS segments ({units} units, {encoding}); "
                    f"the limit is {spec.max_segments}. Shorten it"
                    + (
                        " — Devanagari costs a UCS-2 segment every 67 characters"
                        if encoding == "UCS-2"
                        else ""
                    ),
                )
        return None


_EMOJI = re.compile(
    "[\U0001f000-\U0001faff☀-➿←-⇿️⬀-⯿]"
)


@dataclass(frozen=True, slots=True)
class RendersOnChannel:
    """Subject where one is required, no emoji where they do not belong."""

    check_id: str = "renders_on_channel"
    category: Category = Category.FORMAT

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        spec = brief.spec
        if spec.subject_required and not (draft.subject or "").strip():
            return _finding(self, f"{brief.channel} needs a subject line")
        if not spec.subject_required and draft.subject:
            return _finding(
                self, f"{brief.channel} has no subject line; put everything in the body"
            )
        if not spec.emoji_allowed:
            hit = _EMOJI.search(draft.rendered())
            if hit:
                return _finding(
                    self,
                    f"contains {hit.group(0)!r}; {brief.channel} messages carry "
                    "no emoji"
                    + (
                        " — one forces the whole SMS to UCS-2 and cuts the "
                        "segment from 160 characters to 70"
                        if brief.channel is Channel.SMS
                        else ""
                    ),
                )
        return None


@dataclass(frozen=True, slots=True)
class SmsStaysInGsm7:
    """A Latin-script SMS must not smuggle in a character that costs UCS-2.

    Replaces what an emoji range list was trying and failing to do. The list
    missed ``‼``, ``™``, ``©`` and ``〰``, all outside GSM-7 and all silently
    doubling the per-character cost of every send — but the deeper problem was
    that it enumerated the wrong thing. The property that matters is not "is
    this an emoji", it is "does this force the whole message into UCS-2", and
    that has an exact answer in :func:`~rebound.comms.gsm7_encodable`. A curly
    apostrophe or an en dash, neither of them an emoji, does the same damage.

    Hindi is exempt: Devanagari is UCS-2 by definition and that cost was
    budgeted when the language was chosen.
    """

    check_id: str = "sms_stays_in_gsm7"
    category: Category = Category.FORMAT

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        if brief.channel is not Channel.SMS or brief.language is Language.HI:
            return None
        text = draft.rendered()
        if gsm7_encodable(text):
            return None
        offending = sorted(
            {ch for ch in text if not gsm7_encodable(ch)}
        )
        return _finding(
            self,
            f"uses {''.join(offending)!r}, which is outside GSM-7 — one such "
            "character re-encodes the whole SMS as UCS-2 and cuts the segment "
            "from 160 characters to 70. Use plain ASCII",
        )


#: Enough Devanagari that the message reads as Hindi rather than as English
#: with a Hindi word in it. Not 1.0: a Hindi message legitimately carries
#: "UPI", "SMS" and the merchant's Latin-script brand name.
_HINDI_SCRIPT_FLOOR = 0.6

#: Romanised Hindi function words.
#:
#: The list started with the words a template author reaches for and rejected
#: half of the Hinglish a person actually writes: "Rs.1,299 ka payment fail ho
#: gaya, balance daal dijiye" carried none of them. The postpositions and the
#: common verb tails are what make a sentence Hinglish rather than English,
#: and they were the ones missing. Hinglish is the register this module names
#: as the reason a model earns its place, so a false-reject rate there does
#: not merely annoy — it inflates the fallback headline in the flattering
#: direction.
#:
#: None is an English word except "hum", which is why the floor is two.
_HINGLISH_MARKERS = frozenset(
    {
        # pronouns and postpositions
        "aap", "aapka", "aapke", "aapki", "apka", "apke", "apna", "apne",
        "hum", "hamara", "ka", "ki", "ke", "ko", "se", "mein", "par", "tak",
        # verbs and their tails
        "hai", "hain", "hua", "hui", "ho", "hoga", "hogi", "gaya", "gayi",
        "kata", "kate", "kar", "kare", "karein", "karo", "karna", "karenge",
        "kijiye", "dijiye", "daal", "daalein", "rakhein", "rakhna", "rakhiye",
        "bhej", "bheja", "sakta", "sakte", "raha", "rahi", "tha", "thi",
        "banaye", "milega", "lagega",
        # adverbs, nouns, courtesies
        "kripya", "kripaya", "nahi", "nahin", "abhi", "jaldi", "dobara",
        "phir", "zaroori", "zaroorat", "samay", "liye", "wala", "sirf",
        "paisa", "paise", "khata", "khate", "jama", "bhugtan", "raashi",
        "rashi", "shukriya", "dhanyavaad", "sawaal", "madad",
    }
)
_MIN_HINGLISH_MARKERS = 2


@dataclass(frozen=True, slots=True)
class ScriptMatchesLanguage:
    """The message must be in the script the brief asked for.

    Hinglish is the case that needs a check rather than a convention. It is
    Hindi written in Latin script, so a drafter that "helpfully" answers in
    Devanagari has not produced a Hinglish message — it has produced a Hindi
    one, which on SMS costs a segment the sender did not budget for, and which
    a recipient who reads romanised Hindi but not the script cannot read at all.
    """

    check_id: str = "script_matches_language"
    category: Category = Category.FORMAT

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        text = draft.rendered()
        # A URL is not prose and its script says nothing about the language of
        # the message around it. Counting it rejected every correct Hindi
        # message that carried a payment link: the twenty-odd Latin characters
        # of ``https://pay.example.in/r/7Kd2Qm`` pulled a fully Devanagari
        # sentence down to 44%, so the fallback would have failed its own
        # check on the one message that most needs to be sent. The brand name
        # is deliberately still counted — it is read aloud as part of the
        # sentence, and a "Hindi" message that is mostly Latin brand is a real
        # defect rather than an artefact.
        ratio = devanagari_ratio(_strip_urls(text))
        if brief.language is Language.EN:
            if ratio > 0:
                return _finding(self, "is meant to be English but contains Devanagari")
            return None
        if brief.language is Language.HI:
            if ratio < _HINDI_SCRIPT_FLOOR:
                return _finding(
                    self,
                    f"is meant to be Hindi but only {ratio:.0%} of its letters "
                    "are Devanagari; write it in Devanagari",
                )
            return None
        # Hinglish.
        if ratio > 0:
            return _finding(
                self,
                "is meant to be Hinglish — Hindi in Latin script — but contains "
                "Devanagari; transliterate it",
            )
        words = set(re.findall(r"[a-z]+", text.casefold()))
        if len(words & _HINGLISH_MARKERS) < _MIN_HINGLISH_MARKERS:
            return _finding(
                self,
                "is meant to be Hinglish but reads as plain English; write it "
                "the way an Indian merchant's SMS actually reads",
            )
        return None


@dataclass(frozen=True, slots=True)
class NoInternalCodes:
    """Nothing from our own vocabulary reaches the customer.

    Rail return codes, disposition names and action names are all
    ``SCREAMING_SNAKE`` strings that a model handed them in context will
    cheerfully quote. ``UPI_INSUFFICIENT_FUNDS`` in a consumer SMS is a support
    ticket at best and, on a message asking for money, reads as a scam.
    """

    check_id: str = "no_internal_codes"
    category: Category = Category.FIDELITY
    _shape = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b")

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        text = draft.rendered()
        for code in FAILURE_MODES:
            if code.casefold() in text.casefold():
                return _finding(self, f"quotes the internal failure code {code}")
        # A code-shaped string that is one of our own facts is not a leak.
        # The simulator issues mandate references as MND_0000001, which fits
        # the shape exactly — and a pre-debit notice is *required* to quote its
        # mandate reference. The two checks contradicted each other, so every
        # notice was blocked outright, fallback included: no message went out
        # at all. Unit fixtures missed it because they used a UMRN with no
        # underscore in it.
        ours = {_bare(fact) for fact in brief.quotable_facts}
        for hit in self._shape.finditer(text):
            if _bare(hit.group(0)) not in ours:
                return _finding(
                    self, f"quotes what looks like an internal code: {hit.group(0)}"
                )
        return None


@dataclass(frozen=True, slots=True)
class SenderIsIdentified:
    """The merchant's name must appear.

    An unsigned message about money is indistinguishable from a phishing
    attempt, and Indian consumers are trained — correctly — to ignore those.
    An unsigned reminder does not merely fail to recover; it teaches the
    recipient to ignore the next one.
    """

    check_id: str = "sender_is_identified"
    category: Category = Category.SAFETY

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        # The signature, not the legal entity. Requiring the full registered
        # name made a one-segment SMS unwritable — "Vahan Technologies Private
        # Limited" alone is a fifth of the GSM-7 budget — so the check
        # rejected every correct draft and no message could clear at all.
        signature = brief.merchant.signature
        rendered = draft.rendered().casefold()
        if signature.casefold() not in rendered:
            return _finding(
                self, f"does not name the sender; it must say {signature}"
            )
        return None


# ==========================================================================
# Bounded check: the instruction
# ==========================================================================

#: Phrases that identify each instruction, in all three languages at once.
#:
#: One table rather than three because matching is substring-based and the
#: languages do not collide: no English reminder contains "मंज़ूर" and no Hindi
#: one contains "sufficient balance".
_ASK_MARKERS: dict[Ask, frozenset[str]] = {
    Ask.KEEP_BALANCE: frozenset(
        {
            "balance", "funds", "sufficient", "top up", "top-up",
            "paisa", "paise", "raashi", "rashi", "jama",
            "बैलेंस", "राशि", "शेष", "पैसे", "जमा",
        }
    ),
    Ask.APPROVE_IN_APP: frozenset(
        {
            "approve", "approval", "authorise", "authorize", "pending request",
            "upi app", "mandate request",
            "manzoori", "manjoori", "swikar", "स्वीकृत", "मंज़ूर", "मंजूर",
            "अनुमति", "अनुरोध",
        }
    ),
    Ask.UPDATE_CARD: frozenset(
        {
            "update your card", "new card", "card update", "card details",
            "expired card", "naya card", "card badal",
            "नया कार्ड", "कार्ड अपडेट", "कार्ड बदल",
        }
    ),
    Ask.PAY_NOW_VIA_LINK: frozenset(
        {"pay", "payment", "link", "settle", "bhugtan", "भुगतान", "लिंक"}
    ),
    Ask.AMEND_MANDATE: frozenset(
        {
            "mandate", "autopay", "limit", "ceiling", "e-mandate",
            "मैंडेट", "ऑटोपे", "सीमा", "अधिदेश",
        }
    ),
    Ask.REAUTHORISE_MANDATE: frozenset(
        {
            "mandate", "autopay", "set up", "set-up", "register", "re-register",
            "naya mandate", "मैंडेट", "ऑटोपे", "फिर से", "दोबारा",
        }
    ),
    Ask.NOTHING: frozenset(),
}

#: Instructions that must not appear when a different one was intended.
#:
#: Narrower than the marker table on purpose. Matching a whole phrase like
#: "update your card" is high precision; matching the bare word "card" would
#: reject a legitimate line about which card the mandate sits on. A forbidden
#: list that fires on innocent drafts drives the fallback rate up and teaches
#: everyone to stop reading it.
#: Every instruction needs an entry, not only the ones that felt dangerous.
#: With three of the seven missing, a draft could carry its own instruction
#: *and* a second, wrong one and pass — "please update your card, and keep
#: sufficient balance in the account" was accepted, which is the same wasted
#: contact as the case this check was written for, arriving from the other
#: side.
_CONTRADICTIONS: dict[Ask, frozenset[str]] = {
    Ask.UPDATE_CARD: frozenset(
        {"update your card", "new card", "card update", "naya card", "नया कार्ड"}
    ),
    Ask.APPROVE_IN_APP: frozenset(
        {"approve the request", "approve this request", "pending mandate request"}
    ),
    Ask.REAUTHORISE_MANDATE: frozenset(
        {"new mandate", "naya mandate", "नया मैंडेट", "register the mandate again"}
    ),
    Ask.KEEP_BALANCE: frozenset(
        {
            "keep sufficient balance", "maintain sufficient balance",
            "ensure sufficient balance", "add funds", "top up your account",
            "balance rakhein", "raashi rakhein", "rashi rakhein",
            "राशि रखें", "बैलेंस रखें",
        }
    ),
    Ask.AMEND_MANDATE: frozenset(
        {
            "increase the limit", "update the mandate", "mandate update",
            "autopay limit", "mandate update karein",
            "मैंडेट अपडेट", "ऑटोपे सीमा",
        }
    ),
    # PAY_NOW_VIA_LINK has no entry on purpose. Every phrasing of it is built
    # from "pay", which is too generic to forbid: a balance reminder that says
    # "your payment did not go through" would be rejected for containing the
    # word. The link itself is already constrained by ``LinksAreOurs``, which
    # is the part that could do harm.
}

#: Which instructions are compatible with which. Read as: an ``AMEND_MANDATE``
#: message may talk about re-authorising, because the two repairs shade into
#: each other and a customer sent to the mandate screen can do either.
_COMPATIBLE: dict[Ask, frozenset[Ask]] = {
    Ask.AMEND_MANDATE: frozenset({Ask.REAUTHORISE_MANDATE}),
    Ask.REAUTHORISE_MANDATE: frozenset({Ask.AMEND_MANDATE}),
    # "Please approve the new mandate request in your UPI app" is the natural
    # English for this instruction, and it was unwritable: the phrase "new
    # mandate" belongs to REAUTHORISE_MANDATE, so the check rejected the
    # sentence the check exists to permit.
    Ask.APPROVE_IN_APP: frozenset({Ask.REAUTHORISE_MANDATE}),
    # A pre-debit notice legitimately reminds the customer to have funds
    # ready — that is the standard wording, and it is not an instruction to do
    # anything the notice was not for. It must still not tell them to touch a
    # card or a mandate, so only this one is compatible.
    Ask.NOTHING: frozenset({Ask.KEEP_BALANCE}),
}


@dataclass(frozen=True, slots=True)
class AskIsHonoured:
    """The message must carry its own instruction and no other.

    This is the check the whole division of labour exists to make possible. The
    brief fixes what the customer is being asked to do, derived from the
    failure's disposition; the drafter only chooses words. Without this check
    that division is a comment rather than a constraint, and the first time a
    model decides a short-balance customer should replace their card, the
    customer will replace it and the next debit will fail exactly the same way.
    """

    check_id: str = "ask_is_honoured"
    category: Category = Category.INSTRUCTION

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        text = draft.rendered().casefold()
        required = _ASK_MARKERS[brief.ask]
        if required and not any(marker in text for marker in required):
            return _finding(
                self,
                f"never asks the customer to {brief.ask.replace('_', ' ')}, "
                "which is the only thing this message exists to do",
            )
        allowed = {brief.ask} | _COMPATIBLE.get(brief.ask, frozenset())
        for other, phrases in _CONTRADICTIONS.items():
            if other in allowed:
                continue
            for phrase in phrases:
                if phrase in text:
                    return _finding(
                        self,
                        f"tells the customer to {other.replace('_', ' ')} "
                        f"({phrase!r}), but the failure calls for "
                        f"{brief.ask.replace('_', ' ')}",
                    )
        return None


@dataclass(frozen=True, slots=True)
class PreDebitDisclosure:
    """A pre-debit notice must disclose what it is required to disclose.

    Applies only to :attr:`~rebound.taxonomy.Action.SEND_PRE_DEBIT_NOTIFICATION`.
    The notice is the thing that makes the *following* debit lawful, so a
    notice missing the amount, the date or the mandate reference is not a
    weaker notice — the debit behind it is unnotified.

    The 24-hour window this pairs with lives in :mod:`rebound.regulation` and
    is on that module's unverified list. The three fields are the parts nobody
    disputes; the timing is what needs the citation.
    """

    check_id: str = "pre_debit_disclosure"
    category: Category = Category.FIDELITY

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        from .taxonomy import Action

        if brief.action is not Action.SEND_PRE_DEBIT_NOTIFICATION:
            return None
        text = draft.rendered()
        missing = []

        # Compared through _bare, as everywhere else in this module, so a
        # reference written ICIC 8842 0091 satisfies a brief holding
        # ICIC-8842-0091. Matching raw made this a test of punctuation.
        if _bare(brief.mandate_reference) not in _bare(text):
            missing.append(f"the mandate reference {brief.mandate_reference}")

        if brief.retry_on is None:
            missing.append("a debit date (none was supplied to the brief)")
        elif not self._states_the_date(text, brief):
            missing.append(f"the debit date {brief.retry_on:%d/%m/%Y}")

        # The amount, which this check's own docstring claimed to require and
        # did not. AmountIsExact confirms any figure present is the right one;
        # it does not make a notice state one, and a notice without an amount
        # discloses nothing.
        if not _AMOUNT_PREFIXED.search(text) and not _AMOUNT_SUFFIXED.search(text):
            missing.append(f"the amount {brief.amount_text}")

        if missing:
            return _finding(
                self,
                "is a pre-debit notice and omits " + " and ".join(missing),
            )
        return None

    @staticmethod
    def _states_the_date(text: str, brief: MessageBrief) -> bool:
        """Whether a real date appears, rather than a digit that happens to match.

        The accepted forms used to include the bare day-of-month, a one- or
        two-character substring tested against the whole message. Any stray
        ``5`` — in ``Rs.1,599``, in the reference, in a phone number —
        satisfied the disclosure, so a notice reading "Rs.1,599 will be debited
        soon" passed with no date in it at all and the debit behind it went out
        unnotified.

        A day number now only counts when a month name sits beside it, which is
        what makes "debited on 5 September" a date and "Rs.1,599" not one.
        """
        day = brief.retry_on
        assert day is not None
        for form in (
            day.strftime("%d/%m/%Y"),
            day.strftime("%d-%m-%Y"),
            day.strftime("%d.%m.%Y"),
            day.isoformat(),
            day.strftime("%d/%m/%y"),
        ):
            if form in text:
                return True
        # Day beside a month name, in either order, with the day written with
        # or without its leading zero.
        months = (day.strftime("%b"), day.strftime("%B"))
        days = (str(day.day), f"{day.day:02d}")
        folded = text.casefold()
        return any(
            re.search(
                rf"(?<!\d){re.escape(d)}(?!\d)\s*(?:of\s+)?{re.escape(m.casefold())}"
                rf"|{re.escape(m.casefold())}\s+(?<!\d){re.escape(d)}(?!\d)",
                folded,
            )
            is not None
            for d in days
            for m in months
        )


# ==========================================================================
# Lexical checks
# ==========================================================================

#: Anything that would make this message look like credential phishing.
#:
#: Barred outright, including inside a "never share your OTP" safety footer.
#: Telling a warning from a request needs intent, the token is worthless in a
#: payment reminder either way, and the cost of getting it wrong in the other
#: direction is a customer handing an OTP to whoever asks next.
_CREDENTIAL_MARKERS: tuple[str, ...] = (
    "otp", "o.t.p", "one time password", "one-time password", "upi pin",
    "atm pin", "mpin", "cvv", "card number", "full card number",
    "aadhaar", "aadhar", "net banking password", "security code",
    # The paraphrases. A solicitation does not have to name the thing it
    # wants: "reply with the 6-digit code we just sent" carries none of the
    # words above and is the same attack.
    "verification code", "digit code", "code we sent", "code sent to",
    "share the code", "confirm the code", "sms code",
    "ओटीपी", "पिन", "सीवीवी", "पासवर्ड", "आधार", "कोड",
)

#: Debt-collection language. RBI's fair-practices expectations bar coercion,
#: and a recurring-payments failure is not a default in any case: the customer
#: agreed to pay and a rail said no.
#:
#: Whole words only; the stems below carry everything that inflects.
_COERCION_MARKERS: tuple[str, ...] = (
    "legal action", "legal notice", "court", "courts", "police",
    "recovery agent", "black list", "consequences will", "final warning",
    "immediately or", "kanooni", "kanuni", "vasooli",
    "क़ानूनी", "कानूनी", "अदालत", "पुलिस", "वसूली",
)

#: Markers matched as prefixes, because English inflects and a word list that
#: does not was a check that could not fire.
#:
#: ``"prosecut"`` sat in the whole-word list, where ``_word_present``'s
#: trailing ``(?![a-z])`` made it matchable only by the string "prosecut",
#: which is not a word. "We will prosecute you for this outstanding amount"
#: passed the entire verifier. The same held for ``blacklisted``,
#: ``defaulters``, ``seized``, ``seizure`` and ``passwords`` — every marker
#: whose plural or past tense is the form anyone would actually write.
#:
#: Kept separate from the whole-word list rather than making everything a
#: prefix, because "court" as a prefix matches "courtesy" and a polite message
#: would be rejected for good manners.
_COERCION_STEMS: tuple[str, ...] = (
    "prosecut", "blacklist", "defaulter", "seiz", "criminal", "penalis",
    "penaliz", "threaten", "litigat",
)

#: Credential markers matched as prefixes. ``password`` covers ``passwords``;
#: ``pin`` deliberately stays a whole word, or it would fire on "pink".
_CREDENTIAL_STEMS: tuple[str, ...] = ("password",)

#: Whole-word credential markers short enough to collide if used as prefixes.
_CREDENTIAL_WORDS: tuple[str, ...] = ("pin",)


def _word_present(marker: str, text: str) -> bool:
    if " " in marker or not marker.isascii():
        return marker in text
    return re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", text) is not None


def _stem_present(stem: str, text: str) -> bool:
    """Whether ``text`` contains a word beginning with ``stem``.

    Left boundary only, so ``seiz`` matches ``seized`` and ``seizure`` but not
    the tail of some longer unrelated word.
    """
    return re.search(rf"(?<![a-z]){re.escape(stem)}", text) is not None


@dataclass(frozen=True, slots=True)
class NoCredentialSolicitation:
    """The message must not mention a credential at all.

    A payment reminder that says the word OTP is, from the recipient's side,
    the exact shape of the fraud they are warned about weekly. This check is a
    word list and therefore weak in one direction only: it cannot catch a
    solicitation phrased without any of these words, but nothing it passes
    contains one.
    """

    check_id: str = "no_credential_solicitation"
    category: Category = Category.SAFETY

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        text = draft.rendered().casefold()
        for marker in (*_CREDENTIAL_MARKERS, *_CREDENTIAL_WORDS):
            if _word_present(marker, text):
                return _finding(
                    self,
                    f"mentions {marker!r}; a payment message never refers to a "
                    "PIN, OTP, CVV or password, not even to warn about one",
                )
        for stem in _CREDENTIAL_STEMS:
            if _stem_present(stem, text):
                return _finding(
                    self,
                    f"mentions {stem!r}; a payment message never refers to a "
                    "PIN, OTP, CVV or password, not even to warn about one",
                )
        return None


@dataclass(frozen=True, slots=True)
class NoCoercion:
    """No threats, no collections vocabulary.

    Honestly a weak check: it is a lexicon, and a politely-worded threat that
    avoids every word on the list passes. It is here because the failure it
    guards against is a model reaching for the register it has read most often
    in "overdue payment" text, and that register is full of these exact words.
    """

    check_id: str = "no_coercion"
    category: Category = Category.SAFETY

    def run(self, draft: Draft, brief: MessageBrief) -> Finding | None:
        text = draft.rendered().casefold()
        for marker in _COERCION_MARKERS:
            if _word_present(marker, text):
                return _finding(
                    self,
                    f"uses collections language ({marker!r}); a failed autopay "
                    "is not a default and the message must not threaten",
                )
        for stem in _COERCION_STEMS:
            if _stem_present(stem, text):
                return _finding(
                    self,
                    f"uses collections language ({stem!r}); a failed autopay "
                    "is not a default and the message must not threaten",
                )
        return None


# ==========================================================================


#: Every check, in the order findings are reported.
#:
#: Ordering is deliberate. Fidelity failures come first because they are the
#: ones a repair pass can actually act on, and a model given six complaints
#: fixes the first and drifts on the rest.
CHECKS: tuple[Check, ...] = (
    AmountIsExact(),
    NoFabricatedIdentifiers(),
    LinksAreOurs(),
    PreDebitDisclosure(),
    NoInternalCodes(),
    NoCredentialSolicitation(),
    NoCoercion(),
    SenderIsIdentified(),
    AskIsHonoured(),
    ScriptMatchesLanguage(),
    RendersOnChannel(),
    SmsStaysInGsm7(),
    WithinChannelBudget(),
)


def verify(draft: Draft, brief: MessageBrief) -> tuple[Finding, ...]:
    """Every reason this draft cannot be sent. Empty means it can.

    Runs all checks rather than stopping at the first, because a repair pass
    that fixes one fault and is then told about the next wastes a round trip
    per fault, and because the measured breakdown of *what* models get wrong is
    only meaningful if every draft is scored against every check.
    """
    findings = []
    for check in CHECKS:
        finding = check.run(draft, brief)
        if finding is not None:
            findings.append(finding)
    return tuple(findings)


def feedback_for(findings: tuple[Finding, ...]) -> str:
    """The findings as an instruction to a drafter attempting a repair."""
    lines = "\n".join(f"- The message {finding.detail}." for finding in findings)
    return (
        "The previous draft was rejected by an automated check for these "
        f"reasons:\n{lines}\nRewrite it so that none of them apply. Keep "
        "everything that was already correct."
    )
