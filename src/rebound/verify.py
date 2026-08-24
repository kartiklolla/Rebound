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

#: Top-level domains a link in one of these messages could plausibly use.
#:
#: An allow-list rather than ``\w+\.\w+`` because the alternative matches
#: "Rs.1,299", "e-NACH" and the join between a full stop and the next sentence.
_TLDS = (
    "in|com|co|net|org|io|app|me|biz|info|xyz|link|page|site|online|shop|"
    "store|bank|pay|ind|gov"
)

#: A URL, with or without a scheme.
#:
#: The scheme has to be optional, and that is not a nicety. Indian
#: transactional SMS routinely carries a bare host to save characters, so a
#: model imitating the register writes one — and against a scheme-only pattern
#: ``evil.example.com/pay`` matched nothing, passed every check including the
#: outright bar on voice scripts, and would have been sent. A verifier that
#: only sees the well-formed half of a threat is not a verifier.
_URL = re.compile(
    rf"(?P<scheme>https?://|www\.)[^\s<>\"')\]]+"
    rf"|(?<![@\w.])(?P<bare>(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:{_TLDS})\b"
    rf"(?P<path>/[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)


def _find_urls(text: str) -> list[str]:
    """Every URL in ``text``, with sentence breaks filtered out.

    The bare-host branch has to be case-insensitive to catch a shouted
    ``EVIL.EXAMPLE.COM/pay``, and that makes an English full stop followed by a
    capitalised word look like a host: "keep sufficient balance.In case of
    trouble" matched ``balance.In``, and "the account.In future" matched
    ``account.In``. A missing space after a full stop is a typo a model
    produces, and rejecting the message for it forces a fallback over nothing.

    So a host with no scheme has to look like a host in one of three ways: it
    has a path, or it has three or more labels, or it is written in lower case
    the way a bare host in an SMS actually is. ``balance.In`` is none of those.
    A real host loses nothing — ``vahan.in`` is lower case, ``pay.vahan.in``
    has three labels, and anything with a path qualifies outright.
    """
    urls: list[str] = []
    for match in _URL.finditer(text):
        if match.group("scheme"):
            urls.append(match.group(0))
            continue
        token = match.group("bare")
        host = token.split("/", 1)[0]
        if match.group("path") or host.count(".") >= 2 or host.islower():
            urls.append(token)
    return urls


def _strip_urls(text: str) -> str:
    """``text`` with its URLs blanked out, for checks that must ignore them."""
    for url in _find_urls(text):
        text = text.replace(url, " ")
    return text

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


#: Digit runs at or above this length are treated as identifiers rather than
#: incidental numbers.
#:
#: Four would flag years and times. Six would let a five-digit invented
#: reference through. Five sits above everything a message says in passing —
#: dates, hours, a day of the month — and below every real identifier on these
#: rails, where a UMRN and a customer-care number are both far longer.
_IDENTIFIER_DIGITS = 5

#: A token that mixes letters and digits, which is what a reference number
#: looks like on every rail here.
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
        facts = self._facts(brief)
        permitted = brief.permitted_numerals

        for run in re.findall(r"\d+", text):
            if len(run) >= _IDENTIFIER_DIGITS and run not in permitted:
                return _finding(
                    self,
                    f"contains the number {run}, which is not the amount, the "
                    f"mandate reference {brief.mandate_reference} or the "
                    f"support number — remove it",
                )
        for token in _ALNUM_TOKEN.findall(text):
            if not any(ch.isdigit() for ch in token):
                continue
            if not any(_bare(token) in fact for fact in facts):
                return _finding(
                    self,
                    f"contains the reference {token!r}, which is not one of "
                    "ours — quote only the mandate reference "
                    f"{brief.mandate_reference}",
                )
        return None

    @staticmethod
    def _facts(brief: MessageBrief) -> tuple[str, ...]:
        return tuple(_bare(fact) for fact in brief.quotable_facts)


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
        if not found:
            if brief.ask is Ask.PAY_NOW_VIA_LINK:
                return _finding(
                    self, f"must contain the payment link {brief.link}"
                )
            return None
        if not brief.spec.links_allowed:
            return _finding(
                self, f"{brief.channel} messages carry no links; remove {found[0]}"
            )
        if brief.link is None:
            return _finding(
                self, f"contains a link ({found[0]}) but this message has none to send"
            )
        # Compared without the scheme and without trailing punctuation. A
        # message that writes our own link as a bare host is writing the same
        # link, and rejecting it forces a fallback over a formatting choice —
        # while the case that actually matters, a *different* host, is caught
        # either way.
        ours = _canonical_url(brief.link)
        stray = [url for url in found if _canonical_url(url) != ours]
        if stray:
            return _finding(
                self,
                f"links to {stray[0]}, which is not ours — the only permitted "
                f"URL is {brief.link}",
            )
        return None


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
        if brief.merchant.name.casefold() not in draft.rendered().casefold():
            return _finding(
                self, f"does not name the sender; it must say {brief.merchant.name}"
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
        folded = text.casefold()
        missing = []
        if brief.mandate_reference.casefold() not in folded:
            missing.append(f"the mandate reference {brief.mandate_reference}")
        if brief.retry_on is None:
            missing.append("a debit date (none was supplied to the brief)")
        elif not any(
            form in text
            for form in (
                brief.retry_on.strftime("%d/%m/%Y"),
                brief.retry_on.strftime("%d-%m-%Y"),
                brief.retry_on.isoformat(),
                brief.retry_on.strftime("%d %b"),
                brief.retry_on.strftime("%d %B"),
                str(brief.retry_on.day),
            )
        ):
            missing.append(f"the debit date {brief.retry_on:%d/%m/%Y}")
        if missing:
            return _finding(
                self,
                "is a pre-debit notice and omits " + " and ".join(missing),
            )
        return None


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
    "atm pin", "mpin", "cvv", "card number", "full card number", "password",
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
_COERCION_MARKERS: tuple[str, ...] = (
    "legal action", "legal notice", "court", "police", "recovery agent",
    "blacklist", "black list", "defaulter", "seize", "prosecut", "criminal",
    "consequences will", "final warning", "immediately or",
    "kanooni", "kanuni", "vasooli",
    "क़ानूनी", "कानूनी", "अदालत", "पुलिस", "वसूली",
)


def _word_present(marker: str, text: str) -> bool:
    if " " in marker or not marker.isascii():
        return marker in text
    return re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", text) is not None


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
        for marker in _CREDENTIAL_MARKERS:
            if _word_present(marker, text):
                return _finding(
                    self,
                    f"mentions {marker!r}; a payment message never refers to a "
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
