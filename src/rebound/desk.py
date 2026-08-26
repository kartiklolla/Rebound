"""Turning a brief into a message that is cleared to send.

The loop is generate, verify, repair once, fall back. Each step exists for a
reason that showed up rather than one that was anticipated:

*Generate.* A language model, because the hard part is register — a two-segment
Hinglish SMS that sounds like a merchant rather than a bank circular — and that
is the one thing here templates are genuinely bad at.

*Verify.* :mod:`rebound.verify`, deterministically, against the brief.

*Repair once.* Feeding the findings back recovers most failures, because the
usual fault is a single fabricated detail rather than a bad message. Once, not
until it passes: a model that has failed twice on the same brief is not
converging, and each further attempt costs a call and delays a send.

*Fall back.* A template that is proven to pass every check for every
combination of instruction, channel and language. This is what makes the model
safe to use at all. Without a fallback the failure mode is either sending
something unverified or sending nothing, and sending nothing is a recovery
silently dropped.

The fallback rate is therefore not an embarrassment to be minimised out of
sight. It is the measurement that says how much the model is being trusted,
and :func:`compose` records it per check so the answer is a breakdown rather
than a number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Protocol

from .comms import (
    Ask,
    Channel,
    Draft,
    Language,
    MessageBrief,
)
from .taxonomy import Rail
from .verify import Finding, feedback_for, verify

__all__ = [
    "Attempt",
    "CommsDesk",
    "Composition",
    "Drafter",
    "TemplateDrafter",
    "AnthropicDrafter",
    "DEFAULT_MODEL",
    "SYSTEM_PROMPT",
    "render_template",
]


class Drafter(Protocol):
    """Anything that can turn a brief into a candidate message."""

    name: str

    def draft(
        self,
        brief: MessageBrief,
        *,
        previous: Draft | None = None,
        feedback: str | None = None,
    ) -> Draft: ...


# ==========================================================================
# Templates
# ==========================================================================
#
# These are the fallback, which makes them the most load-bearing prose in the
# system: they are what a customer receives on the worst day, when the model
# is down or has just failed verification twice. Every combination of
# instruction, channel and language below is asserted to pass every check in
# tests/test_desk.py, and that assertion is the reason the model is allowed
# anywhere near a customer.
#
# Three languages, because a language ships when someone who speaks it has
# read its fallback. See ``comms.REVIEWED_LANGUAGES``.

_RAIL_WORDS: dict[Rail, dict[Language, str]] = {
    Rail.UPI_AUTOPAY: {
        Language.EN: "UPI Autopay",
        Language.HINGLISH: "UPI Autopay",
        Language.HI: "UPI ऑटोपे",
    },
    Rail.ENACH: {
        Language.EN: "e-NACH",
        Language.HINGLISH: "e-NACH",
        Language.HI: "e-NACH",
    },
    Rail.CARD_ON_FILE: {
        Language.EN: "card",
        Language.HINGLISH: "card",
        Language.HI: "कार्ड",
    },
}

_LEAD: dict[Language, str] = {
    Language.EN: "{merchant}: your {rail} payment of {amount} did not go through.",
    Language.HINGLISH: "{merchant}: aapka {amount} ka {rail} payment nahi hua.",
    Language.HI: "{merchant}: आपका {amount} का {rail} भुगतान नहीं हो सका।",
}

_ASK_LINE: dict[Ask, dict[Language, str]] = {
    Ask.KEEP_BALANCE: {
        Language.EN: "Please keep sufficient balance in your account and we will try again.",
        Language.HINGLISH: "Kripya khate mein raashi rakhein, hum dobara try karenge.",
        Language.HI: "कृपया खाते में राशि रखें, हम दोबारा प्रयास करेंगे।",
    },
    Ask.APPROVE_IN_APP: {
        Language.EN: "Please approve the pending mandate request from your bank.",
        Language.HINGLISH: "Kripya apne bank ka pending mandate request approve karein.",
        Language.HI: "कृपया अपने बैंक का लंबित अनुरोध स्वीकृत करें।",
    },
    Ask.UPDATE_CARD: {
        Language.EN: "Please update your card details to continue.",
        Language.HINGLISH: "Kripya apna card update karein, tabhi aage jama hoga.",
        Language.HI: "कृपया अपना कार्ड अपडेट करें।",
    },
    Ask.PAY_NOW_VIA_LINK: {
        Language.EN: "You can pay now here: {link}",
        Language.HINGLISH: "Abhi bhugtan karein: {link}",
        Language.HI: "अभी भुगतान करें: {link}",
    },
    Ask.AMEND_MANDATE: {
        Language.EN: (
            "Your autopay limit no longer covers this amount. "
            "Please update the mandate to continue."
        ),
        Language.HINGLISH: (
            "Aapki autopay limit is raashi ke liye kam hai. "
            "Kripya mandate update karein."
        ),
        Language.HI: "आपकी ऑटोपे सीमा कम है। कृपया मैंडेट अपडेट करें।",
    },
    Ask.REAUTHORISE_MANDATE: {
        Language.EN: "Please set up your autopay mandate again to continue.",
        Language.HINGLISH: "Kripya apna autopay mandate dobara set up karein.",
        Language.HI: "कृपया अपना ऑटोपे मैंडेट फिर से बनाएं।",
    },
    Ask.NOTHING: {
        Language.EN: "No action is needed from you.",
        Language.HINGLISH: "Aapko kuch karne ki zaroorat nahi hai.",
        Language.HI: "आपको कुछ करने की ज़रूरत नहीं है।",
    },
}

#: The pre-debit notice does not lead with a failure, because nothing has
#: failed yet — it announces a debit that is about to happen. Its three
#: disclosures (amount, date, mandate reference) are checked by
#: ``verify.PreDebitDisclosure``.
_NOTICE: dict[Language, str] = {
    Language.EN: (
        "{merchant}: {amount} will be debited on {date} under mandate "
        "{reference}. No action is needed from you."
    ),
    Language.HINGLISH: (
        "{merchant}: {amount} {date} ko mandate {reference} se kata jayega. "
        "Aapko kuch karne ki zaroorat nahi hai."
    ),
    Language.HI: (
        "{merchant}: {amount} {date} को मैंडेट {reference} से लिया जाएगा। "
        "आपको कुछ करने की ज़रूरत नहीं है।"
    ),
}

_HELP: dict[Language, str] = {
    Language.EN: "Questions? Call {support}.",
    Language.HINGLISH: "Sawaal hai? {support} par call karein.",
    Language.HI: "प्रश्न हैं? {support} पर कॉल करें।",
}

_SUBJECTS: dict[Language, str] = {
    Language.EN: "{merchant}: your payment of {amount} needs attention",
    Language.HINGLISH: "{merchant}: aapka {amount} ka payment nahi hua",
    Language.HI: "{merchant}: आपका {amount} का भुगतान नहीं हो सका",
}

_NOTICE_SUBJECTS: dict[Language, str] = {
    Language.EN: "{merchant}: {amount} will be debited on {date}",
    Language.HINGLISH: "{merchant}: {amount} {date} ko kata jayega",
    Language.HI: "{merchant}: {amount} {date} को लिया जाएगा",
}


def render_template(brief: MessageBrief) -> Draft:
    """The deterministic message for a brief. Always passes verification.

    Signs with ``merchant.signature``, not ``merchant.name``. The sender check
    was moved off the legal entity when it turned out that "Vahan Technologies
    Private Limited" is a fifth of a GSM-7 segment; the templates were not, so
    they kept emitting it — and a Hindi SMS, already on a 134-unit UCS-2 budget,
    went over. The fallback failed, which means nothing at all would have been
    sent. Found by running scripts/demo.py with a realistic merchant name, on
    the first attempt.

    "Always" is asserted rather than asserted-to: ``test_desk`` renders every
    instruction against every channel and language and runs the full check set
    over each one. A fallback that can fail is not a fallback.
    """
    language = brief.language
    amount = brief.amount_text
    date = brief.retry_on.strftime("%d/%m/%Y") if brief.retry_on else ""

    if brief.ask is Ask.NOTHING and brief.retry_on is not None:
        body = _NOTICE[language].format(
            merchant=brief.merchant.signature,
            amount=amount,
            date=date,
            reference=brief.mandate_reference,
        )
    else:
        lead = _LEAD[language].format(
            merchant=brief.merchant.signature,
            rail=_RAIL_WORDS[brief.rail][language],
            amount=amount,
        )
        instruction = _ASK_LINE[brief.ask][language].format(link=brief.link or "")
        body = f"{lead} {instruction}"

    # SMS is billed by segment and a third segment buys nothing a recipient
    # reads, so the help line is dropped there. Every other channel carries it.
    if brief.channel is not Channel.SMS:
        body = f"{body} {_HELP[language].format(support=brief.merchant.support_number)}"

    subject = None
    if brief.spec.subject_required:
        template = (
            _NOTICE_SUBJECTS if brief.ask is Ask.NOTHING and brief.retry_on else _SUBJECTS
        )
        subject = template[language].format(
            merchant=brief.merchant.signature, amount=amount, date=date
        )

    return Draft(
        body=body, subject=subject, language=language, produced_by="template"
    )


@dataclass(frozen=True, slots=True)
class TemplateDrafter:
    """The fallback, usable on its own as a no-model baseline.

    Ignores ``feedback`` because there is nothing to repair: its output is
    fixed by the brief. That is also what makes it a fair comparison point —
    running the whole system with this as the primary drafter measures what the
    model is worth.
    """

    name: str = "template"

    def draft(
        self,
        brief: MessageBrief,
        *,
        previous: Draft | None = None,
        feedback: str | None = None,
    ) -> Draft:
        return render_template(brief)


# ==========================================================================
# The model
# ==========================================================================

DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """\
You write payment reminders for an Indian merchant. Every message you write is \
checked by a program before it is sent, and anything the program rejects is \
thrown away and replaced by a fixed template, so inventing a detail does not \
get it delivered — it only wastes the message.

Rules, all of them enforced:

1. State the amount exactly as given, once. Never any other figure.
2. Never invent a reference number, a date, a phone number or a URL. Use only \
the ones in the brief. If a fact is not in the brief, leave it out.
3. Say the one thing the brief's INSTRUCTION asks for, and never a different \
one. If the instruction is to keep a balance, do not mention updating a card.
4. Never mention an OTP, PIN, CVV, password or Aadhaar, not even to warn \
about them.
5. Never threaten. This is a failed autopay, not a default — the customer \
agreed to pay and a bank said no. No legal language, no deadlines that sound \
like consequences.
6. Name the merchant.
7. Write in the language asked for. "Hinglish" means Hindi in Latin script — \
no Devanagari at all. "Hindi" means Devanagari.
8. Respect the length limit. SMS is billed per segment and Devanagari uses \
more than twice the space per character.
9. No internal codes, no jargon, no emoji unless told they are allowed.

Reply with JSON only: {"body": "...", "subject": "..."}. Omit "subject" unless \
the brief says a subject is required.\
"""


def _brief_prompt(brief: MessageBrief) -> str:
    lines = [
        f"MERCHANT: {brief.merchant.signature}",
        f"AMOUNT: write it exactly as {brief.amount_text}",
        f"CHANNEL: {brief.channel}",
        f"LANGUAGE: {brief.language}",
        f"INSTRUCTION: {brief.ask.replace('_', ' ')}",
        f"WHAT HAPPENED: a recurring {_RAIL_WORDS[brief.rail][Language.EN]} "
        "debit could not be collected",
        f"SUPPORT NUMBER: {brief.merchant.support_number}",
        f"MANDATE REFERENCE: {brief.mandate_reference}",
    ]
    if brief.retry_on is not None:
        lines.append(f"DATE: {brief.retry_on:%d/%m/%Y}")
    if brief.link is not None:
        lines.append(f"LINK (the only URL allowed): {brief.link}")
    else:
        lines.append("LINK: none — the message must contain no URL at all")
    spec = brief.spec
    if spec.max_segments is not None:
        budget = 134 if brief.language is Language.HI else 300
        lines.append(
            f"LENGTH: at most {budget} characters, and shorter is better"
        )
    else:
        lines.append(f"LENGTH: at most {spec.max_chars} characters")
    lines.append(
        "SUBJECT: required" if spec.subject_required else "SUBJECT: do not write one"
    )
    if not spec.emoji_allowed:
        lines.append("EMOJI: none")
    return "\n".join(lines)


#: The shape a reply must have, enforced by the API rather than by asking.
#:
#: Structured output is the right tool here and it replaces a worse one: the
#: first version asked for JSON in the prompt and parsed the reply tolerantly,
#: which meant "the model wrote a friendly sentence instead of JSON" was a
#: failure mode the desk had to absorb. Constraining the decode removes it at
#: the source. The tolerant parser is still below, because a response that
#: somehow is not the promised shape must degrade to a fallback rather than
#: raise.
REPLY_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "body": {"type": "string"},
        "subject": {"type": "string"},
    },
    "required": ["body"],
    "additionalProperties": False,
}


@dataclass
class AnthropicDrafter:
    """Drafts with a Claude model over the Messages API.

    The client is injected rather than constructed here so that tests can run
    the whole desk — including the repair pass — against a stub, and so that a
    caller can supply a client with their own retry and timeout policy. A
    drafter that builds its own HTTP client is a drafter nobody can test.

    No sampling parameters are passed. An earlier version sent
    ``temperature=0.4`` with a paragraph justifying the value; the parameter
    does not exist on this SDK's ``messages.create``, so the justification was
    for an argument that would have raised ``TypeError`` on the first live
    call. ``test_every_argument_we_send_exists_on_the_real_client`` now checks
    the signature so the next drift is caught without a network call.
    """

    client: object
    model: str = DEFAULT_MODEL
    max_tokens: int = 512

    @property
    def name(self) -> str:
        return f"model:{self.model}"

    def request(
        self,
        brief: MessageBrief,
        *,
        previous: Draft | None = None,
        feedback: str | None = None,
    ) -> dict[str, object]:
        """Exactly what would be sent. Separated so it can be inspected."""
        turns: list[dict[str, str]] = [
            {"role": "user", "content": _brief_prompt(brief)}
        ]
        if previous is not None and feedback is not None:
            turns.append({"role": "assistant", "content": _as_json(previous)})
            turns.append({"role": "user", "content": feedback})
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": turns,
            "output_config": {
                "format": {"type": "json_schema", "schema": REPLY_SCHEMA}
            },
        }

    def draft(
        self,
        brief: MessageBrief,
        *,
        previous: Draft | None = None,
        feedback: str | None = None,
    ) -> Draft:
        response = self.client.messages.create(  # type: ignore[attr-defined]
            **self.request(brief, previous=previous, feedback=feedback)
        )
        return self._parse(response, brief)

    def _parse(self, response: object, brief: MessageBrief) -> Draft:
        # Every part guarded, because this runs on whatever the network
        # returned. A block whose ``text`` was an int raised TypeError inside
        # join, which is an exception from the parse path rather than an empty
        # body and a clean fallback.
        text = "".join(
            block.text
            for block in getattr(response, "content", None) or ()
            if getattr(block, "type", None) == "text"
            and isinstance(getattr(block, "text", None), str)
        ).strip()
        payload = _loads(text)
        body = payload.get("body")
        subject = payload.get("subject")
        # Only a string counts. Anything else — a list, a number, a nested
        # object — becomes an empty body, which fails the length check and
        # falls back. Coercing it with str() would send the repr of a data
        # structure to a customer.
        return Draft(
            body=body.strip() if isinstance(body, str) else "",
            subject=(
                subject.strip()
                if isinstance(subject, str) and subject.strip()
                else None
            ),
            language=brief.language,
            produced_by=self.name,
        )


def _as_json(draft: Draft) -> str:
    payload: dict[str, str] = {"body": draft.body}
    if draft.subject:
        payload["subject"] = draft.subject
    return json.dumps(payload, ensure_ascii=False)


def _loads(text: str) -> dict[str, object]:
    """Parse the model's reply, tolerating a fenced block around the JSON.

    Deliberately tolerant of formatting and strictly intolerant of content: a
    reply that is not JSON at all yields an empty body, which fails
    ``WithinChannelBudget`` and falls back. Guessing at prose here would mean
    sending something that was never checked against the schema the prompt
    asked for.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ==========================================================================
# The desk
# ==========================================================================


@dataclass(frozen=True, slots=True)
class Attempt:
    """One draft and everything wrong with it. Kept whether or not it was used."""

    draft: Draft
    findings: tuple[Finding, ...]

    @property
    def cleared(self) -> bool:
        return not self.findings


@dataclass(frozen=True, slots=True)
class Composition:
    """What the desk produced, and the whole record of how.

    Rejected drafts are kept rather than discarded. They are the only evidence
    of what the model actually did, and a fallback rate reported without them
    is a number no reviewer can check.
    """

    brief: MessageBrief
    attempts: tuple[Attempt, ...]
    sent: Draft | None

    used_fallback: bool = False
    """Whether ``compose`` reached the fallback branch.

    Recorded by the loop rather than inferred from the sent draft. It used to
    be ``sent.produced_by == "template"``, which is a string the drafter writes
    — so running the desk with :class:`TemplateDrafter` as the *primary*
    drafter, which is the no-model baseline, reported 100% fallback for a run
    that never fell back once. A drafter could also simply claim the name and
    launder its own output as the safe path. Whether the fallback ran is a fact
    about control flow and is the loop's to state.
    """

    @property
    def cleared(self) -> bool:
        return self.sent is not None

    @property
    def fell_back(self) -> bool:
        """Whether the message that went out came from the fallback drafter."""
        return self.cleared and self.used_fallback

    @property
    def repaired(self) -> bool:
        """Whether a first draft failed and a later attempt cleared.

        Does not require ``not fell_back``. A run that failed once, was
        repaired, and then failed again and fell back was previously booked as
        neither repaired nor anything else, which quietly lost the repair from
        the tally.
        """
        return self.cleared and any(
            not attempt.cleared for attempt in self.attempts
        )

    @property
    def findings(self) -> tuple[Finding, ...]:
        """Every finding across every attempt, for the measured breakdown."""
        return tuple(f for attempt in self.attempts for f in attempt.findings)

    def record(self) -> dict[str, object]:
        """One row of the audit trail.

        Carries the sent text and the rejected ones. The point of storing a
        rejection is that "the model tried to invent a reference number and was
        stopped" is only demonstrable if the attempt survives.

        Stores ``rendered()`` rather than ``body``. An email subject is part of
        what the checks read and part of what the customer receives, and
        recording only the body meant "the record names what was sent" was
        false for every email.
        """
        return {
            "episode_id": self.brief.episode_id,
            "customer_id": self.brief.customer_id,
            "action": str(self.brief.action),
            "channel": str(self.brief.channel),
            "language": str(self.brief.language),
            "ask": str(self.brief.ask),
            "cleared": self.cleared,
            "fell_back": self.fell_back,
            "attempts": len(self.attempts),
            "repaired": self.repaired,
            "sent": self.sent.rendered() if self.sent else None,
            "sent_subject": self.sent.subject if self.sent else None,
            "produced_by": self.sent.produced_by if self.sent else None,
            "rejected": tuple(
                {
                    "body": attempt.draft.rendered(),
                    "produced_by": attempt.draft.produced_by,
                    "failed": tuple(
                        (f.check_id, f.detail) for f in attempt.findings
                    ),
                }
                for attempt in self.attempts
                if not attempt.cleared
            ),
        }


@dataclass
class CommsDesk:
    """Generate, verify, repair once, fall back.

    ``compose`` is total: it either returns a cleared message or returns a
    composition with ``sent is None``, and never raises on a bad draft. A
    drafter that raises — a timeout, a refusal, malformed JSON — is treated as
    a failed attempt rather than an error, because the alternative is that one
    unreachable API takes down a batch run that had a working fallback the
    whole time.
    """

    drafter: Drafter
    fallback: Drafter = field(default_factory=TemplateDrafter)
    max_repairs: int = 1

    def __post_init__(self) -> None:
        if self.max_repairs < 0:
            # A negative budget made the loop body run zero times, so the
            # drafter was never called and the composition reported a clean
            # fallback as though a model had been tried and had failed.
            raise ValueError("max_repairs cannot be negative")

    def _attempt(self, drafter: Drafter, brief: MessageBrief, **kwargs) -> Attempt:
        """One call to a drafter, with everything that can go wrong contained.

        The drafter is handed a *copy* of the brief and the original is what
        the draft is verified against. ``MessageBrief`` is frozen, which stops
        assignment but not ``object.__setattr__`` — so a drafter could rewrite
        ``brief.link`` to a host it controlled, quote the new value, and have
        the verifier confirm the message matched the brief. It did: a phishing
        host came back from ``compose`` cleared, with ``fell_back=False``. The
        check was not evaded; the ground truth was moved.

        Structurally the same defect the harness had, where the report was read
        off the object handed to the untrusted component. That was fixed by
        cutting the object graph with ``EpisodeView``; this cuts it with a copy.

        A real ``AnthropicDrafter`` cannot do this — a model returns text, not
        code — but ``Drafter`` is an exported extension point, and a guarantee
        that holds only for the implementations shipped today is not one.
        """
        try:
            draft = drafter.draft(replace(brief), **kwargs)
        except Exception as error:  # noqa: BLE001 - see the class docstring
            return Attempt(
                draft=Draft(
                    body="",
                    language=brief.language,
                    produced_by=getattr(drafter, "name", "drafter"),
                ),
                findings=(_drafter_failed(error),),
            )
        if not isinstance(draft, Draft):
            # Everything downstream — verification, the audit record, the send
            # — assumes an immutable Draft. A duck-typed object whose
            # ``rendered()`` returns one thing when verified and another when
            # read back passes every check and delivers something else.
            return Attempt(
                draft=Draft(
                    body="",
                    language=brief.language,
                    produced_by=getattr(drafter, "name", "drafter"),
                ),
                findings=(
                    _drafter_failed(
                        TypeError(
                            f"returned {type(draft).__name__}, not a Draft"
                        )
                    ),
                ),
            )
        return Attempt(draft=draft, findings=verify(draft, brief))

    def compose(self, brief: MessageBrief) -> Composition:
        attempts: list[Attempt] = []
        previous: Draft | None = None
        feedback: str | None = None

        for _ in range(1 + self.max_repairs):
            attempt = self._attempt(
                self.drafter, brief, previous=previous, feedback=feedback
            )
            attempts.append(attempt)
            if attempt.cleared:
                return Composition(
                    brief=brief,
                    attempts=tuple(attempts),
                    sent=attempt.draft,
                    used_fallback=False,
                )
            if any(f.check_id == "drafter_failed" for f in attempt.findings):
                break
            previous, feedback = attempt.draft, feedback_for(attempt.findings)

        if self.fallback is self.drafter or self.fallback == self.drafter:
            # Nothing left to fall back to; the primary already failed. Value
            # equality as well as identity, because both drafters are frozen
            # dataclasses and two TemplateDrafter() instances are equal but not
            # identical — an identity check alone re-rendered the same text and
            # called the second one a fallback.
            return Composition(brief=brief, attempts=tuple(attempts), sent=None)

        final = self._attempt(self.fallback, brief)
        attempts.append(final)
        return Composition(
            brief=brief,
            attempts=tuple(attempts),
            sent=final.draft if final.cleared else None,
            used_fallback=True,
        )


def _drafter_failed(error: Exception) -> Finding:
    from .verify import Category

    return Finding(
        check_id="drafter_failed",
        category=Category.FORMAT,
        detail=f"could not be produced: {type(error).__name__}: {error}",
    )
