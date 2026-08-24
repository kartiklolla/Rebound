"""Tests for the templates and for the generate/verify/repair/fall-back loop.

Two things have to hold or the layer is unsafe rather than merely imperfect.

The first is that the fallback always passes. It is what goes out when the
model is unreachable or has just failed twice, so a fallback that can fail is
not a fallback — it is a second way to send nothing.
:class:`TestEveryTemplateClears` renders every instruction against every
channel and language and runs the full check set over each one.

The second is that nothing unverified is ever returned as sent.
:class:`TestNothingUnverifiedEscapes` puts a drafter that emits deliberately
dangerous text behind the desk and asserts the desk keeps it.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import itertools
import json

import pytest

from rebound.comms import (
    Ask,
    Channel,
    Draft,
    Language,
    MerchantProfile,
    MessageBrief,
    segments_for,
)
from rebound.desk import (
    AnthropicDrafter,
    CommsDesk,
    TemplateDrafter,
    render_template,
)
from rebound.taxonomy import Action, Disposition, Rail
from rebound.verify import verify

MERCHANT = MerchantProfile(
    name="Vahan",
    support_number="18002670001",
    link_host="pay.vahan.in",
    sender_id="VAHANX",
)
LINK = "https://pay.vahan.in/r/7Kd2Qm"

#: The action each instruction actually arrives on, so briefs in these tests
#: are the shape the sequencer really produces rather than a convenient
#: fiction.
ACTION_FOR_ASK = {
    Ask.KEEP_BALANCE: Action.NUDGE_SMS,
    Ask.APPROVE_IN_APP: Action.NUDGE_SMS,
    Ask.UPDATE_CARD: Action.NUDGE_SMS,
    Ask.PAY_NOW_VIA_LINK: Action.SEND_COLLECT_LINK,
    Ask.AMEND_MANDATE: Action.REQUEST_MANDATE_AMENDMENT,
    Ask.REAUTHORISE_MANDATE: Action.REQUEST_REMANDATE,
    Ask.NOTHING: Action.SEND_PRE_DEBIT_NOTIFICATION,
}


def make_brief(
    *,
    ask: Ask = Ask.KEEP_BALANCE,
    channel: Channel = Channel.SMS,
    language: Language = Language.EN,
    amount_paise: int = 129900,
    rail: Rail = Rail.UPI_AUTOPAY,
) -> MessageBrief:
    link = LINK if ask is Ask.PAY_NOW_VIA_LINK else None
    return MessageBrief(
        episode_id="EP_1",
        customer_id="CUST_1",
        action=ACTION_FOR_ASK[ask],
        channel=channel,
        language=language,
        merchant=MERCHANT,
        amount_paise=amount_paise,
        mandate_reference="UMRN2024HDFC0009911",
        bank="HDFC Bank",
        rail=rail,
        disposition=Disposition.RETRY_TIMING,
        ask=ask,
        reference_date=dt.date(2026, 9, 3),
        retry_on=dt.date(2026, 9, 5),
        link=link,
    )


class ScriptedDrafter:
    """Returns prepared bodies in order, so a repair pass can be tested."""

    def __init__(self, *bodies: str, name: str = "model:test"):
        self.name = name
        self.bodies = list(bodies)
        self.calls: list[tuple[Draft | None, str | None]] = []

    def draft(self, brief, *, previous=None, feedback=None) -> Draft:
        self.calls.append((previous, feedback))
        body = self.bodies[min(len(self.calls) - 1, len(self.bodies) - 1)]
        subject = "Vahan: about your payment" if brief.spec.subject_required else None
        return Draft(
            body=body,
            subject=subject,
            language=brief.language,
            produced_by=self.name,
        )


class ExplodingDrafter:
    name = "model:exploding"

    def __init__(self, error: Exception):
        self.error = error
        self.calls = 0

    def draft(self, brief, *, previous=None, feedback=None) -> Draft:
        self.calls += 1
        raise self.error


GOOD_SMS = (
    "Vahan: your UPI Autopay payment of Rs.1,299 did not go through. "
    "Please keep sufficient balance in your account and we will try again."
)


class TestEveryTemplateClears:
    """The fallback must pass every check for every combination it can face.

    This is the assertion the rest of the design rests on. If it fails, the
    model cannot safely be used at all, because the thing that catches the
    model's mistakes has nowhere to hand off to.
    """

    @pytest.mark.parametrize(
        "ask, channel, language",
        list(itertools.product(Ask, Channel, Language)),
    )
    def test_the_template_passes_its_own_verifier(self, ask, channel, language):
        if ask is Ask.PAY_NOW_VIA_LINK and channel is Channel.VOICE:
            # A collect link needs a channel that can carry a URL, and
            # MessageBrief.build refuses the combination outright.
            pytest.skip("a voice script carries no link")
        brief = make_brief(ask=ask, channel=channel, language=language)
        draft = render_template(brief)
        findings = verify(draft, brief)
        assert findings == (), [f.detail for f in findings]

    @pytest.mark.parametrize(
        "amount", [1, 9900, 129900, 1500000, 150000000, 129950]
    )
    def test_the_template_holds_at_every_amount(self, amount):
        # Amount changes length, digit grouping and the number of digit runs
        # the fabrication check sees. A template that fits at ₹1,299 and
        # spills a segment at ₹15,00,000 fails only in production.
        brief = make_brief(amount_paise=amount, language=Language.HI)
        draft = render_template(brief)
        assert verify(draft, brief) == ()

    def test_hindi_costs_two_sms_segments_and_the_others_cost_one(self):
        # Measured, not assumed, and the reason Indian merchants send
        # Hinglish: the same reminder is twice the price in Devanagari.
        counts = {}
        for language in Language:
            body = render_template(
                make_brief(channel=Channel.SMS, language=language)
            ).body
            counts[language] = segments_for(body)[0]
        assert counts[Language.EN] == 1
        assert counts[Language.HINGLISH] == 1
        assert counts[Language.HI] == 2

    def test_sms_templates_drop_the_help_line_and_others_keep_it(self):
        # A third segment buys nothing a recipient reads and costs another
        # send. Everywhere that is not billed by segment keeps the number.
        sms = render_template(make_brief(channel=Channel.SMS)).body
        whatsapp = render_template(make_brief(channel=Channel.WHATSAPP)).body
        assert MERCHANT.support_number not in sms
        assert MERCHANT.support_number in whatsapp

    def test_an_email_template_has_a_subject_and_an_sms_has_none(self):
        assert render_template(make_brief(channel=Channel.EMAIL)).subject
        assert render_template(make_brief(channel=Channel.SMS)).subject is None

    def test_the_pre_debit_notice_does_not_open_with_a_failure(self):
        # Nothing has failed when a notice goes out. It announces a debit.
        body = render_template(
            make_brief(ask=Ask.NOTHING, channel=Channel.SMS)
        ).body
        assert "did not go through" not in body
        assert "will be debited" in body


class TestAgainstRealIdentifiers:
    """Render templates using the identifier formats the simulator actually issues.

    This class exists because a bug hid behind a convenient fixture. Every test
    above used the mandate reference ``UMRN2024HDFC0009911``, invented for
    readability. The simulator issues ``MND_0000001``
    (``sim/world.py``), which matches ``NoInternalCodes``' SCREAMING_SNAKE
    shape — while ``PreDebitDisclosure`` *requires* a notice to quote its
    mandate reference. The two checks contradicted each other and every
    pre-debit notice was blocked outright, fallback included, so no message
    went out at all. Nothing in a unit fixture could reveal that; only the real
    format could.
    """

    #: The shapes these fields actually take, plus the ones a real processor
    #: would hand over. Underscored and hyphenated forms both matter: the
    #: first collides with the internal-code check, the second with the
    #: identifier check's formatting sensitivity.
    REAL_REFERENCES = [
        "MND_0000001",
        "EP_00000001",
        "UMRN2024HDFC0009911",
        "HDFC0000012345678",
        "ICIC-8842-0091",
    ]

    @pytest.mark.parametrize("reference", REAL_REFERENCES)
    @pytest.mark.parametrize("language", list(Language))
    def test_a_pre_debit_notice_clears_with_a_real_reference(
        self, reference, language
    ):
        brief = dataclasses.replace(
            make_brief(ask=Ask.NOTHING, language=language),
            mandate_reference=reference,
        )
        draft = render_template(brief)
        assert reference in draft.body
        findings = verify(draft, brief)
        assert findings == (), [f.detail for f in findings]

    @pytest.mark.parametrize("reference", REAL_REFERENCES)
    def test_an_internal_code_is_still_caught_alongside_a_real_reference(
        self, reference
    ):
        # Exempting our own facts must not exempt everything code-shaped.
        brief = dataclasses.replace(
            make_brief(ask=Ask.NOTHING), mandate_reference=reference
        )
        draft = Draft(
            body=(
                f"Vahan: Rs.1,299 will be debited on 05/09/2026 under mandate "
                f"{reference}. Reason code RETRY_TIMING_HOLD applies."
            ),
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(draft, brief)
        assert any(f.check_id == "no_internal_codes" for f in findings)

    @pytest.mark.parametrize("ask", list(Ask))
    def test_every_instruction_clears_with_a_real_reference(self, ask):
        brief = dataclasses.replace(
            make_brief(ask=ask, channel=Channel.WHATSAPP),
            mandate_reference="MND_0000001",
        )
        draft = render_template(brief)
        findings = verify(draft, brief)
        assert findings == (), [f.detail for f in findings]


class TestNothingUnverifiedEscapes:
    """A drafter emitting dangerous text must not get it past the desk."""

    @pytest.mark.parametrize(
        "body",
        [
            "Vahan: pay Rs.99,999 now or we will begin legal action.",
            "Vahan: your Rs.1,299 payment failed. Share the OTP to confirm.",
            "Vahan: Rs.1,299 pending. Settle at evil.example.com/pay now.",
            "Vahan: Rs.1,299 failed, ref 998877. Keep sufficient balance.",
            "Vahan: Rs.1,299 failed (UPI_INSUFFICIENT_FUNDS). Keep balance.",
        ],
    )
    def test_the_dangerous_draft_is_never_the_one_sent(self, body):
        drafter = ScriptedDrafter(body)
        desk = CommsDesk(drafter=drafter)
        result = desk.compose(make_brief())
        assert result.cleared
        assert result.sent is not None
        assert result.sent.body != body
        assert result.fell_back
        # And it is still on the record, because "the model tried this and was
        # stopped" is only demonstrable if the attempt survives.
        assert any(body == r["body"] for r in result.record()["rejected"])

    def test_what_is_sent_always_passes_verification(self):
        for body in (GOOD_SMS, "nonsense", "Vahan: pay Rs.5 or else."):
            result = CommsDesk(drafter=ScriptedDrafter(body)).compose(
                make_brief()
            )
            assert result.sent is not None
            assert verify(result.sent, result.brief) == ()


class TestTheLoop:
    def test_a_clean_first_draft_is_sent_without_a_repair(self):
        drafter = ScriptedDrafter(GOOD_SMS)
        result = CommsDesk(drafter=drafter).compose(make_brief())
        assert result.sent is not None
        assert result.sent.produced_by == "model:test"
        assert not result.fell_back
        assert not result.repaired
        assert len(drafter.calls) == 1

    def test_a_failed_draft_is_repaired_and_the_repair_is_sent(self):
        drafter = ScriptedDrafter(
            "Vahan: your payment of Rs.9,999 failed. Keep sufficient balance.",
            GOOD_SMS,
        )
        result = CommsDesk(drafter=drafter).compose(make_brief())
        assert result.repaired
        assert not result.fell_back
        assert result.sent is not None and result.sent.body == GOOD_SMS
        assert len(drafter.calls) == 2

    def test_the_repair_pass_is_told_what_was_wrong(self):
        # Without the findings the second call is just a re-roll, and the
        # measured repair rate would be measuring luck.
        drafter = ScriptedDrafter(
            "Vahan: your payment of Rs.9,999 failed. Keep sufficient balance.",
            GOOD_SMS,
        )
        CommsDesk(drafter=drafter).compose(make_brief())
        previous, feedback = drafter.calls[1]
        assert previous is not None
        assert feedback is not None
        assert "1,299" in feedback
        assert "amount" in feedback.casefold()

    def test_two_failures_stop_rather_than_looping(self):
        # A model that has failed twice on one brief is not converging, and
        # each further attempt costs a call and delays the send.
        drafter = ScriptedDrafter("Vahan: pay Rs.5 or we take legal action.")
        result = CommsDesk(drafter=drafter).compose(make_brief())
        assert len(drafter.calls) == 2
        assert result.fell_back

    def test_max_repairs_zero_means_one_attempt(self):
        drafter = ScriptedDrafter("Vahan: pay Rs.5 or we take legal action.")
        result = CommsDesk(drafter=drafter, max_repairs=0).compose(make_brief())
        assert len(drafter.calls) == 1
        assert result.fell_back

    def test_a_drafter_that_raises_falls_back_instead_of_failing_the_batch(self):
        # One unreachable API must not take down a run that had a working
        # fallback the whole time.
        drafter = ExplodingDrafter(TimeoutError("upstream timed out"))
        result = CommsDesk(drafter=drafter).compose(make_brief())
        assert result.cleared
        assert result.fell_back
        assert drafter.calls == 1, "a raising drafter should not be retried"
        failed = result.record()["rejected"][0]["failed"]
        assert failed[0][0] == "drafter_failed"
        assert "TimeoutError" in failed[0][1]

    def test_the_template_alone_needs_no_fallback(self):
        # Running the desk with the template as the primary drafter is the
        # no-model baseline, and it must not report itself as having fallen
        # back to itself.
        template = TemplateDrafter()
        result = CommsDesk(drafter=template, fallback=template).compose(
            make_brief()
        )
        assert result.cleared
        assert len(result.attempts) == 1

    def test_a_broken_fallback_blocks_the_send_rather_than_guessing(self):
        # If both drafters fail, nothing goes out. Sending unverified text
        # because the safe path failed is the one outcome worse than silence.
        bad = ScriptedDrafter("nope", name="model:bad")
        result = CommsDesk(drafter=bad, fallback=ScriptedDrafter("also nope")).compose(
            make_brief()
        )
        assert not result.cleared
        assert result.sent is None


class TestTheRecord:
    def test_the_record_names_what_was_sent_and_what_was_stopped(self):
        drafter = ScriptedDrafter(
            "Vahan: Rs.1,299 failed, ref 998877. Keep sufficient balance."
        )
        record = CommsDesk(drafter=drafter).compose(make_brief()).record()
        assert record["episode_id"] == "EP_1"
        assert record["produced_by"] == "template"
        assert record["fell_back"] is True
        assert record["cleared"] is True
        checks = {c for r in record["rejected"] for c, _ in r["failed"]}
        assert "no_fabricated_identifiers" in checks

    def test_findings_span_every_attempt(self):
        drafter = ScriptedDrafter(
            "Vahan: pay Rs.9,999 or we take legal action.",
            "Vahan: pay Rs.8,888 now. Share your OTP.",
        )
        result = CommsDesk(drafter=drafter).compose(make_brief())
        ids = {f.check_id for f in result.findings}
        assert "no_coercion" in ids
        assert "no_credential_solicitation" in ids


class FakeBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class FakeResponse:
    def __init__(self, text: str):
        self.content = [FakeBlock(text)]


class FakeMessages:
    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        index = min(len(self.requests) - 1, len(self.replies) - 1)
        return FakeResponse(self.replies[index])


class FakeClient:
    def __init__(self, *replies: str):
        self.messages = FakeMessages(*replies)


class TestAnthropicDrafter:
    def test_a_json_reply_becomes_a_draft(self):
        client = FakeClient(json.dumps({"body": GOOD_SMS}))
        draft = AnthropicDrafter(client=client).draft(make_brief())
        assert draft.body == GOOD_SMS
        assert draft.produced_by.startswith("model:")
        assert draft.subject is None

    def test_a_fenced_reply_is_still_parsed(self):
        client = FakeClient(
            "```json\n" + json.dumps({"body": GOOD_SMS}) + "\n```"
        )
        draft = AnthropicDrafter(client=client).draft(make_brief())
        assert draft.body == GOOD_SMS

    def test_a_reply_that_is_not_json_yields_an_empty_body(self):
        # Which fails the length check and falls back. Guessing at prose here
        # means sending something never checked against the asked-for schema.
        client = FakeClient("Sure! Here's a lovely reminder for you.")
        draft = AnthropicDrafter(client=client).draft(make_brief())
        assert draft.body == ""
        result = CommsDesk(drafter=AnthropicDrafter(client=client)).compose(
            make_brief()
        )
        assert result.fell_back

    def test_the_brief_reaches_the_model_and_the_answer_key_does_not(self):
        client = FakeClient(json.dumps({"body": GOOD_SMS}))
        brief = make_brief()
        AnthropicDrafter(client=client).draft(brief)
        prompt = client.messages.requests[0]["messages"][0]["content"]
        assert "Rs.1,299" in prompt
        assert brief.merchant.support_number in prompt
        assert "keep balance" in prompt
        # The customer id is ours, not the customer's, and nothing in the
        # message could ever legitimately quote it.
        assert brief.customer_id not in prompt

    def test_a_repair_call_carries_the_previous_draft_and_the_findings(self):
        client = FakeClient(
            json.dumps({"body": "Vahan: pay Rs.9,999 or we take legal action."}),
            json.dumps({"body": GOOD_SMS}),
        )
        result = CommsDesk(drafter=AnthropicDrafter(client=client)).compose(
            make_brief()
        )
        assert result.repaired
        turns = client.messages.requests[1]["messages"]
        assert [turn["role"] for turn in turns] == ["user", "assistant", "user"]
        assert "9,999" in turns[1]["content"]
        assert "legal action" in turns[2]["content"]

    def test_the_system_prompt_states_the_rules_the_verifier_enforces(self):
        # Not decoration. A prompt that omits a rule the verifier enforces
        # produces a fallback rate that measures the prompt, not the model.
        from rebound.desk import SYSTEM_PROMPT

        lowered = SYSTEM_PROMPT.casefold()
        for phrase in ("otp", "hinglish", "never invent", "threaten", "amount"):
            assert phrase in lowered, phrase

    def test_every_argument_we_send_exists_on_the_real_client(self):
        """Catch SDK drift without a network call.

        This test exists because it would have caught a real bug. The drafter
        sent ``temperature=0.4`` with a paragraph in its docstring justifying
        the value, and ``messages.create`` on this SDK has no such parameter —
        so the first live call would have raised ``TypeError`` and every
        message in the run would have quietly fallen back to a template while
        the report showed a model in use.

        The stub client accepts anything, which is exactly why the stub cannot
        catch this and the signature can.
        """
        anthropic = pytest.importorskip("anthropic")
        import inspect

        real = anthropic.Anthropic(api_key="not-used-no-call-is-made")
        accepted = set(inspect.signature(real.messages.create).parameters)
        sent = set(
            AnthropicDrafter(client=FakeClient("{}")).request(make_brief())
        )
        assert sent <= accepted, f"not accepted by the SDK: {sorted(sent - accepted)}"

    def test_the_reply_shape_is_constrained_by_the_api_not_by_asking(self):
        client = FakeClient(json.dumps({"body": GOOD_SMS}))
        AnthropicDrafter(client=client).draft(make_brief())
        config = client.messages.requests[0]["output_config"]
        assert config["format"]["type"] == "json_schema"
        assert config["format"]["schema"]["required"] == ["body"]

    @pytest.mark.parametrize(
        "payload",
        ['{"body": ["a", "b"]}', '{"body": 42}', '{"subject": "x"}', "[]", '"hi"'],
    )
    def test_a_reply_of_the_wrong_shape_becomes_an_empty_body(self, payload):
        # Never str()-coerced. Coercion would send the repr of a data
        # structure to a customer; an empty body fails the length check and
        # falls back.
        draft = AnthropicDrafter(client=FakeClient(payload)).draft(make_brief())
        assert draft.body == ""

    def test_an_empty_response_does_not_crash(self):
        class Empty:
            content = []

        class Messages:
            def create(self, **kwargs):
                return Empty()

        class Client:
            messages = Messages()

        result = CommsDesk(drafter=AnthropicDrafter(client=Client())).compose(
            make_brief()
        )
        assert result.fell_back

    def test_the_length_budget_given_to_the_model_matches_the_check(self):
        # 134 UCS-2 units is the two-segment Hindi budget. Telling the model
        # 300 there would guarantee a failed draft on every Hindi SMS.
        from rebound.desk import _brief_prompt

        hindi = _brief_prompt(make_brief(language=Language.HI))
        english = _brief_prompt(make_brief(language=Language.EN))
        assert "134" in hindi
        assert "134" not in english
