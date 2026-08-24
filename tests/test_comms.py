"""Tests for the brief and for the checks that stand between a model and a customer.

The layer's whole safety argument is that :mod:`rebound.verify` can reject a
draft. A check with a bug that makes it always return ``None`` still passes
every test that only feeds it good messages, still reports a clean run, and
still lets a fabricated reference number reach a customer. So the central test
here is :class:`TestEveryCheckCanFail`, which constructs a draft that each
check must reject and fails if it does not.

The second concern is the opposite error. A check that rejects correct
messages drives the fallback rate up, and the fallback rate is a headline
number — an inflated one is a false claim about how much the model can be
trusted. :class:`TestGoodMessagesSurvive` hand-writes messages in all three
languages and asserts nothing objects.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rebound.comms import (
    CHANNEL_FOR_ACTION,
    CHANNEL_SPECS,
    REVIEWED_LANGUAGES,
    Ask,
    Channel,
    Draft,
    Language,
    MerchantProfile,
    MessageBrief,
    ask_for,
    devanagari_ratio,
    format_rupees,
    gsm7_encodable,
    segments_for,
)
from rebound.compliance import _APPROVAL, ApprovedAction
from rebound.taxonomy import (
    CUSTOMER_FACING_ACTIONS,
    Action,
    Disposition,
    Rail,
)
from rebound.verify import CHECKS, Category, verify

MERCHANT = MerchantProfile(
    name="Vahan",
    support_number="18002670001",
    link_host="pay.vahan.in",
    sender_id="VAHANX",
)
LINK = "https://pay.vahan.in/r/7Kd2Qm"
TODAY = dt.date(2026, 9, 3)
DEBIT_DAY = dt.date(2026, 9, 5)


def make_brief(**overrides) -> MessageBrief:
    """A brief a correct message can be written for, unless overridden."""
    base = dict(
        episode_id="EP_1",
        customer_id="CUST_1",
        action=Action.NUDGE_SMS,
        channel=Channel.SMS,
        language=Language.EN,
        merchant=MERCHANT,
        amount_paise=129900,
        mandate_reference="UMRN2024HDFC0009911",
        bank="HDFC Bank",
        rail=Rail.UPI_AUTOPAY,
        disposition=Disposition.RETRY_TIMING,
        ask=Ask.KEEP_BALANCE,
        reference_date=TODAY,
        retry_on=DEBIT_DAY,
        link=None,
    )
    base.update(overrides)
    return MessageBrief(**base)


def sms(body: str) -> Draft:
    return Draft(body=body, language=Language.EN, produced_by="test")


#: A message that every check passes. Every failure case below is this text
#: with one thing changed, so a test that fails is pointing at the thing that
#: changed rather than at the fixture.
GOOD_EN = (
    "Vahan: your UPI Autopay payment of Rs.1,299 did not go through. "
    "Please keep sufficient balance in your account and we will try again."
)


class TestSegmentArithmetic:
    """The SMS budget is money, not style, so the arithmetic gets checked."""

    def test_the_rupee_sign_is_not_gsm7(self):
        # This single fact is why every SMS template writes "Rs.". If it were
        # wrong the templates would be paying for UCS-2 and losing more than
        # half the message for nothing.
        assert not gsm7_encodable("₹")
        assert gsm7_encodable("Rs.1,299")

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("a" * 160, (1, 160, "GSM-7")),
            ("a" * 161, (2, 161, "GSM-7")),
            ("a" * 306, (2, 306, "GSM-7")),
            ("a" * 307, (3, 307, "GSM-7")),
            ("क" * 70, (1, 70, "UCS-2")),
            ("क" * 71, (2, 71, "UCS-2")),
            ("क" * 134, (2, 134, "UCS-2")),
            ("क" * 135, (3, 135, "UCS-2")),
        ],
    )
    def test_segment_boundaries(self, text, expected):
        assert segments_for(text) == expected

    def test_extension_characters_cost_two_septets(self):
        # GSM 03.38 sends ~ { } [ ] | ^ \ and € as an escape plus a character.
        # Counting them as one lets a message that bills as two segments look
        # like one.
        assert segments_for("~" * 80)[1] == 160
        assert segments_for("~" * 81)[0] == 2

    def test_one_devanagari_character_re_encodes_the_whole_message(self):
        latin = "a" * 100
        assert segments_for(latin) == (1, 100, "GSM-7")
        assert segments_for(latin + "क")[2] == "UCS-2"
        assert segments_for(latin + "क")[0] == 2


class TestRupeeFormatting:
    @pytest.mark.parametrize(
        "paise, expected",
        [
            (0, "0"),
            (9900, "99"),
            (129900, "1,299"),
            (100000, "1,000"),
            (10000000, "1,00,000"),
            (150000000, "15,00,000"),
            (1000000000, "1,00,00,000"),
            (129950, "1,299.50"),
            (105, "1.05"),
        ],
    )
    def test_indian_digit_grouping(self, paise, expected):
        # Lakhs and crores group in twos after the first three digits.
        # Western grouping would render 1,00,000 as 100,000, which an Indian
        # recipient reads correctly but which is not how any bank writes it.
        assert format_rupees(paise) == expected

    def test_negative_amounts_are_refused(self):
        with pytest.raises(ValueError):
            format_rupees(-1)


class TestTheAskIsDerivedNotChosen:
    def test_every_customer_facing_action_has_a_channel(self):
        # A customer-facing action with no channel entry raises inside
        # MessageBrief.build, so an action added to the taxonomy without a
        # channel would fail at send time rather than here.
        assert CUSTOMER_FACING_ACTIONS <= set(CHANNEL_FOR_ACTION)

    def test_card_and_upi_customer_action_ask_for_different_things(self):
        # The whole reason the ask is derived rather than generated. A card
        # blocks on the card; UPI blocks on an approval the customer holds.
        # Telling a UPI customer to replace a working card wastes a contact
        # and the debit fails again identically.
        card = ask_for(
            Action.NUDGE_SMS, Disposition.CUSTOMER_ACTION, Rail.CARD_ON_FILE
        )
        upi = ask_for(
            Action.NUDGE_SMS, Disposition.CUSTOMER_ACTION, Rail.UPI_AUTOPAY
        )
        assert card is Ask.UPDATE_CARD
        assert upi is Ask.APPROVE_IN_APP

    def test_a_pre_debit_notice_asks_for_nothing(self):
        # A notice announces a debit. Attaching an instruction turns a
        # regulatory disclosure into dunning.
        for disposition in Disposition:
            assert (
                ask_for(
                    Action.SEND_PRE_DEBIT_NOTIFICATION, disposition, Rail.ENACH
                )
                is Ask.NOTHING
            )

    def test_a_timing_failure_never_asks_for_a_card(self):
        assert (
            ask_for(Action.NUDGE_SMS, Disposition.RETRY_TIMING, Rail.CARD_ON_FILE)
            is Ask.KEEP_BALANCE
        )


class TestBriefConstruction:
    @staticmethod
    def _view(**overrides):
        base = dict(
            episode_id="EP_1",
            customer_id="CUST_1",
            rail=Rail.UPI_AUTOPAY,
            disposition=Disposition.RETRY_TIMING,
            cycle_amount_paise=129900,
            mandate_id="UMRN2024HDFC0009911",
            bank="HDFC Bank",
        )
        base.update(overrides)
        return type("View", (), base)()

    def _approval(self, action=Action.NUDGE_SMS, episode="EP_1"):
        return ApprovedAction(
            episode_id=episode,
            action=action,
            at=dt.datetime(2026, 9, 3, 11, 0),
            token=_APPROVAL,
        )

    def test_a_brief_needs_an_approval_that_cannot_be_forged(self):
        # The same discipline the executor uses. There is no path to a message
        # for an action the gate did not permit, because there is no way to
        # mint the approval a brief requires.
        with pytest.raises(PermissionError):
            ApprovedAction(
                episode_id="EP_1",
                action=Action.NUDGE_SMS,
                at=dt.datetime(2026, 9, 3, 11, 0),
            )

    def test_an_approval_for_another_episode_is_refused(self):
        with pytest.raises(ValueError, match="different episode"):
            MessageBrief.build(
                self._approval(episode="EP_OTHER"),
                self._view(),
                merchant=MERCHANT,
                language=Language.EN,
            )

    def test_a_debit_has_no_message(self):
        with pytest.raises(ValueError, match="in front of a customer"):
            MessageBrief.build(
                self._approval(action=Action.RETRY_SAME_RAIL),
                self._view(),
                merchant=MERCHANT,
                language=Language.EN,
            )

    def test_a_collect_link_without_a_link_is_refused(self):
        # Asking someone to pay and giving them nowhere to do it is the one
        # failure the verifier would catch too late — it would fall back to a
        # template that has the same hole.
        with pytest.raises(ValueError, match="nowhere to do it"):
            MessageBrief.build(
                self._approval(action=Action.SEND_COLLECT_LINK),
                self._view(),
                merchant=MERCHANT,
                language=Language.EN,
            )

    def test_an_unreviewed_language_is_refused(self):
        assert Language.EN in REVIEWED_LANGUAGES
        with pytest.raises(ValueError, match="reviewed fallback"):
            MessageBrief.build(
                self._approval(),
                self._view(),
                merchant=MERCHANT,
                language="ta",  # type: ignore[arg-type]
            )

    def test_the_brief_carries_the_derived_ask_not_a_supplied_one(self):
        brief = MessageBrief.build(
            self._approval(),
            self._view(
                disposition=Disposition.CUSTOMER_ACTION, rail=Rail.CARD_ON_FILE
            ),
            merchant=MERCHANT,
            language=Language.EN,
        )
        assert brief.ask is Ask.UPDATE_CARD

    def test_the_reference_date_comes_from_the_approval(self):
        # Not date.today(). A brief that cannot be rebuilt cannot be audited.
        brief = MessageBrief.build(
            self._approval(), self._view(), merchant=MERCHANT, language=Language.EN
        )
        assert brief.reference_date == dt.date(2026, 9, 3)


class TestEveryCheckCanFail:
    """Each check gets a draft it must reject.

    Parametrised over ``CHECKS`` by id so that adding a check without adding a
    case here fails the suite rather than silently shipping an unexercised
    check.
    """

    #: check_id -> (draft, brief) that the check must reject.
    CASES: dict[str, tuple[Draft, MessageBrief]] = {
        "amount_is_exact": (
            sms(GOOD_EN.replace("Rs.1,299", "Rs.12,990")),
            make_brief(),
        ),
        "no_fabricated_identifiers": (
            sms(GOOD_EN + " Ref 884512."),
            make_brief(),
        ),
        "links_are_ours": (
            sms(GOOD_EN + " Pay at https://vahan-secure-pay.example.com/x"),
            make_brief(link=LINK),
        ),
        "pre_debit_disclosure": (
            sms(
                "Vahan: Rs.1,299 will be debited soon. "
                "No action is needed from you."
            ),
            make_brief(
                action=Action.SEND_PRE_DEBIT_NOTIFICATION, ask=Ask.NOTHING
            ),
        ),
        "no_internal_codes": (
            sms(GOOD_EN + " Reason: UPI_INSUFFICIENT_FUNDS."),
            make_brief(),
        ),
        "no_credential_solicitation": (
            sms(
                "Vahan: your payment of Rs.1,299 failed. Please keep "
                "sufficient balance and share the OTP to confirm."
            ),
            make_brief(),
        ),
        "no_coercion": (
            sms(
                "Vahan: your payment of Rs.1,299 failed. Keep sufficient "
                "balance or we will begin legal action against you."
            ),
            make_brief(),
        ),
        "sender_is_identified": (
            sms(GOOD_EN.replace("Vahan: ", "")),
            make_brief(),
        ),
        "ask_is_honoured": (
            sms(
                "Vahan: your payment of Rs.1,299 did not go through. "
                "Please update your card details to continue."
            ),
            make_brief(ask=Ask.KEEP_BALANCE),
        ),
        "script_matches_language": (
            Draft(
                body=(
                    "Vahan: aapka Rs.1,299 ka payment nahi hua. कृपया खाते "
                    "में राशि रखें।"
                ),
                language=Language.HINGLISH,
                produced_by="test",
            ),
            make_brief(language=Language.HINGLISH),
        ),
        "renders_on_channel": (
            Draft(
                body=GOOD_EN,
                subject=None,
                language=Language.EN,
                produced_by="test",
            ),
            make_brief(action=Action.NUDGE_EMAIL, channel=Channel.EMAIL),
        ),
        "sms_stays_in_gsm7": (
            sms(GOOD_EN.replace("go through", "go through‼")),
            make_brief(),
        ),
        "within_channel_budget": (
            Draft(
                body=(
                    "Vahan: आपका Rs.1,299 का भुगतान नहीं हो सका। कृपया खाते "
                    "में पर्याप्त राशि रखें और हम दोबारा प्रयास करेंगे। "
                    "किसी भी प्रश्न के लिए हमें कॉल करें, हम आपकी सहायता "
                    "करने के लिए हमेशा उपलब्ध हैं।"
                ),
                language=Language.HI,
                produced_by="test",
            ),
            make_brief(language=Language.HI),
        ),
    }

    @pytest.mark.parametrize("check", CHECKS, ids=lambda c: c.check_id)
    def test_the_check_rejects_the_draft_it_exists_for(self, check):
        draft, brief = self.CASES[check.check_id]
        finding = check.run(draft, brief)
        assert finding is not None, (
            f"{check.check_id} accepted a draft it exists to reject; the "
            "guarantee it represents does not exist"
        )
        assert finding.check_id == check.check_id

    def test_every_check_has_a_case(self):
        assert {check.check_id for check in CHECKS} == set(self.CASES)

    def test_verify_reports_every_fault_not_just_the_first(self):
        # A repair pass told about one fault at a time costs a round trip per
        # fault, and the measured breakdown of what models get wrong is only
        # meaningful if every draft is scored against every check.
        draft = sms(
            "Your payment of Rs.9,999 failed. Update your card or we will "
            "take legal action. Ref 774411."
        )
        findings = verify(draft, make_brief())
        assert {f.check_id for f in findings} >= {
            "amount_is_exact",
            "no_fabricated_identifiers",
            "no_coercion",
            "sender_is_identified",
            "ask_is_honoured",
        }

    def test_a_wrong_amount_in_an_email_subject_is_caught(self):
        # Checking the body alone passes an email whose subject says one
        # figure and whose body says another.
        draft = Draft(
            body=(
                "Vahan: your UPI Autopay payment of ₹1,299 did not go "
                "through. Please keep sufficient balance in your account."
            ),
            subject="Vahan: your payment of ₹12,990 needs attention",
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(
            draft, make_brief(action=Action.NUDGE_EMAIL, channel=Channel.EMAIL)
        )
        assert any(f.check_id == "amount_is_exact" for f in findings)

    def test_a_message_naming_no_amount_at_all_is_rejected(self):
        draft = sms(
            "Vahan: your UPI Autopay payment did not go through. Please "
            "keep sufficient balance in your account and we will try again."
        )
        findings = verify(draft, make_brief())
        assert any(f.check_id == "amount_is_exact" for f in findings)

    def test_a_voice_script_may_not_carry_a_link(self):
        draft = Draft(
            body=(
                "Vahan: your payment of ₹1,299 did not go through. Please "
                f"keep sufficient balance. Visit {LINK}"
            ),
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(
            draft, make_brief(action=Action.VOICE_CALL, channel=Channel.VOICE)
        )
        assert any(f.check_id == "links_are_ours" for f in findings)

    def test_hinglish_written_as_english_is_rejected(self):
        # Devanagari is not the only way to get Hinglish wrong. A model that
        # ignores the register and answers in English produces a message that
        # passes every other check and is not what was asked for.
        draft = Draft(
            body=GOOD_EN, language=Language.HINGLISH, produced_by="test"
        )
        findings = verify(draft, make_brief(language=Language.HINGLISH))
        assert any(f.check_id == "script_matches_language" for f in findings)


class TestTheReviewFindings:
    """One test per defect an independent review found in the first cut.

    Kept as a named class rather than folded into the cases above because each
    of these passed the original suite. They are the evidence that the suite
    was not, at that point, checking what it claimed to check.
    """

    def test_a_scheme_less_link_is_still_a_link(self):
        # The worst class of defect here: a guarantee that does not exist.
        # Against a scheme-only pattern this matched nothing, passed every
        # check, and would have been read aloud down a phone line.
        draft = Draft(
            body=(
                "Vahan: your payment of ₹1,299 failed. Please keep sufficient "
                "balance, or visit evil.example.com/pay to settle."
            ),
            language=Language.EN,
            produced_by="test",
        )
        for channel, action in (
            (Channel.VOICE, Action.VOICE_CALL),
            (Channel.WHATSAPP, Action.NUDGE_WHATSAPP),
        ):
            findings = verify(
                draft, make_brief(action=action, channel=channel)
            )
            assert any(f.check_id == "links_are_ours" for f in findings), channel

    @pytest.mark.parametrize(
        "text, expected",
        [
            # Real hosts, in the shapes a model actually writes them.
            ("settle at evil.example.com/pay", ["evil.example.com/pay"]),
            ("settle at EVIL.EXAMPLE.COM/pay", ["EVIL.EXAMPLE.COM/pay"]),
            ("visit vahan.in now", ["vahan.in"]),
            ("visit pay.vahan.in/r/7Kd2Qm", ["pay.vahan.in/r/7Kd2Qm"]),
            ("visit evil.example.com.", ["evil.example.com"]),
            ("Pay.Vahan.In is ours", ["Pay.Vahan.In"]),
            # Prose. Making the bare-host branch case-insensitive — needed to
            # catch the shouted form above — turned an English full stop
            # before a capitalised word into a host. A missing space is a typo
            # a model produces, and rejecting the message for it forces a
            # fallback over nothing.
            ("keep sufficient balance.In case of trouble", []),
            ("the account.In future we will retry", []),
            ("contact support.Online help is available", []),
            ("Rs.1,299.Please keep balance", []),
            ("e-NACH.Please retry.", []),
            # An email address is not a link to follow.
            ("mail help@vahan.in", []),
        ],
    )
    def test_a_host_is_told_apart_from_a_sentence_break(self, text, expected):
        from rebound.verify import _find_urls

        assert _find_urls(text) == expected

    def test_a_mandate_reference_beside_the_amount_is_not_a_second_amount(self):
        # A pre-debit notice is legally required to carry the reference, so
        # this false positive fired on the message that most needs to be sent.
        draft = Draft(
            body=(
                "Vahan: mandate UMRN2024HDFC0009911 Rs.1,299 will be debited "
                "on 05/09/2026. No action is needed from you."
            ),
            language=Language.EN,
            produced_by="test",
        )
        brief = make_brief(
            action=Action.SEND_PRE_DEBIT_NOTIFICATION, ask=Ask.NOTHING
        )
        findings = verify(draft, brief)
        assert not any(f.check_id == "amount_is_exact" for f in findings), [
            f.detail for f in findings
        ]

    def test_a_word_ending_in_rs_is_not_an_amount(self):
        from rebound.verify import _AMOUNT_PREFIXED

        assert _AMOUNT_PREFIXED.findall("your hours.500 plan") == []
        assert _AMOUNT_PREFIXED.findall("Rs.1,299") == ["1,299"]

    def test_the_true_failure_date_is_not_a_fabrication(self):
        # reference_date was on the brief and permitted by neither table, so
        # stating the day the debit actually failed was reported as invented.
        draft = Draft(
            body=(
                "Vahan: your autopay of ₹1,299 failed on 03/09/2026. Please "
                "keep sufficient balance in your account."
            ),
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(
            draft, make_brief(channel=Channel.WHATSAPP, retry_on=None)
        )
        assert findings == (), [f.detail for f in findings]

    def test_the_two_fact_tables_cannot_disagree(self):
        # They used to be maintained separately and did disagree. Deriving one
        # from the other is what makes this test hold by construction.
        brief = make_brief()
        for fact in brief.quotable_facts:
            for run in __import__("re").findall(r"\d+", fact):
                assert run in brief.permitted_numerals

    @pytest.mark.parametrize("mangled", ["180026", "2670001", "80026700"])
    def test_a_fragment_of_a_real_identifier_is_a_fabrication(self, mangled):
        # Found by mutation: raising _IDENTIFIER_DIGITS to 50 left the whole
        # suite green, because every case only exercised the token branch.
        #
        # The two branches are not redundant. The token branch tests substring
        # containment, so it accepts any fragment of a true fact; the digit
        # branch tests exact membership, so it rejects one. These are pieces
        # of the real support number 18002670001, and a number that is nearly
        # right is worse than one that is invented — it looks right enough to
        # dial.
        draft = sms(GOOD_EN + f" Call {mangled}.")
        findings = verify(draft, make_brief(channel=Channel.WHATSAPP))
        assert any(
            f.check_id == "no_fabricated_identifiers" for f in findings
        ), mangled

    def test_the_whole_support_number_is_still_fine(self):
        draft = sms(GOOD_EN + " Call 18002670001.")
        findings = verify(draft, make_brief(channel=Channel.WHATSAPP))
        assert findings == (), [f.detail for f in findings]

    def test_a_readable_support_number_is_not_a_fabrication(self):
        # The check must test truth, not formatting. Storing the brief value
        # hyphenated used to make this pass, which is the tell.
        draft = Draft(
            body=GOOD_EN + " Questions? Call 1800-267-0001.",
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(draft, make_brief(channel=Channel.WHATSAPP))
        assert findings == (), [f.detail for f in findings]

    @pytest.mark.parametrize(
        "body",
        [
            "Vahan: Rs.1,299 ka payment fail ho gaya. Account balance check karein.",
            "Vahan: Rs.1,299 ka autopay decline ho gaya, balance daal dijiye.",
            "Vahan: Rs.1,299 nahi kata. Balance rakhiye, hum dobara try karenge.",
        ],
    )
    def test_natural_hinglish_is_not_rejected_as_english(self, body):
        # Half of these were rejected by the first marker list. Hinglish is
        # the register that justifies the model, so false rejects here inflate
        # the fallback rate in exactly the flattering direction.
        findings = verify(
            sms(body), make_brief(language=Language.HINGLISH)
        )
        assert not any(
            f.check_id == "script_matches_language" for f in findings
        ), [f.detail for f in findings]

    def test_a_second_wrong_instruction_is_caught(self):
        # The inverse of the case the check was written for, and just as much
        # a wasted contact. Three of seven asks had no contradiction entry.
        draft = sms(
            "Vahan: Rs.1,299 failed. Please update your card, and keep "
            "sufficient balance in the account."
        )
        brief = make_brief(
            rail=Rail.CARD_ON_FILE,
            disposition=Disposition.CUSTOMER_ACTION,
            ask=Ask.UPDATE_CARD,
        )
        findings = verify(draft, brief)
        assert any(f.check_id == "ask_is_honoured" for f in findings)

    def test_every_ask_has_a_contradiction_entry_or_a_stated_reason(self):
        from rebound.verify import _CONTRADICTIONS

        # PAY_NOW_VIA_LINK is the documented exception: every phrasing is
        # built from "pay", which is too generic to forbid.
        assert set(_CONTRADICTIONS) == set(Ask) - {
            Ask.PAY_NOW_VIA_LINK,
            Ask.NOTHING,
        }

    @pytest.mark.parametrize("char", ["‼", "™", "©", "〰", "’", "—"])
    def test_non_gsm7_characters_are_caught_even_when_not_emoji(self, char):
        # An emoji range list missed all six. A curly apostrophe is not an
        # emoji and costs exactly as much.
        draft = sms(GOOD_EN + f" Thanks{char}")
        findings = verify(draft, make_brief())
        assert any(f.check_id == "sms_stays_in_gsm7" for f in findings), char

    def test_hindi_sms_is_exempt_from_the_gsm7_rule(self):
        # Devanagari is UCS-2 by definition; that cost was budgeted when the
        # language was chosen, and flagging it would reject every Hindi SMS.
        draft = Draft(
            body=(
                "Vahan: आपका Rs.1,299 का UPI ऑटोपे भुगतान नहीं हो सका। "
                "कृपया खाते में राशि रखें।"
            ),
            language=Language.HI,
            produced_by="test",
        )
        findings = verify(draft, make_brief(language=Language.HI))
        assert findings == (), [f.detail for f in findings]


class TestGoodMessagesSurvive:
    """Correct messages must not be rejected.

    A false positive here is not a harmless extra safety margin: it forces a
    fallback, and the fallback rate is reported as evidence of how much the
    model can be trusted. Inflating it with checks that fire on good text
    makes that evidence a lie in the flattering direction.
    """

    @pytest.mark.parametrize(
        "body, brief",
        [
            (GOOD_EN, make_brief()),
            (
                "Vahan: aapka Rs.1,299 ka UPI Autopay payment nahi hua. "
                "Kripya khate mein raashi rakhein, hum dobara try karenge.",
                make_brief(language=Language.HINGLISH),
            ),
            (
                "Vahan: आपका Rs.1,299 का UPI ऑटोपे भुगतान नहीं हो सका। "
                "कृपया खाते में राशि रखें।",
                make_brief(language=Language.HI),
            ),
            (
                "Vahan: your card payment of Rs.1,299 did not go through. "
                "Please update your card details to continue.",
                make_brief(
                    rail=Rail.CARD_ON_FILE,
                    disposition=Disposition.CUSTOMER_ACTION,
                    ask=Ask.UPDATE_CARD,
                ),
            ),
        ],
    )
    def test_nothing_objects(self, body, brief):
        findings = verify(sms(body), brief)
        assert findings == (), [f.detail for f in findings]

    def test_the_support_number_is_not_a_fabricated_identifier(self):
        draft = Draft(
            body=GOOD_EN + " Questions? Call 18002670001.",
            language=Language.EN,
            produced_by="test",
        )
        assert verify(draft, make_brief(channel=Channel.WHATSAPP)) == ()

    def test_a_date_is_not_mistaken_for_money_or_a_reference(self):
        draft = Draft(
            body=(
                "Vahan: your UPI Autopay payment of ₹1,299 did not go "
                "through. Please keep sufficient balance; we will try again "
                "on 05/09/2026."
            ),
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(draft, make_brief(channel=Channel.WHATSAPP))
        assert findings == (), [f.detail for f in findings]

    def test_the_permitted_link_passes(self):
        draft = Draft(
            body=(
                f"Vahan: your payment of ₹1,299 is pending. Pay now: {LINK}"
            ),
            language=Language.EN,
            produced_by="test",
        )
        brief = make_brief(
            action=Action.SEND_COLLECT_LINK,
            channel=Channel.WHATSAPP,
            ask=Ask.PAY_NOW_VIA_LINK,
            link=LINK,
        )
        findings = verify(draft, brief)
        assert findings == (), [f.detail for f in findings]

    def test_a_mandate_reference_we_hold_is_not_a_fabrication(self):
        draft = Draft(
            body=(
                "Vahan: ₹1,299 will be debited on 05/09/2026 under mandate "
                "UMRN2024HDFC0009911. No action is needed from you."
            ),
            language=Language.EN,
            produced_by="test",
        )
        brief = make_brief(
            action=Action.SEND_PRE_DEBIT_NOTIFICATION,
            channel=Channel.WHATSAPP,
            ask=Ask.NOTHING,
        )
        findings = verify(draft, brief)
        assert findings == (), [f.detail for f in findings]


class TestCategories:
    def test_credential_and_coercion_faults_are_labelled_safety(self):
        # The report groups by category, and these two are the lexical checks.
        # Labelling them SAFETY rather than FIDELITY is what stops a reader
        # taking a clean run as proof the messages were never coercive; they
        # were only never coercive in the words on the list.
        by_id = {check.check_id: check for check in CHECKS}
        assert by_id["no_coercion"].category is Category.SAFETY
        assert by_id["no_credential_solicitation"].category is Category.SAFETY
        assert by_id["amount_is_exact"].category is Category.FIDELITY

    def test_check_ids_are_unique(self):
        ids = [check.check_id for check in CHECKS]
        assert len(ids) == len(set(ids))


class TestScriptRatio:
    def test_the_ratio_ignores_digits(self):
        # Counting over all characters makes the same sentence score
        # differently for ₹99 and ₹14,999.
        assert devanagari_ratio("राशि 99") == devanagari_ratio("राशि 14999")

    def test_a_url_does_not_make_a_hindi_message_english(self):
        # The bug this exists for: a payment link's Latin characters pulled a
        # fully Devanagari message to 44% and the fallback failed its own
        # check on the one message that most needs sending.
        body = f"Vahan: आपका ₹1,299 का भुगतान बाकी है। अभी भुगतान करें: {LINK}"
        brief = make_brief(
            action=Action.SEND_COLLECT_LINK,
            channel=Channel.WHATSAPP,
            language=Language.HI,
            ask=Ask.PAY_NOW_VIA_LINK,
            link=LINK,
        )
        findings = verify(
            Draft(body=body, language=Language.HI, produced_by="test"), brief
        )
        assert findings == (), [f.detail for f in findings]


class TestChannelSpecs:
    def test_every_channel_has_a_spec(self):
        assert set(CHANNEL_SPECS) == set(Channel)

    def test_voice_carries_no_links(self):
        assert not CHANNEL_SPECS[Channel.VOICE].links_allowed

    def test_sms_carries_no_emoji(self):
        # One emoji forces UCS-2 and cuts the segment from 160 to 70.
        assert not CHANNEL_SPECS[Channel.SMS].emoji_allowed
