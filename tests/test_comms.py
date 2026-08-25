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
            # Hosts, in the shapes a model actually writes them.
            ("settle at evil.example.com/pay", ["evil.example.com/pay"]),
            ("settle at EVIL.EXAMPLE.COM/pay", ["EVIL.EXAMPLE.COM/pay"]),
            ("visit vahan.in now", ["vahan.in"]),
            ("visit pay.vahan.in/r/7Kd2Qm", ["pay.vahan.in/r/7Kd2Qm"]),
            ("visit evil.example.com.", ["evil.example.com"]),
            ("Pay.Vahan.In is ours", ["Pay.Vahan.In"]),
            # The shapes an allow-list of top-level domains could never see.
            # Every one of these was returned as cleared to send.
            ("pay at vahan-secure.ru/pay", ["vahan-secure.ru/pay"]),
            ("pay at vahan-secure.top", ["vahan-secure.top"]),
            ("pay at 203.0.113.9", ["203.0.113.9"]),
            ("pay at EVIL.CO:8080/pay", ["EVIL.CO:8080/pay"]),
            # No top-level domain at all, and the worst of them in this
            # domain: a tappable payment intent straight to an attacker's VPA.
            ("tap upi://pay?pa=vahan@fraudpsp&am=1299", ["upi://pay?pa=vahan@fraudpsp&am=1299"]),
            # U+2024 ONE DOT LEADER, which resolvers and eyes read as a dot.
            ("pay at vahan-secure․com/pay", ["vahan-secure.com/pay"]),
            # An email or a VPA is a destination too.
            ("mail help@vahan.in", ["help@vahan.in"]),
        ],
    )
    def test_anything_that_could_be_tapped_is_found(self, text, expected):
        from rebound.verify import _find_urls

        assert _find_urls(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "keep sufficient balance.In case of trouble",
            "the account.In future we will retry",
            "call us.in case of trouble",
            "settle the dues.pay by Friday",
            "we could not collect the payment.online banking is unaffected",
            "aapka payment nahi hua.info ke liye call karein",
            "Rs.1,299.Please keep balance",
            "e-NACH.Please retry.",
        ],
    )
    def test_a_missing_space_is_flagged_rather_than_waved_through(self, text):
        """Fail closed. This reverses an earlier decision in this module.

        A previous version filtered these out, on the reasoning that a missing
        space is a typo rather than a link and that rejecting it forces a
        fallback over nothing. That reasoning optimised the reported fallback
        rate instead of the harm, and the two are not comparable: a false
        positive costs one repair round trip, with the offending token named in
        the feedback so the model fixes the space; a false negative costs a
        customer their money.

        It was also load-bearing in the wrong direction. The filter accepted
        any host that was not lowercase and had fewer than three labels, which
        is most of the phishing surface.
        """
        from rebound.verify import _find_urls

        assert _find_urls(text), text

    @pytest.mark.parametrize(
        "text",
        [
            "your payment of Rs.1,299 did not go through",
            "the amount is 1,299.50 including tax",
            "we will retry on 05.09.2026",
            "we will retry on 05/09/2026",
            "we will retry at 10 a.m. tomorrow",
            "reference No.5 in your statement",
            "your e-NACH mandate. Please keep balance.",
            "Vahan Technologies Pvt. Ltd.",
        ],
    )
    def test_ordinary_prose_and_money_are_not_links(self, text):
        # Fail-closed must not mean fail-always. An amount, a decimal, a
        # dotted date, a clock time and a company suffix all carry an internal
        # dot and none of them resolves.
        from rebound.verify import _find_urls

        assert _find_urls(text) == [], text

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

    def test_every_field_a_message_may_quote_is_permitted(self):
        """The two fact tables used to disagree, and one was silently thinner.

        An earlier version of this test asserted that every digit run in
        ``quotable_facts`` appears in ``permitted_numerals`` — which is how
        ``permitted_numerals`` is *defined*, so it was true for every possible
        implementation and could not fail. It is replaced by naming the fields
        outright: a field added to the brief and not to the table is a true
        fact a drafter is punished for stating.
        """
        brief = make_brief(link=LINK)
        facts = set(brief.quotable_facts)
        for field in (
            brief.mandate_reference,
            brief.merchant.support_number,
            brief.merchant.name,
            brief.bank,
            brief.episode_id,
            brief.amount_rupees,
            brief.link,
        ):
            assert field in facts, field
        for day in (brief.retry_on, brief.reference_date):
            assert day.isoformat() in facts
            assert day.strftime("%d/%m/%Y") in facts
        # And the customer id is deliberately absent: it is ours, not theirs,
        # and no message could legitimately carry it.
        assert brief.customer_id not in facts

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
    """Behaviour, not constants.

    These used to read the value out of ``CHANNEL_SPECS`` and assert it equals
    itself, which cannot fail. Each now asserts what the constant is *for*.
    """

    def test_every_channel_has_a_spec(self):
        assert set(CHANNEL_SPECS) == set(Channel)

    def test_a_voice_brief_cannot_be_given_a_link(self):
        # The bar is at construction as well as at verification, and neither
        # half was tested — the desk test that would have covered it skipped
        # the combination, citing this behaviour in a comment.
        approval = ApprovedAction(
            episode_id="EP_1",
            action=Action.VOICE_CALL,
            at=dt.datetime(2026, 9, 3, 11, 0),
            token=_APPROVAL,
        )
        view = type(
            "View",
            (),
            dict(
                episode_id="EP_1",
                customer_id="CUST_1",
                rail=Rail.UPI_AUTOPAY,
                disposition=Disposition.RETRY_TIMING,
                cycle_amount_paise=129900,
                mandate_id="MND_0000001",
                bank="HDFC Bank",
                failure_code="UPI_INSUFFICIENT_FUNDS",
            ),
        )()
        with pytest.raises(ValueError, match="cannot carry a link"):
            MessageBrief.build(
                approval,
                view,
                merchant=MERCHANT,
                language=Language.EN,
                link=LINK,
            )

    def test_a_voice_script_with_a_link_is_rejected_at_verification_too(self):
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

    def test_an_emoji_in_an_sms_is_rejected(self):
        draft = sms(GOOD_EN + " 🎉")
        findings = verify(draft, make_brief())
        assert {"renders_on_channel", "sms_stays_in_gsm7"} & {
            f.check_id for f in findings
        }

    def test_an_emoji_in_a_voice_script_is_rejected(self):
        # Voice is not covered by the GSM-7 check, so this is the only thing
        # standing between an IVR and a script with an emoji in it.
        draft = Draft(
            body=(
                "Vahan: your payment of ₹1,299 did not go through 🎉 Please "
                "keep sufficient balance in your account."
            ),
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(
            draft, make_brief(action=Action.VOICE_CALL, channel=Channel.VOICE)
        )
        assert any(f.check_id == "renders_on_channel" for f in findings)


class TestTheBranchesNothingWasExercising:
    """Guards that survived mutation because no test reached them.

    Each of these is a branch a reviewer found could be deleted outright with
    the whole suite still green.
    """

    def test_a_message_with_no_instruction_at_all_is_rejected(self):
        # The required-marker branch of AskIsHonoured. Every existing case
        # fired the *contradiction* branch instead, so "says nothing at all"
        # was never tested.
        draft = sms(
            "Vahan: your UPI Autopay payment of Rs.1,299 did not go through. "
            "We are sorry for the inconvenience caused to you today."
        )
        findings = verify(draft, make_brief(ask=Ask.KEEP_BALANCE))
        assert any(f.check_id == "ask_is_honoured" for f in findings)

    def test_a_collect_link_message_with_no_link_is_rejected(self):
        draft = Draft(
            body=(
                "Vahan: your payment of ₹1,299 is pending. Please pay now to "
                "continue your subscription."
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
        assert any(f.check_id == "links_are_ours" for f in findings)

    def test_devanagari_in_an_english_message_is_rejected(self):
        draft = Draft(
            body=GOOD_EN + " कृपया ध्यान दें।",
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(draft, make_brief(channel=Channel.WHATSAPP))
        assert any(f.check_id == "script_matches_language" for f in findings)

    def test_hindi_that_is_mostly_latin_is_rejected(self):
        # The 0.6 floor. A message with one Devanagari word in it is not Hindi.
        draft = Draft(
            body=(
                "Vahan: your UPI Autopay payment of Rs.1,299 did not go "
                "through. Please keep balance. राशि"
            ),
            language=Language.HI,
            produced_by="test",
        )
        findings = verify(draft, make_brief(language=Language.HI))
        assert any(f.check_id == "script_matches_language" for f in findings)

    def test_a_subject_on_an_sms_is_rejected(self):
        draft = Draft(
            body=GOOD_EN,
            subject="Vahan: about your payment",
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(draft, make_brief())
        assert any(f.check_id == "renders_on_channel" for f in findings)

    def test_a_message_longer_than_the_channel_allows_is_rejected(self):
        draft = Draft(
            body=(
                "Vahan: your UPI Autopay payment of ₹1,299 did not go "
                "through. Please keep sufficient balance. " + "Thank you. " * 200
            ),
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(draft, make_brief(channel=Channel.WHATSAPP))
        assert any(f.check_id == "within_channel_budget" for f in findings)

    def test_a_message_too_short_to_be_a_message_is_rejected(self):
        draft = sms("Rs.1,299")
        findings = verify(draft, make_brief())
        assert any(f.check_id == "within_channel_budget" for f in findings)

    @pytest.mark.parametrize(
        "phrase",
        [
            "reply with the verification code",
            "share the code we sent you",
            "quote the 6-digit code",
            "tell us your passwords",
        ],
    )
    def test_a_paraphrased_credential_request_is_rejected(self, phrase):
        # Exercised only by the red-team script until now, so deleting the
        # paraphrase markers left the suite green.
        draft = sms(
            f"Vahan: Rs.1,299 failed. Please keep sufficient balance and {phrase}."
        )
        findings = verify(draft, make_brief(channel=Channel.WHATSAPP))
        assert any(
            f.check_id == "no_credential_solicitation" for f in findings
        ), phrase

    def test_an_amount_written_with_the_unit_after_it_is_checked(self):
        # _AMOUNT_SUFFIXED was never consulted by any fixture.
        draft = Draft(
            body=(
                "Vahan: your UPI Autopay payment of 9,999 rupees did not go "
                "through. Please keep sufficient balance."
            ),
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(draft, make_brief(channel=Channel.WHATSAPP))
        assert any(f.check_id == "amount_is_exact" for f in findings)

    def test_the_right_amount_written_with_the_unit_after_it_clears(self):
        draft = Draft(
            body=(
                "Vahan: your UPI Autopay payment of 1,299 rupees did not go "
                "through. Please keep sufficient balance."
            ),
            language=Language.EN,
            produced_by="test",
        )
        findings = verify(draft, make_brief(channel=Channel.WHATSAPP))
        assert findings == (), [f.detail for f in findings]

    def test_a_partial_mandate_reference_is_a_fabrication(self):
        # The token branch accepts any fragment of a true fact, so a fragment
        # of the mandate reference was only caught by the digit branch — and
        # only when it was long enough.
        draft = sms(GOOD_EN + " Ref HDFC0009911.")
        findings = verify(draft, make_brief(channel=Channel.WHATSAPP))
        assert any(f.check_id == "no_fabricated_identifiers" for f in findings)
