"""The comms layer against real episodes, a real gate and real identifiers.

Everything in ``test_comms`` and ``test_desk`` builds a
:class:`~rebound.comms.MessageBrief` by hand. That is the right way to test a
check, and it is exactly how a whole class of defect survives: a fixture chosen
to read nicely is a fixture chosen to avoid the collisions real data has. The
mandate reference used throughout those files is ``UMRN2024HDFC0009911``,
invented for legibility, and the simulator actually issues ``MND_0000001`` —
which matched the internal-code check while the pre-debit disclosure check
required it to be quoted. Every notice was blocked, fallback included, and no
hand-built fixture could have shown it.

So this file builds nothing by hand. It samples a population, collects the
debits that actually failed, adjudicates each candidate action through a real
:class:`~rebound.compliance.ComplianceGate`, builds a brief from the resulting
approval and the episode's own view, and composes a message. If any pairing of
real identifiers, real amounts and real dispositions cannot produce a sendable
message, it fails here rather than in front of a customer.
"""

from __future__ import annotations

import collections
import datetime as dt

import pytest

from rebound.comms import (
    CHANNEL_FOR_ACTION,
    Ask,
    Language,
    MerchantProfile,
    MessageBrief,
)
from rebound.compliance import ComplianceGate, Request, Verdict
from rebound.desk import CommsDesk, TemplateDrafter
from rebound.eval.harness import build_eval_batch
from rebound.sim.world import World
from rebound.taxonomy import (
    CUSTOMER_FACING_ACTIONS,
    Action,
    Disposition,
    legal_actions,
)
from rebound.verify import verify

MERCHANT = MerchantProfile(
    name="Vahan",
    support_number="18002670001",
    link_host="pay.vahan.in",
    sender_id="VAHANX",
)
LINK = "https://pay.vahan.in/r/7Kd2Qm"


@pytest.fixture(scope="module")
def episodes():
    """Real failed debits, with the identifiers the simulator actually issues."""
    world = World(seed=4242)
    customers = world.sample_customers(400)
    mandates = world.sample_mandates(
        customers, dt.date(2024, 1, 1), dt.date(2025, 6, 30)
    )
    # Without calibration the world refuses to sample, so an uncalibrated
    # fixture would not silently produce off-anchor failures — it raises.
    world.calibrate(
        customers,
        mandates,
        [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(14)],
    )
    batch = build_eval_batch(
        world,
        customers,
        mandates,
        start=dt.date(2026, 1, 1),
        end=dt.date(2026, 3, 31),
        max_episodes=250,
        seed=909,
    )
    if not batch:
        pytest.skip("the simulator produced no failures for this seed")
    return [
        world.open_episode(
            episode_id=f"EP_{i:08d}",
            mandate=spec.mandate,
            customer=spec.customer,
            failure_code=spec.failure_code,
            failed_at=spec.failed_at,
            cycles_elapsed=spec.cycles_elapsed,
        )
        for i, spec in enumerate(batch)
    ]


def _composable(episodes):
    """Every (view, approved action) pair the gate actually permits.

    Drawn from the gate rather than assumed, so the set under test is the set
    that can really occur. An action the gate always denies is one no customer
    ever sees, and asserting things about it would be measuring a fiction.
    """
    gate = ComplianceGate()
    pairs = []
    for episode in episodes:
        view = episode.view()
        at = view.failed_at + dt.timedelta(hours=18)
        for action in sorted(CUSTOMER_FACING_ACTIONS, key=str):
            # Structural legality first, because the gate does not check it and
            # the sequencer would never propose an action the taxonomy calls
            # meaningless for this failure. Drawing from the gate alone put
            # REQUEST_REMANDATE on insufficient-funds episodes into this set.
            if action not in legal_actions(view.failure_code):
                continue
            decision = gate.adjudicate(
                Request.from_view(view, action, at), record=False
            )
            if decision.verdict is Verdict.ALLOW and decision.approval:
                pairs.append((view, decision.approval))
    return pairs


class TestRealEpisodesProduceSendableMessages:
    def test_the_gate_permits_something_worth_testing(self, episodes):
        # If this collapses to nothing, every test below passes vacuously.
        pairs = _composable(episodes)
        assert len(pairs) > 100, f"only {len(pairs)} permitted actions"
        actions = {approval.action for _, approval in pairs}
        assert len(actions) >= 3, f"only {actions} ever permitted"

    @pytest.mark.parametrize("language", list(Language))
    def test_every_permitted_action_yields_a_message_that_clears(
        self, episodes, language
    ):
        """The end-to-end assertion: real episode in, sendable message out.

        Runs on the template drafter, which is the fallback. If the fallback
        cannot produce a clearing message for some real combination of
        identifier, amount and disposition, then on that combination the system
        sends nothing at all — the model failing would have nowhere to hand off
        to.
        """
        desk = CommsDesk(drafter=TemplateDrafter())
        failures = []
        composed = 0

        for view, approval in _composable(episodes):
            channel = CHANNEL_FOR_ACTION[approval.action]
            link = (
                LINK
                if approval.action is Action.SEND_COLLECT_LINK
                else None
            )
            brief = MessageBrief.build(
                approval,
                view,
                merchant=MERCHANT,
                language=language,
                retry_on=(approval.at + dt.timedelta(days=2)).date(),
                link=link,
            )
            result = desk.compose(brief)
            composed += 1
            if not result.cleared:
                failures.append(
                    (
                        view.episode_id,
                        view.mandate_id,
                        str(approval.action),
                        str(channel),
                        [f.detail for f in result.findings],
                    )
                )

        assert composed > 100
        assert not failures, failures[:5]

    def test_no_real_message_ever_quotes_the_failure_code(self, episodes):
        # The customer is never told "UPI_INSUFFICIENT_FUNDS". This is checked
        # against real codes drawn from real episodes rather than the one code
        # a unit fixture happens to use.
        desk = CommsDesk(drafter=TemplateDrafter())
        codes = set()
        for view, approval in _composable(episodes):
            brief = MessageBrief.build(
                approval, view, merchant=MERCHANT, language=Language.EN,
                retry_on=(approval.at + dt.timedelta(days=2)).date(),
                link=LINK if approval.action is Action.SEND_COLLECT_LINK else None,
            )
            result = desk.compose(brief)
            codes.add(view.failure_code)
            assert result.sent is not None
            assert view.failure_code not in result.sent.rendered()
        assert len(codes) > 3, f"only saw codes {codes}"

    def test_the_instruction_matches_the_real_disposition(self, episodes):
        """A card failure asks for a card; a UPI one asks for an approval.

        The single most consequential thing the brief fixes, checked against
        the dispositions the taxonomy actually assigns rather than the two a
        unit test picks.
        """
        seen: dict[Disposition, set[Ask]] = collections.defaultdict(set)
        for view, approval in _composable(episodes):
            brief = MessageBrief.build(
                approval, view, merchant=MERCHANT, language=Language.EN,
                retry_on=(approval.at + dt.timedelta(days=2)).date(),
                link=LINK if approval.action is Action.SEND_COLLECT_LINK else None,
            )
            seen[view.disposition].add(brief.ask)

        # A timing failure is never told to touch a card or a mandate.
        for ask in seen.get(Disposition.RETRY_TIMING, ()):
            assert ask in {Ask.KEEP_BALANCE, Ask.PAY_NOW_VIA_LINK}, ask
        # A merchant-side defect never asks the customer for anything.
        assert seen.get(Disposition.MERCHANT_FIX, {Ask.NOTHING}) == {Ask.NOTHING}
        # Terminal episodes produce no customer-facing action at all.
        assert Disposition.TERMINAL not in seen

    def test_a_brief_cannot_be_built_for_an_unadjudicated_action(self, episodes):
        """No path from a real episode to a real message skips the gate.

        This test found that the path existed. ``build`` structurally typed
        *both* of its arguments, which is right for the episode record — that
        comes from whatever system holds it — and wrong for the approval, whose
        entire purpose is to be evidence the gate was consulted. Any object
        carrying three attributes forged one, so the guarantee rested on
        callers choosing not to.
        """
        view = episodes[0].view()
        forgery = type(
            "NotAnApproval",
            (),
            {
                "episode_id": view.episode_id,
                "action": Action.NUDGE_SMS,
                "at": dt.datetime(2026, 3, 1, 11, 0),
            },
        )()
        with pytest.raises(TypeError, match="real ApprovedAction"):
            MessageBrief.build(
                forgery, view, merchant=MERCHANT, language=Language.EN
            )

    def test_a_structurally_illegal_action_is_refused(self, episodes):
        """Compliance and structural legality are different questions.

        The gate answers only the first: it has no rule about dispositions, so
        it approves ``REQUEST_REMANDATE`` on a ``RETRY_TIMING`` episode, which
        the taxonomy calls meaningless there. Composed, that becomes "set up
        your autopay mandate again" for a customer whose balance was briefly
        short — a message nothing downstream can fault, because it is
        internally consistent, quotes the right amount and honours the
        instruction it was given. The instruction was wrong.
        """
        gate = ComplianceGate()
        timing = [
            e for e in episodes if e.view().disposition is Disposition.RETRY_TIMING
        ]
        assert timing, "no timing failures in the batch"
        view = timing[0].view()
        at = view.failed_at + dt.timedelta(hours=18)

        decision = gate.adjudicate(
            Request.from_view(view, Action.REQUEST_REMANDATE, at), record=False
        )
        assert decision.verdict is Verdict.ALLOW, (
            "the gate is expected to permit this; the point is that permitting "
            "it is not the same as it being meaningful"
        )
        with pytest.raises(ValueError, match="not a legal action"):
            MessageBrief.build(
                decision.approval,
                view,
                merchant=MERCHANT,
                language=Language.EN,
            )

    def test_real_amounts_never_break_the_sms_budget(self, episodes):
        """Hindi has 134 UCS-2 units for two segments; amounts vary in width.

        A template that fits at Rs.1,299 and spills a third segment at
        Rs.15,00,000 is a per-message cost overrun nobody notices until the
        bill arrives.
        """
        desk = CommsDesk(drafter=TemplateDrafter())
        widest = 0
        for view, approval in _composable(episodes):
            if CHANNEL_FOR_ACTION[approval.action].value != "sms":
                continue
            brief = MessageBrief.build(
                approval, view, merchant=MERCHANT, language=Language.HI,
                retry_on=(approval.at + dt.timedelta(days=2)).date(),
            )
            result = desk.compose(brief)
            assert result.cleared, [f.detail for f in result.findings]
            widest = max(widest, len(brief.amount_rupees))
        assert widest >= 3, "no amount wide enough to be worth the test"


class TestTheRecordSurvivesRealData:
    def test_every_composition_produces_an_audit_row(self, episodes):
        desk = CommsDesk(drafter=TemplateDrafter())
        rows = []
        for view, approval in _composable(episodes)[:50]:
            brief = MessageBrief.build(
                approval, view, merchant=MERCHANT, language=Language.HINGLISH,
                retry_on=(approval.at + dt.timedelta(days=2)).date(),
                link=LINK if approval.action is Action.SEND_COLLECT_LINK else None,
            )
            rows.append(desk.compose(brief).record())

        assert len(rows) == 50
        for row in rows:
            assert row["episode_id"].startswith("EP_")
            assert row["cleared"] is True
            assert row["sent"]
            # The audit row carries the customer id, which the message never
            # does — the record is for us, the message is for them.
            assert row["customer_id"] not in row["sent"]
