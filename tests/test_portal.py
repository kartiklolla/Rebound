"""Tests for the customer-facing request layer.

Two properties carry the design and both are easy to lose in a refactor: a
request to stop being contacted is never adjudicated, and a permitted request
is never refused for being unprofitable.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

from rebound.compliance import ComplianceGate, Verdict
from rebound.eval.harness import build_eval_batch
from rebound.portal import (
    REQUEST_ACTIONS,
    REQUEST_LABELS,
    CustomerRequest,
    Outcome,
    answer,
)
from rebound.sim.world import World
from rebound.taxonomy import Action, Disposition, legal_actions

NOW = dt.datetime(2026, 1, 15, 14, 0)


@pytest.fixture(scope="module")
def views():
    """One real episode per disposition the simulator produces."""
    world = World(seed=7)
    customers = world.sample_customers(600)
    mandates = world.sample_mandates(
        customers, dt.date(2024, 1, 1), dt.date(2025, 6, 30)
    )
    world.calibrate(
        customers,
        mandates,
        [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(14)],
    )
    batch = build_eval_batch(
        world, customers, mandates, dt.date(2026, 1, 1), dt.date(2026, 2, 28)
    )
    by_disposition = {}
    for spec in batch:
        episode = world.open_episode(
            "EP_T", spec.mandate, spec.customer, spec.failure_code,
            spec.failed_at, spec.cycles_elapsed,
        )
        view = episode.view()
        by_disposition.setdefault(view.disposition, view)
    assert len(by_disposition) >= 4, by_disposition.keys()
    return by_disposition


class TestStopContactIsNotARequest:
    """It is honoured, not adjudicated.

    Routing it through a gate that *could* refuse would encode the idea that it
    is ours to refuse, and the shape of the code is what people read fastest.
    """

    def test_it_is_honoured_for_every_disposition(self, views):
        for disposition, view in views.items():
            verdict = answer(view, CustomerRequest.PAUSE_CONTACT, NOW)
            assert verdict.outcome is Outcome.HONOURED, disposition
            assert verdict.permitted

    def test_no_rule_is_consulted(self, views):
        gate = ComplianceGate()
        view = next(iter(views.values()))
        verdict = answer(view, CustomerRequest.PAUSE_CONTACT, NOW, gate=gate)
        assert verdict.gate is None
        assert not verdict.adjudicated
        assert gate.audit == [], "a stop-contact request reached the gate"

    def test_it_maps_to_no_action(self):
        assert REQUEST_ACTIONS[CustomerRequest.PAUSE_CONTACT] is None


class TestPermissionIsNotWorth:
    def test_a_permitted_request_is_never_refused_for_being_unprofitable(
        self, views
    ):
        """The expected value was estimated for actions *we* initiate.

        Someone who has opened the page and pressed the button is not the
        customer that number was fitted on, so it does not get a veto. It is
        still computed and recorded.
        """

        class Pessimist:
            """A pricer that hates everything."""

            passive_revocation_rate = 0.07

            def price(self, view, candidates):
                import numpy as np

                n = len(candidates)
                return np.zeros(n), np.ones(n)

        refused_for_money = []
        for view in views.values():
            for request in CustomerRequest:
                verdict = answer(view, request, NOW, pricer=Pessimist())
                if verdict.candidate is not None:
                    assert verdict.worth_doing_unprompted is False
                if verdict.outcome is Outcome.DECLINED and verdict.gate is None:
                    continue  # structural, not economic
                if verdict.outcome is Outcome.DECLINED:
                    assert verdict.gate is not None
                    assert verdict.gate.verdict is Verdict.DENY, (
                        "declined without a gate denial behind it"
                    )
                    refused_for_money.append(request)
        assert not [r for r in refused_for_money], refused_for_money

    def test_the_grading_is_recorded_even_when_the_gate_refuses(self, views):
        class Flat:
            passive_revocation_rate = 0.07

            def price(self, view, candidates):
                import numpy as np

                n = len(candidates)
                return np.full(n, 0.4), np.full(n, 0.02)

        graded = 0
        for view in views.values():
            for request in CustomerRequest:
                verdict = answer(view, request, NOW, pricer=Flat())
                if verdict.gate is not None:
                    assert verdict.candidate is not None
                    assert verdict.expected_value_paise is not None
                    graded += 1
        assert graded > 5


class TestStructuralRefusalsExplainThemselves:
    def test_a_terminal_mandate_refuses_before_the_gate_is_consulted(self, views):
        terminal = views.get(Disposition.TERMINAL)
        if terminal is None:
            pytest.skip("no terminal failure in this batch")
        gate = ComplianceGate()
        verdict = answer(terminal, CustomerRequest.RETRY_NOW, NOW, gate=gate)
        assert verdict.outcome is Outcome.DECLINED
        assert verdict.gate is None
        assert gate.audit == []
        assert "cancelled" in verdict.headline

    def test_every_request_maps_to_a_legal_action_or_declines(self, views):
        for view in views.values():
            for request in CustomerRequest:
                action = REQUEST_ACTIONS[request]
                verdict = answer(view, request, NOW)
                if action is None:
                    continue
                if action not in legal_actions(view.failure_code):
                    assert verdict.outcome is Outcome.DECLINED, (
                        f"{request} on {view.failure_code} was not refused"
                    )


class TestEveryRequestIsAnswerable:
    def test_no_request_raises_on_any_disposition(self, views):
        for view in views.values():
            for request in CustomerRequest:
                verdict = answer(view, request, NOW)
                assert verdict.headline
                assert verdict.detail
                assert verdict.outcome in set(Outcome)

    def test_every_request_has_a_customer_facing_label(self):
        assert set(REQUEST_LABELS) == set(CustomerRequest)
        for label in REQUEST_LABELS.values():
            assert label and label[0].isupper()

    def test_a_scheduled_request_says_when(self, views):
        found = False
        for view in views.values():
            for request in CustomerRequest:
                verdict = answer(view, request, dt.datetime(2026, 1, 15, 3, 0))
                if verdict.outcome is Outcome.SCHEDULED:
                    assert verdict.happens_at is not None
                    assert verdict.happens_at > dt.datetime(2026, 1, 15, 3, 0)
                    found = True
        assert found, "no request deferred at 03:00; the fixture proves nothing"

    def test_a_payday_retry_lands_inside_an_execution_window(self, views):
        from rebound.regulation import within_upi_execution_window
        from rebound.taxonomy import Rail

        for view in views.values():
            if view.rail is not Rail.UPI_AUTOPAY:
                continue
            verdict = answer(view, CustomerRequest.RETRY_AFTER_PAYDAY, NOW)
            if verdict.happens_at is not None:
                assert within_upi_execution_window(verdict.happens_at)


class TestStopContactIsActuallyEnforced:
    """The promise, and the mechanism behind it.

    `answer()` used to reply "we will not message you about this payment again"
    and nothing anywhere could have stopped the next nudge — no flag on the
    episode, no rule in the gate, no check in the sequencer. An unenforceable
    undertaking on a system whose pitch is contact discipline is the one a
    compliance-minded reader finds first.

    Three separable pieces: the portal reports, the episode records, the gate
    refuses. Each is tested alone above; this is the chain.
    """

    def _episode(self, spec, world):
        return world.open_episode(
            "EP_SUP", spec.mandate, spec.customer, spec.failure_code,
            spec.failed_at, spec.cycles_elapsed,
        )

    def test_the_verdict_asks_the_caller_to_record_it(self, views):
        view = next(iter(views.values()))
        verdict = answer(view, CustomerRequest.PAUSE_CONTACT, NOW)
        assert verdict.suppresses_contact
        # And nothing else does.
        for request in CustomerRequest:
            if request is CustomerRequest.PAUSE_CONTACT:
                continue
            assert not answer(view, request, NOW).suppresses_contact

    def test_answering_does_not_mutate_the_caller_s_view(self, views):
        """A read-only projection stays read-only.

        An earlier version reached in with object.__setattr__ — the same move
        the comms layer was hardened against when a drafter used it to rewrite
        the brief it was about to be verified against.
        """
        view = next(iter(views.values()))
        before = view.contact_suppressed
        answer(view, CustomerRequest.PAUSE_CONTACT, NOW)
        assert view.contact_suppressed == before is False

    def test_once_recorded_the_gate_refuses_every_contact(self, views):
        from rebound.compliance import Request
        from rebound.taxonomy import CUSTOMER_FACING_ACTIONS

        view = next(
            v for v in views.values()
            if v.disposition is Disposition.RETRY_TIMING
        )
        gate = ComplianceGate()
        at = view.failed_at + dt.timedelta(hours=20)

        suppressed = dataclasses.replace(view, contact_suppressed=True)
        for action in sorted(CUSTOMER_FACING_ACTIONS, key=str):
            if action not in legal_actions(view.failure_code):
                continue
            before = gate.adjudicate(
                Request.from_view(view, action, at), record=False
            )
            after = gate.adjudicate(
                Request.from_view(suppressed, action, at), record=False
            )
            assert after.verdict is Verdict.DENY, action
            assert after.binding is not None
            assert after.binding.rule_id == "POL.CONTACT_SUPPRESSED"
            # And the suppression is the *only* difference: an action the gate
            # already refused is not evidence the rule works.
            if before.verdict is Verdict.DENY:
                assert before.binding.rule_id != "POL.CONTACT_SUPPRESSED"

    def test_a_debit_is_not_a_contact(self, views):
        """Suppression stops messages, not collection.

        A customer asking not to be messaged has not asked us to stop
        collecting a payment they agreed to. Denying the debit too would be a
        different and much larger promise than the one the button makes.
        """
        from rebound.compliance import Request

        view = next(
            v for v in views.values()
            if v.disposition is Disposition.RETRY_TIMING
        )
        suppressed = dataclasses.replace(view, contact_suppressed=True)
        gate = ComplianceGate()
        at = dt.datetime.combine(view.failed_at.date(), dt.time(14, 0))
        decision = gate.adjudicate(
            Request.from_view(suppressed, Action.RETRY_SAME_RAIL, at),
            record=False,
        )
        assert decision.verdict is not Verdict.DENY or (
            decision.binding.rule_id != "POL.CONTACT_SUPPRESSED"
        )

    def test_stop_and_the_pre_debit_notice_stay_permitted(self, views):
        """A gate that can deny STOP can trap an episode open, and a notice is
        a disclosure the following debit depends on rather than a nudge."""
        from rebound.compliance import ContactSuppressed, Request

        view = next(iter(views.values()))
        suppressed = dataclasses.replace(view, contact_suppressed=True)
        at = view.failed_at + dt.timedelta(hours=20)
        for action in (Action.STOP, Action.SEND_PRE_DEBIT_NOTIFICATION):
            ruling = ContactSuppressed().check(
                Request.from_view(suppressed, action, at)
            )
            assert ruling is None, action
