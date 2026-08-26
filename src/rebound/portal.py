"""What a customer whose debit failed can ask for, and how it is answered.

The rest of this system is merchant-initiated: the sequencer decides whether to
contact someone, and the customer is on the receiving end. This module is the
other direction — a customer opens the payment page they were linked to and
asks for something. "Try again now." "Send me a link." "Take it after payday."
"Stop messaging me."

Two things are deliberately separate here, and conflating them would be the
easiest mistake to make.

**Permission is not the same as worth.** :mod:`rebound.compliance` says whether
an action is *allowed*; the sequencer's expected value says whether it is
*worth doing unprompted*. Those are different questions with different owners,
and a customer request only has to clear the first. Declining a customer's own
explicit request on the grounds that our model does not expect it to pay for
itself would be indefensible — the expected value was estimated for actions we
initiate, against a customer who did not ask. Someone who has opened the page
and pressed the button is not that customer. The number is still computed and
still shown to whoever operates this, because "we did it anyway and here is
what we thought it was worth" is a defensible record and "we did not compute
it" is not.

**Some requests are not requests.** "Stop contacting me" is not adjudicated.
There is no verdict, no rule to cite and no circumstance in which the answer is
no. Routing it through a gate that *could* say no would encode the idea that it
is ours to refuse, and the shape of the code is the thing people read fastest.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum

from .compliance import ComplianceGate, Decision, Request, Verdict
from .regulation import next_execution_window_open
from .sequencer import ActionPricer, Candidate
from .taxonomy import Action, Disposition, legal_actions

__all__ = [
    "CustomerRequest",
    "Outcome",
    "RequestVerdict",
    "REQUEST_ACTIONS",
    "REQUEST_LABELS",
    "answer",
]


class CustomerRequest(StrEnum):
    """Everything the customer-facing page offers."""

    RETRY_NOW = "retry_now"
    SEND_PAYMENT_LINK = "send_payment_link"
    RETRY_AFTER_PAYDAY = "retry_after_payday"
    UPDATE_MY_MANDATE = "update_my_mandate"
    PAUSE_CONTACT = "pause_contact"


#: The action each request would actually take. ``None`` for requests that are
#: not actions against the mandate at all.
REQUEST_ACTIONS: dict[CustomerRequest, Action | None] = {
    CustomerRequest.RETRY_NOW: Action.RETRY_SAME_RAIL,
    CustomerRequest.SEND_PAYMENT_LINK: Action.SEND_COLLECT_LINK,
    CustomerRequest.RETRY_AFTER_PAYDAY: Action.RETRY_SAME_RAIL,
    CustomerRequest.UPDATE_MY_MANDATE: Action.REQUEST_MANDATE_AMENDMENT,
    CustomerRequest.PAUSE_CONTACT: None,
}

REQUEST_LABELS: dict[CustomerRequest, str] = {
    CustomerRequest.RETRY_NOW: "Try my payment again now",
    CustomerRequest.SEND_PAYMENT_LINK: "Send me a payment link",
    CustomerRequest.RETRY_AFTER_PAYDAY: "Take it after my salary lands",
    CustomerRequest.UPDATE_MY_MANDATE: "Fix my autopay setup",
    CustomerRequest.PAUSE_CONTACT: "Stop messaging me about this",
}


class Outcome(StrEnum):
    """What the customer is told."""

    DONE = "done"
    """Permitted and taken now."""

    SCHEDULED = "scheduled"
    """Permitted, but not yet. Carries the moment it happens."""

    DECLINED = "declined"
    """Not permitted, and waiting will not change that."""

    HONOURED = "honoured"
    """Not adjudicated at all. See the module docstring."""


@dataclass(frozen=True, slots=True)
class RequestVerdict:
    """One customer request and everything behind the answer.

    Carries the gate decision and the priced candidate side by side rather
    than collapsing them into a single "approved" flag, because the two can
    disagree and the disagreement is the interesting part: an action can be
    entirely permitted and still be one we would never have taken on our own.
    """

    request: CustomerRequest
    action: Action | None
    outcome: Outcome
    headline: str
    """One line, written for the customer."""

    detail: str
    """One line, written for whoever has to defend it later."""

    gate: Decision | None = None
    candidate: Candidate | None = None
    happens_at: dt.datetime | None = None

    suppresses_contact: bool = False
    """The caller must record this on the episode for the promise to hold.

    Reported rather than applied. An earlier version reached into the frozen
    ``EpisodeView`` with ``object.__setattr__`` — which is precisely the move
    the comms layer was hardened against when a drafter used it to rewrite the
    brief it was about to be verified against. A read-only projection is
    read-only for the same reason in both directions.

    So the chain is three separable pieces, each testable on its own: the
    portal reports the customer asked, the episode records it, and
    ``compliance.ContactSuppressed`` refuses every customer-facing action
    afterwards. Before that rule existed the reply was an undertaking with no
    mechanism anywhere behind it.
    """

    @property
    def permitted(self) -> bool:
        return self.outcome in (Outcome.DONE, Outcome.SCHEDULED, Outcome.HONOURED)

    @property
    def adjudicated(self) -> bool:
        return self.gate is not None

    @property
    def expected_value_paise(self) -> float | None:
        return None if self.candidate is None else self.candidate.expected_value_paise

    @property
    def worth_doing_unprompted(self) -> bool | None:
        """Whether we would have chosen this ourselves. Not a gate on the request."""
        ev = self.expected_value_paise
        return None if ev is None else ev > 0


def _payday_retry_time(view: object, now: dt.datetime) -> dt.datetime:
    """A plausible post-salary moment, in the customer's own terms.

    Deliberately naive: the salary day is one of the simulator's hidden latents
    and a merchant genuinely does not know it. The customer is the one asserting
    that money is arriving, so the request carries their claim rather than our
    inference — which is also why this is offered as a button and not as
    something the sequencer schedules on its own.
    """
    candidate = (now + dt.timedelta(days=3)).replace(
        hour=11, minute=0, second=0, microsecond=0
    )
    return next_execution_window_open(candidate)


def answer(
    view: object,
    request: CustomerRequest,
    now: dt.datetime,
    *,
    gate: ComplianceGate | None = None,
    pricer: ActionPricer | None = None,
) -> RequestVerdict:
    """Answer one customer request against one episode.

    ``view`` is anything with the :class:`~rebound.sim.world.EpisodeView`
    shape, matching the discipline everywhere else: this module does not
    import the simulator.
    """
    gate = gate if gate is not None else ComplianceGate()
    action = REQUEST_ACTIONS[request]

    if request is CustomerRequest.PAUSE_CONTACT:
        return RequestVerdict(
            request=request,
            action=None,
            outcome=Outcome.HONOURED,
            headline="Done — we will not message you about this payment again.",
            detail=(
                "Not adjudicated. A request to stop being contacted is honoured "
                "unconditionally; there is no rule to cite because there is no "
                "circumstance in which the answer is no. The caller records "
                "suppresses_contact on the episode, and POL.CONTACT_SUPPRESSED "
                "refuses every customer-facing action from then on."
            ),
            suppresses_contact=True,
        )

    assert action is not None

    # Structural legality first. The taxonomy, not the gate, is what says a
    # closed account cannot be retried — and it is a better answer to the
    # customer, because it explains rather than refuses.
    if action not in legal_actions(getattr(view, "failure_code")):
        return RequestVerdict(
            request=request,
            action=action,
            outcome=Outcome.DECLINED,
            headline=_structural_headline(view, request),
            detail=(
                f"{action} is not a legal action for "
                f"{getattr(view, 'failure_code')} "
                f"({getattr(view, 'disposition')}); refused by the taxonomy "
                "before the gate was consulted"
            ),
        )

    at = (
        _payday_retry_time(view, now)
        if request is CustomerRequest.RETRY_AFTER_PAYDAY
        else now
    )
    decision = gate.adjudicate(Request.from_view(view, action, at), record=True)

    candidate = None
    if pricer is not None:
        candidate = _price(pricer, view, action, decision, at)

    if decision.verdict is Verdict.ALLOW:
        return RequestVerdict(
            request=request,
            action=action,
            outcome=Outcome.DONE,
            headline=_allowed_headline(request, at, now),
            detail=decision.explain(),
            gate=decision,
            candidate=candidate,
            happens_at=at,
        )

    if decision.verdict is Verdict.DEFER and decision.earliest_allowed_at:
        when = decision.earliest_allowed_at
        return RequestVerdict(
            request=request,
            action=action,
            outcome=Outcome.SCHEDULED,
            headline=(
                f"Scheduled for {when:%d %b, %H:%M}. "
                f"{_defer_reason(decision)}"
            ),
            detail=decision.explain(),
            gate=decision,
            candidate=candidate,
            happens_at=when,
        )

    return RequestVerdict(
        request=request,
        action=action,
        outcome=Outcome.DECLINED,
        headline=_denied_headline(decision),
        detail=decision.explain(),
        gate=decision,
        candidate=candidate,
    )


def _price(
    pricer: ActionPricer,
    view: object,
    action: Action,
    decision: Decision,
    at: dt.datetime,
) -> Candidate | None:
    """The sequencer's own grading of this action, for the record.

    Computed even when the gate refused, because "what did you think this was
    worth at the moment you turned it down" is a question an operator will
    have, and recomputing it later against a changed model is not an answer.
    """
    from .economics import attempt_cost_paise, presenting_rail, revocation_cost_paise

    try:
        p_recover, p_revoke = pricer.price(view, [(action, at)])
    except Exception:  # noqa: BLE001 - a grading failure must not refuse a request
        return None
    amount = getattr(view, "cycle_amount_paise")
    return Candidate(
        action=action,
        at=at,
        p_recover=float(p_recover[0]),
        p_revoke=float(p_revoke[0]),
        value_paise=amount,
        cost_paise=attempt_cost_paise(
            action, presenting_rail(action, getattr(view, "rail"))
        ),
        revocation_cost_paise=revocation_cost_paise(amount),
        passive_revocation_rate=getattr(pricer, "passive_revocation_rate", 0.0),
    )


def _structural_headline(view: object, request: CustomerRequest) -> str:
    """Why the taxonomy refused, in the customer's terms.

    Branches on ``mandate_alive``, not on the disposition, and the difference
    is a customer being told their subscription is dead when it is not.
    ``UPI_RETRY_LIMIT_EXCEEDED`` is ``TERMINAL`` with ``mandate_alive=True`` —
    terminal for *this cycle*, not for the mandate — and the taxonomy has
    carried a note since the first commit saying "a system that cannot tell
    those apart writes off live customers". Branching on disposition did
    exactly that, on the demo's front page, for every request on that account.
    """
    disposition = getattr(view, "disposition")
    if disposition is Disposition.TERMINAL:
        if getattr(view, "mandate_alive", False):
            return (
                "This payment has already been tried the maximum number of "
                "times this cycle. Your autopay is still active and we will "
                "collect it with your next cycle — there is nothing to fix."
            )
        return (
            "This mandate has been cancelled, so we cannot collect on it. "
            "Set up a new one to continue."
        )
    if request in (CustomerRequest.RETRY_NOW, CustomerRequest.RETRY_AFTER_PAYDAY):
        return (
            "A retry will not help here — the payment needs a change on your "
            "side first."
        )
    return "That is not something we can do for this payment."


def _allowed_headline(
    request: CustomerRequest, at: dt.datetime, now: dt.datetime
) -> str:
    if request is CustomerRequest.SEND_PAYMENT_LINK:
        return "Sent. Check your WhatsApp for the payment link."
    if request is CustomerRequest.UPDATE_MY_MANDATE:
        return "Sent. Approve the request from your bank to finish."
    if at > now:
        return f"Booked for {at:%d %b, %H:%M}."
    return "Trying now — we will let you know either way."


def _defer_reason(decision: Decision) -> str:
    governing = decision.governing
    if not governing:
        return ""
    rule = governing[0]
    friendly = {
        "REG.EXECUTION_WINDOW": "Your bank only accepts autopay debits in set windows.",
        "POL.QUIET_HOURS": "We do not message customers at night.",
        "REG.PRE_DEBIT_NOTICE": "We have to give you 24 hours' notice first.",
    }
    return friendly.get(rule.rule_id, "")


def _denied_headline(decision: Decision) -> str:
    governing = decision.governing
    if governing and governing[0].rule_id == "REG.RETRY_CAP":
        return (
            "This payment has already been attempted the maximum number of "
            "times this cycle. It will be tried again next cycle."
        )
    if governing and governing[0].rule_id == "REG.AFA_CEILING":
        return (
            "This amount is above the limit for automatic debits. You will "
            "need to approve it directly with your bank."
        )
    if governing and governing[0].rule_id == "POL.SPEND_BUDGET":
        return "We have stopped chasing this payment. Nothing further will be charged."
    return "We cannot do that for this payment right now."
