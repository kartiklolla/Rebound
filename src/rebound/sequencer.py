"""The sequencer: what to do about a failed debit, and when.

This is the component the rest of the system exists to serve. It composes three
things that were built separately and measured separately:

- the **models**, which price how likely an action is to work,
- the **economics**, which price what it costs and what it destroys,
- the **compliance gate**, which decides what is permitted at all.

It is a policy like any other, scored by the same harness as the hand-written
baselines. No special path, no privileged information.

Expected value, not recovery rate
---------------------------------
The instinct is to maximise recovery. The baselines already show where that
leads: ``aggressive_contact`` recovers more debits than the naive ladder and
destroys ₹346,088 per thousand doing it, because every contact carries
revocation risk and recovery rate does not charge for it. It is the worst policy
in the table *and* it wins the metric merchants usually watch.

So each candidate is priced against the counterfactual of stopping now:

    EV = [P(recover | act) − P(recover | stop)] × amount
       − cost(action)
       − [P(revoke | act) − P(revoke | stop)] × LTV destroyed

The third term is why there is a revocation head at all. Without it the
sequencer optimises the same quantity ``aggressive_contact`` optimises, and
arrives at the same place while looking locally correct at every step.

Both terms are **marginal**, and that is not a refinement. Customers revoke
without being contacted — the measured floor is 8.78% under a policy that does
nothing at all — so billing every action for the full revocation probability
charges it for churn that was already coming. At a 12-cycle horizon that term
runs an order of magnitude above the recovery term, so the double-count does not
bias the answer, it *is* the answer: the first working version stopped on 397 of
516 episodes and recovered 0.2306 where the naive ladder recovered 0.5155.

Two heads for two questions
---------------------------
The heads are used where each was measured to be better, rather than blended
into a single number that neither supports.

**When** is chosen by the timing head, which predicts immediate success and
beats the action head on that question on matched rows (ROC 0.9424 against
0.9132). Retry success against insufficient funds runs 0.65 near payday and 0.22
three weeks later — a spread the episode-level label washes out.

**What** is priced by the action head, which predicts whether the episode
ultimately recovers. That is the quantity expected value actually needs: the
value of an action is what the episode ends up doing, not whether this one
presentation collects.

Averaging the two would produce a number that is not a probability of anything.

The gate is upstream of the arithmetic
--------------------------------------
Candidates are filtered before they are priced, never after. A deferred action
re-enters as a candidate *at the time it becomes legal*, which is the point of
the gate returning a moment rather than a refusal — a retry that is blocked now
and permitted at 13:00 is a real option, and a sequencer that only asked
"allowed right now?" would throw it away.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rebound.compliance import ComplianceGate, Request, Verdict
from rebound.economics import attempt_cost_paise, revocation_cost_paise
from rebound.model import (
    COLLECTING_ACTIONS,
    TARGET_REVOKED,
    RecoveryModel,
    TwoHeadedModel,
)
from rebound.policy import Decision, Policy
from rebound.sim.world import EpisodeView
from rebound.taxonomy import Action, get_mode, legal_actions

#: Features the training log has and a deployed sequencer does not.
#:
#: ``EpisodeView`` is scoped to one episode; the ``cust_prior_*`` block is
#: cross-episode customer history that the harness does not put in front of a
#: policy. The tempting fix is to pass zeros at inference — which is train/serve
#: skew of the worst kind, because it is silent: the model is scored on one
#: distribution and deployed on another, every prediction shifts, and nothing
#: raises. A model that reports 0.62 PR-AUC in evaluation and sees a different
#: input shape in production has not been evaluated.
#:
#: So the sequencer's models are fitted *without* them. Train on what can be
#: served. This costs some accuracy against the Claim A numbers, which is the
#: correct price and is reported rather than absorbed: ``fit_for_serving``
#: exists so the gap is a measured quantity instead of an unexamined one.
UNSERVABLE_COLUMNS: tuple[str, ...] = (
    "cust_prior_failures",
    "cust_prior_recoveries",
    "cust_prior_recovery_rate",
    "cust_prior_contacts",
)


def fit_for_serving(
    train: pd.DataFrame,
    order_by: str | None = "decided_at",
    group_by: str | None = "episode_id",
    max_iter: int = 200,
) -> ActionPricer:
    """Fit all three heads on the columns a deployed sequencer can actually see.

    The dropped columns are real signal, and dropping them is still right: a
    feature the serving path cannot produce is not a feature, it is a
    discrepancy waiting to be discovered in production.
    """
    servable = train.drop(columns=list(UNSERVABLE_COLUMNS), errors="ignore")
    heads = TwoHeadedModel(max_iter=max_iter).fit(
        servable, order_by=order_by, group_by=group_by
    )
    revocation = RecoveryModel(target=TARGET_REVOKED, max_iter=max_iter).fit(
        servable, order_by=order_by, group_by=group_by
    )
    return ActionPricer(heads=heads, revocation=revocation)


#: How far ahead the sequencer will consider scheduling an action.
#:
#: Bounded deliberately. An unbounded search would always find some distant
#: moment with a marginally better predicted probability, and "wait longer"
#: is not free — the recovery window closes, and every day a mandate sits in
#: failure is a day the next cycle gets closer.
_HORIZON_HOURS: tuple[int, ...] = (0, 6, 12, 24, 48, 72, 120, 168)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One priced option, kept whether or not it wins.

    The losers are the audit trail. "Why did you call instead of retrying"
    cannot be answered by a record that only stored the winner.
    """

    action: Action
    at: dt.datetime
    p_recover: float
    p_revoke: float
    value_paise: int
    cost_paise: int
    revocation_cost_paise: int

    baseline_recover: float
    """P(recover) if we stop here instead. Usually near zero, not exactly."""

    baseline_revoke: float
    """P(revoke) if we stop here instead.

    The number that makes the arithmetic honest. Customers revoke on their own:
    the measured floor is 8.78% with no contact at all. Charging an action the
    *full* revocation probability bills it for churn that was going to happen
    anyway, and at a 12-cycle horizon that term is an order of magnitude larger
    than the recovery term, so the double-count does not merely bias the answer
    - it decides it.
    """

    @property
    def expected_value_paise(self) -> float:
        """Value of acting, measured against not acting.

        Both terms are marginal, because the comparison is not "act versus a
        world where nothing happens" but "act versus stop here". Stopping does
        not yield zero revocation; it yields the passive hazard.

        With the absolute form, ``p_revoke * 12 * amount`` at a base rate of
        0.07 swamped ``p_recover * amount`` at 0.4, so expected value was
        negative nearly everywhere and the sequencer stopped on 397 of 516
        episodes - recovering 0.2306 where the naive ladder recovered 0.5155.
        It was not being cautious. It was solving the wrong equation.
        """
        return (
            (self.p_recover - self.baseline_recover) * self.value_paise
            - self.cost_paise
            - self.marginal_revocation * self.revocation_cost_paise
        )

    @property
    def marginal_revocation(self) -> float:
        """Extra revocation risk this action carries, floored at zero.

        The floor is the load-bearing part. Left unclamped, 30.7% of candidates
        come back *credited* for reducing revocation, and at a 12-cycle horizon
        an observed delta of −0.0659 pays out +0.79 × amount — enough to make a
        voice call look profitable on any episode in the book.

        Those credits are not a finding, they are an artifact. In the training
        log ``stop`` carries the highest observed revocation rate (0.1038) and
        ``retry_same_rail`` the lowest (0.0582), because the behavioural policy
        stops on episodes that are already lost and a stopped episode never gets
        the chance to recover. The head learns "acting prevents revocation" from
        selection, not from causation.

        So the sequencer declines to be paid for churn it did not prevent. This
        is a guard against a known-bad estimate, not a correction of it: the
        underlying quantity is still unidentified, and the fix is a per-action
        revocation label rather than a clamp.
        """
        return max(0.0, self.p_revoke - self.baseline_revoke)

    def explain(self) -> str:
        return (
            f"{self.action}@{self.at:%m-%d %H:%M} "
            f"p_rec={self.p_recover:.3f} (base {self.baseline_recover:.3f}) "
            f"p_rev={self.p_revoke:.3f} (base {self.baseline_revoke:.3f}, "
            f"marginal {self.marginal_revocation:.3f}) "
            f"ev={self.expected_value_paise / 100:,.0f}"
        )


@dataclass
class ActionPricer:
    """Turns an episode and a candidate action into probabilities.

    Holds the fitted models. Separated from the policy so the policy's logic
    can be tested against a stub pricer with no sklearn anywhere near it —
    which is what makes the stopping rule and the gate interaction testable in
    milliseconds instead of minutes.
    """

    heads: TwoHeadedModel
    revocation: RecoveryModel

    def __post_init__(self) -> None:
        self._shares_encoding = _same_spec(
            self.heads.downstream.spec_, self.revocation.spec_
        )

    def encode(self, frame: pd.DataFrame):
        """Encode once for the action and revocation heads.

        Both are fitted on the same servable columns, so they normally share an
        encoding — worth about a third of the per-decision cost, since encoding
        and predicting cost roughly the same and the whole path is fixed-cost
        per call.

        Guarded, not assumed. If the two specs ever diverge, this returns
        ``None`` and the caller encodes separately. Sharing a matrix across
        models with different column orders or category levels would predict on
        mis-encoded input and return confident nonsense with nothing raising.
        """
        if not self._shares_encoding:
            return None
        return self.heads.downstream.spec_.transform(frame)

    def frame(
        self, episode: EpisodeView, candidates: list[tuple[Action, dt.datetime]]
    ) -> pd.DataFrame:
        """Build one feature row per candidate.

        Mirrors ``dataset._observable_features``. If the two ever drift, the
        model is scored on one distribution and deployed on another — so the
        column set is taken from the fitted spec rather than restated here, and
        anything missing surfaces as a KeyError rather than as a quietly worse
        prediction.
        """
        rows = [
            {
                **_observable(episode, at),
                "action": str(action),
            }
            for action, at in candidates
        ]
        return pd.DataFrame(rows)

    def price(
        self, episode: EpisodeView, candidates: list[tuple[Action, dt.datetime]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(p_recover, p_revoke)`` for each candidate."""
        frame = self.frame(episode, candidates)
        return (
            self.heads.predict_downstream(frame),
            self.revocation.predict_proba(frame),
        )

    def rank_times(
        self, episode: EpisodeView, action: Action, times: list[dt.datetime]
    ) -> list[float]:
        """Immediate-success probability for one action across candidate times.

        Only meaningful for collecting actions — a nudge cannot collect, so its
        immediate label is always 0 and the timing head was never trained on
        those rows.
        """
        frame = self.frame(episode, [(action, at) for at in times])
        return list(self.heads.predict_immediate(frame))


@dataclass
class Sequencer(Policy):
    """Model-driven recovery, bounded by the compliance gate.

    The decision procedure, in order:

    1. Ask the taxonomy which actions the disposition even admits.
    2. Ask the gate which of those are permitted, and when. A deferral becomes
       a candidate at the moment it is permitted, not a discarded option.
    3. For collecting actions, let the timing head choose the moment.
    4. Price every surviving candidate by expected value.
    5. Take the best, or stop if nothing is worth doing.
    """

    pricer: ActionPricer
    gate: ComplianceGate = field(default_factory=ComplianceGate)
    horizon_hours: tuple[int, ...] = _HORIZON_HOURS
    name: str = "rebound_sequencer"
    description: str = (
        "Expected-value sequencer: two-headed recovery model plus a revocation "
        "head, priced against economics, filtered by the compliance gate."
    )
    trail: list[dict[str, object]] = field(default_factory=list)

    def reset(self) -> None:
        self.trail.clear()
        self.gate.audit.clear()

    # -- candidate construction -------------------------------------------

    def _candidate_times(
        self, now: dt.datetime, deadline: dt.datetime
    ) -> list[dt.datetime]:
        times = [now + dt.timedelta(hours=h) for h in self.horizon_hours]
        return [t for t in times if t <= deadline] or [now]

    def _permitted(
        self, episode: EpisodeView, now: dt.datetime, deadline: dt.datetime
    ) -> list[tuple[Action, dt.datetime]]:
        """Actions the gate allows, each at the earliest moment it allows them.

        A DEFER is not a refusal. The gate returns a moment precisely so this
        loop can keep the option and schedule it, and dropping deferrals here
        would discard most of the timing value the gate exists to express.
        """
        options: list[tuple[Action, dt.datetime]] = []
        # Sorted, because ``legal_actions`` returns a frozenset and StrEnum
        # iteration order is salted by PYTHONHASHSEED. That order reaches
        # ``max`` over expected values, and RETRY_SAME_RAIL and RETRY_ALT_RAIL
        # share cost, value and revocation cost — so identical model output
        # produced exact ties broken by list position, and net value moved 2.2%
        # between interpreters running the same code on the same batch. A
        # single-process test cannot see it.
        for action in sorted(legal_actions(episode.failure_code), key=str):
            if action is Action.STOP:
                continue
            decision = self.gate.adjudicate(
                Request.from_view(episode, action, now), record=False
            )
            if decision.verdict is Verdict.ALLOW:
                options.append((action, now))
            elif decision.verdict is Verdict.DEFER:
                when = decision.earliest_allowed_at
                if when is not None and when <= deadline:
                    options.append((action, when))
        return options

    def _expand(
        self, episode: EpisodeView, now: dt.datetime, deadline: dt.datetime
    ) -> list[tuple[Action, dt.datetime]]:
        """Every (action, time) pair worth scoring, in one list.

        Collecting actions get the full candidate grid because the timing head
        has something to say about when to present them. Everything else gets
        the single earliest permitted moment — a nudge's immediate label is
        always 0, so the timing head was never trained on those rows and asking
        it to rank their moments would be reading a number it cannot have.
        """
        pairs: list[tuple[Action, dt.datetime]] = []
        for action, earliest in self._permitted(episode, now, deadline):
            if str(action) not in COLLECTING_ACTIONS:
                pairs.append((action, earliest))
                continue
            for at in self._candidate_times(earliest, deadline):
                if at < earliest or at > deadline:
                    continue
                decision = self.gate.adjudicate(
                    Request.from_view(episode, action, at), record=False
                )
                if decision.verdict is Verdict.ALLOW:
                    pairs.append((action, at))
        return pairs

    # -- the decision ------------------------------------------------------

    def decide(
        self, episode: EpisodeView, now: dt.datetime, deadline: dt.datetime
    ) -> Decision | None:
        pairs = [
            (a, t) for a, t in self._expand(episode, now, deadline) if t <= deadline
        ]
        if not pairs:
            return self._stop(episode, now, "no permitted action before the deadline")

        # One frame, three predictions. The earlier version built a separate
        # frame per action to rank its timings and then another to price the
        # winners — about five sklearn calls per decision on frames of two or
        # three rows, where the per-call overhead dwarfs the arithmetic. The
        # harness killed it at 120 seconds after 4,883 of 6,898 episodes, which
        # is the right verdict: a recovery orchestrator that cannot decide
        # quickly is not deployable, and raising the timeout would have hidden
        # that rather than fixed it.
        # One do-nothing row per distinct candidate time, not one at ``now``.
        #
        # A single baseline at ``now`` looked right and was not: 84% of
        # candidates are scheduled at some other moment, median 48 hours away,
        # and ``_observable`` derives seven features from the timestamp. The
        # difference being maximised was then the action effect confounded with
        # a two-to-seven-day time shift — and the time shift is precisely what
        # the timing head is separately choosing, so the two were fighting over
        # the same quantity.
        times = sorted({at for _, at in pairs})
        baseline_at = {at: len(pairs) + i for i, at in enumerate(times)}
        scored = pairs + [(Action.STOP, at) for at in times]

        frame = self.pricer.frame(episode, scored)
        shared = self.pricer.encode(frame)
        if shared is None:
            p_recover = self.pricer.heads.predict_downstream(frame)
            p_revoke = self.pricer.revocation.predict_proba(frame)
        else:
            p_recover = self.pricer.heads.downstream.predict_proba_prepared(shared)
            p_revoke = self.pricer.revocation.predict_proba_prepared(shared)

        destroyed = revocation_cost_paise(episode.cycle_amount_paise)
        candidates = [
            Candidate(
                action=action,
                at=at,
                p_recover=float(p_recover[i]),
                p_revoke=float(p_revoke[i]),
                value_paise=episode.cycle_amount_paise,
                cost_paise=attempt_cost_paise(action, episode.rail),
                revocation_cost_paise=destroyed,
                baseline_recover=float(p_recover[baseline_at[at]]),
                baseline_revoke=float(p_revoke[baseline_at[at]]),
            )
            for i, (action, at) in enumerate(pairs)
        ]

        # The timing head chooses *when* within each collecting action; the
        # action head then prices *what* across the survivors. Keeping the two
        # questions separate is the point of having two heads, and averaging
        # them would produce a number that is not a probability of anything.
        #
        # Scored only on collecting rows. The immediate head's FeatureSpec was
        # fitted on those rows alone, so every other action falls outside its
        # pinned categories and comes back NaN — running it over the whole
        # frame was a third sklearn call per decision whose output was
        # discarded.
        collecting = [
            i
            for i, c in enumerate(candidates)
            if str(c.action) in COLLECTING_ACTIONS
        ]
        best_time: dict[Action, int] = {
            c.action: i
            for i, c in enumerate(candidates)
            if str(c.action) not in COLLECTING_ACTIONS
        }
        # Only when there is genuinely a choice of moment. With one legal time
        # per collecting action there is nothing to rank, and the call is a
        # fixed ~15ms spent to confirm the only option is the only option.
        rankable = len({candidates[i].action for i in collecting}) < len(collecting)
        if collecting and rankable:
            immediate = self.pricer.heads.predict_immediate(
                frame.iloc[collecting].reset_index(drop=True)
            )
            for rank, i in enumerate(collecting):
                action = candidates[i].action
                incumbent = best_time.get(action)
                if incumbent is None:
                    best_time[action] = i
                    continue
                incumbent_rank = collecting.index(incumbent)
                if immediate[rank] > immediate[incumbent_rank]:
                    best_time[action] = i
        else:
            for i in collecting:
                best_time.setdefault(candidates[i].action, i)

        shortlist = [candidates[i] for i in best_time.values()]
        best = max(shortlist, key=lambda c: (c.expected_value_paise, str(c.action)))

        considered = [c.explain() for c in shortlist]

        if best.expected_value_paise <= 0:
            return self._stop(
                episode,
                now,
                f"best option {best.explain()} has non-positive expected value",
                considered,
            )

        # The gate has the last word even on the winner. Nothing reaches an
        # executor on the strength of an expected value alone.
        approved = self.gate.adjudicate(
            Request.from_view(episode, best.action, best.at)
        )
        if not approved.allowed:
            return self._stop(
                episode, now, f"gate refused: {approved.explain()}", considered
            )

        self._record(episode, now, best.action, best.at, best.expected_value_paise, considered)
        return Decision(
            action=best.action,
            at=best.at,
            reason=(
                f"ev {best.expected_value_paise / 100:,.0f} "
                f"(p_rec {best.p_recover:.3f}, p_rev {best.p_revoke:.3f}) "
                f"over {len(shortlist)} candidates"
            ),
        )

    def _record(
        self,
        episode: EpisodeView,
        now: dt.datetime,
        action: Action,
        at: dt.datetime,
        ev: float,
        considered: list[str],
    ) -> None:
        """One row per decision, written once the branches have resolved.

        The row used to be appended before the stopping check and the gate
        re-check, both of which could then append a second row for the same
        decision. The trail reported ``send_collect_link`` 620 times where 9
        reached the world — two orders of magnitude out on the most-cited
        action, in the record a merchant reads to find out what happened.
        """
        self.trail.append(
            {
                "episode_id": episode.episode_id,
                "at": now,
                "chosen": str(action),
                "scheduled_for": at,
                "expected_value_paise": round(ev, 2),
                "considered": considered,
            }
        )

    def _stop(
        self,
        episode: EpisodeView,
        now: dt.datetime,
        why: str,
        considered: list[str] | None = None,
    ) -> Decision:
        """Stopping is a decision, and it gets logged like one.

        Returning ``None`` would end the episode just as effectively and leave
        no record of why. A merchant asking "why did you give up on this
        customer" is asking about exactly these rows.

        Adjudicated like any other action, even though ``_ALWAYS_PERMITTED``
        guarantees the answer. Skipping it left the compliance trail empty for
        497 of 3,129 decisions — the trail exists so a merchant can ask why
        nothing happened, and stopping *is* nothing happening.
        """
        self.gate.adjudicate(Request.from_view(episode, Action.STOP, now))
        self._record(
            episode, now, Action.STOP, now, 0.0, (considered or []) + [why]
        )
        return Decision(action=Action.STOP, at=now, reason=why)


def _same_spec(left, right) -> bool:
    """Whether two fitted specs encode a frame identically."""
    if left is None or right is None:
        return False
    return (
        list(left.columns) == list(right.columns)
        and left.categories == right.categories
    )


def _observable(episode: EpisodeView, at: dt.datetime) -> dict[str, object]:
    """The feature row for an episode at a moment, minus the action.

    Deliberately built from ``EpisodeView`` alone. If this could reach the live
    episode or the customer, the sequencer would have access the model never had
    at training time and its measured numbers would not transfer.

    Note what is *absent*: the four ``cust_prior_*`` columns. ``EpisodeView``
    does not carry a customer's cross-episode history, so the sequencer cannot
    supply them — see ``UNSERVABLE_COLUMNS``.
    """
    from rebound.regulation import within_upi_execution_window

    return {
        "bank": episode.bank,
        "rail": str(episode.rail),
        "failure_code": episode.failure_code,
        "disposition": str(episode.disposition),
        "mandate_alive": episode.mandate_alive,
        "needs_customer_action": episode.needs_customer_action,
        "is_ambiguous_code": get_mode(episode.failure_code).ambiguous,
        "amount_paise": episode.cycle_amount_paise,
        "ceiling_paise": episode.ceiling_paise,
        "amount_to_ceiling": episode.cycle_amount_paise / episode.ceiling_paise,
        "mandate_age_days": (at.date() - episode.registered_on).days,
        "days_to_expiry": (episode.valid_until - at.date()).days,
        "billing_day": episode.billing_day,
        "cycles_elapsed": episode.cycles_elapsed,
        "days_since_failure": (at - episode.failed_at).total_seconds() / 86400.0,
        "decision_day_of_month": at.day,
        "decision_hour": at.hour,
        "decision_weekday": at.weekday(),
        "within_upi_window": within_upi_execution_window(at),
        "prior_actions": episode.attempts,
        "prior_contacts": episode.contacts_made,
        "notification_sent": episode.notification_sent_at is not None,
        "customer_unblocked": episode.customer_unblocked,
        "mandate_repaired": episode.mandate_repaired,
        "spent_so_far_paise": episode.spent_paise,
    }
