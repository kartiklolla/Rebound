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
The instinct is to maximise recovery. The baselines show where that leads:
``aggressive_contact`` contacts 3.62 times per episode, drives revocation from
the ladder's 0.0460 to 0.1328, and destroys **₹1,941,482 per thousand** against
the ladder's ₹700,439 — the worst net on the board by a wide margin.

An earlier version of this paragraph claimed it "recovers more debits than the
naive ladder" and "wins the metric merchants usually watch". **Both are false.**
It recovers **0.3069 against the ladder's 0.4561** — it recovers *less*, because
it retries exactly once (``baselines.py``) and spends the rest of its budget on
contact. The ₹346,088 was a stale figure from a different metric and config. The
real lesson is narrower and still holds: contact is expensive, and a policy that
does not charge for it loses money.

So each candidate is priced against the counterfactual of stopping now, in
closed form:

    EV = P(recover) × (amount + h × LTV)
       − P(revoke)  × LTV × (1 − h)
       − cost(action)

where ``h`` is P(the customer churns on their own | this episode is never
recovered). Estimated at fit time from the *training* split, where it is
0.0741 ± 0.0021 — not the 0.0700 ± 0.0016 of the full log, which is what
``fit_for_serving`` never sees.

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
beats the action head on that question on matched rows (ROC 0.9282 ± 0.0018
against 0.8996 ± 0.0022, on 22,863 matched collecting rows at the ``max_iter``
this module actually fits with). Retry success against insufficient funds runs 0.65 near payday and 0.22
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
from rebound.economics import (
    LTV_HORIZON_CYCLES as _LTV_HORIZON,
)
from rebound.economics import attempt_cost_paise, revocation_cost_paise
from rebound.fqi import FittedQ
from rebound.model import (
    COLLECTING_ACTIONS,
    TARGET_ACTION_REVOKED,
    RecoveryModel,
    TwoHeadedModel,
)
from rebound.policy import Decision, Policy
from rebound.sim.world import EpisodeView
from rebound.taxonomy import (
    CUSTOMER_FACING_ACTIONS,
    Action,
    get_mode,
    legal_actions,
)

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
#: served.
#:
#: This was described here as costing "some accuracy, reported rather than
#: absorbed" before anything measured it — a claim written because it sounded
#: like the responsible thing to say. Measured, with both arms at identical
#: settings, dropping the four columns **does not cost accuracy at all**:
#:
#:     time split      PR-AUC 0.6002 -> 0.6324   (+0.0322)
#:     customer split  PR-AUC 0.6528 -> 0.6616   (+0.0088)
#:
#: The serving-only model is *better* on both splits. Consistent with the memorisation finding in ``eval/splits``:
#: cross-episode customer history is partly a handle for identifying the
#: customer rather than signal about the decision, and the time split is where
#: that handle pays off and then fails to generalise.
#:
#: (Absolute figures here are not the README's — this comparison ran at
#: ``max_iter=200`` on both arms, against a default of 300. The delta is the
#: measurement; the levels are not comparable across runs.)
UNSERVABLE_COLUMNS: tuple[str, ...] = (
    "cust_prior_failures",
    "cust_prior_recoveries",
    "cust_prior_recovery_rate",
    "cust_prior_contacts",
)


#: Actions that raise the fatigue level for every later contact in the episode.
#:
#: Matches what the world actually counts: ``world.apply`` gates the revocation
#: hazard on nudges and repair requests and increments ``contacts_made`` for
#: those alone. The mandated pre-debit notice reaches the customer but never
#: draws the hazard, so it creates no externality and is excluded.
FATIGUE_ACTIONS: frozenset[str] = frozenset(
    str(a)
    for a in CUSTOMER_FACING_ACTIONS
    if a is not Action.SEND_PRE_DEBIT_NOTIFICATION
)


def estimate_future_contacts(train: pd.DataFrame, max_k: int = 3) -> dict[int, float]:
    """Expected further contacts in an episode, given one is taken at ``k``.

    The multiplier on the fatigue externality. Taking a contact now raises
    ``prior_contacts`` for *every* subsequent contact in the episode, so the
    cost of incurring it is the per-contact increment times however many later
    contacts will pay it.

    Estimated, not bounded. The obvious alternative — "at most
    ``max_contacts - contacts_made`` remain" — is wrong against this data: at
    k=2 the budget bound says zero further contacts and the log shows 0.396,
    because the behavioural policy that generated the log has no compliance gate
    binding it.

    **Known bias, stated rather than buried.** This is measured under the
    behavioural exploration policy, which contacts more freely than the
    sequencer does. It therefore over-estimates the remaining contacts a
    *sequencer-run* episode would see, and so over-charges the externality. That
    is the conservative direction, and given the head is already known to
    under-slope on fatigue by about 20%, erring toward charging more for contact
    is the defensible way to be wrong.
    """
    ordered = train.sort_values(["episode_id", "decision_index"])
    is_contact = ordered["action"].isin(FATIGUE_ACTIONS)
    by_episode = is_contact.groupby(ordered["episode_id"], observed=True)
    remaining = by_episode.transform("sum") - by_episode.cumsum()

    contacts = pd.DataFrame(
        {
            "k": ordered.loc[is_contact, "prior_contacts"].clip(upper=max_k),
            "remaining": remaining[is_contact],
        }
    )
    if contacts.empty:
        return {}
    return {
        int(k): float(v)
        for k, v in contacts.groupby("k", observed=True)["remaining"].mean().items()
    }


def estimate_passive_revocation_rate(train: pd.DataFrame) -> float:
    """P(the customer churns on their own | this episode is never recovered).

    From observable outcomes only. The simulator has a
    ``passive_revocation_rate`` parameter and reading it would be taking the
    generator's answer key — the deployed system has a log, not a generator.

    An episode counts as passive churn when it revoked and **no row in it
    recorded a revocation at an action**. ``close_episode`` writes its
    settlement outcome outside the decision log, so a passive revocation leaves
    no row of its own; 73.4% of all revoked episodes have no action to blame.

    Only the passive share is returned. The action-caused share is already
    charged through ``Candidate.marginal_revocation``, and counting it in both
    places would inflate the value of every recovery.
    """
    per_episode = train.groupby("episode_id", observed=True).agg(
        recovered=("episode_recovered", "max"),
        revoked=("episode_revoked", "max"),
        revoked_at_an_action=("revoked", "max"),
    )
    unrecovered = per_episode[per_episode["recovered"] == 0]
    if unrecovered.empty:
        return 0.0
    passive = (unrecovered["revoked"] == 1) & (
        unrecovered["revoked_at_an_action"] == 0
    )
    return float(passive.mean())


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
    revocation = RecoveryModel(target=TARGET_ACTION_REVOKED, max_iter=max_iter).fit(
        servable, order_by=order_by, group_by=group_by
    )
    return ActionPricer(
        heads=heads,
        revocation=revocation,
        passive_revocation_rate=estimate_passive_revocation_rate(train),
        future_contacts=estimate_future_contacts(train),
    )


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

    passive_revocation_rate: float = 0.0
    """``h`` — P(the customer churns on their own | episode never recovered).

    Estimated from the training log. See
    ``ActionPricer.passive_revocation_rate``.
    """

    @property
    def recovery_value_paise(self) -> float:
        """``A + h·D`` — what one recovery is actually worth.

        Not the amount collected. A recovered episode is **resolved**, and
        ``World.close_episode`` returns early on a resolved episode without
        drawing for passive churn at all. Recovering a payment does not just
        collect it, it removes the customer from the churn pool.

        Measured on the full log: ``P(revoke | recovered) = 0.0000`` across 12,524
        recovered episodes against 0.0952 for unrecovered ones, of which the
        passive share is 0.0700 ± 0.0016. Only the passive share is credited —
        the other 26.6% revoked at an action and are charged separately below, so
        crediting the full figure would count them twice and inflate every
        recovery by 16.5%.

        The figure the running system uses is the *training-split* estimate,
        0.0741 ± 0.0021. These docstrings quote the full log because that is
        where the mechanism was established; ``fit_for_serving`` fits on train
        alone and never sees it.
        """
        return int(
            self.value_paise
            + self.passive_revocation_rate * self.revocation_cost_paise
        )

    @property
    def revocation_charge_paise(self) -> float:
        """``D·(1 − h)`` — what an action-caused revocation actually costs.

        Discounted by ``h`` because the counterfactual is not a customer who
        stays forever. If the action had not been taken and the episode had gone
        unrecovered anyway, that customer faced the passive draw regardless, so
        charging the full ``D`` bills the action for churn it merely brought
        forward.
        """
        return self.revocation_cost_paise * (1.0 - self.passive_revocation_rate)

    fatigue_delta: float = 0.0
    """``p_revoke(k+1) − p_revoke(k)`` for this action, from the head itself.

    Zero for anything that does not raise the fatigue level.
    """

    future_contacts: float = 0.0
    """Expected further contacts in this episode if this one is taken."""

    @property
    def fatigue_externality_paise(self) -> float:
        """What this contact costs the contacts that come after it.

        The defect a one-step expected value cannot see. ``prior_contacts`` is
        identical across every candidate at a decision point, so the head prices
        *today's* fatigue level correctly and the cost of *adding* a level is
        structurally invisible. Taking a contact now raises the revocation
        probability of every subsequent contact in the episode, and a myopic
        objective books none of that.

        It is not a small omission. Correcting the recovery credit alone — which
        is arithmetically right — moved contacts from 0.74 to 1.00 per episode
        and net value from −73,367 to −159,979, because a more accurate
        *myopic* objective simply buys more of the behaviour that makes
        ``aggressive_contact`` the worst policy on the board.

        Charged at ``D(1 − h)`` for the same reason the direct revocation term
        is: the counterfactual is not a customer who would otherwise stay
        forever.

        Both inputs are derived rather than chosen — the increment comes from
        the head's own prediction one fatigue level up, and the multiplier is
        measured from the log. No tuned constant.
        """
        return self.fatigue_delta * self.future_contacts * self.revocation_charge_paise

    @property
    def expected_value_paise(self) -> float:
        """Value of acting, against the counterfactual of stopping now.

        Derived in closed form rather than estimated, which is the correction
        that matters most here::

            stop:  -h·D                                 one passive draw
            act:   p_R·A - p_V·D - (1-p_R-p_V)·h·D      three exclusive branches
            Δ   =  p_R·(A + h·D) - p_V·D·(1-h)

        The previous version subtracted a *model-predicted* baseline for a STOP
        row and clamped the difference. That is a different marginalisation, and
        a wrong one. The ``h·D`` credit is not a baseline to subtract — it
        belongs multiplied by ``p_recover``, because avoiding the passive draw is
        something only a recovery achieves. Subtracting it unconditionally paid
        that credit to actions with ``p_recover = 0``.

        Two of the estimated baselines were also known constants. A STOP row
        terminates the episode, so ``P(recover | stop)`` is exactly 0 by
        construction (the model said 0.0006), and ``revoked`` is structurally 0
        for STOP, both retries and the pre-debit notice (the model said
        0.00082). The clamp guarding against negative marginals is inert under
        the per-action label — it fires on 0.000 of contact candidates.

        Sanity, all exact: ``p_R = p_V = 0`` → ``-cost`` (both branches face the
        same draw); ``p_R = 1`` → ``A + h·D``; ``p_V = 1`` → ``-D(1-h)``.
        """
        return (
            self.p_recover * self.recovery_value_paise
            - self.p_revoke * self.revocation_charge_paise
            - self.fatigue_externality_paise
            - self.cost_paise
        )

    def explain(self) -> str:
        return (
            f"{self.action}@{self.at:%m-%d %H:%M} "
            f"p_rec={self.p_recover:.3f} p_rev={self.p_revoke:.3f} "
            f"fatigue={self.fatigue_externality_paise / 100:,.0f} "
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
    future_contacts: dict[int, float] = field(default_factory=dict)
    """Expected further contacts by current fatigue level. See
    ``estimate_future_contacts``."""

    passive_revocation_rate: float = 0.0
    """P(passive churn | episode never recovered), estimated from the log.

    Fitted rather than hardcoded. A constant measured once at one config and
    pasted into the source is a number that goes stale in silence — and this one
    is multiplied by a 12-cycle horizon, so an error in it is an error in every
    expected value the sequencer computes.
    """

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
        # No STOP baseline rows. The stop counterfactual is -h*D in closed
        # form, so predicting it was estimating a known constant: P(recover |
        # stop) is exactly 0 because a stop terminates the episode, and
        # `revoked` is structurally 0 on a stop row.
        #
        # What the frame *does* carry is a shadow row for every contact
        # candidate, identical except that `prior_contacts` is one higher. The
        # difference between a contact's prediction and its shadow is what that
        # contact does to the fatigue level of every contact after it — the
        # externality a one-step objective cannot otherwise see.
        frame = self.pricer.frame(episode, pairs)
        fatigue_rows = [i for i, (a, _) in enumerate(pairs) if str(a) in FATIGUE_ACTIONS]
        if fatigue_rows:
            shadow = frame.iloc[fatigue_rows].copy()
            shadow["prior_contacts"] = shadow["prior_contacts"] + 1
            frame = pd.concat([frame, shadow], ignore_index=True)

        shared = self.pricer.encode(frame)
        if shared is None:
            p_recover = self.pricer.heads.predict_downstream(frame)
            p_revoke = self.pricer.revocation.predict_proba(frame)
        else:
            p_recover = self.pricer.heads.downstream.predict_proba_prepared(shared)
            p_revoke = self.pricer.revocation.predict_proba_prepared(shared)

        # Clamped at zero: the head's fatigue slope is a coarse step function,
        # so an individual increment can come back slightly negative from tree
        # quantisation. A contact making later contacts *safer* is not a finding
        # this data supports, and paying for it would be the mirror of the
        # revocation-credit bug.
        fatigue_delta = {
            row: max(0.0, float(p_revoke[len(pairs) + rank] - p_revoke[row]))
            for rank, row in enumerate(fatigue_rows)
        }

        destroyed = revocation_cost_paise(episode.cycle_amount_paise)
        contacts_now = min(episode.contacts_made, max(self.pricer.future_contacts, default=0))
        candidates = [
            Candidate(
                action=action,
                at=at,
                p_recover=float(p_recover[i]),
                p_revoke=float(p_revoke[i]),
                value_paise=episode.cycle_amount_paise,
                cost_paise=attempt_cost_paise(action, episode.rail),
                revocation_cost_paise=destroyed,
                passive_revocation_rate=self.pricer.passive_revocation_rate,
                fatigue_delta=fatigue_delta.get(i, 0.0),
                future_contacts=(
                    self.pricer.future_contacts.get(contacts_now, 0.0)
                    if i in fatigue_delta
                    else 0.0
                ),
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


# ==========================================================================
# The fitted-Q policy
# ==========================================================================


@dataclass
class QSequencer(Policy):
    """Acts on a fitted Q function instead of a hand-built expected value.

    Everything the expected value needed as separate machinery collapses into
    one number here.

    **Not shipped.** Across four full-scale seeds the hand-built expected value
    beat this on 4/4 (mean gap over the fixed ladder +164,807 against +100,876),
    and this went *negative* on one seed — losing to a three-line retry ladder.
    It is kept as a measured negative result with a diagnosis, not as a policy.

    **Stopping is not a rule.** ``Q(s, STOP)`` is an action value like any
    other, so the policy stops when stopping is worth more than acting. The
    threshold, the marginal baselines and the passive-churn credit all
    disappear.

    An earlier version of this docstring offered the fitted ``Q(s, STOP)`` of
    -992 against the closed form's -1,002 as corroboration "from a completely
    independent route". **That was wrong and the claim is withdrawn.** On every
    nonzero stop row the logged reward *is* exactly ``-1.0 x amount x
    LTV_HORIZON_CYCLES``, so ``Q(s, STOP)`` is definitionally the same closed
    form with the churn rate estimated rather than supplied. Two names for one
    computation, presented as two witnesses.

    **Timing is not priced reliably.** The state carries ``decision_hour``,
    ``days_since_failure`` and ``within_upi_window``, so in principle ``Q``
    prices *when* and *what* together. In practice the behavioural log has a
    hard floor at ``days_since_failure = 1.0`` — zero of 88,178 rows below it —
    while the candidate grid starts at zero delay, so **68% of this policy's
    decisions fall outside the training support**, in a region where the
    booster's leftmost bin makes the feature constant. That is why it needs a
    ``pricer``, and why the timing head it was meant to absorb is still doing
    the work.

    **The fatigue externality is not a term.** A contact's effect on later
    contacts is exactly what a continuation value is, and it is now carried
    there rather than bolted on with an estimated multiplier.

    The gate still has the last word. Q ranks; compliance disposes.
    """

    q: FittedQ
    pricer: ActionPricer | None = None
    """Supplies the timing head. Without it, ``Q`` picks the moment too.

    It should not. ``Q`` is fitted on logged transitions in which timing was
    never randomised, so ``days_since_failure`` is entangled with *how many
    attempts have already failed* — a decision seven days in is a decision that
    follows failures. Q therefore learns "later decisions are worse" (selection)
    rather than "waiting refills the account" (causation), and acts immediately
    on almost everything: measured, 2,014 of 2,435 decisions scheduled at zero
    delay, mean 1.8 hours against the behavioural log's own 79-hour median. That
    is the ``immediate_retry`` baseline rediscovered, and it scores like it.

    The timing head does not share the defect. It is trained on the immediate
    label over collecting actions — "did this retry, at this moment, collect" —
    which is a causal question about a single decision rather than an
    episode-level outcome. It beats the action head on exactly that question on
    matched rows (ROC 0.9282 vs 0.8996 at the fitted ``max_iter=200``).

    So the same discipline as the two heads applies here: each component is used
    where it was measured to be better. The timing head picks *when*; Q picks
    *what*, and prices the continuation that follows.
    """

    gate: ComplianceGate = field(default_factory=ComplianceGate)
    horizon_hours: tuple[int, ...] = _HORIZON_HOURS
    name: str = "rebound_q"
    description: str = (
        "Fitted Q-iteration over logged episodes; compliance-gated."
    )
    trail: list[dict[str, object]] = field(default_factory=list)

    def reset(self) -> None:
        self.trail.clear()
        self.gate.audit.clear()

    def decide(
        self, episode: EpisodeView, now: dt.datetime, deadline: dt.datetime
    ) -> Decision | None:
        pairs = [
            (a, t) for a, t in self._expand(episode, now, deadline) if t <= deadline
        ]
        pairs = self._choose_times(episode, pairs)
        # STOP is always on the table and always permitted, so it needs no gate
        # round-trip to be a candidate — but it does still get adjudicated when
        # chosen, so the compliance trail records the decision to give up.
        pairs.append((Action.STOP, now))

        frame = pd.DataFrame(
            [{**_observable(episode, at), "action": str(a)} for a, at in pairs]
        )
        values = self.q.predict(frame)
        best = int(np.argmax(values))
        action, at = pairs[best]

        considered = [
            f"{a}@{t:%m-%d %H:%M} q={v / 100:,.0f}"
            for (a, t), v in zip(pairs, values)
        ]

        def record(chosen: Action, when: dt.datetime, value: float) -> None:
            """Written only once the gate has ruled.

            The EV sequencer shipped the other way round for a while and its
            trail reported ``send_collect_link`` 620 times where 9 reached the
            world. Recording the winner before adjudication makes the log a
            record of what was *proposed*.

            ``value`` is passed rather than closed over. An earlier version
            captured ``values[best]``, so on the refusal path it wrote a row
            reading ``chosen="stop"`` carrying the Q of the action the gate had
            just refused — off by ₹1,053 on the reproduction, always in the
            direction of making stopping look better than it was.
            """
            self.trail.append(
                {
                    "episode_id": episode.episode_id,
                    "at": now,
                    "chosen": str(chosen),
                    "scheduled_for": when,
                    "q_paise": round(float(value), 2),
                    "considered": considered,
                }
            )

        # Defence in depth that currently never fires. Every candidate reaching
        # here was cleared by `_expand` under the same rules at the same moment,
        # and every rule in DEFAULT_RULES is a pure function of the Request — so
        # re-asking gets the same answer. Measured: 0 refusals in 414,735 pairs.
        #
        # Kept rather than deleted because the invariant it protects is "nothing
        # executes unadjudicated", and that has to survive a future rule that
        # reads mutable state (a spend counter, a daily contact budget) where
        # asking twice would legitimately differ. It is unreachable today, so
        # the branch below is untested in situ and the tests that exercise it
        # have to force a refusal.
        decision = self.gate.adjudicate(Request.from_view(episode, action, at))
        if not decision.allowed:
            # Stopping is the honest fallback: silently taking the runner-up
            # would mean the recorded reason no longer matches the action.
            self.gate.adjudicate(Request.from_view(episode, Action.STOP, now))
            record(Action.STOP, now, float(values[-1]))
            return Decision(
                action=Action.STOP, at=now, reason=f"gate refused: {decision.explain()}"
            )

        record(action, at, float(values[best]))
        return Decision(
            action=action,
            at=at,
            reason=f"Q={values[best] / 100:,.0f} over {len(pairs)} candidates",
        )

    def _choose_times(
        self, episode: EpisodeView, pairs: list[tuple[Action, dt.datetime]]
    ) -> list[tuple[Action, dt.datetime]]:
        """One moment per action, chosen by the timing head where it applies.

        Collapsing the grid before Q sees it is the point: Q ranks actions, not
        moments. Without a timing head every candidate time is offered to Q and
        it takes the earliest, for the reasons in ``pricer``.
        """
        if self.pricer is None:
            return pairs

        collecting = [
            (i, a, t) for i, (a, t) in enumerate(pairs) if str(a) in COLLECTING_ACTIONS
        ]
        if not collecting:
            return pairs

        frame = self.pricer.frame(episode, [(a, t) for _, a, t in collecting])
        immediate = self.pricer.heads.predict_immediate(frame)

        best: dict[Action, tuple[float, dt.datetime]] = {}
        for (_, action, at), score in zip(collecting, immediate):
            if action not in best or score > best[action][0]:
                best[action] = (float(score), at)

        keep = [(a, t) for a, t in pairs if str(a) not in COLLECTING_ACTIONS]
        keep.extend((action, at) for action, (_, at) in best.items())
        return keep

    # -- candidate construction, shared with the EV sequencer ---------------

    _candidate_times = Sequencer._candidate_times
    _permitted = Sequencer._permitted
    _expand = Sequencer._expand
