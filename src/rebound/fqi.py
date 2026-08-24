"""Fitted Q-iteration over the logged episodes.

Why this exists
---------------
The expected-value sequencer pairs a **one-step cost** with an **episode-level
benefit**. ``p_recover`` is the downstream head — ``P(episode ultimately
recovers | this action)`` — which already contains whatever happens after the
action. So in a three-decision episode, all three decisions are credited with
the same recovery: the nudge is credited for the retry that follows it, and then
the retry is credited again.

That is not a coefficient error and no coefficient fixes it. Correcting the
recovery credit to its true value of 1.84x the amount made the policy *worse*
(net -73,367 to -159,979, contacts 0.74 to 1.00 per episode), because it
amplified a term that was already counted several times over.

The closed-form derivation the EV now uses is correct arithmetic for a decision
that **ends the episode** — three exclusive outcomes, differenced against
stopping. Episodes do not end. There is no term in it for "and then I decide
again", and that missing term is the whole problem.

What this does instead
----------------------
Treats recovery as what it is: a finite-horizon sequential decision problem.

``Q(s, a)`` is the total remaining value of taking action ``a`` in state ``s``
and continuing well afterwards. It is fitted by backward induction over the
logged transitions::

    Q_0(s, a) = 0
    Q_k(s, a) = r(s, a) + max over legal a' of Q_{k-1}(s', a')

The immediate reward comes from the ledger, not from a label: rupees collected,
less what the action cost, less what it destroyed. Continuation stops being an
assumption baked into an episode-level label and becomes something estimated.

Three properties of this problem make it the right tool rather than an academic
detour:

**The horizon is short.** Episodes run two to three decisions, so the recursion
is shallow and value propagates fully in a handful of sweeps.

**The behavioural policy explored.** ``EXPLORATION_WEIGHTS`` puts positive
probability on every action including futile ones, which is exactly the coverage
``max over a'`` needs in order to ask about actions the log did not take here.
The dataset was built this way on purpose.

**Stopping stops being special.** ``Q(s, STOP)`` is just another action value,
so the policy stops when stopping is worth more than acting. The hand-written
stopping rule, the marginal baselines and the fatigue externality all disappear
— the last of these because a contact's effect on later contacts is now carried
by the continuation term rather than bolted on.

What it does not fix
--------------------
The Q function is fitted on states the *behavioural* policy visited. A learned
policy visits a different distribution, and nothing here corrects for that.
Logged propensities exist in the dataset for that correction; they are not used
yet, and that is a stated limitation rather than an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from rebound.economics import LTV_HORIZON_CYCLES
from rebound.model import FeatureSpec
from rebound.taxonomy import Action, legal_actions

#: Columns the training log has that a deployed policy cannot supply.
#: Kept identical to ``sequencer.UNSERVABLE_COLUMNS`` — see that module for why
#: training on features the serving path cannot produce is silent skew.
UNSERVABLE_COLUMNS: tuple[str, ...] = (
    "cust_prior_failures",
    "cust_prior_recoveries",
    "cust_prior_recovery_rate",
    "cust_prior_contacts",
)


@dataclass(frozen=True, slots=True)
class Transitions:
    """Logged ``(s, a, r, s')`` tuples, ready for backward induction."""

    frame: pd.DataFrame
    """One row per decision: the state features, the action taken, and the
    immediate reward. Rows are in episode order."""

    reward: np.ndarray
    """Immediate reward in paise. Recovered, less cost, less destruction."""

    next_row: np.ndarray
    """Index into ``frame`` of the following decision, or -1 when terminal."""

    terminal: np.ndarray
    """True where the log has no further decision for this episode."""

    truncated: np.ndarray
    """True where the episode was *cut off* rather than ended.

    A last row that neither collected, nor lost the mandate, nor chose to stop
    was cut off rather than concluded. Measured: 12,973 of 24,514 last rows
    (52.92% ± 0.32%). They still carry the passive-churn charge, because
    ``close_episode`` genuinely ran — but treating them as evidence that acting
    ends badly conflates "the run was cut short" with "the episode was lost".

    **Two different cut-offs, and an earlier version of this docstring claimed
    they were one.** It said these were "overwhelmingly" the ``max_ladder_steps``
    ceiling. Decomposed:

    - 9,284 (71.56% ± 0.40%) at ``decision_index`` 4 — the step budget.
    - 3,689 (28.44% ± 0.40%) earlier, at mean ``days_since_failure`` 22.4 and a
      maximum of exactly 28.00 — the **recovery window closing**, not the budget.

    The second group is arguably a legitimate ending: the merchant ran out of
    month, not out of patience. Only the first is an artifact.

    Diagnostic only — nothing in the decision path reads this. It is here
    because the induction silently treats both as terminal, and because the
    training log allows five decisions where the evaluation harness allows eight
    (``DEFAULT_MAX_STEPS``), so 3.1% of rollout decisions occur at
    ``prior_actions`` values with **zero** training rows. Aligning the generator
    and the harness is the fix; a flag is not."""

    failure_code: np.ndarray
    """Per row, so the continuation max can be restricted to legal actions."""

    def __len__(self) -> int:
        return len(self.reward)


def build_transitions(log: pd.DataFrame) -> Transitions:
    """Turn a decision log into ``(s, a, r, s')`` tuples.

    The reward is built from the ledger and reconciles exactly against the
    episode totals — ``sum(reward) == episode_net_paise`` for every episode,
    which ``test_rewards_reconcile_with_the_episode_ledger`` asserts. If it did
    not, the Q function would be learning a quantity the evaluation harness does
    not score, and every number downstream would be measuring something else.

    Passive churn needs care. ``close_episode`` settles an abandoned episode
    outside the decision log, so it never appears as a row: 1,698 of 2,312
    revocations have no action to blame. That destruction is attached to the
    **last** decision of the episode, which is where the choice that led to it
    was made.
    """
    required = {
        "episode_id",
        "decision_index",
        "action",
        "failure_code",
        "revoked",
        "succeeded",
        "amount_paise",
        "recovered_paise",
        "cost_paise",
        "episode_destroyed_paise",
    }
    missing = required - set(log.columns)
    if missing:
        raise ValueError(f"log is missing {sorted(missing)}")

    ordered = log.sort_values(["episode_id", "decision_index"]).reset_index(drop=True)

    destroyed_per_revocation = ordered["amount_paise"] * LTV_HORIZON_CYCLES
    row_destroyed = ordered["revoked"].astype(int) * destroyed_per_revocation

    episode = ordered["episode_id"]
    is_last = episode != episode.shift(-1)

    # What close_episode destroyed, which no row records. Recovered exactly:
    # episode_destroyed_paise minus the destruction the rows already account for.
    by_episode = row_destroyed.groupby(episode, observed=True).transform("sum")
    passive = ordered["episode_destroyed_paise"] - by_episode
    passive_on_last = np.where(is_last, passive, 0)

    reward = (
        ordered["recovered_paise"].to_numpy(dtype=float)
        - ordered["cost_paise"].to_numpy(dtype=float)
        - row_destroyed.to_numpy(dtype=float)
        - np.asarray(passive_on_last, dtype=float)
    )

    next_row = np.arange(1, len(ordered) + 1, dtype=int)
    next_row[is_last.to_numpy()] = -1

    # An episode's last row is not always an ending. The generator stops at
    # `max_ladder_steps`, so a run that neither collected, nor lost the mandate,
    # nor chose to stop was simply cut off — 53% of last rows, almost all at
    # decision_index 4. Backward induction is otherwise told the world ends
    # there and charges 12 cycles of LTV for it.
    ended = (
        ordered["succeeded"].astype(bool)
        | ordered["revoked"].astype(bool)
        | (ordered["action"].astype(str) == str(Action.STOP))
    )
    truncated = is_last.to_numpy() & ~ended.to_numpy()

    return Transitions(
        frame=ordered,
        reward=reward,
        next_row=next_row,
        terminal=is_last.to_numpy(),
        truncated=truncated,
        failure_code=ordered["failure_code"].to_numpy(),
    )


@dataclass
class FittedQ:
    """``Q(s, a)`` — total remaining value of an action, fitted by induction.

    Not a probability and not calibrated, deliberately. It is an amount of money
    in paise, which is what a decision between actions actually needs. The
    calibration machinery elsewhere in this project exists because a probability
    gets multiplied by a rupee amount; here the model predicts the rupees.
    """

    sweeps: int = 6
    """Backward-induction passes.

    **This does not converge, and ``deltas_`` shows it.** At ``sweeps=8`` the
    per-sweep target movement runs 27552, 24529, 21660, 21761, 16687, 12987,
    15219 — rising at two points. Extended to 16 it oscillates between roughly
    11,000 and 19,000 paise with no downward trend, against a between-action
    range in mean Q of about 76,000. So on the order of 20% of the signal
    separating one action from another is refit noise, and this parameter is an
    arbitrary stop on an oscillating sequence rather than a converged one.

    An earlier version of this docstring called it "comfortable headroom" and
    claimed ``deltas_`` recorded convergence. It records the opposite, and
    nothing reads it."""

    discount: float = 1.0
    """Undiscounted. The horizon is a handful of decisions inside a 28-day
    recovery window, so a discount rate would be a second assumption doing
    almost no work."""

    max_iter: int = 200
    learning_rate: float = 0.06
    min_samples_leaf: int = 40
    seed: int = 20260821

    ensemble: int = 2
    """How many Q functions to fit on disjoint slices of the log.

    Two is enough, and one is not. A single Q with a plain ``argmax`` suffers
    the winner's curse: over ~20 candidates per decision, the maximum selects
    whichever candidate carries the largest *positive model error* rather than
    the largest true value. Measured, that policy scored -330,652 against the
    hand-built expected value's -49,739, recovering less (0.4094 vs 0.5557)
    while contacting more.

    With two, each one's backward-induction target is evaluated by the *other*
    — the standard double-Q construction, which breaks the correlation between
    selecting an action and believing its inflated value.
    """

    pessimistic: bool = True
    """Take the minimum across the ensemble at decision time rather than the mean.

    The log is fixed and cannot be queried about actions it never took in a
    given state. Where the ensemble disagrees, that disagreement *is* the
    evidence of thin support, and the smaller estimate is the one not resting
    on extrapolation. Standard practice offline, and the opposite of what would
    be right if we could still explore.
    """

    spec_: FeatureSpec | None = None
    models_: list[HistGradientBoostingRegressor] = field(default_factory=list)
    deltas_: list[float] = field(default_factory=list)
    action_levels_: tuple[str, ...] = ()

    # -- fitting -----------------------------------------------------------

    def fit(self, transitions: Transitions) -> FittedQ:
        frame = transitions.frame
        servable = frame.drop(columns=list(UNSERVABLE_COLUMNS), errors="ignore")
        self.spec_ = FeatureSpec.fit(servable)
        self.action_levels_ = tuple(self.spec_.categories["action"])

        states = self.spec_.transform(servable)
        reward = transitions.reward
        live = ~transitions.terminal
        successor = transitions.next_row[live]
        allowed = self._legal_mask(transitions)

        # Episode-atomic partition. Splitting rows would put a decision in one
        # half and its own successor in the other, so each model would be
        # evaluated on continuations it had trained against.
        parts = self._partition(frame)

        targets = [reward.copy() for _ in range(self.ensemble)]
        self.models_ = [None] * self.ensemble  # type: ignore[list-item]
        self.deltas_.clear()

        for sweep in range(self.sweeps):
            for i in range(self.ensemble):
                model = HistGradientBoostingRegressor(
                    max_iter=self.max_iter,
                    learning_rate=self.learning_rate,
                    min_samples_leaf=self.min_samples_leaf,
                    categorical_features="from_dtype",
                    random_state=self.seed + 97 * i + sweep,
                )
                rows = parts[i]
                model.fit(states.iloc[rows], targets[i][rows])
                self.models_[i] = model

            if sweep == self.sweeps - 1:
                break

            moved = []
            for i in range(self.ensemble):
                # Double-Q: model i selects the continuing action, the *other*
                # model prices it. Selecting and valuing with the same
                # estimator is what turns noise into optimism.
                other = (i + 1) % self.ensemble
                continuation = self._best_continuation(
                    servable, successor, allowed, chooser=i, valuer=other
                )
                updated = reward.copy()
                updated[live] += self.discount * continuation
                moved.append(float(np.abs(updated - targets[i]).mean()))
                targets[i] = updated
            self.deltas_.append(float(np.mean(moved)))

        return self

    def _partition(self, frame: pd.DataFrame) -> list[np.ndarray]:
        """Row indices per ensemble member, split on whole episodes."""
        episodes = pd.Index(frame["episode_id"].unique())
        assignment = pd.Series(
            np.random.default_rng(self.seed).integers(0, self.ensemble, len(episodes)),
            index=episodes,
        )
        which = frame["episode_id"].map(assignment).to_numpy()
        return [np.flatnonzero(which == i) for i in range(self.ensemble)]

    def _legal_mask(self, transitions: Transitions) -> dict[str, np.ndarray]:
        """Which actions may be considered at each successor state.

        Taking ``max`` over every action would let the value function assume a
        continuation the taxonomy forbids — a nudge to a closed account — and
        that optimism propagates backwards into every earlier decision.
        """
        codes = transitions.failure_code[transitions.next_row[~transitions.terminal]]
        unique = {code: legal_actions(code) for code in set(codes.tolist())}
        return {
            action: np.array(
                [Action(action) in unique[code] for code in codes], dtype=bool
            )
            for action in self.action_levels_
        }

    def _best_continuation(
        self,
        servable: pd.DataFrame,
        successor: np.ndarray,
        allowed: dict[str, np.ndarray],
        chooser: int,
        valuer: int,
    ) -> np.ndarray:
        """Value of continuing well: ``Q_valuer(s', argmax_a Q_chooser(s', a))``."""
        assert self.spec_ is not None
        successor_states = servable.iloc[successor]

        best_score = np.full(len(successor), -np.inf)
        best_value = np.zeros(len(successor))
        for action in self.action_levels_:
            mask = allowed[action]
            if not mask.any():
                continue
            probe = self.spec_.transform(successor_states.assign(action=action))
            score = self.models_[chooser].predict(probe)
            value = self.models_[valuer].predict(probe)
            take = mask & (score > best_score)
            best_score = np.where(take, score, best_score)
            best_value = np.where(take, value, best_value)

        # A state with no legal action at all can still be stopped.
        return np.where(np.isfinite(best_score), best_value, 0.0)

    # -- prediction --------------------------------------------------------

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """``Q(s, a)`` in paise, for rows already carrying an ``action`` column."""
        if self.spec_ is None or not self.models_:
            raise RuntimeError("model is not fitted")
        servable = frame.drop(columns=list(UNSERVABLE_COLUMNS), errors="ignore")
        prepared = self.spec_.transform(servable)
        stacked = np.vstack([m.predict(prepared) for m in self.models_])
        return stacked.min(axis=0) if self.pessimistic else stacked.mean(axis=0)


def fit_fitted_q(train: pd.DataFrame, **kwargs) -> FittedQ:
    """Build transitions from a training log and fit ``Q`` on them."""
    return FittedQ(**kwargs).fit(build_transitions(train))
