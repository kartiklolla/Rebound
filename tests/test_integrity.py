"""Regression tests for every finding from the adversarial review.

Each test here reproduces an attack that previously worked and asserts it now
fails. They are kept together rather than scattered into the module test files
because they share a purpose: this file is the standing evidence that the
harness cannot be fooled by the component it is measuring.

Full write-up in ``docs/SECURITY_REVIEW.md``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from rebound.economics import Ledger, revocation_cost_paise
from rebound.eval.harness import (
    IntegrityError,
    PolicyFailed,
    _Observed,
    build_eval_batch,
    evaluate_all,
    evaluate_policy,
)
from rebound.eval.metrics import (
    PolicyReport,
    classification_report,
    policy_comparison,
    reliability_table,
)
from rebound.eval.splits import (
    LeakageError,
    Split,
    SplitKind,
    assert_split_is_clean,
    time_split,
)
from rebound.policy import Decision, Policy
from rebound.sim.dataset import (
    FORBIDDEN_COLUMNS,
    OUTCOME_COLUMNS,
    GenerationConfig,
    feature_columns,
    generate_log,
)
from rebound.sim.world import ActionOutcome, Mandate, World, with_amount
from rebound.taxonomy import Action, Rail


@pytest.fixture(scope="module")
def setup():
    world = World(seed=555)
    customers = world.sample_customers(400)
    mandates = world.sample_mandates(
        customers, dt.date(2024, 1, 1), dt.date(2025, 6, 30)
    )
    world.calibrate(
        customers,
        mandates,
        [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(10)],
    )
    batch = build_eval_batch(
        world, customers, mandates, dt.date(2026, 1, 1), dt.date(2026, 1, 31)
    )
    return world, batch


@pytest.fixture(scope="module")
def log():
    return generate_log(
        GenerationConfig(
            n_customers=300,
            start=dt.date(2025, 1, 1),
            end=dt.date(2025, 12, 31),
            seed=606,
        )
    )


# ==========================================================================
# F1/F4/F5/F6/F7 — the policy cannot reach the live episode
# ==========================================================================


class _Snooper(Policy):
    """Captures whatever the harness hands it, so the tests can inspect it."""

    name = "snooper"

    def __init__(self) -> None:
        self.seen = None

    def decide(self, episode, now, deadline):
        self.seen = episode
        return None


def test_policy_never_receives_the_live_episode(setup):
    """The structural fix for the whole tampering class.

    A policy given the live ``Episode`` could set ``resolved``, rebind the
    ledger, rewrite the failure code, or reset the contact counter — all
    demonstrated against an earlier version. It now gets a frozen projection.
    """
    world, batch = setup
    snooper = _Snooper()
    evaluate_policy(world, snooper, batch[:5])
    assert snooper.seen is not None
    assert type(snooper.seen).__name__ == "EpisodeView"


def test_the_episode_view_is_immutable(setup):
    world, batch = setup
    snooper = _Snooper()
    evaluate_policy(world, snooper, batch[:5])
    for field in ("failure_code", "contacts_made", "attempts", "spent_paise"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(snooper.seen, field, 0)


def test_the_view_exposes_no_simulator_latents(setup):
    """The leakage half of the same fix.

    ``Episode.customer`` is a full ``Customer`` carrying ``salary_day``,
    ``balance_health``, ``engagement``, ``churn_intent`` and
    ``preferred_channel`` — the simulator's answer key. A learned policy handed
    that object could read churn intent directly and post numbers that mean
    nothing.
    """
    world, batch = setup
    snooper = _Snooper()
    evaluate_policy(world, snooper, batch[:5])

    exposed = {f.name for f in dataclasses.fields(snooper.seen)}
    leaked = exposed & FORBIDDEN_COLUMNS
    assert not leaked, f"the policy-facing view exposes latents: {leaked}"
    assert not hasattr(snooper.seen, "customer"), (
        "the view must not carry the Customer object; it holds every latent"
    )


def test_the_view_has_no_path_back_to_mutable_state(setup):
    """History is handed over as a tuple of frozen outcomes. If it were the
    live list, a policy could append forged outcomes to it."""
    world, batch = setup
    snooper = _Snooper()
    evaluate_policy(world, snooper, batch[:5])
    assert isinstance(snooper.seen.history, tuple)


# ==========================================================================
# Reconciliation — the report never trusts the episode
# ==========================================================================


def _outcome(**kw) -> ActionOutcome:
    base = dict(
        action=Action.RETRY_SAME_RAIL,
        at=dt.datetime(2026, 1, 2, 9, 0),
        succeeded=False,
        recovered_paise=0,
        cost_paise=50,
        revoked=False,
        destroyed_paise=0,
        detail="test",
    )
    base.update(kw)
    return ActionOutcome(**base)


def _episode(world: World, batch, code: str | None = None):
    spec = batch[0]
    return world.open_episode(
        "EP_T", spec.mandate, spec.customer, code or spec.failure_code,
        spec.failed_at, spec.cycles_elapsed,
    )


def test_reconciliation_catches_a_forged_ledger(setup):
    """The original critical finding, in miniature.

    The report used to be read off the object the untrusted policy held. It is
    now rebuilt from the harness's own observations, and any disagreement
    raises rather than being reported.
    """
    world, batch = setup
    episode = _episode(world, batch)
    observed = _Observed(failure_code=episode.failure_code)
    episode.ledger = Ledger(recovered_paise=10_000_000, spent_paise=0)
    with pytest.raises(IntegrityError, match="does not reconcile"):
        observed.verify(episode)


def test_reconciliation_catches_a_forged_resolved_flag(setup):
    world, batch = setup
    episode = _episode(world, batch)
    observed = _Observed(failure_code=episode.failure_code)
    episode.resolved = True
    with pytest.raises(IntegrityError, match="no observed outcome recovered"):
        observed.verify(episode)


def test_reconciliation_catches_a_rewritten_failure_code(setup):
    """Rewriting the failure code was how a terminal, revoked mandate was made
    to collect money."""
    world, batch = setup
    episode = _episode(world, batch)
    observed = _Observed(failure_code=episode.failure_code)
    episode.failure_code = "UPI_INSUFFICIENT_FUNDS"
    if observed.failure_code != "UPI_INSUFFICIENT_FUNDS":
        with pytest.raises(IntegrityError, match="failure code changed"):
            observed.verify(episode)


def test_reconciliation_catches_a_reset_contact_counter(setup):
    """Resetting ``contacts_made`` suppressed contact fatigue and revocation
    risk, cutting a policy's own churn cost roughly fourfold while under-
    reporting its intrusiveness."""
    world, batch = setup
    episode = _episode(world, batch)
    observed = _Observed(
        failure_code=episode.failure_code,
        outcomes=[_outcome(action=Action.NUDGE_SMS, cost_paise=20)],
    )
    episode.history.append(observed.outcomes[0])
    episode.ledger = episode.ledger.plus_cost(20)
    episode.contacts_made = 0
    with pytest.raises(IntegrityError, match="contacts_made"):
        observed.verify(episode)


def test_reconciliation_catches_a_policy_driving_the_world_itself(setup):
    """A policy that called ``world.apply`` behind the harness's back produced
    real recoveries with an empty audit trail — the most plausible-looking
    forgery of the lot, since the number was not absurd."""
    world, batch = setup
    episode = _episode(world, batch)
    observed = _Observed(failure_code=episode.failure_code)
    episode.history.append(_outcome())
    with pytest.raises(IntegrityError, match="outside the rollout"):
        observed.verify(episode)


def test_the_report_reconciles_with_the_audit_trail(setup):
    """The end-to-end version of the same guarantee: the money in the report is
    the money in the evidence."""
    from rebound.eval.baselines import DispositionAwareRules

    world, batch = setup
    result = evaluate_policy(world, DispositionAwareRules(), batch, seed=11)
    audit = result.audit_frame()
    assert result.report.recovered_paise == int(audit["recovered_paise"].sum())
    assert result.report.spent_paise == int(audit["cost_paise"].sum())


# ==========================================================================
# F2 — negative money
# ==========================================================================


def test_a_negative_mandate_amount_is_rejected_at_construction(setup):
    """One negative amount in a merchant export previously flipped every
    policy's net positive and reported the do-nothing floor as having earned
    money."""
    world, batch = setup
    with pytest.raises(ValueError, match="must be positive"):
        with_amount(batch[0].mandate, -10**9)


def test_mandate_validates_its_own_fields():
    kwargs = dict(
        mandate_id="M1", customer_id="C1", rail=Rail.UPI_AUTOPAY,
        cycle_amount_paise=1000, ceiling_paise=2000, billing_day=5,
        registered_on=dt.date(2025, 1, 1), valid_until=dt.date(2026, 1, 1),
    )
    with pytest.raises(ValueError, match="must be positive"):
        Mandate(**{**kwargs, "cycle_amount_paise": 0})
    with pytest.raises(ValueError, match="cannot be negative"):
        Mandate(**{**kwargs, "ceiling_paise": -1})
    with pytest.raises(ValueError, match="billing_day"):
        Mandate(**{**kwargs, "billing_day": 31})


def test_revocation_cost_rejects_non_positive_amounts():
    with pytest.raises(ValueError, match="must be positive"):
        revocation_cost_paise(-5_000_000)


@pytest.mark.parametrize(
    "method", ["plus_cost", "plus_recovery", "plus_destruction"]
)
def test_the_ledger_rejects_negative_entries(method: str):
    """A negative cost is a rebate, a negative destruction is value created
    from nothing. Both silently improve whatever total they land in."""
    with pytest.raises(ValueError, match="non-negative"):
        getattr(Ledger(), method)(-999_999)


# ==========================================================================
# F3 — split verification
# ==========================================================================


def test_cloned_rows_no_longer_pass_the_leakage_check(log):
    """The twin-row attack.

    Clone the test rows, prefix the ids, shift the dates back past the cut.
    Episode and customer ids are disjoint, so the identity checks passed — and
    an exact feature join recovered the label at accuracy 1.000.
    """
    base = time_split(log)
    twins = base.test.copy()
    twins["episode_id"] = "TWIN_" + twins["episode_id"].astype(str)
    twins["customer_id"] = "TWIN_" + twins["customer_id"].astype(str)
    for column in ("failed_at", "decided_at"):
        twins[column] = pd.to_datetime(twins[column]) - pd.Timedelta(days=400)

    poisoned = Split(
        kind=SplitKind.TIME,
        train=pd.concat([base.train, twins], ignore_index=True),
        test=base.test,
        question="cloned rows with rewritten identifiers",
    )
    with pytest.raises(LeakageError, match="feature twin"):
        assert_split_is_clean(poisoned)


def test_honest_splits_still_pass_the_twin_check(log):
    """The twin check must not fire on legitimate splits, or it is just an
    obstacle people learn to skip."""
    from rebound.eval.splits import all_splits

    for split in all_splits(log).values():
        assert_split_is_clean(split)


def test_split_kind_is_typed_so_checks_cannot_be_skipped():
    """The verifier used to decide which checks to run by comparing a
    free-text ``name`` against string literals, so a split named anything
    unrecognised silently skipped both the temporal and customer checks."""
    assert set(SplitKind) == {SplitKind.TIME, SplitKind.CUSTOMER}
    with pytest.raises(ValueError):
        SplitKind("whatever")


def test_a_single_class_test_side_is_rejected(log):
    """Previously passed, then produced NaN for every discrimination metric
    while the split still reported as verified."""
    base = time_split(log)
    positives = base.test[base.test["episode_recovered"]]
    bad = Split(
        kind=SplitKind.TIME,
        train=base.train,
        test=positives,
        question="all-positive test side",
    )
    with pytest.raises(LeakageError, match="single outcome class"):
        assert_split_is_clean(bad)


def test_a_missing_label_column_is_rejected(log):
    base = time_split(log)
    bad = Split(
        kind=SplitKind.TIME,
        train=base.train,
        test=base.test.drop(columns=["episode_recovered"]),
        question="label dropped",
    )
    with pytest.raises(LeakageError, match="no 'episode_recovered'"):
        assert_split_is_clean(bad)


# ==========================================================================
# F11 — the leak that was an accident waiting to happen
# ==========================================================================


def test_feature_columns_excludes_every_outcome_column(log):
    """The most dangerous finding, and it was not an attack.

    ``FORBIDDEN_COLUMNS`` guards the simulator's latents, so the natural way to
    build a feature matrix leaves ``episode_net_paise`` in — and it predicts
    the label at accuracy 1.000, because it *is* the label restated.
    """
    features = set(feature_columns(log))
    assert not features & OUTCOME_COLUMNS, (
        f"outcome columns are selectable as features: "
        f"{sorted(features & OUTCOME_COLUMNS)}"
    )
    assert not features & FORBIDDEN_COLUMNS
    assert "episode_net_paise" not in features
    assert len(features) > 15, "the allowlist stripped away everything useful"


def test_no_selectable_feature_trivially_determines_the_label(log):
    """A leakage probe rather than a naming check.

    For each candidate feature, find the best single threshold against the
    label. Anything at or near perfect accuracy is the label in disguise, and
    the check does not depend on anyone having remembered to name it.
    """
    labels = log["episode_recovered"].to_numpy().astype(int)
    base_rate = max(labels.mean(), 1 - labels.mean())

    offenders = []
    for column in feature_columns(log):
        values = log[column]
        if not pd.api.types.is_numeric_dtype(values):
            continue
        numeric = values.to_numpy().astype(float)
        if not np.isfinite(numeric).all() or np.ptp(numeric) == 0:
            continue
        cuts = np.quantile(numeric, np.linspace(0.05, 0.95, 19))
        best = max(
            max(
                ((numeric > cut) == labels).mean(),
                ((numeric <= cut) == labels).mean(),
            )
            for cut in np.unique(cuts)
        )
        if best > 0.97:
            offenders.append((column, round(float(best), 4)))

    assert not offenders, (
        f"these selectable features nearly determine the label: {offenders}. "
        f"base rate is {base_rate:.4f}"
    )


# ==========================================================================
# F8/F9 — metric honesty
# ==========================================================================


def test_capacity_is_carried_on_the_report_not_baked_into_a_name():
    """A field named ``precision_at_10pct`` holding precision at 50% capacity
    is the purest available own goal for a project selling honest metrics."""
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, 2000)
    outcomes = (rng.uniform(size=2000) < probs).astype(int)

    at_10 = classification_report(outcomes, probs, capacity=0.10)
    at_50 = classification_report(outcomes, probs, capacity=0.50)

    assert at_10.capacity == 0.10
    assert at_50.capacity == 0.50
    assert at_10.capacity_label == "10%"
    assert at_50.capacity_label == "50%"
    assert at_10.precision_at_capacity != at_50.precision_at_capacity


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5, 2.0])
def test_impossible_capacities_are_rejected(bad: float):
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="capacity"):
        classification_report(y, np.array([0.1, 0.9, 0.2, 0.8]), capacity=bad)


def test_reliability_table_rejects_nan_instead_of_binning_it_high():
    """NaN predictions used to land in the top calibration bin — the very bin
    the docstring says the policy spends most of its money in — and a fabricated
    observed rate was reported for it. The two scoring functions disagreed
    about the same input."""
    y = np.array([0, 1, 0, 1, 1])
    probs = np.array([0.1, 0.9, np.nan, 0.8, 0.7])
    with pytest.raises(ValueError, match="non-finite"):
        reliability_table(y, probs)
    with pytest.raises(ValueError, match="non-finite"):
        classification_report(y, probs)


@pytest.mark.parametrize("bins", [0, -1])
def test_degenerate_bin_counts_are_rejected(bins: int):
    y = np.array([0, 1, 0, 1])
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(ValueError, match="bins"):
        reliability_table(y, probs, bins=bins)


def test_non_binary_labels_are_rejected():
    with pytest.raises(ValueError, match="only 0 and 1"):
        classification_report(np.array([0, 1, 2]), np.array([0.1, 0.5, 0.9]))


def test_duplicate_policy_names_are_rejected():
    """``value_preserved`` silently took the first match as the floor and
    printed the second as an ordinary row."""
    def report(name: str) -> PolicyReport:
        return PolicyReport(
            policy=name, episodes=100, recovery_rate=0.3, revocation_rate=0.05,
            recovered_paise=1000, spent_paise=100, destroyed_paise=200,
            net_paise=700, attempts_per_episode=2.0, contacts_per_episode=1.0,
        )

    with pytest.raises(ValueError, match="duplicate policy names"):
        policy_comparison([report("no_recovery"), report("no_recovery")])


def test_empty_comparison_is_rejected():
    with pytest.raises(ValueError, match="zero reports"):
        policy_comparison([])


# ==========================================================================
# F12 — a broken policy must not take the run down with it
# ==========================================================================


class _Crashing(Policy):
    name = "crashing"

    def decide(self, episode, now, deadline):
        raise RuntimeError("policy under development crashed")


def test_a_crashing_policy_is_isolated(setup):
    """The policy is the component under active development. Losing a whole
    comparison run to one typo is a real cost when the run takes half an hour."""
    from rebound.eval.baselines import FixedLadder, NoRecovery

    world, batch = setup
    results = evaluate_all(
        world, [NoRecovery(), _Crashing(), FixedLadder()], batch[:60]
    )
    assert set(results) == {"no_recovery", "crashing", "fixed_ladder"}
    assert results["crashing"].failed
    assert "policy under development crashed" in results["crashing"].error
    assert not results["fixed_ladder"].failed
    assert results["fixed_ladder"].report.episodes == 60


def test_fail_fast_still_available(setup):
    world, batch = setup
    with pytest.raises(RuntimeError, match="crashed"):
        evaluate_all(world, [_Crashing()], batch[:20], fail_fast=True)


def test_integrity_errors_are_never_isolated(setup):
    """An integrity failure means the numbers cannot be trusted. Swallowing it
    would produce a comparison table containing a figure nobody can stand
    behind, which is worse than no table."""
    import inspect

    from rebound.eval import harness

    source = inspect.getsource(harness.evaluate_all)
    assert "except IntegrityError:" in source
    assert "raise" in source


# ==========================================================================
# F13 — malformed decisions
# ==========================================================================


def _policy_returning(decision):
    class Returner(Policy):
        name = "returner"

        def decide(self, episode, now, deadline):
            return decision

    return Returner()


def test_a_non_action_is_rejected_with_a_useful_message(setup):
    world, batch = setup
    bad = Decision(action="DELETE_CUSTOMER", at=dt.datetime(2026, 1, 5), reason="x")
    with pytest.raises(PolicyFailed, match="taxonomy Action"):
        evaluate_policy(world, _policy_returning(bad), batch[:5])


def test_a_non_datetime_is_rejected(setup):
    world, batch = setup
    bad = Decision(action=Action.RETRY_SAME_RAIL, at="2026-01-05", reason="x")
    with pytest.raises(PolicyFailed, match="must be a datetime"):
        evaluate_policy(world, _policy_returning(bad), batch[:5])


def test_a_timezone_aware_datetime_is_rejected(setup):
    """Previously surfaced as a bare TypeError from an unrelated comparison
    deep inside the world."""
    world, batch = setup
    aware = dt.datetime(2026, 1, 5, tzinfo=dt.timezone.utc)
    bad = Decision(action=Action.RETRY_SAME_RAIL, at=aware, reason="x")
    with pytest.raises(PolicyFailed, match="timezone-naive"):
        evaluate_policy(world, _policy_returning(bad), batch[:5])


def test_a_slow_policy_hits_the_timeout(setup):
    """An unbounded ``decide`` used to hang the harness with no way out."""
    import time as _time

    world, batch = setup

    class Slow(Policy):
        name = "slow"

        def decide(self, episode, now, deadline):
            _time.sleep(0.02)
            return None

    with pytest.raises(PolicyFailed, match="exceeded"):
        evaluate_policy(world, Slow(), batch[:200], timeout_seconds=0.05)
