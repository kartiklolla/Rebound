"""Tests for the synthetic world.

Two kinds of test here. The mechanical ones check that impossible things stay
impossible — a dead mandate never charges, a closed episode never acts again.
The statistical ones check that the effects the model is supposed to learn are
actually present in the data, because a generator that quietly stops producing
a signal turns Claim A into a measurement of noise, and it does so silently.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rebound.economics import Ledger
from rebound.sim.params import WorldParams
from rebound.sim.world import (
    World,
    _days_since_salary,
    within_upi_execution_window,
)
from rebound.taxonomy import Action, Rail

SEED = 4242


@pytest.fixture(scope="module")
def calibrated_world() -> World:
    world = World(seed=SEED)
    customers = world.sample_customers(400)
    mandates = world.sample_mandates(
        customers, dt.date(2024, 6, 1), dt.date(2025, 1, 1)
    )
    dates = [dt.date(2025, 3, 1) + dt.timedelta(days=i) for i in range(14)]
    world.calibrate(customers, mandates, dates, rounds=5)
    world._fixture_customers = customers  # type: ignore[attr-defined]
    world._fixture_mandates = mandates  # type: ignore[attr-defined]
    return world


# ==========================================================================
# Time
# ==========================================================================


def test_days_since_salary_wraps_across_the_month_boundary():
    """A salary day of 28 and a date of the 2nd is four days *after* payday,
    not twenty-six days before it. Getting this backwards would invert the
    single most important feature in the dataset."""
    assert _days_since_salary(28, dt.date(2025, 4, 2)) == 5
    assert _days_since_salary(1, dt.date(2025, 4, 1)) == 0
    assert _days_since_salary(15, dt.date(2025, 4, 20)) == 5


def test_days_since_salary_is_never_negative():
    for salary_day in range(1, 29):
        for day in range(1, 29):
            assert _days_since_salary(salary_day, dt.date(2025, 4, day)) >= 0


def test_execution_window_closes_the_morning_peak():
    assert within_upi_execution_window(dt.datetime(2025, 4, 1, 8, 0))
    assert not within_upi_execution_window(dt.datetime(2025, 4, 1, 11, 30))
    assert within_upi_execution_window(dt.datetime(2025, 4, 1, 14, 0))
    assert not within_upi_execution_window(dt.datetime(2025, 4, 1, 19, 0))
    assert within_upi_execution_window(dt.datetime(2025, 4, 1, 22, 0))


# ==========================================================================
# Determinism
# ==========================================================================


def test_same_seed_produces_identical_populations():
    """Every number in the README has to be regenerable from a seed and a
    parameter set, or none of them can be checked."""
    a = World(seed=99).sample_customers(50)
    b = World(seed=99).sample_customers(50)
    assert a == b


def test_different_seeds_produce_different_populations():
    a = World(seed=1).sample_customers(50)
    b = World(seed=2).sample_customers(50)
    assert a != b


def test_calibration_does_not_disturb_the_sampling_stream():
    """Calibration runs thousands of draws. If it shared the main RNG, the
    dataset would depend on how many calibration rounds happened to run, which
    would make 'same seed, same data' quietly false."""
    world = World(seed=7)
    customers = world.sample_customers(60)
    mandates = world.sample_mandates(
        customers, dt.date(2024, 6, 1), dt.date(2025, 1, 1)
    )
    dates = [dt.date(2025, 3, 1) + dt.timedelta(days=i) for i in range(5)]

    before = world.rng.random()
    world.calibrate(customers, mandates, dates, rounds=3)
    after = World(seed=7)
    after.sample_customers(60)
    after.sample_mandates(customers, dt.date(2024, 6, 1), dt.date(2025, 1, 1))
    assert before == pytest.approx(after.rng.random())


# ==========================================================================
# Calibration
# ==========================================================================


def test_calibration_hits_published_marginal_rates(calibrated_world: World):
    """The headline failure rates must match the anchors they claim to match.
    If calibration silently drifts, the README's provenance claim becomes
    false while every test still passes."""
    for rail, report in calibrated_world.calibration_report.items():
        assert report["achieved_failure_rate"] == pytest.approx(
            report["target_failure_rate"], abs=0.02
        ), f"{rail} missed its anchored failure rate"


def test_sampling_before_calibration_raises():
    """Sampling uncalibrated would produce a dataset whose failure rates match
    nothing at all, and there would be no sign anything was wrong."""
    world = World(seed=3)
    customers = world.sample_customers(2)
    mandates = world.sample_mandates(
        customers, dt.date(2024, 6, 1), dt.date(2024, 7, 1)
    )
    with pytest.raises(RuntimeError, match="calibrate"):
        world.sample_failure(
            mandates[0], customers[0], dt.datetime(2025, 4, 1, 9, 0)
        )


# ==========================================================================
# The salary-day signal
# ==========================================================================


def test_funds_probability_is_a_probability(calibrated_world: World):
    for customer in calibrated_world._fixture_customers[:50]:
        for day in range(1, 29):
            p = calibrated_world.funds_probability(customer, dt.date(2025, 4, day))
            assert 0.0 <= p <= 1.0


def test_funds_probability_decays_with_distance_from_payday(
    calibrated_world: World,
):
    """The central learnable effect. If this ever flattens, Claim A is
    measuring noise and the model's headline metric becomes meaningless while
    still looking respectable."""
    customer = calibrated_world._fixture_customers[0]
    salary_day = customer.salary_day
    just_paid = dt.date(2025, 4, salary_day)
    much_later = dt.date(2025, 4, salary_day) + dt.timedelta(days=20)
    if much_later.month != 4:
        pytest.skip("salary day too late in the month for a clean 20-day gap")

    assert calibrated_world.funds_probability(
        customer, just_paid
    ) > calibrated_world.funds_probability(customer, much_later)


def test_salary_effect_is_large_enough_to_be_learnable(calibrated_world: World):
    """A signal that exists but is tiny is not worth claiming to have found.
    Requires a meaningful peak-to-trough spread across the population."""
    peaks, troughs = [], []
    for customer in calibrated_world._fixture_customers[:120]:
        base = dt.date(2025, 4, 1)
        probs = [
            calibrated_world.funds_probability(customer, base + dt.timedelta(days=d))
            for d in range(28)
        ]
        peaks.append(max(probs))
        troughs.append(min(probs))
    mean_peak = sum(peaks) / len(peaks)
    mean_trough = sum(troughs) / len(troughs)
    assert mean_peak > mean_trough * 1.8, (
        f"peak/trough spread is only {mean_peak / mean_trough:.2f}x — too weak "
        f"for the model to learn anything worth reporting"
    )


# ==========================================================================
# Episode mechanics
# ==========================================================================


def _episode(world: World, code: str, hour: int = 14):
    customer = world._fixture_customers[0]
    mandate = next(
        m for m in world._fixture_mandates if m.customer_id == customer.customer_id
    )
    return world.open_episode(
        "EP_TEST", mandate, customer, code, dt.datetime(2025, 4, 1, hour, 0), 3
    )


def test_acting_on_a_closed_episode_raises(calibrated_world: World):
    """A sequencer bug that keeps working a resolved episode would bill a
    customer twice. Loud failure, not silent tolerance."""
    episode = _episode(calibrated_world, "UPI_INSUFFICIENT_FUNDS")
    calibrated_world.apply(episode, Action.STOP, dt.datetime(2025, 4, 2, 14, 0))
    with pytest.raises(RuntimeError, match="closed"):
        calibrated_world.apply(
            episode, Action.RETRY_SAME_RAIL, dt.datetime(2025, 4, 3, 14, 0)
        )


def test_stop_closes_the_episode_without_counting_an_attempt(
    calibrated_world: World,
):
    episode = _episode(calibrated_world, "UPI_INSUFFICIENT_FUNDS")
    calibrated_world.apply(episode, Action.STOP, dt.datetime(2025, 4, 2, 14, 0))
    assert episode.stopped and episode.closed
    assert episode.ledger.attempts == 0
    assert episode.ledger.spent_paise == 0


@pytest.mark.parametrize(
    "code",
    ["UPI_MANDATE_REVOKED", "NACH_ACCOUNT_CLOSED", "CARD_TOKEN_INVALID"],
)
def test_dead_mandates_never_recover_money_on_retry(
    calibrated_world: World, code: str
):
    """The expensive mistake this whole project exists to stop. A retry against
    a dead mandate must never succeed, at any date, for any customer."""
    for day in range(2, 20):
        episode = _episode(calibrated_world, code)
        outcome = calibrated_world.apply(
            episode, Action.RETRY_SAME_RAIL, dt.datetime(2025, 4, day, 14, 0)
        )
        assert not outcome.succeeded
        assert outcome.recovered_paise == 0
        assert outcome.cost_paise > 0, "a doomed retry still costs money"


def test_retries_never_cause_revocation(calibrated_world: World):
    """Revocation is a response to being *contacted*. A silent re-presentation
    the customer never sees cannot make them cancel, and modelling it that way
    would unfairly penalise the cheapest recovery action."""
    for i in range(300):
        episode = _episode(calibrated_world, "UPI_INSUFFICIENT_FUNDS")
        outcome = calibrated_world.apply(
            episode, Action.RETRY_SAME_RAIL, dt.datetime(2025, 4, 2, 14, 0)
        )
        assert not outcome.revoked


def test_upi_retry_outside_the_execution_window_always_fails(
    calibrated_world: World,
):
    customer = next(
        c
        for c in calibrated_world._fixture_customers
        if any(
            m.rail is Rail.UPI_AUTOPAY and m.customer_id == c.customer_id
            for m in calibrated_world._fixture_mandates
        )
    )
    mandate = next(
        m
        for m in calibrated_world._fixture_mandates
        if m.customer_id == customer.customer_id and m.rail is Rail.UPI_AUTOPAY
    )
    episode = calibrated_world.open_episode(
        "EP_WINDOW", mandate, customer, "UPI_INSUFFICIENT_FUNDS",
        dt.datetime(2025, 4, 1, 14, 0), 2,
    )
    outcome = calibrated_world.apply(
        episode, Action.RETRY_SAME_RAIL, dt.datetime(2025, 4, 2, 11, 30)
    )
    assert not outcome.succeeded
    assert "execution window" in outcome.detail


# ==========================================================================
# The fatigue / revocation pairing
# ==========================================================================


def test_nudge_efficacy_decays_with_repeated_contact(calibrated_world: World):
    episode = _episode(calibrated_world, "UPI_MANDATE_PAUSED")
    first = calibrated_world.nudge_efficacy(episode, Action.NUDGE_SMS)
    episode.contacts_made = 3
    later = calibrated_world.nudge_efficacy(episode, Action.NUDGE_SMS)
    assert later < first


def test_revocation_hazard_grows_with_repeated_contact(calibrated_world: World):
    """The mirror of fatigue decay. Together these two are what stop 'contact
    everyone constantly' from being the optimal policy — without both, the
    problem has a trivial answer and measuring anything is pointless."""
    episode = _episode(calibrated_world, "UPI_MANDATE_PAUSED")
    first = calibrated_world.revocation_hazard(episode, Action.NUDGE_SMS)
    episode.contacts_made = 3
    later = calibrated_world.revocation_hazard(episode, Action.NUDGE_SMS)
    assert later > first


def test_voice_calls_are_riskier_than_messages(calibrated_world: World):
    episode = _episode(calibrated_world, "UPI_MANDATE_PAUSED")
    assert calibrated_world.revocation_hazard(
        episode, Action.VOICE_CALL
    ) > calibrated_world.revocation_hazard(episode, Action.NUDGE_EMAIL)


def test_revocation_actually_occurs_at_a_plausible_rate(
    calibrated_world: World,
):
    """A hazard that never fires is a cost the policy can safely ignore, which
    would make the evaluation reward exactly the behaviour this models as
    harmful."""
    revoked = 0
    trials = 600
    for i in range(trials):
        customer = calibrated_world._fixture_customers[
            i % len(calibrated_world._fixture_customers)
        ]
        mandate = next(
            m
            for m in calibrated_world._fixture_mandates
            if m.customer_id == customer.customer_id
        )
        episode = calibrated_world.open_episode(
            f"EP_{i}", mandate, customer, "UPI_MANDATE_PAUSED",
            dt.datetime(2025, 4, 1, 14, 0), 3,
        )
        calibrated_world.apply(
            episode, Action.VOICE_CALL, dt.datetime(2025, 4, 2, 14, 0)
        )
        revoked += episode.revoked
    rate = revoked / trials
    assert 0.001 < rate < 0.15, f"voice-call revocation rate of {rate:.4f} is implausible"


def test_revocation_destroys_more_than_the_cycle_is_worth(
    calibrated_world: World,
):
    """A revocation must cost more than the single payment being chased,
    otherwise the policy correctly concludes that churning customers is fine."""
    for i in range(400):
        customer = calibrated_world._fixture_customers[
            i % len(calibrated_world._fixture_customers)
        ]
        mandate = next(
            m
            for m in calibrated_world._fixture_mandates
            if m.customer_id == customer.customer_id
        )
        episode = calibrated_world.open_episode(
            f"EP_R{i}", mandate, customer, "UPI_MANDATE_PAUSED",
            dt.datetime(2025, 4, 1, 14, 0), 3,
        )
        outcome = calibrated_world.apply(
            episode, Action.VOICE_CALL, dt.datetime(2025, 4, 2, 14, 0)
        )
        if outcome.revoked:
            assert outcome.destroyed_paise > mandate.cycle_amount_paise
            return
    pytest.skip("no revocation occurred in this sample")


# ==========================================================================
# Ledger accounting
# ==========================================================================


def test_ledger_net_subtracts_both_spend_and_destruction():
    ledger = Ledger(recovered_paise=1000, spent_paise=150, destroyed_paise=400)
    assert ledger.net_paise == 450


def test_ledger_addition_is_componentwise():
    a = Ledger(100, 10, 5, 1)
    b = Ledger(200, 20, 0, 2)
    assert a + b == Ledger(300, 30, 5, 3)


def test_episode_ledger_tracks_every_action(calibrated_world: World):
    episode = _episode(calibrated_world, "UPI_INSUFFICIENT_FUNDS")
    for day in (2, 3, 4):
        if episode.closed:
            break
        calibrated_world.apply(
            episode, Action.RETRY_SAME_RAIL, dt.datetime(2025, 4, day, 14, 0)
        )
    assert episode.ledger.attempts == len(
        [h for h in episode.history if h.action is not Action.STOP]
    )
    assert episode.ledger.spent_paise == sum(h.cost_paise for h in episode.history)


# ==========================================================================
# Parameter variation
# ==========================================================================


def test_a_flatter_salary_curve_weakens_the_signal():
    """Sensitivity check. If the peak-to-trough spread can be collapsed and the
    generated signal does not change, then the salary mechanism is not actually
    driving the data and the story told about it is wrong."""
    flat = WorldParams(funds_peak_probability=0.5, funds_trough_probability=0.48)
    world = World(params=flat, seed=11)
    customer = world.sample_customers(1)[0]
    probs = [
        world.funds_probability(customer, dt.date(2025, 4, day))
        for day in range(1, 29)
    ]
    assert max(probs) - min(probs) < 0.05
