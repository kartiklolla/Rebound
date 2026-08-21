"""Tests for the train/test splits.

Every test here is a leakage test in some form. A split that quietly shares
information produces a model with excellent held-out metrics and no value, and
nothing about that failure announces itself.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rebound.eval.splits import (
    LeakageError,
    Split,
    all_splits,
    assert_split_is_clean,
    customer_split,
    split_report,
    time_split,
)
from rebound.sim.dataset import GenerationConfig, generate_log

CONFIG = GenerationConfig(
    n_customers=400,
    start=dt.date(2025, 1, 1),
    end=dt.date(2025, 12, 31),
    seed=808,
)


@pytest.fixture(scope="module")
def log():
    return generate_log(CONFIG)


# ==========================================================================
# Episode atomicity — the one that matters most
# ==========================================================================


@pytest.mark.parametrize("split_name", ["time", "customer"])
def test_no_episode_spans_the_boundary(log, split_name: str):
    """``episode_recovered`` is an episode-level label.

    If a single episode had rows on both sides, the label of its test rows
    would be readable directly from its train rows — not a subtle statistical
    leak but a literal copy of the answer.
    """
    split = all_splits(log)[split_name]
    shared = set(split.train["episode_id"]) & set(split.test["episode_id"])
    assert not shared, f"{split_name} split shares {len(shared)} episodes"


@pytest.mark.parametrize("split_name", ["time", "customer"])
def test_clean_split_assertion_passes(log, split_name: str):
    assert_split_is_clean(all_splits(log)[split_name])


# ==========================================================================
# Time split
# ==========================================================================


def test_time_split_is_strictly_ordered(log):
    """No training decision may occur at or after the first test failure.
    Otherwise the model has seen the future it is being asked to predict."""
    split = time_split(log)
    assert split.train["decided_at"].max() < split.test["failed_at"].min()


def test_time_split_drops_straddling_episodes_rather_than_assigning_them(log):
    """Assigning a straddling episode to train leaks post-cut information
    backwards; assigning it to test scores the model on an episode it trained
    on. Dropping is the only clean option, and the count is reported."""
    split = time_split(log)
    assert split.dropped_rows > 0, (
        "expected some episodes to straddle the cut in a log this size"
    )
    assert "dropped rather than assigned" in split.dropped_reason
    assert len(split.train) + len(split.test) + split.dropped_rows == len(log)


def test_time_split_respects_the_requested_fraction(log):
    split = time_split(log, test_fraction=0.25)
    share = len(split.test) / (len(split.train) + len(split.test))
    assert 0.15 < share < 0.35


def test_time_split_rejects_impossible_fractions(log):
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            time_split(log, test_fraction=bad)


# ==========================================================================
# Customer split
# ==========================================================================


def test_customer_split_has_disjoint_customers(log):
    split = customer_split(log)
    assert not set(split.train["customer_id"]) & set(split.test["customer_id"])


def test_customer_split_keeps_every_row(log):
    """Unlike the time split there is nothing to drop — every customer belongs
    to exactly one side."""
    split = customer_split(log)
    assert len(split.train) + len(split.test) == len(log)


def test_customer_split_is_stable_across_processes(log):
    """Python's builtin hash is salted per process. Using it would silently
    produce a different split on every run, and the resulting variance would
    look like ordinary run-to-run noise rather than a reproducibility bug."""
    first = customer_split(log)
    second = customer_split(log)
    assert set(first.test["customer_id"]) == set(second.test["customer_id"])


def test_customer_split_salt_changes_the_partition(log):
    a = customer_split(log, salt="salt-a")
    b = customer_split(log, salt="salt-b")
    assert set(a.test["customer_id"]) != set(b.test["customer_id"])


def test_customer_split_respects_the_requested_fraction(log):
    split = customer_split(log, test_fraction=0.3)
    customers = log["customer_id"].nunique()
    share = split.test["customer_id"].nunique() / customers
    assert 0.2 < share < 0.4


# ==========================================================================
# The verifier itself
# ==========================================================================


def test_leakage_verifier_catches_shared_episodes(log):
    """The verifier must actually fail on a bad split. A safety check that
    cannot fail is decoration."""
    bad = Split(name="custom", train=log, test=log, question="deliberately broken")
    with pytest.raises(LeakageError, match="episodes"):
        assert_split_is_clean(bad)


def test_leakage_verifier_catches_shared_customers(log):
    episodes = log["episode_id"].unique()
    half = set(episodes[: len(episodes) // 2])
    bad = Split(
        name="customer",
        train=log[log["episode_id"].isin(half)],
        test=log[~log["episode_id"].isin(half)],
        question="episodes are disjoint but customers are not",
    )
    with pytest.raises(LeakageError, match="customers"):
        assert_split_is_clean(bad)


def test_leakage_verifier_catches_an_empty_side(log):
    bad = Split(
        name="time", train=log, test=log.iloc[0:0], question="empty test side"
    )
    with pytest.raises(LeakageError, match="empty"):
        assert_split_is_clean(bad)


# ==========================================================================
# Reporting
# ==========================================================================


def test_split_report_covers_both_splits(log):
    report = split_report(all_splits(log))
    assert set(report["split"]) == {"time", "customer"}
    assert (report["train_rows"] > 0).all()
    assert (report["test_rows"] > 0).all()


def test_both_splits_retain_enough_data_to_train_on(log):
    for name, split in all_splits(log).items():
        assert len(split.train) > 500, f"{name} training set is too small"
        assert len(split.test) > 200, f"{name} test set is too small"
        assert split.test["episode_recovered"].nunique() == 2, (
            f"{name} test set has only one outcome class; every metric would "
            f"be undefined or meaningless"
        )
