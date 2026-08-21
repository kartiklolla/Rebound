"""Train/test splits — time-based and customer-based.

Both are built and both are reported, because they answer different questions
and a model can pass one while failing the other.

**Time split.** Train on an earlier window, test on a strictly later one. This
is the deployment condition — a model trained on history and run against next
month — so it carries the headline numbers. It is the split that catches
temporal leakage and drift.

**Customer split.** Whole customers held out, disjoint sets. This is the
cold-start condition, and it is the split that matters most given how this
dataset is built. The ``cust_prior_*`` features exist specifically so the model
can infer latent traits from a customer's own history; that is exactly the
mechanism by which it could instead memorise individual customers. A model that
is strong on the time split and weak on the customer split has learned people
rather than structure, and will fail on every new signup.

A gap between the two is a finding, not an embarrassment. It is reported.

Episode atomicity
-----------------
Every split cuts on **whole episodes**, never on rows.

``episode_recovered`` is an episode-level label. If a single episode had rows on
both sides of a split, the label of its test rows would be directly readable
from its train rows — not a subtle statistical leak but a literal copy of the
answer. Both splitters enforce this, and ``assert_split_is_clean`` re-checks it
independently of how the split was produced.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class Split:
    """A train/test partition, with an account of what it cost to make."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    question: str
    """What generalisation question this split actually answers."""

    dropped_rows: int = 0
    dropped_reason: str = ""

    def __repr__(self) -> str:
        return (
            f"Split({self.name!r}, train={len(self.train):,}, "
            f"test={len(self.test):,}, dropped={self.dropped_rows:,})"
        )

    @property
    def summary(self) -> dict[str, object]:
        return {
            "split": self.name,
            "question": self.question,
            "train_rows": len(self.train),
            "test_rows": len(self.test),
            "train_episodes": self.train["episode_id"].nunique(),
            "test_episodes": self.test["episode_id"].nunique(),
            "train_customers": self.train["customer_id"].nunique(),
            "test_customers": self.test["customer_id"].nunique(),
            "dropped_rows": self.dropped_rows,
            "dropped_reason": self.dropped_reason,
        }


def time_split(frame: pd.DataFrame, test_fraction: float = 0.3) -> Split:
    """Split forward in time, on episode boundaries, with a hard embargo.

    Episodes are assigned by when they *started*. An episode that begins before
    the cut but is still being worked after it belongs to neither side and is
    dropped.

    Dropping rather than assigning is the conservative choice. Putting a
    straddling episode in train leaks post-cut information backwards; putting it
    in test means the model was trained on the same episode it is being scored
    on. Neither is acceptable, and the number dropped is reported rather than
    quietly absorbed.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    bounds = frame.groupby("episode_id").agg(
        started=("failed_at", "min"), ended=("decided_at", "max")
    )
    cut = bounds["started"].quantile(1.0 - test_fraction)

    train_ids = set(bounds.index[bounds["ended"] < cut])
    test_ids = set(bounds.index[bounds["started"] >= cut])
    straddling = set(bounds.index) - train_ids - test_ids

    train = frame[frame["episode_id"].isin(train_ids)]
    test = frame[frame["episode_id"].isin(test_ids)]
    dropped = int(frame["episode_id"].isin(straddling).sum())

    return Split(
        name="time",
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        question=(
            "Does the model generalise forward in time? This is the deployment "
            "condition: trained on history, run against next month."
        ),
        dropped_rows=dropped,
        dropped_reason=(
            f"{len(straddling):,} episodes straddled the cut at {cut} and were "
            f"dropped rather than assigned to either side"
        ),
    )


def customer_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.3,
    salt: str = "rebound-customer-split-v1",
) -> Split:
    """Split on disjoint customers.

    Assignment is by a **stable hash** of the customer id, not by Python's
    builtin ``hash``, which is salted per process and would silently produce a
    different split on every run — making results irreproducible in a way that
    looks like ordinary run-to-run variance.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    customers = frame["customer_id"].unique()
    assignment = {
        customer_id: _stable_unit_interval(customer_id, salt)
        for customer_id in customers
    }
    test_customers = {
        customer_id
        for customer_id, value in assignment.items()
        if value < test_fraction
    }

    is_test = frame["customer_id"].isin(test_customers)
    return Split(
        name="customer",
        train=frame[~is_test].reset_index(drop=True),
        test=frame[is_test].reset_index(drop=True),
        question=(
            "Does the model generalise to customers it has never seen? This is "
            "the cold-start condition, and the split that catches a model "
            "memorising per-customer history instead of learning structure."
        ),
    )


def _stable_unit_interval(key: str, salt: str) -> float:
    """Deterministic value in [0, 1) for a key, stable across processes."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def all_splits(frame: pd.DataFrame, test_fraction: float = 0.3) -> dict[str, Split]:
    """Both splits, keyed by name. Every reported metric is produced for both."""
    return {
        "time": time_split(frame, test_fraction),
        "customer": customer_split(frame, test_fraction),
    }


# --------------------------------------------------------------------------
# Independent verification
# --------------------------------------------------------------------------


class LeakageError(AssertionError):
    """Raised when a split shares information across the train/test boundary."""


def assert_split_is_clean(split: Split) -> None:
    """Re-derive the split's integrity from the data, not from how it was made.

    Deliberately independent of the splitting code. A bug in ``time_split``
    that also lives in its own internal checks would be invisible; this
    interrogates the resulting frames directly, which is the only version worth
    trusting.
    """
    train, test = split.train, split.test

    if train.empty or test.empty:
        raise LeakageError(f"{split.name} split produced an empty side")

    shared_episodes = set(train["episode_id"]) & set(test["episode_id"])
    if shared_episodes:
        raise LeakageError(
            f"{split.name} split shares {len(shared_episodes)} episodes across "
            f"the boundary. episode_recovered is an episode-level label, so the "
            f"test rows' answer is readable directly from the train rows."
        )

    if split.name == "customer":
        shared_customers = set(train["customer_id"]) & set(test["customer_id"])
        if shared_customers:
            raise LeakageError(
                f"customer split shares {len(shared_customers)} customers; the "
                f"whole point is that the test set is people never seen before"
            )

    if split.name == "time":
        latest_train = train["decided_at"].max()
        earliest_test = test["failed_at"].min()
        if latest_train >= earliest_test:
            raise LeakageError(
                f"time split overlaps: last training decision at {latest_train} "
                f"is not before the first test failure at {earliest_test}"
            )


def split_report(splits: dict[str, Split]) -> pd.DataFrame:
    """Side-by-side account of both splits, for the README."""
    return pd.DataFrame([split.summary for split in splits.values()])
