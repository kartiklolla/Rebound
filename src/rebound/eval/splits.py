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
from enum import StrEnum

import pandas as pd


class SplitKind(StrEnum):
    """What kind of boundary a split draws, and therefore which checks apply.

    A typed field rather than the free-text ``name`` it replaced. The verifier
    used to decide which integrity checks to run by comparing ``name`` against
    string literals, so a split named anything unrecognised — ``"whatever"``,
    or a typo — silently skipped both the temporal and the customer-disjointness
    check while still reporting as verified.

    That is a bad failure in any function; in one whose documented purpose is
    being "independent of the splitting code" it defeats the entire point.
    """

    TIME = "time"
    CUSTOMER = "customer"


@dataclass(frozen=True, slots=True)
class Split:
    """A train/test partition, with an account of what it cost to make."""

    kind: SplitKind
    train: pd.DataFrame
    test: pd.DataFrame
    question: str
    """What generalisation question this split actually answers."""

    dropped_rows: int = 0
    dropped_reason: str = ""

    @property
    def name(self) -> str:
        return str(self.kind)

    def __repr__(self) -> str:
        return (
            f"Split({self.kind.value!r}, train={len(self.train):,}, "
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
        kind=SplitKind.TIME,
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
        kind=SplitKind.CUSTOMER,
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


def assert_split_is_clean(split: Split, twin_tolerance: float = 0.01) -> None:
    """Re-derive the split's integrity from the data, not from how it was made.

    Deliberately independent of the splitting code. A bug in ``time_split``
    that also lives in its own internal checks would be invisible; this
    interrogates the resulting frames directly.

    Checks, in order of how badly each one has failed in the past:

    1. Neither side is empty.
    2. The test side has both outcome classes — otherwise every downstream
       metric is NaN or degenerate, and the split "passes" while being useless.
    3. No episode spans the boundary (episode-level label).
    4. No *feature twin* spans the boundary. Identity checks on
       ``episode_id``/``customer_id`` are trivially defeated by renaming: an
       earlier version passed a split built by cloning the test rows with
       prefixed ids and shifted dates, from which the label could be recovered
       by an exact feature join at accuracy 1.000. Realistically this is how
       resampling, augmentation, or a de-duplication bug leaks.
    5. Kind-specific: temporal ordering for TIME, customer disjointness for
       CUSTOMER. Dispatched on a typed :class:`SplitKind`, never a free-text
       name — the string version silently skipped both for any unrecognised
       value.
    """
    train, test = split.train, split.test

    if train.empty or test.empty:
        raise LeakageError(f"{split.name} split produced an empty side")

    if "episode_recovered" not in test.columns:
        raise LeakageError(
            f"{split.name} split has no 'episode_recovered' column; the label "
            f"is missing and nothing downstream can be scored"
        )
    if test["episode_recovered"].nunique() < 2:
        raise LeakageError(
            f"{split.name} split's test side has a single outcome class. Every "
            f"discrimination metric would be NaN while the split still looked "
            f"valid."
        )

    shared_episodes = set(train["episode_id"]) & set(test["episode_id"])
    if shared_episodes:
        raise LeakageError(
            f"{split.name} split shares {len(shared_episodes)} episodes across "
            f"the boundary. episode_recovered is an episode-level label, so the "
            f"test rows' answer is readable directly from the train rows."
        )

    _assert_no_feature_twins(split, twin_tolerance)

    if split.kind is SplitKind.CUSTOMER:
        shared_customers = set(train["customer_id"]) & set(test["customer_id"])
        if shared_customers:
            raise LeakageError(
                f"customer split shares {len(shared_customers)} customers; the "
                f"whole point is that the test set is people never seen before"
            )

    if split.kind is SplitKind.TIME:
        latest_train = train["decided_at"].max()
        earliest_test = test["failed_at"].min()
        if latest_train >= earliest_test:
            raise LeakageError(
                f"time split overlaps: last training decision at {latest_train} "
                f"is not before the first test failure at {earliest_test}"
            )


#: Columns compared when looking for duplicated rows across the boundary.
#:
#: Deliberately excludes identifiers and timestamps — those are exactly what a
#: cloned row has had rewritten. What survives cloning is the feature content,
#: so that is what gets hashed.
_TWIN_COLUMNS: tuple[str, ...] = (
    "failure_code",
    "rail",
    "amount_paise",
    "ceiling_paise",
    "billing_day",
    "cycles_elapsed",
    "decision_index",
    "action",
    "prior_attempts",
    "prior_contacts",
    "cust_prior_failures",
    "cust_prior_recoveries",
    "cust_prior_recovery_rate",
)


def _assert_no_feature_twins(split: Split, tolerance: float) -> None:
    """Fail if too many test rows have an exact feature duplicate in train.

    A small overlap is expected and harmless — with discrete features, some
    rows genuinely coincide. A large one means rows were copied, and the label
    can be looked up rather than predicted.
    """
    columns = [c for c in _TWIN_COLUMNS if c in split.train.columns]
    if not columns:
        return

    def fingerprint(frame: pd.DataFrame) -> pd.Series:
        # Missing values are given an explicit token rather than being cast.
        # ``astype(str)`` leaves NaN as a float under pandas 3, which makes the
        # join raise — and a fingerprint that silently skipped NaN-bearing
        # columns would quietly stop detecting clones in exactly the features
        # that carry missingness.
        return (
            frame[columns].astype("string").fillna("<missing>").agg("|".join, axis=1)
        )

    train_prints = set(fingerprint(split.train))
    test_prints = fingerprint(split.test)
    twins = int(test_prints.isin(train_prints).sum())
    share = twins / len(split.test)

    if share > tolerance:
        raise LeakageError(
            f"{split.name} split: {twins:,} of {len(split.test):,} test rows "
            f"({share:.1%}) have an exact feature twin in train, above the "
            f"{tolerance:.1%} tolerance. Disjoint ids do not make a split "
            f"clean if the rows were copied — the label can be joined rather "
            f"than predicted."
        )


def split_report(splits: dict[str, Split]) -> pd.DataFrame:
    """Side-by-side account of both splits, for the README."""
    return pd.DataFrame([split.summary for split in splits.values()])
