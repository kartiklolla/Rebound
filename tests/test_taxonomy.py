"""Invariant tests for the failure taxonomy.

Most of these are not "does the lookup work" tests. They are guards on
properties the rest of the system silently assumes, written so that breaking
one fails loudly here rather than showing up as an inflated metric later.
"""

from __future__ import annotations

import dataclasses

import pytest

from rebound.taxonomy import (
    ALT_RAILS,
    AMBIGUOUS_CODES,
    FAILURE_MODES,
    Action,
    Disposition,
    FailureMode,
    Rail,
    UnknownFailureCode,
    codes_for_rail,
    disposition_of,
    get_mode,
    is_terminal,
    legal_actions,
)

ALL_CODES = sorted(FAILURE_MODES)


def test_registry_keys_match_their_codes():
    for key, mode in FAILURE_MODES.items():
        assert key == mode.code


def test_every_rail_has_failure_modes():
    for rail in Rail:
        assert codes_for_rail(rail), f"{rail} has no failure modes"


def test_rails_partition_the_taxonomy():
    """Every code belongs to exactly one rail, and none are orphaned."""
    per_rail = [set(codes_for_rail(rail)) for rail in Rail]
    union: set[str] = set().union(*per_rail)
    assert union == set(ALL_CODES)
    total = sum(len(codes) for codes in per_rail)
    assert total == len(ALL_CODES), "a code is claimed by more than one rail"


@pytest.mark.parametrize("code", ALL_CODES)
def test_stop_is_always_legal(code: str):
    """Giving up is legal for every failure. The sequencer relies on this to
    always have at least one action available and never deadlock."""
    assert Action.STOP in legal_actions(code)


@pytest.mark.parametrize("code", ALL_CODES)
def test_terminal_failures_admit_nothing_but_stop(code: str):
    if is_terminal(code):
        assert legal_actions(code) == {Action.STOP}


@pytest.mark.parametrize("code", ALL_CODES)
def test_dead_mandates_are_never_re_presented(code: str):
    """If the mandate did not survive the failure, re-presenting a debit
    against it cannot succeed. Any path that allows it is burning gateway fees
    on a guaranteed decline."""
    mode = get_mode(code)
    if not mode.mandate_alive:
        assert Action.RETRY_SAME_RAIL not in legal_actions(code)


@pytest.mark.parametrize("code", ALL_CODES)
def test_customer_action_failures_can_reach_the_customer(code: str):
    """If only the customer can unblock the debit, at least one action must
    actually reach them. Otherwise the disposition is a dead end that looks
    recoverable."""
    mode = get_mode(code)
    if mode.needs_customer_action:
        reaching = {
            Action.NUDGE_SMS,
            Action.NUDGE_WHATSAPP,
            Action.NUDGE_EMAIL,
            Action.VOICE_CALL,
            Action.SEND_COLLECT_LINK,
        }
        assert legal_actions(code) & reaching, (
            f"{code} needs customer action but no action reaches the customer"
        )


@pytest.mark.parametrize("code", ALL_CODES)
def test_terminal_failures_never_need_customer_action(code: str):
    """A contradiction guard. 'Dead' and 'the customer could fix it' cannot
    both be true — if the customer can fix it, the correct disposition is
    CUSTOMER_ACTION or MANDATE_REPAIR, not TERMINAL."""
    mode = get_mode(code)
    if mode.disposition is Disposition.TERMINAL:
        assert not mode.needs_customer_action


def test_every_disposition_is_exercised():
    """A disposition with no failure mode behind it is dead abstraction."""
    used = {mode.disposition for mode in FAILURE_MODES.values()}
    assert used == set(Disposition)


def test_taxonomy_encodes_no_probabilities():
    """Evaluation-integrity guard, and the reason this file exists.

    The taxonomy must describe structure only. If a recovery rate, prior, or
    weight is ever added to FailureMode, the model would be able to read the
    generator's answer key through the taxonomy and every held-out metric in
    the README would become meaningless.

    Adding a numeric field here should require deleting this test — which is a
    deliberate speed bump, not an obstacle to route around.
    """
    numeric = {
        field.name: field.type
        for field in dataclasses.fields(FailureMode)
        if field.type in {"float", "int", float, int}
    }
    assert not numeric, (
        f"numeric fields found on FailureMode: {numeric}. The taxonomy encodes "
        f"structure, not probabilities — see the module docstring."
    )


def test_alt_rails_cover_every_rail_and_exclude_self():
    assert set(ALT_RAILS) == set(Rail)
    for rail, alternatives in ALT_RAILS.items():
        assert rail not in alternatives, f"{rail} lists itself as a fallback"
        assert len(set(alternatives)) == len(alternatives), "duplicate fallback"


def test_ambiguous_codes_are_flagged_and_retryable():
    """Issuer catch-all codes hide their real cause, so they must not be
    classified as terminal — writing them off is how live customers get
    dropped."""
    assert AMBIGUOUS_CODES, "expected at least one ambiguous issuer code"
    for code in AMBIGUOUS_CODES:
        assert not is_terminal(code)


def test_unknown_code_raises_rather_than_defaulting():
    with pytest.raises(UnknownFailureCode):
        get_mode("NACH_SOME_CODE_THE_RAIL_ADDED_LAST_TUESDAY")


def test_disposition_of_matches_get_mode():
    for code in ALL_CODES:
        assert disposition_of(code) is get_mode(code).disposition
