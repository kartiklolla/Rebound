"""Tests for the world parameters and their provenance register."""

from __future__ import annotations

from dataclasses import fields

import pytest

from rebound.sim.params import (
    AFA_EXEMPT_CEILING_PAISE,
    FAILURE_MIX,
    MAX_RETRIES_PER_CYCLE,
    PARAM_PROVENANCE,
    PRE_DEBIT_NOTIFICATION_HOURS,
    UPI_EXECUTION_WINDOWS,
    Provenance,
    WorldParams,
    provenance_summary,
)
from rebound.taxonomy import FAILURE_MODES, Rail, get_mode

PARAM_NAMES = [f.name for f in fields(WorldParams)]


@pytest.mark.parametrize("name", PARAM_NAMES)
def test_every_parameter_has_documented_provenance(name: str):
    """The point of the provenance register.

    A parameter without a source entry is a magic number, and a simulator built
    from magic numbers cannot support any claim at all. Adding a parameter must
    mean adding its justification in the same commit.
    """
    assert name in PARAM_PROVENANCE, (
        f"{name!r} has no entry in PARAM_PROVENANCE. Add one — including "
        f"Provenance.ASSUMED with an honest note, if that is the truth."
    )
    source = PARAM_PROVENANCE[name]
    assert source.note.strip(), f"{name!r} has an empty provenance note"


@pytest.mark.parametrize("name", PARAM_NAMES)
def test_anchored_and_derived_parameters_cite_a_source(name: str):
    """ASSUMED may stand alone. ANCHORED and DERIVED may not — those words are
    a claim about the outside world and need something behind them."""
    source = PARAM_PROVENANCE[name]
    if source.provenance in (Provenance.ANCHORED, Provenance.DERIVED):
        assert source.source.strip(), (
            f"{name!r} is marked {source.provenance} but cites no source. "
            f"Either cite one or downgrade it to ASSUMED."
        )


def test_no_orphaned_provenance_entries():
    """Entries for parameters that no longer exist mean the register has
    drifted from the code and can no longer be trusted as documentation."""
    orphans = set(PARAM_PROVENANCE) - set(PARAM_NAMES)
    assert not orphans, f"provenance entries for non-existent params: {orphans}"


def test_provenance_summary_accounts_for_every_parameter():
    summary = provenance_summary()
    assert sum(summary.values()) == len(PARAM_NAMES)


def test_failure_rates_are_probabilities():
    p = WorldParams()
    for rate in (p.upi_failure_rate, p.enach_failure_rate, p.card_failure_rate):
        assert 0.0 < rate < 1.0


def test_failure_mix_codes_exist_in_the_taxonomy():
    """A mix referencing a code the taxonomy does not know would crash at
    sampling time, deep inside a generation run."""
    for rail, mix in FAILURE_MIX.items():
        for code in mix:
            assert code in FAILURE_MODES, f"{code} is not in the taxonomy"


def test_failure_mix_codes_belong_to_their_rail():
    """A card failure code in the UPI mix would silently generate impossible
    data that the model would then dutifully learn from."""
    for rail, mix in FAILURE_MIX.items():
        for code in mix:
            assert get_mode(code).rail is rail, (
                f"{code} is a {get_mode(code).rail} code but appears in the "
                f"{rail} mix"
            )


def test_every_rail_has_a_failure_mix():
    assert set(FAILURE_MIX) == set(Rail)


def test_every_mix_contains_an_insufficient_funds_code():
    """Calibration splits each rail's target rate into an NSF share and a
    remainder. A mix with no NSF code makes that split degenerate."""
    for rail, mix in FAILURE_MIX.items():
        assert any("INSUFFICIENT_FUNDS" in code for code in mix), rail


def test_mix_weights_are_positive():
    for rail, mix in FAILURE_MIX.items():
        for code, weight in mix.items():
            assert weight > 0, f"{rail}/{code} has non-positive weight"


def test_execution_windows_are_ordered_and_disjoint():
    """Overlapping windows would make the compliance layer's answer depend on
    iteration order, which is the kind of bug that only shows up in a demo."""
    for start, end in UPI_EXECUTION_WINDOWS:
        assert start < end, f"window {start}-{end} is inverted"
    ordered = sorted(UPI_EXECUTION_WINDOWS)
    assert list(UPI_EXECUTION_WINDOWS) == ordered, "windows are not sorted"
    for (_, end), (next_start, _) in zip(ordered, ordered[1:]):
        assert end < next_start, "execution windows overlap"


def test_execution_windows_leave_peak_hours_closed():
    """The whole point of the rule is that the late-morning peak is shut. If a
    change ever opens it, the timing problem this project exists to solve
    quietly disappears."""
    import datetime as dt

    peak = dt.time(11, 30)
    assert not any(
        start <= peak <= end for start, end in UPI_EXECUTION_WINDOWS
    ), "the 10:00-13:00 peak window should be closed to mandate executions"


def test_regulatory_constants_are_sane():
    assert PRE_DEBIT_NOTIFICATION_HOURS == 24
    assert MAX_RETRIES_PER_CYCLE >= 1
    assert AFA_EXEMPT_CEILING_PAISE > 0
