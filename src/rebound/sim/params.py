"""Every parameter of the synthetic world, in one file, each with its provenance.

Why this file exists in this shape
----------------------------------
The first question any serious reader asks about a simulator-based result is
"where did these numbers come from?" The usual answer is that they are scattered
across the generator as magic constants and nobody, including the author,
can reconstruct which were researched and which were invented on a Tuesday.

So every parameter here carries a ``Provenance``:

``ANCHORED``
    Traceable to a published figure, with the citation attached.
``DERIVED``
    Computed from anchored figures, with the reasoning attached.
``ASSUMED``
    Judgement. No source. Labelled honestly rather than dressed up.

``test_params.py`` fails if any field of ``WorldParams`` lacks an entry. The
README reports how many parameters fall in each bucket, including the
unflattering count.

A note on the ASSUMED ones: they are mostly *shapes* rather than levels — how
fast nudge efficacy decays with repeated contact, how concentrated the salary
effect is. Claim B is a relative comparison between policies run against the
same world, so it is robust to these being somewhat wrong in level. It is not
robust to them being wrong in sign, and where that risk is real it is noted on
the parameter.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, fields
from enum import StrEnum

from rebound.taxonomy import Rail


class Provenance(StrEnum):
    ANCHORED = "anchored"
    DERIVED = "derived"
    ASSUMED = "assumed"


@dataclass(frozen=True, slots=True)
class ParamSource:
    provenance: Provenance
    note: str
    source: str = ""


# ==========================================================================
# Regulatory constants — re-exported, not owned
# ==========================================================================
#
# These live in ``rebound.regulation`` now. They were never simulator
# parameters: a parameter is the generator's opinion, and getting one wrong
# shifts a measured number. A regulatory constant is a claim about the world,
# and getting one wrong makes the system do something non-compliant while
# reporting that it complied — which no amount of testing against our own
# simulator can catch, because the simulator would be wrong the same way.
#
# Re-exported here so existing importers keep working.

from rebound.regulation import (  # noqa: E402
    AFA_EXEMPT_CEILING_PAISE,
    MAX_EXECUTIONS_PER_CYCLE,
    MAX_RETRIES_PER_CYCLE,
    PRE_DEBIT_NOTIFICATION_HOURS,
    UPI_EXECUTION_WINDOWS,
)

__all_regulatory__ = (
    "AFA_EXEMPT_CEILING_PAISE",
    "MAX_EXECUTIONS_PER_CYCLE",
    "MAX_RETRIES_PER_CYCLE",
    "PRE_DEBIT_NOTIFICATION_HOURS",
    "UPI_EXECUTION_WINDOWS",
)


# ==========================================================================
# World parameters
# ==========================================================================


@dataclass(frozen=True, slots=True)
class WorldParams:
    """Behavioural parameters of the synthetic world.

    Frozen and passed explicitly rather than read from module scope, so that
    sensitivity analysis is a matter of constructing a variant rather than
    monkey-patching globals.
    """

    # -- failure incidence ------------------------------------------------
    upi_failure_rate: float = 0.55
    enach_failure_rate: float = 0.31
    card_failure_rate: float = 0.28

    # -- customer latents -------------------------------------------------
    salary_day_early_share: float = 0.78
    balance_health_alpha: float = 2.2
    balance_health_beta: float = 2.0
    engagement_alpha: float = 1.8
    engagement_beta: float = 2.6
    churn_intent_alpha: float = 1.1
    churn_intent_beta: float = 9.0

    # -- the salary-day effect -------------------------------------------
    funds_peak_probability: float = 0.88
    funds_trough_probability: float = 0.17
    funds_peak_window_days: int = 3
    funds_decay_days: float = 11.0

    # -- nudge response ---------------------------------------------------
    nudge_base_efficacy: float = 0.34
    channel_match_multiplier: float = 1.7
    contact_fatigue_decay: float = 0.62
    voice_efficacy_multiplier: float = 1.9
    nudge_top_up_lift: float = 0.55

    # -- revocation risk --------------------------------------------------
    revocation_base_rate: float = 0.021
    revocation_fatigue_growth: float = 1.45
    passive_revocation_rate: float = 0.09

    # -- transient failures ----------------------------------------------
    outage_mean_hours: float = 5.5
    bank_outage_dispersion: float = 0.9

    # -- mandate population ----------------------------------------------
    alt_rail_availability: float = 0.41
    mean_cycle_amount_paise: int = 84_900
    cycle_amount_log_sigma: float = 0.85


DEFAULT_PARAMS = WorldParams()


# ==========================================================================
# Provenance register
# ==========================================================================

_NPCI_NACH = (
    "NPCI NACH debit data reported by Business Standard: ~31% of presentations "
    "failed by volume, ~24.9% by value."
)
_NPCI_UPI_AUTOPAY = (
    "NPCI UPI Autopay data reported in trade press: mandate execution success "
    "fell from ~50% (Jan 2024) to ~30% (Nov 2025) as volumes grew ~10x."
)
_NPCI_REVOCATIONS = (
    "Reported NPCI figures: >20 million UPI Autopay mandates revoked per month, "
    "attributed largely to insufficient customer balances."
)

PARAM_PROVENANCE: dict[str, ParamSource] = {
    # -- failure incidence ------------------------------------------------
    "upi_failure_rate": ParamSource(
        Provenance.DERIVED,
        "Published success of ~30% implies ~70% failure. Using 0.55 instead, "
        "deliberately conservative: the headline figure counts executions "
        "blocked by NPCI execution windows, which this world models as a "
        "separate mechanism. Taking 0.70 would double-count them.",
        _NPCI_UPI_AUTOPAY,
    ),
    "enach_failure_rate": ParamSource(
        Provenance.ANCHORED,
        "Used directly at the published volume-terms rate.",
        _NPCI_NACH,
    ),
    "card_failure_rate": ParamSource(
        Provenance.ASSUMED,
        "No India-specific published figure found for recurring card-on-file "
        "failure. Placed below eNACH on the reasoning that card limits are "
        "revolving where bank balances are not. Weakly held.",
    ),
    # -- customer latents -------------------------------------------------
    "salary_day_early_share": ParamSource(
        Provenance.ASSUMED,
        "Share of salaried customers credited in the first week of the month. "
        "The salary-day effect itself is well attested; this specific "
        "concentration is judgement.",
    ),
    "balance_health_alpha": ParamSource(
        Provenance.ASSUMED,
        "Beta shape for chronic-vs-occasional low balance. Mildly "
        "right-of-centre: most accounts are usually funded.",
    ),
    "balance_health_beta": ParamSource(
        Provenance.ASSUMED, "Paired with balance_health_alpha."
    ),
    "engagement_alpha": ParamSource(
        Provenance.ASSUMED,
        "Beta shape for responsiveness to merchant contact. Left-skewed: most "
        "customers ignore most messages, which is the realistic case and the "
        "one that makes nudge spend easy to waste.",
    ),
    "engagement_beta": ParamSource(
        Provenance.ASSUMED, "Paired with engagement_alpha."
    ),
    "churn_intent_alpha": ParamSource(
        Provenance.DERIVED,
        "Tuned so aggregate revocation incidence under the historical ladder "
        "lands near reported levels. Strongly left-skewed: a small minority "
        "carry real churn intent, and they are the ones a nudge converts from "
        "a lapsed payment into a cancelled subscription.",
        _NPCI_REVOCATIONS,
    ),
    "churn_intent_beta": ParamSource(
        Provenance.DERIVED, "Paired with churn_intent_alpha.", _NPCI_REVOCATIONS
    ),
    # -- the salary-day effect -------------------------------------------
    "funds_peak_probability": ParamSource(
        Provenance.ASSUMED,
        "Probability an account is funded in the days just after salary "
        "credit. The *existence* of this effect is the central learnable "
        "signal; its exact height is judgement.",
    ),
    "funds_trough_probability": ParamSource(
        Provenance.ASSUMED,
        "Probability an account is funded at the furthest point from salary "
        "credit. The peak-to-trough ratio is what the model has to discover.",
    ),
    "funds_peak_window_days": ParamSource(
        Provenance.ASSUMED, "Days after salary credit before decay begins."
    ),
    "funds_decay_days": ParamSource(
        Provenance.ASSUMED,
        "Exponential decay constant from peak to trough across the month.",
    ),
    # -- nudge response ---------------------------------------------------
    "nudge_base_efficacy": ParamSource(
        Provenance.ASSUMED,
        "Probability a maximally engaged customer acts on a first, "
        "channel-matched nudge. Sets the ceiling on what nudging can achieve.",
    ),
    "channel_match_multiplier": ParamSource(
        Provenance.ASSUMED,
        "Uplift when a nudge lands on the customer's preferred channel. "
        "Channel preference is latent, so this is a real thing for the model "
        "to infer from response history rather than read off a field.",
    ),
    "contact_fatigue_decay": ParamSource(
        Provenance.ASSUMED,
        "Multiplicative efficacy decay per prior contact in the episode. "
        "Direction is safe — repeated chasing works less well. Magnitude is "
        "judgement, and it is the parameter Claim B is most sensitive to, so "
        "it gets a sensitivity sweep in the evaluation.",
    ),
    "voice_efficacy_multiplier": ParamSource(
        Provenance.ASSUMED,
        "A phone call is harder to ignore than a message. Paired with a "
        "correspondingly higher revocation risk, which is the whole tension: "
        "the most effective channel is also the most dangerous.",
    ),
    "nudge_top_up_lift": ParamSource(
        Provenance.ASSUMED,
        "How much a customer who responds to a nudge closes the gap toward "
        "being funded — i.e. they top up the account. Without this a nudge "
        "does nothing for insufficient funds, which is the largest failure "
        "class on every rail, and nudging would look useless where it is in "
        "fact the main lever merchants actually have.",
    ),
    # -- revocation risk --------------------------------------------------
    "revocation_base_rate": ParamSource(
        Provenance.DERIVED,
        "Per-contact revocation hazard at unit intrusiveness for a customer of "
        "median churn intent, tuned against reported revocation volumes.",
        _NPCI_REVOCATIONS,
    ),
    "revocation_fatigue_growth": ParamSource(
        Provenance.ASSUMED,
        "Growth in revocation hazard per prior contact. The mirror of "
        "contact_fatigue_decay: chasing harder both works less and costs more. "
        "This pairing is what stops 'contact everyone constantly' from being "
        "the optimal policy.",
    ),
    "passive_revocation_rate": ParamSource(
        Provenance.DERIVED,
        "Probability that an unresolved failed payment ends in revocation with "
        "no merchant contact at all, for a customer of median churn intent. "
        "Customers cancel on their own; reported revocation volumes are "
        "attributed largely to insufficient balances rather than to dunning. "
        "Without this the only cause of churn in the world is merchant "
        "contact, which makes 'never speak to anyone' optimal and hands the "
        "comparison to any policy that only retries. It also inverts the "
        "central economics: recovering a payment should *prevent* churn, and "
        "that is only true if unrecovered payments cause it.",
        _NPCI_REVOCATIONS,
    ),
    # -- transient failures ----------------------------------------------
    "outage_mean_hours": ParamSource(
        Provenance.ASSUMED,
        "Mean duration of a bank-side technical outage. Governs how long to "
        "wait before re-presenting a transient failure.",
    ),
    "bank_outage_dispersion": ParamSource(
        Provenance.ASSUMED,
        "Log-normal dispersion of outage duration across banks. Non-zero so "
        "that bank identity carries real signal and the model has something "
        "genuine to learn from it.",
    ),
    # -- mandate population ----------------------------------------------
    "alt_rail_availability": ParamSource(
        Provenance.ASSUMED,
        "Share of customers holding a usable mandate on a second rail. Gates "
        "how often rail fallback is even an option.",
    ),
    "mean_cycle_amount_paise": ParamSource(
        Provenance.ASSUMED,
        "Median recurring debit around ₹849 — a subscription/SIP-shaped "
        "population rather than an EMI-shaped one.",
    ),
    "cycle_amount_log_sigma": ParamSource(
        Provenance.ASSUMED,
        "Log-normal spread of cycle amounts. Wide enough that some debits sit "
        "above the AFA-exempt ceiling, so that failure mode occurs naturally "
        "instead of being injected.",
    ),
}


# ==========================================================================
# Failure-code mix
# ==========================================================================

#: Conditional distribution over failure codes, given that a debit failed.
#:
#: Insufficient funds dominates every rail — the one strongly attested fact in
#: this table, and the reason timing beats persistence. The rest of each mix is
#: shaped by judgement about relative frequency and is labelled accordingly.
#: Weights are unnormalised; the world normalises them.
FAILURE_MIX: dict[Rail, dict[str, float]] = {
    Rail.UPI_AUTOPAY: {
        "UPI_INSUFFICIENT_FUNDS": 0.52,
        "UPI_PSP_UNAVAILABLE": 0.13,
        "UPI_AFA_NOT_COMPLETED": 0.10,
        "UPI_MANDATE_REVOKED": 0.09,
        "UPI_MANDATE_PAUSED": 0.05,
        "UPI_AMOUNT_EXCEEDS_MANDATE": 0.04,
        "UPI_RETRY_LIMIT_EXCEEDED": 0.03,
        "UPI_MANDATE_EXPIRED": 0.02,
        "UPI_ACCOUNT_BLOCKED": 0.02,
    },
    Rail.ENACH: {
        "NACH_INSUFFICIENT_FUNDS": 0.58,
        "NACH_TECHNICAL_DECLINE": 0.11,
        "NACH_MANDATE_CANCELLED": 0.09,
        "NACH_MANDATE_NOT_REGISTERED": 0.06,
        "NACH_AMOUNT_EXCEEDS_MANDATE": 0.05,
        "NACH_ACCOUNT_DORMANT": 0.04,
        "NACH_MANDATE_DEFECT": 0.03,
        "NACH_ACCOUNT_CLOSED": 0.03,
        "NACH_DEBIT_NOT_PERMITTED": 0.01,
    },
    Rail.CARD_ON_FILE: {
        "CARD_INSUFFICIENT_FUNDS": 0.34,
        "CARD_DO_NOT_HONOUR": 0.24,
        "CARD_AFA_REQUIRED": 0.12,
        "CARD_PRE_DEBIT_NOTIFICATION_MISSING": 0.09,
        "CARD_EXPIRED": 0.07,
        "CARD_ISSUER_UNAVAILABLE": 0.06,
        "CARD_TOKEN_INVALID": 0.04,
        "CARD_STANDING_INSTRUCTION_CANCELLED": 0.03,
        "CARD_BLOCKED": 0.01,
    },
}

FAILURE_MIX_PROVENANCE = ParamSource(
    Provenance.DERIVED,
    "Insufficient funds set as the dominant mode on every rail, which is the "
    "well-attested part. The share of CARD_DO_NOT_HONOUR is deliberately "
    "large: issuer catch-all declines are known to be a substantial slice of "
    "card failures, and it is the code where a learned model has the most to "
    "offer over a rules ladder, since its true cause is unknowable a priori. "
    "Relative ordering within the remainder is judgement.",
    f"{_NPCI_NACH} {_NPCI_REVOCATIONS}",
)


def provenance_summary() -> dict[Provenance, int]:
    """Count of ``WorldParams`` fields by provenance.

    Reported in the README, including the ASSUMED count. A simulator whose
    parameters are mostly assumptions is not disqualifying; one that hides how
    many are assumptions is.
    """
    counts = dict.fromkeys(Provenance, 0)
    for field in fields(WorldParams):
        counts[PARAM_PROVENANCE[field.name].provenance] += 1
    return counts
