"""The policy interface.

A policy looks at an open episode and decides what to do next, and when. That
is the whole contract. Everything else — the learned model, the compliance
gate, the sequencer — plugs in behind it, which is what lets a hand-written
rules baseline and a model-driven sequencer be scored by identical code.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rebound.sim.world import EpisodeView
from rebound.taxonomy import Action


@dataclass(frozen=True, slots=True)
class Decision:
    """One action, the moment to take it, and why.

    ``reason`` is not decoration. The track requires an audit trail, and a
    decision log that records *what* without *why* cannot be reviewed by the
    risk team that has to answer for it.
    """

    action: Action
    at: dt.datetime
    reason: str = ""


class Policy(ABC):
    """Decides what to do about a failed debit."""

    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def decide(
        self,
        episode: EpisodeView,
        now: dt.datetime,
        deadline: dt.datetime,
    ) -> Decision | None:
        """Next action for this episode, or ``None`` to stop working it.

        ``episode`` is a frozen :class:`~rebound.sim.world.EpisodeView`, not the
        live episode. A policy cannot mutate the run it is participating in, and
        cannot see the simulator's customer latents. Both were reachable through
        the live object and both were exploited during red-teaming; see
        ``docs/SECURITY_REVIEW.md``.

        ``now`` is the time of the last event; ``deadline`` is when the
        recovery window closes and the next billing cycle takes over.

        Returning ``None`` means the policy is finished — distinct from
        returning ``Action.STOP``, which is an explicit, logged decision to give
        up. Both end the episode; only one of them says so out loud.
        """

    def reset(self) -> None:
        """Clear any per-batch state. Called before each evaluation run."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
