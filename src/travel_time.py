"""
Step 4 - Travel-time model: from beliefs to a road cost for A*.

Combines
    - the Bayes net belief P(Status | evidence) per road  (step 2)
    - the HMM belief B(X_t) over traffic per road          (step 3)
into the cost function `cost(road)` that the search of step 1 takes.

Rules (agreed assumptions, see report section 2):
    A6  A road is unusable when P(Blocked) > BLOCKED_THRESHOLD (0.5).
        Otherwise its status factor is the expected factor GIVEN that it is
        not blocked:
            status_factor = (P(Open) * 1.0 + P(Congested) * 1.5) / (P(Open) + P(Congested))
    A7  Traffic factor = expected factor under the HMM belief:
            traffic_factor = sum_x B(x) * TRAFFIC_FACTORS[x]
        Roads without traffic history get factor 1.0.

    expected travel time = base_travel_time * status_factor * traffic_factor

Multiplying the two expectations is only exact if status and traffic are
independent. They are not really ("Congested" and "High traffic" describe a
similar situation), so this may double-count. This is listed as a limitation.

Every factor is >= 1, so the A* heuristic of step 1 stays admissible and
consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.bayes import road_status_beliefs
from src.data import Road, Scenario
from src.hmm import traffic_beliefs

BLOCKED_THRESHOLD = 0.5
STATUS_FACTORS = {"Open": 1.0, "Congested": 1.5}  # Blocked = unusable
TRAFFIC_FACTORS = {"Low": 1.0, "Medium": 1.25, "High": 1.5}


@dataclass
class TravelTimeModel:
    status_beliefs: dict[tuple[str, str], dict[str, float]]
    traffic_beliefs: dict[tuple[str, str], dict[str, float]]
    blocked_threshold: float = BLOCKED_THRESHOLD
    traffic_factors: dict[str, float] = field(default_factory=lambda: dict(TRAFFIC_FACTORS))
    use_status: bool = True  # False: ignore the Bayes net (for experiments)
    use_traffic: bool = True  # False: ignore the HMM (for experiments)

    @classmethod
    def from_scenario(cls, scenario: Scenario, **options) -> "TravelTimeModel":
        return cls(road_status_beliefs(scenario), traffic_beliefs(scenario), **options)

    # --- the three ingredients -------------------------------------------------
    def is_blocked(self, road: Road) -> bool:
        if not self.use_status:
            return False
        return self.status_beliefs[(road.from_node, road.to_node)]["Blocked"] > self.blocked_threshold

    def status_factor(self, road: Road) -> float:
        """Expected slowdown from the road status, given that the road is not blocked."""
        if not self.use_status:
            return 1.0
        b = self.status_beliefs[(road.from_node, road.to_node)]
        usable = b["Open"] + b["Congested"]
        return (b["Open"] * STATUS_FACTORS["Open"] + b["Congested"] * STATUS_FACTORS["Congested"]) / usable

    def traffic_factor(self, road: Road) -> float:
        """Expected slowdown from traffic under the HMM belief (1.0 without history)."""
        belief = self.traffic_beliefs.get((road.from_node, road.to_node))
        if not self.use_traffic or belief is None:
            return 1.0
        return sum(p * self.traffic_factors[level] for level, p in belief.items())

    # --- the cost function used by the search ------------------------------------
    def cost(self, road: Road) -> Optional[float]:
        """Expected travel time in seconds, or None if the road is treated as blocked."""
        if self.is_blocked(road):
            return None
        return road.base_travel_time * self.status_factor(road) * self.traffic_factor(road)
