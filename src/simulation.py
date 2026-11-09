"""
Step 6 - Online, event-driven simulation.

The planner may not look into the future (A15): an incident only exists for
the planner from its timestamp on. We therefore simulate time and re-plan at
every EVENT:
    - an incident becomes known (its timestamp), or
    - a responder becomes free while incidents are still waiting.

At each event:
    1. reveal incidents with timestamp <= now and add them to `pending`;
    2. build the feasible options (capability + route with A*, step 4 costs);
    3. ask the allocator (CSP or greedy, step 5) for a tentative assignment;
    4. COMMIT only the pairs whose responder can leave now. A committed
       assignment is final (A11): the responder drives the route, handles the
       incident and becomes free at the incident location (A9).
    5. Everything else stays pending and is re-planned at the next event.

Step 4 means a busy responder can be planned for an incident (A12) but is
only committed when it is free; by then a better choice may have appeared.

Travel times are the EXPECTED travel times of the model: the simulation has
no ground truth about the real road state, so arrival = departure + expected
travel time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from src.allocation import ALLOCATORS, Option, ResponderState, build_options
from src.data import Incident, Scenario
from src.search import SearchResult, astar
from src.travel_time import TravelTimeModel

EPS = 1e-9


@dataclass
class Assignment:
    responder_id: str
    incident_id: str
    path: list[str]
    departure: float  # "time" in plan.json
    arrival: float
    finish: float  # arrival + time_required
    deadline: float
    timestamp: float

    @property
    def response_time(self) -> float:
        return self.arrival - self.timestamp

    @property
    def deadline_met(self) -> bool:
        return self.arrival <= self.deadline + EPS

    def to_json(self) -> dict:
        return {
            "responder_id": self.responder_id,
            "incident_id": self.incident_id,
            "path": self.path,
            "time": round(self.departure, 2),
        }


@dataclass
class SimulationResult:
    assignments: list[Assignment]
    unhandled: list[Incident]  # never assigned (no capable responder could reach them)
    decisions: int  # number of times the allocator was called
    log: list[str] = field(default_factory=list)


def make_route_function(scenario: Scenario, model: TravelTimeModel) -> Callable[[str, str], Optional[SearchResult]]:
    """A* on the model costs, cached: road beliefs do not change during the
    simulation (A2, A8), so a route between two nodes never changes."""
    cache: dict[tuple[str, str], Optional[SearchResult]] = {}

    def route(start: str, goal: str) -> Optional[SearchResult]:
        if (start, goal) not in cache:
            cache[(start, goal)] = astar(scenario.graph, start, goal, model.cost)
        return cache[(start, goal)]

    return route


def initial_states(scenario: Scenario) -> list[ResponderState]:
    """Busy responders become free at their start location after
    time_until_available seconds (assumption A16)."""
    return [
        ResponderState(r, r.start_location, 0.0 if r.available else r.time_until_available)
        for r in scenario.responders
    ]


def simulate(scenario: Scenario, model: TravelTimeModel, allocator="csp") -> SimulationResult:
    """`allocator` is a name from ALLOCATORS ("csp", "greedy") or a function
    with the same signature (used by the tests)."""
    allocate = ALLOCATORS[allocator] if isinstance(allocator, str) else allocator
    route = make_route_function(scenario, model)
    states = initial_states(scenario)
    state_of = {st.responder.id: st for st in states}

    future = list(scenario.incidents)  # sorted by timestamp, not yet known
    pending: list[Incident] = []
    assignments: list[Assignment] = []
    log: list[str] = []
    decisions = 0
    now = 0.0

    while future or pending:
        # 1. Reveal incidents that became known.
        while future and future[0].timestamp <= now + EPS:
            pending.append(future.pop(0))

        # 2-4. Plan and commit what can start now.
        if pending:
            options = build_options(now, pending, states, route)
            choice: dict[str, Option] = allocate(pending, options)
            decisions += 1
            by_id = {i.id: i for i in pending}
            for inc_id, opt in sorted(choice.items()):
                st = state_of[opt.responder_id]
                if st.free_at > now + EPS:
                    continue  # tentative: responder still busy, re-plan later
                inc = by_id[inc_id]
                a = Assignment(opt.responder_id, inc_id, list(opt.path), now, opt.arrival,
                               opt.arrival + inc.time_required, inc.deadline, inc.timestamp)
                assignments.append(a)
                st.location, st.free_at = inc.location, a.finish
                pending.remove(inc)
                log.append(f"t={now:7.1f}  {a.responder_id:>4} -> {inc_id:<4} arrive {a.arrival:7.1f}"
                           f"  deadline {inc.deadline:g}{'' if a.deadline_met else '  LATE'}")

        # 5. Next event: next incident, or a responder becoming free while work is waiting.
        candidates = [future[0].timestamp] if future else []
        if pending:
            candidates += [st.free_at for st in states if st.free_at > now + EPS]
        if not candidates:
            break  # pending incidents that no responder can ever reach
        now = min(candidates)

    return SimulationResult(assignments, pending, decisions, log)
