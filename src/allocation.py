"""
Step 5 - Constraint-based allocation of responders to incidents.

At one decision moment `now`, the simulation (step 6) gives us
    - the pending incidents (known, not yet handled),
    - the state of every responder (where it is, from when it is free),
    - a route function (A* on the travel-time model of step 4).
We decide which responder handles which pending incident NEXT.

Constraint optimisation problem (solved with OR-Tools CP-SAT):

    variables   x[r, i] in {0, 1}   responder r goes to incident i next
                only created for pairs that pass the unary constraints
                (domain filtering / node consistency):
                    - r has the capability required by i
                    - i's location is reachable from r's location
    constraints sum_i x[r, i] <= 1   a responder takes at most one next incident
                sum_r x[r, i] <= 1   an incident gets at most one responder (A10)
    objective   minimise
                  sum_{r,i} x[r,i] * w_i * (response_time(r,i) + LATE * late(r,i))
                + sum_i (1 - sum_r x[r,i]) * w_i * UNASSIGNED

    w_i             priority weight: Critical 4, High 3, Medium 2, Low 1 (A14)
    response_time   arrival - incident timestamp (seconds)
    late            1 if arrival > deadline, else 0

The penalties are chosen so that the objective is lexicographic:
    1. serve as many (weighted) incidents as possible     UNASSIGNED >> LATE
    2. then meet as many (weighted) deadlines as possible LATE >> any response time
    3. then minimise weighted response time
So deadlines act as hard constraints whenever they can be met, and a late
responder is still sent when no one can make it (A13).

The assignment is TENTATIVE: the simulation only commits the pairs whose
responder can leave now. The rest is re-planned at the next event.

A greedy allocator is included as a baseline for the experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from src.data import Incident, Responder
from src.search import SearchResult

PRIORITY_WEIGHT = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
LATE_PENALTY = 100_000  # seconds-equivalent cost of missing a deadline
UNASSIGNED_PENALTY = 1_000_000  # seconds-equivalent cost of leaving an incident unassigned

RouteFunction = Callable[[str, str], Optional[SearchResult]]


@dataclass
class ResponderState:
    responder: Responder
    location: str  # where it is (or will be) when it becomes free
    free_at: float  # simulation time from which it can leave


@dataclass(frozen=True)
class Option:
    """One feasible (responder, incident) pair at the current decision moment."""

    responder_id: str
    incident_id: str
    departure: float  # max(now, responder free_at)
    arrival: float  # departure + expected travel time
    path: tuple[str, ...]
    late: bool  # arrival > deadline
    response_time: float  # arrival - incident timestamp

    def cost(self, incident: Incident) -> int:
        """Objective contribution if this option is chosen (integer for CP-SAT)."""
        w = PRIORITY_WEIGHT[incident.priority]
        return round(w * (self.response_time + (LATE_PENALTY if self.late else 0)))


def unassigned_cost(incident: Incident) -> int:
    return PRIORITY_WEIGHT[incident.priority] * UNASSIGNED_PENALTY


def build_options(
    now: float, pending: list[Incident], states: list[ResponderState], route: RouteFunction
) -> list[Option]:
    """All pairs that satisfy the unary constraints (capability, reachability)."""
    options = []
    for inc in pending:
        for st in states:
            if inc.required_capability not in st.responder.capabilities:
                continue
            result = route(st.location, inc.location)
            if result is None:  # unreachable with the current road beliefs
                continue
            departure = max(now, st.free_at)
            arrival = departure + result.cost
            options.append(Option(
                st.responder.id, inc.id, departure, arrival, tuple(result.path),
                late=arrival > inc.deadline, response_time=arrival - inc.timestamp,
            ))
    return options


def total_cost(choice: dict[str, Option], pending: list[Incident]) -> int:
    """Objective value of a complete assignment (used by tests and experiments)."""
    by_id = {i.id: i for i in pending}
    cost = sum(opt.cost(by_id[i]) for i, opt in choice.items())
    return cost + sum(unassigned_cost(i) for i in pending if i.id not in choice)


# -----------------------------------------------------------------------------
# CSP allocator (OR-Tools CP-SAT)
# -----------------------------------------------------------------------------
def csp_allocate(pending: list[Incident], options: list[Option], time_limit: float = 10.0) -> dict[str, Option]:
    """Optimal tentative assignment incident_id -> Option."""
    from ortools.sat.python import cp_model  # imported here so the rest works without OR-Tools

    if not options:
        return {}
    by_id = {i.id: i for i in pending}
    model = cp_model.CpModel()

    # Variables: one boolean per feasible pair.
    x = {(o.responder_id, o.incident_id): model.NewBoolVar(f"x_{o.responder_id}_{o.incident_id}") for o in options}

    # A responder takes at most one next incident.
    for r in {o.responder_id for o in options}:
        model.AddAtMostOne([x[r, i] for (rr, i) in x if rr == r])

    # An incident gets at most one responder. `assigned[i]` is 1 if it gets one.
    assigned = {}
    for i in by_id:
        vars_i = [x[r, ii] for (r, ii) in x if ii == i]
        assigned[i] = sum(vars_i) if vars_i else 0
        if vars_i:
            model.AddAtMostOne(vars_i)

    # Objective (see module docstring).
    model.Minimize(
        sum(o.cost(by_id[o.incident_id]) * x[o.responder_id, o.incident_id] for o in options)
        + sum(unassigned_cost(by_id[i]) * (1 - assigned[i]) for i in by_id)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1  # single thread: deterministic results
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no solution (status {solver.StatusName(status)})")

    return {o.incident_id: o for o in options if solver.Value(x[o.responder_id, o.incident_id])}


# -----------------------------------------------------------------------------
# Greedy baseline
# -----------------------------------------------------------------------------
def greedy_allocate(pending: list[Incident], options: list[Option]) -> dict[str, Option]:
    """Most urgent incident first (priority, then deadline); each takes the
    responder that arrives earliest among those not yet taken this round."""
    rank = {p: k for k, p in enumerate(PRIORITY_WEIGHT)}  # Critical = 0
    taken: set[str] = set()
    choice: dict[str, Option] = {}
    for inc in sorted(pending, key=lambda i: (rank[i.priority], i.deadline, i.timestamp)):
        candidates = [o for o in options if o.incident_id == inc.id and o.responder_id not in taken]
        if candidates:
            best = min(candidates, key=lambda o: o.arrival)
            choice[inc.id] = best
            taken.add(best.responder_id)
    return choice


# -----------------------------------------------------------------------------
# Exhaustive allocator (reference)
# -----------------------------------------------------------------------------
def exhaustive_allocate(pending: list[Incident], options: list[Option], limit: int = 200_000) -> dict[str, Option]:
    """Depth-first search over every assignment, minimising exactly the same
    objective as the CSP (`total_cost`). Exponential, so only usable for small
    decision moments; it is the reference that the CSP is tested against and
    lets the experiments run without OR-Tools. Ties: the first optimal
    assignment found (incident order, 'unassigned' tried first) is kept."""
    per_incident = [[o for o in options if o.incident_id == inc.id] for inc in pending]
    best: dict = {"cost": None, "choice": {}}
    visited = [0]

    def dfs(k: int, used: frozenset, choice: dict[str, Option]) -> None:
        visited[0] += 1
        if visited[0] > limit:
            raise RuntimeError("decision moment too large for exhaustive search")
        if k == len(pending):
            cost = total_cost(choice, pending)
            if best["cost"] is None or cost < best["cost"]:
                best["cost"], best["choice"] = cost, dict(choice)
            return
        dfs(k + 1, used, choice)  # incident k stays unassigned
        for o in per_incident[k]:
            if o.responder_id not in used:  # constraint: one incident per responder
                choice[o.incident_id] = o
                dfs(k + 1, used | {o.responder_id}, choice)
                del choice[o.incident_id]

    dfs(0, frozenset(), {})
    return best["choice"]


ALLOCATORS = {"csp": csp_allocate, "greedy": greedy_allocate, "exhaustive": exhaustive_allocate}
