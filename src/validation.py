"""
Step 7a - Validation of a plan, independent of the simulation.

The validator reads the plan (the list that is written to plan.json) and
the scenario, and REPLAYS every responder from scratch. It does not trust
any time computed by the simulation: arrival and finish times are recomputed
from the path with the travel-time model.

Checks (an error makes the plan invalid):
    structure   every entry has responder_id, incident_id, path (list of
                node ids) and time (number)
    identity    responder and incident exist; an incident appears at most once
    capability  the responder has the capability the incident requires
    path        non-empty; starts where the responder is at that moment
                (start location, or the location of its previous incident);
                ends at the incident; every consecutive pair is a road
    roads       no road on the path is treated as blocked (P(Blocked) > threshold)
    online      the start time is not before the incident's timestamp
    overlap     the start time is not before the responder is free (its
                time_until_available, or the finish of its previous incident)

Warnings (allowed, reported):
    late        arrival after the deadline (A13 allows sending a late responder)

`allow_delay=True` turns the validator into an EVALUATOR used by the
experiments: a plan made with one road model is replayed under another.
A responder that is not free yet then simply leaves later, and a route over
a road that the other model considers blocked fails (the incident counts
as unhandled) instead of being an error.

Limitation: timing is checked against the expected travel times of our own
model; there is no ground truth for the real road state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.data import Scenario
from src.travel_time import TravelTimeModel

EPS = 1e-6


@dataclass
class ReplayedAssignment:
    responder_id: str
    incident_id: str
    path: list[str]
    planned_departure: float  # "time" in the plan
    departure: float  # = planned, or later when allow_delay shifts it
    arrival: float
    finish: float
    timestamp: float
    deadline: float
    priority: str

    @property
    def response_time(self) -> float:
        return self.arrival - self.timestamp

    @property
    def deadline_met(self) -> bool:
        return self.arrival <= self.deadline + EPS


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    replayed: list[ReplayedAssignment] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)  # incident ids whose route was unusable (allow_delay only)

    @property
    def valid(self) -> bool:
        return not self.errors


def load_plan(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _structure_ok(k: int, entry, report: ValidationReport) -> bool:
    if not isinstance(entry, dict):
        report.errors.append(f"entry {k}: not an object")
        return False
    missing = [key for key in ("responder_id", "incident_id", "path", "time") if key not in entry]
    if missing:
        report.errors.append(f"entry {k}: missing {missing}")
        return False
    if not isinstance(entry["path"], list) or not entry["path"] or not all(isinstance(n, str) for n in entry["path"]):
        report.errors.append(f"entry {k}: path must be a non-empty list of node ids")
        return False
    if not isinstance(entry["time"], (int, float)) or isinstance(entry["time"], bool):
        report.errors.append(f"entry {k}: time must be a number")
        return False
    return True


def validate_plan(
    plan: list[dict], scenario: Scenario, model: TravelTimeModel, allow_delay: bool = False
) -> ValidationReport:
    report = ValidationReport()
    responders = {r.id: r for r in scenario.responders}
    incidents = {i.id: i for i in scenario.incidents}
    graph = scenario.graph

    if not isinstance(plan, list):
        report.errors.append("plan must be a JSON list")
        return report

    # --- structure and identity -------------------------------------------------
    entries = []
    seen_incidents: set[str] = set()
    for k, entry in enumerate(plan):
        if not _structure_ok(k, entry, report):
            continue
        rid, iid = entry["responder_id"], entry["incident_id"]
        if rid not in responders:
            report.errors.append(f"entry {k}: unknown responder {rid}")
            continue
        if iid not in incidents:
            report.errors.append(f"entry {k}: unknown incident {iid}")
            continue
        if iid in seen_incidents:
            report.errors.append(f"entry {k}: incident {iid} assigned more than once")
            continue
        seen_incidents.add(iid)
        entries.append((k, entry))

    # --- replay every responder in time order --------------------------------
    for rid, r in responders.items():
        location = r.start_location
        free_at = 0.0 if r.available else r.time_until_available
        mine = sorted((e for e in entries if e[1]["responder_id"] == rid), key=lambda e: (e[1]["time"], e[0]))
        for k, entry in mine:
            inc = incidents[entry["incident_id"]]
            path, time = entry["path"], float(entry["time"])
            where = f"entry {k} ({rid} -> {inc.id})"

            if inc.required_capability not in r.capabilities:
                report.errors.append(f"{where}: {rid} lacks capability '{inc.required_capability}'")
            if path[0] != location:
                report.errors.append(f"{where}: path starts at {path[0]}, but {rid} is at {location}")
            if path[-1] != inc.location:
                report.errors.append(f"{where}: path ends at {path[-1]}, incident is at {inc.location}")
            if time < inc.timestamp - EPS:
                report.errors.append(f"{where}: starts at {time:g} before the incident is known ({inc.timestamp:g})")

            departure = time
            if time < free_at - EPS:
                if allow_delay:
                    departure = free_at
                else:
                    report.errors.append(f"{where}: starts at {time:g} but {rid} is busy until {free_at:g}")

            # Road by road: must exist, must not be treated as blocked.
            travel, usable = 0.0, True
            for a, b in zip(path, path[1:]):
                if (a, b) not in graph.roads:
                    report.errors.append(f"{where}: no road {a}->{b}")
                    usable = False
                    break
                cost = model.cost(graph.road(a, b))
                if cost is None:
                    if allow_delay:
                        report.failed.append(inc.id)
                    else:
                        report.errors.append(f"{where}: road {a}->{b} is treated as blocked")
                    usable = False
                    break
                travel += cost
            if not usable:
                continue  # responder state unchanged: it never reached the incident

            arrival = departure + travel
            finish = arrival + inc.time_required
            replayed = ReplayedAssignment(rid, inc.id, path, time, departure, arrival, finish,
                                          inc.timestamp, inc.deadline, inc.priority)
            report.replayed.append(replayed)
            if not replayed.deadline_met:
                report.warnings.append(f"{where}: arrives at {arrival:.1f}, after deadline {inc.deadline:g}")
            location, free_at = inc.location, finish

    return report
