"""
Step 0 - Sanity checks on the input data.

Before building anything on top of the data we verify that it is consistent.
Each check returns human-readable messages, split into:

    errors   - the data is inconsistent; later steps would crash or be wrong
    warnings - not fatal, but it influences a design choice (e.g. A* heuristic)

These checks also serve as documentation of what we ASSUME about the input.
"""

from __future__ import annotations

import math
from collections import deque

from src.data import (
    PRIORITIES,
    REPORT_SOURCES,
    ROAD_STATUSES,
    TRAFFIC_LEVELS,
    WEATHER_VALUES,
    Scenario,
)

TOLERANCE = 1e-6  # allowed rounding error when probabilities should sum to 1


def _check_distribution(name: str, dist: dict[str, float], domain: tuple[str, ...], errors: list[str]) -> None:
    """A probability distribution must cover the domain, be >= 0 and sum to 1."""
    if set(dist) != set(domain):
        errors.append(
            f"{name}: keys {sorted(dist)} differ from expected {sorted(domain)}")
    if any(p < 0 for p in dist.values()):
        errors.append(f"{name}: negative probability")
    if abs(sum(dist.values()) - 1.0) > TOLERANCE:
        errors.append(f"{name}: sums to {sum(dist.values()):.6f} instead of 1")


def reachable_from(scenario: Scenario, start: str) -> set[str]:
    """Breadth-first search: all nodes reachable from `start` via directed roads."""
    seen = {start}
    frontier = deque([start])
    while frontier:
        node = frontier.popleft()
        for nxt in scenario.graph.neighbours(node):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def check_scenario(scenario: Scenario) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    g = scenario.graph

    # --- Graph -------------------------------------------------------------
    for (frm, to), road in g.roads.items():
        if frm not in g.nodes or to not in g.nodes:
            errors.append(f"road {frm}->{to} uses an unknown node")
            continue
        if road.distance <= 0 or road.speed <= 0:
            errors.append(
                f"road {frm}->{to} has non-positive distance or speed")
        # A* will use straight-line distance as a lower bound on road distance.
        # That is only valid (admissible) if no road is shorter than the straight line.
        a, b = g.nodes[frm], g.nodes[to]
        straight = math.hypot(a.x - b.x, a.y - b.y)
        if road.distance < straight - TOLERANCE:
            warnings.append(
                f"road {frm}->{to}: distance {road.distance} < straight line {straight:.1f}; "
                "Euclidean A* heuristic would NOT be admissible"
            )

    # Every node should be able to reach every other node (strong connectivity),
    # otherwise some incidents are unreachable even with all roads open.
    for node in g.nodes:
        missing = set(g.nodes) - reachable_from(scenario, node)
        if missing:
            warnings.append(
                f"from {node} these nodes are unreachable: {sorted(missing)}")

    # --- Responders --------------------------------------------------------
    ids = [r.id for r in scenario.responders]
    if len(ids) != len(set(ids)):
        errors.append("duplicate responder ids")
    for r in scenario.responders:
        if r.start_location not in g.nodes:
            errors.append(
                f"responder {r.id} starts at unknown node {r.start_location}")
        if r.available and r.time_until_available != 0:
            warnings.append(
                f"responder {r.id} is Available but has time_until_available={r.time_until_available}")
        if not r.available and r.time_until_available <= 0:
            warnings.append(
                f"responder {r.id} is Busy but time_until_available <= 0")

    # --- Incidents ---------------------------------------------------------
    ids = [i.id for i in scenario.incidents]
    if len(ids) != len(set(ids)):
        errors.append("duplicate incident ids")
    all_capabilities = set().union(*(r.capabilities for r in scenario.responders))
    for i in scenario.incidents:
        if i.location not in g.nodes:
            errors.append(f"incident {i.id} at unknown node {i.location}")
        if i.priority not in PRIORITIES:
            errors.append(f"incident {i.id} has unknown priority {i.priority}")
        if i.deadline < i.timestamp:
            errors.append(
                f"incident {i.id}: deadline {i.deadline} before timestamp {i.timestamp}")
        if i.time_required < 0:
            errors.append(f"incident {i.id}: negative time_required")
        if i.required_capability not in all_capabilities:
            warnings.append(
                f"incident {i.id}: no responder has capability '{i.required_capability}'")

    # --- Road reports ------------------------------------------------------
    for rep in scenario.road_reports:
        where = f"report {rep.from_node}->{rep.to_node} ({rep.source})"
        if (rep.from_node, rep.to_node) not in g.roads:
            errors.append(f"{where}: road does not exist")
        if rep.weather not in WEATHER_VALUES:
            errors.append(f"{where}: unknown weather {rep.weather}")
        if rep.source not in REPORT_SOURCES:
            errors.append(f"{where}: unknown source {rep.source}")
        if rep.reported_status not in ROAD_STATUSES:
            errors.append(f"{where}: unknown status {rep.reported_status}")
    # One road should have a single (local) weather value.
    weather_per_road: dict[tuple[str, str], set[str]] = {}
    for rep in scenario.road_reports:
        weather_per_road.setdefault(
            (rep.from_node, rep.to_node), set()).add(rep.weather)
    for road, weathers in weather_per_road.items():
        if len(weathers) > 1:
            warnings.append(
                f"road {road[0]}->{road[1]} has conflicting weather values {sorted(weathers)}")

    # --- CPTs ---------------------------------------------------------------
    rp = scenario.road_probabilities
    _check_distribution("P(Weather)", rp.weather_prior, WEATHER_VALUES, errors)
    for w in WEATHER_VALUES:
        _check_distribution(f"P(Status | {w})", rp.status_given_weather.get(
            w, {}), ROAD_STATUSES, errors)
    for src in REPORT_SOURCES:
        for s in ROAD_STATUSES:
            _check_distribution(
                f"P({src} report | {s})", rp.report_given_status[src].get(
                    s, {}), ROAD_STATUSES, errors
            )

    # --- HMM ----------------------------------------------------------------
    tp = scenario.traffic_probabilities
    _check_distribution("HMM initial", tp.initial, TRAFFIC_LEVELS, errors)
    for lvl in TRAFFIC_LEVELS:
        _check_distribution(f"HMM transition from {lvl}", tp.transition.get(
            lvl, {}), TRAFFIC_LEVELS, errors)
        _check_distribution(f"HMM emission from {lvl}", tp.emission.get(
            lvl, {}), TRAFFIC_LEVELS, errors)
    for (frm, to), seq in scenario.traffic_observations.items():
        if (frm, to) not in g.roads:
            errors.append(f"traffic observations for unknown road {frm}->{to}")
        bad = [o for o in seq if o not in TRAFFIC_LEVELS]
        if bad:
            errors.append(
                f"traffic observations {frm}->{to}: unknown values {bad}")

    return errors, warnings


def summarize(scenario: Scenario) -> str:
    """Short overview of a scenario, printed by run.py."""
    g = scenario.graph
    busy = [r.id for r in scenario.responders if not r.available]
    lines = [
        f"Scenario {scenario.name}",
        f"  nodes: {len(g.nodes)}   directed roads: {len(g.roads)}",
        f"  responders: {len(scenario.responders)}  (busy at t=0: {busy or 'none'})",
        f"  incidents: {len(scenario.incidents)}  (timestamps "
        f"{scenario.incidents[0].timestamp:g}-{scenario.incidents[-1].timestamp:g} s)",
        f"  road reports: {len(scenario.road_reports)} on "
        f"{len({(r.from_node, r.to_node) for r in scenario.road_reports})} roads",
        f"  roads with traffic history: {len(scenario.traffic_observations)}",
    ]
    return "\n".join(lines)
