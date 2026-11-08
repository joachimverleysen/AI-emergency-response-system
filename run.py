"""
Entry point of the emergency response decision-support system.

Usage:
    python run.py src/Scenarios/Scenario1

Current state (step 2): load a scenario, run the sanity checks, show for
every incident the travel time of each capable responder under normal road
conditions (A*), and show the Bayesian belief about the status of every
reported road. Later steps will add traffic filtering, allocation and
finally write plan.json and statistics.txt.
"""

import argparse
import sys

from src.bayes import road_status_beliefs
from src.checks import check_scenario, summarize
from src.data import Scenario, load_scenario
from src.search import astar


def print_routing_overview(scenario: Scenario) -> None:
    """For each incident: A* travel time (normal conditions) of every capable
    responder from its start location. Only a demonstration of step 1; the
    real allocation comes in step 5."""
    print("\nRouting under normal conditions (A*, seconds from start location):")
    for inc in scenario.incidents:
        options = []
        for r in scenario.responders:
            if inc.required_capability not in r.capabilities:
                continue
            route = astar(scenario.graph, r.start_location, inc.location)
            if route is not None:
                options.append((route.cost, r.id, route.path))
        options.sort()
        best = options[0] if options else None
        line = f"  {inc.id:<4} {inc.priority:<8} at {inc.location:<4} needs {inc.required_capability:<20}"
        if best is None:
            print(line + " -> no capable responder can reach it")
        else:
            cost, rid, path = best
            others = ", ".join(f"{o[1]} {o[0]:.0f}s" for o in options[1:])
            print(line + f" -> fastest {rid} {cost:.1f}s via {'-'.join(path)}"
                  + (f" (others: {others})" if others else ""))


def print_road_beliefs(scenario: Scenario) -> None:
    """P(Status | weather, reports) for every road that has reports.
    Roads without reports all get the same prior belief (printed once)."""
    beliefs = road_status_beliefs(scenario)
    reported = {(r.from_node, r.to_node) for r in scenario.road_reports}
    print("\nRoad status beliefs (Bayesian network, P(Open) / P(Congested) / P(Blocked)):")
    for road in sorted(reported):
        d = beliefs[road]
        flag = "  -> treated as BLOCKED (P(Blocked) > 0.5)" if d["Blocked"] > 0.5 else ""
        print(f"  {road[0]:>3}->{road[1]:<4} {d['Open']:.3f} / {d['Congested']:.3f} / {d['Blocked']:.3f}{flag}")
    unreported = [r for r in scenario.graph.roads if r not in reported]
    if unreported:
        d = beliefs[unreported[0]]
        print(f"  {len(unreported)} roads without reports: "
              f"{d['Open']:.3f} / {d['Congested']:.3f} / {d['Blocked']:.3f} (prior, weather unknown)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Emergency response planner")
    parser.add_argument(
        "scenario", help="path to a scenario folder, e.g. src/Scenarios/Scenario1")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    print(summarize(scenario))

    errors, warnings = check_scenario(scenario)
    for w in warnings:
        print(f"  WARNING: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print("Input data is inconsistent; stopping.")
        return 1
    print("  input checks passed")

    print_routing_overview(scenario)
    print_road_beliefs(scenario)
    return 0


if __name__ == "__main__":
    sys.exit(main())
