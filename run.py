"""
Entry point of the emergency response decision-support system.

Usage:
    python run.py src/Scenarios/Scenario1

Current state (step 1): load a scenario, run the sanity checks, and show for
every incident the travel time of each capable responder under normal road
conditions (A*). Later steps will add road-state inference, traffic
filtering, allocation and finally write plan.json and statistics.txt.
"""

import argparse
import sys

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
