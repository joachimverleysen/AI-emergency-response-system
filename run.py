"""
Entry point of the emergency response decision-support system.

Usage:
    python run.py src/Scenarios/Scenario1

Current state (step 0): load a scenario, run the sanity checks and print a
summary. Later steps will add routing, road-state inference, traffic
filtering, allocation and finally write plan.json and statistics.txt.
"""

import argparse
import sys

from src.checks import check_scenario, summarize
from src.data import load_scenario


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
