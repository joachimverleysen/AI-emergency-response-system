"""
Entry point of the emergency response decision-support system.

Usage:
    python run.py src/Scenarios/Scenario1

    python run.py src/Scenarios/Scenario1 --allocator greedy --details

Pipeline: load a scenario and check it, build the travel-time model (Bayes
net + HMM), run the online simulation with the CSP allocator (OR-Tools), the
greedy baseline or the exhaustive reference, write plan.json, read it back,
validate it independently and write statistics.txt.
--details also prints the road beliefs, traffic beliefs and expected travel
times.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from src.checks import check_scenario, summarize
from src.data import Scenario, load_scenario
from src.hmm import viterbi
from src.search import astar, base_cost
from src.simulation import simulate
from src.stats import compute_statistics, format_statistics
from src.travel_time import TravelTimeModel
from src.validation import load_plan, validate_plan


def print_road_beliefs(scenario: Scenario, model: TravelTimeModel) -> None:
    """Bayes net: P(Status | evidence) for every road that has reports.
    Roads without reports all get the same prior belief (printed once)."""
    reported = {(r.from_node, r.to_node) for r in scenario.road_reports}
    print("\nRoad status (Bayes net, P(Open) / P(Congested) / P(Blocked)):")
    for road in sorted(reported):
        d = model.status_beliefs[road]
        flag = f"  -> BLOCKED (P(Blocked) > {model.blocked_threshold})" if model.is_blocked(
            scenario.graph.road(*road)) else ""
        print(f"  {road[0]:>3}->{road[1]:<4} {d['Open']:.3f} / {d['Congested']:.3f} / {d['Blocked']:.3f}{flag}")
    unreported = [r for r in scenario.graph.roads if r not in reported]
    if unreported:
        d = model.status_beliefs[unreported[0]]
        print(f"  {len(unreported)} roads without reports: "
              f"{d['Open']:.3f} / {d['Congested']:.3f} / {d['Blocked']:.3f} (Weather hidden)")


def print_traffic_beliefs(scenario: Scenario, model: TravelTimeModel) -> None:
    """HMM: belief after the last observation, the resulting traffic factor,
    and the Viterbi sequence (analysis only)."""
    print("\nTraffic (HMM, B(Low) / B(Medium) / B(High) after the last observation):")
    hmm = scenario.traffic_probabilities
    for road, obs in sorted(scenario.traffic_observations.items()):
        b = model.traffic_beliefs[road]
        factor = model.traffic_factor(scenario.graph.road(*road))
        seq = "".join(o[0] for o in obs)
        vit = "".join(x[0] for x in viterbi(obs, hmm))
        print(f"  {road[0]:>3}->{road[1]:<4} {b['Low']:.2f} / {b['Medium']:.2f} / {b['High']:.2f}"
              f"  factor x{factor:.3f}  observed {seq:<10} Viterbi {vit}")


def print_routing_overview(scenario: Scenario, model: TravelTimeModel) -> None:
    """For each incident: expected travel time (A* on model costs) of every
    capable responder from its start location, next to the normal time.
    Only a demonstration; the real allocation comes in step 5."""
    print("\nExpected travel time per incident (A*, seconds from start location; normal time in brackets):")
    for inc in scenario.incidents:
        options, unreachable = [], []
        for r in scenario.responders:
            if inc.required_capability not in r.capabilities:
                continue
            route = astar(scenario.graph, r.start_location, inc.location, model.cost)
            if route is None:
                unreachable.append(r.id)
                continue
            normal = astar(scenario.graph, r.start_location, inc.location, base_cost)
            options.append((route.cost, r.id, route.path, normal.cost))
        options.sort()
        line = f"  {inc.id:<4} {inc.priority:<8} at {inc.location:<4} needs {inc.required_capability:<20}"
        if not options:
            print(line + " -> NO capable responder can reach it")
            continue
        cost, rid, path, normal = options[0]
        others = ", ".join(f"{o[1]} {o[0]:.0f}s" for o in options[1:])
        print(line + f" -> fastest {rid} {cost:.1f}s ({normal:.1f}s) via {'-'.join(path)}"
              + (f"; others: {others}" if others else "")
              + (f"; cut off: {', '.join(unreachable)}" if unreachable else ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Emergency response planner")
    parser.add_argument(
        "scenario", help="path to a scenario folder, e.g. src/Scenarios/Scenario1")
    parser.add_argument("--allocator", choices=["csp", "greedy", "exhaustive"], default="csp",
                        help="csp = OR-Tools constraint model (default), greedy = baseline, "
                             "exhaustive = reference search (small scenarios only)")
    parser.add_argument("--out", help="output folder (default: output/<scenario name>)")
    parser.add_argument("--details", action="store_true",
                        help="also print road beliefs, traffic beliefs and travel times")
    args = parser.parse_args()

    if args.allocator == "csp" and importlib.util.find_spec("ortools") is None:
        print("OR-Tools is not installed: run 'pip install ortools' or use --allocator greedy")
        return 2

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

    model = TravelTimeModel.from_scenario(scenario)
    if args.details:
        print_road_beliefs(scenario, model)
        print_traffic_beliefs(scenario, model)
        print_routing_overview(scenario, model)

    result = simulate(scenario, model, args.allocator)
    print(f"\nOnline simulation ({args.allocator} allocator, {result.decisions} decisions):")
    for line in result.log:
        print("  " + line)
    if result.unhandled:
        print(f"  unhandled: {', '.join(i.id for i in result.unhandled)}")

    out = Path(args.out) if args.out else Path("output") / scenario.name
    out.mkdir(parents=True, exist_ok=True)
    plan = [a.to_json() for a in sorted(result.assignments, key=lambda a: (a.departure, a.responder_id))]
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out / 'plan.json'} ({len(plan)} assignments)")

    # Validate the plan as written to disk, independently of the simulation.
    report = validate_plan(load_plan(out / "plan.json"), scenario, model)
    for e in report.errors:
        print(f"  INVALID: {e}")
    print(f"Plan validation: {'OK' if report.valid else 'FAILED'} "
          f"({len(report.errors)} errors, {len(report.warnings)} late arrivals)")

    stats = compute_statistics(scenario, report)
    notes = [f"Scenario: {scenario.name}; allocator: {args.allocator}; "
             f"blocked threshold: {model.blocked_threshold}; plan valid: {report.valid}"]
    text = format_statistics(stats, notes)
    (out / "statistics.txt").write_text(text, encoding="utf-8")
    print(f"Wrote {out / 'statistics.txt'}\n")
    print(text.split("\nDetails")[0])
    return 0 if report.valid else 1


if __name__ == "__main__":
    sys.exit(main())
