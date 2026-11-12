"""
Step 8 - Experiments for the report.

Usage:
    python experiments.py            # writes output/experiments.md and prints it

Every plan is evaluated with the validator in evaluation mode
(validate_plan(..., allow_delay=True)) under a REFERENCE model: the full
model (Bayes net + HMM, threshold 0.5). Without ground truth this is our best
belief about the roads. Evaluating every plan under the same reference makes
plans made with different assumptions comparable: a plan made with a
"normal roads" model is re-timed with the expected travel times of the
reference, and a route over a road the reference treats as blocked fails.

E1  Allocators: greedy, exhaustive (optimal per decision), CSP (if OR-Tools)
E2  Value of the Bayes net and the HMM: plan with less information
E3  Blocking threshold: speed versus risk of driving into a blocked road
E4  Traffic factors: how much the chosen factors matter
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

from src.data import load_scenario
from src.simulation import simulate
from src.stats import compute_statistics
from src.travel_time import TravelTimeModel
from src.validation import validate_plan

SCENARIOS = sorted(Path("src/Scenarios").glob("Scenario*"))
HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None


def run(scenario, planning_model, reference_model, allocator="exhaustive"):
    """Plan with `planning_model`, evaluate under `reference_model`."""
    result = simulate(scenario, planning_model, allocator)
    plan = [a.to_json() for a in result.assignments]
    report = validate_plan(plan, scenario, reference_model, allow_delay=True)
    return result, plan, report, compute_statistics(scenario, report)


def route_risk(plan, scenario, reference_model) -> float:
    """Mean over assignments of P(at least one road on the path is blocked),
    with the Bayes net beliefs and roads assumed independent:
        P(route blocked) = 1 - prod_r (1 - P(Blocked_r))"""
    risks = []
    for e in plan:
        p_open = math.prod(1 - reference_model.status_beliefs[(a, b)]["Blocked"]
                           for a, b in zip(e["path"], e["path"][1:]))
        risks.append(1 - p_open)
    return sum(risks) / len(risks) if risks else 0.0


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def fmt(stats):
    avg = f"{stats.average_response:.1f}" if stats.average_response is not None else "n/a"
    return [f"{stats.handled}/{stats.total_incidents}", f"{stats.deadlines_met}/{stats.total_incidents}",
            f"{stats.pct_met_all:.1f}%", avg]


def main() -> None:
    out: list[str] = ["# Experiments", ""]
    scenarios = [load_scenario(f) for f in SCENARIOS]
    reference = {s.name: TravelTimeModel.from_scenario(s) for s in scenarios}

    # E1 ------------------------------------------------------------------------
    out += ["## E1 - Allocators", "",
            row(["Scenario", "Allocator", "Handled", "Deadlines met", "Met %", "Avg response (s)"]),
            row(["---"] * 6)]
    allocators = ["greedy", "exhaustive"] + (["csp"] if HAS_ORTOOLS else [])
    for s in scenarios:
        for alg in allocators:
            *_, stats = run(s, reference[s.name], reference[s.name], alg)
            out.append(row([s.name, alg] + fmt(stats)))
    if not HAS_ORTOOLS:
        out += ["", "OR-Tools not installed: CSP rows omitted."]

    # E2 ------------------------------------------------------------------------
    variants = {
        "normal roads (no BN, no HMM)": dict(use_status=False, use_traffic=False),
        "HMM only": dict(use_status=False),
        "Bayes net only": dict(use_traffic=False),
        "Bayes net + HMM (full)": {},
    }
    out += ["", "## E2 - Value of the Bayes net and the HMM (evaluated under the full model)", "",
            row(["Scenario", "Planning model", "Handled", "Deadlines met", "Met %", "Avg response (s)",
                 "Failed (blocked road)"]),
            row(["---"] * 7)]
    for s in scenarios:
        for name, opts in variants.items():
            _, _, report, stats = run(s, TravelTimeModel.from_scenario(s, **opts), reference[s.name])
            out.append(row([s.name, name] + fmt(stats) + [", ".join(report.failed) or "-"]))

    # E3 ------------------------------------------------------------------------
    out += ["", "## E3 - Blocking threshold (each plan evaluated under its OWN model)", "",
            row(["Scenario", "Threshold", "Roads treated blocked", "Deadlines met", "Met %", "Avg response (s)",
                 "Mean route risk"]),
            row(["---"] * 7)]
    for s in scenarios:
        for th in (0.3, 0.5, 0.55, 0.9):
            model = TravelTimeModel.from_scenario(s, blocked_threshold=th)
            _, plan, _, stats = run(s, model, model)
            blocked = sum(model.is_blocked(r) for r in s.graph.roads.values())
            out.append(row([s.name, th, blocked] + fmt(stats)[1:] + [f"{route_risk(plan, s, model):.3f}"]))

    # E4 ------------------------------------------------------------------------
    factor_sets = {"none (1/1/1)": (1.0, 1.0, 1.0), "mild (1/1.1/1.2)": (1.0, 1.1, 1.2),
                   "chosen (1/1.25/1.5)": (1.0, 1.25, 1.5), "strong (1/1.5/2)": (1.0, 1.5, 2.0)}
    out += ["", "## E4 - Traffic factors Low/Medium/High (evaluated under the full model)", "",
            row(["Scenario", "Factors", "Handled", "Deadlines met", "Met %", "Avg response (s)"]),
            row(["---"] * 6)]
    for s in scenarios:
        for name, (lo, me, hi) in factor_sets.items():
            model = TravelTimeModel.from_scenario(s, traffic_factors={"Low": lo, "Medium": me, "High": hi})
            *_, stats = run(s, model, reference[s.name])
            out.append(row([s.name, name] + fmt(stats)))

    text = "\n".join(out) + "\n"
    Path("output").mkdir(exist_ok=True)
    Path("output/experiments.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
