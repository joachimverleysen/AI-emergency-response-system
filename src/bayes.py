"""
Step 2 - Bayesian network for the (hidden) status of each road.

For one directed road the network looks like this:

            Weather
               |
               v
            Status                 (Open / Congested / Blocked, hidden)
          /    |    \\
         v     v     v
    Report_1 Report_2 ... Report_n (one node per received report)

CPTs (from road_probabilities.json):
    P(Weather)                      weather_prior
    P(Status | Weather)             road_status_given_weather
    P(Report_k | Status)            citizen_report / police_report / sensor_report,
                                    depending on the source of report k

Query:      P(Status | evidence)
Evidence:   the reported statuses, and the weather if it was reported for
            this road. Weather is local to a road (scenario README), so for
            a road without reports the weather is unknown and is summed out.

The module has two parts:
  1. A small, general discrete Bayesian network with the two exact inference
     algorithms from the course: inference by enumeration and variable
     elimination. Both give the same answer; the tests check this.
  2. Road-specific code that builds the network for one road and returns
     P(Status | evidence) for every road in the city.

Key assumption (A5 in the report): reports are conditionally independent
given the true status. This is encoded by the structure: the only parent of
every Report node is Status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from src.data import ROAD_STATUSES, WEATHER_VALUES, RoadProbabilities, RoadReport, Scenario

Assignment = dict[str, str]  # variable name -> value


# =============================================================================
# Part 1 - a general discrete Bayesian network
# =============================================================================
@dataclass
class Variable:
    name: str
    domain: tuple[str, ...]
    parents: tuple[str, ...]
    # cpt[parent_values][value] = P(variable = value | parents = parent_values)
    # parent_values is a tuple in the same order as `parents` (empty tuple for a root).
    cpt: dict[tuple[str, ...], dict[str, float]]

    def prob(self, value: str, assignment: Assignment) -> float:
        parent_values = tuple(assignment[p] for p in self.parents)
        return self.cpt[parent_values][value]


@dataclass
class BayesNet:
    # Variables must be added in topological order (parents before children).
    variables: list[Variable] = field(default_factory=list)

    def add(self, var: Variable) -> None:
        known = {v.name for v in self.variables}
        assert all(p in known for p in var.parents), f"add parents of {var.name} first"
        self.variables.append(var)

    def variable(self, name: str) -> Variable:
        return next(v for v in self.variables if v.name == name)


def normalize(dist: dict[str, float]) -> dict[str, float]:
    total = sum(dist.values())
    if total == 0:
        raise ValueError("evidence has probability 0")
    return {k: v / total for k, v in dist.items()}


# ---------------------------------------------------------------------------
# Inference by enumeration (CS188 note 6)
#
#   P(Q | e) = alpha * sum_{h} P(Q, e, h)
#
# Take the rows of the joint consistent with the evidence e, sum out the
# hidden variables h, and normalize. Each row of the joint comes from the
# chain rule for Bayes nets:
#   P(x1, ..., xn) = prod_i P(xi | parents(Xi))
# ---------------------------------------------------------------------------
def enumeration_ask(query: str, evidence: Assignment, bn: BayesNet) -> dict[str, float]:
    result = {}
    for value in bn.variable(query).domain:
        result[value] = _enumerate_all(bn.variables, {**evidence, query: value})
    return normalize(result)


def _enumerate_all(variables: list[Variable], assignment: Assignment) -> float:
    """Sum of the joint probability over all values of the unassigned variables."""
    if not variables:
        return 1.0
    first, rest = variables[0], variables[1:]
    if first.name in assignment:
        # Observed (or query) variable: multiply by its CPT entry and continue.
        return first.prob(assignment[first.name], assignment) * _enumerate_all(rest, assignment)
    # Hidden variable: sum over all its values.
    return sum(
        first.prob(value, assignment) * _enumerate_all(rest, {**assignment, first.name: value})
        for value in first.domain
    )


# ---------------------------------------------------------------------------
# Variable elimination (CS188 note 6)
#
# Same answer as enumeration, but works with factors (unnormalized tables).
# To eliminate a hidden variable X:  1. join all factors involving X,
#                                    2. sum out X.
# ---------------------------------------------------------------------------
@dataclass
class Factor:
    variables: tuple[str, ...]
    table: dict[tuple[str, ...], float]  # values (in order of `variables`) -> number


def _make_factor(var: Variable, evidence: Assignment, bn: BayesNet) -> Factor:
    """CPT of `var` as a factor, restricted to the evidence.

    Observed variables are fixed to their value and disappear from the factor.
    """
    scope = [v for v in (*var.parents, var.name) if v not in evidence]
    domains = [bn.variable(v).domain for v in scope]
    table = {}
    for values in product(*domains):
        assignment = {**evidence, **dict(zip(scope, values))}
        table[values] = var.prob(assignment[var.name], assignment)
    return Factor(tuple(scope), table)


def _join(f1: Factor, f2: Factor, bn: BayesNet) -> Factor:
    """Join (pointwise multiply) two factors."""
    scope = list(f1.variables) + [v for v in f2.variables if v not in f1.variables]
    domains = [bn.variable(v).domain for v in scope]
    table = {}
    for values in product(*domains):
        a = dict(zip(scope, values))
        table[values] = (
            f1.table[tuple(a[v] for v in f1.variables)] * f2.table[tuple(a[v] for v in f2.variables)]
        )
    return Factor(tuple(scope), table)


def _sum_out(var: str, factors: list[Factor], bn: BayesNet) -> list[Factor]:
    """Join all factors that involve `var`, then sum `var` out."""
    with_var = [f for f in factors if var in f.variables]
    without_var = [f for f in factors if var not in f.variables]
    if not with_var:
        return factors
    joined = with_var[0]
    for f in with_var[1:]:
        joined = _join(joined, f, bn)
    i = joined.variables.index(var)
    scope = joined.variables[:i] + joined.variables[i + 1:]
    table: dict[tuple[str, ...], float] = {}
    for values, p in joined.table.items():
        key = values[:i] + values[i + 1:]
        table[key] = table.get(key, 0.0) + p
    return without_var + [Factor(scope, table)]


def elimination_ask(query: str, evidence: Assignment, bn: BayesNet) -> dict[str, float]:
    factors: list[Factor] = []
    # Process variables from the leaves upward (reverse topological order).
    for var in reversed(bn.variables):
        factors.append(_make_factor(var, evidence, bn))
        if var.name != query and var.name not in evidence:  # hidden variable
            factors = _sum_out(var.name, factors, bn)
    # Only factors over the query variable (or constants) remain.
    result = factors[0]
    for f in factors[1:]:
        result = _join(result, f, bn)
    return normalize({value: result.table[(value,)] for value in bn.variable(query).domain})


# =============================================================================
# Part 2 - the road-status network
# =============================================================================
def build_road_network(reports: list[RoadReport], probs: RoadProbabilities) -> tuple[BayesNet, Assignment]:
    """Network + evidence for ONE road, given the reports about that road."""
    bn = BayesNet()
    bn.add(Variable("Weather", WEATHER_VALUES, (), {(): probs.weather_prior}))
    bn.add(Variable(
        "Status", ROAD_STATUSES, ("Weather",),
        {(w,): probs.status_given_weather[w] for w in WEATHER_VALUES},
    ))
    evidence: Assignment = {}
    for k, rep in enumerate(reports, start=1):
        name = f"Report{k}_{rep.source}"
        cpt = probs.report_given_status[rep.source]  # P(reported | true status)
        bn.add(Variable(name, ROAD_STATUSES, ("Status",), {(s,): cpt[s] for s in ROAD_STATUSES}))
        evidence[name] = rep.reported_status
    weathers = {rep.weather for rep in reports}
    if len(weathers) == 1:  # weather of this road is known
        evidence["Weather"] = weathers.pop()
    return bn, evidence


def road_status_posterior(
    reports: list[RoadReport], probs: RoadProbabilities, method: str = "enumeration"
) -> dict[str, float]:
    """P(Status | evidence) for one road."""
    bn, evidence = build_road_network(reports, probs)
    ask = enumeration_ask if method == "enumeration" else elimination_ask
    return ask("Status", evidence, bn)


def road_status_beliefs(scenario: Scenario, method: str = "enumeration") -> dict[tuple[str, str], dict[str, float]]:
    """P(Status | evidence) for EVERY directed road of the scenario.

    Roads without reports get the prior P(Status) = sum_w P(w) P(Status | w).
    """
    reports_per_road: dict[tuple[str, str], list[RoadReport]] = {}
    for rep in scenario.road_reports:
        reports_per_road.setdefault((rep.from_node, rep.to_node), []).append(rep)
    return {
        road: road_status_posterior(reports_per_road.get(road, []), scenario.road_probabilities, method)
        for road in scenario.graph.roads
    }
