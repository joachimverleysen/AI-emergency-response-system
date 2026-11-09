"""
Tests for step 5 (allocation).

Run from the project root with:
    python -m unittest discover tests

The CSP tests need OR-Tools (pip install ortools); without it they are skipped.
"""

import importlib.util
import unittest
from pathlib import Path

from src.allocation import (
    Option, ResponderState, build_options, csp_allocate, greedy_allocate, total_cost,
)
from src.data import Incident, Responder, load_scenario
from src.simulation import make_route_function, simulate
from src.travel_time import TravelTimeModel
from tests.oracle import exhaustive_allocate

HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None
SCENARIOS = Path(__file__).resolve().parent.parent / "src" / "Scenarios"
ALL = [load_scenario(f) for f in sorted(SCENARIOS.glob("Scenario*"))]


def incident(id, priority="High", deadline=100.0, timestamp=0.0):
    return Incident(id, "X", priority, deadline, "Rescue", timestamp, 10.0)


def option(r, i, arrival, deadline=100.0):
    return Option(r, i, 0.0, arrival, ("A", "X"), late=arrival > deadline, response_time=arrival)


# A conflict where greedy fails:
#   A (Critical) can be served by R1 in 10 s or by R2 in 12 s.
#   B (High) can only be served by R1, in 20 s.
# Greedy handles A first and takes R1, so B gets nobody.
# Optimal: A <- R2, B <- R1.
CONFLICT_PENDING = [incident("A", "Critical"), incident("B", "High")]
CONFLICT_OPTIONS = [option("R1", "A", 10), option("R2", "A", 12), option("R1", "B", 20)]

# Deadlines dominate response time:
#   R1 -> A 10 s, R2 -> A 100 s; R1 -> B 44 s (on time), R2 -> B 46 s (late, deadline 45).
#   Fastest total: R1->A + R2->B = 56 s, but B is late.
#   Deadline-first: R2->A + R1->B = 144 s, nobody late.  <- must be chosen
DEADLINE_PENDING = [incident("A", "High", deadline=200), incident("B", "High", deadline=45)]
DEADLINE_OPTIONS = [option("R1", "A", 10, 200), option("R2", "A", 100, 200),
                    option("R1", "B", 44, 45), option("R2", "B", 46, 45)]


def decision_moments():
    """(pending, options) at every decision of every scenario's simulation."""
    moments = []

    def recorder(pending, options):
        moments.append((list(pending), list(options)))
        return exhaustive_allocate(pending, options)

    for s in ALL:
        simulate(s, TravelTimeModel.from_scenario(s), recorder)
    return moments


class TestOptions(unittest.TestCase):
    def test_capability_and_reachability(self):
        s = ALL[0]
        model = TravelTimeModel.from_scenario(s)
        states = [ResponderState(r, r.start_location, 0.0) for r in s.responders]
        options = build_options(0.0, s.incidents, states, make_route_function(s, model))
        by_id = {r.id: r for r in s.responders}
        inc = {i.id: i for i in s.incidents}
        for o in options:
            self.assertIn(inc[o.incident_id].required_capability, by_id[o.responder_id].capabilities)
            self.assertEqual(o.path[0], by_id[o.responder_id].start_location)
            self.assertEqual(o.path[-1], inc[o.incident_id].location)

    def test_busy_responder_departs_when_free(self):
        r = Responder("R", "Ambulance", "N1", frozenset({"Rescue"}), False, 30.0)
        states = [ResponderState(r, "N1", 30.0)]
        s = ALL[0]
        opts = build_options(0.0, [Incident("I", "N1", "High", 100, "Rescue", 0, 5)], states,
                             make_route_function(s, TravelTimeModel.from_scenario(s)))
        self.assertEqual(opts[0].departure, 30.0)


class TestGreedyAndOracle(unittest.TestCase):
    def test_greedy_fails_on_conflict(self):
        greedy = greedy_allocate(CONFLICT_PENDING, CONFLICT_OPTIONS)
        self.assertEqual(set(greedy), {"A"})  # B left without responder

    def test_oracle_solves_conflict(self):
        best = exhaustive_allocate(CONFLICT_PENDING, CONFLICT_OPTIONS)
        self.assertEqual({i: o.responder_id for i, o in best.items()}, {"A": "R2", "B": "R1"})

    def test_oracle_prefers_meeting_deadlines(self):
        best = exhaustive_allocate(DEADLINE_PENDING, DEADLINE_OPTIONS)
        self.assertEqual({i: o.responder_id for i, o in best.items()}, {"A": "R2", "B": "R1"})


@unittest.skipUnless(HAS_ORTOOLS, "OR-Tools not installed")
class TestCSP(unittest.TestCase):
    def test_conflict(self):
        best = csp_allocate(CONFLICT_PENDING, CONFLICT_OPTIONS)
        self.assertEqual({i: o.responder_id for i, o in best.items()}, {"A": "R2", "B": "R1"})

    def test_deadlines_first(self):
        best = csp_allocate(DEADLINE_PENDING, DEADLINE_OPTIONS)
        self.assertEqual({i: o.responder_id for i, o in best.items()}, {"A": "R2", "B": "R1"})

    def test_no_options(self):
        self.assertEqual(csp_allocate(CONFLICT_PENDING, []), {})

    def test_matches_exhaustive_search_on_every_decision(self):
        # The CSP must reach the optimal objective value at every decision
        # moment that occurs in the three scenarios.
        for k, (pending, options) in enumerate(decision_moments()):
            with self.subTest(decision=k):
                csp = csp_allocate(pending, options)
                oracle = exhaustive_allocate(pending, options)
                self.assertEqual(total_cost(csp, pending), total_cost(oracle, pending))
                # Constraint check: one incident per responder, one responder per incident.
                responders = [o.responder_id for o in csp.values()]
                self.assertEqual(len(responders), len(set(responders)))


if __name__ == "__main__":
    unittest.main()
