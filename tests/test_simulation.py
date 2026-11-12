"""
Tests for step 6 (online simulation).

Run from the project root with:
    python -m unittest discover tests

Runs with the greedy allocator and the exhaustive oracle (no OR-Tools needed);
the CSP run is skipped when OR-Tools is not installed.
"""

import importlib.util
import unittest
from pathlib import Path

from src.data import load_scenario
from src.search import path_cost
from src.simulation import simulate
from src.travel_time import TravelTimeModel
from tests.oracle import exhaustive_allocate, last_optimal_allocate

HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None
SCENARIOS = Path(__file__).resolve().parent.parent / "src" / "Scenarios"
ALL = [load_scenario(f) for f in sorted(SCENARIOS.glob("Scenario*"))]


class PlanInvariants:
    """Checks every simulation result must satisfy (mixed into the test classes)."""

    allocator = None

    def run_all(self):
        for s in ALL:
            model = TravelTimeModel.from_scenario(s)
            yield s, model, simulate(s, model, self.allocator)

    def test_invariants(self):
        for s, model, result in self.run_all():
            with self.subTest(scenario=s.name):
                responders = {r.id: r for r in s.responders}
                incidents = {i.id: i for i in s.incidents}
                ids = [a.incident_id for a in result.assignments]
                # Every incident handled at most once; handled + unhandled = all.
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(set(ids) | {i.id for i in result.unhandled}, set(incidents))

                # Replay each responder's assignments in time order.
                for rid, r in responders.items():
                    location = r.start_location
                    free_at = 0.0 if r.available else r.time_until_available
                    mine = sorted((a for a in result.assignments if a.responder_id == rid),
                                  key=lambda a: a.departure)
                    for a in mine:
                        inc = incidents[a.incident_id]
                        self.assertIn(inc.required_capability, r.capabilities)
                        self.assertEqual(a.path[0], location)  # starts where the responder is
                        self.assertEqual(a.path[-1], inc.location)
                        self.assertGreaterEqual(a.departure, inc.timestamp - 1e-9)  # online
                        self.assertGreaterEqual(a.departure, free_at - 1e-9)  # no overlap
                        cost = path_cost(s.graph, a.path, model.cost)
                        self.assertIsNotNone(cost)  # only usable roads
                        self.assertAlmostEqual(a.arrival, a.departure + cost)
                        self.assertAlmostEqual(a.finish, a.arrival + inc.time_required)
                        location, free_at = inc.location, a.finish

    def test_all_incidents_handled(self):
        # With the current road beliefs every incident has a capable responder
        # that can reach it (step 4), so nothing should stay unhandled.
        for s, _, result in self.run_all():
            self.assertEqual(result.unhandled, [], s.name)


class TestGreedySimulation(PlanInvariants, unittest.TestCase):
    allocator = "greedy"

    def test_busy_responder_waits(self):
        # Scenario 1: R4 is busy for 10 s, so it cannot leave before t = 10.
        s = ALL[0]
        result = simulate(s, TravelTimeModel.from_scenario(s), "greedy")
        for a in result.assignments:
            if a.responder_id == "R4":
                self.assertGreaterEqual(a.departure, 10.0)


class TestOracleSimulation(PlanInvariants, unittest.TestCase):
    allocator = staticmethod(exhaustive_allocate)

    def test_ties_change_the_outcome(self):
        # Scenario 3, t = 0: I4 (First aid) can go to R1 or R2, both at H1 with
        # the same arrival time, so both choices are optimal at that moment.
        # At t = 1140, I11 (Medical emergency) appears at the same node N8 and
        # only R1 or R3 can handle it. If R1 was sent to I4 it is already there;
        # if R2 was sent, R1 must still drive from H1 and arrives too late.
        # A planner that cannot see the future cannot know which tie is better,
        # so two OPTIMAL allocators can give different simulations.
        s = ALL[2]
        model = TravelTimeModel.from_scenario(s)
        first = simulate(s, model, exhaustive_allocate)
        last = simulate(s, model, last_optimal_allocate)
        self.assertEqual(sum(a.deadline_met for a in first.assignments), 13)
        self.assertEqual(sum(a.deadline_met for a in last.assignments), 12)


@unittest.skipUnless(HAS_ORTOOLS, "OR-Tools not installed")
class TestCSPSimulation(PlanInvariants, unittest.TestCase):
    allocator = "csp"


if __name__ == "__main__":
    unittest.main()
