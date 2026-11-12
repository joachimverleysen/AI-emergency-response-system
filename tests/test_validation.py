"""
Tests for step 7 (plan validation and statistics).

Run from the project root with:
    python -m unittest discover tests

Strategy: plans produced by the simulation must pass; then every check is
tested by breaking a valid plan in one specific way ("mutation tests").
"""

import copy
import json
import unittest
from pathlib import Path

from src.data import load_scenario
from src.simulation import simulate
from src.stats import compute_statistics, format_statistics
from src.travel_time import TravelTimeModel
from src.validation import validate_plan

SCENARIOS = Path(__file__).resolve().parent.parent / "src" / "Scenarios"
ALL = [load_scenario(f) for f in sorted(SCENARIOS.glob("Scenario*"))]
S1 = ALL[0]
MODEL1 = TravelTimeModel.from_scenario(S1)


def plan_of(scenario, model, allocator="exhaustive"):
    """Plan as it is written to plan.json, round-tripped through JSON."""
    result = simulate(scenario, model, allocator)
    return json.loads(json.dumps([a.to_json() for a in result.assignments]))


PLAN1 = plan_of(S1, MODEL1)


def entry(plan, incident_id):
    return next(e for e in plan if e["incident_id"] == incident_id)


class TestValidPlans(unittest.TestCase):
    def test_simulated_plans_are_valid(self):
        for s in ALL:
            model = TravelTimeModel.from_scenario(s)
            for allocator in ("greedy", "exhaustive"):
                with self.subTest(scenario=s.name, allocator=allocator):
                    report = validate_plan(plan_of(s, model, allocator), s, model)
                    self.assertEqual(report.errors, [])
                    self.assertEqual(len(report.replayed), len(s.incidents))

    def test_replay_matches_simulation(self):
        # The validator recomputes times on its own; they must agree with the simulation.
        result = simulate(S1, MODEL1, "exhaustive")
        report = validate_plan(PLAN1, S1, MODEL1)
        sim = {a.incident_id: a.arrival for a in result.assignments}
        for r in report.replayed:
            self.assertAlmostEqual(r.arrival, sim[r.incident_id])

    def test_late_arrival_is_a_warning_not_an_error(self):
        report = validate_plan(PLAN1, S1, MODEL1)
        self.assertTrue(report.valid)
        self.assertEqual(len(report.warnings), 3)  # I1, I6, I7 in Scenario 1


class TestMutations(unittest.TestCase):
    """Break the valid Scenario 1 plan in one way and expect one specific error."""

    def assertError(self, plan, fragment):
        report = validate_plan(plan, S1, MODEL1)
        self.assertFalse(report.valid)
        self.assertTrue(any(fragment in e for e in report.errors), report.errors)

    def mutated(self):
        return copy.deepcopy(PLAN1)

    def test_plan_not_a_list(self):
        self.assertError({"responder_id": "R1"}, "must be a JSON list")

    def test_missing_key(self):
        plan = self.mutated()
        del plan[0]["path"]
        self.assertError(plan, "missing")

    def test_time_not_a_number(self):
        plan = self.mutated()
        plan[0]["time"] = "0"
        self.assertError(plan, "time must be a number")

    def test_unknown_responder(self):
        plan = self.mutated()
        plan[0]["responder_id"] = "R99"
        self.assertError(plan, "unknown responder")

    def test_unknown_incident(self):
        plan = self.mutated()
        plan[0]["incident_id"] = "I99"
        self.assertError(plan, "unknown incident")

    def test_incident_twice(self):
        plan = self.mutated()
        plan.append(copy.deepcopy(plan[0]))
        self.assertError(plan, "more than once")

    def test_wrong_capability(self):
        plan = self.mutated()
        e = entry(plan, "I2")  # Fire suppression, done by fire truck R1
        e["responder_id"] = "R3"  # police unit
        self.assertError(plan, "lacks capability")

    def test_path_starts_elsewhere(self):
        plan = self.mutated()
        entry(plan, "I4")["path"] = ["N4", "N2"]  # R2 starts at H1
        self.assertError(plan, "path starts at")

    def test_path_ends_elsewhere(self):
        plan = self.mutated()
        entry(plan, "I4")["path"] = ["H1", "N4"]  # I4 is at N2
        self.assertError(plan, "path ends at")

    def test_missing_road(self):
        plan = self.mutated()
        entry(plan, "I4")["path"] = ["H1", "N4", "N1", "N2"]  # no road N4->N1
        self.assertError(plan, "no road")

    def test_blocked_road(self):
        plan = self.mutated()
        entry(plan, "I2")["path"] = ["N1", "N2"]  # N1->N2 has P(Blocked) = 0.527
        self.assertError(plan, "treated as blocked")

    def test_before_incident_known(self):
        plan = self.mutated()
        entry(plan, "I5")["time"] = 10.0  # I5 appears at t = 50
        self.assertError(plan, "before the incident is known")

    def test_responder_still_busy(self):
        plan = self.mutated()
        entry(plan, "I1")["time"] = 0.0  # R4 is busy until t = 10
        self.assertError(plan, "busy until")

    def test_rounded_down_time_is_caught(self):
        # Regression: plan.json used to round times to 2 decimals, which could
        # put a departure a fraction before the responder was free.
        plan = self.mutated()
        e = entry(plan, "I5")
        e["time"] = float(f"{e['time']:.2f}") - 0.001
        self.assertError(plan, "busy until")


class TestEvaluationMode(unittest.TestCase):
    def test_delay_instead_of_error(self):
        plan = copy.deepcopy(PLAN1)
        entry(plan, "I1")["time"] = 0.0  # R4 busy until 10
        report = validate_plan(plan, S1, MODEL1, allow_delay=True)
        self.assertTrue(report.valid)
        i1 = next(r for r in report.replayed if r.incident_id == "I1")
        self.assertEqual(i1.departure, 10.0)

    def test_blocked_route_fails_the_incident(self):
        # A plan made WITHOUT the Bayes net may drive over a road the full model
        # considers blocked: in evaluation mode that incident is not handled.
        plan = copy.deepcopy(PLAN1)
        entry(plan, "I2")["path"] = ["N1", "N2"]
        report = validate_plan(plan, S1, MODEL1, allow_delay=True)
        self.assertIn("I2", report.failed)
        self.assertNotIn("I2", {r.incident_id for r in report.replayed})


class TestStatistics(unittest.TestCase):
    def test_scenario1_numbers(self):
        stats = compute_statistics(S1, validate_plan(PLAN1, S1, MODEL1))
        self.assertEqual(stats.handled, 7)
        self.assertEqual(stats.responders_used, 4)
        self.assertEqual(stats.deadlines_met, 4)
        self.assertAlmostEqual(stats.pct_met_all, 400 / 7)
        self.assertEqual(stats.per_priority["Critical"], (2, 2))

    def test_average_is_mean_of_arrival_minus_timestamp(self):
        report = validate_plan(PLAN1, S1, MODEL1)
        stats = compute_statistics(S1, report)
        expected = sum(r.arrival - r.timestamp for r in report.replayed) / len(report.replayed)
        self.assertAlmostEqual(stats.average_response, expected)

    def test_unhandled_counts_as_missed(self):
        plan = [e for e in PLAN1 if e["incident_id"] != "I4"]  # I4 was on time
        stats = compute_statistics(S1, validate_plan(plan, S1, MODEL1))
        self.assertEqual(stats.unhandled_ids, ["I4"])
        self.assertEqual(stats.handled, 6)
        self.assertAlmostEqual(stats.pct_met_all, 300 / 7)  # 3 met out of ALL 7
        self.assertAlmostEqual(stats.pct_met_handled, 300 / 6)

    def test_empty_plan(self):
        stats = compute_statistics(S1, validate_plan([], S1, MODEL1))
        self.assertEqual(stats.handled, 0)
        self.assertIsNone(stats.average_response)
        self.assertEqual(stats.pct_met_all, 0.0)

    def test_required_lines(self):
        text = format_statistics(compute_statistics(S1, validate_plan(PLAN1, S1, MODEL1)))
        for line in ("7 incidents handled", "4 responders used", "Average response time: ", "Deadlines met: 57.1%"):
            self.assertIn(line, text)


if __name__ == "__main__":
    unittest.main()
