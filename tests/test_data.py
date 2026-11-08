"""
Tests for step 0 (data loading and input checks).

Run from the project root with:
    python -m unittest discover tests
"""

import unittest
from pathlib import Path

from src.checks import check_scenario
from src.data import load_scenario

SCENARIOS = Path(__file__).resolve().parent.parent / "src" / "Scenarios"


class TestLoading(unittest.TestCase):
    def setUp(self):
        self.s1 = load_scenario(SCENARIOS / "Scenario1")

    def test_counts_scenario1(self):
        # Hand-counted from the files of Scenario1.
        self.assertEqual(len(self.s1.graph.nodes), 5)
        self.assertEqual(len(self.s1.graph.roads), 14)
        self.assertEqual(len(self.s1.responders), 4)
        self.assertEqual(len(self.s1.incidents), 7)
        # blank lines must be skipped
        self.assertEqual(len(self.s1.road_reports), 18)

    def test_types_are_converted(self):
        road = self.s1.graph.road("H1", "N1")
        self.assertEqual(road.distance, 458.0)
        self.assertAlmostEqual(road.base_travel_time, 458 / 15)
        r1 = next(r for r in self.s1.responders if r.id == "R1")
        self.assertEqual(r1.capabilities, frozenset(
            {"Fire suppression", "Rescue"}))
        r4 = next(r for r in self.s1.responders if r.id == "R4")
        self.assertFalse(r4.available)
        self.assertEqual(r4.time_until_available, 10.0)

    def test_incidents_sorted_by_timestamp(self):
        times = [i.timestamp for i in self.s1.incidents]
        self.assertEqual(times, sorted(times))

    def test_traffic_sequences_ordered(self):
        seq = self.s1.traffic_observations[("H1", "N1")]
        self.assertEqual(seq[:4], ["Low", "Low", "Low", "Medium"])

    def test_report_cpt_indexed_by_source(self):
        cpt = self.s1.road_probabilities.report_given_status
        self.assertAlmostEqual(cpt["Police"]["Blocked"]["Blocked"], 0.90)


class TestChecks(unittest.TestCase):
    def test_all_scenarios_pass(self):
        for folder in sorted(SCENARIOS.glob("Scenario*")):
            with self.subTest(scenario=folder.name):
                errors, _ = check_scenario(load_scenario(folder))
                self.assertEqual(errors, [])

    def test_detects_bad_probability(self):
        s = load_scenario(SCENARIOS / "Scenario1")
        s.traffic_probabilities.initial["Low"] = 0.9  # now sums to 1.3
        errors, _ = check_scenario(s)
        self.assertTrue(any("HMM initial" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
