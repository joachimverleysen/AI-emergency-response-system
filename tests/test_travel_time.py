"""
Tests for step 4 (travel-time model combining Bayes net and HMM).

Run from the project root with:
    python -m unittest discover tests
"""

import unittest
from pathlib import Path

from src.data import load_scenario
from src.search import astar, base_cost, straight_line_heuristic, ucs
from src.travel_time import TravelTimeModel

SCENARIOS = Path(__file__).resolve().parent.parent / "src" / "Scenarios"
ALL = [load_scenario(f) for f in sorted(SCENARIOS.glob("Scenario*"))]


class TestTravelTime(unittest.TestCase):
    def test_hand_computed_cost(self):
        # Scenario 1, road N2->N1: P(Open)=0.006, P(Congested)=0.993 (Bayes net)
        s = ALL[0]
        model = TravelTimeModel.from_scenario(s)
        road = s.graph.road("N2", "N1")
        b = model.status_beliefs[("N2", "N1")]
        status = (b["Open"] * 1.0 + b["Congested"] * 1.5) / (b["Open"] + b["Congested"])
        t = model.traffic_beliefs[("N2", "N1")]
        traffic = t["Low"] * 1.0 + t["Medium"] * 1.25 + t["High"] * 1.5
        self.assertAlmostEqual(model.cost(road), 775 / 15 * status * traffic)

    def test_blocked_roads_have_no_cost(self):
        s = ALL[0]
        model = TravelTimeModel.from_scenario(s)
        self.assertIsNone(model.cost(s.graph.road("H1", "N3")))  # P(Blocked) = 0.998
        self.assertIsNone(model.cost(s.graph.road("N1", "N2")))  # P(Blocked) = 0.527
        self.assertIsNotNone(model.cost(s.graph.road("N2", "N1")))  # P(Blocked) = 0.001

    def test_threshold_changes_borderline_roads_only(self):
        s = ALL[0]
        strict = TravelTimeModel.from_scenario(s, blocked_threshold=0.55)
        self.assertIsNotNone(strict.cost(s.graph.road("N1", "N2")))  # 0.527 < 0.55
        self.assertIsNone(strict.cost(s.graph.road("H1", "N3")))  # 0.998 > 0.55

    def test_costs_never_below_normal(self):
        # All factors >= 1, which keeps the A* heuristic admissible.
        for s in ALL:
            model = TravelTimeModel.from_scenario(s)
            for road in s.graph.roads.values():
                c = model.cost(road)
                if c is not None:
                    self.assertGreaterEqual(c, base_cost(road) - 1e-9)

    def test_switches_reproduce_normal_costs(self):
        s = ALL[0]
        model = TravelTimeModel.from_scenario(s, use_status=False, use_traffic=False)
        for road in s.graph.roads.values():
            self.assertAlmostEqual(model.cost(road), base_cost(road))

    def test_astar_still_optimal_with_model_costs(self):
        for s in ALL:
            cost = TravelTimeModel.from_scenario(s).cost
            g = s.graph
            for a in g.nodes:
                for b in g.nodes:
                    u, r = ucs(g, a, b, cost), astar(g, a, b, cost)
                    with self.subTest(scenario=s.name, start=a, goal=b):
                        self.assertEqual(u is None, r is None)
                        if u is not None:
                            self.assertAlmostEqual(u.cost, r.cost)
                            self.assertLessEqual(straight_line_heuristic(g, b)(a), r.cost + 1e-9)


if __name__ == "__main__":
    unittest.main()
