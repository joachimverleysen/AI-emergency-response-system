"""
Tests for step 1 (UCS and A*).

Run from the project root with:
    python -m unittest discover tests
"""

import unittest
from pathlib import Path

from src.data import load_scenario
from src.search import astar, base_cost, path_cost, straight_line_heuristic, ucs

SCENARIOS = Path(__file__).resolve().parent.parent / "src" / "Scenarios"
ALL = [load_scenario(f) for f in sorted(SCENARIOS.glob("Scenario*"))]


def all_simple_path_costs(graph, start, goal):
    """Independent oracle: depth-first enumeration of every path without
    repeated nodes. Exponential, so only used on the small Scenario 1."""
    costs = []

    def dfs(node, visited, cost):
        if node == goal:
            costs.append(cost)
            return
        for nxt in graph.neighbours(node):
            if nxt not in visited:
                dfs(nxt, visited | {nxt}, cost + graph.road(node, nxt).base_travel_time)

    dfs(start, {start}, 0.0)
    return costs


class TestSearch(unittest.TestCase):
    def test_hand_computed_example(self):
        # Scenario 1, from H1 to N2 (checked by hand in city_graph.json):
        #   direct          721/15           = 48.07 s   <- optimal
        #   via N4    340/30 + 615/15        = 52.33 s
        #   via N1    458/15 + 775/15        = 82.20 s
        g = ALL[0].graph
        result = astar(g, "H1", "N2")
        self.assertEqual(result.path, ["H1", "N2"])
        self.assertAlmostEqual(result.cost, 721 / 15)

    def test_start_equals_goal(self):
        result = ucs(ALL[0].graph, "N1", "N1")
        self.assertEqual(result.path, ["N1"])
        self.assertEqual(result.cost, 0.0)

    def test_ucs_matches_exhaustive_dfs(self):
        g = ALL[0].graph
        for a in g.nodes:
            for b in g.nodes:
                with self.subTest(start=a, goal=b):
                    self.assertAlmostEqual(ucs(g, a, b).cost, min(all_simple_path_costs(g, a, b)))

    def test_astar_equals_ucs_and_paths_are_valid(self):
        for s in ALL:
            g = s.graph
            for a in g.nodes:
                for b in g.nodes:
                    with self.subTest(scenario=s.name, start=a, goal=b):
                        u, r = ucs(g, a, b), astar(g, a, b)
                        self.assertAlmostEqual(u.cost, r.cost)
                        self.assertEqual(r.path[0], a)
                        self.assertEqual(r.path[-1], b)
                        # The returned cost must equal the cost of the returned path.
                        self.assertAlmostEqual(path_cost(g, r.path), r.cost)

    def test_heuristic_is_admissible(self):
        # h(n) may never exceed the true optimal cost from n to the goal.
        for s in ALL:
            g = s.graph
            for goal in g.nodes:
                h = straight_line_heuristic(g, goal)
                for n in g.nodes:
                    with self.subTest(scenario=s.name, node=n, goal=goal):
                        self.assertLessEqual(h(n), ucs(g, n, goal).cost + 1e-9)

    def test_astar_expands_fewer_nodes_in_total(self):
        for s in ALL:
            g = s.graph
            total_ucs = sum(ucs(g, a, b).expanded for a in g.nodes for b in g.nodes)
            total_astar = sum(astar(g, a, b).expanded for a in g.nodes for b in g.nodes)
            with self.subTest(scenario=s.name):
                self.assertLessEqual(total_astar, total_ucs)

    def test_blocked_road_is_avoided(self):
        g = ALL[0].graph

        def without_direct(road):
            return None if (road.from_node, road.to_node) == ("H1", "N2") else base_cost(road)

        result = astar(g, "H1", "N2", without_direct)
        self.assertNotEqual(result.path, ["H1", "N2"])
        self.assertIsNotNone(path_cost(g, result.path, without_direct))

    def test_unreachable_returns_none(self):
        g = ALL[0].graph

        def everything_blocked(road):
            return None

        self.assertIsNone(astar(g, "H1", "N2", everything_blocked))

    def test_path_cost_rejects_missing_road(self):
        g = ALL[0].graph
        self.assertIsNone(path_cost(g, ["N1", "N3"]))  # no direct road in Scenario 1


if __name__ == "__main__":
    unittest.main()
