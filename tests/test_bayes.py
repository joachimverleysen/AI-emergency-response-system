"""
Tests for step 2 (Bayesian network for road status).

Run from the project root with:
    python -m unittest discover tests
"""

import unittest
from pathlib import Path

from src.bayes import road_status_beliefs, road_status_posterior
from src.data import ROAD_STATUSES, WEATHER_VALUES, RoadReport, load_scenario

SCENARIOS = Path(__file__).resolve().parent.parent / "src" / "Scenarios"
ALL = [load_scenario(f) for f in sorted(SCENARIOS.glob("Scenario*"))]
PROBS = ALL[0].road_probabilities  # identical CPTs in all scenarios


def closed_form(reports, probs):
    """Independent check: the formula derived in the report, written directly.

        P(S | w, r_1..r_n)  ∝  P(S | w) * prod_k P(r_k | S)                (weather known)
        P(S | r_1..r_n)     ∝  sum_w P(w) P(S | w) * prod_k P(r_k | S)     (weather unknown)
    """
    weathers = {r.weather for r in reports}
    unnorm = {}
    for s in ROAD_STATUSES:
        if len(weathers) == 1:
            prior = probs.status_given_weather[next(iter(weathers))][s]
        else:
            prior = sum(probs.weather_prior[w] * probs.status_given_weather[w][s] for w in WEATHER_VALUES)
        likelihood = 1.0
        for r in reports:
            likelihood *= probs.report_given_status[r.source][s][r.reported_status]
        unnorm[s] = prior * likelihood
    total = sum(unnorm.values())
    return {s: p / total for s, p in unnorm.items()}


def report(source, status, weather="Rain"):
    return RoadReport("A", "B", weather, source, status)


class TestRoadStatus(unittest.TestCase):
    def assertDistEqual(self, d1, d2, places=9):
        for s in ROAD_STATUSES:
            self.assertAlmostEqual(d1[s], d2[s], places=places)

    def test_hand_computed_example(self):
        # Scenario 1, road N1->N2, Rain; Citizen=Blocked, Police=Blocked, Sensor=Congested
        #   Open:      0.6 * 0.10 * 0.02 * 0.10 = 0.000120
        #   Congested: 0.3 * 0.20 * 0.08 * 0.80 = 0.003840
        #   Blocked:   0.1 * 0.70 * 0.90 * 0.07 = 0.004410
        #   sum = 0.008370  ->  (0.0143, 0.4588, 0.5269)
        reps = [report("Citizen", "Blocked"), report("Police", "Blocked"), report("Sensor", "Congested")]
        post = road_status_posterior(reps, PROBS)
        self.assertAlmostEqual(post["Open"], 0.000120 / 0.008370)
        self.assertAlmostEqual(post["Congested"], 0.003840 / 0.008370)
        self.assertAlmostEqual(post["Blocked"], 0.004410 / 0.008370)

    def test_no_reports_gives_marginal_prior(self):
        # sum_w P(w) P(S|w) = (0.685, 0.18, 0.135)
        post = road_status_posterior([], PROBS)
        self.assertDistEqual(post, {"Open": 0.685, "Congested": 0.18, "Blocked": 0.135})

    def test_enumeration_equals_elimination_equals_formula(self):
        for s in ALL:
            reports_per_road = {}
            for r in s.road_reports:
                reports_per_road.setdefault((r.from_node, r.to_node), []).append(r)
            for road in s.graph.roads:
                reps = reports_per_road.get(road, [])
                with self.subTest(scenario=s.name, road=road):
                    enum = road_status_posterior(reps, s.road_probabilities, "enumeration")
                    ve = road_status_posterior(reps, s.road_probabilities, "elimination")
                    self.assertDistEqual(enum, ve)
                    self.assertDistEqual(enum, closed_form(reps, s.road_probabilities))

    def test_unknown_weather_is_summed_out(self):
        # A report without a single consistent weather value -> weather is hidden.
        reps = [report("Police", "Open", "Rain"), report("Sensor", "Open", "Storm")]
        self.assertDistEqual(road_status_posterior(reps, PROBS), closed_form(reps, PROBS))
        self.assertDistEqual(road_status_posterior(reps, PROBS, "elimination"), closed_form(reps, PROBS))

    def test_more_blocked_reports_raise_p_blocked(self):
        reps = [report("Citizen", "Blocked")]
        p1 = road_status_posterior(reps, PROBS)["Blocked"]
        p2 = road_status_posterior(reps + [report("Sensor", "Blocked")], PROBS)["Blocked"]
        self.assertGreater(p2, p1)

    def test_police_more_reliable_than_citizen(self):
        # Same report, more reliable source -> stronger belief.
        citizen = road_status_posterior([report("Citizen", "Blocked")], PROBS)["Blocked"]
        police = road_status_posterior([report("Police", "Blocked")], PROBS)["Blocked"]
        self.assertGreater(police, citizen)

    def test_beliefs_cover_every_road_and_sum_to_one(self):
        for s in ALL:
            beliefs = road_status_beliefs(s)
            self.assertEqual(set(beliefs), set(s.graph.roads))
            for road, dist in beliefs.items():
                self.assertAlmostEqual(sum(dist.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
