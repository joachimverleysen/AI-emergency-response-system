"""
Tests for step 3 (HMM: forward algorithm, Viterbi, stationary distribution).

Run from the project root with:
    python -m unittest discover tests
"""

import unittest
from itertools import product
from pathlib import Path

from src.data import TrafficProbabilities, load_scenario
from src.hmm import forward, states, stationary_distribution, time_elapse, traffic_beliefs, viterbi

SCENARIOS = Path(__file__).resolve().parent.parent / "src" / "Scenarios"
ALL = [load_scenario(f) for f in sorted(SCENARIOS.glob("Scenario*"))]

# The weather example of CS188 note 8.
WEATHER = TrafficProbabilities(
    initial={"sun": 0.8, "rain": 0.2},
    transition={"sun": {"sun": 0.6, "rain": 0.4}, "rain": {"sun": 0.1, "rain": 0.9}},
    emission={"sun": {"good": 0.8, "bad": 0.2}, "rain": {"good": 0.3, "bad": 0.7}},
)


def joint(xs, es, hmm):
    """P(x_0, x_1..x_t, e_1..e_t) by the chain rule of the HMM."""
    p = hmm.initial[xs[0]]
    for i, e in enumerate(es, start=1):
        p *= hmm.transition[xs[i - 1]][xs[i]] * hmm.emission[xs[i]][e]
    return p


def brute_force_filter(es, hmm):
    """Independent oracle: inference by enumeration over all state sequences."""
    totals = {x: 0.0 for x in states(hmm)}
    for xs in product(states(hmm), repeat=len(es) + 1):
        totals[xs[-1]] += joint(xs, es, hmm)
    z = sum(totals.values())
    return {x: v / z for x, v in totals.items()}


def brute_force_viterbi(es, hmm):
    """Independent oracle: the sequence x_0..x_t with the highest joint, without x_0."""
    best = max(product(states(hmm), repeat=len(es) + 1), key=lambda xs: joint(xs, es, hmm))
    return list(best[1:])


class TestCourseExample(unittest.TestCase):
    def test_time_elapse(self):
        # Note 8: B'(W1 = sun) = 0.6*0.8 + 0.1*0.2 = 0.5
        self.assertAlmostEqual(time_elapse(WEATHER.initial, WEATHER)["sun"], 0.5)

    def test_forward_one_step(self):
        # Note 8: F1 = good gives B(W1 = sun) = 8/11
        self.assertAlmostEqual(forward(["good"], WEATHER)[-1]["sun"], 8 / 11)

    def test_stationary_distribution(self):
        # Note 8: P(W_inf) = (sun 0.2, rain 0.8)
        d = stationary_distribution(WEATHER)
        self.assertAlmostEqual(d["sun"], 0.2)
        self.assertAlmostEqual(d["rain"], 0.8)


class TestTrafficHMM(unittest.TestCase):
    def test_forward_matches_enumeration_on_all_roads(self):
        for s in ALL:
            hmm = s.traffic_probabilities
            for road, obs in s.traffic_observations.items():
                with self.subTest(scenario=s.name, road=road):
                    got, want = forward(obs, hmm)[-1], brute_force_filter(obs, hmm)
                    for x in states(hmm):
                        self.assertAlmostEqual(got[x], want[x])

    def test_viterbi_matches_brute_force_on_all_roads(self):
        for s in ALL:
            hmm = s.traffic_probabilities
            for road, obs in s.traffic_observations.items():
                with self.subTest(scenario=s.name, road=road):
                    self.assertEqual(viterbi(obs, hmm), brute_force_viterbi(obs, hmm))

    def test_repeated_low_makes_low_more_certain(self):
        hmm = ALL[0].traffic_probabilities
        one, two = forward(["Low"], hmm)[-1]["Low"], forward(["Low", "Low"], hmm)[-1]["Low"]
        self.assertGreater(two, one)

    def test_beliefs_only_for_roads_with_history(self):
        for s in ALL:
            self.assertEqual(set(traffic_beliefs(s)), set(s.traffic_observations))


if __name__ == "__main__":
    unittest.main()
