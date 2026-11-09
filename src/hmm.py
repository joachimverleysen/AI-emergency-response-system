"""
Step 3 - Hidden Markov model for traffic on a road.

Notation follows CS188 note 8.

    state variable     X_i  traffic on the road at step i      (Low / Medium / High, hidden)
    evidence variable  E_i  observed traffic at step i         (Low / Medium / High)

    initial distribution  P(X_0)            traffic_probabilities.json "initial"
    transition model      P(X_{i+1} | X_i)  "transition"   (stationary: same for every i)
    sensor model          P(E_i | X_i)      "emission"     (stationary as well)

Following the note, X_0 has no observation; observation k of a road (k = 1..t,
ordered by observation_nr) is the evidence e_k for X_k.

Belief distributions:
    B'(X_{i+1}) = P(X_{i+1} | e_1..e_i)       after the time elapse update
    B(X_{i+1})  = P(X_{i+1} | e_1..e_{i+1})   after the observation update

Forward algorithm, one step = time elapse update + observation update:
    B'(X_{i+1}) = sum_{x_i} P(X_{i+1} | x_i) B(x_i)
    B(X_{i+1})  ∝ P(e_{i+1} | X_{i+1}) B'(X_{i+1})

What we use in planning (assumption A8): the belief B(X_t) after the LAST
observation of a road. observation_nr is not simulation time, so we do not
predict further ahead.

Viterbi is implemented as analysis: the single most likely sequence of
traffic states. It is reported, not used for planning.
"""

from __future__ import annotations

from src.data import Scenario, TrafficProbabilities

Belief = dict[str, float]  # traffic level -> probability


def states(hmm: TrafficProbabilities) -> list[str]:
    """The possible values of X (for traffic: Low, Medium, High)."""
    return list(hmm.initial)


def normalize(dist: Belief) -> Belief:
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()}


# -----------------------------------------------------------------------------
# Forward algorithm
# -----------------------------------------------------------------------------
def time_elapse(belief: Belief, hmm: TrafficProbabilities) -> Belief:
    """B'(X_{i+1}) = sum_{x_i} P(X_{i+1} | x_i) B(x_i)   (mini-forward step)"""
    return {
        nxt: sum(hmm.transition[cur][nxt] * belief[cur] for cur in states(hmm))
        for nxt in states(hmm)
    }


def observe(belief_prime: Belief, evidence: str, hmm: TrafficProbabilities) -> Belief:
    """B(X_{i+1}) ∝ P(e_{i+1} | X_{i+1}) B'(X_{i+1}), then normalize."""
    return normalize({x: hmm.emission[x][evidence] * belief_prime[x] for x in states(hmm)})


def forward(observations: list[str], hmm: TrafficProbabilities) -> list[Belief]:
    """Beliefs B(X_0), B(X_1), ..., B(X_t). B(X_0) is the initial distribution."""
    beliefs = [dict(hmm.initial)]
    for e in observations:
        beliefs.append(observe(time_elapse(beliefs[-1], hmm), e, hmm))
    return beliefs


def filtered_belief(observations: list[str], hmm: TrafficProbabilities) -> Belief:
    """B(X_t) = P(X_t | e_1..e_t): belief after the last observation."""
    return forward(observations, hmm)[-1]


# -----------------------------------------------------------------------------
# Viterbi: most likely sequence x_1..x_t given e_1..e_t
#
#   m_1(x)     = P(e_1 | x) * max_{x_0} P(x | x_0) P(x_0)
#   m_{i+1}(x) = P(e_{i+1} | x) * max_{x_i} P(x | x_i) m_i(x_i)
#
# Same structure as the forward algorithm, with max instead of sum. For each
# state we remember which predecessor gave the max (back pointer) and follow
# the pointers back from the best final state.
# -----------------------------------------------------------------------------
def viterbi(observations: list[str], hmm: TrafficProbabilities) -> list[str]:
    if not observations:
        return []
    m = dict(hmm.initial)  # plays the role of m_0 = P(X_0)
    back_pointers: list[dict[str, str]] = []
    for e in observations:
        new_m, pointers = {}, {}
        for x in states(hmm):
            best_prev = max(states(hmm), key=lambda prev: hmm.transition[prev][x] * m[prev])
            pointers[x] = best_prev
            new_m[x] = hmm.emission[x][e] * hmm.transition[best_prev][x] * m[best_prev]
        # Scaling every entry by the same constant does not change any max;
        # it only prevents the numbers from becoming tiny on long sequences.
        m = normalize(new_m)
        back_pointers.append(pointers)
    # Follow the back pointers from the most likely final state.
    path = [max(states(hmm), key=lambda x: m[x])]
    for pointers in reversed(back_pointers[1:]):
        path.append(pointers[path[-1]])
    return list(reversed(path))


# -----------------------------------------------------------------------------
# Stationary distribution (used for discussion in the report)
# -----------------------------------------------------------------------------
def stationary_distribution(hmm: TrafficProbabilities, tolerance: float = 1e-12) -> Belief:
    """Repeat the mini-forward step until P(X_{t+1}) = P(X_t)."""
    belief = dict(hmm.initial)
    while True:
        nxt = time_elapse(belief, hmm)
        if max(abs(nxt[x] - belief[x]) for x in states(hmm)) < tolerance:
            return nxt
        belief = nxt


# -----------------------------------------------------------------------------
# All roads of a scenario
# -----------------------------------------------------------------------------
def traffic_beliefs(scenario: Scenario) -> dict[tuple[str, str], Belief]:
    """Filtered belief after the last observation, for every road WITH history.

    Roads without traffic observations are not in the result (assumption A7:
    they get traffic factor 1.0).
    """
    hmm = scenario.traffic_probabilities
    return {road: filtered_belief(obs, hmm) for road, obs in scenario.traffic_observations.items()}
