# AI-emergency-response-system

Decision-support system that assigns emergency responders to incidents in a
storm-hit city and plans their routes (AI practicum assignment).

Components: A* search (`src/search.py`), Bayes net for road status
(`src/bayes.py`), hidden Markov model for traffic (`src/hmm.py`), travel-time
model (`src/travel_time.py`), constraint-based allocation with OR-Tools
(`src/allocation.py`), an online event-driven simulation (`src/simulation.py`),
an independent plan validator (`src/validation.py`) and statistics (`src/stats.py`).

## Run

```
pip install -r requirements.txt
python run.py src/Scenarios/Scenario1                 # CSP allocator (OR-Tools)
python run.py src/Scenarios/Scenario1 --allocator greedy --details
python run.py src/Scenarios/Scenario1 --allocator exhaustive   # reference, no OR-Tools needed
```

Output is written to `output/<scenario>/plan.json` and `output/<scenario>/statistics.txt`.
All times are in seconds.

## Experiments

```
python experiments.py      # writes output/experiments.md
```

## Test

```
python -m unittest discover tests
```

Tests that need OR-Tools are skipped when it is not installed.
