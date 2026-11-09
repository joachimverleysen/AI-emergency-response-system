# AI-emergency-response-system

Decision-support system that assigns emergency responders to incidents in a
storm-hit city and plans their routes (AI practicum assignment).

Components: A* search (`src/search.py`), Bayes net for road status
(`src/bayes.py`), hidden Markov model for traffic (`src/hmm.py`), travel-time
model (`src/travel_time.py`), constraint-based allocation with OR-Tools
(`src/allocation.py`) and an online event-driven simulation (`src/simulation.py`).

## Run

```
pip install -r requirements.txt
python run.py src/Scenarios/Scenario1                 # CSP allocator (OR-Tools)
python run.py src/Scenarios/Scenario1 --allocator greedy --details
```

Output is written to `output/<scenario>/plan.json`.

## Test

```
python -m unittest discover tests
```

Tests that need OR-Tools are skipped when it is not installed.
