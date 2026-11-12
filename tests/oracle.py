"""
Test helpers around the exhaustive allocator.

exhaustive_allocate (in src/allocation.py) minimises exactly the same
objective as the CP-SAT model by depth-first search over all assignments.
It is the independent reference for the CSP allocator. The helpers below
enumerate ALL optimal assignments, to show the effect of tie-breaking.
"""

from src.allocation import exhaustive_allocate, total_cost  # noqa: F401  (re-exported for the tests)


def all_optimal_allocations(pending, options):
    """Every assignment that reaches the optimal objective (ties included),
    in a fixed order. Used to show that the choice between equally good
    assignments changes the rest of the simulation."""
    from itertools import product

    per_incident = [[None] + [o for o in options if o.incident_id == inc.id] for inc in pending]
    best, solutions = None, []
    for combo in product(*per_incident):
        used = [o.responder_id for o in combo if o is not None]
        if len(used) != len(set(used)):
            continue  # a responder used twice violates the constraint
        choice = {o.incident_id: o for o in combo if o is not None}
        cost = total_cost(choice, pending)
        if best is None or cost < best:
            best, solutions = cost, [choice]
        elif cost == best:
            solutions.append(choice)
    return solutions


def last_optimal_allocate(pending, options):
    """Like exhaustive_allocate, but breaks ties the other way."""
    return all_optimal_allocations(pending, options)[-1]
