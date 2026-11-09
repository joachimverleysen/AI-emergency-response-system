"""
Test helper: an exhaustive allocator that minimises EXACTLY the same objective
as the CP-SAT model (src.allocation.total_cost), by depth-first search over
all assignments. Exponential, only for small decision moments. Used as an
independent oracle for the CSP allocator.
"""

from src.allocation import Option, total_cost


def exhaustive_allocate(pending, options, limit=200_000):
    per_incident = [[o for o in options if o.incident_id == inc.id] for inc in pending]
    best = {"cost": None, "choice": {}}
    visited = [0]

    def dfs(k, used, choice):
        visited[0] += 1
        if visited[0] > limit:
            raise RuntimeError("decision moment too large for exhaustive search")
        if k == len(pending):
            c = total_cost(choice, pending)
            if best["cost"] is None or c < best["cost"]:
                best["cost"], best["choice"] = c, dict(choice)
            return
        dfs(k + 1, used, choice)  # incident k unassigned
        for o in per_incident[k]:
            if o.responder_id not in used:
                choice[o.incident_id] = o
                dfs(k + 1, used | {o.responder_id}, choice)
                del choice[o.incident_id]

    dfs(0, frozenset(), {})
    return best["choice"]
