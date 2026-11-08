"""
Step 1 - Route planning with Uniform-Cost Search (UCS) and A*.

Problem formulation (as a search problem):
    state         a node of the city graph
    initial state the responder's current location
    goal test     state == incident location
    actions       drive along one directed road leaving the current node
    step cost     travel time of that road in seconds (given by `cost_fn`)

The step cost is NOT fixed in this module. It is passed in as a function
`cost_fn(road) -> seconds or None`, where None means "road is unusable".
In step 1 we use the normal travel time (distance / speed). In step 4 the
same search will receive a cost function built from the Bayesian network
and the HMM, without changing this file.

UCS and A* are the same algorithm with a different priority:
    UCS:  f(n) = g(n)            (cost so far)
    A*:   f(n) = g(n) + h(n)     (cost so far + estimate of remaining cost)
With h(n) = 0, A* is exactly UCS. That is how it is implemented below.

Heuristic (A*):
    h(n) = straight_line_distance(n, goal) / max_speed

Why this is admissible (never overestimates): every road is at least as
long as the straight line between its endpoints (checked in step 0), and
no road is faster than max_speed. By the triangle inequality, any route
from n to the goal is at least straight_line_distance(n, goal) long, and
driving it takes at least that distance / max_speed seconds. Multiplying
travel times by factors >= 1 (congestion, traffic) keeps this true.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Callable, Optional

from src.data import CityGraph, Road

# A cost function returns the travel time of a road in seconds,
# or None if the road may not be used.
CostFunction = Callable[[Road], Optional[float]]
Heuristic = Callable[[str], float]


def base_cost(road: Road) -> float:
    """Travel time under normal conditions (Open road, no traffic)."""
    return road.base_travel_time


@dataclass
class SearchResult:
    path: list[str]  # nodes from start to goal (inclusive)
    cost: float  # total travel time in seconds
    expanded: int  # number of nodes taken from the frontier and expanded


def straight_line_heuristic(graph: CityGraph, goal: str) -> Heuristic:
    """h(n) = Euclidean distance from n to the goal / fastest speed in the graph."""
    gx, gy = graph.nodes[goal].x, graph.nodes[goal].y
    vmax = graph.max_speed

    def h(node: str) -> float:
        n = graph.nodes[node]
        return math.hypot(n.x - gx, n.y - gy) / vmax

    return h


def best_first_search(
    graph: CityGraph,
    start: str,
    goal: str,
    cost_fn: CostFunction = base_cost,
    heuristic: Heuristic = lambda node: 0.0,
) -> Optional[SearchResult]:
    """Graph search that always expands the frontier node with the lowest f = g + h.

    Returns None when the goal cannot be reached (e.g. all routes blocked).
    """
    # Frontier: priority queue of (f, tie_breaker, node).
    # The tie_breaker (a counter) keeps the order deterministic when f is equal.
    counter = 0
    frontier: list[tuple[float, int, str]] = [(heuristic(start), counter, start)]

    g: dict[str, float] = {start: 0.0}  # best known cost from start to each node
    parent: dict[str, Optional[str]] = {start: None}  # to rebuild the path
    expanded: set[str] = set()  # nodes already expanded ("explored set")

    while frontier:
        _, _, node = heapq.heappop(frontier)

        # The same node can be in the frontier several times with different
        # priorities; only the first (cheapest) time it is popped counts.
        if node in expanded:
            continue

        # Goal test when a node is EXPANDED, not when it is generated.
        # Only then is g(goal) guaranteed to be optimal.
        if node == goal:
            return SearchResult(_rebuild_path(parent, goal), g[goal], len(expanded) + 1)

        expanded.add(node)

        for neighbour in graph.neighbours(node):
            step = cost_fn(graph.road(node, neighbour))
            if step is None:  # road unusable
                continue
            new_g = g[node] + step
            if neighbour not in g or new_g < g[neighbour]:
                g[neighbour] = new_g
                parent[neighbour] = node
                counter += 1
                heapq.heappush(frontier, (new_g + heuristic(neighbour), counter, neighbour))

    return None  # frontier empty: goal unreachable


def _rebuild_path(parent: dict[str, Optional[str]], goal: str) -> list[str]:
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return list(reversed(path))


def ucs(graph: CityGraph, start: str, goal: str, cost_fn: CostFunction = base_cost) -> Optional[SearchResult]:
    """Uniform-Cost Search: best-first search with h(n) = 0."""
    return best_first_search(graph, start, goal, cost_fn)


def astar(graph: CityGraph, start: str, goal: str, cost_fn: CostFunction = base_cost) -> Optional[SearchResult]:
    """A* search with the straight-line heuristic."""
    return best_first_search(graph, start, goal, cost_fn, straight_line_heuristic(graph, goal))


def path_cost(graph: CityGraph, path: list[str], cost_fn: CostFunction = base_cost) -> Optional[float]:
    """Total cost of an explicit path; None if a road is missing or unusable.

    Used by the tests now and by the plan validator later.
    """
    total = 0.0
    for a, b in zip(path, path[1:]):
        if (a, b) not in graph.roads:
            return None
        step = cost_fn(graph.road(a, b))
        if step is None:
            return None
        total += step
    return total
