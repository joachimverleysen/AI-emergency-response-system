"""
Step 0 - Loading the scenario data.

This module turns the raw files of one scenario folder into plain Python
objects (dataclasses). Nothing "intelligent" happens here: the goal is that
every later module (search, Bayes, HMM, allocation, simulation) works with
clean, typed data instead of raw CSV strings.

Units (from src/Scenarios/README.md):
    distance     metres
    speed        metres per second
    all times    seconds (timestamps, deadlines, durations, availability)

Files in a scenario folder:
    city_graph.json            nodes (x, y) and directed roads
    responders.csv             emergency units
    incidents.csv              incident stream (revealed over time)
    road_reports.csv           noisy reports about road status
    road_probabilities.json    CPTs of the Bayesian network
    observed_traffic.csv       historical traffic observations per road
    traffic_probabilities.json HMM parameters (initial, transition, emission)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Domain values. Used for validation and as a fixed ordering of states.
# ---------------------------------------------------------------------------
WEATHER_VALUES = ("Clear", "Rain", "Storm")
ROAD_STATUSES = ("Open", "Congested", "Blocked")
REPORT_SOURCES = ("Citizen", "Police", "Sensor")
TRAFFIC_LEVELS = ("Low", "Medium", "High")
PRIORITIES = ("Critical", "High", "Medium", "Low")


# ---------------------------------------------------------------------------
# City graph
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Node:
    id: str
    name: str
    x: float
    y: float
    type: str  # "normal", "hospital", "police", "fire_station"


@dataclass(frozen=True)
class Road:
    """A *directed* road. A two-way street appears as two Road objects."""

    from_node: str
    to_node: str
    distance: float  # metres
    road_type: str  # "main" or "secondary"
    speed: float  # metres / second

    @property
    def base_travel_time(self) -> float:
        """Travel time in seconds under normal (Open, no traffic) conditions."""
        return self.distance / self.speed


@dataclass
class CityGraph:
    nodes: dict[str, Node]
    roads: dict[tuple[str, str], Road]  # key: (from_node, to_node)

    def neighbours(self, node_id: str) -> list[str]:
        """Nodes reachable from `node_id` by one directed road."""
        return [to for (frm, to) in self.roads if frm == node_id]

    def road(self, from_node: str, to_node: str) -> Road:
        return self.roads[(from_node, to_node)]

    @property
    def max_speed(self) -> float:
        return max(r.speed for r in self.roads.values())


# ---------------------------------------------------------------------------
# Responders and incidents
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Responder:
    id: str
    type: str  # Ambulance, PoliceUnit, FireTruck
    start_location: str
    capabilities: frozenset[str]
    available: bool  # True = "Available", False = "Busy"
    time_until_available: float  # seconds (0 if available)


@dataclass(frozen=True)
class Incident:
    id: str
    location: str
    priority: str  # Critical, High, Medium, Low
    # ABSOLUTE simulation time (s) by which a responder must ARRIVE
    deadline: float
    required_capability: str
    timestamp: float  # simulation time (s) at which the incident becomes known
    time_required: float  # seconds the responder is busy on site after arrival


# ---------------------------------------------------------------------------
# Bayesian network inputs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoadReport:
    from_node: str
    to_node: str
    weather: str  # weather local to this road
    source: str  # Citizen, Police, Sensor
    reported_status: str  # Open, Congested, Blocked


@dataclass
class RoadProbabilities:
    """Conditional probability tables (CPTs) of the Bayesian network.

    weather_prior[w]                        = P(Weather = w)
    status_given_weather[w][s]              = P(Status = s | Weather = w)
    report_given_status[src][s][r]          = P(Report_src = r | Status = s)
    """

    weather_prior: dict[str, float]
    status_given_weather: dict[str, dict[str, float]]
    report_given_status: dict[str, dict[str, dict[str, float]]]


# ---------------------------------------------------------------------------
# Hidden Markov model inputs
# ---------------------------------------------------------------------------
@dataclass
class TrafficProbabilities:
    """HMM parameters.

    initial[i]        = P(X_1 = i)
    transition[i][j]  = P(X_{t+1} = j | X_t = i)
    emission[i][o]    = P(E_t = o | X_t = i)
    """

    initial: dict[str, float]
    transition: dict[str, dict[str, float]]
    emission: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# Everything of one scenario together
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    graph: CityGraph
    responders: list[Responder]
    incidents: list[Incident]  # sorted by timestamp
    road_reports: list[RoadReport]
    road_probabilities: RoadProbabilities
    # One observation sequence per directed road, ordered by observation_nr.
    traffic_observations: dict[tuple[str, str], list[str]]
    traffic_probabilities: TrafficProbabilities


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dicts, skipping blank lines.

    The provided CSV files contain empty lines between blocks; those rows
    are dropped here so that no other code has to worry about them.
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        return [
            {k.strip(): (v or "").strip()
             for k, v in row.items() if k is not None}
            for row in rows
            if any((v or "").strip() for v in row.values() if isinstance(v, str))
        ]


def load_graph(path: Path) -> CityGraph:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {
        n["id"]: Node(n["id"], n["name"], float(
            n["x"]), float(n["y"]), n["type"])
        for n in data["nodes"]
    }
    roads = {}
    for r in data["roads"]:
        road = Road(r["from"], r["to"], float(r["distance"]),
                    r["road_type"], float(r["speed"]))
        roads[(road.from_node, road.to_node)] = road
    return CityGraph(nodes, roads)


def load_responders(path: Path) -> list[Responder]:
    return [
        Responder(
            id=row["responder_id"],
            type=row["type"],
            start_location=row["start_location"],
            capabilities=frozenset(
                c.strip() for c in row["capabilities"].split(";") if c.strip()),
            available=row["availability"] == "Available",
            time_until_available=float(row["time_until_available"] or 0),
        )
        for row in _read_csv(path)
    ]


def load_incidents(path: Path) -> list[Incident]:
    incidents = [
        Incident(
            id=row["incident_id"],
            location=row["location"],
            priority=row["priority"],
            deadline=float(row["deadline"]),
            required_capability=row["required_capability"],
            timestamp=float(row["timestamp"]),
            time_required=float(row["time_required"]),
        )
        for row in _read_csv(path)
    ]
    return sorted(incidents, key=lambda i: i.timestamp)


def load_road_reports(path: Path) -> list[RoadReport]:
    return [
        RoadReport(row["from"], row["to"], row["weather"],
                   row["source"], row["reported_status"])
        for row in _read_csv(path)
    ]


def load_road_probabilities(path: Path) -> RoadProbabilities:
    data = json.loads(path.read_text(encoding="utf-8"))
    # JSON keys are "citizen_report", "police_report", "sensor_report";
    # we index them by the source names used in road_reports.csv.
    reports = {src: data[f"{src.lower()}_report"] for src in REPORT_SOURCES}
    return RoadProbabilities(data["weather_prior"], data["road_status_given_weather"], reports)


def load_traffic_observations(path: Path) -> dict[tuple[str, str], list[str]]:
    per_road: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for row in _read_csv(path):
        key = (row["from"], row["to"])
        per_road.setdefault(key, []).append(
            (int(row["observation_nr"]), row["traffic_observation"]))
    # Sort every sequence by observation_nr and keep only the observations.
    return {key: [obs for _, obs in sorted(seq)] for key, seq in per_road.items()}


def load_traffic_probabilities(path: Path) -> TrafficProbabilities:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TrafficProbabilities(data["initial"], data["transition"], data["emission"])


def load_scenario(folder: str | Path) -> Scenario:
    folder = Path(folder)
    return Scenario(
        name=folder.name,
        graph=load_graph(folder / "city_graph.json"),
        responders=load_responders(folder / "responders.csv"),
        incidents=load_incidents(folder / "incidents.csv"),
        road_reports=load_road_reports(folder / "road_reports.csv"),
        road_probabilities=load_road_probabilities(
            folder / "road_probabilities.json"),
        traffic_observations=load_traffic_observations(
            folder / "observed_traffic.csv"),
        traffic_probabilities=load_traffic_probabilities(
            folder / "traffic_probabilities.json"),
    )
