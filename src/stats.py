"""
Step 7b - System statistics (statistics.txt).

Computed from the VALIDATED replay of the plan (src/validation.py), not from
the simulation's own bookkeeping.

Definitions (time unit: seconds of simulation time):
    incidents handled   incidents with a valid, replayed assignment
    responders used     distinct responders that appear in those assignments
    response time       arrival at the incident - incident timestamp
                        (waiting for a busy responder is included)
    average response    mean response time over HANDLED incidents only;
                        an unhandled incident has no arrival, so no response time
    deadlines met       incidents reached by their deadline / ALL incidents;
                        an unhandled incident counts as a missed deadline.
                        The share among handled incidents is reported as well.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data import PRIORITIES, Scenario
from src.validation import ValidationReport


@dataclass
class Statistics:
    total_incidents: int
    handled: int
    unhandled_ids: list[str]
    responders_used: int
    total_responders: int
    average_response: float | None  # None when nothing was handled
    max_response: float | None
    deadlines_met: int
    per_priority: dict[str, tuple[int, int]]  # priority -> (met, total)

    @property
    def pct_met_all(self) -> float:
        return 100.0 * self.deadlines_met / self.total_incidents if self.total_incidents else 100.0

    @property
    def pct_met_handled(self) -> float:
        return 100.0 * self.deadlines_met / self.handled if self.handled else 0.0


def compute_statistics(scenario: Scenario, report: ValidationReport) -> Statistics:
    done = report.replayed
    handled_ids = {a.incident_id for a in done}
    met = {a.incident_id for a in done if a.deadline_met}
    responses = [a.response_time for a in done]
    per_priority = {}
    for p in PRIORITIES:
        of_p = [i for i in scenario.incidents if i.priority == p]
        if of_p:
            per_priority[p] = (sum(i.id in met for i in of_p), len(of_p))
    return Statistics(
        total_incidents=len(scenario.incidents),
        handled=len(done),
        unhandled_ids=[i.id for i in scenario.incidents if i.id not in handled_ids],
        responders_used=len({a.responder_id for a in done}),
        total_responders=len(scenario.responders),
        average_response=sum(responses) / len(responses) if responses else None,
        max_response=max(responses) if responses else None,
        deadlines_met=len(met),
        per_priority=per_priority,
    )


def format_statistics(stats: Statistics, notes: list[str] | None = None) -> str:
    avg = f"{stats.average_response:.1f} s" if stats.average_response is not None else "n/a (nothing handled)"
    mx = f"{stats.max_response:.1f} s" if stats.max_response is not None else "n/a"
    lines = [
        "System statistics:",
        "*****",
        f"{stats.handled} incidents handled",
        f"{stats.responders_used} responders used",
        f"Average response time: {avg}",
        f"Deadlines met: {stats.pct_met_all:.1f}%",
        "",
        "Details",
        "-------",
        f"Incidents: {stats.total_incidents} total, {stats.handled} handled, "
        f"{len(stats.unhandled_ids)} unhandled ({', '.join(stats.unhandled_ids) or 'none'})",
        f"Responders used: {stats.responders_used} of {stats.total_responders}",
        f"Deadlines met: {stats.deadlines_met} of {stats.total_incidents} incidents "
        f"({stats.pct_met_all:.1f}%); among handled incidents {stats.pct_met_handled:.1f}%",
        "Per priority (met / total): "
        + ", ".join(f"{p} {m}/{t}" for p, (m, t) in stats.per_priority.items()),
        f"Maximum response time: {mx}",
        "",
        "How these numbers are computed",
        "------------------------------",
        "Time unit: seconds of simulation time.",
        "Response time = arrival at the incident - incident timestamp (includes waiting for a busy responder).",
        "Average and maximum response time are over handled incidents only; "
        "an unhandled incident has no arrival time.",
        "Deadlines met = incidents reached by their (absolute) deadline / ALL incidents; "
        "an unhandled incident counts as missed.",
        "All times are recomputed by the plan validator from the paths, using the expected travel times "
        "of the road model (Bayes net + HMM); there is no ground truth for the real road state.",
    ]
    if notes:
        lines += [""] + notes
    return "\n".join(lines) + "\n"
