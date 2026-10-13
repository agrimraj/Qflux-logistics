"""Scaling analysis: where does each approach stop being practical?

Runtimes beyond what we actually run are *projections*: we time a small solve
on this machine, measure the cost per basic step, and multiply by the known
step count of each algorithm. They are estimates, not measurements.
"""

from __future__ import annotations

import math

from shared.mapgen import euclidean_distance_matrix, generate_cities
from classical.solvers import brute_force, held_karp, heuristic_tsp

_calibration: dict[str, float] | None = None


def brute_steps(n: int) -> float:
    return max(1.0, math.factorial(n - 1) / 2) * n


def held_karp_steps(n: int) -> float:
    return n * n * 2.0 ** (n - 1)


def calibrate() -> dict[str, float]:
    """Seconds per basic step for the two exact solvers on *this* computer."""
    global _calibration
    if _calibration is None:
        m9 = euclidean_distance_matrix(generate_cities(9, 0))
        m13 = euclidean_distance_matrix(generate_cities(13, 0))
        b = min(brute_force(m9).elapsed_seconds for _ in range(2))
        h = held_karp(m13).elapsed_seconds
        _calibration = {"brute": b / brute_steps(9), "held_karp": h / held_karp_steps(13)}
    return _calibration


def human_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.0f} ms"
    if seconds < 120:
        return f"{seconds:.1f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.0f} h"
    if seconds < 2 * 31_557_600:
        return f"{seconds / 86400:.0f} days"
    years = seconds / 31_557_600
    if years < 1e6:
        return f"{years:,.0f} years"
    return f"{years:.1e} years"


def human_bytes(b: float) -> str:
    if b >= 1024 ** 6:
        return f"10^{math.log10(b):.0f} bytes"
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if b < 1024:
            return f"{b:.0f} {unit}"
        b /= 1024
    return f"{b:.0f} PB"


def breakdown_table(max_n: int = 30) -> list[dict]:
    cal = calibrate()
    rows = []
    for n in range(4, max_n + 1):
        qubits = (n - 1) ** 2
        tours = math.factorial(n - 1) // 2
        brute_s = brute_steps(n) * cal["brute"]
        hk_s = held_karp_steps(n) * cal["held_karp"]
        rows.append({
            "n": n,
            "valid_tours": tours,
            "qubits": qubits,
            "search_space_log10": round(qubits * math.log10(2), 1),
            "valid_fraction_log10": round(math.log10(tours) - qubits * math.log10(2), 1),
            "brute_seconds": brute_s,
            "brute_human": human_time(brute_s),
            "held_karp_seconds": hk_s,
            "held_karp_human": human_time(hk_s),
            "held_karp_memory": human_bytes(n * 2 ** (n - 1) * 100),  # ~100 B per dict entry
            "statevector_memory": human_bytes(16 * 2.0 ** qubits),
        })
    return rows


def measured_scaling(sizes: list[int], seed: int = 1, brute_limit: int = 11,
                     exact_limit: int = 18) -> list[dict]:
    """Actually run the solvers for each size (used by the UI's scaling chart)."""
    rows = []
    for n in sizes:
        m = euclidean_distance_matrix(generate_cities(n, seed))
        h = heuristic_tsp(m)
        hk = held_karp(m) if n <= exact_limit else None
        bf = brute_force(m) if n <= brute_limit else None
        rows.append({
            "n": n,
            "heuristic_seconds": h.elapsed_seconds,
            "held_karp_seconds": hk.elapsed_seconds if hk else None,
            "brute_seconds": bf.elapsed_seconds if bf else None,
            "gap_percent": (h.length / hk.length - 1) * 100 if hk else None,
        })
    return rows
