"""One experiment = one map + one encoding + tuned QAOA angles + a classical baseline.

The same Problem object is used for the ideal simulation, the noisy
simulation and the real IBM job, so all three run the identical circuit.

Two problems are supported:

* "route" - the full one-hot TSP route (n cities -> (n-1)^2 qubits). Fits real
  hardware only for about 4 cities, because the circuit gets deep quickly.
* "split" - decide which of two vans delivers each stop (1 qubit per stop),
  with each van taking half of the stops; the classical solver then plans
  each van's route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time

from classical.solvers import held_karp, route_length
from quantum_simulation.qubo import QUBO, build_qubo, decode, qubo_energy
from quantum_simulation.simulator import TunedQAOA, tune_qaoa
from quantum_simulation.vehicle_split import build_split_qubo, split_groups
from shared.instance import MapInstance, build_instance, spec_key


def _sub_route(matrix, stops: list[int]) -> tuple[list[int], float]:
    """Optimal route for one van: depot -> its stops -> depot (Held-Karp)."""
    if not stops:
        return [0, 0], 0.0
    nodes = [0] + stops
    sub = [[matrix[x][y] for y in nodes] for x in nodes]
    if len(nodes) == 2:
        return [0, stops[0], 0], 2 * sub[0][1]
    sol = held_karp(sub)
    return [nodes[k] for k in sol.route], sol.length


# --------------------------------------------------------------------------
# Problem set-up (cached so that simulate and submit use identical circuits)
# --------------------------------------------------------------------------

@dataclass
class Problem:
    mode: str
    n: int
    seed: int | None
    layers: int
    map: MapInstance
    matrix: list[list[float]]
    qubo: QUBO
    tuned: TunedQAOA
    best_energy: float
    best_states: set[int]
    baseline: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    def evaluate(self, bits: int) -> dict | None:
        """What a measured bitstring means in delivery terms."""
        if bits in self._cache:
            return self._cache[bits]
        if self.mode == "route":
            r = decode(bits, self.n)
            out = None if r is None else {"routes": [r], "length": route_length(r, self.matrix)}
        else:
            a, b = split_groups(bits, self.n)
            if len(b) != (self.n - 1) // 2:       # van capacity rule broken
                self._cache[bits] = None
                return None
            ra, la = _sub_route(self.matrix, a)
            rb, lb = _sub_route(self.matrix, b)
            out = {"routes": [ra, rb], "length": la + lb, "groups": [a, b]}
        self._cache[bits] = out
        return out


LIMITS = {"route": (3, 5), "split": (4, 15)}


_PROBLEMS: dict[tuple, Problem] = {}


def get_problem(mode: str, n: int, seed: int, layers: int) -> Problem:
    """Problem on a made-up random map (kept for tests and the CLI)."""
    return get_problem_for_map(mode, {"kind": "random", "cities": n, "seed": seed}, layers)


def get_problem_for_map(mode: str, map_spec: dict, layers: int,
                        retry_fallback: bool = False) -> Problem:
    """Build (or reuse) the experiment for any map, random or real.

    The result is cached so that "Simulate" and "Submit to IBM" use exactly the
    same distances and the same circuit. If road distances could not be fetched
    (the map fell back to straight lines), `retry_fallback=True` tries again.
    """
    if mode not in LIMITS:
        raise ValueError(f"unknown problem '{mode}'")
    key = (mode, spec_key(map_spec), int(layers))
    cached = _PROBLEMS.get(key)
    if cached is not None and not (retry_fallback and cached.map.notes):
        return cached
    problem = _make_problem(mode, build_instance(map_spec), int(layers))
    if len(_PROBLEMS) > 16:
        _PROBLEMS.clear()
    _PROBLEMS[key] = problem
    return problem


def _make_problem(mode: str, inst: MapInstance, layers: int) -> Problem:
    n, seed = inst.n, inst.seed
    lo, hi = LIMITS[mode]
    if not lo <= n <= hi:
        raise ValueError(f"'{mode}' supports {lo}-{hi} places including the depot (you have {n})")
    matrix = inst.matrix
    q = build_qubo(matrix) if mode == "route" else build_split_qubo(matrix)
    tuned = tune_qaoa(q, layers)
    energies = tuned.energies
    if mode == "route":
        valid = [s for s in range(1 << q.n_qubits) if decode(s, n) is not None]
    else:
        valid = [s for s in range(1 << q.n_qubits) if bin(s).count("1") == (n - 1) // 2]
    best_e = min(energies[s] for s in valid)
    best_states = {s for s in valid if abs(energies[s] - best_e) < 1e-6}
    p = Problem(mode, n, seed, layers, inst, matrix, q, tuned, best_e, best_states)

    # Classical baseline to compare against.
    t0 = time.perf_counter()
    if mode == "route":
        hk = held_karp(matrix)
        p.baseline = {"name": "Held-Karp (exact)", "length": hk.length, "routes": [hk.route]}
    else:
        best = min(valid, key=lambda s: p.evaluate(s)["length"])
        ev = p.evaluate(best)
        p.baseline = {"name": "Try every fair split + exact routes", "length": ev["length"],
                      "routes": ev["routes"], "splits_checked": len(valid)}
    p.baseline["seconds"] = time.perf_counter() - t0
    return p



def summarise(problem: Problem, counts: dict[int, int]) -> dict:
    shots = sum(counts.values()) or 1
    valid = best = 0
    best_found, best_len = None, math.inf
    energy_sum = 0.0
    for bits, c in counts.items():
        ev = problem.evaluate(bits)
        energy_sum += c * qubo_energy(problem.qubo, bits)
        if ev is None:
            continue
        valid += c
        if ev["length"] <= problem.baseline["length"] + 1e-6:   # as good as the classical optimum
            best += c
        if ev["length"] < best_len:
            best_len, best_found = ev["length"], ev
    energies = problem.tuned.energies
    e_max, e_min = max(energies), problem.best_energy
    mean_e = energy_sum / shots
    out = {
        "shots": shots,
        "p_valid": valid / shots,
        "p_best": best / shots,
        # 1.0 = always the best answer, 0.0 = always the worst one
        "quality": (e_max - mean_e) / (e_max - e_min) if e_max > e_min else 1.0,
        "best_length": best_len if best_found else None,
        "best_routes": best_found["routes"] if best_found else None,
        "gap_percent": ((best_len / problem.baseline["length"] - 1) * 100) if best_found else None,
    }
    return out


def random_reference(problem: Problem) -> dict:
    total = 1 << problem.qubo.n_qubits
    energies = problem.tuned.energies
    e_max, e_min = max(energies), problem.best_energy
    mean = sum(energies) / total
    n_valid = sum(1 for s in range(total) if problem.evaluate(s) is not None)
    n_best = sum(1 for s in range(total) if problem.evaluate(s) is not None
                 and problem.evaluate(s)["length"] <= problem.baseline["length"] + 1e-6)
    return {"p_valid": n_valid / total, "p_best": n_best / total,
            "quality": (e_max - mean) / (e_max - e_min) if e_max > e_min else 1.0}


def ideal_counts(problem: Problem, shots: int, seed: int = 11) -> dict[int, int]:
    rng = random.Random(seed)
    probs = problem.tuned.probs
    counts: dict[int, int] = {}
    for s in rng.choices(range(len(probs)), weights=probs, k=shots):
        counts[s] = counts.get(s, 0) + 1
    return counts



def problem_info(problem: Problem) -> dict:
    return {
        "mode": problem.mode, "n": problem.n, "seed": problem.seed, "layers": problem.layers,
        "qubits": problem.qubo.n_qubits, "zz_terms": len(problem.qubo.quadratic),
        "params": [round(x, 4) for x in problem.tuned.params],
        "tuning_evaluations": problem.tuned.evaluations,
        "cities": problem.map.points,
        "map": problem.map.summary(),
        "baseline": problem.baseline,
        "random": random_reference(problem),
    }
