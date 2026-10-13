"""Run every experiment behind the main result and return plain numbers.

Statistics used
---------------
* Sampled probabilities (ideal samples, noisy simulator, hardware) are shown with
  a binomial standard error  sqrt(p (1 - p) / shots).
* The noisy simulator is run with 3 different seeds; we report mean and the
  spread (standard deviation) across those runs.
* The full-route experiment averages over several random maps (mean +- std).
* "Ideal (exact)" numbers come straight from the simulated quantum state, so
  they have no sampling error.
"""

from __future__ import annotations

import json
import math
import statistics
import time

from classical.solvers import held_karp, heuristic_tsp
from quantum_hardware.experiment import (Problem, _sub_route, get_problem_for_map,
                                         ideal_counts, random_reference, summarise)
from quantum_simulation.simulator import solve_qaoa
from shared.mapgen import euclidean_distance_matrix, generate_cities
from shared.scaling import measured_scaling

from . import config as C


def binomial_se(p: float, shots: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / shots) if shots else 0.0


def _with_errors(summary: dict) -> dict:
    out = dict(summary)
    out["p_valid_se"] = binomial_se(summary["p_valid"], summary["shots"])
    out["p_best_se"] = binomial_se(summary["p_best"], summary["shots"])
    out.pop("best_routes", None)
    return out


def exact_ideal(problem: Problem) -> dict:
    """Probabilities read directly from the ideal quantum state (no sampling)."""
    p_valid = p_best = 0.0
    for state, prob in enumerate(problem.tuned.probs):
        ev = problem.evaluate(state)
        if ev is None:
            continue
        p_valid += prob
        if ev["length"] <= problem.baseline["length"] + 1e-6:
            p_best += prob
    return {"p_valid": p_valid, "p_best": p_best}


def sweep_heuristic(problem: Problem) -> dict:
    """A fast classical heuristic for the same two-van problem.

    Sort the stops by compass direction around the depot, try every way of
    cutting that circle into two fair halves, and route each van exactly.
    """
    t0 = time.perf_counter()
    pts = {p["id"]: p for p in problem.map.points}
    depot = pts[0]
    stops = sorted((i for i in pts if i != 0),
                   key=lambda i: math.atan2(pts[i]["y"] - depot["y"], pts[i]["x"] - depot["x"]))
    k, half = len(stops), len(stops) // 2
    best = None
    for start in range(k):
        ring = stops[start:] + stops[:start]
        van_b, van_a = ring[:half], ring[half:]
        (ra, la), (rb, lb) = _sub_route(problem.matrix, van_a), _sub_route(problem.matrix, van_b)
        if best is None or la + lb < best[0]:
            best = (la + lb, [ra, rb])
    return {"length": best[0], "routes": best[1], "seconds": time.perf_counter() - t0,
            "gap_percent": (best[0] / problem.baseline["length"] - 1) * 100}


def load_hardware(problem: Problem) -> dict | None:
    """Read a saved IBM hardware run (made by run_hardware.py), if there is one."""
    if not C.HARDWARE_FILE.exists():
        return None
    data = json.loads(C.HARDWARE_FILE.read_text())
    counts = {int(k): int(v) for k, v in data["counts"].items()}
    out = _with_errors(summarise(problem, counts))
    out.update({k: data.get(k) for k in ("backend", "job_id", "date_utc", "qpu_seconds",
                                          "two_qubit_gates", "depth")})
    same = (data.get("map_spec") == C.MAIN_MAP and data.get("mode") == C.MAIN_MODE
            and data.get("layers") == C.MAIN_LAYERS)
    out["matches_current_experiment"] = bool(same)
    return out


def run_main_experiment(log=print) -> dict:
    """Two vans on the Kolkata map: random vs ideal vs noisy simulator vs IBM hardware."""
    import quantum_hardware as hw

    t0 = time.perf_counter()
    problem = get_problem_for_map(C.MAIN_MODE, C.MAIN_MAP, C.MAIN_LAYERS)
    log(f"  map: {problem.n} places, {problem.map.distance_source}; "
        f"{problem.qubo.n_qubits} qubits; tuned in {time.perf_counter() - t0:.1f} s")

    result = {
        "problem": {"places": [p["name"] for p in problem.map.points], "qubits": problem.qubo.n_qubits,
                    "distance_source": problem.map.distance_source, "layers": C.MAIN_LAYERS,
                    "params": problem.tuned.params, "tuning_evaluations": problem.tuned.evaluations,
                    "shots": C.SHOTS, "noisy_seeds": C.NOISY_SEEDS},
        "classical_exact": {"name": problem.baseline["name"], "length": problem.baseline["length"],
                            "seconds": problem.baseline["seconds"],
                            "splits_checked": problem.baseline.get("splits_checked"),
                            "routes": problem.baseline["routes"]},
        "classical_heuristic": sweep_heuristic(problem),
        "random": random_reference(problem),
        "ideal_exact": exact_ideal(problem),
        "ideal_sampled": _with_errors(summarise(problem, ideal_counts(problem, C.SHOTS, C.IDEAL_SAMPLE_SEED))),
    }

    noisy_runs, stats = [], {}
    for seed in C.NOISY_SEEDS:
        counts, stats = hw.noisy_counts(problem, C.SHOTS, C.FAKE_BACKEND, seed=seed)
        noisy_runs.append(_with_errors(summarise(problem, counts)))
        log(f"  noisy simulator seed {seed}: valid {noisy_runs[-1]['p_valid']:.3f}, "
            f"best {noisy_runs[-1]['p_best']:.4f}")
    result["noisy"] = {
        "backend": C.FAKE_BACKEND, "runs": noisy_runs,
        "two_qubit_gates": stats.get("two_qubit_gates"), "depth": stats.get("depth"),
        "p_valid": statistics.mean(r["p_valid"] for r in noisy_runs),
        "p_valid_sd": statistics.stdev(r["p_valid"] for r in noisy_runs),
        "p_best": statistics.mean(r["p_best"] for r in noisy_runs),
        "p_best_sd": statistics.stdev(r["p_best"] for r in noisy_runs),
        "best_length": min(r["best_length"] for r in noisy_runs if r["best_length"]),
    }
    result["hardware"] = load_hardware(problem)
    result["best_plan_routes"] = problem.baseline["routes"]
    result["points"] = problem.map.points
    result["seconds"] = time.perf_counter() - t0
    return result


def run_route_experiment(log=print) -> list[dict]:
    """Full-route (one-hot) QUBO on made-up maps: how often does QAOA return the optimum?"""
    rows = []
    for n in C.ROUTE_CITIES:
        for layers in C.ROUTE_LAYERS:
            if n == 5 and layers == 2:
                continue                       # slow in pure simulation; p=1 is enough here
            p_opt, p_valid, p_rand, secs = [], [], [], []
            for seed in C.ROUTE_SEEDS[n]:
                m = euclidean_distance_matrix(generate_cities(n, seed))
                r = solve_qaoa(m, layers=layers, shots=1024, seed=seed)
                p_opt.append(r.p_optimal)
                p_valid.append(r.p_valid)
                p_rand.append(r.p_optimal_random)
                secs.append(r.solution.elapsed_seconds if r.solution else float("nan"))
            hk_secs = [held_karp(euclidean_distance_matrix(generate_cities(n, s))).elapsed_seconds
                       for s in C.ROUTE_SEEDS[n]]
            sd = (lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0)
            row = {"cities": n, "qubits": (n - 1) ** 2, "layers": layers, "maps": len(C.ROUTE_SEEDS[n]),
                   "p_optimal": statistics.mean(p_opt), "p_optimal_sd": sd(p_opt),
                   "p_valid": statistics.mean(p_valid), "p_valid_sd": sd(p_valid),
                   "p_optimal_random": statistics.mean(p_rand),
                   "qaoa_seconds": statistics.mean(secs), "held_karp_seconds": statistics.mean(hk_secs)}
            rows.append(row)
            log(f"  route n={n} p={layers}: P(optimal) {row['p_optimal']:.4f} +- {row['p_optimal_sd']:.4f} "
                f"(random {row['p_optimal_random']:.4f}) over {row['maps']} maps")
    return rows


def run_scaling(log=print) -> list[dict]:
    rows = measured_scaling(C.SCALING_SIZES, C.SCALING_SEED, brute_limit=C.BRUTE_LIMIT)
    for r in rows:
        r["qubits_full_route"] = (r["n"] - 1) ** 2
        r["qubits_two_vans"] = r["n"] - 2
    log(f"  classical scaling measured for n = {C.SCALING_SIZES[0]}..{C.SCALING_SIZES[-1]}")
    # one larger heuristic run, to show the fast classical method on a bigger map
    big = heuristic_tsp(euclidean_distance_matrix(generate_cities(60, C.SCALING_SEED)))
    rows.append({"n": 60, "heuristic_seconds": big.elapsed_seconds, "held_karp_seconds": None,
                 "brute_seconds": None, "gap_percent": None,
                 "qubits_full_route": 59 ** 2, "qubits_two_vans": 58})
    return rows
