"""An exact QAOA simulator (state-vector), written from scratch.

It stores all 2^qubits amplitudes of the quantum state, which is why it breaks
down quickly: n=4 cities -> 9 qubits (512 numbers), n=5 -> 16 qubits (65 536),
n=6 -> 25 qubits (33 million numbers, ~0.5 GB). Real hardware avoids this
memory wall but adds noise (see ../quantum_hardware).

Works with plain Python; installing numpy makes it ~100x faster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time

try:  # numpy makes the simulator ~100x faster, but is optional
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without numpy
    np = None

from classical.solvers import TSPSolution, route_length

from .qubo import QUBO, all_energies, build_qubo, decode


# --------------------------------------------------------------------------
# State-vector simulation
# --------------------------------------------------------------------------

def _apply_layer_numpy(psi, phase_e, gamma, beta, m):
    psi = psi * np.exp(-1j * gamma * phase_e)
    c, s = math.cos(beta), -1j * math.sin(beta)
    psi = psi.reshape([2] * m)
    for axis in range(m):
        a0 = np.take(psi, 0, axis=axis)
        a1 = np.take(psi, 1, axis=axis)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=axis)
    return psi.reshape(-1)


def _apply_layer_python(psi, phase_e, gamma, beta, m):
    psi = [amp * complex(math.cos(gamma * e), -math.sin(gamma * e)) for amp, e in zip(psi, phase_e)]
    c, s = math.cos(beta), complex(0, -math.sin(beta))
    size = len(psi)
    for q in range(m):
        step = 1 << q
        for base in range(0, size, step << 1):
            for i in range(base, base + step):
                a0, a1 = psi[i], psi[i + step]
                psi[i], psi[i + step] = c * a0 + s * a1, s * a0 + c * a1
    return psi


def qaoa_state(phase_e, params: list[float], m: int):
    p = len(params) // 2
    size = 1 << m
    amp = 1 / math.sqrt(size)
    if np is not None:
        psi = np.full(size, amp, dtype=np.complex128)
        for layer in range(p):
            psi = _apply_layer_numpy(psi, phase_e, params[layer], params[p + layer], m)
        return psi
    psi = [complex(amp, 0)] * size
    for layer in range(p):
        psi = _apply_layer_python(psi, phase_e, params[layer], params[p + layer], m)
    return psi


def _probabilities(psi):
    if np is not None:
        return np.abs(psi) ** 2
    return [abs(a) ** 2 for a in psi]


def _expectation(probs, energies) -> float:
    if np is not None:
        return float(np.dot(probs, energies))
    return sum(p * e for p, e in zip(probs, energies))


def _nelder_mead(f, x0: list[float], step: float = 0.3, iters: int = 120):
    """Tiny derivative-free optimiser (so no SciPy is needed)."""
    dim = len(x0)
    pts = [x0[:]] + [[x0[j] + (step if j == i else 0) for j in range(dim)] for i in range(dim)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(dim + 1), key=vals.__getitem__)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        centroid = [sum(p[j] for p in pts[:-1]) / dim for j in range(dim)]
        refl = [centroid[j] + (centroid[j] - pts[-1][j]) for j in range(dim)]
        fr = f(refl)
        if fr < vals[0]:
            exp = [centroid[j] + 2 * (centroid[j] - pts[-1][j]) for j in range(dim)]
            fe = f(exp)
            pts[-1], vals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = refl, fr
        else:
            con = [centroid[j] + 0.5 * (pts[-1][j] - centroid[j]) for j in range(dim)]
            fc = f(con)
            if fc < vals[-1]:
                pts[-1], vals[-1] = con, fc
            else:
                for i in range(1, dim + 1):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(dim)]
                    vals[i] = f(pts[i])
    best = min(range(dim + 1), key=vals.__getitem__)
    return pts[best], vals[best]


@dataclass
class QAOAResult:
    solution: TSPSolution | None   # best valid tour seen in the samples
    n_qubits: int
    layers: int
    shots: int
    circuit_evaluations: int
    params: list[float]
    p_optimal: float               # probability one shot returns the optimal tour
    p_valid: float                 # probability one shot is a valid tour at all
    p_optimal_random: float        # same, for random guessing (for comparison)
    valid_shots: int
    expected_shots_to_optimal: float | None
    statevector_bytes: int
    backend: str
    convergence: list[float] = field(default_factory=list)


MAX_QAOA_CITIES = 5


@dataclass
class TunedQAOA:
    params: list[float]       # [gamma_1..gamma_p, beta_1..beta_p]
    scale: float              # energies were divided by this before applying gamma
    energies: list[float]     # energy of every bitstring
    probs: list[float]        # ideal probability of measuring every bitstring
    evaluations: int
    history: list[float]


def tune_qaoa(q: QUBO, layers: int = 1) -> TunedQAOA:
    """Find good QAOA angles for any QUBO using the exact simulator.

    This is the expensive "classical outer loop" of QAOA. We do it on the laptop
    so that real quantum hardware is only needed for the final, tuned circuit.
    """
    m = q.n_qubits
    if m > 20:
        raise ValueError(f"{m} qubits is too many to tune on a laptop simulator")
    energies = all_energies(q)
    # Scale energies so the angles gamma live in a sensible range.
    scale = max(abs(min(energies)), abs(max(energies))) or 1.0
    phase_e = energies / scale if np is not None else [e / scale for e in energies]
    evaluations = 0
    history: list[float] = []

    def objective(params):
        nonlocal evaluations
        evaluations += 1
        val = _expectation(_probabilities(qaoa_state(phase_e, params, m)), phase_e)
        history.append(val * scale)
        return val

    # Coarse grid for the first layer, then let Nelder-Mead refine everything.
    best = (math.inf, [0.5] * layers + [0.4] * layers)
    for g in [0.2 * k for k in range(1, 16)]:
        for b in [0.1 * k for k in range(1, 8)]:
            params = [g] * layers + [b] * layers
            v = objective(params)
            if v < best[0]:
                best = (v, params)
    params, _ = _nelder_mead(objective, best[1], iters=40 * layers if np is not None else 15 * layers)
    probs = _probabilities(qaoa_state(phase_e, params, m))
    return TunedQAOA(
        params=[float(x) for x in params], scale=float(scale),
        energies=energies.tolist() if np is not None else energies,
        probs=probs.tolist() if np is not None else probs,
        evaluations=evaluations, history=history)


def solve_qaoa(matrix: list[list[float]], layers: int = 2, shots: int = 1024,
               seed: int = 7, max_cities: int = MAX_QAOA_CITIES) -> QAOAResult:
    """Run QAOA on a simulated quantum computer and sample tours from it."""
    n = len(matrix)
    if n < 3:
        raise ValueError("QAOA needs at least 3 cities")
    if n > max_cities:
        q = (n - 1) ** 2
        raise ValueError(
            f"{n} cities need {q} qubits; simulating that needs {16 * 2 ** q / 1e9:,.1f} GB "
            f"of memory. This is exactly where the approach breaks down on a laptop.")
    start = time.perf_counter()
    q = build_qubo(matrix)
    m = q.n_qubits
    t = tune_qaoa(q, layers)
    energies, params, probs_list = t.energies, t.params, t.probs
    evaluations, history = t.evaluations, t.history

    # Identify valid tours and the optimum from the energy table.
    valid = [s for s in range(1 << m) if decode(s, n) is not None]
    opt_e = min(energies[s] for s in valid)
    optimal_states = [s for s in valid if abs(energies[s] - opt_e) < 1e-6]
    p_opt = sum(probs_list[s] for s in optimal_states)
    p_valid = sum(probs_list[s] for s in valid)

    # Measure the circuit `shots` times, like a real device would.
    rng = random.Random(seed)
    samples = rng.choices(range(1 << m), weights=probs_list, k=shots)
    best_route, best_len, valid_shots = None, math.inf, 0
    for s in samples:
        r = decode(s, n)
        if r is None:
            continue
        valid_shots += 1
        length = route_length(r, matrix)
        if length < best_len:
            best_route, best_len = r, length
    elapsed = time.perf_counter() - start
    solution = None
    if best_route is not None:
        solution = TSPSolution("qaoa (simulated)", best_route, best_len, elapsed,
                               bool(abs(best_len - opt_e) < 1e-6))
    return QAOAResult(
        solution=solution, n_qubits=m, layers=layers, shots=shots,
        circuit_evaluations=evaluations, params=[round(x, 4) for x in params],
        p_optimal=float(p_opt), p_valid=float(p_valid),
        p_optimal_random=len(optimal_states) / (1 << m),
        valid_shots=valid_shots,
        expected_shots_to_optimal=(1 / p_opt) if p_opt > 0 else None,
        statevector_bytes=16 * (1 << m),
        backend="numpy" if np is not None else "pure-python",
        convergence=[float(v) for v in history[:: max(1, len(history) // 60)]],
    )
