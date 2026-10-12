"""Encode the delivery-route problem (TSP) as a QUBO.

QUBO = Quadratic Unconstrained Binary Optimisation: find the bitstring x that
minimises  offset + sum_a h_a x_a + sum_{a<b} J_ab x_a x_b.  QAOA works on
exactly this kind of problem.

Encoding (one-hot, depot fixed)
-------------------------------
City 0 is the depot and is always first and last, so only cities 1..n-1 need
to be placed in positions 1..n-1.  Binary variable

    x[i, t] = 1  if city i is visited at position t

gives m = (n-1)^2 variables, i.e. (n-1)^2 qubits.

    cost    = sum of distances along the encoded route
    penalty = A * sum_t (1 - sum_i x[i,t])^2      (one city per position)
            + A * sum_i (1 - sum_t x[i,t])^2      (each city exactly once)

A valid tour has penalty 0, so the lowest-energy bitstring is the optimal tour.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # numpy makes the simulator ~100x faster, but is optional
    import numpy as np
except ImportError:  # pragma: no cover - exercised only without numpy
    np = None


# --------------------------------------------------------------------------
# QUBO
# --------------------------------------------------------------------------

@dataclass
class QUBO:
    n_cities: int
    n_qubits: int
    penalty: float
    offset: float
    linear: list[float]                      # h[a]
    quadratic: dict[tuple[int, int], float]  # J[(a, b)] with a < b


def var_index(city: int, pos: int, n: int) -> int:
    """Qubit index for x[city, pos]; city and pos both run from 1 to n-1."""
    return (city - 1) * (n - 1) + (pos - 1)


def build_qubo(matrix: list[list[float]], penalty: float | None = None) -> QUBO:
    n = len(matrix)
    k = n - 1                       # cities to place / positions to fill
    m = k * k
    if penalty is None:
        # Larger than any possible gain from breaking a rule.
        penalty = 2.0 * max(max(row) for row in matrix)
    lin = [0.0] * m
    quad: dict[tuple[int, int], float] = {}

    def add(a: int, b: int, w: float) -> None:
        if a == b:
            lin[a] += w            # x*x == x for binary variables
            return
        key = (a, b) if a < b else (b, a)
        quad[key] = quad.get(key, 0.0) + w

    # --- distance part ---
    for i in range(1, n):
        lin[var_index(i, 1, n)] += matrix[0][i]          # depot -> first
        lin[var_index(i, k, n)] += matrix[i][0]          # last -> depot
    for t in range(1, k):
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    add(var_index(i, t, n), var_index(j, t + 1, n), matrix[i][j])

    # --- penalties: (1 - sum x)^2 = 1 - sum x + 2*sum_{a<b} x_a x_b (binary) ---
    offset = 0.0
    groups = [[var_index(i, t, n) for i in range(1, n)] for t in range(1, n)]
    groups += [[var_index(i, t, n) for t in range(1, n)] for i in range(1, n)]
    for g in groups:
        offset += penalty
        for a in g:
            lin[a] -= penalty
        for x in range(len(g)):
            for y in range(x + 1, len(g)):
                add(g[x], g[y], 2 * penalty)
    return QUBO(n, m, penalty, offset, lin, quad)


def qubo_energy(q: QUBO, bits: int) -> float:
    e = q.offset
    for a, h in enumerate(q.linear):
        if bits >> a & 1:
            e += h
    for (a, b), w in q.quadratic.items():
        if bits >> a & 1 and bits >> b & 1:
            e += w
    return e


def all_energies(q: QUBO):
    """Energy of every one of the 2^m bitstrings (the diagonal of the cost Hamiltonian)."""
    m = q.n_qubits
    if np is not None:
        idx = np.arange(1 << m, dtype=np.int64)
        bits = [((idx >> a) & 1).astype(np.float64) for a in range(m)]
        e = np.full(1 << m, q.offset)
        for a, h in enumerate(q.linear):
            e += h * bits[a]
        for (a, b), w in q.quadratic.items():
            e += w * bits[a] * bits[b]
        return e
    return [qubo_energy(q, s) for s in range(1 << m)]


def decode(bits: int, n: int) -> list[int] | None:
    """Turn a bitstring into a route [0, ..., 0], or None if it breaks a rule."""
    k = n - 1
    route = [0]
    for t in range(1, n):
        here = [i for i in range(1, n) if bits >> var_index(i, t, n) & 1]
        if len(here) != 1:
            return None
        route.append(here[0])
    route.append(0)
    if len(set(route[1:-1])) != k:
        return None
    return route
