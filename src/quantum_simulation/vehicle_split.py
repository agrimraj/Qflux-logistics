"""Encode the two-van problem as a QUBO (1 qubit per stop).

Instead of asking the quantum computer for a whole route ((n-1)^2 qubits), we
only ask it which van delivers each stop. A classical solver then plans each
van's route. This needs far fewer qubits and much shallower circuits.
"""

from __future__ import annotations

from .qubo import QUBO


def build_split_qubo(matrix: list[list[float]]) -> QUBO:
    """Split stops 1..n-1 between van A and van B (each van has limited space).

    Rule (capacity): van B takes exactly floor(k/2) of the k stops, van A the rest.
    Goal: put stops on the same van when sharing a trip saves distance. The
    "saving" of serving i and j on one trip is the classic Clarke-Wright value
        s_ij = d(depot, i) + d(depot, j) - d(i, j).
    Energy = -(total saving inside each van) + penalty * (|B| - k//2)^2.

    Stop 1 is always in van A (removes the A/B mirror symmetry, saves a qubit).
    Qubit j means stop j+2; bit = 1 -> van B.
    """
    n = len(matrix)
    k = n - 1                 # stops
    m = n - 2                 # qubits
    target = k // 2
    sav = [[matrix[0][i] + matrix[0][j] - matrix[i][j] for j in range(n)] for i in range(n)]
    lin = [0.0] * m
    quad: dict[tuple[int, int], float] = {}
    offset = 0.0

    def add_q(a, b, w):
        key = (a, b) if a < b else (b, a)
        quad[key] = quad.get(key, 0.0) + w

    for i in range(1, n):
        for j in range(i + 1, n):
            w = -sav[i][j]          # reward (negative energy) for sharing a van
            if i == 1:              # same van iff x_j = 0
                offset += w
                lin[j - 2] -= w
            else:                   # [x_i == x_j] = 1 - x_i - x_j + 2 x_i x_j
                offset += w
                lin[i - 2] -= w
                lin[j - 2] -= w
                add_q(i - 2, j - 2, 2 * w)
    # Capacity penalty A * (sum x - target)^2, expanded for binary x.
    penalty = max(sum(abs(sav[i][j]) for j in range(1, n) if j != i) for i in range(1, n))
    offset += penalty * target * target
    for a in range(m):
        lin[a] += penalty * (1 - 2 * target)
        for b in range(a + 1, m):
            add_q(a, b, 2 * penalty)
    return QUBO(n, m, penalty, offset, lin, quad)


def split_groups(bits: int, n: int) -> tuple[list[int], list[int]]:
    a, b = [1], []
    for stop in range(2, n):
        (b if bits >> (stop - 2) & 1 else a).append(stop)
    return a, b
