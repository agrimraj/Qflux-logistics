"""Exact and approximate solvers for symmetric TSP."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class TSPSolution:
    method: str
    route: list[int]
    length: float
    elapsed_seconds: float
    optimal: bool


def _validate(matrix: list[list[float]]) -> int:
    n = len(matrix)
    if n < 2 or any(len(row) != n for row in matrix):
        raise ValueError("distance matrix must be square and at least 2x2")
    return n


def route_length(route: list[int], matrix: list[list[float]]) -> float:
    return sum(matrix[a][b] for a, b in zip(route, route[1:]))


def held_karp(matrix: list[list[float]]) -> TSPSolution:
    """Solve TSP exactly in O(n^2 2^n) time and O(n 2^n) memory.

    City 0 is fixed as the depot, removing rotational symmetry.
    """
    n = _validate(matrix)
    start_time = time.perf_counter()
    # (visited mask over cities 1..n-1, final city) -> (cost, predecessor)
    dp: dict[tuple[int, int], tuple[float, int]] = {}
    for j in range(1, n):
        dp[(1 << (j - 1), j)] = (matrix[0][j], 0)

    full = (1 << (n - 1)) - 1
    for size in range(2, n):
        # Enumerate masks of the desired population without external dependencies.
        for mask in range(1, full + 1):
            if mask.bit_count() != size:
                continue
            for j in range(1, n):
                bit = 1 << (j - 1)
                if not mask & bit:
                    continue
                previous = mask ^ bit
                best_cost, best_parent = math.inf, -1
                for k in range(1, n):
                    if previous & (1 << (k - 1)):
                        candidate = dp[(previous, k)][0] + matrix[k][j]
                        if candidate < best_cost:
                            best_cost, best_parent = candidate, k
                dp[(mask, j)] = (best_cost, best_parent)

    length, last = min((dp[(full, j)][0] + matrix[j][0], j) for j in range(1, n))
    backwards = [last]
    mask = full
    while True:
        parent = dp[(mask, last)][1]
        mask ^= 1 << (last - 1)
        if parent == 0:
            break
        backwards.append(parent)
        last = parent
    route = [0] + list(reversed(backwards)) + [0]
    return TSPSolution("held-karp", route, length, time.perf_counter() - start_time, True)


def _nearest_neighbor(matrix: list[list[float]], first: int) -> list[int]:
    n = len(matrix)
    route = [0, first]
    remaining = set(range(1, n)) - {first}
    while remaining:
        current = route[-1]
        nxt = min(remaining, key=lambda j: (matrix[current][j], j))
        route.append(nxt)
        remaining.remove(nxt)
    return route + [0]


def _two_opt(route: list[int], matrix: list[list[float]], max_passes: int) -> list[int]:
    route = route[:]
    n = len(route) - 1
    for _ in range(max_passes):
        best_delta, best_move = -1e-12, None
        for i in range(1, n - 1):
            a, b = route[i - 1], route[i]
            for k in range(i + 1, n):
                c, d = route[k], route[k + 1]
                delta = matrix[a][c] + matrix[b][d] - matrix[a][b] - matrix[c][d]
                if delta < best_delta:
                    best_delta, best_move = delta, (i, k)
        if best_move is None:
            break
        i, k = best_move
        route[i:k + 1] = reversed(route[i:k + 1])
    return route


def heuristic_tsp(matrix: list[list[float]], starts: int | None = None,
                  max_passes: int = 100) -> TSPSolution:
    """Deterministic multi-start nearest-neighbor followed by best-improvement 2-opt."""
    n = _validate(matrix)
    if max_passes < 0:
        raise ValueError("max_passes cannot be negative")
    start_time = time.perf_counter()
    count = n - 1 if starts is None else max(1, min(starts, n - 1))
    candidates = range(1, count + 1)
    best_route: list[int] | None = None
    best_length = math.inf
    for first in candidates:
        route = _two_opt(_nearest_neighbor(matrix, first), matrix, max_passes)
        length = route_length(route, matrix)
        if length < best_length:
            best_route, best_length = route, length
    assert best_route is not None
    return TSPSolution("nearest-neighbor+2-opt", best_route, best_length,
                       time.perf_counter() - start_time, False)



def brute_force(matrix: list[list[float]]) -> TSPSolution:
    """Check every possible tour, one by one (exhaustive search).

    With city 0 fixed there are (n-1)! orderings; each tour and its mirror image
    have the same length, so only (n-1)!/2 are checked. This is the slowest
    *honest* exact method: its runtime grows factorially, so it is the natural
    "try everything" baseline to compare against a quantum sampler.
    """
    import itertools

    n = _validate(matrix)
    start_time = time.perf_counter()
    if n <= 3:
        route = list(range(n)) + [0]
        return TSPSolution("brute-force", route, route_length(route, matrix),
                           time.perf_counter() - start_time, True)
    best_length, best_route = math.inf, None
    row0 = matrix[0]
    for perm in itertools.permutations(range(1, n)):
        if perm[0] > perm[-1]:          # skip mirror images
            continue
        length = row0[perm[0]] + row0[perm[-1]]
        for a, b in zip(perm, perm[1:]):
            length += matrix[a][b]
        if length < best_length:
            best_length, best_route = length, perm
    route = [0, *best_route, 0]
    return TSPSolution("brute-force", route, best_length,
                       time.perf_counter() - start_time, True)
