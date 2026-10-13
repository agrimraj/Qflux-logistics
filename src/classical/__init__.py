"""Classical solvers for the delivery-route problem (TSP).

    solvers.py    heuristic (nearest neighbour + 2-opt), Held-Karp, brute force
    benchmark.py  repeatable JSON/CSV benchmarks with run times and gaps
"""

from .solvers import TSPSolution, brute_force, held_karp, heuristic_tsp, route_length

__all__ = ["TSPSolution", "brute_force", "held_karp", "heuristic_tsp", "route_length"]
