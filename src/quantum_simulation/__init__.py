"""Quantum side, simulated on a normal computer (no quantum hardware needed).

    qubo.py           TSP route -> QUBO, with (n-1)^2 qubits
    vehicle_split.py  two-van split -> QUBO, with 1 qubit per stop
    simulator.py      exact state-vector QAOA simulator + angle tuning
"""

from .qubo import QUBO, all_energies, build_qubo, decode, qubo_energy, var_index
from .simulator import MAX_QAOA_CITIES, QAOAResult, TunedQAOA, solve_qaoa, tune_qaoa
from .vehicle_split import build_split_qubo, split_groups

__all__ = [
    "QUBO", "QAOAResult", "TunedQAOA", "MAX_QAOA_CITIES", "all_energies", "build_qubo",
    "build_split_qubo", "decode", "qubo_energy", "solve_qaoa", "split_groups", "tune_qaoa", "var_index",
]
