"""Run the QAOA circuit on IBM quantum hardware, or on a noisy copy of it.

    circuits.py    QUBO + tuned angles -> Qiskit circuit, compiled for a chip
    experiment.py  one map + encoding + tuned angles + classical baseline, and
                   how measured bitstrings are scored
    ibm_runner.py  noisy simulation, submitting jobs to IBM, checking job status

Needs:  pip install -r requirements-hardware.txt
"""

from .circuits import DEFAULT_FAKE, qaoa_circuit, to_ising
from .experiment import Problem, get_problem, get_problem_for_map, ideal_counts, problem_info, random_reference, summarise
from .ibm_runner import job_status, noisy_counts, submit

__all__ = [
    "DEFAULT_FAKE", "Problem", "get_problem", "get_problem_for_map", "ideal_counts", "job_status", "noisy_counts",
    "problem_info", "qaoa_circuit", "random_reference", "submit", "summarise", "to_ising",
]
