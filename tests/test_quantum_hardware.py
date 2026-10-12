"""Hardware tests. The circuit tests run only if Qiskit is installed;
none of these tests contact IBM or need an API key."""

import importlib.util
import unittest

from quantum_hardware.experiment import get_problem, ideal_counts, summarise
from quantum_simulation.qubo import all_energies
from quantum_simulation.vehicle_split import split_groups

HAS_QISKIT = all(importlib.util.find_spec(m) for m in ("qiskit", "qiskit_ibm_runtime"))


class ExperimentTests(unittest.TestCase):
    def test_lowest_energy_split_respects_van_capacity(self):
        problem = get_problem("split", 8, 5, 1)
        energies = list(all_energies(problem.qubo))
        best = min(range(len(energies)), key=energies.__getitem__)
        _, van_b = split_groups(best, 8)
        self.assertEqual(len(van_b), 7 // 2)
        self.assertIsNotNone(problem.evaluate(best))

    def test_ideal_samples_are_scored(self):
        problem = get_problem("route", 4, 42, 1)
        result = summarise(problem, ideal_counts(problem, 2000))
        self.assertGreater(result["p_valid"], 0)
        self.assertGreaterEqual(result["gap_percent"], -1e-9)


@unittest.skipUnless(HAS_QISKIT, "Qiskit not installed")
class CircuitTests(unittest.TestCase):
    def test_circuit_matches_the_simulator(self):
        from qiskit.quantum_info import Statevector

        from quantum_hardware.circuits import qaoa_circuit

        problem = get_problem("route", 4, 42, 1)
        qc = qaoa_circuit(problem.qubo, problem.tuned.params, problem.tuned.scale)
        qc.remove_final_measurements()
        probs = Statevector(qc).probabilities()
        for a, b in zip(probs, problem.tuned.probs):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
