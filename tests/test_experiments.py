"""Checks on the main-result experiment against exact classical references."""

import unittest

from experiments import config as C
from experiments.main_result import binomial_se, exact_ideal, sweep_heuristic
from quantum_hardware.experiment import get_problem_for_map, random_reference


class MainExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = get_problem_for_map(C.MAIN_MODE, C.MAIN_MAP, C.MAIN_LAYERS)

    def test_uses_ten_kolkata_places_and_eight_qubits(self):
        self.assertEqual(self.problem.n, 10)
        self.assertEqual(self.problem.qubo.n_qubits, 8)

    def test_classical_optimum_checks_all_70_fair_splits(self):
        self.assertEqual(self.problem.baseline["splits_checked"], 70)    # C(8, 4)

    def test_qaoa_is_better_than_random_guessing(self):
        ideal, rnd = exact_ideal(self.problem), random_reference(self.problem)
        self.assertGreater(ideal["p_valid"], rnd["p_valid"])
        self.assertGreater(ideal["p_best"], rnd["p_best"])
        self.assertAlmostEqual(sum(self.problem.tuned.probs), 1.0, places=9)   # a valid quantum state

    def test_heuristic_never_beats_the_exact_optimum(self):
        self.assertGreaterEqual(sweep_heuristic(self.problem)["gap_percent"], -1e-9)

    def test_binomial_error(self):
        self.assertAlmostEqual(binomial_se(0.5, 100), 0.05)


if __name__ == "__main__":
    unittest.main()
