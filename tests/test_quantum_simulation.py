import unittest

from classical.solvers import held_karp
from quantum_simulation.qubo import build_qubo, decode, qubo_energy, var_index
from quantum_simulation.simulator import solve_qaoa
from quantum_simulation.vehicle_split import build_split_qubo, split_groups
from shared.mapgen import euclidean_distance_matrix, generate_cities


class QuboTests(unittest.TestCase):
    def test_energy_of_optimal_tour_equals_its_length(self):
        matrix = euclidean_distance_matrix(generate_cities(4, 2))
        best = held_karp(matrix)
        bits = sum(1 << var_index(c, t, 4) for t, c in enumerate(best.route[1:-1], 1))
        self.assertEqual(decode(bits, 4), best.route)
        self.assertAlmostEqual(qubo_energy(build_qubo(matrix), bits), best.length)
        self.assertIsNone(decode(0, 4))  # an empty bitstring is not a tour

    def test_vehicle_split_uses_one_qubit_per_stop(self):
        matrix = euclidean_distance_matrix(generate_cities(8, 5))
        q = build_split_qubo(matrix)
        self.assertEqual(q.n_qubits, 6)          # 7 stops, stop 1 fixed in van A
        a, b = split_groups(0, 8)
        self.assertEqual(a, list(range(1, 8)))
        self.assertEqual(b, [])


class SimulatorTests(unittest.TestCase):
    def test_qaoa_beats_random_guessing(self):
        r = solve_qaoa(euclidean_distance_matrix(generate_cities(4, 3)), layers=1)
        self.assertEqual(r.n_qubits, 9)
        self.assertGreater(r.p_optimal, r.p_optimal_random)
        self.assertIsNotNone(r.solution)

    def test_refuses_sizes_a_laptop_cannot_simulate(self):
        with self.assertRaises(ValueError):
            solve_qaoa(euclidean_distance_matrix(generate_cities(6, 1)))


if __name__ == "__main__":
    unittest.main()
