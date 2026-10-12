import json
import tempfile
import unittest
from pathlib import Path

from classical.benchmark import benchmark
from classical.solvers import brute_force, held_karp, heuristic_tsp, route_length
from shared.mapgen import City, euclidean_distance_matrix, generate_cities, load_instance, save_instance


class MapTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        self.assertEqual(generate_cities(10, 7), generate_cities(10, 7))
        self.assertNotEqual(generate_cities(10, 7), generate_cities(10, 8))

    def test_instance_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            cities = generate_cities(5, 4)
            save_instance(path, cities, 4)
            self.assertEqual(load_instance(path), cities)


class SolverTests(unittest.TestCase):
    def test_unit_square(self):
        square = [City(0, 0, 0), City(1, 1, 0), City(2, 1, 1), City(3, 0, 1)]
        solution = held_karp(euclidean_distance_matrix(square))
        self.assertAlmostEqual(solution.length, 4.0)
        self.assertEqual(solution.route[0], 0)
        self.assertEqual(solution.route[-1], 0)
        self.assertEqual(set(solution.route[:-1]), set(range(4)))

    def test_heuristic_is_valid_and_not_better_than_optimum(self):
        matrix = euclidean_distance_matrix(generate_cities(9, 99))
        exact, heuristic = held_karp(matrix), heuristic_tsp(matrix)
        self.assertEqual(len(heuristic.route), 10)
        self.assertEqual(len(set(heuristic.route[:-1])), 9)
        self.assertAlmostEqual(heuristic.length, route_length(heuristic.route, matrix))
        self.assertGreaterEqual(heuristic.length + 1e-9, exact.length)

    def test_brute_force_matches_held_karp(self):
        for n in (4, 6, 8):
            matrix = euclidean_distance_matrix(generate_cities(n, n))
            self.assertAlmostEqual(brute_force(matrix).length, held_karp(matrix).length)

    def test_benchmark_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            rows = benchmark([5], [4], 5, output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(json.loads(output.read_text())["results"]), 1)


if __name__ == "__main__":
    unittest.main()
