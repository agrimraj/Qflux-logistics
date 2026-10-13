"""Real-map tests. A tiny fake OSRM/Nominatim server stands in for the internet,
so these tests run offline and check that we read the real APIs' format correctly."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import threading
import unittest
from unittest import mock

from shared import realmap
from shared.instance import build_instance
from shared.realmap import PRESETS, Place, haversine_m, parse_places

PLACES = [{"name": "Depot", "lat": 22.5678, "lon": 88.3710},
          {"name": "A", "lat": 22.5851, "lon": 88.3468},
          {"name": "B", "lat": 22.5448, "lon": 88.3426}]


class FakeOSM(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/table/v1/driving/"):
            n = len(self.path.split("/driving/")[1].split("?")[0].split(";"))
            if n == 3:   # deliberately asymmetric, like one-way streets
                body = {"code": "Ok", "distances": [[0, 4000, 5000], [4200, 0, 6000], [5000, 6400, 0]]}
            else:        # an irregular street network: every pair stretched differently
                body = {"code": "Ok", "distances": [[0 if i == j else 1000.0 * (3 + (i * 7 + j * 7) % 5)
                                                     for j in range(n)] for i in range(n)]}
        elif self.path.startswith("/route/v1/driving/"):
            body = {"code": "Ok", "routes": [{"distance": 9000.0, "geometry": {
                "type": "LineString", "coordinates": [[88.371, 22.5678], [88.36, 22.57], [88.3468, 22.5851]]}}]}
        elif self.path.startswith("/search?"):
            body = [{"display_name": "Victoria Memorial, Kolkata, India", "lat": "22.5448", "lon": "88.3426"}]
        else:
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


class RealMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), FakeOSM)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.env = mock.patch.dict(os.environ, {"OSRM_URL": url, "NOMINATIM_URL": url})
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.server.shutdown()

    def test_haversine_known_distance(self):
        # Kolkata -> Delhi is about 1,300 km in a straight line
        d = haversine_m(Place("Kolkata", 22.5726, 88.3639), Place("Delhi", 28.6139, 77.2090))
        self.assertAlmostEqual(d / 1000, 1305, delta=15)

    def test_road_matrix_is_made_symmetric(self):
        m = realmap.road_matrix(parse_places(PLACES))
        self.assertEqual(m[0][1], 4100)          # average of 4000 and 4200
        self.assertEqual(m[1][2], m[2][1])

    def test_route_geometry_is_lat_lon(self):
        path, metres = realmap.road_geometry(parse_places(PLACES))
        self.assertEqual(path[0], [22.5678, 88.371])
        self.assertEqual(metres, 9000.0)

    def test_geocode(self):
        hits = realmap.geocode("Victoria Memorial")
        self.assertEqual(hits[0]["name"], "Victoria Memorial")
        self.assertAlmostEqual(hits[0]["lat"], 22.5448)

    def test_instance_uses_road_distances(self):
        inst = build_instance({"kind": "real", "distance": "road", "places": PLACES})
        self.assertEqual(inst.units, "m")
        self.assertIn("road", inst.distance_source)
        self.assertEqual(inst.notes, [])

    def test_falls_back_to_straight_line_without_internet(self):
        with mock.patch.dict(os.environ, {"OSRM_URL": "http://127.0.0.1:9"}):
            inst = build_instance({"kind": "real", "distance": "road", "places": PRESETS["kolkata"]})
        self.assertIn("straight", inst.distance_source)
        self.assertTrue(inst.notes)

    def test_dashboard_solves_a_real_map(self):
        from dashboard.server import solve_request
        out = solve_request({"map": {"kind": "real", "distance": "straight", "places": PRESETS["kolkata"]},
                             "methods": ["heuristic", "held-karp"]})
        self.assertEqual(out["map"]["units"], "m")
        self.assertEqual(len(out["cities"]), 10)
        self.assertTrue(all(r["route"][0] == 0 for r in out["results"]))

    def test_quantum_hardware_uses_street_distances(self):
        """On a real map with "road", the QUBO, and therefore the circuit sent to IBM,
        is built from the street distances, not straight lines."""
        from dashboard.server import hw_simulate
        from quantum_hardware.experiment import get_problem_for_map

        places = PRESETS["kolkata"][:6]
        road = {"kind": "real", "distance": "road", "places": places}
        straight = {**road, "distance": "straight"}
        info = hw_simulate({"mode": "split", "layers": 1, "shots": 500, "map": road})
        self.assertIn("road", info["map"]["distance_source"])
        p_road = get_problem_for_map("split", road, 1)
        p_straight = get_problem_for_map("split", straight, 1)
        self.assertEqual(p_road.matrix[1][2], 1000.0 * (3 + 21 % 5))   # the fake street distance
        self.assertNotEqual(p_road.qubo.linear, p_straight.qubo.linear)
        self.assertNotAlmostEqual(p_road.baseline["length"], p_straight.baseline["length"])

    def test_rejects_bad_coordinates(self):
        with self.assertRaises(ValueError):
            parse_places([{"lat": 200, "lon": 0}] * 3)


if __name__ == "__main__":
    unittest.main()
