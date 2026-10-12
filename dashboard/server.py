"""A tiny local web dashboard. Uses only the Python standard library.

Run it with:   python app.py        (from the project folder)
          or:  python cli.py ui
Then open http://127.0.0.1:8000 in a browser (it opens automatically).
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import traceback
import webbrowser

from shared.scaling import breakdown_table, measured_scaling
from shared.instance import build_instance
from shared.realmap import PRESETS, Place, geocode, road_geometry
from quantum_simulation.qubo import build_qubo
from quantum_simulation.simulator import MAX_QAOA_CITIES, solve_qaoa
from classical.solvers import brute_force, held_karp, heuristic_tsp

WEB = Path(__file__).resolve().parent / "web"
PAGE = WEB / "index.html"
STATIC_TYPES = {".js": "text/javascript", ".css": "text/css"}
LIMITS = {"brute-force": 12, "held-karp": 20, "qaoa": MAX_QAOA_CITIES, "heuristic": 500}


def _solution_dict(sol, extra=None):
    d = {"method": sol.method, "route": sol.route, "length": sol.length,
         "seconds": sol.elapsed_seconds, "optimal": sol.optimal}
    if extra:
        d.update(extra)
    return d


def map_spec(body: dict) -> dict:
    """The map to solve: a real map sent by the page, or a random one from cities + seed."""
    spec = body.get("map")
    if spec and spec.get("kind") == "real":
        return {"kind": "real", "distance": spec.get("distance", "road"),
                "places": [{"name": p.get("name", ""), "lat": round(float(p["lat"]), 6),
                            "lon": round(float(p["lon"]), 6)} for p in spec.get("places", [])]}
    return {"kind": "random", "cities": int(body.get("cities", 8)), "seed": int(body.get("seed", 42))}


def solve_request(body: dict) -> dict:
    methods = body.get("methods") or ["heuristic", "held-karp"]
    inst = build_instance(map_spec(body))
    n, matrix = inst.n, inst.matrix
    if not 3 <= n <= 500:
        raise ValueError("the map needs between 3 and 500 places")
    results, skipped = [], []
    for method in methods:
        if n > LIMITS.get(method, 0):
            skipped.append({"method": method,
                            "reason": f"limited to {LIMITS[method]} cities"})
            continue
        if method == "heuristic":
            results.append(_solution_dict(heuristic_tsp(matrix)))
        elif method == "held-karp":
            results.append(_solution_dict(held_karp(matrix)))
        elif method == "brute-force":
            results.append(_solution_dict(brute_force(matrix)))
        elif method == "qaoa":
            r = solve_qaoa(matrix, layers=int(body.get("layers", 2)),
                           shots=int(body.get("shots", 1024)))
            extra = {k: getattr(r, k) for k in (
                "n_qubits", "layers", "shots", "circuit_evaluations", "params", "p_optimal",
                "p_valid", "p_optimal_random", "valid_shots", "expected_shots_to_optimal",
                "statevector_bytes", "backend", "convergence")}
            if r.solution is None:
                skipped.append({"method": "qaoa", "reason": "no valid tour in any shot", **extra})
            else:
                results.append(_solution_dict(r.solution, extra))
    best = min((r["length"] for r in results if r["optimal"]), default=None)
    for r in results:
        r["gap_percent"] = (r["length"] / best - 1) * 100 if best else None
    q = (n - 1) ** 2
    return {"n": n, "seed": inst.seed, "cities": inst.points, "map": inst.summary(),
            "results": results, "skipped": skipped,
            "qubo": {"qubits": q,
                     "terms": len(build_qubo(matrix).quadratic) if n <= 12 else None}}


def _to_json(obj):
    """numpy numbers (e.g. numpy.float64) are not JSON-serialisable; convert them."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def hw_problem(body: dict):
    import quantum_hardware as hardware
    return hardware, hardware.get_problem_for_map(body.get("mode", "split"), map_spec(body),
                                                  int(body.get("layers", 1)))


def geometry_request(body: dict) -> dict:
    """Street-level path for one route (a list of places in visiting order)."""
    places = [Place(str(p.get("name", "")), float(p["lat"]), float(p["lon"]))
              for p in body.get("places", [])]
    if len(places) < 2 or len({(p.lat, p.lon) for p in places}) < 2:
        return {"path": [[p.lat, p.lon] for p in places], "source": "straight"}
    if body.get("distance", "road") == "road":
        try:
            path, metres = road_geometry(places)
            return {"path": path, "metres": metres, "source": "road"}
        except Exception as exc:
            return {"path": [[p.lat, p.lon] for p in places], "source": "straight",
                    "note": f"road shape unavailable: {exc}"}
    return {"path": [[p.lat, p.lon] for p in places], "source": "straight"}


def hw_simulate(body: dict) -> dict:
    hardware, problem = hw_problem(body)
    shots = max(100, min(int(body.get("shots", 4000)), 20000))
    out = hardware.problem_info(problem)
    out["ideal"] = hardware.summarise(problem, hardware.ideal_counts(problem, shots))
    if body.get("noisy"):
        if problem.qubo.n_qubits > 12:
            out["noisy_error"] = "noisy simulation is limited to 12 qubits (it gets very slow)"
        else:
            try:
                counts, stats = hardware.noisy_counts(problem, shots, body.get("fake") or hardware.DEFAULT_FAKE)
                out["noisy"] = {**hardware.summarise(problem, counts), **stats}
            except RuntimeError as exc:          # qiskit not installed
                out["noisy_error"] = str(exc)
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _send(self, status: int, payload, content_type="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload, default=_to_json).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif self.path.startswith("/api/breakdown"):
            self._send(200, breakdown_table(30))
        elif self.path == "/api/presets":
            self._send(200, PRESETS)
        elif self.path.startswith("/api/geocode"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query).get("q", [""])[0].strip()
            try:
                self._send(200, geocode(q) if q else [])
            except Exception as exc:
                self._send(502, {"error": f"place search needs internet ({type(exc).__name__}: {exc})"})
        elif self.path.startswith("/vendor/"):
            f = (WEB / self.path.lstrip("/")).resolve()
            if WEB in f.parents and f.is_file() and f.suffix in STATIC_TYPES:
                self._send(200, f.read_bytes(), STATIC_TYPES[f.suffix])
            else:
                self._send(404, {"error": "not found"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/solve":
                self._send(200, solve_request(body))
            elif self.path == "/api/scaling":
                sizes = [int(s) for s in body.get("sizes", [6, 8, 10, 11, 12, 14, 16])]
                sizes = [s for s in sizes if 3 <= s <= 20]
                self._send(200, measured_scaling(sizes, int(body.get("seed", 1)),
                                                 brute_limit=int(body.get("brute_limit", 11))))
            elif self.path == "/api/geometry":
                self._send(200, geometry_request(body))
            elif self.path == "/api/hw/simulate":
                self._send(200, hw_simulate(body))
            elif self.path == "/api/hw/submit":
                hardware, problem = hw_problem(body)
                self._send(200, hardware.submit(
                    problem, shots=max(100, min(int(body.get("shots", 4000)), 20000)),
                    token=body.get("token"), instance=body.get("instance"),
                    backend_name=body.get("backend")))
            elif self.path == "/api/hw/status":
                from quantum_hardware import job_status
                self._send(200, job_status(str(body.get("job_id", ""))))
            else:
                self._send(404, {"error": "not found"})
        except (ValueError, RuntimeError) as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # show the error in the page instead of crashing
            traceback.print_exc()
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def run(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"TSP dashboard running at {url}  (press Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
