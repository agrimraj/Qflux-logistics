"""Command-line interface.

    python cli.py solve --cities 10 --method both
    python cli.py solve --cities 4 --method qaoa
    python cli.py solve --example kolkata --method both          # real map, road distances
    python cli.py solve --places my_shops.json --distance road
    python cli.py benchmark --sizes 8,10,12 --output results.csv
    python cli.py ui
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import json

from classical.benchmark import benchmark
from shared.instance import build_instance
from shared.realmap import PRESETS
from shared.mapgen import euclidean_distance_matrix, generate_cities, load_instance, save_instance
from classical.solvers import brute_force, held_karp, heuristic_tsp


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(x) for x in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python cli.py", description="Classical Euclidean TSP baseline")
    sub = p.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="write a deterministic JSON instance")
    gen.add_argument("--cities", type=int, required=True)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--output", required=True)
    solve = sub.add_parser("solve", help="solve generated or saved instance")
    source = solve.add_mutually_exclusive_group(required=True)
    source.add_argument("--cities", type=int)
    source.add_argument("--input", help="saved made-up map (from 'generate')")
    source.add_argument("--places", help="real map: JSON list of {name, lat, lon}; first = depot")
    source.add_argument("--example", choices=sorted(PRESETS), help="built-in real map")
    solve.add_argument("--distance", choices=["road", "straight"], default="road",
                       help="real maps only: road (needs internet) or straight line")
    solve.add_argument("--seed", type=int, default=42)
    solve.add_argument("--method", choices=["exact", "heuristic", "both", "brute", "qaoa"],
                       default="both")
    solve.add_argument("--exact-limit", type=int, default=20)
    solve.add_argument("--starts", type=int)
    bench = sub.add_parser("benchmark", help="compare runtime and solution quality")
    bench.add_argument("--sizes", type=_csv_ints, default=[8, 10, 12, 14, 16])
    bench.add_argument("--seeds", type=_csv_ints, default=[1, 2, 3])
    bench.add_argument("--exact-limit", type=int, default=18)
    bench.add_argument("--output", default="benchmark.json")
    ui = sub.add_parser("ui", help="open the web dashboard in your browser")
    ui.add_argument("--port", type=int, default=8000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "generate":
        cities = generate_cities(args.cities, args.seed)
        save_instance(args.output, cities, args.seed)
        print(f"Wrote {len(cities)} cities to {args.output}")
        return 0
    if args.command == "solve":
        if args.places or args.example:
            raw = json.loads(Path(args.places).read_text()) if args.places else PRESETS[args.example]
            inst = build_instance({"kind": "real", "distance": args.distance, "places": raw})
            cities, matrix = inst.points, inst.matrix
            print(f"Real map: {len(cities)} places, {inst.distance_source}. " + " ".join(inst.notes),
                  file=sys.stderr)
        else:
            cities = load_instance(args.input) if args.input else generate_cities(args.cities, args.seed)
            matrix = euclidean_distance_matrix(cities)
        results = []
        if args.method in ("exact", "both"):
            if len(cities) > args.exact_limit:
                raise SystemExit(f"exact solve refused for n={len(cities)}; raise --exact-limit knowingly")
            results.append(held_karp(matrix))
        if args.method in ("heuristic", "both"):
            results.append(heuristic_tsp(matrix, args.starts))
        if args.method == "brute":
            if len(cities) > 12:
                raise SystemExit("brute force refused above 12 cities (it would take hours)")
            results.append(brute_force(matrix))
        if args.method == "qaoa":
            from quantum_simulation.simulator import solve_qaoa
            r = solve_qaoa(matrix)
            print(f"QAOA: {r.n_qubits} qubits, P(optimal)={r.p_optimal:.4f}, "
                  f"P(valid)={r.p_valid:.4f}, {r.circuit_evaluations} circuit evaluations")
            if r.solution:
                results.append(r.solution)
        print(json.dumps([{"method": s.method, "route": s.route, "length": s.length,
                           "seconds": s.elapsed_seconds, "optimal": s.optimal} for s in results], indent=2))
        return 0
    if args.command == "ui":
        from dashboard.server import run
        run(port=args.port)
        return 0
    rows = benchmark(args.sizes, args.seeds, args.exact_limit, args.output)
    print(f"Wrote {len(rows)} benchmark rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

