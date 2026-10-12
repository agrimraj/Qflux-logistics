"""Reproducible benchmark harness."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time

from shared.mapgen import euclidean_distance_matrix, generate_cities
from .solvers import held_karp, heuristic_tsp


def benchmark(sizes: list[int], seeds: list[int], exact_limit: int = 18,
              output: str | Path | None = None) -> list[dict]:
    rows: list[dict] = []
    for n in sizes:
        if n < 2:
            raise ValueError("all sizes must be at least 2")
        for seed in seeds:
            matrix = euclidean_distance_matrix(generate_cities(n, seed))
            heuristic = heuristic_tsp(matrix)
            exact = held_karp(matrix) if n <= exact_limit else None
            rows.append({
                "n": n,
                "seed": seed,
                "heuristic_length": heuristic.length,
                "heuristic_seconds": heuristic.elapsed_seconds,
                "exact_length": exact.length if exact else None,
                "exact_seconds": exact.elapsed_seconds if exact else None,
                "gap_percent": ((heuristic.length / exact.length - 1) * 100) if exact else None,
                "exact_skipped": exact is None,
            })
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".csv":
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        else:
            payload = {
                "metadata": {"python": sys.version, "platform": platform.platform(),
                             "generated_unix_time": time.time(), "exact_limit": exact_limit},
                "results": rows,
            }
            path.write_text(json.dumps(payload, indent=2) + "\n")
    return rows

