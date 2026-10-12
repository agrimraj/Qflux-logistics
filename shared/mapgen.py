"""Deterministic Euclidean TSP instance generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path


@dataclass(frozen=True)
class City:
    id: int
    x: float
    y: float


def generate_cities(n: int, seed: int = 42, width: float = 1000.0,
                    height: float = 1000.0) -> list[City]:
    """Generate reproducible points using a local PRNG (no global-state changes)."""
    if n < 2:
        raise ValueError("n must be at least 2")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    rng = random.Random(seed)
    return [City(i, rng.uniform(0, width), rng.uniform(0, height)) for i in range(n)]


def euclidean_distance_matrix(cities: list[City]) -> list[list[float]]:
    n = len(cities)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(cities[i].x - cities[j].x, cities[i].y - cities[j].y)
            matrix[i][j] = matrix[j][i] = d
    return matrix


def save_instance(path: str | Path, cities: list[City], seed: int | None = None) -> None:
    Path(path).write_text(json.dumps({"seed": seed, "cities": [asdict(c) for c in cities]}, indent=2) + "\n")


def load_instance(path: str | Path) -> list[City]:
    data = json.loads(Path(path).read_text())
    raw = data["cities"] if isinstance(data, dict) else data
    cities = [City(int(c.get("id", i)), float(c["x"]), float(c["y"])) for i, c in enumerate(raw)]
    if len(cities) < 2:
        raise ValueError("instance must contain at least two cities")
    if [c.id for c in cities] != list(range(len(cities))):
        raise ValueError("city ids must be consecutive integers starting at 0")
    return cities

