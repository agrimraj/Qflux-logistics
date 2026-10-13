"""One description of "the map" that every solver (classical or quantum) uses.

A map spec is a small dict, e.g.

    {"kind": "random", "cities": 10, "seed": 42}
    {"kind": "real", "distance": "road",
     "places": [{"name": "Depot", "lat": 22.56, "lon": 88.37}, ...]}

`build_instance(spec)` turns it into points (for drawing) plus a distance
matrix (for solving). The solvers only ever see the distance matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math

from .mapgen import euclidean_distance_matrix, generate_cities
from .realmap import parse_places, road_matrix, straight_matrix


@dataclass
class MapInstance:
    kind: str                    # "random" or "real"
    points: list[dict]           # [{id, name, x, y, lat?, lon?}], x/y in a 0..1000 box
    matrix: list[list[float]]    # distance between every pair of points
    units: str                   # "" (abstract units) or "m" (metres)
    distance_source: str
    seed: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.points)

    def summary(self) -> dict:
        return {"kind": self.kind, "units": self.units, "distance_source": self.distance_source,
                "seed": self.seed, "notes": self.notes, "points": self.points}


def spec_key(spec: dict) -> str:
    """A canonical string for a spec, so identical maps share one cache entry."""
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


_CACHE: dict[str, MapInstance] = {}


def build_instance(spec: dict) -> MapInstance:
    key = spec_key(spec)
    if key not in _CACHE:
        inst = _build(key)
        if inst.notes:            # a fallback was used: try the internet again next time
            return inst
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = inst
    return _CACHE[key]


def _build(key: str) -> MapInstance:
    spec = json.loads(key)
    kind = spec.get("kind", "random")
    if kind == "random":
        n, seed = int(spec.get("cities", 8)), int(spec.get("seed", 42))
        cities = generate_cities(n, seed)
        return MapInstance(
            "random", [{"id": c.id, "name": "Depot" if c.id == 0 else f"City {c.id}",
                        "x": c.x, "y": c.y} for c in cities],
            euclidean_distance_matrix(cities), "", "straight line on a made-up map", seed)
    if kind != "real":
        raise ValueError(f"unknown map kind '{kind}'")

    places = parse_places(spec.get("places", []))
    notes: list[str] = []
    source = "straight line (great-circle)"
    if spec.get("distance", "road") == "road":
        try:
            matrix = road_matrix(places)
            source = "road distance (OpenStreetMap / OSRM)"
        except Exception as exc:  # no internet, service busy, unreachable place...
            matrix = straight_matrix(places)
            notes.append(f"Road distances unavailable ({exc}); used straight-line distances instead.")
    else:
        matrix = straight_matrix(places)
    return MapInstance("real", _project(places), matrix, "m", source, None, notes)


def _project(places) -> list[dict]:
    """Flatten lat/lon into the dashboard's 0..1000 drawing box (keeps the shape)."""
    lat0 = sum(p.lat for p in places) / len(places)
    xs = [p.lon * math.cos(math.radians(lat0)) for p in places]
    ys = [-p.lat for p in places]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    ox = (1000 - (max(xs) - min(xs)) / span * 1000) / 2
    oy = (1000 - (max(ys) - min(ys)) / span * 1000) / 2
    return [{"id": i, "name": p.name, "lat": p.lat, "lon": p.lon,
             "x": ox + (xs[i] - min(xs)) / span * 1000, "y": oy + (ys[i] - min(ys)) / span * 1000}
            for i, p in enumerate(places)]
