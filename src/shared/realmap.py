"""Real-world maps: places with latitude/longitude, and real distances between them.

Two ways to measure distance:

* "straight" - great-circle ("as the crow flies") distance with the haversine
  formula. Works offline.
* "road"     - actual driving distance along streets, from the free OSRM
  routing service built on OpenStreetMap. Needs internet.

Online services used (both free, both ask for light, fair use):
    OSRM       https://router.project-osrm.org   (override with env OSRM_URL)
    Nominatim  https://nominatim.openstreetmap.org (override with env NOMINATIM_URL)
For anything beyond a demo, run your own OSRM server and point OSRM_URL at it.

Road distances can differ by direction (one-way streets). Our solvers assume
A->B equals B->A, so we use the average of the two directions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import urllib.parse
import urllib.request

USER_AGENT = "qaoa-delivery-routing/2.0 (IBM hackathon demo; https://github.com)"
TIMEOUT = 20


def osrm_url() -> str:
    return os.environ.get("OSRM_URL", "https://router.project-osrm.org").rstrip("/")


def nominatim_url() -> str:
    return os.environ.get("NOMINATIM_URL", "https://nominatim.openstreetmap.org").rstrip("/")


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float


def parse_places(raw: list[dict]) -> list[Place]:
    places = []
    for i, p in enumerate(raw):
        lat, lon = float(p["lat"]), float(p["lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError(f"place {i} has an invalid latitude/longitude")
        places.append(Place(str(p.get("name") or f"Stop {i}"), lat, lon))
    if len(places) < 3:
        raise ValueError("add at least 3 places (a depot and two stops)")
    return places


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------

def haversine_m(a: Place, b: Place) -> float:
    """Great-circle distance in metres."""
    r = 6_371_000.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp, dl = p2 - p1, math.radians(b.lon - a.lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def straight_matrix(places: list[Place]) -> list[list[float]]:
    return [[haversine_m(a, b) for b in places] for a in places]


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _coords(places: list[Place]) -> str:
    return ";".join(f"{p.lon:.6f},{p.lat:.6f}" for p in places)   # OSRM wants lon,lat


def road_matrix(places: list[Place]) -> list[list[float]]:
    """Driving distances in metres from OSRM, made symmetric."""
    data = _get_json(f"{osrm_url()}/table/v1/driving/{_coords(places)}?annotations=distance")
    if not isinstance(data, dict) or data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data.get('message') if isinstance(data, dict) else data}")
    d = data["distances"]
    n = len(places)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if d[i][j] is None or d[j][i] is None:
                raise RuntimeError(f"no road found between '{places[i].name}' and '{places[j].name}'")
            out[i][j] = out[j][i] = (d[i][j] + d[j][i]) / 2
    return out


def road_geometry(places: list[Place]) -> tuple[list[list[float]], float]:
    """The street-level path visiting `places` in order: ([[lat, lon], ...], metres)."""
    data = _get_json(f"{osrm_url()}/route/v1/driving/{_coords(places)}"
                     "?overview=full&geometries=geojson")
    if not isinstance(data, dict) or data.get("code") != "Ok":
        raise RuntimeError("OSRM could not build the route")
    route = data["routes"][0]
    return [[lat, lon] for lon, lat in route["geometry"]["coordinates"]], float(route["distance"])


def geocode(query: str, limit: int = 5) -> list[dict]:
    """Search for a place by name or address (OpenStreetMap Nominatim)."""
    q = urllib.parse.urlencode({"q": query, "format": "json", "limit": limit})
    data = _get_json(f"{nominatim_url()}/search?{q}")
    return [{"name": r.get("display_name", query).split(",")[0],
             "full_name": r.get("display_name", ""),
             "lat": float(r["lat"]), "lon": float(r["lon"])} for r in data]


# --------------------------------------------------------------------------
# A ready-made example (approximate locations of well-known places)
# --------------------------------------------------------------------------

PRESETS: dict[str, list[dict]] = {
    "kolkata": [
        {"name": "Depot: Sealdah", "lat": 22.5678, "lon": 88.3710},
        {"name": "Howrah Bridge", "lat": 22.5851, "lon": 88.3468},
        {"name": "Esplanade", "lat": 22.5646, "lon": 88.3511},
        {"name": "Park Street", "lat": 22.5525, "lon": 88.3525},
        {"name": "Victoria Memorial", "lat": 22.5448, "lon": 88.3426},
        {"name": "College Street", "lat": 22.5760, "lon": 88.3639},
        {"name": "Gariahat", "lat": 22.5180, "lon": 88.3665},
        {"name": "Kalighat", "lat": 22.5203, "lon": 88.3423},
        {"name": "Science City", "lat": 22.5401, "lon": 88.3960},
        {"name": "Salt Lake Sector V", "lat": 22.5760, "lon": 88.4330},
    ],
}
