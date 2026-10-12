"""Code used by both the classical and the quantum sides.

    mapgen.py   reproducible random maps (same seed -> same map) and distances
    scaling.py  where each approach stops being practical (measured + projected)
"""

from .mapgen import City, euclidean_distance_matrix, generate_cities, load_instance, save_instance

__all__ = ["City", "euclidean_distance_matrix", "generate_cities", "load_instance", "save_instance"]
