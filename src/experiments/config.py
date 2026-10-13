"""Fixed settings for the main result, so every run reproduces the same numbers."""

from pathlib import Path

from shared.realmap import PRESETS

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
HARDWARE_FILE = RESULTS_DIR / "hardware_run.json"
SUMMARY_FILE = RESULTS_DIR / "summary.json"

# ---- Main experiment: two vans on a real map (Kolkata, 10 places -> 8 qubits) ----
# Straight-line (great-circle) distances keep the main result reproducible offline
# and stable over time; road distances are available in the dashboard.
MAIN_MAP = {"kind": "real", "distance": "straight", "places": PRESETS["kolkata"]}
MAIN_MODE = "split"
MAIN_LAYERS = 1               # QAOA depth p
SHOTS = 4000                  # measurements per run
IDEAL_SAMPLE_SEED = 11        # seed for sampling the ideal distribution
NOISY_SEEDS = [11, 12, 13]    # three independent noisy-simulator runs
FAKE_BACKEND = "fake_kingston"

# ---- Supporting experiment: full-route encoding on made-up maps ----
ROUTE_CITIES = [4, 5]                 # 9 and 16 qubits
ROUTE_SEEDS = {4: list(range(1, 11)), 5: [1, 2, 3]}   # maps averaged over
ROUTE_LAYERS = [1, 2]

# ---- Classical scaling ----
SCALING_SIZES = [6, 7, 8, 9, 10, 11, 12, 13, 14]
SCALING_SEED = 1
BRUTE_LIMIT = 11
