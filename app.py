"""Start the dashboard:   python app.py   [port]

Then open http://127.0.0.1:8000 (your browser opens it automatically).
"""
import sys
from pathlib import Path

# Make the project folders importable no matter where this is run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard.server import run  # noqa: E402

if __name__ == "__main__":
    run(port=int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
