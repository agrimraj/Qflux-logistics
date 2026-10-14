"""Reproduce the main result:   python reproduce.py

Runs every experiment with fixed seeds, writes results/summary.json and the
figures in figures/. No IBM account or internet connection is needed.

    python reproduce.py --show             # ...and open the figures when done
    python reproduce.py --figures-only     # redraw figures from results/summary.json

If results/hardware_run.json exists (made by run_hardware.py), the real IBM
hardware run is added to the comparison automatically.
"""

import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from experiments import config as C  # noqa: E402
from experiments.figures import make_all  # noqa: E402
from experiments.main_result import run_main_experiment, run_route_experiment, run_scaling  # noqa: E402


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "qiskit", "qiskit-aer", "qiskit-ibm-runtime", "matplotlib"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def show(paths) -> None:
    """Open the figure files in the computer's default image viewer."""
    import os
    import subprocess
    for path in paths:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)            # macOS: Preview
        elif os.name == "nt":
            os.startfile(str(path))                                      # Windows
        else:
            subprocess.run(["xdg-open", str(path)], check=False)        # Linux


def main() -> None:
    if "--figures-only" in sys.argv:          # redraw figures from an existing summary.json
        figs = make_all(json.loads(C.SUMMARY_FILE.read_text()))
        print("Redrew " + ", ".join(str(f.relative_to(C.ROOT)) for f in figs))
        if "--show" in sys.argv:
            show(figs)
        return
    t0 = time.perf_counter()
    print("1/3  Two vans on the Kolkata map (ideal vs noisy simulator vs IBM hardware)...")
    main_res = run_main_experiment()
    print("2/3  Full-route encoding on made-up maps...")
    route = run_route_experiment()
    print("3/3  Classical solver scaling...")
    scaling = run_scaling()

    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "versions": versions(), "main": main_res, "route": route, "scaling": scaling}
    C.RESULTS_DIR.mkdir(exist_ok=True)
    C.SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n")
    figs = make_all(summary)

    m = main_res
    print("\nMain result: two vans, 10 Kolkata places, 8 qubits, p = 1")
    print(f"  classical optimum            {m['classical_exact']['length'] / 1000:.2f} km "
          f"({m['classical_exact']['splits_checked']} fair splits checked in "
          f"{m['classical_exact']['seconds'] * 1000:.1f} ms)")
    print(f"  classical sweep heuristic    {m['classical_heuristic']['length'] / 1000:.2f} km "
          f"(+{m['classical_heuristic']['gap_percent']:.1f}%)")
    print(f"  {'source':<28} {'valid shots':>12} {'optimal shots':>14}")
    rows = [("random guess", m["random"]["p_valid"], 0, m["random"]["p_best"], 0),
            ("ideal simulator", m["ideal_sampled"]["p_valid"], m["ideal_sampled"]["p_valid_se"],
             m["ideal_sampled"]["p_best"], m["ideal_sampled"]["p_best_se"]),
            ("noisy simulator (3 runs)", m["noisy"]["p_valid"], m["noisy"]["p_valid_sd"],
             m["noisy"]["p_best"], m["noisy"]["p_best_sd"])]
    if m["hardware"]:
        h = m["hardware"]
        rows.append((f"IBM {h['backend']}", h["p_valid"], h["p_valid_se"], h["p_best"], h["p_best_se"]))
    for name, pv, pve, pb, pbe in rows:
        print(f"  {name:<28} {pv * 100:>6.1f}% ±{pve * 100:<4.1f} {pb * 100:>7.2f}% ±{pbe * 100:.2f}")
    if not m["hardware"]:
        print("  (no IBM hardware run yet: see run_hardware.py)")
    elif not m["hardware"]["matches_current_experiment"]:
        print("  WARNING: results/hardware_run.json was made with different settings")
    print(f"\nWrote {C.SUMMARY_FILE.relative_to(C.ROOT)} and " +
          ", ".join(str(f.relative_to(C.ROOT)) for f in figs))
    print(f"Total time: {time.perf_counter() - t0:.0f} s")
    if "--show" in sys.argv:
        show(figs)
    else:
        print("Tip: add --show to open the figures, or open figures/main_result.png")


if __name__ == "__main__":
    main()
