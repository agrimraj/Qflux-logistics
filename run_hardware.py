"""Run the main-result circuit on real IBM quantum hardware (optional bonus).

Needs an IBM Quantum account. Your API key is read from the environment and is
NEVER written to any file:

    export QISKIT_IBM_TOKEN="your-key"          # Windows: set QISKIT_IBM_TOKEN=your-key
    python run_hardware.py                       # least busy real device
    python run_hardware.py --backend ibm_kingston
    python run_hardware.py --job-id <ID>         # re-download a job you already ran
    python run_hardware.py --backend fake_kingston   # offline rehearsal, no account

The result is saved to results/hardware_run.json. Then run `python reproduce.py`
to add it to the comparison and the main figure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from experiments import config as C  # noqa: E402
from quantum_hardware.run_job import run  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", help="device name, e.g. ibm_kingston (default: least busy)")
    p.add_argument("--shots", type=int, default=C.SHOTS)
    p.add_argument("--instance", help="IBM Cloud instance CRN (optional)")
    p.add_argument("--job-id", help="download the results of an existing job instead of submitting")
    a = p.parse_args()
    run(a.backend, a.shots, a.instance, a.job_id)
