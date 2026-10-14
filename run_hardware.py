"""Run the main-result circuit on real IBM quantum hardware (optional bonus).

    python run_hardware.py                       # asks for your API key, uses the least busy device
    python run_hardware.py --backend ibm_kingston
    python run_hardware.py --job-id <ID>         # re-download a job you already ran
    python run_hardware.py --backend fake_kingston   # offline rehearsal, no account needed

Your IBM Quantum API key is NEVER written to any file. The script uses, in order:
  1. the QISKIT_IBM_TOKEN environment variable, if set;
  2. an account saved earlier with QiskitRuntimeService.save_account(...);
  3. otherwise it asks you to paste the key (typing is hidden).

The result is saved to results/hardware_run.json. Then run `python reproduce.py`
to add it to the comparison and the main figure.
"""

import argparse
from getpass import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from experiments import config as C  # noqa: E402


def has_saved_account() -> bool:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        return bool(QiskitRuntimeService.saved_accounts())
    except Exception:
        return False


def get_token(rehearsal: bool) -> str | None:
    if rehearsal:
        return None
    token = os.environ.get("QISKIT_IBM_TOKEN", "").strip()
    if token:
        print("Using the API key from the QISKIT_IBM_TOKEN environment variable.")
        return token
    if has_saved_account():
        print("Using your saved IBM Quantum account.")
        return None
    print("No API key found. Get one at https://quantum.cloud.ibm.com (it is not saved anywhere).")
    token = getpass("Paste your IBM Quantum API key and press Enter (nothing will show while you paste): ").strip()
    if not token:
        sys.exit("No key entered, so nothing was sent to IBM.")
    return token


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", help="device name, e.g. ibm_kingston (default: least busy)")
    p.add_argument("--shots", type=int, default=C.SHOTS)
    p.add_argument("--instance", help="IBM Cloud instance CRN (optional)")
    p.add_argument("--job-id", help="download the results of an existing job instead of submitting")
    a = p.parse_args()

    rehearsal = bool(a.backend and a.backend.startswith("fake_"))
    token = get_token(rehearsal)
    from quantum_hardware.run_job import run  # imported late so --help works without Qiskit

    try:
        run(a.backend, a.shots, a.instance, a.job_id, token=token)
    except KeyboardInterrupt:
        sys.exit("\nStopped. If a job was submitted, re-download it later with --job-id <ID>.")
    except Exception as exc:  # show a short message instead of a long traceback
        msg = str(exc)
        if "token" in msg.lower() or "instances" in msg.lower() or "account" in msg.lower():
            msg += "\n-> Check that you pasted the whole API key (no spaces), then try again."
        sys.exit(f"\nHardware run failed: {type(exc).__name__}: {msg}")
