"""Run the main-result circuit on a real IBM quantum computer and save the raw counts.

Called by ../../run_hardware.py. The saved file (results/hardware_run.json) holds
the backend name, date, job ID and every measured bitstring, so the comparison in
reproduce.py can be re-checked later without re-running the job.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import time

from experiments import config as C

from .circuits import _require_qiskit, compile_for, fake_backend, qaoa_circuit
from .experiment import get_problem_for_map, summarise
from .ibm_runner import _counts_from_pub, _service


def _save(path, record: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def run(backend_name: str | None = None, shots: int = C.SHOTS, instance: str | None = None,
        job_id: str | None = None, log=print) -> dict:
    _require_qiskit()
    from qiskit_ibm_runtime import SamplerV2

    problem = get_problem_for_map(C.MAIN_MODE, C.MAIN_MAP, C.MAIN_LAYERS)
    qc = qaoa_circuit(problem.qubo, problem.tuned.params, problem.tuned.scale)
    rehearsal = bool(backend_name and backend_name.startswith("fake_"))

    if rehearsal:
        backend = fake_backend(backend_name)
        log(f"Offline rehearsal on {backend.name} (a noisy copy, NOT real hardware)")
    else:
        service = _service(None, instance)          # key comes from QISKIT_IBM_TOKEN or a saved account
        if job_id:
            job = service.job(job_id)
            backend = job.backend()
            log(f"Fetching existing job {job_id} from {backend.name}")
        elif backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False,
                                         min_num_qubits=problem.qubo.n_qubits)
    isa, stats = compile_for(qc, backend)
    log(f"Circuit: {problem.qubo.n_qubits} qubits, {stats['two_qubit_gates']} two-qubit gates, "
        f"depth {stats['depth']} on {backend.name}")

    if not job_id:
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = shots
        if not rehearsal:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XpXm"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"
        job = sampler.run([isa])
        job_id = job.job_id()
        log(f"Submitted job {job_id}. Waiting (real devices have a queue)...")

    last = None
    while True:
        status = job.status()
        status = getattr(status, "name", str(status))
        if status != last:
            log(f"  status: {status}")
            last = status
        if status in ("DONE", "COMPLETED"):
            break
        if status in ("ERROR", "CANCELLED", "FAILED"):
            raise RuntimeError(f"job {job_id} ended with status {status}")
        time.sleep(5)

    pub = job.result()[0]
    counts = _counts_from_pub(pub)
    try:
        qpu_seconds = job.usage()
    except Exception:
        qpu_seconds = None
    try:
        created = job.creation_date.astimezone(timezone.utc).isoformat(timespec="seconds")
    except Exception:
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")

    record = {
        "backend": backend.name, "real_hardware": not rehearsal, "job_id": job_id,
        "date_utc": created, "shots": sum(counts.values()), "qpu_seconds": qpu_seconds,
        "two_qubit_gates": stats["two_qubit_gates"], "depth": stats["depth"],
        "error_suppression": None if rehearsal else "dynamical decoupling (XpXm) + gate twirling",
        "map_spec": C.MAIN_MAP, "mode": C.MAIN_MODE, "layers": C.MAIN_LAYERS,
        "params": problem.tuned.params,
        "counts": {str(k): v for k, v in sorted(counts.items())},
    }
    path = C.RESULTS_DIR / ("hardware_rehearsal.json" if rehearsal else "hardware_run.json")
    _save(path, record)
    s = summarise(problem, counts)
    log(f"Valid shots {s['p_valid'] * 100:.1f}%, optimal-plan shots {s['p_best'] * 100:.2f}%, "
        f"best plan {s['best_length'] / 1000:.2f} km (gap {s['gap_percent']:.1f}%)")
    log(f"Saved {path.relative_to(C.ROOT)}")
    return record
