"""Run the tuned QAOA circuit on a noisy simulator or on real IBM hardware.

API keys
--------
Nobody's key is stored in this project. Each person pastes their own IBM
Quantum API key into the dashboard (or sets the QISKIT_IBM_TOKEN environment
variable, or saves it once with QiskitRuntimeService.save_account). The key is
kept in memory only for the job you submit.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import threading
import time

from .circuits import DEFAULT_FAKE, _require_qiskit, compile_for, fake_backend, qaoa_circuit
from .experiment import Problem, summarise


def _counts_from_pub(pub_result) -> dict[int, int]:
    data = pub_result.data
    reg = getattr(data, "meas", None) or next(iter(data.values()))
    return {int(k, 2): v for k, v in reg.get_counts().items()}


def _noisy_counts(problem: Problem, shots: int, fake: str = DEFAULT_FAKE, seed: int = 3):
    """Simulate the circuit with the noise of a real IBM chip (no account needed)."""
    _require_qiskit()
    from qiskit_aer import AerSimulator

    backend = fake_backend(fake)
    qc = qaoa_circuit(problem.qubo, problem.tuned.params, problem.tuned.scale)
    isa, stats = compile_for(qc, backend)
    sim = AerSimulator.from_backend(backend)
    t0 = time.perf_counter()
    raw = sim.run(isa, shots=shots, seed_simulator=seed).result().get_counts()
    stats["seconds"] = time.perf_counter() - t0
    return {int(k.replace(" ", ""), 2): v for k, v in raw.items()}, stats



# --------------------------------------------------------------------------
# Real IBM hardware (asynchronous: jobs wait in a queue)
# --------------------------------------------------------------------------

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _service(token: str | None, instance: str | None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = (token or "").strip() or os.environ.get("QISKIT_IBM_TOKEN")
    kwargs = {}
    if instance and instance.strip():
        kwargs["instance"] = instance.strip()
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, **kwargs)
    try:  # fall back to an account saved with QiskitRuntimeService.save_account(...)
        return QiskitRuntimeService(**kwargs)
    except Exception as exc:
        raise ValueError("No IBM API key given. Paste your key, set QISKIT_IBM_TOKEN, "
                         "or save it with QiskitRuntimeService.save_account().") from exc


def _submit(problem: Problem, shots: int, token: str | None = None, instance: str | None = None,
           backend_name: str | None = None) -> dict:
    """Send the tuned circuit to IBM. Returns immediately with a job id."""
    _require_qiskit()
    from qiskit_ibm_runtime import SamplerV2

    backend_name = (backend_name or "").strip()
    if backend_name.startswith("fake_"):
        backend = fake_backend(backend_name)        # local rehearsal, no account
    else:
        service = _service(token, instance)
        if backend_name:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(operational=True, simulator=False,
                                         min_num_qubits=problem.qubo.n_qubits)
    qc = qaoa_circuit(problem.qubo, problem.tuned.params, problem.tuned.scale)
    isa, stats = compile_for(qc, backend)
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    if not backend_name.startswith("fake_"):
        # Free error-suppression: keep idle qubits steady and randomise gate noise.
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.num_randomizations = "auto"
    job = sampler.run([isa])
    job_id = job.job_id()
    with _LOCK:
        _JOBS[job_id] = {"job": job, "problem": problem, "stats": stats,
                         "submitted": time.time(), "local": backend_name.startswith("fake_")}
    return {"job_id": job_id, **stats}


def _job_status(job_id: str) -> dict:
    with _LOCK:
        entry = _JOBS.get(job_id)
    if entry is None:
        raise ValueError("unknown job id (jobs are forgotten when the dashboard restarts; "
                         "results stay visible at quantum.cloud.ibm.com)")
    job = entry["job"]
    status = job.status()
    status = getattr(status, "name", str(status))
    out = {"job_id": job_id, "status": status, "waited_seconds": time.time() - entry["submitted"],
           **entry["stats"]}
    if status in ("DONE", "COMPLETED"):
        if "summary" not in entry:
            counts = _counts_from_pub(job.result()[0])
            entry["summary"] = summarise(entry["problem"], counts)
            try:
                entry["qpu_seconds"] = job.usage()
            except Exception:
                entry["qpu_seconds"] = None
        out["summary"] = entry["summary"]
        out["qpu_seconds"] = entry["qpu_seconds"]
    elif status in ("ERROR", "CANCELLED", "FAILED"):
        try:
            out["error"] = str(job.error_message())
        except Exception:
            out["error"] = "job failed"
    return out


# --------------------------------------------------------------------------
# Qiskit's compiled core crashes if it is used from several different threads
# (the dashboard answers each click on a new thread). So every Qiskit call
# goes through this one long-lived worker thread.
# --------------------------------------------------------------------------

_QISKIT_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qiskit")


def noisy_counts(problem: Problem, shots: int, fake: str = DEFAULT_FAKE, seed: int = 3):
    return _QISKIT_THREAD.submit(_noisy_counts, problem, shots, fake, seed).result()


def submit(problem: Problem, shots: int, token: str | None = None, instance: str | None = None,
           backend_name: str | None = None) -> dict:
    return _QISKIT_THREAD.submit(_submit, problem, shots, token, instance, backend_name).result()


def job_status(job_id: str) -> dict:
    return _QISKIT_THREAD.submit(_job_status, job_id).result()
