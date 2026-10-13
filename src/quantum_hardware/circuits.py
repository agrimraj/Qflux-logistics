"""Turn a QUBO plus tuned angles into a Qiskit circuit for IBM hardware.

Needs:  pip install qiskit qiskit-ibm-runtime qiskit-aer
"""

from __future__ import annotations

from quantum_simulation.qubo import QUBO

INSTALL_HINT = "Install the IBM packages first:  pip install -r requirements-hardware.txt"
DEFAULT_FAKE = "fake_kingston"   # a local copy of IBM's 156-qubit Heron r2 chip


def _require_qiskit():
    try:
        import qiskit  # noqa: F401
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc



def to_ising(q: QUBO) -> tuple[list[float], dict[tuple[int, int], float]]:
    """Rewrite x in {0,1} as x = (1 - z)/2 with z in {+1,-1} (Pauli Z)."""
    h = [-w / 2 for w in q.linear]
    J: dict[tuple[int, int], float] = {}
    for (a, b), w in q.quadratic.items():
        J[(a, b)] = J.get((a, b), 0.0) + w / 4
        h[a] -= w / 4
        h[b] -= w / 4
    return h, J


def qaoa_circuit(q: QUBO, params: list[float], scale: float):
    """The QAOA circuit, matching the laptop simulator gate for gate."""
    from qiskit import QuantumCircuit

    h, J = to_ising(q)
    p = len(params) // 2
    qc = QuantumCircuit(q.n_qubits)
    qc.h(range(q.n_qubits))
    for layer in range(p):
        gamma, beta = params[layer], params[p + layer]
        for (a, b), w in J.items():
            qc.rzz(2 * gamma * w / scale, a, b)
        for a, w in enumerate(h):
            if w:
                qc.rz(2 * gamma * w / scale, a)
        qc.rx(2 * beta, range(q.n_qubits))
    qc.measure_all()
    return qc


def fake_backend(name: str):
    from qiskit_ibm_runtime import fake_provider

    cls_name = "Fake" + name.removeprefix("fake_").replace("_", " ").title().replace(" ", "")
    if not hasattr(fake_provider, cls_name):
        raise ValueError(f"unknown fake backend '{name}' (try fake_kingston, fake_fez, fake_torino)")
    return getattr(fake_provider, cls_name)()


def compile_for(qc, backend):
    from qiskit.transpiler import generate_preset_pass_manager

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=7)
    isa = pm.run(qc)
    ops = isa.count_ops()
    two_q = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx", "rzz"))
    return isa, {"two_qubit_gates": int(two_q), "depth": int(isa.depth()),
                 "backend": backend.name}
