from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator


def noisy_expectation(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    qubits: list[int] | range,
    backend: Any | None = None,
    shots: int = 4000,
) -> float:
    if backend is not None:
        sim = AerSimulator.from_backend(backend)
        circ = transpile(circuit, backend=backend)
    else:
        sim = AerSimulator(method="automatic")
        circ = circuit.copy()
    circ.save_expectation_value(hamiltonian, qubits)
    result = sim.run(circ, shots=shots).result()
    return float(result.data().get("expectation_value", 0.0))
