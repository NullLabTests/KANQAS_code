from __future__ import annotations

import logging
from typing import Any

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("ibm_runtime")

try:
    from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService, SamplerV2, Session
    from qiskit_ibm_runtime.accounts import AccountManager

    _HAS_IBM_RUNTIME = True
except ImportError:
    _HAS_IBM_RUNTIME = False
    QiskitRuntimeService = None
    EstimatorV2 = None
    SamplerV2 = None
    Session = None


class IBMQuantumBackend:
    def __init__(
        self,
        instance: str | None = None,
        channel: str = "ibm_quantum",
        token: str | None = None,
        max_qubits: int = 127,
        min_qubits: int = 4,
    ):
        if not _HAS_IBM_RUNTIME:
            raise ImportError("qiskit-ibm-runtime is required for IBM Quantum backend operations")

        self.instance = instance
        self.channel = channel
        self.token = token
        self.max_qubits = max_qubits
        self.min_qubits = min_qubits
        self.service: QiskitRuntimeService | None = None
        self.backend: Any = None
        self.backend_name: str = ""

    def authenticate(self) -> QiskitRuntimeService:
        if self.token:
            self.service = QiskitRuntimeService(
                channel=self.channel,
                token=self.token,
                instance=self.instance,
            )
        else:
            try:
                self.service = QiskitRuntimeService(
                    channel=self.channel,
                    instance=self.instance,
                )
            except Exception:
                logger.warning("No IBM Quantum credentials found. Using simulator fallback.")
                self.service = None
        return self.service

    def select_backend(self, num_qubits: int) -> str | None:
        if self.service is None:
            self.authenticate()
        if self.service is None:
            logger.warning("Cannot select backend without IBM Quantum service")
            return None

        try:
            backends = self.service.backends(
                min_num_qubits=max(num_qubits, self.min_qubits),
                max_num_qubits=self.max_qubits,
                operational=True,
                simulator=False,
            )
            if not backends:
                logger.warning("No suitable operational backends found")
                return None
            backend = backends[0]
            self.backend = backend
            self.backend_name = backend.name
            logger.info(f"Selected backend: {backend.name} ({backend.num_qubits} qubits)")
            return backend.name
        except Exception as e:
            logger.error(f"Failed to select backend: {e}")
            return None

    def get_estimator(self) -> EstimatorV2 | None:
        if self.backend is None:
            logger.error("No backend selected. Call select_backend first.")
            return None
        return EstimatorV2(backend=self.backend)

    def get_sampler(self) -> SamplerV2 | None:
        if self.backend is None:
            logger.error("No backend selected. Call select_backend first.")
            return None
        return SamplerV2(backend=self.backend)

    def create_session(self) -> Session | None:
        if self.backend is None:
            logger.error("No backend selected. Call select_backend first.")
            return None
        return Session(backend=self.backend)

    def get_backend_properties(self) -> dict[str, Any]:
        if self.backend is None:
            return {}
        props = self.backend.properties()
        if props is None:
            return {"qubit_count": self.backend.num_qubits}
        return {
            "qubit_count": self.backend.num_qubits,
            "backend_name": self.backend.name,
            "backend_version": self.backend.backend_version,
            "basis_gates": self.backend.configuration().basis_gates,
            "max_shots": self.backend.configuration().max_shots,
            "qubit_fidelities": {
                q: {
                    "T1": props.qubit_property(q, "T1"),
                    "T2": props.qubit_property(q, "T2"),
                    "frequency": props.qubit_property(q, "frequency"),
                    "readout_error": props.qubit_property(q, "readout_error"),
                }
                for q in range(min(self.backend.num_qubits, 10))
            },
            "gate_errors": {
                g: props.gate_error(g, (0, 1))
                for g in props.gates
                if hasattr(props, "gate_error")
            },
        }

    @staticmethod
    def estimate_job_cost(circuit_depth: int, num_qubits: int, shots: int = 4000) -> float:
        base_cost = 0.05
        depth_factor = circuit_depth / 100
        qubit_factor = num_qubits / 10
        shot_factor = shots / 4000
        return base_cost * depth_factor * qubit_factor * shot_factor
