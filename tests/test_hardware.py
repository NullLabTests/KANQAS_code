from __future__ import annotations

import pytest

from chemistry.molecule import MolecularHamiltonian


class TestHardwareModule:
    def test_imports(self):
        try:
            from hardware.ibm_runtime import IBMQuantumBackend
            from hardware.noise_aware_trainer import NoiseAwareTrainer
            from hardware.hardware_eval import HardwareEvaluator

            assert IBMQuantumBackend is not None
            assert NoiseAwareTrainer is not None
            assert HardwareEvaluator is not None
        except ImportError as e:
            if "IBM" in str(e):
                pytest.skip("IBM runtime not installed")
            raise

    @pytest.mark.smoke
    def test_noise_aware_trainer_creation(self):
        mol = MolecularHamiltonian.h2(bond_length=0.74)
        if mol is not None:
            from hardware.noise_aware_trainer import NoiseAwareTrainer

            trainer = NoiseAwareTrainer(molecule=mol, num_qubits=4)
            assert trainer is not None
            assert trainer.molecule.name == "H2"
            assert trainer.num_qubits == 4

    def test_ibm_backend_creation(self):
        try:
            from hardware.ibm_runtime import IBMQuantumBackend

            backend = IBMQuantumBackend()
            assert backend is not None
            assert backend.max_qubits == 127
            assert backend.min_qubits == 4
        except ImportError:
            pytest.skip("IBM runtime not installed")

    def test_backend_properties_no_service(self):
        try:
            from hardware.ibm_runtime import IBMQuantumBackend

            backend = IBMQuantumBackend()
            props = backend.get_backend_properties()
            assert isinstance(props, dict)
        except ImportError:
            pytest.skip("IBM runtime not installed")

    def test_cost_estimation(self):
        try:
            from hardware.ibm_runtime import IBMQuantumBackend

            cost = IBMQuantumBackend.estimate_job_cost(circuit_depth=50, num_qubits=4, shots=4000)
            assert isinstance(cost, float)
            assert cost > 0
        except ImportError:
            pytest.skip("IBM runtime not installed")
