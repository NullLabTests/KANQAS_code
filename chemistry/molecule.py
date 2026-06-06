from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qiskit.quantum_info import SparsePauliOp

try:
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
    from qiskit_nature.second_q.operators import SparseLabelOp
    from qiskit_nature.second_q.properties import ParticleNumber

    _HAS_QISKIT_NATURE = True
except ImportError:
    _HAS_QISKIT_NATURE = False


@dataclass
class MolecularHamiltonian:
    name: str
    num_qubits: int
    hamiltonian: SparsePauliOp
    reference_energy: float
    bond_lengths: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)

    @classmethod
    def from_pyscf(
        cls,
        name: str,
        geometry: str,
        mapping: str = "jordan_wigner",
        charge: int = 0,
        multiplicity: int = 1,
        basis: str = "sto-3g",
        bond_lengths: list[float] | None = None,
    ) -> MolecularHamiltonian:
        if not _HAS_QISKIT_NATURE:
            raise ImportError("qiskit-nature is required for molecular Hamiltonian generation")

        mapper = JordanWignerMapper() if mapping == "jordan_wigner" else ParityMapper()

        if bond_lengths is not None:
            energies = []
            for bond in bond_lengths:
                geo = geometry.format(bond_length=bond)
                driver = PySCFDriver(atom=geo, charge=charge, spin=multiplicity - 1, basis=basis)
                problem = driver.run()
                second_q_op = problem.hamiltonian.second_q_op()
                qubit_op = mapper.map(second_q_op)
                energy = problem.reference_energy
                energies.append(energy)
            ham = cls(
                name=name,
                num_qubits=qubit_op.num_qubits,
                hamiltonian=qubit_op,
                reference_energy=energies[0],
                bond_lengths=bond_lengths,
                energies=energies,
            )
        else:
            driver = PySCFDriver(atom=geometry, charge=charge, spin=multiplicity - 1, basis=basis)
            problem = driver.run()
            second_q_op = problem.hamiltonian.second_q_op()
            qubit_op = mapper.map(second_q_op)
            ham = cls(
                name=name,
                num_qubits=qubit_op.num_qubits,
                hamiltonian=qubit_op,
                reference_energy=problem.reference_energy,
            )
        return ham

    @classmethod
    def h2(
        cls,
        bond_length: float = 0.74,
        mapping: str = "jordan_wigner",
        basis: str = "sto-3g",
    ) -> MolecularHamiltonian:
        geometry = f"H 0 0 0; H 0 0 {bond_length}"
        return cls.from_pyscf("H2", geometry, mapping=mapping, basis=basis)

    @classmethod
    def lih(
        cls,
        bond_length: float = 2.0,
        mapping: str = "jordan_wigner",
        basis: str = "sto-3g",
    ) -> MolecularHamiltonian:
        geometry = f"Li 0 0 0; H 0 0 {bond_length}"
        return cls.from_pyscf("LiH", geometry, mapping=mapping, basis=basis, charge=0, multiplicity=1)

    @classmethod
    def beh2(
        cls,
        bond_length: float = 1.3,
        mapping: str = "jordan_wigner",
        basis: str = "sto-3g",
        fragment: bool = False,
    ) -> MolecularHamiltonian:
        if fragment:
            be_geo = f"Be 0 0 0"
            h1_geo = f"H 0 1.3 0"
            h2_geo = f"H 0 -1.3 0"
            geometry = f"{be_geo}; {h1_geo}; {h2_geo}"
        else:
            geometry = f"Be 0 0 0; H 0 0 {bond_length}; H 0 0 -{bond_length}"
        return cls.from_pyscf(
            "BeH2",
            geometry,
            mapping=mapping,
            basis=basis,
            charge=0,
            multiplicity=1,
        )

    def to_torch(self) -> torch.Tensor:
        return torch.tensor(self.hamiltonian.to_matrix(), dtype=torch.complex128)

    def to_sparse(self) -> SparsePauliOp:
        return self.hamiltonian

    def estimate_energy(self, circuit: Any, backend: Any | None = None, shots: int = 10000) -> float:
        from qiskit.quantum_info import SparsePauliOp, Statevector

        if isinstance(self.hamiltonian, SparsePauliOp):
            hamiltonian = self.hamiltonian
        else:
            hamiltonian = SparsePauliOp(self.hamiltonian)

        if backend is None:
            sv = Statevector(circuit)
            energy = sv.expectation_value(hamiltonian).real
        else:
            try:
                from qiskit_ibm_runtime import EstimatorV2

                estimator = EstimatorV2(backend=backend)
                pub = (circuit, [hamiltonian])
                result = estimator.run([pub]).result()
                energy = result[0].data.evs[0]
            except ImportError:
                sv = Statevector(circuit)
                energy = sv.expectation_value(hamiltonian).real
        return energy

    def exact_diagonalization(self) -> float:
        from qiskit.quantum_info import SparsePauliOp

        if isinstance(self.hamiltonian, SparsePauliOp):
            matrix = self.hamiltonian.to_matrix(sparse=False)
        else:
            matrix = SparsePauliOp(self.hamiltonian).to_matrix(sparse=False)
        eigenvalues = np.linalg.eigvalsh(matrix)
        return float(eigenvalues[0])
