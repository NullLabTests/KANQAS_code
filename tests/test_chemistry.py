from __future__ import annotations

import pytest
import torch

from chemistry.molecule import MolecularHamiltonian


class TestMolecularHamiltonian:
    @pytest.mark.smoke
    def test_h2_creation(self):
        mol = MolecularHamiltonian.h2(bond_length=0.74)
        assert mol.name == "H2"
        assert mol.num_qubits == 4
        assert mol.hamiltonian is not None
        assert isinstance(mol.reference_energy, float)

    @pytest.mark.slow
    def test_h2_exact_diagonalization(self):
        mol = MolecularHamiltonian.h2(bond_length=0.74)
        exact = mol.exact_diagonalization()
        assert isinstance(exact, float)
        assert exact < 0

    def test_h2_at_different_bond(self):
        mol = MolecularHamiltonian.h2(bond_length=0.5)
        assert mol.num_qubits == 4
        assert isinstance(mol.reference_energy, float)

    def test_lih_creation(self):
        mol = MolecularHamiltonian.lih(bond_length=2.0)
        assert mol.name == "LiH"
        assert mol.num_qubits > 4
        assert mol.hamiltonian is not None

    def test_beh2_creation(self):
        mol = MolecularHamiltonian.beh2(bond_length=1.3)
        assert mol.name == "BeH2"
        assert mol.hamiltonian is not None

    def test_beh2_fragment_creation(self):
        mol = MolecularHamiltonian.beh2(bond_length=1.3, fragment=True)
        assert mol.name == "BeH2"
        assert mol.hamiltonian is not None

    @pytest.mark.slow
    def test_estimate_energy(self):
        from qiskit import QuantumCircuit

        mol = MolecularHamiltonian.h2(bond_length=0.74)
        circ = QuantumCircuit(4)
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(0, 2)
        circ.cx(0, 3)
        energy = mol.estimate_energy(circ)
        assert isinstance(energy, float)
