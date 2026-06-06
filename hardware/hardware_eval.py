from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

from chemistry.molecule import MolecularHamiltonian
from hardware.ibm_runtime import IBMQuantumBackend
from hardware.noise_aware_trainer import NoiseAwareTrainer
from hardware.noisy_utils import noisy_expectation

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("hardware_eval")

console = Console()

try:
    from qiskit_ibm_runtime import EstimatorV2
except ImportError:
    EstimatorV2 = None


class HardwareEvaluator:
    def __init__(
        self,
        molecule: MolecularHamiltonian,
        ibm_token: str | None = None,
        instance: str | None = None,
        output_dir: str = "results/hardware_eval",
    ):
        self.molecule = molecule
        self.ibm_token = ibm_token
        self.instance = instance
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ibm_backend = IBMQuantumBackend(token=ibm_token, instance=instance)
        self.backend_name: str = ""

    def _simulator_evaluation(self, circuit: QuantumCircuit) -> dict[str, Any]:
        noiseless_energy = self.molecule.estimate_energy(circuit, backend=None)
        noisy_energy = noisy_expectation(
            circuit, self.molecule.hamiltonian, range(self.molecule.num_qubits),
        )
        return {
            "noiseless_energy": float(noiseless_energy),
            "noisy_sim_energy": float(noisy_energy),
            "exact_energy": self.molecule.exact_diagonalization(),
        }

    def _hardware_evaluation(self, circuit: QuantumCircuit, shots: int = 4000) -> dict[str, Any]:
        if EstimatorV2 is None:
            logger.error("qiskit-ibm-runtime not available")
            return {"hardware_energy": 0.0, "error": "missing dependency"}

        if self.ibm_backend.backend is None:
            self.ibm_backend.token = self.ibm_token
            self.ibm_backend.authenticate()
            self.ibm_backend.select_backend(self.molecule.num_qubits)
        if self.ibm_backend.backend is None:
            logger.error("Cannot connect to IBM Quantum")
            return {"hardware_energy": 0.0, "error": "no backend"}

        try:
            estimator = self.ibm_backend.get_estimator()
            pub = (circuit, [self.molecule.hamiltonian])
            job = estimator.run([pub], shots=shots)
            result = job.result()
            hardware_energy = float(result[0].data.evs[0])
            return {
                "hardware_energy": hardware_energy,
                "job_id": job.job_id(),
                "backend": self.ibm_backend.backend_name,
            }
        except Exception as e:
            logger.error(f"Hardware evaluation failed: {e}")
            return {"hardware_energy": 0.0, "error": str(e)}

    def evaluate(
        self,
        circuit: QuantumCircuit,
        run_on_hardware: bool = False,
    ) -> dict[str, Any]:
        logger.info("Evaluating circuit on simulator...")
        sim_results = self._simulator_evaluation(circuit)
        results = {**sim_results}

        results["gate_counts"] = dict(circuit.count_ops())
        results["depth"] = circuit.depth()
        results["num_qubits"] = circuit.num_qubits
        results["num_gates"] = sum(circuit.count_ops().values())

        if run_on_hardware:
            logger.info("Evaluating circuit on real IBM Quantum hardware...")
            hw_results = self._hardware_evaluation(circuit)
            results.update(hw_results)
            if "hardware_energy" in hw_results:
                results["sim_vs_hw_diff"] = abs(
                    results.get("noisy_sim_energy", 0) - hw_results.get("hardware_energy", 0)
                )

        self._print_results_table(results)
        np.save(self.output_dir / "hardware_eval_results.npy", results)
        return results

    def evaluate_curriculum(
        self,
        trainer: NoiseAwareTrainer,
        run_on_hardware: bool = False,
    ) -> dict[str, Any]:
        logger.info("Running noise-aware training then evaluating best circuit...")
        train_results = trainer.train()
        best_circuit = train_results["best_circuit"]
        eval_results = self.evaluate(best_circuit, run_on_hardware=run_on_hardware)
        eval_results["training_best_energy"] = train_results["best_energy"]
        eval_results["training_best_cx"] = train_results["best_cx"]
        eval_results["training_best_depth"] = train_results["best_depth"]
        np.save(self.output_dir / "hardware_curriculum_results.npy", eval_results)
        return eval_results

    def _print_results_table(self, results: dict[str, Any]) -> None:
        table = Table(title=f"Hardware Evaluation - {self.molecule.name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Molecule", self.molecule.name)
        table.add_row("Qubits", str(results.get("num_qubits", "N/A")))
        table.add_row("Depth", str(results.get("depth", "N/A")))
        table.add_row("Gates", str(results.get("num_gates", "N/A")))
        table.add_row("Gate Counts", str(results.get("gate_counts", {})))
        table.add_row("Exact GS Energy", f"{results.get('exact_energy', 0):.8f}")
        table.add_row("Noiseless Energy", f"{results.get('noiseless_energy', 0):.8f}")
        table.add_row("Noisy Sim Energy", f"{results.get('noisy_sim_energy', 0):.8f}")
        if "hardware_energy" in results:
            table.add_row("Hardware Energy", f"{results['hardware_energy']:.8f}")
        if "sim_vs_hw_diff" in results:
            table.add_row("Sim vs HW Diff", f"{results['sim_vs_hw_diff']:.8f}")
        if "training_best_energy" in results:
            table.add_row("Training Best E", f"{results['training_best_energy']:.6f}")
        console.print(table)

    def compare_simulator_hardware(self, circuits: list[QuantumCircuit], labels: list[str]) -> None:
        table = Table(title="Simulator vs Hardware Comparison")
        table.add_column("Label", style="cyan")
        table.add_column("Exact E", style="green")
        table.add_column("Noiseless", style="blue")
        table.add_column("Noisy Sim", style="yellow")
        table.add_column("Hardware", style="red")
        table.add_column("Diff", style="magenta")
        for circ, label in zip(circuits, labels):
            exact = self.molecule.exact_diagonalization()
            noiseless = self.molecule.estimate_energy(circ, backend=None)
            sim_res = self._simulator_evaluation(circ)
            hw_res = self._hardware_evaluation(circ)
            table.add_row(
                label,
                f"{exact:.6f}",
                f"{noiseless:.6f}",
                f"{sim_res['noisy_sim_energy']:.6f}",
                f"{hw_res.get('hardware_energy', 0):.6f}",
                f"{abs(sim_res['noisy_sim_energy'] - hw_res.get('hardware_energy', 0)):.6f}",
            )
        console.print(table)


if __name__ == "__main__":
    mol = MolecularHamiltonian.h2(bond_length=0.74)
    from qiskit import QuantumCircuit

    test_circ = QuantumCircuit(2)
    test_circ.h(0)
    test_circ.cx(0, 1)
    evaluator = HardwareEvaluator(mol)
    evaluator.evaluate(test_circ, run_on_hardware=False)
