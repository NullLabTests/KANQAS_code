from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from chemistry.molecule import MolecularHamiltonian
from hardware.hardware_eval import HardwareEvaluator
from hardware.ibm_runtime import get_backend
from hardware.noise_aware_trainer import NoiseAwareTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("phase_d")
console = Console()


def train_chemical_precision(
    molecule_name: str,
    bond_length: float,
    backend_mode: str,
    backend_name: str | None,
    episodes: int = 1000,
    num_layers: int = 12,
) -> dict[str, Any]:
    logger.info(f"Training {molecule_name} at R={bond_length:.2f}A ({episodes} episodes, {num_layers} layers)")
    molecule = _get_molecule(molecule_name, bond_length)
    exact = molecule.exact_diagonalization()

    backend = get_backend(mode=backend_mode, name=backend_name, num_qubits=molecule.num_qubits)
    trainer = NoiseAwareTrainer(
        molecule,
        real_device_backend=backend,
        backend_mode=backend_mode,
        backend_name=backend_name,
        num_qubits=molecule.num_qubits,
        output_dir=f"results/phase_d/{molecule_name}_{bond_length:.2f}",
    )
    trainer.config["general"]["episodes"] = episodes
    trainer.config["env"]["num_layers"] = num_layers
    trainer.config["env"]["depth_penalty"] = 0.001
    trainer.config["env"]["gate_penalty"] = 0.0003
    trainer.config["env"]["accept_err"] = 3.0
    trainer.config["env"]["shift_threshold_ball"] = 0.5
    trainer.config["agent"]["epsilon_decay"] = 0.9995
    trainer.config["agent"]["epsilon_min"] = 0.02
    trainer.config["agent"]["memory_size"] = 200000
    trainer.config["agent"]["batch_size"] = 128
    result = trainer.train()

    error = abs(result["best_energy"] - exact)
    logger.info(f"  Best E={result['best_energy']:.6f} | Exact={exact:.6f} | Error={error:.6f}")
    return result | {"molecule": molecule_name, "bond_length": bond_length, "exact_energy": exact}


def _get_molecule(name: str, bond_length: float) -> MolecularHamiltonian:
    name = name.lower()
    if name in ("h2", "h₂"):
        return MolecularHamiltonian.h2(bond_length=bond_length)
    elif name in ("lih", "lih"):
        return MolecularHamiltonian.lih(bond_length=bond_length)
    elif name in ("beh2", "beh₂", "beh2"):
        return MolecularHamiltonian.beh2(bond_length=bond_length)
    raise ValueError(f"Unknown molecule: {name}")


def evaluate_on_real_hardware(
    circuits: list[Any],
    labels: list[str],
    molecule: MolecularHamiltonian,
    token: str | None = None,
    instance: str | None = None,
    output_dir: str = "results/hardware",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    backend = get_backend(mode="real", num_qubits=molecule.num_qubits, token=token, instance=instance)
    if backend is None:
        logger.warning("No real backend available. Skipping hardware evaluation.")
        return {"status": "skipped", "reason": "no real backend"}

    results = []
    for circ, label in zip(circuits, labels):
        evaluator = HardwareEvaluator(molecule, backend_mode="real", ibm_token=token, instance=instance)
        evaluator.backend = backend
        logger.info(f"Evaluating {label} on {backend.name}...")
        eval_result = evaluator._hardware_evaluation(circ)
        eval_result["label"] = label
        eval_result["noiseless"] = molecule.estimate_energy(circ)
        eval_result["backend"] = backend.name
        results.append(eval_result)

    table = Table(title=f"Real Hardware Evaluation - {molecule.name} on {backend.name}")
    table.add_column("Label", style="cyan")
    table.add_column("Noiseless (Ha)", style="green")
    table.add_column("Hardware (Ha)", style="red")
    table.add_column("Job ID", style="yellow")
    for r in results:
        table.add_row(r["label"], f"{r.get('noiseless', 0):.6f}", f"{r.get('hardware_energy', 0):.6f}", r.get("job_id", "N/A"))
    console.print(table)

    save = {"backend": backend.name, "results": results}
    np.save(out / "real_hardware_results.npy", save)
    with open(out / "real_hardware_results.json", "w") as f:
        json.dump(save, f, indent=2, default=str)
    logger.info(f"Results saved to {out / 'real_hardware_results.json'}")
    return save


def main():
    parser = argparse.ArgumentParser(description="KANQAS-NISQ Phase D: Chemical Precision + Hardware")
    parser.add_argument("--molecule", default="H2", choices=["H2", "LiH", "BeH2"])
    parser.add_argument("--bond-length", type=float, default=0.74)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--backend", choices=["fake", "real"], default="fake")
    parser.add_argument("--backend-name", default=None, help="Backend name (e.g. Brisbane, Kyoto)")
    parser.add_argument("--ibm-token", default=None)
    parser.add_argument("--ibm-instance", default=None)
    args = parser.parse_args()

    t0 = time.time()
    train_result = train_chemical_precision(
        args.molecule, args.bond_length,
        backend_mode=args.backend,
        backend_name=args.backend_name,
        episodes=args.episodes,
        num_layers=args.layers,
    )

    if args.backend == "real" and train_result.get("best_circuit"):
        molecule = _get_molecule(args.molecule, args.bond_length)
        hw_result = evaluate_on_real_hardware(
            [train_result["best_circuit"]],
            [f"{args.molecule}_best"],
            molecule,
            token=args.ibm_token,
            instance=args.ibm_instance,
        )
        train_result["real_hardware"] = hw_result

    elapsed = time.time() - t0
    console.print(f"\n[bold green]Phase D complete in {elapsed:.0f}s[/bold green]")
    return train_result


if __name__ == "__main__":
    main()
