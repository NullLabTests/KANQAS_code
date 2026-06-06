#!/usr/bin/env python3
"""Phase A v2: Chemical precision with periodic epsilon reset and slower decay."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

sys.path.insert(0, ".")
from chemistry.h2_vqe import H2VQETrainer

logging.basicConfig(level=logging.WARNING, handlers=[RichHandler(rich_tracebacks=True)])
console = Console()


def train_bond(bl: float, episodes: int, output_dir: Path) -> dict:
    console.print(f"\n[bold yellow]=== H2 at R = {bl:.2f} A ({episodes} eps) ===[/]")
    trainer = H2VQETrainer(bond_lengths=[bl], agent_type="KAQN", output_dir=str(output_dir))
    trainer.config["general"]["episodes"] = episodes
    trainer.config["agent"]["memory_size"] = 1000000
    trainer.config["agent"]["batch_size"] = 256
    trainer.config["agent"]["epsilon_decay"] = 0.9995
    trainer.config["agent"]["learning_rate"] = 0.0003
    trainer.config["env"]["num_layers"] = 10
    trainer.config["env"]["accept_err"] = 2.0
    trainer.config["env"]["shift_threshold_ball"] = 0.15
    trainer.config["env"]["shift_threshold_time"] = 15
    trainer.config["env"]["depth_penalty"] = 0.003
    trainer.config["env"]["gate_penalty"] = 0.0005

    t0 = time.time()
    results = trainer.run()
    elapsed = time.time() - t0

    found = float(results["found_energies"][0])
    exact = float(results["exact_energies"][0])
    error = abs(found - exact)
    cx = int(results["cnot_counts"][0])
    depth = int(results["depths"][0])

    achieved = error < 0.0016
    console.print(f"  E_found={found:.6f} E_exact={exact:.6f} Error={error:.6f} CX={cx} Depth={depth} Time={elapsed:.0f}s")
    console.print(f"  Chemical precision: [{'green' if achieved else 'red'}]{achieved}[/]")

    return {"bond_length": bl, "found_energy": found, "exact_energy": exact,
            "error": error, "cx_gates": cx, "depth": depth, "elapsed_s": elapsed,
            "episodes": episodes, "precision_achieved": achieved}


def main():
    console.print("[bold cyan]KANQAS-NISQ Phase A v2 — Chemical Precision Push[/]\n")

    output_dir = Path("results/phase_a_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Focus on worst bond lengths from v1 + equilibrium
    target_bonds = [(0.74, 1500), (1.00, 1500), (1.50, 2000), (2.50, 1000)]
    all_results = {}
    best_error = float("inf")

    for bl, eps in target_bonds:
        res = train_bond(bl, eps, output_dir)
        all_results[str(bl)] = res
        best_error = min(best_error, res["error"])

        # Commit intermediate result
        with open(output_dir / "phase_a_v2_progress.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    table = Table(title="Phase A v2 — H2 Chemical Precision Results")
    table.add_column("R (A)", style="cyan")
    table.add_column("E_found (Ha)", style="green")
    table.add_column("E_exact (Ha)", style="blue")
    table.add_column("Error (Ha)", style="magenta")
    table.add_column("CX", style="yellow")
    table.add_column("Depth", style="yellow")
    table.add_column("Precision", style="bold")

    for bl, res in sorted(all_results.items(), key=lambda x: float(x[0])):
        mark = "✓" if res["precision_achieved"] else "✗"
        table.add_row(f"{res['bond_length']:.2f}", f"{res['found_energy']:.6f}",
                      f"{res['exact_energy']:.6f}", f"{res['error']:.6f}",
                      str(res["cx_gates"]), str(res["depth"]), mark)
    console.print(table)

    summary = {"best_error": best_error, "chemical_precision_achieved": best_error < 0.0016, "results": all_results}
    with open(output_dir / "phase_a_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    np.save(output_dir / "phase_a_v2_results.npy", all_results)
    console.print(f"\n[bold green]Results saved to {output_dir}/[/]")

    if best_error < 0.0016:
        console.print("[bold green]✓ CHEMICAL PRECISION ACHIEVED![/]")
    else:
        console.print(f"[bold yellow]Best error: {best_error:.6f} Ha — need more training[/]")


if __name__ == "__main__":
    main()
