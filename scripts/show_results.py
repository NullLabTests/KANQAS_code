from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

results_dir = Path("results/chemistry")
all_results = []

for subdir in results_dir.iterdir():
    if subdir.is_dir():
        summary_file = subdir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                data = json.load(f)
            all_results.append(data)
        else:
            npy_file = subdir / "summary.npy"
            if npy_file.exists():
                data = np.load(npy_file, allow_pickle=True).item()
                if isinstance(data, dict):
                    all_results.append(data)

if not all_results:
    # Also check h2_vqe_results.npy files
    for npy_file in results_dir.rglob("*_results.npy"):
        try:
            data = np.load(npy_file, allow_pickle=True).item()
            if isinstance(data, dict) and "bond_lengths" in data:
                for i in range(len(data["bond_lengths"])):
                    all_results.append({
                        "bond_length": float(data["bond_lengths"][i]),
                        "found_energy": float(data["found_energies"][i]),
                        "exact_energy": float(data["exact_energies"][i]),
                        "error": abs(float(data["found_energies"][i]) - float(data["exact_energies"][i])),
                        "cnot": int(data["cnot_counts"][i]),
                        "depth": int(data["depths"][i]),
                    })
        except Exception:
            pass

all_results.sort(key=lambda x: (x.get("molecule", "H2"), x.get("bond_length", 0)))

table = Table(title="KANQAS-NISQ Chemistry Results Summary")
table.add_column("Molecule", style="cyan")
table.add_column("R (Å)", style="blue")
table.add_column("RL E (Ha)", style="yellow")
table.add_column("Final E (Ha)", style="green")
table.add_column("Exact (Ha)", style="white")
table.add_column("Error (Ha)", style="magenta")
table.add_column("CX", style="red")
table.add_column("Depth", style="red")
table.add_column("Chem Prec", style="bold")

for r in all_results:
    name = r.get("molecule", "H2")
    bl = r.get("bond_length", 0)
    rl = r.get("rl_energy", r.get("found_energy", 0))
    fe = r.get("found_energy", 0)
    ex = r.get("exact_energy", 0)
    err = r.get("error", abs(fe - ex))
    cx = r.get("cnot", r.get("cx", 0))
    dp = r.get("depth", 0)
    cp = "✅" if err < 0.0016 else "❌"
    table.add_row(name, f"{bl:.2f}", f"{rl:.6f}", f"{fe:.6f}", f"{ex:.6f}", f"{err:.6f}", str(cx), str(dp), cp)

console.print(table)
