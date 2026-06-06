from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from qiskit import QuantumCircuit

from agents.KAQN import KAQN


class KANVisualizer:
    def __init__(self, output_dir: str = "results/interpretability"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_spline_activations(
        self,
        kan_agent: KAQN,
        layer_idx: int = 0,
        num_samples: int = 100,
        save_name: str = "spline_activations.png",
    ) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        kan_module = kan_agent.policy_net[0]
        input_dim = kan_agent.state_size
        for i in range(min(6, input_dim)):
            x = torch.linspace(-1, 1, num_samples).unsqueeze(1)
            x_extended = torch.zeros(num_samples, input_dim)
            x_extended[:, i] = x.squeeze()
            with torch.no_grad():
                output = kan_module(x_extended)
            axes[i].plot(x.numpy(), output[:, 0].numpy(), label=f"Input {i}")
            axes[i].set_title(f"Spline Activation - Input {i}")
            axes[i].set_xlabel("Input value")
            axes[i].set_ylabel("Output")
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()
        return fig

    def plot_gate_preference_heatmap(
        self,
        circuit: QuantumCircuit,
        save_name: str = "gate_preference_heatmap.png",
    ) -> plt.Figure:
        num_qubits = circuit.num_qubits
        gate_matrix = np.zeros((num_qubits, num_qubits))
        for instr in circuit.data:
            qargs = [circuit.find_bit(q).index for q in instr.qubits]
            if len(qargs) == 2:
                gate_matrix[qargs[0], qargs[1]] += 1
                gate_matrix[qargs[1], qargs[0]] += 1
            else:
                gate_matrix[qargs[0], qargs[0]] += 1
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(gate_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(num_qubits))
        ax.set_yticks(range(num_qubits))
        ax.set_xlabel("Target Qubit")
        ax.set_ylabel("Control Qubit")
        ax.set_title("Gate Preference Heatmap")
        for i in range(num_qubits):
            for j in range(num_qubits):
                val = int(gate_matrix[i, j])
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax, label="Gate count")
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()
        return fig

    def plot_activation_trajectories(
        self,
        energy_history: list[float],
        cx_history: list[int],
        depth_history: list[int],
        save_name: str = "activation_trajectories.png",
    ) -> plt.Figure:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        axes[0].plot(energy_history, color="blue", alpha=0.7)
        axes[0].set_title("Energy Trajectory")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Energy (Ha)")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(cx_history, color="red", alpha=0.7)
        axes[1].set_title("CNOT Gate Count Trajectory")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("CNOT Count")
        axes[1].grid(True, alpha=0.3)
        axes[2].plot(depth_history, color="green", alpha=0.7)
        axes[2].set_title("Circuit Depth Trajectory")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Depth")
        axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()
        return fig

    def plot_circuit_evolution(
        self,
        circuits: list[QuantumCircuit],
        episode_indices: list[int],
        save_name: str = "circuit_evolution.png",
    ) -> plt.Figure:
        n = len(circuits)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, circ, ep in zip(axes, circuits, episode_indices):
            gates = circ.count_ops()
            labels = list(gates.keys())
            values = list(gates.values())
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
            ax.bar(labels, values, color=colors)
            ax.set_title(f"Episode {ep}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()
        return fig

    def plot_energy_curve(
        self,
        bond_lengths: list[float],
        found_energies: list[float],
        exact_energies: list[float],
        molecule_name: str = "",
        save_name: str = "energy_curve.png",
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(bond_lengths, exact_energies, "b-", linewidth=2, label="Exact (FCI)")
        ax.plot(
            bond_lengths,
            found_energies,
            "r--o",
            linewidth=2,
            markersize=6,
            label="KANQAS-Discovered",
        )
        ax.fill_between(
            bond_lengths,
            exact_energies,
            found_energies,
            alpha=0.2,
            color="gray",
            label="Error",
        )
        ax.set_xlabel("Bond Length (A)")
        ax.set_ylabel("Energy (Ha)")
        ax.set_title(f"KANQAS-NISQ VQE Energy Curve - {molecule_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=150, bbox_inches="tight")
        plt.close()
        return fig

    def create_summary_dashboard_plots(
        self,
        results: dict[str, Any],
        molecule_name: str = "H2",
    ) -> None:
        if "bond_lengths" in results and "found_energies" in results:
            self.plot_energy_curve(
                results["bond_lengths"],
                results["found_energies"],
                results.get("exact_energies", results.get("exact_energies", [])),
                molecule_name=molecule_name,
            )
        if "energy_history" in results:
            self.plot_activation_trajectories(
                results.get("energy_history", []),
                results.get("cx_history", []),
                results.get("depth_history", []),
            )


if __name__ == "__main__":
    viz = KANVisualizer()
    print(f"KANVisualizer ready. Output dir: {viz.output_dir}")
