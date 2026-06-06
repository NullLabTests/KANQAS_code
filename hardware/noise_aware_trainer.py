from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from rich.logging import RichHandler

from agents.KAQN import KAQN
from chemistry.molecule import MolecularHamiltonian
from chemistry.vqe_env import VQEEnv
from utils import dictionary_of_actions

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("noise_aware_trainer")

try:
    import mitiq

    _HAS_MITIQ = True
except ImportError:
    _HAS_MITIQ = False


class NoiseAwareTrainer:
    def __init__(
        self,
        molecule: MolecularHamiltonian,
        real_device_backend: Any | None = None,
        num_qubits: int = 4,
        device: torch.device = torch.device("cpu"),
        output_dir: str = "results/noise_aware",
        max_cx_gates: int = 100,
    ):
        self.molecule = molecule
        self.real_device_backend = real_device_backend
        self.num_qubits = num_qubits
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_cx_gates = max_cx_gates
        self.config = self._default_config()
        self.noisy_sim: AerSimulator | None = None
        self.energy_history: list[float] = []
        self.cx_history: list[int] = []
        self.depth_history: list[int] = []

    def _default_config(self) -> dict[str, Any]:
        return {
            "general": {"episodes": 200},
            "env": {
                "num_layers": 6,
                "fn_type": "fidelty_reward",
                "curriculum_type": "MovingThreshold",
                "accept_err": 0.05,
                "shift_threshold_ball": 0.02,
                "shift_threshold_time": 20,
                "success_thresh": 5,
                "succ_radius_shift": 10,
                "succes_switch": 1.0,
                "depth_penalty": 0.02,
                "gate_penalty": 0.005,
            },
            "problem": {
                "type": "noise_aware_vqe",
                "noise": 1,
                "noise_prob_1q": 0.001,
                "noise_prob_2q": 0.01,
            },
            "agent": {
                "agent_type": "KAQN",
                "agent_class": "KAQN",
                "angles": False,
                "en_state": True,
                "threshold_in_state": True,
                "memory_size": 100000,
                "batch_size": 64,
                "learning_rate": 0.001,
                "final_gamma": 0.99,
                "epsilon_min": 0.01,
                "epsilon_decay": 0.995,
                "update_target_net": 10,
                "dropout": 0.1,
                "neurons": [64, 32],
                "kan_seed": 42,
                "k": 3,
                "grid": 5,
                "memory_reset_switch": 10,
                "memory_reset_threshold": 0.01,
            },
        }

    def _apply_zne(self, circuit: QuantumCircuit, shots: int = 4000) -> float:
        if not _HAS_MITIQ:
            logger.warning("Mitiq not installed; returning noisy expectation value")
            if self.noisy_sim:
                circ_copy = circuit.copy()
                circ_copy.save_expectation_value(self.molecule.hamiltonian, range(self.num_qubits))
                result = self.noisy_sim.run(circ_copy, shots=shots).result()
                return float(result.data().get("expectation_value", 0.0))
            return 0.0
        noisy_value = self._noisy_expectation(circuit, shots)
        return noisy_value

    def _noisy_expectation(self, circuit: QuantumCircuit, shots: int = 4000) -> float:
        if self.noisy_sim is None:
            if self.real_device_backend:
                self.noisy_sim = AerSimulator.from_backend(self.real_device_backend)
            else:
                self.noisy_sim = AerSimulator(method="automatic")
        circ_copy = circuit.copy()
        circ_copy.save_expectation_value(self.molecule.hamiltonian, range(self.num_qubits))
        result = self.noisy_sim.run(circ_copy, shots=shots).result()
        return float(result.data().get("expectation_value", 0.0))

    def _prune_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        ops = circuit.count_ops()
        if ops.get("cx", 0) > self.max_cx_gates:
            logger.warning(f"Circuit has {ops['cx']} CX gates, exceeding limit of {self.max_cx_gates}")
        return circuit

    def train(self) -> dict[str, Any]:
        logger.info(f"Starting noise-aware training for {self.molecule.name}")
        conf = copy.deepcopy(self.config)
        conf["env"]["num_qubits"] = self.num_qubits
        env = VQEEnv(conf, self.molecule, self.device)
        agent = KAQN(self.config, env.action_size, env.state_size, self.device)
        translate = dictionary_of_actions(self.num_qubits)
        episodes = conf["general"]["episodes"]
        for ep in range(episodes):
            state = env.reset()
            for step in range(env.num_layers + 1):
                ill_actions = env.update_illegal_actions()
                action_idx, _ = agent.act(state, ill_actions)
                action = translate[action_idx]
                next_state, reward, done = env.step(action)
                agent.remember(
                    state,
                    torch.tensor(action_idx, device=self.device),
                    reward,
                    next_state,
                    torch.tensor(done, device=self.device),
                )
                state = next_state
                if len(agent.memory) > conf["agent"]["batch_size"]:
                    agent.replay(conf["agent"]["batch_size"])
                if done:
                    break
            circ = env.make_circuit()
            circ = self._prune_circuit(circ)
            noise_energy = self._noisy_expectation(circ)
            self.energy_history.append(noise_energy)
            self.cx_history.append(circ.count_ops().get("cx", 0))
            self.depth_history.append(circ.depth())
            if ep % 10 == 0:
                logger.info(
                    f"Ep {ep}/{episodes} | Noisy E={noise_energy:.6f} | "
                    f"CX={circ.count_ops().get('cx', 0)} | Depth={circ.depth()}"
                )
        best_idx = int(np.argmin(self.energy_history))
        best_circuit = env.make_circuit()
        best_circuit = self._prune_circuit(best_circuit)
        result = {
            "energy_history": self.energy_history,
            "cx_history": self.cx_history,
            "depth_history": self.depth_history,
            "best_energy": float(np.min(self.energy_history)),
            "best_circuit": best_circuit,
            "best_cx": best_circuit.count_ops().get("cx", 0),
            "best_depth": best_circuit.depth(),
            "exact_energy": self.molecule.exact_diagonalization(),
            "final_circuit": env.make_circuit(),
        }
        np.save(self.output_dir / "noise_aware_results.npy", result)
        logger.info(f"Noise-aware training complete. Best energy: {result['best_energy']:.6f}")
        return result

    def apply_zne_mitigation(self, circuit: QuantumCircuit) -> dict[str, Any]:
        if not _HAS_MITIQ:
            logger.warning("Mitiq not available; skipping ZNE")
            return {"mitigated_energy": 0.0, "unmitigated_energy": 0.0}

        unmitigated = self._noisy_expectation(circuit)
        zne_circuit = copy.deepcopy(circuit)
        from mitiq import zne

        mitigated = zne.execute_with_zne(
            zne_circuit,
            lambda c: self._noisy_expectation(c),
            scale_noise=mitiq.zne.scaling.fold_global,
        )
        logger.info(f"ZNE: unmitigated={unmitigated:.6f} mitigated={mitigated:.6f}")
        return {"mitigated_energy": mitigated, "unmitigated_energy": unmitigated}


if __name__ == "__main__":
    mol = MolecularHamiltonian.h2()
    trainer = NoiseAwareTrainer(mol)
    trainer.train()
