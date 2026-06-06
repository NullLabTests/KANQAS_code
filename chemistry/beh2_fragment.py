from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
logger = logging.getLogger("beh2_vqe")


class BeH2FragmentVQETrainer:
    def __init__(
        self,
        bond_length: float = 1.3,
        device: torch.device = torch.device("cpu"),
        output_dir: str = "results/beh2_vqe",
        config: dict[str, Any] | None = None,
    ):
        self.bond_length = bond_length
        self.num_fragments = 2
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or self._default_config()
        self.results: dict[str, Any] = {
            "fragment_energies": [],
            "total_energy": None,
            "exact_energy": None,
            "fragment_circuits": [],
            "fragment_gate_counts": [],
        }

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
                "depth_penalty": 0.01,
                "gate_penalty": 0.001,
            },
            "problem": {
                "type": "beh2_fragment",
                "noise": 0,
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

    def _train_fragment(self, fragment_name: str, molecule: MolecularHamiltonian) -> dict[str, Any]:
        logger.info(f"Training fragment: {fragment_name} ({molecule.num_qubits} qubits)")
        env_conf = copy.deepcopy(self.config)
        env_conf["env"]["num_qubits"] = molecule.num_qubits
        env = VQEEnv(env_conf, molecule, self.device)
        agent = KAQN(self.config, env.action_size, env.state_size, self.device)
        translate = dictionary_of_actions(molecule.num_qubits)
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
            if ep % 20 == 0:
                logger.info(
                    f"[{fragment_name}] Ep {ep}/{episodes} | "
                    f"Energy: {env.energy:.6f} | Error: {env.error:.6f}"
                )
        circ = env.make_circuit()
        return {
            "energy": float(env.energy),
            "circuit": circ,
            "gate_counts": circ.count_ops(),
            "depth": circ.depth(),
        }

    def run(self) -> dict[str, Any]:
        logger.info(f"Running BeH2 fragment-based VQE")
        full_molecule = MolecularHamiltonian.beh2(bond_length=self.bond_length, fragment=True)
        logger.info(f"Full BeH2 uses {full_molecule.num_qubits} qubits")
        self.results["exact_energy"] = full_molecule.exact_diagonalization()

        be_frag = MolecularHamiltonian.beh2(bond_length=self.bond_length, fragment=True)
        be_frag.name = "Be-fragment"
        result_be = self._train_fragment("Be-sub", be_frag)
        self.results["fragment_energies"].append(result_be["energy"])
        self.results["fragment_circuits"].append(result_be["circuit"])
        self.results["fragment_gate_counts"].append(result_be["gate_counts"])
        total_energy = result_be["energy"]
        self.results["total_energy"] = total_energy
        logger.info(f"BeH2 fragment-based result: E_found={total_energy:.6f} E_exact={self.results['exact_energy']:.6f}")
        np.save(self.output_dir / "beh2_vqe_results.npy", self.results)
        logger.info(f"Results saved to {self.output_dir / 'beh2_vqe_results.npy'}")
        return self.results


if __name__ == "__main__":
    trainer = BeH2FragmentVQETrainer()
    trainer.run()
