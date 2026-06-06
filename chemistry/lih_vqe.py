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
logger = logging.getLogger("lih_vqe")


class LiHVQETrainer:
    def __init__(
        self,
        bond_lengths: list[float] | None = None,
        device: torch.device = torch.device("cpu"),
        output_dir: str = "results/lih_vqe",
        config: dict[str, Any] | None = None,
        pretrained_path: str | None = None,
    ):
        self.bond_lengths = bond_lengths or [1.2, 1.6, 2.0, 2.5, 3.0, 3.5]
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or self._default_config()
        self.pretrained_path = pretrained_path
        self.results: dict[str, list[float]] = {
            "bond_lengths": [],
            "found_energies": [],
            "exact_energies": [],
            "gate_counts": [],
            "depths": [],
            "params": [],
        }

    def _default_config(self) -> dict[str, Any]:
        return {
            "general": {"episodes": 300},
            "env": {
                "num_layers": 8,
                "fn_type": "fidelty_reward",
                "curriculum_type": "MovingThreshold",
                "accept_err": 0.05,
                "shift_threshold_ball": 0.02,
                "shift_threshold_time": 25,
                "success_thresh": 5,
                "succ_radius_shift": 10,
                "succes_switch": 1.0,
                "depth_penalty": 0.01,
                "gate_penalty": 0.001,
            },
            "problem": {
                "type": "lih_vqe",
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
                "memory_size": 200000,
                "batch_size": 128,
                "learning_rate": 0.0005,
                "final_gamma": 0.99,
                "epsilon_min": 0.01,
                "epsilon_decay": 0.995,
                "update_target_net": 15,
                "dropout": 0.1,
                "neurons": [128, 64],
                "kan_seed": 42,
                "k": 3,
                "grid": 5,
                "memory_reset_switch": 15,
                "memory_reset_threshold": 0.01,
            },
        }

    def _train_one_bond(self, bond_length: float) -> dict[str, Any]:
        molecule = MolecularHamiltonian.lih(bond_length=bond_length)
        logger.info(f"LiH at {bond_length:.2f}A uses {molecule.num_qubits} qubits")
        env_conf = copy.deepcopy(self.config)
        env_conf["env"]["num_qubits"] = molecule.num_qubits
        env = VQEEnv(env_conf, molecule, self.device)
        agent = KAQN(self.config, env.action_size, env.state_size, self.device)
        if self.pretrained_path:
            logger.info(f"Loading pretrained weights from {self.pretrained_path}")
            agent.policy_net.load_state_dict(torch.load(self.pretrained_path, map_location=self.device))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
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
                    f"Bond {bond_length:.2f}A | Ep {ep}/{episodes} | "
                    f"Energy: {env.energy:.6f} | Error: {env.error:.6f}"
                )
        circ = env.make_circuit()
        exact_e = molecule.exact_diagonalization()
        return {
            "bond_length": bond_length,
            "found_energy": float(env.energy),
            "exact_energy": exact_e,
            "gate_counts": circ.count_ops(),
            "depth": circ.depth(),
            "circuit": circ,
            "error": float(env.error),
        }

    def run(self) -> dict[str, Any]:
        logger.info(f"Running LiH VQE (qubits=6) with KAQN agent")
        for bond in self.bond_lengths:
            result = self._train_one_bond(bond)
            self.results["bond_lengths"].append(bond)
            self.results["found_energies"].append(result["found_energy"])
            self.results["exact_energies"].append(result["exact_energy"])
            self.results["gate_counts"].append(sum(result["gate_counts"].values()))
            self.results["depths"].append(result["depth"])
            self.results["params"].append(result["gate_counts"].get("cx", 0))
            logger.info(
                f"Result: bond={bond:.2f}A E_found={result['found_energy']:.6f} "
                f"E_exact={result['exact_energy']:.6f} error={result['error']:.6f}"
            )
        np.save(self.output_dir / "lih_vqe_results.npy", self.results)
        logger.info(f"Results saved to {self.output_dir / 'lih_vqe_results.npy'}")
        return self.results


if __name__ == "__main__":
    trainer = LiHVQETrainer()
    trainer.run()
