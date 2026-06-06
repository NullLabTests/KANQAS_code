from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler

from agents.DDQN import DDQN
from agents.KAQN import KAQN
from chemistry.molecule import MolecularHamiltonian
from chemistry.vqe_env import VQEEnv
from curricula import MovingThreshold
from utils import dictionary_of_actions

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("h2_vqe")


class H2VQETrainer:
    def __init__(
        self,
        bond_lengths: list[float] | None = None,
        agent_type: str = "KAQN",
        device: torch.device = torch.device("cpu"),
        output_dir: str = "results/h2_vqe",
        config: dict[str, Any] | None = None,
    ):
        self.bond_lengths = bond_lengths or [0.5, 0.74, 1.0, 1.5, 2.0, 2.5]
        self.agent_type = agent_type
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or self._default_config()
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
                "type": "h2_vqe",
                "noise": 0,
                "noise_prob_1q": 0.001,
                "noise_prob_2q": 0.01,
            },
            "agent": {
                "agent_type": "KAQN",
                "agent_class": "KAQN",
                "angles": False,
                "en_state": False,
                "threshold_in_state": False,
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

    def _make_agent(self, env: VQEEnv) -> DDQN | KAQN:
        if self.agent_type == "KAQN":
            return KAQN(self.config, env.action_size, env.state_size, self.device)
        return DDQN(self.config, env.action_size, env.state_size, self.device)

    def _train_one_bond(self, bond_length: float) -> dict[str, Any]:
        molecule = MolecularHamiltonian.h2(bond_length=bond_length)
        env_conf = copy.deepcopy(self.config)
        env_conf["env"]["num_qubits"] = molecule.num_qubits
        env_conf["env"]["curriculum_type"] = "MovingThreshold"
        env_conf["env"]["accept_err"] = 0.05
        env = VQEEnv(env_conf, molecule, self.device)
        agent = self._make_agent(env)
        translate = dictionary_of_actions(molecule.num_qubits)

        episodes = self.config["general"]["episodes"]
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
                batch_size = self.config["agent"]["batch_size"]
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                if done:
                    break
            if ep % 20 == 0:
                logger.info(f"Bond {bond_length:.2f}A | Ep {ep}/{episodes} | Energy: {env.energy:.6f} | Error: {env.error:.6f}")

        circ = env.make_circuit()
        gate_counts = circ.count_ops()
        exact_e = molecule.exact_diagonalization()
        return {
            "bond_length": bond_length,
            "found_energy": float(env.energy),
            "exact_energy": exact_e,
            "gate_counts": gate_counts,
            "depth": circ.depth(),
            "circuit": circ,
            "error": float(env.error),
        }

    def run(self) -> dict[str, Any]:
        logger.info(f"Running H2 VQE with {self.agent_type} agent")
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
        np.save(self.output_dir / "h2_vqe_results.npy", self.results)
        logger.info(f"Results saved to {self.output_dir / 'h2_vqe_results.npy'}")
        return self.results


if __name__ == "__main__":
    trainer = H2VQETrainer()
    trainer.run()
