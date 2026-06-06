from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from qiskit import QuantumCircuit
from rich.logging import RichHandler
from scipy.optimize import minimize

from agents.DDQN import DDQN
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
        self.results: dict[str, Any] = {
            "bond_lengths": [],
            "found_energies": [],
            "exact_energies": [],
            "rl_energies": [],
            "gate_counts": [],
            "depths": [],
            "cnot_counts": [],
            "best_circuits": [],
            "energy_histories": [],
            "error_histories": [],
        }

    def _default_config(self) -> dict[str, Any]:
        return {
            "general": {"episodes": 500},
            "env": {
                "num_layers": 8,
                "fn_type": "fidelty_reward",
                "curriculum_type": "MovingThreshold",
                "accept_err": 1.5,
                "shift_threshold_ball": 0.1,
                "shift_threshold_time": 10,
                "success_thresh": 3,
                "succ_radius_shift": 5,
                "succes_switch": 3.0,
                "depth_penalty": 0.005,
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
                "memory_size": 200000,
                "batch_size": 128,
                "learning_rate": 0.0005,
                "final_gamma": 0.99,
                "epsilon_min": 0.01,
                "epsilon_decay": 0.9995,
                "update_target_net": 20,
                "dropout": 0.1,
                "neurons": [128, 64],
                "kan_seed": 42,
                "k": 3,
                "grid": 5,
                "memory_reset_switch": 25,
                "memory_reset_threshold": 0.05,
            },
        }

    def _make_agent(self, env: VQEEnv) -> DDQN | KAQN:
        if self.agent_type == "KAQN":
            return KAQN(self.config, env.action_size, env.state_size, self.device)
        return DDQN(self.config, env.action_size, env.state_size, self.device)

    def _convert_to_continuous(self, circuit: QuantumCircuit) -> tuple[QuantumCircuit, list[Any]]:
        from qiskit.circuit import Parameter
        params: list[Parameter] = []
        new_circ = QuantumCircuit(circuit.num_qubits, name=f"{circuit.name}_continuous")
        for instruction, qargs, cargs in circuit.data:
            q = qargs[0]
            if instruction.name == 'x':
                p = Parameter(f'θ{len(params)}')
                params.append(p)
                new_circ.rx(np.pi + p, q)
            elif instruction.name == 'y':
                p = Parameter(f'θ{len(params)}')
                params.append(p)
                new_circ.ry(np.pi + p, q)
            elif instruction.name == 'z':
                p = Parameter(f'θ{len(params)}')
                params.append(p)
                new_circ.rz(np.pi + p, q)
            elif instruction.name == 'h':
                new_circ.h(q)
            elif instruction.name == 't':
                new_circ.t(q)
            elif instruction.name == 'cx':
                new_circ.cx(q, qargs[1])
            else:
                new_circ.append(instruction, qargs, cargs)
        nq = circuit.num_qubits
        hea_reps = min(2, max(1, nq // 2))
        for _ in range(hea_reps):
            for q in range(nq):
                p = Parameter(f'θ{len(params)}')
                params.append(p)
                new_circ.ry(p, q)
            for q in range(nq - 1):
                new_circ.cx(q, q + 1)
            for q in range(nq):
                p = Parameter(f'θ{len(params)}')
                params.append(p)
                new_circ.rz(p, q)
        return new_circ, params

    def _post_optimize(
        self, circuit: QuantumCircuit, molecule: MolecularHamiltonian,
        max_iter: int = 1000,
    ) -> tuple[float, QuantumCircuit]:
        try:
            circ, params = self._convert_to_continuous(circuit)
        except Exception:
            logger.warning("Could not convert circuit to continuous, skipping post-optimization")
            return float('inf'), circuit
        if not params:
            return float('inf'), circuit

        x0 = np.zeros(len(params))
        bounds = [(-np.pi, np.pi)] * len(params)
        best_x = x0.copy()
        best_e = float('inf')

        def objective(x: np.ndarray) -> float:
            nonlocal best_x, best_e
            bound_circ = circ.assign_parameters({p: v for p, v in zip(params, x)})
            e = molecule.estimate_energy(bound_circ)
            if e < best_e:
                best_e = e
                best_x = x.copy()
            return e

        logger.info(f"  Post-optimizing {len(params)} continuous parameters with COBYLA (max {max_iter} iters)...")
        from scipy.optimize import minimize
        minimize(objective, x0, method='COBYLA', bounds=bounds,
                 options={'maxiter': max_iter, 'rhobeg': 0.5, 'catol': 1e-6})

        bound_circ = circ.assign_parameters({p: v for p, v in zip(params, best_x)})
        logger.info(f"  Post-optimized energy: {best_e:.6f} (from {len(params)} params)")
        return best_e, bound_circ

    def _train_one_bond(self, bond_length: float) -> dict[str, Any]:
        molecule = MolecularHamiltonian.h2(bond_length=bond_length)
        exact_e = molecule.exact_diagonalization()

        env_conf = copy.deepcopy(self.config)
        env_conf["env"]["num_qubits"] = molecule.num_qubits
        env = VQEEnv(env_conf, molecule, self.device)
        agent = self._make_agent(env)
        translate = dictionary_of_actions(molecule.num_qubits)

        episodes = self.config["general"]["episodes"]
        best_energy = float("inf")
        best_circuit = None
        energy_history: list[float] = []
        error_history: list[float] = []
        best_episode = 0

        epsilon_reset_interval = 300
        for ep in range(episodes):
            if ep > 0 and ep % epsilon_reset_interval == 0:
                agent.epsilon = min(agent.epsilon + 0.15, 0.3)
                logger.info(f"  Epsilon reset to {agent.epsilon:.3f} at episode {ep}")
            state = env.reset()
            episode_energy = float("inf")
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
                episode_energy = min(episode_energy, float(env.energy))
                batch_size = self.config["agent"]["batch_size"]
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                if done:
                    break

            energy_history.append(episode_energy)
            error = abs(episode_energy - exact_e)
            error_history.append(error)

            if episode_energy < best_energy:
                best_energy = episode_energy
                best_circuit = env.make_circuit()
                best_episode = ep

            if ep % 50 == 0:
                logger.info(
                    f"Bond {bond_length:.2f}A | Ep {ep}/{episodes} | "
                    f"E={episode_energy:.6f} | |E-Egs|={error:.6f} | "
                    f"Best={best_energy:.6f} | Eps={agent.epsilon:.3f}"
                )

            if error < 0.0016:
                logger.info(f"Chemical precision reached at episode {ep}!")
                break

        circ = best_circuit or env.make_circuit()
        best_rl_energy = float(best_energy)

        do_optimize = self.config.get("general", {}).get("post_optimize", False)
        if do_optimize:
            max_iter = self.config.get("general", {}).get("post_optimize_iter", 1000)
            opt_energy, opt_circ = self._post_optimize(circ, molecule, max_iter)
            if opt_energy < best_energy:
                best_energy = opt_energy
                circ = opt_circ

        gate_counts = circ.count_ops()
        logger.info(
            f"Bond {bond_length:.2f}A complete | RL E={best_rl_energy:.6f} | "
            f"Post-opt E={best_energy:.6f} | "
            f"Exact GS={exact_e:.6f} | Error={abs(best_energy - exact_e):.6f} | "
            f"CX={gate_counts.get('cx', 0)} | Depth={circ.depth()}"
        )
        return {
            "bond_length": bond_length,
            "found_energy": float(best_energy),
            "exact_energy": exact_e,
            "gate_counts": gate_counts,
            "depth": circ.depth(),
            "circuit": circ,
            "error": float(abs(best_energy - exact_e)),
            "energy_history": energy_history,
            "error_history": error_history,
            "best_episode": best_episode,
            "rl_energy": best_rl_energy,
        }

    def run(self) -> dict[str, Any]:
        logger.info(f"Running H2 VQE with {self.agent_type} agent over {len(self.bond_lengths)} bond lengths")
        key_map = {
            "bond_lengths": "bond_length",
            "found_energies": "found_energy",
            "exact_energies": "exact_energy",
            "rl_energies": "rl_energy",
            "gate_counts": None,
            "depths": None,
            "cnot_counts": None,
            "best_circuits": None,
            "energy_histories": None,
            "error_histories": None,
        }
        for bond in self.bond_lengths:
            result = self._train_one_bond(bond)
            for key, singular in key_map.items():
                if key == "gate_counts":
                    val = sum(result.get("gate_counts", {}).values())
                elif key == "cnot_counts":
                    val = result.get("gate_counts", {}).get("cx", 0)
                elif key == "depths":
                    val = result.get("depth", 0)
                elif key == "best_circuits":
                    val = result.get("circuit")
                elif key == "energy_histories":
                    val = result.get("energy_history", [])
                elif key == "error_histories":
                    val = result.get("error_history", [])
                else:
                    val = result.get(singular, 0)
                self.results[key].append(val)
        np.save(self.output_dir / "h2_vqe_results.npy", self.results, allow_pickle=True)
        logger.info(f"Results saved to {self.output_dir / 'h2_vqe_results.npy'}")
        return self.results


if __name__ == "__main__":
    trainer = H2VQETrainer()
    trainer.run()
