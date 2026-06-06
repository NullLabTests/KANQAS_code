from __future__ import annotations

from typing import Any

import numpy as np
import torch
from qiskit import QuantumCircuit

from chemistry.molecule import MolecularHamiltonian
from environment import CircuitEnv


class VQEEnv(CircuitEnv):
    def __init__(
        self,
        conf: dict[str, Any],
        molecule: MolecularHamiltonian,
        device: torch.device = torch.device("cpu"),
    ):
        self.molecule = molecule
        self.num_qubits = molecule.num_qubits
        if "env" in conf:
            conf["env"]["num_qubits"] = molecule.num_qubits
        else:
            conf["num_qubits"] = molecule.num_qubits

        super().__init__(conf, device=device)

        self.depth_penalty_weight = conf.get("env", {}).get("depth_penalty", 0.01)
        self.gate_penalty_weight = conf.get("env", {}).get("gate_penalty", 0.001)
        self.gs_energy = molecule.reference_energy
        self.prev_energy = 0.0

    def get_cost_func(self, x: torch.Tensor | None = None) -> float:
        circ = self.make_circuit()
        energy = self.molecule.estimate_energy(circ)
        return float(energy)

    def reward_fn(self, energy: float) -> float:
        depth_penalty = self.depth_penalty_weight * self.step_counter
        num_cnots = self.make_circuit().count_ops().get("cx", 0)
        gate_penalty = self.gate_penalty_weight * num_cnots
        energy_error = abs(energy - self.gs_energy)
        if energy_error <= self.done_threshold:
            return 50.0 * (1.0 - depth_penalty - gate_penalty)
        return -energy_error - depth_penalty - gate_penalty

    def make_circuit(self, state: torch.Tensor | None = None) -> QuantumCircuit:
        if state is None:
            state = self.state.clone()
        circ = QuantumCircuit(self.num_qubits)
        for i in range(self.num_layers):
            cnot_pos = np.where(state[i][0 : self.num_qubits] == 1)
            targ, ctrl = cnot_pos[0], cnot_pos[1]
            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    circ.cx(int(ctrl[r]), int(targ[r]))
            rot_pos = np.where(state[i][self.num_qubits : self.num_qubits + 5] == 1)
            rot_direction_list, rot_qubit_list = rot_pos[0], rot_pos[1]
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    qubit = int(rot_qubit_list[pos])
                    if r == 0:
                        circ.x(qubit)
                    elif r == 1:
                        circ.y(qubit)
                    elif r == 2:
                        circ.z(qubit)
                    elif r == 3:
                        circ.h(qubit)
                    elif r == 4:
                        circ.t(qubit)
        return circ

    def reset(self) -> torch.Tensor:
        state = torch.zeros((self.num_layers, self.num_qubits + 5, self.num_qubits))
        self.state = state
        self.reset_env_variables()
        return state[:, : self.num_qubits + 5].reshape(-1).to(self.device)

    def step(self, action: list[int], train_flag: bool = True) -> tuple[torch.Tensor, torch.Tensor, int]:
        next_state = self.state.clone()
        self.step_counter += 1
        ctrl = action[0]
        targ = (action[0] + action[1]) % self.num_qubits
        which_qubit = action[2]
        which_oneq_gate = action[3]
        self.action = action
        if which_qubit < self.num_qubits:
            gate_tensor = self.moments[which_qubit]
        elif ctrl < self.num_qubits:
            gate_tensor = max(self.moments[ctrl], self.moments[targ])
        if ctrl < self.num_qubits:
            next_state[gate_tensor][targ][ctrl] = 1
        elif which_qubit < self.num_qubits:
            next_state[gate_tensor][self.num_qubits + which_oneq_gate - 1][which_qubit] = 1
        if which_qubit < self.num_qubits:
            self.moments[which_qubit] += 1
        elif ctrl < self.num_qubits:
            max_of_two = max(self.moments[ctrl], self.moments[targ])
            self.moments[ctrl] = max_of_two + 1
            self.moments[targ] = max_of_two + 1
        self.current_action = action
        self.update_illegal_actions()
        self.state = next_state.clone()
        self.prev_cost = self.error
        energy = float(self.get_cost_func())
        self.energy = energy
        self.prev_energy = energy
        self.error = float(abs(energy - self.gs_energy))
        rwd = self.reward_fn(energy)
        self.save_circ = self.make_circuit()
        energy_done = int(self.error >= self.done_threshold)
        layers_done = self.step_counter == (self.num_layers - 1)
        done = int(energy_done or layers_done)
        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
        next_state = next_state[:, : self.num_qubits + 5]
        return next_state.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done
