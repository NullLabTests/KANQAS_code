from __future__ import annotations

import copy
from sys import stdout
from typing import Any

import numpy as np
import qiskit.quantum_info as qi
import torch
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, mixed_unitary_error

import curricula
from utils import dictionary_of_actions


class CircuitEnv:
    def __init__(self, conf: dict[str, Any], device: torch.device):
        self.num_qubits = conf["env"]["num_qubits"]
        self.num_layers = conf["env"]["num_layers"]
        self.fn_type = conf["env"]["fn_type"]
        self.state_type = conf["problem"]["type"]
        self.noise = int(conf["problem"]["noise"])

        self.cnot_rwd_weight = float(conf["env"].get("cnot_rwd_weight", 1.0))
        p_single = float(conf["problem"].get("noise_prob_1q", 0.001))
        p_double = float(conf["problem"].get("noise_prob_2q", 0.01))

        if self.noise:
            X = qi.Operator.from_label("X")
            I = qi.Operator.from_label("I")
            dep_error = depolarizing_error(p_double, num_qubits=2)
            prob_X = p_single
            prob_I = 1 - p_single
            S_noise = mixed_unitary_error([(X, prob_X), (I, prob_I)])
            self.noise_m = NoiseModel()
            self.noise_m.add_all_qubit_quantum_error(
                S_noise, ["h", "t", "x", "y", "z"], list(range(self.num_qubits))
            )
            self.noise_m.add_all_qubit_quantum_error(dep_error, "cx")

        self.state_with_angles = conf["agent"].get("angles", False)
        self.current_number_of_cnots = 0
        self.curriculum_dict: dict[str, Any] = {}
        self.device = device
        self.done_threshold = conf["env"]["accept_err"]
        self.curriculum_dict[self.state_type] = curricula.__dict__[
            conf["env"]["curriculum_type"]
        ](conf["env"], target_energy=0)

        stdout.flush()
        self.state_size = self.num_layers * self.num_qubits * (self.num_qubits + 5)
        self.step_counter = -1
        self.error = 0.4
        self.moments = [0] * self.num_qubits
        self.illegal_actions: list[list[int]] = [[]] * self.num_qubits
        self.energy = 0.0
        self.prev_energy = 0.0
        self.action_size = self.num_qubits * (self.num_qubits + 4)
        self.previous_action = [0, 0, 0, 0]
        if self.num_qubits == 2:
            self.init_state = DensityMatrix(
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                )
            )
        elif self.num_qubits == 3:
            self.init_state = DensityMatrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                    ]
                )
            )

    def step(
        self, action: list[int], train_flag: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
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
        cost_func = self.get_cost_func()
        energy = cost_func
        self.energy = cost_func
        self.error = float(abs(energy))
        rwd = self.reward_fn(energy)
        self.prev_cost = np.copy(energy) if isinstance(energy, np.ndarray) else float(energy)
        self.save_circ = self.make_circuit()
        energy_done = int(self.error >= self.done_threshold)
        layers_done = self.step_counter == (self.num_layers - 1)
        done = int(energy_done or layers_done)
        self.previous_action = copy.deepcopy(action)
        if energy < self.curriculum.lowest_cost and train_flag:
            self.curriculum.lowest_cost = copy.copy(energy)
        if done:
            self.curriculum.update_threshold(energy_done=energy_done)
            self.done_threshold = self.curriculum.get_current_threshold()
            self.curriculum_dict[str(self.current_prob)] = copy.deepcopy(self.curriculum)
        if self.state_with_angles:
            return next_state.view(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done
        next_state = next_state[:, : self.num_qubits + 5]
        return next_state.reshape(-1).to(self.device), torch.tensor(rwd, dtype=torch.float32, device=self.device), done

    def reset(self) -> torch.Tensor:
        state = torch.zeros((self.num_layers, self.num_qubits + 5, self.num_qubits))
        self.state = state
        self.reset_env_variables()
        if self.state_with_angles:
            return state.reshape(-1).to(self.device)
        state = state[:, : self.num_qubits + 5]
        return state.reshape(-1).to(self.device)

    def reset_env_variables(self):
        self.current_prob = self.state_type
        self.curriculum = copy.deepcopy(self.curriculum_dict[str(self.current_prob)])
        self.done_threshold = copy.deepcopy(self.curriculum.get_current_threshold())
        self.current_number_of_cnots = 0
        self.current_action = [self.num_qubits] * 4
        self.illegal_actions = [[]] * self.num_qubits
        self.step_counter = -1
        self.moments = [0] * self.num_qubits
        self.error = 0.4
        self.prev_cost = self.get_cost_func(self.state) if self.state is not None else 0.0

    def make_circuit(self, x: torch.Tensor | None = None) -> QuantumCircuit:
        state = x.clone() if x is not None else self.state.clone()
        circuit = QuantumCircuit(self.num_qubits)
        for i in range(self.num_layers):
            cnot_pos = np.where(state[i][0 : self.num_qubits] == 1)
            targ = cnot_pos[0]
            ctrl = cnot_pos[1]
            if len(ctrl) != 0:
                for r in range(len(ctrl)):
                    circuit.cx(int(ctrl[r]), int(targ[r]))
            rot_pos = np.where(state[i][self.num_qubits : self.num_qubits + 5] == 1)
            rot_direction_list, rot_qubit_list = rot_pos[0], rot_pos[1]
            if len(rot_qubit_list) != 0:
                for pos, r in enumerate(rot_direction_list):
                    qubit = int(rot_qubit_list[pos])
                    if r == 0:
                        circuit.x(qubit)
                    elif r == 1:
                        circuit.y(qubit)
                    elif r == 2:
                        circuit.z(qubit)
                    elif r == 3:
                        circuit.h(qubit)
                    elif r == 4:
                        circuit.t(qubit)
        return circuit

    def get_cost_func(self, x: torch.Tensor | None = None) -> float:
        circ = self.make_circuit()
        if self.num_qubits == 2:
            bell_state = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1])
            target = DensityMatrix(np.outer(bell_state, bell_state.conj()))
        elif self.num_qubits == 3:
            ghz_state = (1 / np.sqrt(2)) * np.array([1, 0, 0, 0, 0, 0, 0, 1])
            target = DensityMatrix(np.outer(ghz_state, ghz_state.conj()))
        if self.noise:
            circ.save_density_matrix()
            sim_density = AerSimulator(noise_model=self.noise_m)
            job = sim_density.run(circ)
            result = job.result().data()
            dm_evo = result["density_matrix"]
        else:
            dm_evo = self.init_state.evolve(circ)
        fid = state_fidelity(DensityMatrix(dm_evo), DensityMatrix(target))
        return float(fid)

    def reward_fn(self, x: float | None = None) -> float:
        if self.fn_type == "fidelty_reward":
            scalar = 50
            if self.error >= self.done_threshold:
                return scalar * self.error
            return self.error
        return 0.0

    def update_illegal_actions(self) -> list[int]:
        action = self.current_action
        illegal_action = self.illegal_actions
        ctrl, targ = action[0], (action[0] + action[1]) % self.num_qubits
        rot_qubit, rot_axis = action[2], action[3]
        if ctrl < self.num_qubits:
            are_you_empty = sum(sum(l) for l in illegal_action)
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    if len(ill_ac) != 0:
                        ill_ac_targ = (ill_ac[0] + ill_ac[1]) % self.num_qubits
                        if ill_ac[2] == self.num_qubits:
                            if ctrl == ill_ac[0] or ctrl == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            elif targ == ill_ac[0] or targ == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if ctrl == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            elif targ == ill_ac[2]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
            else:
                illegal_action[0] = action
        if rot_qubit < self.num_qubits:
            are_you_empty = sum(sum(l) for l in illegal_action)
            if are_you_empty != 0:
                for ill_ac_no, ill_ac in enumerate(illegal_action):
                    if len(ill_ac) != 0:
                        ill_ac_targ = (ill_ac[0] + ill_ac[1]) % self.num_qubits
                        if ill_ac[0] == self.num_qubits:
                            if rot_qubit == ill_ac[2] and rot_axis != ill_ac[3]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            elif rot_qubit != ill_ac[2]:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                        else:
                            if rot_qubit == ill_ac[0]:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            elif rot_qubit == ill_ac_targ:
                                illegal_action[ill_ac_no] = []
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
                            else:
                                for i in range(1, self.num_qubits):
                                    if len(illegal_action[i]) == 0:
                                        illegal_action[i] = action
                                        break
            else:
                illegal_action[0] = action
        for indx in range(self.num_qubits):
            for jndx in range(indx + 1, self.num_qubits):
                if illegal_action[indx] == illegal_action[jndx]:
                    if jndx != indx + 1:
                        illegal_action[indx] = []
                    else:
                        illegal_action[jndx] = []
                    break
        for indx in range(self.num_qubits - 1):
            if len(illegal_action[indx]) == 0:
                illegal_action[indx] = illegal_action[indx + 1]
                illegal_action[indx + 1] = []
        illegal_action_decode = []
        for key, contain in dictionary_of_actions(self.num_qubits).items():
            for ill_action in illegal_action:
                if ill_action == contain:
                    illegal_action_decode.append(key)
        self.illegal_actions = illegal_action
        return illegal_action_decode
