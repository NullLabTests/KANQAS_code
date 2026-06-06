from __future__ import annotations

import copy
import random
from collections import namedtuple
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from utils import dictionary_of_actions, dict_of_actions_revert_q


class DDQN:
    def __init__(
        self,
        conf: dict[str, Any],
        action_size: int,
        state_size: int,
        device: torch.device,
    ):
        env_cfg = conf.get("env", conf)
        agent_cfg = conf.get("agent", conf)
        self.num_qubits = env_cfg.get("num_qubits", conf.get("num_qubits", 2))
        self.num_layers = env_cfg.get("num_layers", conf.get("num_layers", 6))
        memory_size = agent_cfg.get("memory_size", conf.get("memory_size", 10000))
        self.final_gamma = agent_cfg.get("final_gamma", conf.get("final_gamma", 0.99))
        self.epsilon_min = agent_cfg.get("epsilon_min", conf.get("epsilon_min", 0.01))
        self.epsilon_decay = agent_cfg.get("epsilon_decay", conf.get("epsilon_decay", 0.995))
        learning_rate = agent_cfg.get("learning_rate", conf.get("learning_rate", 0.001))
        self.update_target_net = agent_cfg.get("update_target_net", conf.get("update_target_net", 10))
        neuron_list = agent_cfg.get("neurons", conf.get("neurons", [64, 32]))
        drop_prob = agent_cfg.get("dropout", conf.get("dropout", 0.1))
        self.with_angles = agent_cfg.get("angles", conf.get("angles", False))

        self.memory_reset_switch = agent_cfg.get("memory_reset_switch", conf.get("memory_reset_switch", False))
        self.memory_reset_threshold = agent_cfg.get("memory_reset_threshold", conf.get("memory_reset_threshold", 0.0))
        self.memory_reset_counter = 0

        self.action_size = action_size
        self.state_size = state_size
        self.state_size = self.state_size + 1 if agent_cfg.get("en_state", conf.get("en_state", False)) else self.state_size
        self.state_size = (
            self.state_size + 1
            if agent_cfg.get("threshold_in_state", conf.get("threshold_in_state", False))
            else self.state_size
        )

        self.translate = dictionary_of_actions(self.num_qubits)
        self.rev_translate = dict_of_actions_revert_q(self.num_qubits)
        self.policy_net = self._build_mlp(neuron_list, drop_prob).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()

        self.gamma = torch.Tensor(
            [np.round(np.power(self.final_gamma, 1 / self.num_layers), 2)]
        ).to(device)
        self.memory = ReplayMemory(memory_size)
        self.epsilon = 1.0
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.device = device
        self.step_counter = 0
        self.Transition = namedtuple(
            "Transition", ("state", "action", "reward", "next_state", "done")
        )

    def _build_mlp(self, neuron_list: list[int], p: float) -> nn.Sequential:
        layer_list: list[nn.Module] = []
        dims = [self.state_size] + neuron_list
        for i in range(len(dims) - 1):
            layer_list.append(nn.Linear(dims[i], dims[i + 1]))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(p=p))
        layer_list.append(nn.Linear(dims[-1], self.action_size))
        return nn.Sequential(*layer_list)

    def remember(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state: torch.Tensor, ill_action: list[int]) -> tuple[int, bool]:
        state = state.unsqueeze(0)
        if torch.rand(1).item() <= self.epsilon:
            rand_ac = torch.randint(self.action_size, (1,)).item()
            while rand_ac in ill_action:
                rand_ac = torch.randint(self.action_size, (1,)).item()
            return rand_ac, True
        act_values = self.policy_net.forward(state)
        act_values[0][ill_action] = float("-inf")
        return torch.argmax(act_values[0]).item(), False

    def replay(self, batch_size: int) -> float:
        if self.step_counter % self.update_target_net == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.step_counter += 1
        transitions = self.memory.sample(batch_size)
        batch = self.Transition(*zip(*transitions))
        next_state_batch = torch.stack(batch.next_state)
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward)
        done_batch = torch.stack(batch.done)

        state_action_values = self.policy_net.forward(state_batch).gather(
            1, action_batch.unsqueeze(1)
        )

        next_state_values = self.target_net.forward(next_state_batch)
        next_state_actions = (
            self.policy_net.forward(next_state_batch).max(1)[1].detach()
        )
        next_state_values = next_state_values.gather(
            1, next_state_actions.unsqueeze(1)
        ).squeeze(1)

        expected = (
            next_state_values * self.gamma
        ) * (1 - done_batch) + reward_batch
        expected = expected.view(-1, 1)

        assert state_action_values.shape == expected.shape, "Shape mismatch"
        cost = self._fit(state_action_values, expected)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
        return cost

    def _fit(self, output: torch.Tensor, target: torch.Tensor) -> float:
        self.optim.zero_grad()
        loss = self.loss_fn(output, target)
        loss.backward()
        self.optim.step()
        return loss.item()


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: list[Any] = []
        self.position = 0
        self.Transition = namedtuple(
            "Transition", ("state", "action", "reward", "next_state", "done")
        )

    def push(self, *args: torch.Tensor):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Any]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

    def clean_memory(self):
        self.memory = []
        self.position = 0
