from __future__ import annotations

import copy

import pytest
import torch

from agents.DDQN import DDQN, ReplayMemory
from agents.KAQN import KAQN


@pytest.fixture
def basic_config():
    return {
        "env": {"num_qubits": 2},
        "agent": {
            "num_qubits": 2,
            "num_layers": 6,
            "memory_size": 10000,
            "final_gamma": 0.99,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001,
            "update_target_net": 10,
            "neurons": [64, 32],
            "dropout": 0.1,
            "angles": False,
            "en_state": False,
            "kan_seed": 42,
            "k": 3,
            "grid": 5,
        },
    }


@pytest.fixture
def device():
    return torch.device("cpu")


class TestDDQN:
    def test_initialization(self, basic_config, device):
        agent = DDQN(basic_config["agent"], action_size=12, state_size=84, device=device)
        assert agent.action_size == 12
        assert agent.state_size == 84
        assert agent.epsilon == 1.0
        assert agent.memory is not None

    def test_act(self, basic_config, device):
        agent = DDQN(basic_config["agent"], action_size=12, state_size=84, device=device)
        agent.epsilon = 0.0
        state = torch.randn(84)
        action, eps = agent.act(state, [])
        assert isinstance(action, int)
        assert 0 <= action < 12

    def test_act_with_illegal(self, basic_config, device):
        agent = DDQN(basic_config["agent"], action_size=12, state_size=84, device=device)
        agent.epsilon = 0.0
        state = torch.randn(84)
        action, eps = agent.act(state, [0, 1, 2])
        assert action not in [0, 1, 2]

    def test_remember(self, basic_config, device):
        agent = DDQN(basic_config["agent"], action_size=12, state_size=84, device=device)
        agent.remember(
            torch.randn(84),
            torch.tensor(0),
            torch.tensor(1.0),
            torch.randn(84),
            torch.tensor(0),
        )
        assert len(agent.memory) == 1

    def test_replay(self, basic_config, device):
        agent = DDQN(basic_config["agent"], action_size=12, state_size=84, device=device)
        for _ in range(128):
            agent.remember(
                torch.randn(84),
                torch.tensor(0),
                torch.tensor(1.0),
                torch.randn(84),
                torch.tensor(0),
            )
        loss = agent.replay(64)
        assert isinstance(loss, float)


class TestKAQN:
    def test_initialization(self, basic_config, device):
        agent = KAQN(basic_config["agent"], action_size=12, state_size=84, device=device)
        assert agent.action_size == 12
        assert agent.state_size == 84
        assert agent.epsilon == 1.0

    @pytest.mark.smoke
    def test_act(self, basic_config, device):
        agent = KAQN(basic_config["agent"], action_size=12, state_size=84, device=device)
        agent.epsilon = 0.0
        state = torch.randn(84)
        action, eps = agent.act(state, [])
        assert isinstance(action, int)
        assert 0 <= action < 12

    def test_replay(self, basic_config, device):
        agent = KAQN(basic_config["agent"], action_size=12, state_size=84, device=device)
        for _ in range(128):
            agent.remember(
                torch.randn(84),
                torch.tensor(0),
                torch.tensor(1.0),
                torch.randn(84),
                torch.tensor(0),
            )
        loss = agent.replay(64)
        assert isinstance(loss, float)


class TestReplayMemory:
    def test_push_and_sample(self):
        memory = ReplayMemory(capacity=100)
        for i in range(50):
            memory.push(torch.randn(10), torch.tensor(i), torch.tensor(1.0), torch.randn(10), torch.tensor(0))
        assert len(memory) == 50
        sample = memory.sample(10)
        assert len(sample) == 10

    def test_capacity(self):
        memory = ReplayMemory(capacity=10)
        for i in range(20):
            memory.push(torch.randn(10), torch.tensor(i), torch.tensor(1.0), torch.randn(10), torch.tensor(0))
        assert len(memory) == 10

    def test_clean_memory(self):
        memory = ReplayMemory(capacity=100)
        for i in range(20):
            memory.push(torch.randn(10), torch.tensor(i), torch.tensor(1.0), torch.randn(10), torch.tensor(0))
        assert len(memory) == 20
        memory.clean_memory()
        assert len(memory) == 0
