from __future__ import annotations

import pytest
import torch

from environment import CircuitEnv


@pytest.fixture
def bell_config():
    return {
        "env": {
            "num_qubits": 2,
            "num_layers": 6,
            "fn_type": "fidelty_reward",
            "curriculum_type": "MovingThreshold",
            "accept_err": 0.05,
            "shift_threshold_ball": 0.02,
            "shift_threshold_time": 20,
            "success_thresh": 5,
            "succ_radius_shift": 10,
            "succes_switch": 1.0,
        },
        "problem": {
            "type": "bell",
            "noise": 0,
            "noise_prob_1q": 0.001,
            "noise_prob_2q": 0.01,
        },
        "agent": {"angles": False, "en_state": False},
    }


@pytest.fixture
def device():
    return torch.device("cpu")


class TestCircuitEnv:
    @pytest.mark.smoke
    def test_initialization(self, bell_config, device):
        env = CircuitEnv(bell_config, device)
        assert env.num_qubits == 2
        assert env.num_layers == 6
        assert env.action_size == 12
        assert env.state_size == 6 * 2 * (2 + 5)

    def test_reset(self, bell_config, device):
        env = CircuitEnv(bell_config, device)
        state = env.reset()
        assert isinstance(state, torch.Tensor)
        assert state.shape[0] == env.state_size

    def test_step(self, bell_config, device):
        env = CircuitEnv(bell_config, device)
        env.reset()
        action = [0, 1, 2, 3]
        next_state, reward, done = env.step(action)
        assert isinstance(next_state, torch.Tensor)
        assert isinstance(reward, torch.Tensor)
        assert done in (0, 1)

    def test_make_circuit(self, bell_config, device):
        env = CircuitEnv(bell_config, device)
        env.reset()
        circ = env.make_circuit()
        assert circ.num_qubits == 2
        assert circ.name is not None

    def test_get_cost_func(self, bell_config, device):
        env = CircuitEnv(bell_config, device)
        env.reset()
        cost = env.get_cost_func()
        assert isinstance(cost, float)

    def test_reward_fn(self, bell_config, device):
        env = CircuitEnv(bell_config, device)
        reward = env.reward_fn(0.5)
        assert isinstance(reward, float)

    @pytest.mark.smoke
    def test_full_episode(self, bell_config, device):
        env = CircuitEnv(bell_config, device)
        state = env.reset()
        for step in range(env.num_layers):
            action = [0, 1, 2, 3]
            next_state, reward, done = env.step(action)
            state = next_state
            if done:
                break
        assert env.step_counter >= 0
