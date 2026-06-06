from __future__ import annotations

import pytest

from utils import dictionary_of_actions, dict_of_actions_revert_q, get_config


class TestUtils:
    def test_dictionary_of_actions_2q(self):
        actions = dictionary_of_actions(2)
        assert len(actions) == 2 * (2 + 4)
        for k, v in actions.items():
            assert isinstance(k, int)
            assert len(v) == 4

    def test_dictionary_of_actions_4q(self):
        actions = dictionary_of_actions(4)
        assert len(actions) == 4 * (4 + 4)
        for k, v in actions.items():
            assert k in actions
            assert len(v) == 4

    def test_dict_revert_qubits(self):
        forward = dictionary_of_actions(3)
        backward = dict_of_actions_revert_q(3)
        assert len(forward) == len(backward)

    def test_action_structure(self):
        actions = dictionary_of_actions(3)
        for k, v in actions.items():
            if v[0] < 3:
                assert v[1] in range(1, 3)
            else:
                assert v[2] in range(3)
                assert v[3] in range(1, 6)

    def test_get_config(self):
        config = get_config("KAQN/", "2q_bell_state_seed1.cfg")
        assert config is not None
        assert "env" in config
        assert "agent" in config
        assert "general" in config
