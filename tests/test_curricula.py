from __future__ import annotations

import pytest

from curricula import MovingThreshold, SuccesCountThreshold, VanillaCurriculum


class TestMovingThreshold:
    @pytest.fixture
    def config(self):
        return {
            "accept_err": 0.1,
            "shift_threshold_ball": 0.02,
            "shift_threshold_time": 10,
            "success_thresh": 5,
            "succ_radius_shift": 3,
            "succes_switch": 1.0,
        }

    def test_initialization(self, config):
        mt = MovingThreshold(config, target_energy=0.0)
        assert mt.current_threshold == 0.1
        assert mt.lowest_cost == 0.1
        assert mt.success_counter == 0

    def test_get_current_threshold(self, config):
        mt = MovingThreshold(config, target_energy=0.0)
        assert mt.get_current_threshold() == 0.1

    def test_reduce_amortisation(self, config):
        mt = MovingThreshold(config, target_energy=0.0)
        for _ in range(5):
            mt.reduce_amortisation()
        assert mt.current_threshold <= 0.1

    def test_greedy_shift(self, config):
        mt = MovingThreshold(config, target_energy=0.0)
        mt.lowest_cost = 0.05
        for _ in range(15):
            mt.greedy_shift()
        assert mt.current_threshold > 0

    def test_update_threshold(self, config):
        mt = MovingThreshold(config, target_energy=0.0)
        mt.update_threshold(energy_done=True)
        assert mt.call_counter > 0


class TestSuccesCountThreshold:
    @pytest.fixture
    def config(self):
        return {"accept_err": 0.1, "success_thresh": 3}

    def test_initialization(self, config):
        sct = SuccesCountThreshold(config, target_energy=0.0)
        assert sct.current_threshold == 0.1

    def test_greedy_shift(self, config):
        sct = SuccesCountThreshold(config, target_energy=0.0)
        sct.lowest_cost = 0.05
        for _ in range(4):
            sct.greedy_shift()
        assert sct.current_threshold <= 0.1


class TestVanillaCurriculum:
    @pytest.fixture
    def config(self):
        return {
            "accept_err": 0.5,
            "thresholds": [0.5, 0.2, 0.1],
            "switch_episodes": [100, 200, 300],
        }

    def test_initialization(self, config):
        vc = VanillaCurriculum(config, target_energy=0.0)
        assert vc.current_threshold == 0.5

    def test_get_current_threshold_initial(self, config):
        vc = VanillaCurriculum(config, target_energy=0.0)
        assert vc.get_current_threshold() == 0.5

    def test_update_and_threshold(self, config):
        vc = VanillaCurriculum(config, target_energy=0.0)
        for _ in range(150):
            vc.update_threshold()
        assert vc.get_current_threshold() == 0.2
