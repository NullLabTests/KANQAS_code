from __future__ import annotations

import configparser
import json
from itertools import product
from typing import Any
from warnings import simplefilter

simplefilter(action="ignore", category=DeprecationWarning)


def get_config(
    config_name: str,
    experiment_name: str,
    path: str = "configuration_files",
    verbose: bool = True,
) -> dict[str, Any]:
    config_dict: dict[str, Any] = {}
    config = configparser.ConfigParser()
    config.read(f"{path}/{config_name}{experiment_name}")
    for section in config.sections():
        config_dict[section] = {}
        for key, val in config.items(section):
            try:
                config_dict[section].update({key: int(val)})
            except ValueError:
                config_dict[section].update({key: val})
            floats = [
                "learning_rate",
                "dropout",
                "alpha",
                "beta",
                "beta_incr",
                "shift_threshold_ball",
                "succes_switch",
                "tolearance_to_thresh",
                "memory_reset_threshold",
                "fake_min_energy",
                "_true_en",
            ]
            strings = [
                "ham_type",
                "fn_type",
                "geometry",
                "method",
                "agent_type",
                "agent_class",
                "init_seed",
                "init_path",
                "init_thresh",
                "method",
                "mapping",
                "optim_alg",
                "curriculum_type",
            ]
            lists = [
                "episodes",
                "neurons",
                "accept_err",
                "epsilon_decay",
                "epsilon_min",
                "epsilon_decay",
                "final_gamma",
                "memory_clean",
                "update_target_net",
                "epsilon_restart",
                "thresholds",
                "switch_episodes",
            ]
            if key in floats:
                config_dict[section].update({key: float(val)})
            elif key in strings:
                config_dict[section].update({key: str(val)})
            elif key in lists:
                config_dict[section].update({key: json.loads(val)})
    if "DEFAULT" in config_dict:
        del config_dict["DEFAULT"]
    return config_dict


def dictionary_of_actions(num_qubits: int) -> dict[int, list[int]]:
    dictionary: dict[int, list[int]] = {}
    i = 0
    for c, x in product(range(num_qubits), range(1, num_qubits)):
        dictionary[i] = [c, x, num_qubits, 0]
        i += 1
    for r, h in product(range(num_qubits), range(1, 6)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary


def dict_of_actions_revert_q(num_qubits: int) -> dict[int, list[int]]:
    dictionary: dict[int, list[int]] = {}
    i = 0
    for c, x in product(
        range(num_qubits - 1, -1, -1), range(num_qubits - 1, 0, -1)
    ):
        dictionary[i] = [c, x, num_qubits, 0]
        i += 1
    for r, h in product(range(num_qubits - 1, -1, -1), range(1, 6)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary


def load_yaml_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for loading YAML configs")
    with open(path) as f:
        return yaml.safe_load(f)
