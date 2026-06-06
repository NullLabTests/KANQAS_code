from __future__ import annotations

import argparse
import copy
import logging
import pathlib
import random
import sys
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler

import agents
from environment import CircuitEnv
from utils import get_config

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("kanqas")

torch.set_num_threads(1)


def get_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KANQAS-NISQ: Hardware-aware Curriculum RL Quantum Architecture Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --experiment bell --config 2q_bell_state_seed1 --agent KAQN
  python main.py --experiment h2 --bond-lengths 0.5 0.74 1.0 --episodes 100
  python main.py --experiment lih --bond-lengths 1.5 2.0 2.5 --episodes 200
  python main.py --experiment hardware --backend ibm_brisbane --mode eval
  python main.py --experiment dashboard
        """,
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--config", type=str, default="2q_bell_state_seed1", help="Config file name (without .cfg)")
    parser.add_argument("--experiment", type=str, default="bell", choices=["bell", "ghz", "h2", "lih", "beh2", "hardware", "dashboard"], help="Experiment type")
    parser.add_argument("--experiment_name", type=str, default="KANQAS/", help="Experiment save subfolder")
    parser.add_argument("--agent", type=str, default="KAQN", choices=["KAQN", "DDQN"], help="Agent type")
    parser.add_argument("--backend", type=str, default=None, help="IBM Quantum backend name")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "hardware", "dashboard"], help="Execution mode")
    parser.add_argument("--episodes", type=int, default=None, help="Number of training episodes")
    parser.add_argument("--bond-lengths", type=float, nargs="+", default=None, help="Bond lengths for VQE")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--noisy", action="store_true", help="Enable noise simulation")
    parser.add_argument("--ibm-token", type=str, default=None, help="IBM Quantum API token")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args(argv)


class Saver:
    def __init__(self, results_path: str, experiment_seed: int):
        self.stats_file: dict[str, Any] = {"train": {}, "test": {}}
        self.exp_seed = experiment_seed
        self.rpath = results_path

    def get_new_episode(self, mode: str, episode_no: int):
        if mode == "train":
            self.stats_file[mode][episode_no] = {
                "loss": [],
                "actions": [],
                "errors": [],
                "done_threshold": 0,
                "bond_distance": 0,
                "opt_ang": [],
                "save_circ": [],
                "time": [],
            }
        elif mode == "test":
            self.stats_file[mode][episode_no] = {
                "actions": [],
                "errors": [],
                "done_threshold": 0,
                "bond_distance": 0,
                "opt_ang": [],
                "save_circ": [],
            }

    def save_file(self):
        np.save(f"{self.rpath}/summary_{self.exp_seed}.npy", self.stats_file)

    def validate_stats(self, episode: int, mode: str):
        assert len(self.stats_file[mode][episode]["actions"]) == len(self.stats_file[mode][episode]["errors"]), "Stats mismatch"


def modify_state(state: torch.Tensor, env: CircuitEnv, conf: dict[str, Any]) -> torch.Tensor:
    if conf["agent"]["en_state"]:
        state = torch.cat((state, torch.tensor(env.prev_energy, dtype=torch.float, device=device).view(1)))
    if "threshold_in_state" in conf["agent"].keys() and conf["agent"]["threshold_in_state"]:
        state = torch.cat((state, torch.tensor(env.done_threshold, dtype=torch.float, device=device).view(1)))
    return state


def one_episode(
    episode_no: int,
    env: CircuitEnv,
    agent: DDQN.KAQN | KAQN.KAQN,
    episodes: int,
    conf: dict[str, Any],
):
    agent.saver.get_new_episode("train", episode_no)
    state = env.reset()
    agent.saver.stats_file["train"][episode_no]["done_threshold"] = env.done_threshold
    state = modify_state(state, env, conf)
    agent.policy_net.train()
    for itr in range(env.num_layers + 1):
        ill_action_from_env = env.update_illegal_actions()
        action, _ = agent.act(state, ill_action_from_env)
        assert isinstance(action, int)
        agent.saver.stats_file["train"][episode_no]["actions"].append(action)
        next_state, reward, done = env.step(agent.translate[action])
        next_state = modify_state(next_state, env, conf)
        agent.remember(state, torch.tensor(action, device=device), reward, next_state, torch.tensor(done, device=device))
        state = next_state.clone()
        assert isinstance(env.error, float)
        agent.saver.stats_file["train"][episode_no]["errors"].append(env.error)
        if agent.memory_reset_switch:
            if env.error < agent.memory_reset_threshold:
                agent.memory_reset_counter += 1
            if agent.memory_reset_counter == agent.memory_reset_switch:
                agent.memory.clean_memory()
                agent.memory_reset_switch = False
                agent.memory_reset_counter = False
        if done:
            if episode_no % 20 == 0:
                logger.info(f"episode: {episode_no}/{episodes}, score: {itr}, e: {agent.epsilon:.2}, rwd: {reward}")
            break
        if len(agent.memory) > conf["agent"]["batch_size"]:
            ratio = conf["agent"].get("replay_ratio", 1)
            if itr % ratio == 0:
                loss = agent.replay(conf["agent"]["batch_size"])
                assert isinstance(loss, float)
                agent.saver.stats_file["train"][episode_no]["loss"].append(loss)
                agent.saver.validate_stats(episode_no, "train")


def train(agent, env, episodes: int, seed: int, output_path: str, threshold: float):
    for e in range(episodes):
        one_episode(e, env, agent, episodes, conf)
        if e % 20 == 0 and e > 0:
            agent.saver.save_file()
            torch.save(agent.policy_net.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_model.pth")
            torch.save(agent.optim.state_dict(), f"{output_path}/thresh_{threshold}_{seed}_optim.pth")
            torch.save(
                {i: a._asdict() for i, a in enumerate(agent.memory.memory)},
                f"{output_path}/thresh_{threshold}_{seed}_replay_buffer.pth",
            )


def run_bell_ghz(args: argparse.Namespace):
    results_path = "results/"
    path_part = f"{results_path}{args.experiment_name}{args.config}"
    pathlib.Path(path_part).mkdir(parents=True, exist_ok=True)
    conf = get_config(args.experiment_name, f"{args.config}.cfg", verbose=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    if conf["env"]["type"] == "classic":
        environment = CircuitEnv(conf, device=device)
    agent_mod = getattr(agents, conf["agent"]["agent_type"])
    agent_class = getattr(agent_mod, conf["agent"]["agent_class"])
    agent = agent_class(conf, environment.action_size, environment.state_size, device)
    agent.saver = Saver(f"{results_path}{args.experiment_name}{args.config}", args.seed)
    if conf["agent"]["init_net"]:
        path = f"{results_path}{conf['agent']['init_net']}{args.seed}"
        agent.policy_net.load_state_dict(torch.load(path + "_model.pth"))
        agent.target_net.load_state_dict(torch.load(path + "_model.pth"))
        agent.optim.load_state_dict(torch.load(path + "_optim.pth"))
        agent.policy_net.eval()
        agent.target_net.eval()
        replay = torch.load(f"{path}_replay_buffer.pth")
        for i in replay.keys():
            agent.remember(**replay[i])
        if not conf["agent"]["epsilon_restart"]:
            agent.epsilon = agent.epsilon_min
    train(agent, environment, conf["general"]["episodes"], args.seed, path_part, conf["env"]["accept_err"])
    agent.saver.save_file()
    final_path = f"{path_part}/thresh_{conf['env']['accept_err']}_{args.seed}"
    torch.save(agent.policy_net.state_dict(), f"{final_path}_model.pth")
    torch.save(agent.optim.state_dict(), f"{final_path}_optim.pth")
    logger.info(f"Bell/GHZ experiment complete. Results in {path_part}")


def run_chemistry(args: argparse.Namespace):
    from chemistry.h2_vqe import H2VQETrainer
    from chemistry.lih_vqe import LiHVQETrainer
    from chemistry.beh2_fragment import BeH2FragmentVQETrainer

    output_base = f"results/{args.experiment}_vqe"
    pathlib.Path(output_base).mkdir(parents=True, exist_ok=True)

    if args.experiment == "h2":
        bl = args.bond_lengths or [0.5, 0.74, 1.0, 1.5, 2.0, 2.5]
        trainer = H2VQETrainer(bond_lengths=bl, agent_type=args.agent, device=device, output_dir=output_base)
        trainer.run()
    elif args.experiment == "lih":
        bl = args.bond_lengths or [1.2, 1.6, 2.0, 2.5, 3.0, 3.5]
        trainer = LiHVQETrainer(bond_lengths=bl, device=device, output_dir=output_base)
        trainer.run()
    elif args.experiment == "beh2":
        trainer = BeH2FragmentVQETrainer(bond_length=1.3, device=device, output_dir=output_base)
        trainer.run()


def run_hardware(args: argparse.Namespace):
    from chemistry.molecule import MolecularHamiltonian
    from hardware.hardware_eval import HardwareEvaluator
    from hardware.noise_aware_trainer import NoiseAwareTrainer

    if args.experiment == "h2":
        molecule = MolecularHamiltonian.h2(bond_length=0.74)
    elif args.experiment == "lih":
        molecule = MolecularHamiltonian.lih(bond_length=2.0)
    elif args.experiment == "beh2":
        molecule = MolecularHamiltonian.beh2(bond_length=1.3)
    else:
        molecule = MolecularHamiltonian.h2(bond_length=0.74)

    logger.info(f"Hardware mode: {args.mode}, molecule: {molecule.name}, {molecule.num_qubits} qubits")

    if args.mode == "train":
        trainer = NoiseAwareTrainer(molecule=molecule, num_qubits=molecule.num_qubits, device=device)
        results = trainer.train()
        logger.info(f"Training complete. Best energy: {results['best_energy']:.6f}")
    elif args.mode == "eval":
        evaluator = HardwareEvaluator(molecule, ibm_token=args.ibm_token)
        trainer = NoiseAwareTrainer(molecule=molecule, num_qubits=molecule.num_qubits, device=device)
        evaluator.evaluate_curriculum(trainer, run_on_hardware=(args.backend is not None))
    elif args.mode == "hardware":
        evaluator = HardwareEvaluator(molecule, ibm_token=args.ibm_token)
        trainer = NoiseAwareTrainer(molecule=molecule, num_qubits=molecule.num_qubits, device=device)
        evaluator.evaluate_curriculum(trainer, run_on_hardware=True)


def run_dashboard():
    logger.info("Starting Streamlit dashboard...")
    import subprocess
    subprocess.run(["streamlit", "run", "interpretability/streamlit_dashboard.py"])


def setup_wandb(args: argparse.Namespace):
    try:
        import wandb
        wandb.init(
            project="kanqas-nisq",
            config=vars(args),
            name=f"{args.experiment}_{args.agent}_{args.seed}",
        )
        return wandb
    except ImportError:
        logger.warning("wandb not installed. Skipping.")
        return None


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    device = torch.device(f"cpu:{0}")
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")

    logger.info(f"KANQAS-NISQ v2.0 | Device: {device} | Experiment: {args.experiment}")
    logger.info(f"Args: {args}")

    wandb_run = setup_wandb(args) if args.wandb else None

    if args.experiment in ("bell", "ghz"):
        run_bell_ghz(args)
    elif args.experiment in ("h2", "lih", "beh2"):
        if args.mode == "hardware":
            run_hardware(args)
        else:
            run_chemistry(args)
    elif args.experiment == "hardware":
        run_hardware(args)
    elif args.experiment == "dashboard":
        run_dashboard()
    else:
        console.print(f"[red]Unknown experiment: {args.experiment}[/red]")
        console.print("[yellow]Use --help for available options[/yellow]")
        sys.exit(1)

    logger.info("Done.")

# Keep the original import for backward compatibility
conf: dict[str, Any] = {}
device: torch.device = torch.device("cpu:0")
