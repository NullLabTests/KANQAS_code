from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np

sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)

from chemistry.molecule import MolecularHamiltonian
from chemistry.vqe_env import VQEEnv
from agents.KAQN import KAQN
from utils import dictionary_of_actions

bond_length = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
output_dir = f"results/chemistry/lih_BL{bond_length:.2f}"

import torch
device = torch.device('cpu')

mol = MolecularHamiltonian.lih(bond_length=bond_length)
exact_e = mol.exact_diagonalization()
print(f"LiH R={bond_length:.2f}A | {mol.num_qubits} qubits | Exact GS={exact_e:.6f} Ha")

conf = {
    'general': {'episodes': 1500},
    'env': {
        'num_layers': 16, 'fn_type': 'fidelty_reward', 'curriculum_type': 'MovingThreshold',
        'accept_err': 5.0, 'shift_threshold_ball': 0.5, 'shift_threshold_time': 20,
        'success_thresh': 5, 'succ_radius_shift': 10, 'succes_switch': 5.0,
        'depth_penalty': 0.0005, 'gate_penalty': 0.0001,
    },
    'problem': {'type': 'lih_vqe', 'noise': 0, 'noise_prob_1q': 0.001, 'noise_prob_2q': 0.01},
    'agent': {
        'agent_type': 'KAQN', 'agent_class': 'KAQN', 'angles': False,
        'en_state': False, 'threshold_in_state': False,
        'memory_size': 500000, 'batch_size': 256,
        'learning_rate': 0.0003, 'final_gamma': 0.99,
        'epsilon_min': 0.02, 'epsilon_decay': 0.9998,
        'update_target_net': 30, 'dropout': 0.1,
        'neurons': [128, 64], 'kan_seed': 42, 'k': 3, 'grid': 5,
        'memory_reset_switch': 50, 'memory_reset_threshold': 0.01,
    },
}

env = VQEEnv(conf, mol, device)
agent = KAQN(conf, env.action_size, env.state_size, device)
translate = dictionary_of_actions(mol.num_qubits)

episodes = conf['general']['episodes']
best_energy = float('inf')
best_circuit = None
epsilon_reset_interval = 500

t0 = time.time()
for ep in range(episodes):
    if ep > 0 and ep % epsilon_reset_interval == 0:
        agent.epsilon = min(agent.epsilon + 0.15, 0.3)
    state = env.reset()
    episode_energy = float('inf')
    for step in range(env.num_layers + 1):
        ill = env.update_illegal_actions()
        a, _ = agent.act(state, ill)
        ns, r, d = env.step(translate[a])
        agent.remember(state, torch.tensor(a, device=device), r, ns, torch.tensor(d, device=device))
        state = ns
        episode_energy = min(episode_energy, float(env.energy))
        if len(agent.memory) > conf['agent']['batch_size']:
            agent.replay(conf['agent']['batch_size'])
        if d:
            break
    if episode_energy < best_energy:
        best_energy = episode_energy
        best_circuit = env.make_circuit()
    if ep % 100 == 0:
        err = abs(episode_energy - exact_e)
        print(f"  Ep {ep}/{episodes} | E={episode_energy:.6f} | err={err:.6f} | best={best_energy:.6f} | eps={agent.epsilon:.3f}")

# Post-optimization with HEA
print("  Post-optimizing with HEA...")
from scipy.optimize import minimize
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit

def build_param_circ(circ):
    params = []
    new_circ = QuantumCircuit(circ.num_qubits)
    for inst, qargs, cargs in circ.data:
        q = qargs[0]
        if inst.name == 'x':
            p = Parameter(f't{len(params)}'); params.append(p)
            new_circ.rx(float(np.pi) + p, q)
        elif inst.name == 'y':
            p = Parameter(f't{len(params)}'); params.append(p)
            new_circ.ry(float(np.pi) + p, q)
        elif inst.name == 'z':
            p = Parameter(f't{len(params)}'); params.append(p)
            new_circ.rz(float(np.pi) + p, q)
        elif inst.name == 'h':
            new_circ.h(q)
        elif inst.name == 't':
            new_circ.t(q)
        elif inst.name == 'cx':
            new_circ.cx(q, qargs[1])
        else:
            new_circ.append(inst, qargs, cargs)
    nq = circ.num_qubits
    for _ in range(2):
        for q2 in range(nq):
            p = Parameter(f't{len(params)}'); params.append(p)
            new_circ.ry(p, q2)
        for q2 in range(nq - 1):
            new_circ.cx(q2, q2 + 1)
        for q2 in range(nq):
            p = Parameter(f't{len(params)}'); params.append(p)
            new_circ.rz(p, q2)
    return new_circ, params

param_circ, params = build_param_circ(best_circuit or env.make_circuit())
x0 = np.zeros(len(params))
bounds = [(-np.pi, np.pi)] * len(params)
best_x, best_pe = x0.copy(), float('inf')

def obj(x):
    global best_x, best_pe
    bc = param_circ.assign_parameters({p: v for p, v in zip(params, x)})
    e = mol.estimate_energy(bc)
    if e < best_pe:
        best_pe = e; best_x = x.copy()
    return e

minimize(obj, x0, method='COBYLA', bounds=bounds, options={'maxiter': 1000, 'rhobeg': 0.5, 'catol': 1e-6})

elapsed = time.time() - t0
final_circ = param_circ.assign_parameters({p: v for p, v in zip(params, best_x)})
gc = final_circ.count_ops()
print(f"LiH R={bond_length:.2f}A | RL={best_energy:.6f} | PostOpt={best_pe:.6f} | Exact={exact_e:.6f} | Error={abs(best_pe-exact_e):.6f} | CX={gc.get('cx',0)} | Time={elapsed:.0f}s")

summary = {
    'bond_length': bond_length, 'rl_energy': float(best_energy),
    'found_energy': float(best_pe), 'exact_energy': float(exact_e),
    'error': abs(best_pe - exact_e), 'cnot': gc.get('cx', 0),
    'depth': final_circ.depth(), 'elapsed_seconds': elapsed,
    'chemical_precision': abs(best_pe - exact_e) < 0.0016,
}
with open(f"{output_dir}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
np.save(f"{output_dir}/summary.npy", summary)
