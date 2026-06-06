from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np

sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)

from chemistry.h2_vqe import H2VQETrainer

bond_length = float(sys.argv[1]) if len(sys.argv) > 1 else 0.74
output_dir = f"results/chemistry/h2_BL{bond_length:.2f}"

trainer = H2VQETrainer(bond_lengths=[bond_length], agent_type='KAQN', output_dir=output_dir)
trainer.config['general']['episodes'] = 1500
trainer.config['general']['post_optimize'] = True
trainer.config['general']['post_optimize_iter'] = 800
trainer.config['env']['num_layers'] = 20
trainer.config['env']['depth_penalty'] = 0.0003
trainer.config['env']['gate_penalty'] = 0.0001
trainer.config['env']['accept_err'] = 4.0
trainer.config['env']['shift_threshold_ball'] = 0.3
trainer.config['env']['shift_threshold_time'] = 15
trainer.config['agent']['epsilon_decay'] = 0.9998
trainer.config['agent']['epsilon_min'] = 0.02
trainer.config['agent']['memory_size'] = 500000
trainer.config['agent']['batch_size'] = 256
trainer.config['agent']['learning_rate'] = 0.0003
trainer.config['agent']['update_target_net'] = 30

t0 = time.time()
r = trainer.run()
elapsed = time.time() - t0

summary = {
    'bond_length': float(r['bond_lengths'][0]),
    'rl_energy': float(r.get('rl_energies', [0])[0] or 0),
    'found_energy': float(r['found_energies'][0]),
    'exact_energy': float(r['exact_energies'][0]),
    'error': abs(float(r['found_energies'][0]) - float(r['exact_energies'][0])),
    'cnot': int(r['cnot_counts'][0]),
    'depth': int(r['depths'][0]),
    'elapsed_seconds': elapsed,
    'chemical_precision': abs(float(r['found_energies'][0]) - float(r['exact_energies'][0])) < 0.0016,
}

with open(f"{output_dir}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
np.save(f"{output_dir}/summary.npy", summary)

print(f"=== H2 R={bond_length:.2f} | RL={summary['rl_energy']:.6f} | Final={summary['found_energy']:.6f} | Exact={summary['exact_energy']:.6f} | Error={summary['error']:.6f} | ChemPrec={summary['chemical_precision']} | Time={elapsed:.0f}s ===")
