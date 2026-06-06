from __future__ import annotations

import json
import logging
import time

import numpy as np

from chemistry.h2_vqe import H2VQETrainer

logging.basicConfig(level=logging.INFO)

trainer = H2VQETrainer(bond_lengths=[0.74], agent_type='KAQN', output_dir='results/chemistry/h2_R074_long')
trainer.config['general']['episodes'] = 2000
trainer.config['general']['post_optimize'] = True
trainer.config['general']['post_optimize_iter'] = 500
trainer.config['env']['num_layers'] = 16
trainer.config['env']['depth_penalty'] = 0.0005
trainer.config['env']['gate_penalty'] = 0.0002
trainer.config['env']['accept_err'] = 3.0
trainer.config['env']['shift_threshold_ball'] = 0.5
trainer.config['agent']['epsilon_decay'] = 0.9998
trainer.config['agent']['epsilon_min'] = 0.01
trainer.config['agent']['memory_size'] = 500000
trainer.config['agent']['batch_size'] = 256
trainer.config['agent']['learning_rate'] = 0.0003
t0 = time.time()
r = trainer.run()
elapsed = time.time() - t0
summary = {
    'bond_lengths': [float(x) for x in r['bond_lengths']],
    'found_energies': [float(x) for x in r['found_energies']],
    'exact_energies': [float(x) for x in r['exact_energies']],
    'errors': [abs(float(r['found_energies'][i]) - float(r['exact_energies'][i])) for i in range(len(r['bond_lengths']))],
    'cnot_counts': [int(x) for x in r['cnot_counts']],
    'depths': [int(x) for x in r['depths']],
    'elapsed_seconds': elapsed,
}
with open('results/chemistry/h2_R074_long/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
np.save('results/chemistry/h2_R074_long/summary.npy', summary)
print(f'=== H2 R=0.74 DONE: found={summary["found_energies"][0]:.6f} exact={summary["exact_energies"][0]:.6f} error={summary["errors"][0]:.6f} time={elapsed:.0f}s ===')
