# KANQAS-NISQ Progress Log

## 2026-06-06 — Phase A Long-Run Campaign (Day 1)

### Current Best Errors (H₂, STO-3G, 4 qubits)
| Bond (Å) | KAQN Energy (Ha) | Exact FCI (Ha) | Error (Ha) | CX | Depth | Episodes |
|----------|-----------------|----------------|------------|-----|-------|----------|
| 0.50     | -1.967628       | -2.113514      | 0.145886   | —   | —     | 1500     |
| 0.74     | -1.831864       | -1.852388      | 0.020525   | 3   | 5     | 1500     |
| 1.00     | -1.630328       | -1.630328      | 0.000000   | —   | —     | 1500     |
| 1.50     | -1.350934       | -1.350934      | 0.000000   | 5   | 6     | 2000     |
| 2.00     | -1.213230       | -1.213230      | 0.000000   | —   | —     | 1000     |
| 2.50     | -1.143309       | -1.147726      | 0.004416   | —   | —     | 1000     |

### Key Bugfix
- `gs_energy` was set to HF energy (`molecule.reference_energy`) instead of FCI ground state (`molecule.exact_diagonalization()`). Fixed in checkpoint-2.
- Post-optimization (COBYLA) added for converting discrete rotations to continuous Rx/Ry/Rz(θ).

### Phase B (Hardware Pipeline)
- Noise-aware training with FakeBrisbane backend (127 qubit, ECR gates, NoiseModel.from_backend)
- ZNE error mitigation via Mitiq (available, works on short circuits)
- `get_backend(mode='fake'|'real')` in ibm_runtime.py
- `run_phase_d.py --backend {fake,real}` CLI

### Phase C (Dashboard)
- 5-tab Streamlit app running on port 8501
- Hardware Results tab added (noise-aware training history, best circuit display)

### Phase D (Infrastructure)
- `run_phase_d.py` pipeline: train → evaluate → real hardware submit (if token available)
- Results directories: results/chemistry/, results/hardware/, results/benchmark/

### Next Goals
1. Run 1000+ episode H₂ training at R=0.74Å targeting error < 0.0016 Ha
2. Run H₂ at all 6 bond lengths (1000+ eps each), save CSVs + convergence plots
3. LiH (6 qubit) long training
4. Real IBM hardware evaluation with saved job IDs
5. Streamlit dashboard with live results from results/chemistry/

### Git Tags
- checkpoint-1: Initial Phase A structure
- checkpoint-2: gs_energy bugfix + post-optimization
- checkpoint-3: Phase B hardware pipeline + dashboard
- checkpoint-4: FakeBrisbane default + run_phase_d.py

### Repository
- https://github.com/NullLabTests/KANQAS_code
- Branch: feature/phase-A-chemical-precision
