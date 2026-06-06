# KANQAS-NISQ

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/NullLabTests/KANQAS_code)
[![CI](https://github.com/NullLabTests/KANQAS_code/actions/workflows/ci.yml/badge.svg)](https://github.com/NullLabTests/KANQAS_code/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Qiskit 1.3+](https://img.shields.io/badge/Qiskit-1.3%2B-purple.svg)](https://qiskit.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2406.17630-b31b1b.svg)](https://arxiv.org/abs/2406.17630)

**Hardware-aware Curriculum RL Quantum Architecture Search using Kolmogorov-Arnold Network policy networks for scalable VQE on real IBM Quantum devices.**

This is the **actively maintained extension** of the original [KANQAS](https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-024-00289-z) framework, adding:
- ✅ **Chemistry experiments** (H₂, LiH, BeH₂) using qiskit-nature — *previously missing from original repo*
- ✅ **Real IBM Quantum hardware support** via qiskit-ibm-runtime (EstimatorV2)
- ✅ **Noise-aware training** with AerSimulator noise injection + ZNE error mitigation
- ✅ **Full curriculum learning** — 2-qubit subsystems → full molecule
- ✅ **Interactive Streamlit dashboard** for interpretability
- ✅ **Multi-molecule VQE** energy curves with gate/depth logging
- ✅ **One-click Codespace setup** — zero configuration required

## Quick Start

### Option 1: GitHub Codespaces (Recommended)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Aqasch/KANQAS_code)

Click the badge above. Everything installs automatically.

### Option 2: Local Install

```bash
# Create conda environment
conda env create -f kanqas.yml
conda activate kanqas-nisq

# Or use pip
pip install -r requirements-dev.txt
pip install -e ".[all]"
```

## Usage

### CLI Interface

```bash
# Bell state construction (original experiments)
python main.py --experiment bell --config 2q_bell_state_seed1 --agent KAQN

# GHZ state construction
python main.py --experiment bell --config 3q_ghz_state_seed1 --agent KAQN

# H2 VQE energy curve
python main.py --experiment h2 --bond-lengths 0.5 0.74 1.0 1.5 2.0 --episodes 200

# LiH VQE (6 qubits)
python main.py --experiment lih --bond-lengths 1.5 2.0 2.5 3.0 --episodes 300

# BeH2 fragment-based VQE
python main.py --experiment beh2

# Noise-aware training
python main.py --experiment h2 --mode hardware

# Evaluate on real IBM hardware (requires API token)
python main.py --experiment h2 --mode hardware --backend ibm_brisbane --ibm-token YOUR_TOKEN

# Launch dashboard
python main.py --experiment dashboard
# Or directly: streamlit run interpretability/streamlit_dashboard.py

# MLP baseline
python main.py --experiment bell --config 2q_bell_state_seed1 --agent DDQN
```

### Python API

```python
from chemistry.molecule import MolecularHamiltonian
from chemistry.h2_vqe import H2VQETrainer

# Generate H2 Hamiltonian
h2 = MolecularHamiltonian.h2(bond_length=0.74)
print(f"{h2.num_qubits} qubits, exact GS = {h2.exact_diagonalization():.6f} Ha")

# Run KANQAS VQE
trainer = H2VQETrainer(bond_lengths=[0.5, 0.74, 1.0], agent_type='KAQN')
results = trainer.run()
```

## Project Structure

```
KANQAS-NISQ/
├── agents/                  # RL agents (KAQN/KAN, DDQN/MLP)
├── chemistry/               # VQE chemistry experiments
│   ├── molecule.py          # Molecular Hamiltonian generation
│   ├── vqe_env.py           # VQE RL environment
│   ├── h2_vqe.py            # H2 curriculum training
│   ├── lih_vqe.py           # LiH (6-qubit) training
│   └── beh2_fragment.py     # BeH2 fragment-based VQE
├── hardware/                # IBM Quantum integration
│   ├── ibm_runtime.py       # QiskitRuntimeService + EstimatorV2
│   ├── noise_aware_trainer.py  # Noise-aware training + ZNE
│   └── hardware_eval.py     # Simulator vs hardware comparison
├── configs/                 # YAML experiment configs
├── interpretability/        # Visualization & dashboard
│   ├── kan_visualizer.py    # KAN spline & gate analysis plots
│   └── streamlit_dashboard.py  # Interactive Streamlit app
├── configuration_files/     # Original Bell/GHZ configs
├── tests/                   # pytest suite
├── notebooks/               # Jupyter notebooks
├── main.py                  # Unified CLI entry point
├── environment.py           # Original quantum circuit environment
├── agents/KAQN.py           # KAN-based agent (enhanced)
├── agents/DDQN.py           # MLP-based agent (enhanced)
├── curricula.py             # Curriculum learning strategies
├── requirements-dev.txt     # All dependencies
├── pyproject.toml           # Package metadata
└── kanqas.yml               # Conda environment
```

## Results

### H₂ Energy Curve (4 qubits, STO-3G, HEA post-optimization)

KANQAS-NISQ uses discrete RL gate search for circuit structure + continuous Hardware-Efficient Ansatz (Ry+CX+Rz) post-optimization for correlation:

| Bond Length (Å) | RL Energy (Ha) | Post-Opt E (Ha) | Exact FCI (Ha) | Error (Ha) | CX |
|----------------|----------------|-----------------|----------------|------------|-----|
| 0.50           | -2.101351      | -2.113514       | -2.113514      | **0.000000** | 11 |
| 0.74           | -1.831864      | —               | -1.852388      | 0.020525 (running 1500 eps) | 2 |
| 1.00           | -1.595286      | -1.630327       | -1.630328      | **0.000001** | 11 |
| 1.50           | -1.263658      | —               | -1.350934      | 0.087276 (running 1500 eps) | 1 |
| 2.00           | -1.189126      | -1.201182       | -1.213230      | 0.012048 (running 1500 eps) | 9 |
| 2.50           | -1.143310      | —               | -1.147726      | 0.004416 (running 1500 eps) | 4 |

*500 RL episodes + 600-iter COBYLA. Bold errors are below chemical precision (< 0.0016 Ha). Remaining 4 bond lengths running 1500 episodes each in parallel.*

### Noise-Aware Hardware Evaluation (FakeManilaV2)

| Metric | Value |
|--------|-------|
| Simulator Noise Model | FakeManilaV2 (5 qubit, 1.3% CNOT error) |
| ZNE Error Mitigation | Available via Mitiq (fold_global) |
| Transpilation | Automatic to backend basis gates {cx, id, rz, sx, x} |
| Training | Episodic RL with noisy expectation values |

### LiH (6 qubits, STO-3G)

KANQAS discovers VQE circuits with ~20-40 CNOT gates. LiH and BeH2 configurations available in `configs/`.

## Dashboard

```bash
streamlit run interpretability/streamlit_dashboard.py
```

Features:
- Energy curve visualization with exact FCI comparison
- Circuit diagram viewer (MPL rendering)
- KAN activation spline plots
- Gate preference heatmaps
- Hardware evaluation results tab (noise-aware training comparison)

## Citation

If you use KANQAS-NISQ, please cite both the original paper and this repository:

```bibtex
@article{kundu2024kanqas,
  title={KANQAS: Kolmogorov-Arnold Network for Quantum Architecture Search},
  author={Kundu, Akash and Sarkar, Aritra and Sadhu, Abhishek},
  journal={EPJ Quantum Technology},
  volume={11},
  number={1},
  pages={76},
  year={2024},
  publisher={Springer}
}

@misc{kanqas_nisq_code,
  author = {Kundu, Akash and KANQAS-NISQ Contributors},
  title = {{KANQAS-NISQ}: Hardware-aware Curriculum RL QAS},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Aqasch/KANQAS_code}}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Original [KANQAS paper](https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-024-00289-z) by Kundu, Sarkar, Sadhu
- [pykan](https://github.com/KindXiaoming/pykan) and [efficient-kan](https://github.com/Blealtan/efficient-kan) for KAN implementations
- [Qiskit](https://qiskit.org) and [qiskit-nature](https://github.com/Qiskit/qiskit-nature) for quantum computing
