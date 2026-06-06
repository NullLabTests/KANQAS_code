from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer

from agents.KAQN import KAQN
from chemistry.molecule import MolecularHamiltonian
from interpretability.kan_visualizer import KANVisualizer

st.set_page_config(
    page_title="KANQAS-NISQ Dashboard",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_results(path: str) -> dict | None:
    path = Path(path)
    if path.exists():
        try:
            return np.load(path, allow_pickle=True).item()
        except Exception as e:
            st.error(f"Failed to load {path}: {e}")
    return None


def main():
    st.title("⚛ KANQAS-NISQ Interpretability Dashboard")
    st.markdown(
        "Hardware-aware Curriculum RL Quantum Architecture Search using "
        "KAN policy networks for scalable VQE"
    )

    with st.sidebar:
        st.header("Configuration")
        molecule_choice = st.selectbox(
            "Molecule",
            ["H2", "LiH", "BeH2"],
            index=0,
        )
        result_path = st.text_input(
            "Results path",
            value=f"results/{molecule_choice.lower()}_vqe/h2_vqe_results.npy",
        )
        theme = st.selectbox("Theme", ["Dark", "Light"], index=0)
        st.markdown("---")
        st.markdown("**Quick Links**")
        st.markdown("[KANQAS Paper](https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-024-00289-z)")
        st.markdown("[GitHub Repo](https://github.com/Aqasch/KANQAS_code)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Energy Curves",
        "Circuit Visualization",
        "KAN Activations",
        "Gate Analysis",
    ])

    with tab1:
        st.header("Energy Curves")
        results = load_results(result_path)
        if results and "bond_lengths" in results:
            viz = KANVisualizer()
            fig = viz.plot_energy_curve(
                results["bond_lengths"],
                results["found_energies"],
                results["exact_energies"],
                molecule_name=molecule_choice,
            )
            st.pyplot(fig)

            col1, col2, col3 = st.columns(3)
            with col1:
                best_error = min(
                    abs(np.array(results["found_energies"]) - np.array(results["exact_energies"]))
                )
                st.metric("Best Energy Error", f"{best_error:.6f} Ha")
            with col2:
                st.metric("Points Computed", str(len(results["bond_lengths"])))
            with col3:
                if "depths" in results:
                    avg_depth = np.mean(results["depths"])
                    st.metric("Avg Circuit Depth", f"{avg_depth:.1f}")
        else:
            st.info("No results loaded. Run an experiment first.")
            if st.button("Run H2 demo"):
                st.write("Running demo... (placeholder)")

    with tab2:
        st.header("Circuit Visualization")
        if results and "final_circuit" in results:
            circ = results["final_circuit"]
            if isinstance(circ, QuantumCircuit):
                fig, ax = circuit_drawer(circ, output="mpl", style=theme.lower())
                st.pyplot(fig)
                ops = circ.count_ops()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Qubits", circ.num_qubits)
                with col2:
                    st.metric("Depth", circ.depth())
                with col3:
                    st.metric("Total Gates", sum(ops.values()))
                with col4:
                    st.metric("CNOT Gates", ops.get("cx", 0))
            else:
                st.error("Circuit object is not a QuantumCircuit instance")
        else:
            st.info("No circuit available. Run an experiment first.")

    with tab3:
        st.header("KAN Activation Visualization")
        st.info(
            "KAN activation splines show how the learned Kolmogorov-Arnold Network "
            "transforms each input feature into quantum gate selections."
        )
        if results and "agent" in results:
            agent = results["agent"]
            if isinstance(agent, KAQN):
                viz = KANVisualizer()
                fig = viz.plot_spline_activations(agent)
                st.pyplot(fig)
            else:
                st.warning("Loaded agent is not a KAQN instance")
        else:
            st.info("No trained KAN agent available in results.")

    with tab4:
        st.header("Gate Analysis")
        st.markdown("""
        The gate preference heatmap shows which qubits are most frequently targeted 
        by the KAN-discovered circuit architecture. This reveals structural patterns 
        in the learned ansatz.
        """)
        if results and "final_circuit" in results:
            circ = results["final_circuit"]
            if isinstance(circ, QuantumCircuit):
                viz = KANVisualizer()
                fig = viz.plot_gate_preference_heatmap(circ)
                st.pyplot(fig)
                with st.expander("Gate Count Details"):
                    ops = circ.count_ops()
                    st.json({str(k): int(v) for k, v in ops.items()})
            else:
                st.error("Circuit object is not a QuantumCircuit instance")
        else:
            st.info("No circuit available for gate analysis.")

    with st.expander("About KANQAS-NISQ"):
        st.markdown("""
        **KANQAS-NISQ** extends the original KANQAS framework with:
        - Hardware-aware training with noise injection
        - Real IBM Quantum device evaluation
        - Chemistry experiments (H₂, LiH, BeH₂) via qiskit-nature
        - Curriculum learning with MovingThreshold
        - Full interpretability dashboard
        - ZNE error mitigation
        """)

    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Qiskit, PyTorch, and Streamlit | "
        "[Original KANQAS Paper](https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-024-00289-z)"
    )


if __name__ == "__main__":
    main()
