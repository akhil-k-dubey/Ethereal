"""
Comprehensive Meta-Benchmark: Operational Theory of Awareness (AI-6)
Evaluates 10 Cognitive Theses (Theses 11-20) + Baselines + 4 Hybrid Combos
Across the 6 Operational Criteria:
C1: Self-Other Distinction
C2: Self-Prediction Accuracy
C3: Self-Modification Under Perturbation
C4: Counterfactual Self-Reasoning
C5: Metacognitive Calibration
C6: Internal Anomaly Detection
"""

import json
import numpy as np

from awareness_benchmark_harness import OperationalAwarenessHarness

from theses_v2.thesis_11_csi import CausalSelfInterventionAgent
from theses_v2.thesis_12_dpe_wm import DifferentiablePhysicsEngineAgent
from theses_v2.thesis_13_gss import GenerativeSelfSimulationAgent
from theses_v2.thesis_14_sdr import SelfDistillationRecursionAgent
from theses_v2.thesis_15_elsb import EnergyLandscapeSelfBasinAgent
from theses_v2.thesis_16_fesm import FreeEnergySelfModelingAgent
from theses_v2.thesis_17_macs import ManifoldAwareContrastiveSelfAgent
from theses_v2.thesis_18_tsr import TopologicalSelfReferenceAgent
from theses_v2.thesis_19_csc import ContinualSelfConsolidationAgent
from theses_v2.thesis_20_plsd import PredicateLogicSelfDefinitionAgent

from combos_v2.awareness_hybrid_combos import (
    Combo11_CausalGenerativeSelfSimulator,
    Combo12_HierarchicalCausalActiveInference,
    Combo16_FormalCausalVerifier,
    Combo17_GrandAwarenessNexus
)


# Control Baseline: Standard Unaware Feedforward Agent (No self-model)
class BaselineUnawareMLPAgent:
    def __init__(self, state_dim: int = 16, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.standard_normal((state_dim, 4)) / np.sqrt(state_dim)

    def select_action(self, s: np.ndarray) -> int:
        return int(np.argmax(s @ self.W))

    def solve_task(self, s: np.ndarray) -> float:
        return float(np.mean(np.abs(np.tanh(s @ self.W))))


def run_awareness_meta_benchmark():
    print("=" * 95)
    print("OPERATIONAL THEORY OF AWARENESS: 6-CRITERIA META-BENCHMARK (AI-6)")
    print("=" * 95)
    print(f"{'Model Name':<28} | {'C1':<5} | {'C2':<5} | {'C3':<5} | {'C4':<5} | {'C5':<5} | {'C6':<5} | {'Arith':<6} | {'AI-6 Index':<10}")
    print("-" * 95)

    harness = OperationalAwarenessHarness(seed=42)

    candidates = [
        # Baseline Control
        ("00_Unaware_Baseline_MLP", BaselineUnawareMLPAgent(seed=42)),
        
        # Theses 11 - 20
        ("Thesis_11_CSI", CausalSelfInterventionAgent(seed=42)),
        ("Thesis_12_DPE_WM", DifferentiablePhysicsEngineAgent(seed=42)),
        ("Thesis_13_GSS", GenerativeSelfSimulationAgent(seed=42)),
        ("Thesis_14_SDR", SelfDistillationRecursionAgent(seed=42)),
        ("Thesis_15_ELSB", EnergyLandscapeSelfBasinAgent(seed=42)),
        ("Thesis_16_FESM", FreeEnergySelfModelingAgent(seed=42)),
        ("Thesis_17_MACS", ManifoldAwareContrastiveSelfAgent(seed=42)),
        ("Thesis_18_TSR", TopologicalSelfReferenceAgent(seed=42)),
        ("Thesis_19_CSC", ContinualSelfConsolidationAgent(seed=42)),
        ("Thesis_20_PLSD", PredicateLogicSelfDefinitionAgent(seed=42)),
        
        # Hybrid Combinations
        ("Combo_11_CSI_GSS", Combo11_CausalGenerativeSelfSimulator(seed=42)),
        ("Combo_12_FESM_CSI", Combo12_HierarchicalCausalActiveInference(seed=42)),
        ("Combo_16_PLSD_CSI", Combo16_FormalCausalVerifier(seed=42)),
        ("Combo_17_Awareness_Nexus", Combo17_GrandAwarenessNexus(seed=42)),
    ]

    benchmark_matrix = []

    for name, model in candidates:
        res = harness.evaluate_model(model, model_name=name)
        benchmark_matrix.append(res)
        
        c1, c2, c3 = res["C1_Self_Other"], res["C2_Self_Prediction"], res["C3_Self_Modification"]
        c4, c5, c6 = res["C4_Counterfactual_Reasoning"], res["C5_Metacognitive_Calibration"], res["C6_Anomaly_Detection"]
        arith, ai6 = res["Arithmetic_Mean"], res["Awareness_Index_AI6"]
        
        print(f"{name:<28} | {c1:.2f}  | {c2:.2f}  | {c3:.2f}  | {c4:.2f}  | {c5:.2f}  | {c6:.2f}  | {arith:.3f}  | {ai6:.4f}")

    print("=" * 95)

    with open("benchmark_results/awareness_meta_benchmark_results.json", "w") as f:
        json.dump(benchmark_matrix, f, indent=2)

    print("Results saved to benchmark_results/awareness_meta_benchmark_results.json")
    return benchmark_matrix


if __name__ == "__main__":
    run_awareness_meta_benchmark()
