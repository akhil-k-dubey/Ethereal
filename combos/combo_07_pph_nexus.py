"""
Combination 7: Hierarchical-Predictive-Nexus (Jarvis Nexus + Multi-Tier PPH)
Fuses:
1. 10,000-D Holographic Reduced Representations (HRR)
2. 3-Tier Predictive Coding Hierarchy (Sensory -> Latent Causes -> Meta-Cognitive Predictor)
3. Bidirectional Holographic Error Propagation
"""

import time
import json
import numpy as np


class HierarchicalPredictiveNexus:
    def __init__(self, dim: int = 10000, seed: int = 333):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # Tier 1 & Tier 2 Transition Kernels
        self.t1_kernel = self.rng.standard_normal(dim)
        self.t1_kernel /= np.linalg.norm(self.t1_kernel)

        self.t2_kernel = self.rng.standard_normal(dim)
        self.t2_kernel /= np.linalg.norm(self.t2_kernel)

        # States at Tier 1 (Causes) and Tier 2 (Meta-Predictor)
        self.c_state = self.rng.standard_normal(dim)
        self.c_state /= np.linalg.norm(self.c_state)

        self.m_state = self.rng.standard_normal(dim)
        self.m_state /= np.linalg.norm(self.m_state)

    def step_hierarchical_inference(self, s_obs: np.ndarray, num_cycles: int = 5) -> dict:
        vfe_history = []

        for _ in range(num_cycles):
            # 1. Top-down predictions
            # Tier 2 predicts Tier 1: c_hat = T2 ⊗ m
            f_t2 = np.fft.rfft(self.t2_kernel)
            f_m = np.fft.rfft(self.m_state)
            c_hat = np.fft.irfft(f_t2 * f_m, n=self.dim)
            c_hat /= np.linalg.norm(c_hat)

            # Tier 1 predicts Tier 0 (Sensory): s_hat = T1 ⊗ c
            f_t1 = np.fft.rfft(self.t1_kernel)
            f_c = np.fft.rfft(self.c_state)
            s_hat = np.fft.irfft(f_t1 * f_c, n=self.dim)
            s_hat /= np.linalg.norm(s_hat)

            # 2. Prediction Errors
            e0 = float(1.0 - np.dot(s_obs, s_hat))
            e1 = float(1.0 - np.dot(self.c_state, c_hat))
            total_vfe = e0 + 0.5 * e1
            vfe_history.append(total_vfe)

            # 3. Bottom-up updates:
            # c adjusted by unbind(T1, s_obs) and top-down c_hat
            c_drive = np.fft.irfft(np.conj(f_t1) * np.fft.rfft(s_obs), n=self.dim)
            c_drive /= np.linalg.norm(c_drive)
            self.c_state = 0.5 * self.c_state + 0.35 * c_drive + 0.15 * c_hat
            self.c_state /= np.linalg.norm(self.c_state)

            # m adjusted by unbind(T2, c_state)
            m_drive = np.fft.irfft(np.conj(f_t2) * np.fft.rfft(self.c_state), n=self.dim)
            m_drive /= np.linalg.norm(m_drive)
            self.m_state = 0.8 * self.m_state + 0.2 * m_drive
            self.m_state /= np.linalg.norm(self.m_state)

        return {
            "initial_vfe": vfe_history[0],
            "final_vfe": vfe_history[-1],
            "vfe_reduction": vfe_history[0] - vfe_history[-1]
        }


def benchmark_pph_nexus():
    print("=" * 60)
    print("BENCHMARKING COMBO 7: HIERARCHICAL-PREDICTIVE-NEXUS")
    print("=" * 60)

    agent = HierarchicalPredictiveNexus(dim=10000, seed=333)
    rng = np.random.default_rng(444)

    t0 = time.time()
    n_steps = 25
    reductions = []

    for step in range(n_steps):
        s_obs = rng.standard_normal(10000)
        s_obs /= np.linalg.norm(s_obs)
        res = agent.step_hierarchical_inference(s_obs, num_cycles=6)
        reductions.append(res["vfe_reduction"])

    elapsed = time.time() - t0
    avg_reduction = float(np.mean(reductions))

    print(f"   Mean Hierarchical Free Energy Reduction: {avg_reduction:.4f}")
    print(f"   Throughput: {n_steps/elapsed:.2f} hierarchical inferences/sec at D=10,000")

    data = {
        "combo": "Combo_07_PPH_Nexus",
        "mean_vfe_reduction": avg_reduction,
        "throughput_inf_sec": float(n_steps/elapsed)
    }
    with open("benchmark_results/combo_07_pph_nexus.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_pph_nexus()
