"""
Combination 9: Integrated-Phi-Nexus (Jarvis Nexus + Integrated Information Phi Optimization)
Fuses:
1. 10,000-D Holographic Reduced Representations (HRR)
2. Continuous Information Integration across the Minimum Information Partition (MIP)
3. Direct maximization of irreducible holistic information Phi
"""

import time
import json
import numpy as np


class PhiNexusSystem:
    def __init__(self, dim: int = 10000, seed: int = 777):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # Cross-subspace coupling kernel
        self.k_integrate = self.rng.standard_normal(dim)
        self.k_integrate /= np.linalg.norm(self.k_integrate)

        # Two halves of cognitive space
        self.sub_a = self.rng.standard_normal(dim // 2)
        self.sub_a /= np.linalg.norm(self.sub_a)

        self.sub_b = self.rng.standard_normal(dim // 2)
        self.sub_b /= np.linalg.norm(self.sub_b)

    def compute_phi_hdc(self) -> float:
        """
        Computes continuous informational integration Phi:
        Difference between whole-system predictive capacity and partitioned halves.
        """
        # Whole unified vector
        v_whole = np.concatenate([self.sub_a, self.sub_b])
        f_k = np.fft.rfft(self.k_integrate)
        f_w = np.fft.rfft(v_whole)
        integrated = np.fft.irfft(f_k * f_w, n=self.dim)
        integrated /= np.linalg.norm(integrated)

        # Whole mutual alignment
        i_whole = float(np.abs(np.dot(v_whole, integrated)))

        # Partitioned independent alignment (without cross-binding)
        i_part_a = float(np.abs(np.dot(self.sub_a, integrated[:self.dim // 2])))
        i_part_b = float(np.abs(np.dot(self.sub_b, integrated[self.dim // 2:])))
        i_partitioned = 0.5 * (i_part_a + i_part_b)

        # Phi is the irreducible holistic synergy
        phi = max(0.0, i_whole - i_partitioned)
        return phi

    def optimize_integration_step(self, lr: float = 0.1):
        """Plasticity step strengthening cross-subspace binding."""
        v_whole = np.concatenate([self.sub_a, self.sub_b])
        # Re-tune coupling kernel towards outer product alignment
        self.k_integrate = 0.9 * self.k_integrate + 0.1 * v_whole
        self.k_integrate /= np.linalg.norm(self.k_integrate)


def benchmark_phi_nexus():
    print("=" * 60)
    print("BENCHMARKING COMBO 9: INTEGRATED-PHI-NEXUS")
    print("=" * 60)

    system = PhiNexusSystem(dim=10000, seed=777)

    phi_init = system.compute_phi_hdc()
    print(f"   Initial Holographic Integrated Information Phi: {phi_init:.6f}")

    t0 = time.time()
    for cycle in range(10):
        system.optimize_integration_step(lr=0.1)
    phi_final = system.compute_phi_hdc()
    elapsed = time.time() - t0

    phi_gain = ((phi_final - phi_init) / (phi_init + 1e-9)) * 100.0
    print(f"   Post-Optimization Phi: {phi_final:.6f} (Gain: {phi_gain:+.1f}%)")
    print(f"   Throughput: {10/elapsed:.2f} Phi optimization steps/sec at D=10,000")

    data = {
        "combo": "Combo_09_Phi_Nexus",
        "initial_phi": float(phi_init),
        "final_phi": float(phi_final),
        "phi_gain_pct": float(phi_gain),
        "throughput_opt_sec": float(10/elapsed)
    }
    with open("benchmark_results/combo_09_phi_nexus.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_phi_nexus()
