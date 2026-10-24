"""
Combination 3: KAN-Nexus (Jarvis Nexus + Kolmogorov-Arnold Splines)
Fuses:
1. 10,000-D Holographic Reduced Representations (HRR)
2. Kolmogorov-Arnold Spline Operators for Strange-Loop Stabilization
3. Suppresses holographic crosstalk noise through smooth continuous univariate spline manifolds
"""

import time
import json
import numpy as np


class KANSplineOperator:
    """Univariate spline basis function operator over high-dimensional vectors."""
    def __init__(self, dim: int = 10000, n_basis: int = 4, seed: int = 505):
        self.dim = dim
        self.n_basis = n_basis
        self.rng = np.random.default_rng(seed)

        # Learnable spline coefficients per coordinate
        self.coeffs = self.rng.standard_normal((dim, n_basis)) * 0.05
        # Linear residual
        self.w_lin = self.rng.standard_normal(dim) * 0.1

    def forward(self, v: np.ndarray) -> np.ndarray:
        # Basis: [v, v^2, sin(pi*v), cos(pi*v)]
        v_clip = np.clip(v, -2.0, 2.0)
        b1 = v_clip
        b2 = v_clip**2
        b3 = np.sin(np.pi * v_clip)
        b4 = np.cos(np.pi * v_clip)
        basis = np.stack([b1, b2, b3, b4], axis=-1)  # (dim, 4)

        spline_term = np.sum(self.coeffs * basis, axis=-1)
        out = self.w_lin * v + spline_term
        norm = np.linalg.norm(out)
        return out / norm if norm > 1e-12 else out


class KANNexusAgent:
    def __init__(self, dim: int = 10000, seed: int = 505):
        self.dim = dim
        self.kan_op = KANSplineOperator(dim=dim, seed=seed)
        self.rng = np.random.default_rng(seed)

        self.k_self = self.rng.standard_normal(dim)
        self.k_self /= np.linalg.norm(self.k_self)

        self.state = self.rng.standard_normal(dim)
        self.state /= np.linalg.norm(self.state)

    def strange_loop_iteration(self, external_obs: np.ndarray, num_cycles: int = 5) -> list[float]:
        stabilities = []
        curr = self.state.copy()

        for _ in range(num_cycles):
            # 1. KAN Spline Transformation
            v_kan = self.kan_op.forward(curr)

            # 2. Holographic Binding with Self-Key
            f_k = np.fft.rfft(self.k_self)
            f_kan = np.fft.rfft(v_kan)
            bound = np.fft.irfft(f_k * f_kan, n=self.dim)
            bound /= np.linalg.norm(bound)

            # 3. Superposition with sensory observation and prior state
            next_state = 0.5 * curr + 0.3 * bound + 0.2 * external_obs
            next_state /= np.linalg.norm(next_state)

            sim = float(np.dot(curr, next_state))
            stabilities.append(sim)
            curr = next_state

        self.state = curr
        return stabilities


def benchmark_kan_nexus():
    print("=" * 60)
    print("BENCHMARKING COMBO 3: KAN-NEXUS (KOLMOGOROV-ARNOLD + JARVIS)")
    print("=" * 60)

    agent = KANNexusAgent(dim=10000, seed=505)
    rng = np.random.default_rng(606)

    t0 = time.time()
    n_trials = 20
    final_stabs = []

    for trial in range(n_trials):
        obs = rng.standard_normal(10000)
        obs /= np.linalg.norm(obs)
        stabs = agent.strange_loop_iteration(obs, num_cycles=6)
        final_stabs.append(stabs[-1])
        if (trial + 1) % 5 == 0:
            print(f"   Trial {trial+1:02d} | KAN Strange-Loop Stability: {stabs[-1]:.6f}")

    elapsed = time.time() - t0
    avg_stab = float(np.mean(final_stabs))

    print(f"\nSummary Combo 3 (KAN-Nexus):")
    print(f"   Mean Invariant Stability: {avg_stab:.6f}")
    print(f"   Throughput: {n_trials/elapsed:.2f} KAN-Nexus trials/sec at D=10,000")

    data = {
        "combo": "Combo_03_KAN_Nexus",
        "mean_invariant_stability": avg_stab,
        "throughput_trials_sec": float(n_trials/elapsed)
    }
    with open("benchmark_results/combo_03_kan_nexus.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_kan_nexus()
