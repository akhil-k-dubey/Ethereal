"""
Thesis 9: KAN + Holographic Self-Embed (KHSE)
Architecture: Kolmogorov-Arnold Networks with Learnable B-Spline Edges + HRR Holographic Self-Embedding (Liu 2024/Plate)

KAN Layer: Replaces scalar weights with learnable cubic spline basis functions on edges.
Coupled with Holographic Binding for self-referential attractor stability.
"""

import time
import json
import numpy as np


class KANLayer:
    """1D Univariate Spline Activation Function per edge."""
    def __init__(self, in_features: int, out_features: int, n_grid: int = 5, seed: int = 42):
        self.in_features = in_features
        self.out_features = out_features
        self.n_grid = n_grid
        self.rng = np.random.default_rng(seed)

        # Base linear weights
        self.w_base = self.rng.standard_normal((out_features, in_features)) * 0.1
        # Spline coefficients on Chebyshev/polynomial basis
        self.c_spline = self.rng.standard_normal((out_features, in_features, n_grid)) * 0.05

    def evaluate_basis(self, x: np.ndarray) -> np.ndarray:
        """Evaluates basis functions [x, x^2, x^3, sin(pi*x), cos(pi*x)]."""
        # x shape: (in_features,)
        return np.stack([
            x,
            x**2,
            x**3,
            np.sin(np.pi * x),
            np.cos(np.pi * x)
        ], axis=-1)  # (in_features, n_grid)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Base residual
        base = self.w_base @ np.tanh(x)

        # Spline transformation
        basis = self.evaluate_basis(np.clip(x, -2.0, 2.0))  # (in_features, n_grid)
        # Sum over grid and in_features
        spline_out = np.einsum('oig,ig->o', self.c_spline, basis)
        return base + spline_out


class KANHolographicSelfEmbed:
    def __init__(self, dim: int = 32, seed: int = 42):
        self.dim = dim
        self.kan1 = KANLayer(dim, dim, seed=seed)
        self.kan2 = KANLayer(dim, dim, seed=seed + 1)
        # Holographic self-key
        rng = np.random.default_rng(seed + 99)
        self.k_self = rng.standard_normal(dim)
        self.k_self /= np.linalg.norm(self.k_self)

    def forward_strange_loop(self, x: np.ndarray, num_iterations: int = 5) -> tuple[np.ndarray, list[float]]:
        curr = x / np.linalg.norm(x)
        stabilities = []

        for _ in range(num_iterations):
            # 1. KAN Spline Transformation
            h = self.kan1.forward(curr)
            out_kan = self.kan2.forward(h)
            out_kan /= (np.linalg.norm(out_kan) + 1e-9)

            # 2. Holographic Binding (Circular Convolution via FFT)
            f_key = np.fft.rfft(self.k_self)
            f_out = np.fft.rfft(out_kan)
            bound = np.fft.irfft(f_key * f_out, n=self.dim)
            bound /= (np.linalg.norm(bound) + 1e-9)

            # 3. Superposition with current state
            next_state = 0.6 * curr + 0.4 * bound
            next_state /= np.linalg.norm(next_state)

            # Track cosine similarity to measure attractor stability
            sim = float(np.dot(curr, next_state))
            stabilities.append(sim)
            curr = next_state

        return curr, stabilities


def benchmark_khse():
    print("=" * 60)
    print("BENCHMARKING THESIS 9: KAN + HOLOGRAPHIC SELF-EMBED")
    print("=" * 60)

    khse = KANHolographicSelfEmbed(dim=32, seed=42)
    rng = np.random.default_rng(101)

    drift_scores = []
    t0 = time.time()
    n_trials = 25

    for trial in range(n_trials):
        v_init = rng.standard_normal(32)
        v_final, stabs = khse.forward_strange_loop(v_init, num_iterations=8)
        # Final step stability: similarity of consecutive iterations
        drift_scores.append(stabs[-1])
        if (trial + 1) % 5 == 0:
            print(f"   Trial {trial+1:02d} | Consecutive Convergence Sim: {stabs[-1]:.6f}")

    elapsed = time.time() - t0
    avg_stability = float(np.mean(drift_scores))

    print(f"\nSummary Thesis 9 (KHSE):")
    print(f"   Mean Fixed-Point Stability: {avg_stability:.6f} (1.0 = perfect smooth invariant)")
    print(f"   Throughput: {n_trials/elapsed:.2f} strange-loop iterations/sec")

    data = {
        "thesis": "Thesis_09_KHSE",
        "fixed_point_stability": float(avg_stability),
        "throughput_loops_sec": float(n_trials/elapsed)
    }
    with open("benchmark_results/thesis_09_khse.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_khse()
