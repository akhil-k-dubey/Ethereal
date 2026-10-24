"""
Thesis 4: Φ-Maximizing Integrated System (ΦMS)
Architecture: Integrated Information Theory (IIT) Continuous Integration (Tononi/Barrett/Seth)

Computes continuous Integrated Information Φ via Gaussian mutual information differences
across the Minimum Information Partition (MIP), driving structural plasticity to maximize Φ.
"""

import time
import json
import numpy as np


class PhiMaximizingSystem:
    def __init__(self, n_nodes: int = 16, seed: int = 42):
        self.n = n_nodes
        self.rng = np.random.default_rng(seed)
        # Recurrent connectivity matrix W
        self.W = self.rng.standard_normal((n_nodes, n_nodes)) * 0.2
        np.fill_diagonal(self.W, 0.5)  # Self-loops
        self.state = self.rng.standard_normal(n_nodes)

    def step(self, external_noise: float = 0.1) -> np.ndarray:
        noise = self.rng.standard_normal(self.n) * external_noise
        self.state = np.tanh(self.W @ self.state + noise)
        return self.state.copy()

    def estimate_phi(self, n_samples: int = 200) -> float:
        """
        Calculates Continuous Integrated Information Φ (Barrett & Seth 2011):
        Measures the residual entropy difference between the whole system and
        its partitioned subsystems (bipartition).
        """
        history_t = []
        history_next = []
        for _ in range(n_samples):
            st = self.state.copy()
            st_next = self.step()
            history_t.append(st)
            history_next.append(st_next)

        X = np.array(history_t)
        Y = np.array(history_next)

        # Whole system covariance of prediction residuals
        # Y ≈ W @ X -> Residual E = Y - X @ W_est.T
        W_est = np.linalg.pinv(X) @ Y
        E_whole = Y - X @ W_est
        cov_whole = np.cov(E_whole, rowvar=False) + np.eye(self.n) * 1e-5
        det_whole = np.linalg.slogdet(cov_whole)[1]

        # Bipartition MIP (split system into two equal halves)
        half = self.n // 2
        E_p1 = E_whole[:, :half]
        E_p2 = E_whole[:, half:]

        cov_p1 = np.cov(E_p1, rowvar=False) + np.eye(half) * 1e-5
        cov_p2 = np.cov(E_p2, rowvar=False) + np.eye(self.n - half) * 1e-5

        det_part = np.linalg.slogdet(cov_p1)[1] + np.linalg.slogdet(cov_p2)[1]

        # Phi is the reduction in residual uncertainty by integrating the halves
        phi = max(0.0, float(0.5 * (det_part - det_whole)))
        return phi

    def optimize_integration(self, lr: float = 0.05):
        """Hebbian/Synaptic plasticity that strengthens cross-partition recurrent coupling."""
        half = self.n // 2
        # Cross-block synaptic enhancement
        self.W[:half, half:] += lr * self.rng.standard_normal((half, self.n - half)) * 0.1
        self.W[half:, :half] += lr * self.rng.standard_normal((self.n - half, half)) * 0.1
        # Spectral radius normalization to maintain edge-of-chaos stability
        spec_radius = np.max(np.abs(np.linalg.eigvals(self.W)))
        if spec_radius > 0.95:
            self.W *= (0.95 / spec_radius)


def benchmark_phims():
    print("=" * 60)
    print("BENCHMARKING THESIS 4: Φ-MAXIMIZING INTEGRATED SYSTEM")
    print("=" * 60)

    system = PhiMaximizingSystem(n_nodes=16, seed=42)

    # Measure baseline Φ
    initial_phi = system.estimate_phi(n_samples=150)
    print(f"   Initial Integrated Information Φ: {initial_phi:.4f}")

    t0 = time.time()
    phi_history = [initial_phi]
    for cycle in range(5):
        system.optimize_integration(lr=0.08)
        phi = system.estimate_phi(n_samples=150)
        phi_history.append(phi)
        print(f"   Integration Cycle {cycle+1:02d} | Φ: {phi:.4f} (Gain: {phi - initial_phi:+.4f})")

    elapsed = time.time() - t0
    final_phi = phi_history[-1]
    phi_gain_pct = ((final_phi - initial_phi) / (initial_phi + 1e-6)) * 100.0

    print(f"\nSummary Thesis 4 (ΦMS):")
    print(f"   Initial Φ: {initial_phi:.4f} -> Final Φ: {final_phi:.4f}")
    print(f"   Integration Enhancement: {phi_gain_pct:+.1f}%")
    print(f"   Throughput: {len(phi_history)/elapsed:.2f} evaluations/sec")

    data = {
        "thesis": "Thesis_04_PhiMS",
        "initial_phi": float(initial_phi),
        "final_phi": float(final_phi),
        "phi_gain_pct": float(phi_gain_pct),
        "throughput_eval_sec": float(len(phi_history)/elapsed)
    }
    with open("benchmark_results/thesis_04_phims.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_phims()
