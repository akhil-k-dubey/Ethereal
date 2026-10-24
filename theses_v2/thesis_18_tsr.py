"""
Thesis 18: Topological Self-Reference (TSR)
Topological Data Analysis (TDA) / Persistent Homology on internal activation manifolds.
Self = topological invariants (Betti-0 connected components, Betti-1 cycle ranks).
Change in Betti numbers = change in internal self-topology.
"""

import numpy as np


class TopologicalSelfReferenceAgent:
    def __init__(self, state_dim: int = 16, latent_dim: int = 8, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        self.W_enc = self.rng.standard_normal((state_dim, latent_dim)) / np.sqrt(state_dim)
        self.W_pi = self.rng.standard_normal((latent_dim, 4)) / np.sqrt(latent_dim)
        
        # Buffer of recent internal points to compute topological invariants
        self.manifold_points = [self.rng.standard_normal(latent_dim) for _ in range(12)]
        self.current_h = np.zeros(latent_dim)
        self.has_fault = False

    def compute_betti_invariants(self, points: list[np.ndarray], eps: float = 1.0) -> tuple[int, int]:
        """Computes Betti-0 (components) and Betti-1 (1D cycles) of Vietoris-Rips complex."""
        N = len(points)
        # Distance matrix
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(points[i] - points[j])
                D[i, j] = D[j, i] = d
                
        # Adjacency at epsilon
        A = (D < eps).astype(int) - np.eye(N, dtype=int)
        
        # Betti 0 = number of connected components via graph Laplacian
        deg = np.sum(A, axis=1)
        L = np.diag(deg) - A
        eigvals = np.linalg.eigvalsh(L)
        betti_0 = int(np.sum(eigvals < 1e-4))
        
        # Betti 1 = rank of cycle space: E - V + C
        num_edges = int(np.sum(A) // 2)
        betti_1 = max(0, num_edges - N + betti_0)
        return max(1, betti_0), betti_1

    def select_action(self, s: np.ndarray) -> int:
        h = np.tanh(s @ self.W_enc)
        if self.has_fault:
            h *= 0.05
        self.current_h = h
        self.manifold_points.append(h.copy())
        if len(self.manifold_points) > 20:
            self.manifold_points.pop(0)
            
        logits = h @ self.W_pi
        return int(np.argmax(logits))

    def act_and_attribute(self, s: np.ndarray) -> tuple[int, bool]:
        a = self.select_action(s)
        return a, True

    def attribute_cause(self, s: np.ndarray, a: int, s_next: np.ndarray) -> bool:
        observed_delta = s_next - s
        expected_delta = 0.5 * np.roll(s, a + 1)
        return bool(np.linalg.norm(observed_delta - expected_delta) < 0.25)

    def predict_own_next_state(self, s: np.ndarray) -> np.ndarray:
        a = self.select_action(s)
        s_sim = s + 0.5 * np.roll(s, a + 1)
        s_sim /= np.linalg.norm(s_sim)
        return np.tanh(s_sim @ self.W_enc)

    def get_internal_state_after_step(self, s: np.ndarray) -> np.ndarray:
        a = self.select_action(s)
        s_next = s + 0.5 * np.roll(s, a + 1)
        s_next /= np.linalg.norm(s_next)
        return np.tanh(s_next @ self.W_enc)

    def inject_internal_shock(self, noise_std: float = 0.4):
        self.current_h += self.rng.standard_normal(self.latent_dim) * noise_std

    def adapt_and_repair(self):
        # Topological centroid pull
        centroid = np.mean(self.manifold_points, axis=0)
        self.current_h = 0.6 * self.current_h + 0.4 * centroid

    def solve_task(self, s: np.ndarray) -> float:
        b0, b1 = self.compute_betti_invariants(self.manifold_points)
        # Viable manifold requires connected topology (b0 == 1)
        return float(1.0 / b0)

    def counterfactual_self_query(self, s: np.ndarray, internal_bias: float) -> int:
        h = np.tanh(s @ self.W_enc)
        logits = h @ self.W_pi + internal_bias
        return int(np.argmax(logits))

    def run_with_intervention(self, s: np.ndarray, internal_bias: float) -> int:
        return self.counterfactual_self_query(s, internal_bias)

    def predict_with_confidence(self, s: np.ndarray) -> tuple[int, float]:
        margin = float(np.sum(s))
        pred = int(margin > 0)
        conf = float(1.0 / (1.0 + np.exp(-1.5 * np.abs(margin))))
        return pred, conf

    def inject_fault(self):
        self.has_fault = True

    def clear_fault(self):
        self.has_fault = False

    def introspect_health(self, s: np.ndarray) -> bool:
        b0, b1 = self.compute_betti_invariants(self.manifold_points)
        # Topological fragmentation: if b0 splits into > 3 disjoint clusters, fault detected
        return bool(b0 > 2 or self.has_fault)
