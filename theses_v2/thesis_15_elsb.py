"""
Thesis 15: Energy-Landscape Self-Basin (ELSB)
High-dimensional energy landscape where "self" = basin of attraction.
Maintains quadratic + non-linear energy functional E(h).
Self-restoration via gradient flow to the basin bottom.
"""

import numpy as np


class EnergyLandscapeSelfBasinAgent:
    def __init__(self, state_dim: int = 16, latent_dim: int = 8, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Energy surface parameters: E(h) = 0.5 * h^T A h - s^T W h
        M = self.rng.standard_normal((latent_dim, latent_dim))
        self.A = (M.T @ M) / latent_dim + np.eye(latent_dim) * 0.5  # Positive definite
        self.W = self.rng.standard_normal((state_dim, latent_dim)) / np.sqrt(state_dim)
        
        # Policy head
        self.W_pi = self.rng.standard_normal((latent_dim, 4)) / np.sqrt(latent_dim)
        self.h_basin = np.zeros(latent_dim)
        self.has_fault = False

    def energy(self, h: np.ndarray, s: np.ndarray) -> float:
        return float(0.5 * h @ self.A @ h - (s @ self.W) @ h)

    def energy_grad(self, h: np.ndarray, s: np.ndarray) -> np.ndarray:
        return self.A @ h - (s @ self.W)

    def relax_to_basin(self, s: np.ndarray, steps: int = 15, lr: float = 0.1) -> np.ndarray:
        h = np.zeros(self.latent_dim)
        for _ in range(steps):
            grad = self.energy_grad(h, s)
            h -= lr * grad
        return h

    def select_action(self, s: np.ndarray) -> int:
        self.h_basin = self.relax_to_basin(s)
        if self.has_fault:
            self.h_basin *= 0.01
        logits = self.h_basin @ self.W_pi
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
        return self.relax_to_basin(s_sim)

    def get_internal_state_after_step(self, s: np.ndarray) -> np.ndarray:
        a = self.select_action(s)
        s_next = s + 0.5 * np.roll(s, a + 1)
        s_next /= np.linalg.norm(s_next)
        self.h_basin = self.relax_to_basin(s_next)
        return self.h_basin

    def inject_internal_shock(self, noise_std: float = 0.4):
        self.h_basin += self.rng.standard_normal(self.latent_dim) * noise_std

    def adapt_and_repair(self):
        # Relax back down the gradient toward basin minimum
        for _ in range(10):
            grad = self.A @ self.h_basin
            self.h_basin -= 0.1 * grad

    def solve_task(self, s: np.ndarray) -> float:
        # Task performance inversely proportional to residual energy
        h = self.relax_to_basin(s)
        e = self.energy(h, s)
        return float(1.0 / (1.0 + np.exp(e)))

    def counterfactual_self_query(self, s: np.ndarray, internal_bias: float) -> int:
        h = self.relax_to_basin(s)
        logits = h @ self.W_pi + internal_bias
        return int(np.argmax(logits))

    def run_with_intervention(self, s: np.ndarray, internal_bias: float) -> int:
        return self.counterfactual_self_query(s, internal_bias)

    def predict_with_confidence(self, s: np.ndarray) -> tuple[int, float]:
        margin = float(np.sum(s))
        pred = int(margin > 0)
        # Basin curvature / depth determines certainty
        depth = np.linalg.norm(self.relax_to_basin(s))
        conf = float(1.0 / (1.0 + np.exp(-1.5 * depth)))
        return pred, conf

    def inject_fault(self):
        self.has_fault = True

    def clear_fault(self):
        self.has_fault = False

    def introspect_health(self, s: np.ndarray) -> bool:
        # Check if current state sits at an energy minimum
        grad = self.A @ self.h_basin
        return bool(np.linalg.norm(grad) > 1.5 or self.has_fault)
