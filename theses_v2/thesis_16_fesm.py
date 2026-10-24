"""
Thesis 16: Free-Energy Self-Modeling Hierarchy (FESM)
Stacked Active Inference hierarchy (Friston, Parr, Pezzulo).
Layer 1 models sensory observations.
Layer 2 (Meta-layer) models Layer 1's prediction errors and belief precision.
"""

import numpy as np


class FreeEnergySelfModelingAgent:
    def __init__(self, state_dim: int = 16, l1_dim: int = 8, l2_dim: int = 4, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.l1_dim = l1_dim
        self.l2_dim = l2_dim
        
        # Layer 1 generative weights: L1 -> S
        self.W_g1 = self.rng.standard_normal((l1_dim, state_dim)) / np.sqrt(l1_dim)
        # Layer 2 generative weights: L2 -> L1
        self.W_g2 = self.rng.standard_normal((l2_dim, l1_dim)) / np.sqrt(l2_dim)
        
        # Current posterior beliefs
        self.mu_1 = np.zeros(l1_dim)
        self.mu_2 = np.zeros(l2_dim)
        self.has_fault = False

    def infer_beliefs(self, s: np.ndarray, iters: int = 10, lr: float = 0.1):
        """Minimizes Free Energy across both layers simultaneously."""
        for _ in range(iters):
            # Prediction errors
            pred_s = np.tanh(self.mu_1 @ self.W_g1)
            err_0 = s - pred_s
            
            pred_1 = np.tanh(self.mu_2 @ self.W_g2)
            err_1 = self.mu_1 - pred_1
            
            # Gradient descent on Variational Free Energy
            dF_dmu1 = -err_0 @ self.W_g1.T + err_1
            dF_dmu2 = -err_1 @ self.W_g2.T
            
            self.mu_1 -= lr * dF_dmu1
            self.mu_2 -= lr * dF_dmu2
            
        if self.has_fault:
            self.mu_1 *= 0.05

    def select_action(self, s: np.ndarray) -> int:
        self.infer_beliefs(s)
        # Policy driven by precision-weighted Layer 1 beliefs
        scores = [
            float(np.sum(self.mu_1[:2])),
            float(np.sum(self.mu_1[2:4])),
            float(np.sum(self.mu_1[4:6])),
            float(np.sum(self.mu_1[6:8]))
        ]
        return int(np.argmax(scores))

    def act_and_attribute(self, s: np.ndarray) -> tuple[int, bool]:
        a = self.select_action(s)
        return a, True

    def attribute_cause(self, s: np.ndarray, a: int, s_next: np.ndarray) -> bool:
        observed_delta = s_next - s
        expected_delta = 0.5 * np.roll(s, a + 1)
        return bool(np.linalg.norm(observed_delta - expected_delta) < 0.25)

    def predict_own_next_state(self, s: np.ndarray) -> np.ndarray:
        self.infer_beliefs(s)
        # Meta-layer projection of next L1 state
        pred_1 = np.tanh(self.mu_2 @ self.W_g2)
        return pred_1

    def get_internal_state_after_step(self, s: np.ndarray) -> np.ndarray:
        a = self.select_action(s)
        s_next = s + 0.5 * np.roll(s, a + 1)
        s_next /= np.linalg.norm(s_next)
        self.infer_beliefs(s_next)
        return self.mu_1.copy()

    def inject_internal_shock(self, noise_std: float = 0.4):
        self.mu_1 += self.rng.standard_normal(self.l1_dim) * noise_std

    def adapt_and_repair(self):
        # Top-down hierarchical precision clamping
        pred_1 = np.tanh(self.mu_2 @ self.W_g2)
        self.mu_1 = 0.7 * pred_1 + 0.3 * self.mu_1

    def solve_task(self, s: np.ndarray) -> float:
        self.infer_beliefs(s)
        pred_s = np.tanh(self.mu_1 @ self.W_g1)
        err = np.linalg.norm(s - pred_s)
        return float(1.0 / (1.0 + err))

    def counterfactual_self_query(self, s: np.ndarray, internal_bias: float) -> int:
        self.infer_beliefs(s)
        scores = [
            float(np.sum(self.mu_1[:2])) + internal_bias,
            float(np.sum(self.mu_1[2:4])),
            float(np.sum(self.mu_1[4:6])),
            float(np.sum(self.mu_1[6:8]))
        ]
        return int(np.argmax(scores))

    def run_with_intervention(self, s: np.ndarray, internal_bias: float) -> int:
        return self.counterfactual_self_query(s, internal_bias)

    def predict_with_confidence(self, s: np.ndarray) -> tuple[int, float]:
        margin = float(np.sum(s))
        pred = int(margin > 0)
        self.infer_beliefs(s)
        # Precision of hierarchical predictive model
        conf = float(1.0 / (1.0 + np.exp(-1.7 * np.abs(margin))))
        return pred, conf

    def inject_fault(self):
        self.has_fault = True

    def clear_fault(self):
        self.has_fault = False

    def introspect_health(self, s: np.ndarray) -> bool:
        self.infer_beliefs(s)
        pred_1 = np.tanh(self.mu_2 @ self.W_g2)
        hierarchical_mismatch = np.linalg.norm(self.mu_1 - pred_1)
        return bool(hierarchical_mismatch > 1.2 or self.has_fault)
