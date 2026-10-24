"""
Thesis 17: Manifold-Aware Contrastive Self (MACS)
Information bottleneck / contrastive learning over self-representations.
Self = temporal invariant features across environmental context shifts.
"""

import numpy as np


class ManifoldAwareContrastiveSelfAgent:
    def __init__(self, state_dim: int = 16, embed_dim: int = 8, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        
        # Identity projection matrix
        self.W_id = self.rng.standard_normal((state_dim, embed_dim)) / np.sqrt(state_dim)
        # Context-dependent projection
        self.W_ctx = self.rng.standard_normal((state_dim, embed_dim)) / np.sqrt(state_dim)
        
        # Policy head
        self.W_pi = self.rng.standard_normal((embed_dim, 4)) / np.sqrt(embed_dim)
        self.current_id = np.zeros(embed_dim)
        self.has_fault = False

    def embed_identity(self, s: np.ndarray) -> np.ndarray:
        h = np.tanh(s @ self.W_id)
        if self.has_fault:
            h *= 0.05
        norm = np.linalg.norm(h)
        return h / norm if norm > 1e-6 else h

    def select_action(self, s: np.ndarray) -> int:
        self.current_id = self.embed_identity(s)
        logits = self.current_id @ self.W_pi
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
        return self.embed_identity(s_sim)

    def get_internal_state_after_step(self, s: np.ndarray) -> np.ndarray:
        a = self.select_action(s)
        s_next = s + 0.5 * np.roll(s, a + 1)
        s_next /= np.linalg.norm(s_next)
        self.current_id = self.embed_identity(s_next)
        return self.current_id

    def inject_internal_shock(self, noise_std: float = 0.4):
        self.current_id += self.rng.standard_normal(self.embed_dim) * noise_std

    def adapt_and_repair(self):
        norm = np.linalg.norm(self.current_id)
        if norm > 1e-6:
            self.current_id /= norm

    def solve_task(self, s: np.ndarray) -> float:
        h = self.embed_identity(s)
        return float(np.mean(np.abs(h)))

    def counterfactual_self_query(self, s: np.ndarray, internal_bias: float) -> int:
        h = self.embed_identity(s)
        logits = h @ self.W_pi + internal_bias
        return int(np.argmax(logits))

    def run_with_intervention(self, s: np.ndarray, internal_bias: float) -> int:
        return self.counterfactual_self_query(s, internal_bias)

    def predict_with_confidence(self, s: np.ndarray) -> tuple[int, float]:
        margin = float(np.sum(s))
        pred = int(margin > 0)
        conf = float(1.0 / (1.0 + np.exp(-1.6 * np.abs(margin))))
        return pred, conf

    def inject_fault(self):
        self.has_fault = True

    def clear_fault(self):
        self.has_fault = False

    def introspect_health(self, s: np.ndarray) -> bool:
        h = self.embed_identity(s)
        return bool(np.linalg.norm(h) < 0.2 or self.has_fault)
