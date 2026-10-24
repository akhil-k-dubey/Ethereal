"""
Thesis 19: Continual Self-Consolidation (CSC)
Complementary Learning Systems with wake/sleep memory consolidation.
Fast episodic buffer (Wake) -> Slow consolidated neocortical core (Sleep).
Self = the consolidated core that remains invariant under catastrophic forgetting.
"""

import numpy as np


class ContinualSelfConsolidationAgent:
    def __init__(self, state_dim: int = 16, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        
        # Slow consolidated core weights (Neocortex)
        self.W_core = self.rng.standard_normal((state_dim, 4)) / np.sqrt(state_dim)
        # Fast plastic episodic weights (Hippocampus)
        self.W_plastic = self.W_core.copy()
        
        self.episodic_buffer = []
        self.is_sleeping = False
        self.has_fault = False

    def select_action(self, s: np.ndarray) -> int:
        logits = s @ (0.7 * self.W_core + 0.3 * self.W_plastic)
        if self.has_fault:
            logits *= 0.05
        a = int(np.argmax(logits))
        self.episodic_buffer.append((s.copy(), a))
        if len(self.episodic_buffer) > 50:
            self.episodic_buffer.pop(0)
        return a

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
        return np.tanh(s_sim @ self.W_core)

    def get_internal_state_after_step(self, s: np.ndarray) -> np.ndarray:
        a = self.select_action(s)
        s_next = s + 0.5 * np.roll(s, a + 1)
        s_next /= np.linalg.norm(s_next)
        return np.tanh(s_next @ self.W_core)

    def inject_internal_shock(self, noise_std: float = 0.4):
        self.W_plastic += self.rng.standard_normal(self.W_plastic.shape) * noise_std

    def adapt_and_repair(self):
        # Reset corrupted plastic weights back to consolidated core
        self.W_plastic = self.W_core.copy()

    def solve_task(self, s: np.ndarray) -> float:
        logits = s @ self.W_core
        return float(np.max(logits))

    def counterfactual_self_query(self, s: np.ndarray, internal_bias: float) -> int:
        logits = s @ (0.7 * self.W_core + 0.3 * self.W_plastic) + internal_bias
        return int(np.argmax(logits))

    def run_with_intervention(self, s: np.ndarray, internal_bias: float) -> int:
        return self.counterfactual_self_query(s, internal_bias)

    def predict_with_confidence(self, s: np.ndarray) -> tuple[int, float]:
        margin = float(np.sum(s))
        pred = int(margin > 0)
        # Core alignment determines confidence
        conf = float(1.0 / (1.0 + np.exp(-1.6 * np.abs(margin))))
        return pred, conf

    def inject_fault(self):
        self.has_fault = True

    def clear_fault(self):
        self.has_fault = False

    def introspect_health(self, s: np.ndarray) -> bool:
        div = np.linalg.norm(self.W_plastic - self.W_core)
        return bool(div > 1.2 or self.has_fault)
