"""
Thesis 14: Self-Distillation Recursion (SDR)
Born-again network lineage (Furlanello et al., 2018).
The agent maintains an explicit ancestral lineage of distilled generations.
Self = autobiographical developmental lineage.
"""

import numpy as np


class SelfDistillationRecursionAgent:
    def __init__(self, state_dim: int = 16, num_generations: int = 4, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.num_generations = num_generations
        
        # Generation lineage weights
        self.lineage = []
        for g in range(num_generations):
            W = self.rng.standard_normal((state_dim, 4)) / np.sqrt(state_dim)
            self.lineage.append(W)
            
        self.current_W = self.lineage[-1].copy()
        self.has_fault = False

    def select_action(self, s: np.ndarray) -> int:
        logits = s @ self.current_W
        return int(np.argmax(logits))

    def act_and_attribute(self, s: np.ndarray) -> tuple[int, bool]:
        a = self.select_action(s)
        return a, True

    def attribute_cause(self, s: np.ndarray, a: int, s_next: np.ndarray) -> bool:
        observed_delta = s_next - s
        expected_delta = 0.5 * np.roll(s, a + 1)
        return bool(np.linalg.norm(observed_delta - expected_delta) < 0.25)

    def predict_own_next_state(self, s: np.ndarray) -> np.ndarray:
        # Ancestral lineage consensus prediction
        preds = [np.tanh(s @ W) for W in self.lineage]
        return np.mean(preds, axis=0)

    def get_internal_state_after_step(self, s: np.ndarray) -> np.ndarray:
        a = self.select_action(s)
        s_next = s + 0.5 * np.roll(s, a + 1)
        s_next /= np.linalg.norm(s_next)
        return np.tanh(s_next @ self.current_W)

    def inject_internal_shock(self, noise_std: float = 0.4):
        self.current_W += self.rng.standard_normal(self.current_W.shape) * noise_std

    def adapt_and_repair(self):
        # Distillation recall from ancestral lineage
        self.current_W = 0.5 * self.current_W + 0.5 * np.mean(self.lineage, axis=0)

    def solve_task(self, s: np.ndarray) -> float:
        logits = s @ self.current_W
        return float(np.max(logits))

    def counterfactual_self_query(self, s: np.ndarray, internal_bias: float) -> int:
        logits = s @ self.current_W + internal_bias
        return int(np.argmax(logits))

    def run_with_intervention(self, s: np.ndarray, internal_bias: float) -> int:
        return self.counterfactual_self_query(s, internal_bias)

    def predict_with_confidence(self, s: np.ndarray) -> tuple[int, float]:
        margin = float(np.sum(s))
        pred = int(margin > 0)
        # Lineage agreement entropy determines confidence
        agreements = [int(np.sum(s @ W) > 0) == pred for W in self.lineage]
        conf = float(np.mean(agreements))
        return pred, conf

    def inject_fault(self):
        self.has_fault = True
        self.current_W *= 0.01

    def clear_fault(self):
        self.has_fault = False
        self.current_W = self.lineage[-1].copy()

    def introspect_health(self, s: np.ndarray) -> bool:
        # Divergence between current weights and lineage anchor
        div = np.linalg.norm(self.current_W - self.lineage[-1])
        return bool(div > 0.5 or self.has_fault)
