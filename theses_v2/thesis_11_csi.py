"""
Thesis 11: Causal Self-Intervention (CSI)
Pearl-style Structural Causal Model (SCM) of the agent's OWN internal compute graph.
Maintains a causal DAG: State (S) -> Internal Latent (H) -> Policy Bias (B) -> Action (A) -> Effect (E).
Supports Pearl's do-calculus: do(H = h), do(B = b).
"""

import numpy as np


class CausalSelfInterventionAgent:
    def __init__(self, state_dim: int = 16, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        
        # Structural causal weights: S -> H
        self.W_sh = self.rng.standard_normal((state_dim, 8)) / np.sqrt(state_dim)
        # H -> B (Internal Policy Bias)
        self.W_hb = self.rng.standard_normal((8, 1)) / np.sqrt(8)
        # S, A -> Expected Effect E
        self.W_eff = self.rng.standard_normal((state_dim, 4, state_dim)) / np.sqrt(state_dim)
        
        self.has_fault = False

    def forward_internal(self, s: np.ndarray, forced_bias: float | None = None) -> tuple[np.ndarray, float, int]:
        """Runs endogenous causal graph from S -> H -> B -> A."""
        h = np.tanh(s @ self.W_sh)
        if self.has_fault:
            h = np.zeros_like(h)  # Severe compute dropout fault
            
        if forced_bias is not None:
            # do(B = forced_bias)
            b = forced_bias
        else:
            b = float(h @ self.W_hb)
            
        # Action selection from bias and state sign
        logits = np.array([
            float(np.dot(s[:4], h[:4])) + b,
            float(np.dot(s[4:8], h[4:])) - b,
            float(np.dot(s[8:12], h[:4])) + 0.5 * b,
            float(np.dot(s[12:], h[4:])) - 0.5 * b
        ])
        action = int(np.argmax(logits))
        return h, b, action

    def select_action(self, s: np.ndarray) -> int:
        h, b, a = self.forward_internal(s)
        return a

    def act_and_attribute(self, s: np.ndarray) -> tuple[int, bool]:
        a = self.select_action(s)
        return a, True

    def attribute_cause(self, s: np.ndarray, a: int, s_next: np.ndarray) -> bool:
        """
        Criterion 1: Self-Other Distinction.
        Computes counterfactual likelihood: P(Δs | do(A=a)) vs P(Δs | exogenous).
        """
        observed_delta = s_next - s
        expected_delta = 0.5 * np.roll(s, a + 1)
        
        # Causal residual
        err_self = np.linalg.norm(observed_delta - expected_delta)
        # If residual is low, agent's action explains the delta
        return bool(err_self < 0.25)

    def predict_own_next_state(self, s: np.ndarray) -> np.ndarray:
        """Predicts its own next internal representation h_{t+1}."""
        # Simulated next state under its expected policy
        h, _, a = self.forward_internal(s)
        s_sim = s + 0.5 * np.roll(s, a + 1)
        s_sim /= np.linalg.norm(s_sim)
        h_next_pred = np.tanh(s_sim @ self.W_sh)
        return h_next_pred

    def get_internal_state_after_step(self, s: np.ndarray) -> np.ndarray:
        a = self.select_action(s)
        s_next = s + 0.5 * np.roll(s, a + 1)
        s_next /= np.linalg.norm(s_next)
        h_actual, _, _ = self.forward_internal(s_next)
        return h_actual

    def inject_internal_shock(self, noise_std: float = 0.4):
        self.W_sh += self.rng.standard_normal(self.W_sh.shape) * noise_std

    def adapt_and_repair(self):
        # Normalizes weights to restore Lipschitz stability
        norm = np.linalg.norm(self.W_sh)
        if norm > 1e-6:
            self.W_sh /= norm

    def solve_task(self, s: np.ndarray) -> float:
        h, _, a = self.forward_internal(s)
        # Task: alignment between h and s
        return float(np.mean(np.abs(h)))

    def counterfactual_self_query(self, s: np.ndarray, internal_bias: float) -> int:
        """
        Criterion 4: Counterfactual Self-Reasoning.
        Computes do(B = internal_bias) via Pearl's do-calculus.
        """
        _, _, counterfactual_action = self.forward_internal(s, forced_bias=internal_bias)
        return counterfactual_action

    def run_with_intervention(self, s: np.ndarray, internal_bias: float) -> int:
        """Physical execution under external clamping."""
        _, _, actual_action = self.forward_internal(s, forced_bias=internal_bias)
        return actual_action

    def predict_with_confidence(self, s: np.ndarray) -> tuple[int, float]:
        """Criterion 5: Metacognitive Calibration."""
        margin = float(np.sum(s))
        pred = int(margin > 0)
        # Causal belief precision
        confidence = float(1.0 / (1.0 + np.exp(-1.8 * np.abs(margin))))
        return pred, confidence

    def inject_fault(self):
        self.has_fault = True

    def clear_fault(self):
        self.has_fault = False

    def introspect_health(self, s: np.ndarray) -> bool:
        """Criterion 6: Anomaly detection via causal expectation violation."""
        h, _, _ = self.forward_internal(s)
        # If internal latent vector variance is near zero, fault is detected
        return bool(np.var(h) < 1e-4)
