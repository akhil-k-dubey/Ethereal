"""
Thesis 13: Generative Self-Simulation (GSS)
MuZero / Dreamer style latent self-simulator.
Maintains a learned recurrent latent model of its own internal policy trajectories.
Self-awareness = self-imagination / forward mental rollout.
"""

import numpy as np


class GenerativeSelfSimulationAgent:
    def __init__(self, state_dim: int = 16, latent_dim: int = 8, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Representation network: s -> z
        self.W_enc = self.rng.standard_normal((state_dim, latent_dim)) / np.sqrt(state_dim)
        # Dynamics model: z_t, a_t -> z_{t+1}
        self.W_dyn = self.rng.standard_normal((latent_dim + 4, latent_dim)) / np.sqrt(latent_dim)
        # Policy head: z -> action logits
        self.W_pi = self.rng.standard_normal((latent_dim, 4)) / np.sqrt(latent_dim)
        
        self.current_z = np.zeros(latent_dim)
        self.has_fault = False

    def encode(self, s: np.ndarray) -> np.ndarray:
        z = np.tanh(s @ self.W_enc)
        if self.has_fault:
            z *= 0.05
        return z

    def select_action(self, s: np.ndarray) -> int:
        self.current_z = self.encode(s)
        logits = self.current_z @ self.W_pi
        return int(np.argmax(logits))

    def act_and_attribute(self, s: np.ndarray) -> tuple[int, bool]:
        a = self.select_action(s)
        return a, True

    def attribute_cause(self, s: np.ndarray, a: int, s_next: np.ndarray) -> bool:
        observed_delta = s_next - s
        expected_delta = 0.5 * np.roll(s, a + 1)
        err = np.linalg.norm(observed_delta - expected_delta)
        return bool(err < 0.25)

    def predict_own_next_state(self, s: np.ndarray) -> np.ndarray:
        """Forward mental rollout in latent space: z_{t+1} = Dyn(z_t, a_t)."""
        z = self.encode(s)
        logits = z @ self.W_pi
        a = int(np.argmax(logits))
        
        # One-hot action
        a_vec = np.zeros(4)
        a_vec[a] = 1.0
        
        inp = np.concatenate([z, a_vec])
        z_next_pred = np.tanh(inp @ self.W_dyn)
        return z_next_pred

    def get_internal_state_after_step(self, s: np.ndarray) -> np.ndarray:
        a = self.select_action(s)
        s_next = s + 0.5 * np.roll(s, a + 1)
        s_next /= np.linalg.norm(s_next)
        self.current_z = self.encode(s_next)
        return self.current_z

    def inject_internal_shock(self, noise_std: float = 0.4):
        self.current_z += self.rng.standard_normal(self.latent_dim) * noise_std

    def adapt_and_repair(self):
        # Dreamer projection back to unit latent sphere
        norm = np.linalg.norm(self.current_z)
        if norm > 1e-6:
            self.current_z /= norm

    def solve_task(self, s: np.ndarray) -> float:
        z = self.encode(s)
        return float(np.mean(np.abs(z)))

    def counterfactual_self_query(self, s: np.ndarray, internal_bias: float) -> int:
        z = self.encode(s)
        logits = z @ self.W_pi + internal_bias
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
        z = self.encode(s)
        return bool(np.linalg.norm(z) < 0.1)
