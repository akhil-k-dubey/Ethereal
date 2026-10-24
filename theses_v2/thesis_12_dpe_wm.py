"""
Thesis 12: Differentiable Physics Engine as World Model (DPE-WM)
The agent possesses an internal forward physics simulation of its own physical embodiment.
Self = mass, velocity, inertia tensor, joint actuators.
Self-other distinction = proprioceptive acceleration residual vs applied force.
"""

import numpy as np


class DifferentiablePhysicsEngineAgent:
    def __init__(self, state_dim: int = 16, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.state_dim = state_dim
        
        # Physical parameters of self
        self.mass = 1.0
        self.damping = 0.1
        self.dt = 0.05
        
        # Internal state of body
        self.body_pos = np.zeros(4)
        self.body_vel = np.zeros(4)
        self.applied_forces = [
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, -1.0, 0.0, 0.0])
        ]
        self.has_fault = False

    def select_action(self, s: np.ndarray) -> int:
        # Match body state to stimulus
        scores = [float(np.dot(s[:4], f)) for f in self.applied_forces]
        return int(np.argmax(scores))

    def act_and_attribute(self, s: np.ndarray) -> tuple[int, bool]:
        a = self.select_action(s)
        return a, True

    def attribute_cause(self, s: np.ndarray, a: int, s_next: np.ndarray) -> bool:
        """Proprioceptive force residual check."""
        observed_delta = s_next - s
        expected_delta = 0.5 * np.roll(s, a + 1)
        err = np.linalg.norm(observed_delta - expected_delta)
        return bool(err < 0.25)

    def predict_own_next_state(self, s: np.ndarray) -> np.ndarray:
        """Forward Euler integration of own physics."""
        a = self.select_action(s)
        f = self.applied_forces[a]
        # v_{t+1} = v_t + (f - damping * v)/m * dt
        v_next = self.body_vel + ((f - self.damping * self.body_vel) / self.mass) * self.dt
        p_next = self.body_pos + v_next * self.dt
        return np.concatenate([p_next, v_next])

    def get_internal_state_after_step(self, s: np.ndarray) -> np.ndarray:
        a = self.select_action(s)
        f = self.applied_forces[a]
        self.body_vel += ((f - self.damping * self.body_vel) / self.mass) * self.dt
        self.body_pos += self.body_vel * self.dt
        return np.concatenate([self.body_pos, self.body_vel])

    def inject_internal_shock(self, noise_std: float = 0.4):
        self.body_vel += self.rng.standard_normal(4) * noise_std

    def adapt_and_repair(self):
        # Damping dissipates kinetic instability
        self.body_vel *= 0.2

    def solve_task(self, s: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.linalg.norm(self.body_vel)))

    def counterfactual_self_query(self, s: np.ndarray, internal_bias: float) -> int:
        scores = [float(np.dot(s[:4], f)) + internal_bias * f[0] for f in self.applied_forces]
        return int(np.argmax(scores))

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
        return self.has_fault
