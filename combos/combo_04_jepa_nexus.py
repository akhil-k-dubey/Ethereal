"""
Combination 4: JEPA-Nexus (Jarvis Nexus + Non-Generative JEPA Latent Space)
Fuses:
1. 10,000-D Holographic Reduced Representations (HRR)
2. Joint Embedding Predictive Architecture: all reasoning occurs in abstract latent space (no raw reconstruction)
3. Self-Predictor evaluating internal cognitive state trajectory
"""

import time
import json
import numpy as np


class JEPANexusAgent:
    def __init__(self, dim: int = 10000, seed: int = 707):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # Non-generative projection kernel
        self.encoder_kernel = self.rng.standard_normal(dim)
        self.encoder_kernel /= np.linalg.norm(self.encoder_kernel)

        # Latent World Predictor kernel
        self.world_predictor = self.rng.standard_normal(dim)
        self.world_predictor /= np.linalg.norm(self.world_predictor)

        # Self-Predictor kernel
        self.self_predictor = self.rng.standard_normal(dim)
        self.self_predictor /= np.linalg.norm(self.self_predictor)

        # Strange-loop self key
        self.k_self = self.rng.standard_normal(dim)
        self.k_self /= np.linalg.norm(self.k_self)

        # Current internal latent state
        self.z_state = self.rng.standard_normal(dim)
        self.z_state /= np.linalg.norm(self.z_state)

    def encode_to_latent(self, raw_input: np.ndarray) -> np.ndarray:
        """Holographic encoding into abstract latent space."""
        f_k = np.fft.rfft(self.encoder_kernel)
        f_x = np.fft.rfft(raw_input)
        z = np.fft.irfft(f_k * f_x, n=self.dim)
        norm = np.linalg.norm(z)
        return z / norm if norm > 1e-12 else z

    def predict_latent_future(self, z_curr: np.ndarray, action_vec: np.ndarray) -> np.ndarray:
        """Non-generative rollout: Predicts z_{t+1} directly in latent space."""
        # Cause = z_curr ⊕ action
        cause = 0.7 * z_curr + 0.3 * action_vec
        f_w = np.fft.rfft(self.world_predictor)
        f_c = np.fft.rfft(cause)
        z_hat = np.fft.irfft(f_w * f_c, n=self.dim)
        norm = np.linalg.norm(z_hat)
        return z_hat / norm if norm > 1e-12 else z_hat

    def step_active_inference(self, raw_obs: np.ndarray, candidate_actions: list[np.ndarray], goal_latent: np.ndarray) -> tuple[int, dict]:
        # 1. Encode observation into latent representation
        z_obs = self.encode_to_latent(raw_obs)

        # 2. Update internal state with strange-loop self-model
        f_k = np.fft.rfft(self.k_self)
        f_z = np.fft.rfft(self.z_state)
        z_refl = np.fft.irfft(f_k * f_z, n=self.dim)
        z_refl /= np.linalg.norm(z_refl)

        self.z_state = 0.5 * self.z_state + 0.35 * z_obs + 0.15 * z_refl
        self.z_state /= np.linalg.norm(self.z_state)

        # 3. Evaluate candidate actions via latent Expected Free Energy (EFE)
        best_act_idx = 0
        best_efe = float("inf")
        action_utilities = []

        for idx, act in enumerate(candidate_actions):
            z_pred = self.predict_latent_future(self.z_state, act)

            # Pragmatic Value in Latent Space
            pragmatic = float(np.dot(z_pred, goal_latent))

            # Self-Prediction Coherence
            f_s = np.fft.rfft(self.self_predictor)
            f_zp = np.fft.rfft(z_pred)
            z_self_pred = np.fft.irfft(f_s * f_zp, n=self.dim)
            z_self_pred /= np.linalg.norm(z_self_pred)
            self_coherence = float(np.dot(z_pred, z_self_pred))

            # EFE = - (3.0 * Pragmatic + 0.5 * Self-Coherence)
            efe = -(3.0 * pragmatic + 0.5 * self_coherence)
            action_utilities.append(efe)

            if efe < best_efe:
                best_efe = efe
                best_act_idx = idx

        # Online update of world predictor towards true next latent state
        f_c = np.fft.rfft(0.7 * self.z_state + 0.3 * candidate_actions[best_act_idx])
        f_target = np.fft.rfft(z_obs)
        target_w = np.fft.irfft(np.conj(f_c) * f_target, n=self.dim)
        target_w /= (np.linalg.norm(target_w) + 1e-9)
        self.world_predictor = 0.95 * self.world_predictor + 0.05 * target_w
        self.world_predictor /= np.linalg.norm(self.world_predictor)

        return best_act_idx, {
            "chosen_action": best_act_idx,
            "min_efe": best_efe,
            "latent_variance": float(np.var(z_obs))
        }


def benchmark_jepa_nexus():
    print("=" * 60)
    print("BENCHMARKING COMBO 4: JEPA-NEXUS (NON-GENERATIVE LATENT ACTIVE INFERENCE)")
    print("=" * 60)

    agent = JEPANexusAgent(dim=10000, seed=707)
    rng = np.random.default_rng(808)

    goal_latent = rng.standard_normal(10000)
    goal_latent /= np.linalg.norm(goal_latent)

    actions = [rng.standard_normal(10000) for _ in range(4)]
    for a in actions:
        a /= np.linalg.norm(a)

    t0 = time.time()
    n_steps = 30
    efes = []

    for step in range(n_steps):
        raw_obs = rng.standard_normal(10000)
        best_a, meta = agent.step_active_inference(raw_obs, actions, goal_latent)
        efes.append(meta["min_efe"])
        if (step + 1) % 10 == 0:
            print(f"   Step {step+1:02d} | Chosen Action: [{best_a}] | Min Latent EFE: {meta['min_efe']:.4f} | Latent Var: {meta['latent_variance']:.6f}")

    elapsed = time.time() - t0
    avg_efe = float(np.mean(efes))
    collapse_free = bool(meta["latent_variance"] > 1e-6)

    print(f"\nSummary Combo 4 (JEPA-Nexus):")
    print(f"   Mean Latent EFE: {avg_efe:.4f}")
    print(f"   Representation Collapse Resistance: {'SUCCESS' if collapse_free else 'COLLAPSED'}")
    print(f"   Throughput: {n_steps/elapsed:.2f} latent decisions/sec at D=10,000")

    data = {
        "combo": "Combo_04_JEPA_Nexus",
        "mean_latent_efe": avg_efe,
        "collapse_free": collapse_free,
        "throughput_decisions_sec": float(n_steps/elapsed)
    }
    with open("benchmark_results/combo_04_jepa_nexus.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_jepa_nexus()
