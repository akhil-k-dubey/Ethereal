"""
Combination 2: Liquid-Nexus (Jarvis Nexus + Liquid Continuous-Time ODE)
Fuses:
1. 10,000-D Holographic Reduced Representations (HRR)
2. Continuous-Time Neural ODE: dv/dt = - [1/tau + f(v)] * v + unbind(M, s(t)) + (K_self ⊗ v)
3. Fluid Free Energy dissipation without discrete step boundaries
"""

import time
import json
import numpy as np


class LiquidNexusAgent:
    def __init__(self, dim: int = 10000, tau: float = 0.5, seed: int = 303):
        self.dim = dim
        self.tau = tau
        self.rng = np.random.default_rng(seed)

        # Continuous state vector on hypersphere
        self.v = self.rng.standard_normal(dim)
        self.v /= np.linalg.norm(self.v)

        # Self-referential strange loop kernel
        self.k_self = self.rng.standard_normal(dim)
        self.k_self /= np.linalg.norm(self.k_self)

        # Sensory mapping hypervector
        self.m_obs = self.rng.standard_normal(dim)
        self.m_obs /= np.linalg.norm(self.m_obs)

    def continuous_derivative(self, v_curr: np.ndarray, s_t: np.ndarray) -> np.ndarray:
        """Computes continuous time derivative dv/dt in 10,000-D space."""
        # 1. Holographic Sensory Inversion: unbind(M_obs, s_t)
        f_m = np.fft.rfft(self.m_obs)
        f_s = np.fft.rfft(s_t)
        sensory_drive = np.fft.irfft(np.conj(f_m) * f_s, n=self.dim)
        sensory_drive /= (np.linalg.norm(sensory_drive) + 1e-9)

        # 2. Strange-Loop Self-Feedback: K_self ⊗ v_curr
        f_k = np.fft.rfft(self.k_self)
        f_v = np.fft.rfft(v_curr)
        self_feedback = np.fft.irfft(f_k * f_v, n=self.dim)
        self_feedback /= (np.linalg.norm(self_feedback) + 1e-9)

        # 3. Liquid Conductance: state-dependent decay
        conductance = (1.0 / self.tau) + 0.2 * np.linalg.norm(v_curr)

        # dv/dt = -conductance * v + 0.6 * sensory_drive + 0.4 * self_feedback
        dv_dt = -conductance * v_curr + 0.6 * sensory_drive + 0.4 * self_feedback
        return dv_dt

    def integrate_rk4(self, s_t: np.ndarray, dt: float = 0.05) -> tuple[np.ndarray, float]:
        """RK4 integration step in 10,000-D continuous space."""
        k1 = self.continuous_derivative(self.v, s_t)
        k2 = self.continuous_derivative(self.v + 0.5 * dt * k1, s_t)
        k3 = self.continuous_derivative(self.v + 0.5 * dt * k2, s_t)
        k4 = self.continuous_derivative(self.v + dt * k3, s_t)

        self.v += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.v /= np.linalg.norm(self.v)

        # Compute instantaneous Variational Free Energy
        # s_pred = M_obs ⊗ v
        f_m = np.fft.rfft(self.m_obs)
        f_v = np.fft.rfft(self.v)
        s_pred = np.fft.irfft(f_m * f_v, n=self.dim)
        s_pred /= (np.linalg.norm(s_pred) + 1e-9)

        # Online associative update of observation model
        f_s = np.fft.rfft(s_t)
        target_m = np.fft.irfft(np.conj(f_v) * f_s, n=self.dim)
        target_m /= (np.linalg.norm(target_m) + 1e-9)
        self.m_obs = 0.92 * self.m_obs + 0.08 * target_m
        self.m_obs /= np.linalg.norm(self.m_obs)

        vfe = 1.0 - float(np.dot(s_t / np.linalg.norm(s_t), s_pred))
        return self.v.copy(), vfe


def benchmark_liquid_nexus():
    print("=" * 60)
    print("BENCHMARKING COMBO 2: LIQUID-NEXUS (CONTINUOUS-TIME ODE)")
    print("=" * 60)

    agent = LiquidNexusAgent(dim=10000, tau=0.5, seed=303)
    rng = np.random.default_rng(404)

    # Dynamic time-varying sensory signal (continuous harmonic trajectory)
    t_points = np.linspace(0, 5, 50)
    dt = t_points[1] - t_points[0]
    base_s = rng.standard_normal(10000)

    vfe_history = []
    t0 = time.time()

    for t in t_points:
        s_t = base_s * np.sin(t * 1.5) + rng.standard_normal(10000) * 0.1
        _, vfe = agent.integrate_rk4(s_t, dt=dt)
        vfe_history.append(vfe)

    elapsed = time.time() - t0
    final_vfe = float(np.mean(vfe_history[-10:]))
    dissipation_rate = float((vfe_history[0] - final_vfe) / vfe_history[0] * 100.0)

    print(f"   Initial Continuous VFE: {vfe_history[0]:.4f} -> Final VFE: {final_vfe:.4f}")
    print(f"   Continuous Free Energy Dissipation: {dissipation_rate:+.1f}%")
    print(f"   Continuous Integration Throughput: {len(t_points)/elapsed:.2f} RK4 steps/sec")

    data = {
        "combo": "Combo_02_Liquid_Nexus",
        "initial_vfe": float(vfe_history[0]),
        "final_vfe": final_vfe,
        "dissipation_rate_pct": dissipation_rate,
        "throughput_rk4_sec": float(len(t_points)/elapsed)
    }
    with open("benchmark_results/combo_02_liquid_nexus.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_liquid_nexus()
