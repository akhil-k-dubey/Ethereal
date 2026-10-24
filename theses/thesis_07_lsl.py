"""
Thesis 7: Liquid Strange Loop (LSL)
Architecture: Liquid Time-Constant Neural Network ODE with Hofstadter Strange-Loop Kernel (Hasani/Hofstadter)

Continuous-time ODE state evolution:
dx/dt = - [1/tau + f(x)] * x + W_in * u(t) + W_rec(x) * x
where W_rec(x) is state-dependent (self-modulating liquid strange loop).
Integrated via Runge-Kutta 4 (RK4).
"""

import time
import json
import numpy as np


class LiquidStrangeLoop:
    def __init__(self, state_dim: int = 32, in_dim: int = 8, tau: float = 1.0, seed: int = 777):
        self.dim = state_dim
        self.in_dim = in_dim
        self.tau = tau
        self.rng = np.random.default_rng(seed)

        # Baseline weights
        self.W_in = self.rng.standard_normal((state_dim, in_dim)) * 0.2
        self.W_rec_base = self.rng.standard_normal((state_dim, state_dim)) * 0.15
        np.fill_diagonal(self.W_rec_base, 0.0)

        # State vector
        self.state = self.rng.standard_normal(state_dim) * 0.1

    def ode_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Continuous-time Liquid ODE:
        Non-linear time-constant modulation + self-referential strange-loop recurrent kernel.
        """
        # State-dependent conductance (the 'liquid' component)
        conductance = (1.0 / self.tau) + 0.5 * np.log1p(np.exp(x**2))

        # Strange-loop self-modulation: W_rec adapts to outer product of current state!
        self_modulation = 0.1 * np.outer(x, x) / (np.linalg.norm(x)**2 + 1e-6)
        W_effective = self.W_rec_base + self_modulation

        dx_dt = -conductance * x + (self.W_in @ u) + np.tanh(W_effective @ x)
        return dx_dt

    def step_rk4(self, u: np.ndarray, dt: float = 0.05) -> np.ndarray:
        """4th-order Runge-Kutta numerical integration."""
        k1 = self.ode_derivative(self.state, u)
        k2 = self.ode_derivative(self.state + 0.5 * dt * k1, u)
        k3 = self.ode_derivative(self.state + 0.5 * dt * k2, u)
        k4 = self.ode_derivative(self.state + dt * k3, u)

        self.state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.state.copy()


def benchmark_lsl():
    print("=" * 60)
    print("BENCHMARKING THESIS 7: LIQUID STRANGE LOOP (LSL)")
    print("=" * 60)

    lsl = LiquidStrangeLoop(state_dim=32, in_dim=8, tau=0.8, seed=777)
    rng = np.random.default_rng(888)

    # Drive with complex multi-frequency continuous signal
    t_span = np.linspace(0, 10, 200)
    dt = t_span[1] - t_span[0]

    trajectory = []
    t0 = time.time()
    for t in t_span:
        u = np.sin([t * 1.0, t * 2.5, t * 0.7, t * 4.1, t * 0.3, t * 1.9, t * 3.3, t * 0.5])
        st = lsl.step_rk4(u, dt=dt)
        trajectory.append(st)

    elapsed = time.time() - t0
    trajectory = np.array(trajectory)

    # Benchmark: Attractor smoothness, stability, and Lyapunov-like continuity
    diffs = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
    continuity_smoothness = float(np.std(diffs))
    trajectory_energy = float(np.mean(np.linalg.norm(trajectory, axis=1)))

    print(f"   Trajectory Energy: {trajectory_energy:.4f}")
    print(f"   Continuous Flow Smoothness: {continuity_smoothness:.5f} (Low variance = fluid continuity)")
    print(f"   Throughput: {len(t_span)/elapsed:.2f} integration steps/sec")

    data = {
        "thesis": "Thesis_07_LSL",
        "trajectory_energy": float(trajectory_energy),
        "continuous_smoothness": float(continuity_smoothness),
        "throughput_steps_sec": float(len(t_span)/elapsed)
    }
    with open("benchmark_results/thesis_07_lsl.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_lsl()
