"""
Thesis 10: Energy-Based Autopoietic Field (EBAF)
Architecture: Energy-Based Models (LeCun) with Maturana/Varela Autopoietic Homeostasis Boundaries

Components:
- Energy function E(x, s, a) integrating sensory energy, action effort, and homeostatic self-viability.
- Autopoietic boundary: The system actively acts on the world to prevent its own internal state
  from exiting the viable homeostatic manifold (self-preservation / autopoiesis).
"""

import time
import json
import numpy as np


class EnergyBasedAutopoieticField:
    def __init__(self, state_dim: int = 16, action_dim: int = 4, viability_radius: float = 2.0, seed: int = 333):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.r_viability = viability_radius
        self.rng = np.random.default_rng(seed)

        # Homeostatic target setpoint s_0
        self.s_homeo = np.zeros(state_dim)

        # Current internal state
        self.s = self.s_homeo.copy()

        # Coupling matrices
        self.W_sens = self.rng.standard_normal((state_dim, state_dim)) * 0.2
        self.W_act = self.rng.standard_normal((state_dim, action_dim)) * 0.3

    def compute_energy(self, x_env: np.ndarray, a: np.ndarray) -> tuple[float, float, float]:
        """
        Total Energy E = E_sensory + E_effort + E_autopoietic_boundary
        """
        # Environmental mismatch
        s_pred = np.tanh(self.W_sens @ x_env + self.W_act @ a)
        e_sens = float(np.sum((self.s - s_pred)**2))

        # Action effort
        e_act = float(0.1 * np.sum(a**2))

        # Autopoietic boundary penalty: spikes quadratically outside viability radius!
        dist_from_homeo = np.linalg.norm(self.s - self.s_homeo)
        e_viability = float(max(0.0, dist_from_homeo - self.r_viability)**2 * 10.0)

        total_e = e_sens + e_act + e_viability
        return total_e, e_sens, e_viability

    def step_action(self, x_env: np.ndarray, num_langevin_steps: int = 10, lr: float = 0.08) -> tuple[np.ndarray, dict]:
        """Selects action via Langevin gradient descent on the energy field."""
        a = np.zeros(self.action_dim)

        for _ in range(num_langevin_steps):
            # Numerical gradient dE/da
            eps = 1e-4
            grad_a = np.zeros(self.action_dim)
            base_e, _, _ = self.compute_energy(x_env, a)
            for i in range(self.action_dim):
                a_pert = a.copy()
                a_pert[i] += eps
                e_pert, _, _ = self.compute_energy(x_env, a_pert)
                grad_a[i] = (e_pert - base_e) / eps

            # Gradient descent step
            a -= lr * grad_a

        # World updates internal state s: action actively restores homeostasis
        self.s = np.tanh(0.7 * self.s + 0.9 * (self.W_act @ a) + 0.15 * x_env)

        tot_e, sens_e, viab_e = self.compute_energy(x_env, a)
        viable = bool(np.linalg.norm(self.s - self.s_homeo) <= self.r_viability)

        return a, {
            "total_energy": tot_e,
            "sensory_energy": sens_e,
            "boundary_energy": viab_e,
            "dist_homeo": float(np.linalg.norm(self.s - self.s_homeo)),
            "viable": viable
        }


def benchmark_ebaf():
    print("=" * 60)
    print("BENCHMARKING THESIS 10: ENERGY-BASED AUTOPOIETIC FIELD (EBAF)")
    print("=" * 60)

    ebaf = EnergyBasedAutopoieticField(state_dim=16, action_dim=4, viability_radius=3.5, seed=333)
    rng = np.random.default_rng(444)

    logs = []
    t0 = time.time()
    n_steps = 40

    for step in range(n_steps):
        # Environmental disruption pushes system away from homeostatic viability
        x_env = rng.standard_normal(16) * 1.5
        _, meta = ebaf.step_action(x_env, num_langevin_steps=12, lr=0.1)
        logs.append(meta)
        if (step + 1) % 10 == 0:
            print(f"   Step {step+1:02d} | Total E: {meta['total_energy']:.3f} | Dist Homeo: {meta['dist_homeo']:.3f} | Viable: {meta['viable']}")

    elapsed = time.time() - t0
    viability_rate = np.mean([1.0 if l["viable"] else 0.0 for l in logs])
    avg_energy = np.mean([l["total_energy"] for l in logs])

    print(f"\nSummary Thesis 10 (EBAF):")
    print(f"   Autopoietic Viability Rate: {viability_rate*100:.1f}%")
    print(f"   Mean Energy: {avg_energy:.4f}")
    print(f"   Throughput: {n_steps/elapsed:.2f} Langevin cycles/sec")

    data = {
        "thesis": "Thesis_10_EBAF",
        "autopoietic_viability_rate": float(viability_rate),
        "mean_energy": float(avg_energy),
        "throughput_cycles_sec": float(n_steps/elapsed)
    }
    with open("benchmark_results/thesis_10_ebaf.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_ebaf()
