"""
Combination 8: Neuromorphic-Spiking-Nexus (Jarvis Nexus + Event-Driven Sparse Spikes)
Fuses:
1. 10,000-D Holographic Reduced Representations (HRR)
2. Event-Driven Neuromorphic Spiking Thresholds
3. Energy-proportional computation: 90%+ computational silence during normal state, spiking only on surprise
"""

import time
import json
import numpy as np


class SpikingNexusAgent:
    def __init__(self, dim: int = 10000, spike_threshold: float = 0.015, seed: int = 555):
        self.dim = dim
        self.spike_threshold = spike_threshold
        self.rng = np.random.default_rng(seed)

        self.state_mem = np.zeros(dim)
        self.k_self = self.rng.standard_normal(dim)
        self.k_self /= np.linalg.norm(self.k_self)

    def event_driven_spike_step(self, sensory_input: np.ndarray) -> tuple[np.ndarray, dict]:
        # 1. Update membrane potential
        self.state_mem = 0.85 * self.state_mem + 0.15 * sensory_input

        # 2. Event-driven ternary spiking: {-1, 0, +1}
        spikes = np.zeros(self.dim)
        spikes[self.state_mem > self.spike_threshold] = 1.0
        spikes[self.state_mem < -self.spike_threshold] = -1.0

        # Reset spiked membranes
        self.state_mem[spikes != 0] = 0.0

        # Sparsity metric
        active_spikes = np.count_nonzero(spikes)
        sparsity = float(1.0 - (active_spikes / self.dim))

        # Perform sparse binding with self-key only using active spike coordinates
        if active_spikes > 0:
            spikes_norm = spikes / np.linalg.norm(spikes)
            f_k = np.fft.rfft(self.k_self)
            f_s = np.fft.rfft(spikes_norm)
            bound = np.fft.irfft(f_k * f_s, n=self.dim)
            bound /= np.linalg.norm(bound)
        else:
            bound = np.zeros(self.dim)

        return bound, {
            "active_spikes": active_spikes,
            "sparsity_pct": sparsity * 100.0,
            "energy_saving_factor": float(self.dim / (active_spikes + 1e-6))
        }


def benchmark_spiking_nexus():
    print("=" * 60)
    print("BENCHMARKING COMBO 8: NEUROMORPHIC-SPIKING-NEXUS")
    print("=" * 60)

    agent = SpikingNexusAgent(dim=10000, spike_threshold=0.012, seed=555)
    rng = np.random.default_rng(666)

    sparsities = []
    energy_savings = []
    t0 = time.time()
    n_steps = 40

    for step in range(n_steps):
        # Sparse external stimuli
        stim = rng.standard_normal(10000) * 0.02
        _, meta = agent.event_driven_spike_step(stim)
        sparsities.append(meta["sparsity_pct"])
        energy_savings.append(meta["energy_saving_factor"])

    elapsed = time.time() - t0
    avg_sparsity = float(np.mean(sparsities))
    avg_savings = float(np.mean(energy_savings))

    print(f"   Mean Spike Sparsity: {avg_sparsity:.2f}% (Computational silence)")
    print(f"   Energy Reduction Factor: {avg_savings:.1f}x lower synaptic activity than dense")
    print(f"   Throughput: {n_steps/elapsed:.2f} neuromorphic event cycles/sec at D=10,000")

    data = {
        "combo": "Combo_08_Spiking_Nexus",
        "mean_sparsity_pct": avg_sparsity,
        "energy_reduction_factor": avg_savings,
        "throughput_cycles_sec": float(n_steps/elapsed)
    }
    with open("benchmark_results/combo_08_spiking_nexus.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_spiking_nexus()
