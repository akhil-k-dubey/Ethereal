"""
Thesis 8: Spiking Self-Modeling Brain (SSMB)
Architecture: Neuromorphic Leaky Integrate-and-Fire (LIF) Network with Local STDP & Self-Monitoring Population

Components:
- Primary Sensory/Processing Population (LIF neurons)
- Collateral Self-Monitoring Population: Receives efference copies of spikes to model own firing state
- STDP: Spike-Timing-Dependent Plasticity (local biological learning without backpropagation)
"""

import time
import json
import numpy as np


class SpikingSelfModel:
    def __init__(self, n_primary: int = 64, n_self: int = 16, seed: int = 101):
        self.n_primary = n_primary
        self.n_self = n_self
        self.rng = np.random.default_rng(seed)

        # Membrane parameters
        self.decay = 0.90
        self.v_th = 2.5

        # Membrane potentials
        self.v_prim = np.zeros(n_primary)
        self.v_self = np.zeros(n_self)

        # Synaptic weights
        self.W_in = self.rng.uniform(0.05, 0.20, (n_primary, 16))
        # Self-monitoring synapses (primary -> self-population)
        self.W_self = self.rng.uniform(0.1, 0.35, (n_self, n_primary))

        # Spike trace histories for STDP
        self.trace_prim = np.zeros(n_primary)
        self.trace_self = np.zeros(n_self)

    def step(self, input_current: np.ndarray, stdp_lr: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        # 1. Update Primary Population
        i_prim = self.W_in @ input_current
        self.v_prim = self.decay * self.v_prim + i_prim
        spikes_prim = (self.v_prim >= self.v_th).astype(np.float64)
        self.v_prim[spikes_prim > 0] = 0.0  # Reset

        # 2. Update Self-Monitoring Population
        i_self = self.W_self @ spikes_prim
        self.v_self = self.decay * self.v_self + i_self
        spikes_self = (self.v_self >= self.v_th).astype(np.float64)
        self.v_self[spikes_self > 0] = 0.0  # Reset

        # 3. STDP Local Synaptic Plasticity
        # Pre: spikes_prim, Post: spikes_self
        self.trace_prim = 0.9 * self.trace_prim + spikes_prim
        self.trace_self = 0.9 * self.trace_self + spikes_self

        # Potentiation when post spikes after pre
        dw_stdp = stdp_lr * (np.outer(spikes_self, self.trace_prim) - 0.5 * np.outer(self.trace_self, spikes_prim))
        self.W_self = np.clip(self.W_self + dw_stdp, 0.0, 1.0)

        return spikes_prim, spikes_self


def benchmark_ssmb():
    print("=" * 60)
    print("BENCHMARKING THESIS 8: SPIKING SELF-MODELING BRAIN (SSMB)")
    print("=" * 60)

    ssmb = SpikingSelfModel(n_primary=64, n_self=16, seed=101)
    rng = np.random.default_rng(202)

    total_prim_spikes = 0
    total_self_spikes = 0
    t0 = time.time()
    n_steps = 100

    rates_prim = []
    rates_self = []

    for _ in range(n_steps):
        stim = rng.uniform(0.0, 1.0, 16)
        s_prim, s_self = ssmb.step(stim, stdp_lr=0.015)
        total_prim_spikes += np.sum(s_prim)
        total_self_spikes += np.sum(s_self)
        rates_prim.append(float(np.mean(s_prim)))
        rates_self.append(float(np.mean(s_self)))

    elapsed = time.time() - t0

    # Correlation between primary activity and self-monitoring activity
    corr = float(np.corrcoef(rates_prim, rates_self)[0, 1])
    sparsity = float(1.0 - (total_prim_spikes / (n_steps * 64)))

    print(f"   Primary Spike Sparsity: {sparsity*100:.1f}%")
    print(f"   Self-Monitoring Correlation: {corr:.4f} (Biological self-tracking)")
    print(f"   Throughput: {n_steps/elapsed:.2f} biological ms steps/sec")

    data = {
        "thesis": "Thesis_08_SSMB",
        "spike_sparsity": float(sparsity),
        "self_monitoring_correlation": float(corr),
        "throughput_steps_sec": float(n_steps/elapsed)
    }
    with open("benchmark_results/thesis_08_ssmb.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_ssmb()
