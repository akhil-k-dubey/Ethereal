"""
Combination 1: GWS-Nexus (Jarvis Nexus + Global Workspace Competition)
Fuses:
1. 10,000-D Holographic Reduced Representations Substrate
2. Hofstadter Strange-Loop Self-Model
3. Multi-Specialist Global Workspace Competition (Perception, Reasoning, Action, Memory)
4. Conscious Ignition & Universal Broadcast with Dedicated Self-Slot
"""

import time
import json
import numpy as np


class GWSNexusAgent:
    def __init__(self, dim: int = 10000, seed: int = 101):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

        # 1. Self-Slot Vector (The Subjective Anchor)
        self.self_slot = self.rng.standard_normal(dim)
        self.self_slot /= np.linalg.norm(self.self_slot)

        # 2. Strange-Loop Kernel
        self.k_self = self.rng.standard_normal(dim)
        self.k_self /= np.linalg.norm(self.k_self)

        # 3. Four Specialist Modules in 10,000-D space
        self.specialists = {
            "Perception": self.rng.standard_normal(dim),
            "Semantic": self.rng.standard_normal(dim),
            "Action": self.rng.standard_normal(dim),
            "Episodic": self.rng.standard_normal(dim)
        }
        for name in self.specialists:
            self.specialists[name] /= np.linalg.norm(self.specialists[name])

        # 4. Global Workspace Buffer
        self.workspace = np.zeros(dim)
        self.ignition_threshold = 0.75

    def step(self, external_signal: np.ndarray, beta: float = 4.0) -> dict:
        # Context binds Workspace with Strange-Loop Self-Slot
        f_key = np.fft.rfft(self.k_self)
        f_self = np.fft.rfft(self.self_slot)
        self_loop = np.fft.irfft(f_key * f_self, n=self.dim)
        self_loop /= np.linalg.norm(self_loop)

        context = 0.6 * self.workspace + 0.4 * self_loop
        context /= (np.linalg.norm(context) + 1e-9)

        # Specialist Proposals and Saliences
        saliences = {}
        proposals = {}
        for name, spec_vec in self.specialists.items():
            prop = np.tanh(spec_vec + 0.25 * external_signal)
            prop /= np.linalg.norm(prop)
            proposals[name] = prop
            # Dimension-normalized salience
            sal = float(np.abs(np.dot(prop, context)) * 2.5 + np.std(prop) * np.sqrt(self.dim) * 0.8)
            saliences[name] = sal

        # Self-Slot also generates reflective salience
        self_sal = float(np.abs(np.dot(self.self_slot, context)) * 3.0 + 0.6)
        saliences["SelfSlot"] = self_sal
        proposals["SelfSlot"] = self.self_slot

        # Softmax Ignition Competition
        names = list(saliences.keys())
        sal_vals = np.array([saliences[n] for n in names])
        exp_sal = np.exp(beta * (sal_vals - np.max(sal_vals)))
        probs = exp_sal / np.sum(exp_sal)

        winner = str(np.random.choice(names, p=probs))
        winner_vec = proposals[winner]
        ignited = bool(saliences[winner] >= self.ignition_threshold)

        if ignited:
            # Conscious Ignition: Workspace binds winner with Self-Slot
            f_w = np.fft.rfft(winner_vec)
            f_s = np.fft.rfft(self.self_slot)
            bound_conscious = np.fft.irfft(f_w * f_s, n=self.dim)
            bound_conscious /= np.linalg.norm(bound_conscious)

            self.workspace = 0.7 * winner_vec + 0.3 * bound_conscious
            self.workspace /= np.linalg.norm(self.workspace)

            # Universal Broadcast: all specialists absorb the ignited content
            for name in self.specialists:
                self.specialists[name] = 0.8 * self.specialists[name] + 0.2 * self.workspace
                self.specialists[name] /= np.linalg.norm(self.specialists[name])

            # Self-Slot integrates the conscious event
            self.self_slot = 0.85 * self.self_slot + 0.15 * self.workspace
            self.self_slot /= np.linalg.norm(self.self_slot)

        coherence = float(np.dot(self.workspace, self.self_slot))
        return {
            "winner": winner,
            "ignited": ignited,
            "broadcast_coherence": coherence,
            "max_salience": float(np.max(sal_vals))
        }


def benchmark_gws_nexus():
    print("=" * 60)
    print("BENCHMARKING COMBO 1: GWS-NEXUS (GLOBAL WORKSPACE + JARVIS)")
    print("=" * 60)

    agent = GWSNexusAgent(dim=10000, seed=101)
    rng = np.random.default_rng(202)

    logs = []
    t0 = time.time()
    for t in range(25):
        sig = rng.standard_normal(10000)
        res = agent.step(sig)
        logs.append(res)
        if (t + 1) % 5 == 0:
            print(f"   Step {t+1:02d} | Winner: [{res['winner']}] | Ignited: {res['ignited']} | Coherence: {res['broadcast_coherence']:.4f}")

    elapsed = time.time() - t0
    coherence = np.mean([l["broadcast_coherence"] for l in logs])
    ignition_rate = np.mean([1.0 if l["ignited"] else 0.0 for l in logs])

    print(f"\nSummary Combo 1 (GWS-Nexus):")
    print(f"   Mean Conscious Broadcast Coherence: {coherence:.4f}")
    print(f"   Ignition Rate: {ignition_rate*100:.1f}%")
    print(f"   Throughput: {25/elapsed:.2f} conscious cycles/sec")

    data = {
        "combo": "Combo_01_GWS_Nexus",
        "broadcast_coherence": float(coherence),
        "ignition_rate": float(ignition_rate),
        "throughput_cycles_sec": float(25/elapsed)
    }
    with open("benchmark_results/combo_01_gws_nexus.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_gws_nexus()
