"""
Thesis 3: Global Workspace + Self-Slot (GWS)
Architecture: Global Neuronal Workspace with Conscious Ignition & Self-Slot (Baars/Dehaene)

Components:
- Specialist Modules:
  * Perception Specialist (Sensory features)
  * Language/Semantic Specialist (Relational concepts)
  * Action/Motor Specialist (Affordances)
  * Memory Specialist (Episodic recall)
- Dedicated Self-Slot: Encodes the system's current subjective stance, meta-confidence, and identity.
- Competitive Attention / Ignition:
  Non-linear ignition threshold. When a specialist wins competition, its state is broadcast
  globally across the entire network, binding with the Self-Slot.
"""

import time
import json
import numpy as np


class SpecialistModule:
    def __init__(self, name: str, dim: int, seed: int):
        self.name = name
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.state = self.rng.standard_normal(dim)
        self.state /= np.linalg.norm(self.state)

    def generate_proposal(self, context: np.ndarray) -> tuple[np.ndarray, float]:
        """Proposes a representation and its intrinsic salience/urgency."""
        # Non-linear feature processing
        proposal = np.tanh(self.state + 0.3 * context)
        proposal /= np.linalg.norm(proposal)
        # Salience: variance + energy
        salience = float(np.std(proposal) * 10.0 + np.abs(np.dot(proposal, context)))
        return proposal, salience

    def receive_broadcast(self, broadcast_vector: np.ndarray, lr: float = 0.2):
        """Updates internal state based on the globally ignited workspace."""
        self.state = (1.0 - lr) * self.state + lr * broadcast_vector
        self.state /= np.linalg.norm(self.state)


class GlobalWorkspaceSystem:
    def __init__(self, workspace_dim: int = 512, seed: int = 707):
        self.dim = workspace_dim
        self.rng = np.random.default_rng(seed)

        # Specialist modules
        self.specialists = [
            SpecialistModule("Perception", workspace_dim, seed=seed + 1),
            SpecialistModule("Semantic", workspace_dim, seed=seed + 2),
            SpecialistModule("Action", workspace_dim, seed=seed + 3),
            SpecialistModule("Memory", workspace_dim, seed=seed + 4),
        ]

        # The Self-Slot (The Conscious Self-Model)
        self.self_slot = self.rng.standard_normal(workspace_dim)
        self.self_slot /= np.linalg.norm(self.self_slot)

        # Global Workspace Buffer
        self.workspace_state = np.zeros(workspace_dim)
        self.ignition_threshold = 1.2
        self.broadcast_history = []

    def step(self, external_stimulus: np.ndarray) -> dict:
        # Context includes current workspace + Self-Slot
        context = 0.6 * self.workspace_state + 0.4 * self.self_slot
        context /= (np.linalg.norm(context) + 1e-9)

        # Collect proposals & salience from all specialists
        proposals = []
        saliences = []
        for s in self.specialists:
            prop, sal = s.generate_proposal(context + 0.2 * external_stimulus)
            proposals.append(prop)
            saliences.append(sal)

        # Self-Slot also generates its own reflective salience
        self_salience = float(np.abs(np.dot(self.self_slot, context)) * 2.5)

        # Competitive Selection via Softmax Ignition
        all_saliences = np.array(saliences + [self_salience])
        beta = 3.0  # Competitive temperature
        exp_sal = np.exp(beta * (all_saliences - np.max(all_saliences)))
        probs = exp_sal / np.sum(exp_sal)

        winner_idx = int(np.random.choice(len(all_saliences), p=probs))
        ignited = bool(all_saliences[winner_idx] >= self.ignition_threshold)

        if winner_idx < len(self.specialists):
            winning_name = self.specialists[winner_idx].name
            winning_vector = proposals[winner_idx]
            self_engaged = False
        else:
            winning_name = "Self-Slot"
            winning_vector = self.self_slot
            self_engaged = True

        # Non-linear Ignition into Global Workspace:
        # Binds the winning content with the Self-Slot!
        if ignited:
            self.workspace_state = np.tanh(0.7 * winning_vector + 0.3 * self.self_slot)
            self.workspace_state /= np.linalg.norm(self.workspace_state)

            # Global Broadcast: all specialists receive the ignited workspace
            for s in self.specialists:
                s.receive_broadcast(self.workspace_state)

            # Self-Slot update: conscious self integrates what was broadcast
            self.self_slot = np.tanh(0.85 * self.self_slot + 0.15 * winning_vector)
            self.self_slot /= np.linalg.norm(self.self_slot)

        self.broadcast_history.append((winning_name, ignited))

        return {
            "winner": winning_name,
            "ignited": ignited,
            "self_slot_engaged": self_engaged,
            "max_salience": float(np.max(all_saliences)),
            "broadcast_coherence": float(np.dot(self.workspace_state, self.self_slot))
        }


def benchmark_gws():
    print("=" * 60)
    print("BENCHMARKING THESIS 3: GLOBAL WORKSPACE + SELF-SLOT")
    print("=" * 60)

    gws = GlobalWorkspaceSystem(workspace_dim=512, seed=707)
    rng = np.random.default_rng(999)

    bench_logs = []
    winners = []
    t0 = time.time()

    for trial in range(30):
        stim = rng.standard_normal(512)
        res = gws.step(stim)
        bench_logs.append(res)
        winners.append(res["winner"])
        if (trial + 1) % 10 == 0:
            print(f"   Trial {trial+1:02d} | Winner: [{res['winner']}] | Ignited: {res['ignited']} | Coherence: {res['broadcast_coherence']:.4f}")

    elapsed = time.time() - t0
    ignition_rate = np.mean([1.0 if r["ignited"] else 0.0 for r in bench_logs])
    self_engagement_rate = np.mean([1.0 if r["self_slot_engaged"] else 0.0 for r in bench_logs])
    avg_coherence = np.mean([r["broadcast_coherence"] for r in bench_logs])

    print(f"\nSummary Thesis 3 (GWS):")
    print(f"   Conscious Ignition Rate: {ignition_rate*100:.1f}%")
    print(f"   Self-Slot Engagement Rate: {self_engagement_rate*100:.1f}%")
    print(f"   Mean Broadcast Self-Coherence: {avg_coherence:.4f}")
    print(f"   Throughput: {30/elapsed:.2f} cycles/sec")

    data = {
        "thesis": "Thesis_03_GWS",
        "ignition_rate": float(ignition_rate),
        "self_slot_engagement_rate": float(self_engagement_rate),
        "mean_broadcast_self_coherence": float(avg_coherence),
        "throughput_cycles_sec": float(30/elapsed)
    }
    with open("benchmark_results/thesis_03_gws.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_gws()
