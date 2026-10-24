"""
Hyperdimensional Active Inference (HAI) - Generation 3: Active Inference & Counterfactual Imagination
Architecture: Expected Free Energy (EFE) + Counterfactual Rollouts + Autopoietic Strange Loop

Key Innovations:
1. Expected Free Energy (EFE) Action Selection:
   G(a) = α * Pragmatic_Value(a) + β * Epistemic_Curiosity(a) + γ * Self_Coherence(a)
   - Pragmatic Value: Alignment with Prior Preference (Goal hypervector V_goal)
   - Epistemic Value: Information gain / novelty orthogonal to Episodic Memory
   - Self-Coherence: Autopoietic preservation of the Strange-Loop self-identity
2. Counterfactual Imagination in Hyperspace:
   Simulates forward outcomes via Hyperdimensional Transition Kernel T ⊗ (V ⊕ φ(A))
   without interacting with the physical environment.
3. Benchmark: Latent Topology Navigation under Ambiguity & Surprise
"""

import time
import json
import numpy as np


class HyperdimensionalSpaceV3:
    """High-Performance 10,000-D Vector Symbolic Substrate."""

    def __init__(self, dim: int = 10000, seed: int = 2026):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def random_vector(self) -> np.ndarray:
        v = self.rng.standard_normal(self.dim)
        return v / np.linalg.norm(v)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        res = np.fft.irfft(fa * fb, n=self.dim)
        norm = np.linalg.norm(res)
        return res / norm if norm > 1e-12 else res

    def unbind(self, key: np.ndarray, bound: np.ndarray) -> np.ndarray:
        f_key = np.fft.rfft(key)
        f_bound = np.fft.rfft(bound)
        res = np.fft.irfft(np.conj(f_key) * f_bound, n=self.dim)
        norm = np.linalg.norm(res)
        return res / norm if norm > 1e-12 else res

    def superpose(self, vectors: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
        if weights is None:
            weights = [1.0] * len(vectors)
        acc = np.zeros(self.dim, dtype=np.float64)
        for w, v in zip(weights, vectors):
            acc += w * v
        norm = np.linalg.norm(acc)
        return acc / norm if norm > 1e-12 else acc


    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class GridLatentWorld:
    """
    Non-trivial 4x4 Grid Environment represented as high-dimensional graph.
    Nodes: 16 discrete states, each emitting a unique continuous 10,000-D sensory pattern.
    Actions: 0: North, 1: South, 2: East, 3: West
    Goal: State (3, 3)
    """

    def __init__(self, space: HyperdimensionalSpaceV3, grid_size: int = 4):
        self.space = space
        self.size = grid_size
        self.num_states = grid_size * grid_size
        self.state_vectors = [space.random_vector() for _ in range(self.num_states)]
        self.current_coord = (0, 0)
        self.goal_coord = (grid_size - 1, grid_size - 1)

    def coord_to_idx(self, r: int, c: int) -> int:
        return r * self.size + c

    def reset(self) -> np.ndarray:
        self.current_coord = (0, 0)
        idx = self.coord_to_idx(*self.current_coord)
        return self.state_vectors[idx]

    def step(self, action_idx: int) -> tuple[np.ndarray, bool, dict]:
        r, c = self.current_coord
        if action_idx == 0:    # North
            r = max(0, r - 1)
        elif action_idx == 1:  # South
            r = min(self.size - 1, r + 1)
        elif action_idx == 2:  # East
            c = min(self.size - 1, c + 1)
        elif action_idx == 3:  # West
            c = max(0, c - 1)

        self.current_coord = (r, c)
        idx = self.coord_to_idx(r, c)
        done = (self.current_coord == self.goal_coord)
        obs = self.state_vectors[idx]
        return obs, done, {"coord": self.current_coord, "state_idx": idx}


class HAIAgentV3:
    """
    Generation 3 Agent:
    Active Inference via Expected Free Energy & Hyperdimensional Rollouts.
    """

    def __init__(self, space: HyperdimensionalSpaceV3, action_names: list[str]):
        self.space = space
        self.action_names = action_names
        self.num_actions = len(action_names)

        # High-dimensional action basis
        self.action_vectors = [self.space.random_vector() for _ in range(self.num_actions)]

        # Distributed Holographic Associative World Memory:
        # Forward: (State ⊗ Action) ⊗ Next_State
        self.world_memory = np.zeros(self.space.dim, dtype=np.float64)
        # Backward: (Next_State ⊗ Action) ⊗ Prev_State (for Retrograde Planning)
        self.bwd_memory = np.zeros(self.space.dim, dtype=np.float64)
        self.memory_count = 0

        # Strange-Loop Self-Model
        self.k_self = self.space.random_vector()
        self.s_kernel = self.space.random_vector()

        # Current Internal Belief Vector
        self.internal_state = self.space.random_vector()

        # Episodic Memory of visited subspace in current episode
        self.episodic_memory: list[np.ndarray] = []

        # Known state prototypes library for cleanup & distance
        self.known_states: list[np.ndarray] = []

        # Goal vector and Holographic Goal Gradient Field
        self.goal_vector: np.ndarray | None = None
        self.goal_gradient_field = np.zeros(self.space.dim, dtype=np.float64)

    def set_goal(self, goal_obs: np.ndarray):
        self.goal_vector = goal_obs.copy()
        self.recompute_goal_gradient()

    def recompute_goal_gradient(self, hops: int = 5):
        """
        Holographic Retrograde Goal Propagation:
        Propagates goal attraction backward through the transition graph in hyperdimensional space.
        Φ_0 = Goal
        Φ_{t+1} = Φ_t ⊕ (Retrograde_Expansion(Φ_t))
        """
        if self.goal_vector is None or self.memory_count == 0:
            if self.goal_vector is not None:
                self.goal_gradient_field = self.goal_vector.copy()
            return

        field = self.goal_vector.copy()
        decay = 0.75
        for _ in range(hops):
            # For each action, find predecessors: unbind( (field ⊗ a), bwd_memory )
            predecessors = []
            for act_vec in self.action_vectors:
                query = self.space.bind(field, act_vec)
                pred_states = self.space.unbind(query, self.bwd_memory)
                predecessors.append(pred_states)

            step_field = self.space.superpose(predecessors)
            field = self.space.superpose([field, step_field], weights=[1.0, decay])
            decay *= 0.75

        self.goal_gradient_field = field

    def reset_episode(self, start_obs: np.ndarray):
        """Resets episodic memory for new episode."""
        self.episodic_memory = []
        self.update_internal_belief(start_obs, lr=0.9)

    def update_internal_belief(self, observation: np.ndarray, lr: float = 0.5):
        """Bayesian-ish belief update in HD space."""
        self.current_sensory_state = observation.copy()
        self_refl = self.space.bind(self.k_self, self.internal_state)
        self.internal_state = self.space.superpose(
            [self.internal_state, observation, self_refl],
            weights=[1.0 - lr, lr * 0.8, lr * 0.2]
        )
        self.episodic_memory.append(self.internal_state.copy())
        if len(self.episodic_memory) > 15:
            self.episodic_memory.pop(0)

        # Track observed state in known library
        if not any(self.space.similarity(observation, ks) > 0.85 for ks in self.known_states):
            self.known_states.append(observation.copy())

    def counterfactual_rollout(self, action_idx: int) -> np.ndarray:
        """Projects outcome of hypothetical action without executing it."""
        if self.memory_count == 0:
            return self.internal_state

        act_vec = self.action_vectors[action_idx]
        base_s = getattr(self, "current_sensory_state", self.internal_state)
        query_key = self.space.bind(base_s, act_vec)
        v_pred_raw = self.space.unbind(query_key, self.world_memory)

        # Associative clean-up against known states
        best_match = v_pred_raw
        best_sim = 0.0
        for ks in self.known_states:
            sim = self.space.similarity(v_pred_raw, ks)
            if sim > best_sim:
                best_sim = sim
                best_match = ks

        if best_sim > 0.08:
            return best_match
        return v_pred_raw

    def compute_expected_free_energy(self, action_idx: int) -> tuple[float, dict]:
        """
        Calculates Expected Free Energy G(a).
        Uses both direct goal similarity AND Holographic Goal Gradient Field.
        """
        v_hyp = self.counterfactual_rollout(action_idx)

        # 1. Pragmatic Value (Direct Goal + Backward Goal Gradient Field):
        direct_goal_sim = 0.0
        if self.goal_vector is not None:
            direct_goal_sim = self.space.similarity(v_hyp, self.goal_vector)

        gradient_sim = 0.0
        if np.linalg.norm(self.goal_gradient_field) > 1e-6:
            gradient_sim = self.space.similarity(v_hyp, self.goal_gradient_field)

        pragmatic_total = direct_goal_sim * 3.0 + gradient_sim * 1.5

        # 2. Epistemic Value (Curiosity / Novelty within current episode):
        if self.episodic_memory:
            sims = [self.space.similarity(v_hyp, past_v) for past_v in self.episodic_memory[-6:]]
            max_sim_past = max(sims)
            novelty = 1.0 - max_sim_past
        else:
            novelty = 1.0

        # 3. Autopoietic Self-Coherence:
        self_pred = self.space.bind(self.s_kernel, v_hyp)
        self_coherence = self.space.similarity(v_hyp, self_pred)

        # Expected Free Energy:
        w_prag = 4.0
        w_epis = 1.0
        w_self = 0.2

        expected_value = (w_prag * pragmatic_total) + (w_epis * novelty) + (w_self * self_coherence)
        expected_free_energy = -expected_value

        return expected_free_energy, {
            "pragmatic": pragmatic_total,
            "epistemic": novelty,
            "self_coherence": self_coherence,
            "efe": expected_free_energy
        }

    def select_action(self, temperature: float = 0.1) -> tuple[int, dict]:
        """Action selection via Softmax over Negative Expected Free Energy."""
        efes = []
        details = []
        for a_idx in range(self.num_actions):
            efe, d = self.compute_expected_free_energy(a_idx)
            efes.append(efe)
            details.append(d)

        efes = np.array(efes)
        logits = -efes / temperature
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        chosen_action = int(np.random.choice(self.num_actions, p=probs))

        return chosen_action, {
            "chosen_action": chosen_action,
            "action_name": self.action_names[chosen_action],
            "action_probs": probs.tolist(),
            "details": details[chosen_action]
        }

    def train_world_model_step(self, prev_state: np.ndarray, action_idx: int, next_obs: np.ndarray, lr: float = 0.2):
        """Holographic binding into World Transition Memory (Forward & Backward)."""
        act_vec = self.action_vectors[action_idx]

        # Forward transition: (S ⊗ A) ⊗ S'
        fwd_key = self.space.bind(prev_state, act_vec)
        fwd_record = self.space.bind(fwd_key, next_obs)
        self.world_memory += fwd_record

        # Backward transition: (S' ⊗ A) ⊗ S
        bwd_key = self.space.bind(next_obs, act_vec)
        bwd_record = self.space.bind(bwd_key, prev_state)
        self.bwd_memory += bwd_record

        self.memory_count += 1


def benchmark_v3():
    print("=" * 60)
    print("BENCHMARKING HAI GENERATION 3: ACTIVE INFERENCE & EFE")
    print("=" * 60)

    dim = 10000
    space = HyperdimensionalSpaceV3(dim=dim, seed=2026)
    env = GridLatentWorld(space, grid_size=4)
    action_names = ["North", "South", "East", "West"]

    # 1. Benchmark: HAI Active Inference vs Random Walk vs Greedy Reflex
    goal_obs = env.state_vectors[env.coord_to_idx(3, 3)]

    # --- Run HAI Agent ---
    print("\n1. Evaluating HAI Agent with Expected Free Energy (EFE)...")
    hai_agent = HAIAgentV3(space, action_names)
    hai_agent.set_goal(goal_obs)

    # Pre-train world model via exploratory steps across the grid
    obs = env.reset()
    for _ in range(50):
        a_rnd = np.random.randint(4)
        n_obs, _, _ = env.step(a_rnd)
        hai_agent.train_world_model_step(obs, a_rnd, n_obs, lr=0.25)
        obs = n_obs

    # Retrograde goal propagation across discovered transitions
    hai_agent.set_goal(goal_obs)
    hai_agent.recompute_goal_gradient(hops=6)

    # Run navigation episodes
    num_episodes = 5
    hai_steps_to_goal = []
    hai_epistemic_history = []
    hai_pragmatic_history = []

    t0 = time.time()
    for ep in range(num_episodes):
        obs = env.reset()
        hai_agent.reset_episode(obs)
        steps = 0
        done = False
        max_steps = 30

        while not done and steps < max_steps:
            prev_obs = obs.copy()
            action, meta = hai_agent.select_action(temperature=0.08)
            obs, done, info = env.step(action)
            hai_agent.train_world_model_step(prev_obs, action, obs, lr=0.2)
            hai_agent.update_internal_belief(obs)

            hai_epistemic_history.append(meta['details']['epistemic'])
            hai_pragmatic_history.append(meta['details']['pragmatic'])
            steps += 1

        hai_steps_to_goal.append(steps)
        print(f"   Episode {ep+1}: Reached Goal in {steps} steps (Done: {done})")

    elapsed_hai = time.time() - t0

    # --- Run Random Baseline ---
    print("\n2. Evaluating Random Policy Baseline...")
    random_steps = []
    for ep in range(num_episodes):
        env.reset()
        steps = 0
        done = False
        max_steps = 30
        while not done and steps < max_steps:
            a = np.random.randint(4)
            _, done, _ = env.step(a)
            steps += 1
        random_steps.append(steps)

    avg_hai_steps = float(np.mean(hai_steps_to_goal))
    avg_rnd_steps = float(np.mean(random_steps))
    efficiency_gain = (avg_rnd_steps - avg_hai_steps) / avg_rnd_steps * 100.0

    print(f"\nBenchmark Summary V3:")
    print(f"   HAI Mean Steps to Goal: {avg_hai_steps:.1f} / 30")
    print(f"   Random Baseline Steps to Goal: {avg_rnd_steps:.1f} / 30")
    print(f"   Navigation Efficiency Improvement: +{efficiency_gain:.1f}%")
    print(f"   Mean Epistemic Curiosity Value: {np.mean(hai_epistemic_history):.4f}")
    print(f"   Mean Pragmatic Goal Alignment: {np.mean(hai_pragmatic_history):.4f}")
    print(f"   Decision Cycle Throughput: {sum(hai_steps_to_goal)/elapsed_hai:.2f} cycles/sec")

    results = {
        "generation": 3,
        "name": "HAI_ActiveInference_EFE",
        "dimension": dim,
        "mean_steps_to_goal": avg_hai_steps,
        "random_baseline_steps": avg_rnd_steps,
        "efficiency_gain_pct": efficiency_gain,
        "mean_epistemic_curiosity": float(np.mean(hai_epistemic_history)),
        "mean_pragmatic_alignment": float(np.mean(hai_pragmatic_history)),
        "throughput_cycles_sec": float(sum(hai_steps_to_goal)/elapsed_hai)
    }

    with open("results_v3_active_inference.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    benchmark_v3()
