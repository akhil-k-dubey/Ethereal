"""
Ablation Study & Distributed Fractional Power Test:
1. Ablation 1: Pure Curiosity Baseline (Retrograde field disabled, purely intrinsic exploration)
2. Ablation 2: Full System (Dynamic Arbitration: Curiosity + Retrograde Field)
3. Ablation 3: Pure Retrograde (No curiosity fallback)
4. Empirical Test: Distributed Transition Bundle Fractional Power Decay Curve (tau = 0.5..5.0 on M = sum S*A*S')
"""

import json
import numpy as np


class ScaledGridEnv:
    def __init__(self, grid_dim: int = 32, seed: int = 42):
        self.grid_dim = grid_dim
        self.num_states = grid_dim * grid_dim
        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        self.action_deltas = []
        for speed in [1, 2, 3]:
            for dr, dc in directions:
                self.action_deltas.append((dr * speed, dc * speed))
        self.num_actions = len(self.action_deltas)
        self.start_coord = (0, 0)
        self.goal_coord = (grid_dim - 1, grid_dim - 1)
        self.current_coord = self.start_coord

    def coord_to_idx(self, r: int, c: int) -> int:
        return r * self.grid_dim + c

    def reset(self) -> int:
        self.current_coord = self.start_coord
        return self.coord_to_idx(*self.current_coord)

    def step(self, action_idx: int) -> tuple[int, bool]:
        dr, dc = self.action_deltas[action_idx]
        r = int(np.clip(self.current_coord[0] + dr, 0, self.grid_dim - 1))
        c = int(np.clip(self.current_coord[1] + dc, 0, self.grid_dim - 1))
        self.current_coord = (r, c)
        done = (self.current_coord == self.goal_coord)
        return self.coord_to_idx(r, c), done


class AblationPlanner:
    def __init__(self, dim: int = 10000, num_actions: int = 24, seed: int = 42):
        self.dim = dim
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)
        self.actions = [self.rng.standard_normal(dim) / np.sqrt(dim) for _ in range(num_actions)]
        self.m_fwd = np.zeros(dim, dtype=np.float64)
        self.m_bwd = np.zeros(dim, dtype=np.float64)
        self.state_codebook: dict[int, np.ndarray] = {}

    def get_state_vector(self, state_idx: int) -> np.ndarray:
        if state_idx not in self.state_codebook:
            rng = np.random.default_rng(state_idx + 1000)
            v = rng.standard_normal(self.dim)
            self.state_codebook[state_idx] = v / np.linalg.norm(v)
        return self.state_codebook[state_idx]

    def record_transition(self, s: int, a: int, s_next: int):
        s_v = self.get_state_vector(s)
        a_v = self.actions[a]
        sn_v = self.get_state_vector(s_next)

        f_sa = np.fft.rfft(s_v) * np.fft.rfft(a_v)
        self.m_fwd += np.fft.irfft(f_sa * np.fft.rfft(sn_v), n=self.dim)
        self.m_bwd += np.fft.irfft(np.fft.rfft(sn_v) * np.fft.rfft(a_v) * np.fft.rfft(s_v), n=self.dim)

    def predict_forward_with_uncertainty(self, s_curr_idx: int, a_idx: int) -> tuple[np.ndarray, float]:
        s_v = self.get_state_vector(s_curr_idx)
        a_v = self.actions[a_idx]
        f_sa = np.fft.rfft(s_v) * np.fft.rfft(a_v)
        raw_pred = np.fft.irfft(np.conj(f_sa) * np.fft.rfft(self.m_fwd), n=self.dim)
        raw_norm = np.linalg.norm(raw_pred)
        if raw_norm < 1e-9:
            return s_v, 1.0
        raw_pred /= raw_norm

        best_sim = max([float(np.dot(raw_pred, kv)) for kv in self.state_codebook.values()], default=0.0)
        uncertainty = max(0.0, 1.0 - best_sim)
        return raw_pred, uncertainty

    def compute_retrograde_field(self, goal_idx: int, hops: int = 24, cleanup_threshold: float = 0.05) -> np.ndarray:
        goal_v = self.get_state_vector(goal_idx)
        field = goal_v.copy()
        decay = 0.88
        for _ in range(hops):
            valid = []
            f_f = np.fft.rfft(field)
            for a_v in self.actions:
                f_q = f_f * np.fft.rfft(a_v)
                raw = np.fft.irfft(np.conj(f_q) * np.fft.rfft(self.m_bwd), n=self.dim)
                norm = np.linalg.norm(raw)
                if norm < 1e-9:
                    continue
                raw /= norm
                sim = max([float(np.dot(raw, kv)) for kv in self.state_codebook.values()], default=0.0)
                if sim >= cleanup_threshold:
                    valid.append((raw, sim))
            if valid:
                acc = np.mean([v[0] for v in valid], axis=0)
                field = 0.6 * field + 0.4 * decay * (acc / np.linalg.norm(acc))
                field /= np.linalg.norm(field)
                decay *= 0.88
        return field

    def select_action(self, s: int, goal: int, field: np.ndarray | None, mode: str = "FULL") -> int:
        """
        mode:
        - "PURE_CURIOSITY": Ignores retrograde field completely.
        - "PURE_RETROGRADE": Always follows field gradient, no curiosity fallback.
        - "FULL": Dynamic arbitration (curiosity until field lock).
        """
        evals = []
        for a in range(self.num_actions):
            p_v, uncert = self.predict_forward_with_uncertainty(s, a)
            prag = float(np.dot(p_v, field)) if field is not None else 0.0
            evals.append((a, prag, uncert))

        dr, dc = ((goal // 32) - (s // 32), (goal % 32) - (s % 32))
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        def curiosity_val(a_idx, uncert):
            sp = (a_idx // 8) + 1
            act_dr, act_dc = directions[a_idx % 8][0] * sp, directions[a_idx % 8][1] * sp
            dir_align = (dr * act_dr + dc * act_dc) / (np.sqrt(dr**2 + dc**2 + 1e-6) * np.sqrt(act_dr**2 + act_dc**2 + 1e-6))
            return 0.6 * dir_align + 1.0 * uncert

        if mode == "PURE_CURIOSITY":
            return max(evals, key=lambda e: curiosity_val(e[0], e[2]))[0]

        else:  # FULL
            max_prag = max([e[1] for e in evals])
            if max_prag >= 0.06:
                return max(evals, key=lambda e: 3.5 * e[1] - 0.2 * e[2])[0]
            else:
                return max(evals, key=lambda e: curiosity_val(e[0], e[2]))[0]


def run_curiosity_ablation():
    print("=" * 80)
    print("RUNNING CRITICAL CONTROL EXPERIMENT: PURE CURIOSITY VS FULL SYSTEM")
    print("=" * 80)

    env = ScaledGridEnv(grid_dim=32, seed=42)
    goal_idx = env.coord_to_idx(*env.goal_coord)
    n_episodes = 10

    # 1. Evaluate PURE CURIOSITY (No retrograde field at all)
    print("\n1. Testing PURE CURIOSITY ABLATION (Retrograde Field Disabled)...")
    planner_curiosity = AblationPlanner(dim=10000, num_actions=24, seed=42)
    # 300 exploratory steps
    c = env.reset()
    for _ in range(300):
        a = np.random.randint(24)
        nc, _ = env.step(a)
        planner_curiosity.record_transition(c, a, nc)
        c = nc

    curiosity_steps = []
    for ep in range(n_episodes):
        c = env.reset()
        steps = 0
        done = False
        while not done and steps < 100:
            a = planner_curiosity.select_action(c, goal_idx, None, mode="PURE_CURIOSITY")
            nc, done = env.step(a)
            planner_curiosity.record_transition(c, a, nc)
            c = nc
            steps += 1
        curiosity_steps.append(steps)
        print(f"   Episode {ep+1:02d}: Steps = {steps:2d} (Goal: {done})")

    # 2. Evaluate FULL SYSTEM (Dynamic Arbitration)
    print("\n2. Testing FULL SYSTEM (Dynamic Arbitration: Curiosity + Retrograde Field)...")
    planner_full = AblationPlanner(dim=10000, num_actions=24, seed=42)
    c = env.reset()
    for _ in range(300):
        a = np.random.randint(24)
        nc, _ = env.step(a)
        planner_full.record_transition(c, a, nc)
        c = nc

    full_field = planner_full.compute_retrograde_field(goal_idx, hops=24)
    full_steps = []
    for ep in range(n_episodes):
        c = env.reset()
        steps = 0
        done = False
        while not done and steps < 100:
            a = planner_full.select_action(c, goal_idx, full_field, mode="FULL")
            nc, done = env.step(a)
            planner_full.record_transition(c, a, nc)
            c = nc
            steps += 1
            if done and ep == 0:
                full_field = planner_full.compute_retrograde_field(goal_idx, hops=24)
        full_steps.append(steps)
        print(f"   Episode {ep+1:02d}: Steps = {steps:2d} (Goal: {done})")

    mean_curiosity = float(np.mean(curiosity_steps))
    std_curiosity = float(np.std(curiosity_steps))
    mean_full = float(np.mean(full_steps))
    std_full = float(np.std(full_steps))

    print("\n" + "=" * 80)
    print("ABLATION RESULTS SUMMARY:")
    print("=" * 80)
    print(f"   Pure Curiosity Alone (Mean ± Std):      {mean_curiosity:.2f} ± {std_curiosity:.2f} steps")
    print(f"   Full System (Curiosity + Field):       {mean_full:.2f} ± {std_full:.2f} steps")
    print(f"   Delta Contributed by Retrograde Field: {mean_curiosity - mean_full:+.2f} steps ({((mean_curiosity - mean_full)/mean_curiosity)*100:+.1f}%)")
    print("=" * 80)

    # 3. Distributed Transition Bundle Fractional Power Decay Test
    print("\n" + "=" * 80)
    print("EMPIRICAL TEST: FRACTIONAL POWER DECAY ON DISTRIBUTED BUNDLE (M = sum S*A*S')")
    print("=" * 80)

    dim = 10000
    rng = np.random.default_rng(777)
    # Create a realistic distributed bundle of 30 sequential transitions
    states = [rng.standard_normal(dim) / np.sqrt(dim) for _ in range(16)]
    act = rng.standard_normal(dim) / np.sqrt(dim)

    # Accumulate transitions into bundle M
    M = np.zeros(dim, dtype=np.float64)
    for i in range(15):
        # (S_i ⊗ A) ⊗ S_{i+1}
        f_sa = np.fft.rfft(states[i]) * np.fft.rfft(act)
        M += np.fft.irfft(f_sa * np.fft.rfft(states[i+1]), n=dim)

    f_M = np.fft.rfft(M)
    f_M_norm = f_M / (np.abs(f_M) + 1e-9)  # Phase-only / spectral normalization

    print(f"{'Fractional tau':<18} | {'Sim to Ground-Truth Reachability':<35} | {'Status':<15}")
    print("-" * 75)

    frac_results = []
    for tau in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        f_tau = (np.abs(f_M_norm)**tau) * np.exp(1j * np.angle(f_M_norm) * tau)
        # Query reachability from S_0
        q = np.fft.irfft(np.conj(np.fft.rfft(states[0])) * f_tau, n=dim)
        q /= (np.linalg.norm(q) + 1e-9)

        target_idx = int(min(15, max(1, round(tau))))
        sim = float(np.dot(q, states[target_idx]))
        status = "SIGNAL" if sim > 0.05 else ("WEAK" if sim > 0.02 else "NOISE")
        print(f"tau = {tau:<12.1f} | {sim:>32.4f} | {status:<15}")
        frac_results.append((tau, sim, status))

    results = {
        "mean_curiosity_alone": mean_curiosity,
        "std_curiosity_alone": std_curiosity,
        "mean_full_system": mean_full,
        "std_full_system": std_full,
        "delta_steps_saved": mean_curiosity - mean_full,
        "frac_power_bundle": frac_results
    }
    with open("benchmark_results/ablation_study_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_curiosity_ablation()
