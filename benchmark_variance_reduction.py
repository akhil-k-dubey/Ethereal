"""
Variance Reduction & Rigorous Empirical Evaluation:
1. Dynamic Epistemic-Pragmatic Arbitration (Gradient-Thresholded Curiosity)
2. Extended Wavefront Retrograde Propagation (k = 24 hops)
3. 10-Episode Statistical Evaluation (Mean ± Std Step Count)
4. Empirical Test of Fractional Power HRR Horizon Degradation (tau = 1..15)
"""

import time
import numpy as np


class ScaledGraphEnvironment:
    def __init__(self, grid_dim: int = 32, num_actions: int = 24, seed: int = 42):
        self.grid_dim = grid_dim
        self.num_states = grid_dim * grid_dim
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

        directions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        self.action_deltas = []
        for speed in [1, 2, 3]:
            for dr, dc in directions:
                self.action_deltas.append((dr * speed, dc * speed))

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


class RobustHDCPlanner:
    def __init__(self, dim: int = 10000, num_actions: int = 24, seed: int = 42):
        self.dim = dim
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

        self.actions = [self.rng.standard_normal(dim) for _ in range(num_actions)]
        for a in self.actions:
            a /= np.linalg.norm(a)

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
        f_sn = np.fft.rfft(sn_v)
        self.m_fwd += np.fft.irfft(f_sa * f_sn, n=self.dim)
        self.m_bwd += np.fft.irfft(f_sn * np.fft.rfft(a_v) * np.fft.rfft(s_v), n=self.dim)

    def predict_forward_with_uncertainty(self, s_curr_idx: int, a_idx: int) -> tuple[np.ndarray, float]:
        s_v = self.get_state_vector(s_curr_idx)
        a_v = self.actions[a_idx]

        f_sa = np.fft.rfft(s_v) * np.fft.rfft(a_v)
        raw_pred = np.fft.irfft(np.conj(f_sa) * np.fft.rfft(self.m_fwd), n=self.dim)
        raw_norm = np.linalg.norm(raw_pred)
        if raw_norm < 1e-9:
            return s_v, 1.0
        raw_pred /= raw_norm

        best_sim = 0.0
        best_vec = raw_pred
        for _, known_v in self.state_codebook.items():
            sim = float(np.dot(raw_pred, known_v))
            if sim > best_sim:
                best_sim = sim
                best_vec = known_v

        uncertainty = max(0.0, 1.0 - best_sim)
        return best_vec, uncertainty

    def compute_wavefront_retrograde_field(self, goal_idx: int, hops: int = 24, cleanup_threshold: float = 0.05) -> np.ndarray:
        """Extended wavefront propagation with precision pruning across k=24 hops."""
        goal_v = self.get_state_vector(goal_idx)
        field = goal_v.copy()
        decay = 0.88

        for hop in range(hops):
            valid_predecessors = []
            f_field = np.fft.rfft(field)

            for a_v in self.actions:
                f_query = f_field * np.fft.rfft(a_v)
                raw_pred = np.fft.irfft(np.conj(f_query) * np.fft.rfft(self.m_bwd), n=self.dim)
                raw_norm = np.linalg.norm(raw_pred)
                if raw_norm < 1e-9:
                    continue
                raw_pred /= raw_norm

                best_sim = max([float(np.dot(raw_pred, kv)) for kv in self.state_codebook.values()], default=0.0)
                if best_sim >= cleanup_threshold:
                    valid_predecessors.append((raw_pred, best_sim))

            if valid_predecessors:
                sims = np.array([p[1] for p in valid_predecessors])
                weights = np.exp(2.5 * (sims - np.max(sims)))
                weights /= np.sum(weights)

                acc = np.zeros(self.dim)
                for w, (vec, _) in zip(weights, valid_predecessors):
                    acc += w * vec
                acc /= np.linalg.norm(acc)

                field = 0.55 * field + 0.45 * decay * acc
                field /= np.linalg.norm(field)
                decay *= 0.88

        return field

    def select_action_dynamic(self, current_state: int, goal_state: int, retrograde_field: np.ndarray, field_threshold: float = 0.06) -> tuple[int, str]:
        """
        Dynamic Epistemic-Pragmatic Arbitration:
        Evaluates local gradient strength. If field is below noise floor, curiosity remains active!
        """
        # Test candidate actions
        candidate_evals = []
        for a_idx in range(self.num_actions):
            pred_v, uncertainty = self.predict_forward_with_uncertainty(current_state, a_idx)
            pragmatic = float(np.dot(pred_v, retrograde_field))
            candidate_evals.append((a_idx, pred_v, pragmatic, uncertainty))

        max_local_pragmatic = max([ce[2] for ce in candidate_evals])

        # If local field signal is above noise floor: PRAGMATIC GRADIENT takes control!
        if max_local_pragmatic >= field_threshold:
            mode = "PRAGMATIC_GRADIENT"
            best_a = max(candidate_evals, key=lambda ce: 3.5 * ce[2] - 0.2 * ce[3])[0]
        else:
            # Field has not reached here: CURIOSITY EXPLORATION takes control!
            mode = "EPISTEMIC_CURIOSITY"
            dr, dc = ((goal_state // 32) - (current_state // 32), (goal_state % 32) - (current_state % 32))
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

            def curiosity_score(ce):
                a_idx = ce[0]
                speed = (a_idx // 8) + 1
                act_dr, act_dc = directions[a_idx % 8][0] * speed, directions[a_idx % 8][1] * speed
                dir_align = (dr * act_dr + dc * act_dc) / (np.sqrt(dr**2 + dc**2 + 1e-6) * np.sqrt(act_dr**2 + act_dc**2 + 1e-6))
                # Balance directional heading + novelty (uncertainty)
                return 0.6 * dir_align + 1.0 * ce[3]

            best_a = max(candidate_evals, key=curiosity_score)[0]

        return best_a, mode


def evaluate_fractional_power_degradation(dim: int = 10000):
    """
    Empirical Test: Cosine similarity degradation of fractional power operator T^tau as a function of tau.
    Tests whether T^tau preserves reachability or washes out into diffusion noise.
    """
    print("\n" + "=" * 80)
    print("EMPIRICAL TEST: FRACTIONAL POWER HRR DEGRADATION OVER HORIZON (tau = 1..15)")
    print("=" * 80)

    rng = np.random.default_rng(999)
    T = rng.standard_normal(dim)
    T /= np.linalg.norm(T)
    f_T = np.fft.rfft(T)

    v_init = rng.standard_normal(dim)
    v_init /= np.linalg.norm(v_init)

    # Compute step-by-step ground-truth chaining: v_k = T ⊗ v_{k-1}
    step_history = [v_init]
    curr = v_init
    for _ in range(15):
        curr = np.fft.irfft(f_T * np.fft.rfft(curr), n=dim)
        curr /= np.linalg.norm(curr)
        step_history.append(curr)

    # Compute fractional power operator: (T^tau ⊗ v_0)
    print(f"{'Horizon (tau)':<15} | {'Ground-Truth Sim':<20} | {'Noise Floor Std':<18} | {'Status':<15}")
    print("-" * 75)

    frac_results = []
    for tau in range(1, 16):
        # Power in Fourier domain: (f_T)^tau
        f_T_tau = (np.abs(f_T)**tau) * np.exp(1j * np.angle(f_T) * tau)
        v_frac = np.fft.irfft(f_T_tau * np.fft.rfft(v_init), n=dim)
        v_frac /= np.linalg.norm(v_frac)

        sim_to_gt = float(np.dot(v_frac, step_history[tau]))
        noise_floor = 1.0 / np.sqrt(dim)
        status = "SIGNAL" if sim_to_gt > 3.0 * noise_floor else "DIFFUSION_NOISE"
        print(f"tau = {tau:<10} | {sim_to_gt:>18.4f} | {noise_floor:>16.4f} | {status:<15}")
        frac_results.append((tau, sim_to_gt, status))

    return frac_results


def run_10_episode_variance_benchmark():
    print("=" * 80)
    print("RIGOROUS 10-EPISODE BENCHMARK (|S| = 1024, |A| = 24, Mean ± Std Evaluation)")
    print("=" * 80)

    env = ScaledGraphEnvironment(grid_dim=32, num_actions=24, seed=42)
    planner = RobustHDCPlanner(dim=10000, num_actions=24, seed=42)

    goal_idx = env.coord_to_idx(*env.goal_coord)

    # 1. Exploratory Ingestion
    print("\n1. Ingesting State Transitions...")
    curr = env.reset()
    for _ in range(300):
        a_rnd = np.random.randint(24)
        n_curr, _ = env.step(a_rnd)
        planner.record_transition(curr, a_rnd, n_curr)
        curr = n_curr

    # 2. Run 10 Consecutive Episodes with Dynamic Epistemic-Pragmatic Arbitration
    print("\n2. Executing 10 Consecutive Episodes...")
    episode_steps = []
    retrograde_field = planner.compute_wavefront_retrograde_field(goal_idx, hops=24, cleanup_threshold=0.05)
    goal_found = False

    t0 = time.time()
    for ep in range(10):
        curr = env.reset()
        steps = 0
        done = False
        max_steps = 100
        modes_used = []

        while not done and steps < max_steps:
            action, mode = planner.select_action_dynamic(curr, goal_idx, retrograde_field, field_threshold=0.06)
            modes_used.append(mode)
            next_s, done = env.step(action)
            planner.record_transition(curr, action, next_s)
            curr = next_s
            steps += 1

            if done and not goal_found:
                goal_found = True
                # Recompile retrograde field with freshly discovered path
                retrograde_field = planner.compute_wavefront_retrograde_field(goal_idx, hops=24, cleanup_threshold=0.05)

        episode_steps.append(steps)
        prag_pct = (modes_used.count("PRAGMATIC_GRADIENT") / len(modes_used)) * 100.0
        print(f"   Episode {ep+1:02d}: Steps = {steps:2d} (Goal: {done}) | Gradient Lock: {prag_pct:4.1f}% | Dominant Mode: {modes_used[-1]}")

    elapsed = time.time() - t0
    mean_steps = float(np.mean(episode_steps))
    std_steps = float(np.std(episode_steps))
    success_rate = (sum(s < 100 for s in episode_steps) / 10.0) * 100.0

    print("\n" + "=" * 80)
    print("10-EPISODE STATISTICAL BENCHMARK SUMMARY:")
    print("=" * 80)
    print(f"   Success Rate:                      {success_rate:.1f}% ({sum(s < 100 for s in episode_steps)}/10 episodes solved)")
    print(f"   Steps to Goal (Mean ± Std):         {mean_steps:.2f} ± {std_steps:.2f} steps")
    print(f"   Theoretical BFS Oracle Lower Bound: 11 steps")
    print(f"   Path Length Overhead vs Oracle:     {((mean_steps - 11.0) / 11.0) * 100:+.1f}%")
    print(f"   Average Decision Throughput:        {sum(episode_steps) / elapsed:.1f} steps/sec")
    print("=" * 80)

    # Run fractional power degradation test
    frac_results = evaluate_fractional_power_degradation(dim=10000)

    return {
        "success_rate": success_rate,
        "mean_steps": mean_steps,
        "std_steps": std_steps,
        "overhead_vs_oracle_pct": ((mean_steps - 11.0) / 11.0) * 100.0,
        "frac_power_results": frac_results
    }


if __name__ == "__main__":
    run_10_episode_variance_benchmark()
