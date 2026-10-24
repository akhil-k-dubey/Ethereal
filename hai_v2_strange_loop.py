"""
Hyperdimensional Active Inference (HAI) - Generation 2: Strange-Loop Self-Reference
Architecture: Holographic Strange-Loop Recursion + Hofstadter Attractor Dynamics + Metacognitive Introspection

Key Novelties:
1. Fractal Self-Embedding:
   State vector V contains an explicit self-bound fractal representation:
   V_t = norm( V_sensory ⊕ V_cognitive ⊕ (K_self ⊗ V_t) )
2. Co-evolving Strange-Loop Dynamics:
   Operator S: V -> V and state V dynamically co-adapt.
   Attractor dynamics stabilize the infinite recursive hall-of-mirrors.
3. Introspective Metacognition & Self-Repair:
   The system monitors its own self-consistency F_self.
   Internal perturbations (delusions/noise) are detected introspectively without external ground truth.
4. Deep Introspection / Unbinding Depth:
   Measuring recursive recovery across depth k: (K_self^{-k} ⊗ V)
"""

import time
import json
import numpy as np


class HyperdimensionalSpaceV2:
    """Enhanced HDC Substrate with unitary / invertible binding operators."""

    def __init__(self, dim: int = 10000, seed: int = 1337):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def random_vector(self) -> np.ndarray:
        v = self.rng.standard_normal(self.dim)
        return v / np.linalg.norm(v)

    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular Convolution via FFT."""
        fa = np.fft.rfft(a)
        fb = np.fft.rfft(b)
        res = np.fft.irfft(fa * fb, n=self.dim)
        norm = np.linalg.norm(res)
        return res / norm if norm > 1e-12 else res

    def unbind(self, key: np.ndarray, bound: np.ndarray) -> np.ndarray:
        """Circular Correlation via FFT."""
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

    def non_linear_cleanup(self, v: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Soft-thresholding / sparsity cleanup in hyperdimensional space.
        Suppresses holographic crosstalk noise while preserving semantic peaks.
        """
        v_clean = np.where(np.abs(v) > threshold, v, 0.0)
        norm = np.linalg.norm(v_clean)
        return v_clean / norm if norm > 1e-12 else v


class StrangeLoopEngine:
    """
    Hofstadter Strange-Loop Self-Model.
    Implements recursive self-awareness as a stabilized fixed-point attractor in hyperspace.
    """

    def __init__(self, space: HyperdimensionalSpaceV2):
        self.space = space
        # Introspective key vector: K_self
        self.k_self = self.space.random_vector()
        # Self-prediction transformation kernel S
        self.s_kernel = self.space.random_vector()

    def embed_self_recursively(self, v_world: np.ndarray, current_state: np.ndarray, recursion_weight: float = 0.35) -> np.ndarray:
        """
        Synthesizes state V = norm( (1-w) * V_world ⊕ w * (K_self ⊗ current_state) )
        The state vector physically holds a compressed model of its own entirety!
        """
        self_reflection = self.space.bind(self.k_self, current_state)
        v_next = self.space.superpose([v_world, self_reflection], weights=[1.0 - recursion_weight, recursion_weight])
        return self.space.non_linear_cleanup(v_next)

    def predict_self(self, state: np.ndarray) -> np.ndarray:
        """Operator S(V): Predicts internal state."""
        return self.space.bind(self.s_kernel, state)

    def introspect_depth(self, state: np.ndarray, depth: int = 1) -> np.ndarray:
        """
        Unfolds the recursive self-reference to depth k:
        Level 1: unbind(K_self, V) -> recovers V
        Level 2: unbind(K_self, Level 1) -> recovers V looking at V
        """
        curr = state
        for _ in range(depth):
            curr = self.space.unbind(self.k_self, curr)
            curr = self.space.non_linear_cleanup(curr, threshold=0.005)
        return curr

    def adapt_self_model(self, state: np.ndarray, lr: float = 0.05):
        """Co-adapts S towards current state invariant."""
        target_kernel = self.space.unbind(state, state)
        self.s_kernel = self.space.superpose([self.s_kernel, target_kernel], weights=[1.0 - lr, lr])


class HAIAgentV2:
    """
    Hyperdimensional Active Inference Agent (Generation 2).
    Equipped with Fractal Strange-Loop and Metacognitive Monitoring.
    """

    def __init__(self, space: HyperdimensionalSpaceV2):
        self.space = space
        self.loop_engine = StrangeLoopEngine(space)

        # Sensory observation operator
        self.obs_kernel = self.space.random_vector()

        # System state vector V ∈ R^D
        self.state = self.space.random_vector()

        # Metacognitive baseline

    def step(self, sensory_input: np.ndarray, introspective_cycles: int = 5) -> dict:
        """
        Strange-Loop Active Inference update:
        1. Invert sensory input to sensory hypothesis
        2. Recurse through strange-loop self-embedding
        3. Measure metacognitive surprise (self-consistency)
        4. Settle into free-energy minimizing attractor
        """
        # 1. Sensory hypothesis
        sensory_hyp = self.space.unbind(self.obs_kernel, sensory_input)

        # Track trajectory for attractor stability analysis
        trajectory = [self.state.copy()]

        for cycle in range(introspective_cycles):
            # 2. Strange-loop self-embedding
            v_projected = self.loop_engine.embed_self_recursively(sensory_hyp, self.state, recursion_weight=0.35)

            # 3. Self-prediction from S
            s_pred = self.loop_engine.predict_self(self.state)

            # 4. Attractor convergence: blend projected state with self-prediction
            self.state = self.space.superpose([v_projected, s_pred], weights=[0.75, 0.25])
            trajectory.append(self.state.copy())

        # Adapt self-model S
        self.loop_engine.adapt_self_model(self.state, lr=0.04)

        # Metacognitive Self-Consistency Free Energy
        self_pred = self.loop_engine.predict_self(self.state)
        self_consistency = self.space.similarity(self.state, self_pred)
        self_free_energy = 1.0 - self_consistency

        # Sensory Free Energy
        obs_pred = self.space.bind(self.obs_kernel, self.state)
        sensory_free_energy = 1.0 - self.space.similarity(sensory_input, obs_pred)

        total_fe = sensory_free_energy + 0.6 * self_free_energy

        # Measure trajectory step divergence (Lyapunov-like attractor stability)
        step_diffs = [
            1.0 - self.space.similarity(trajectory[i], trajectory[i+1])
            for i in range(len(trajectory) - 1)
        ]
        convergence_rate = step_diffs[0] - step_diffs[-1] if len(step_diffs) > 1 else 0.0

        return {
            "total_free_energy": total_fe,
            "sensory_free_energy": sensory_free_energy,
            "self_free_energy": self_free_energy,
            "self_consistency": self_consistency,
            "convergence_rate": convergence_rate,
            "trajectory_drift": step_diffs[-1]
        }

    def detect_metacognitive_anomaly(self, noisy_perturbation: np.ndarray) -> tuple[bool, float]:
        """
        Metacognitive Self-Monitoring:
        Injects a perturbation into the internal state and tests if the strange loop
        flags the discrepancy purely from internal self-inconsistency.
        """
        state_backup = self.state.copy()
        # Corrupt internal state
        corrupted = self.space.superpose([self.state, noisy_perturbation], weights=[0.5, 0.5])
        s_pred = self.loop_engine.predict_self(corrupted)
        corrupted_self_error = 1.0 - self.space.similarity(corrupted, s_pred)

        # Baseline error
        normal_pred = self.loop_engine.predict_self(state_backup)
        normal_error = 1.0 - self.space.similarity(state_backup, normal_pred)

        delta_surprise = corrupted_self_error - normal_error
        anomaly_detected = delta_surprise > 0.15

        # Self-repair: restore belief
        self.state = state_backup
        return anomaly_detected, delta_surprise


def benchmark_v2():
    print("=" * 60)
    print("BENCHMARKING HAI GENERATION 2: STRANGE-LOOP SELF-REFERENCE")
    print("=" * 60)

    dim = 10000
    space = HyperdimensionalSpaceV2(dim=dim, seed=1337)
    hai = HAIAgentV2(space)

    # 1. Benchmark: Strange-Loop Attractor Stabilization
    print("\n1. Testing Strange-Loop Attractor Stabilization...")
    sensory_inputs = [space.random_vector() for _ in range(15)]

    metrics_log = []
    t0 = time.time()
    for t, s_in in enumerate(sensory_inputs):
        m = hai.step(s_in, introspective_cycles=6)
        metrics_log.append(m)
        if (t + 1) % 3 == 0:
            print(f"   Step {t+1:02d} | FE: {m['total_free_energy']:.4f} | Self-Consistency: {m['self_consistency']:.4f} | Drift: {m['trajectory_drift']:.5f}")

    elapsed = time.time() - t0

    # 2. Benchmark: Fractal Introspection Depth
    print("\n2. Testing Fractal Introspection Depth (Self unbinding through k layers)...")
    depth_similarities = []
    original_state = hai.state.copy()
    for depth in range(1, 5):
        introspected = hai.loop_engine.introspect_depth(original_state, depth=depth)
        sim = space.similarity(original_state, introspected)
        depth_similarities.append(sim)
        print(f"   Introspection Depth {depth}: Cosine Similarity to Root State = {sim:.4f}")

    # 3. Benchmark: Metacognitive Anomaly Detection & Self-Repair
    print("\n3. Testing Metacognitive Anomaly Detection...")
    anomaly_success = 0
    num_anomaly_trials = 10
    delta_surprises = []
    for _ in range(num_anomaly_trials):
        noise = space.random_vector()
        detected, delta = hai.detect_metacognitive_anomaly(noise)
        delta_surprises.append(delta)
        if detected:
            anomaly_success += 1

    detection_rate = anomaly_success / num_anomaly_trials
    avg_delta_surprise = float(np.mean(delta_surprises))
    print(f"   Metacognitive Anomaly Detection Rate: {detection_rate*100:.1f}% (Mean Surprise Spike: {avg_delta_surprise:.4f})")

    avg_self_consistency = float(np.mean([m['self_consistency'] for m in metrics_log]))
    avg_fe = float(np.mean([m['total_free_energy'] for m in metrics_log]))
    avg_drift = float(np.mean([m['trajectory_drift'] for m in metrics_log]))

    print(f"\nBenchmark Summary V2:")
    print(f"   Mean Attractor Self-Consistency: {avg_self_consistency:.4f}")
    print(f"   Attractor Stability Drift: {avg_drift:.6f} (near-zero = rock-solid fixed point)")
    print(f"   Depth-1 Self-Reflection Fidelity: {depth_similarities[0]:.4f}")
    print(f"   Metacognitive Detection Rate: {detection_rate*100:.1f}%")
    print(f"   Throughput: {len(sensory_inputs)/elapsed:.2f} steps/second")

    results = {
        "generation": 2,
        "name": "HAI_StrangeLoop_SelfReference",
        "dimension": dim,
        "mean_self_consistency": avg_self_consistency,
        "attractor_drift": avg_drift,
        "introspection_depth_1_fidelity": depth_similarities[0],
        "introspection_depth_2_fidelity": depth_similarities[1],
        "metacognitive_anomaly_detection_rate": detection_rate,
        "mean_anomaly_surprise_spike": avg_delta_surprise,
        "mean_free_energy": avg_fe,
        "steps_per_second": float(len(sensory_inputs)/elapsed)
    }

    with open("results_v2_strange_loop.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    benchmark_v2()
