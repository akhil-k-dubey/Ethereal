"""
Thesis 2: Predictive Processing Hierarchy (PPH)
Architecture: Hierarchical Predictive Coding with Meta-Predictive Self-Modeling (Clark/Friston)

Components:
- 4-Tier Hierarchy:
  * L0: Sensory Surface
  * L1: Spatial/Temporal Features
  * L2: Abstract Latent Causes
  * L3: Meta-Predictor (Predicts the system's own internal prediction errors)
- Top-down predictive cascades & bottom-up precision-weighted prediction error propagation.
"""

import time
import json
import numpy as np


class PPHLayer:
    def __init__(self, in_dim: int, out_dim: int, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        # Generative weights (predicting lower layer)
        self.W = self.rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / (in_dim + out_dim))
        self.state = np.zeros(out_dim)
        self.precision = np.ones(in_dim)  # Inverse variance / attention

    def predict_down(self) -> np.ndarray:
        return np.tanh(self.W @ self.state)

    def update_state(self, error_own: np.ndarray, error_below: np.ndarray, lr: float = 0.1):
        # Gradient ascent on joint log-evidence:
        # State driven by bottom-up prediction error from layer below, opposed by top-down error
        grad = (self.W.T @ (self.precision * error_below)) - error_own
        self.state += lr * grad
        self.state = np.clip(self.state, -5.0, 5.0)


class PredictiveProcessingHierarchy:
    def __init__(self, dims: list[int] = [64, 128, 64, 32], seed: int = 101):
        self.dims = dims
        self.layers = [
            PPHLayer(dims[i], dims[i+1], seed=seed + i)
            for i in range(len(dims) - 1)
        ]
        # Meta-prediction layer: predicts the magnitude of prediction error at L2
        self.meta_w = np.random.default_rng(seed + 99).standard_normal(dims[-1]) * 0.1
        self.meta_predicted_error = 0.0

    def step(self, sensory_input: np.ndarray, num_inference_steps: int = 10, lr: float = 0.12) -> dict:
        error_history = []
        meta_accuracy_history = []

        for step in range(num_inference_steps):
            # 1. Top-down predictions
            preds = [layer.predict_down() for layer in self.layers]

            # 2. Bottom-up prediction errors
            errors = []
            # Error at sensory layer (L0)
            errors.append(sensory_input - preds[0])
            # Errors at intermediate layers
            for i in range(len(self.layers) - 1):
                errors.append(self.layers[i].state - preds[i+1])
            # Error at top layer (prior is zero)
            errors.append(-self.layers[-1].state * 0.1)

            total_vfe = sum(np.sum(e**2) for e in errors)
            error_history.append(total_vfe)

            # Meta-prediction of system's own surprise
            actual_top_error = np.mean(errors[-2]**2)
            meta_pred = float(np.dot(self.layers[-1].state, self.meta_w)**2)
            meta_acc = 1.0 - abs(actual_top_error - meta_pred) / (actual_top_error + 1e-6)
            meta_accuracy_history.append(max(0.0, meta_acc))

            # 3. State updates (belief propagation)
            for i in range(len(self.layers)):
                self.layers[i].update_state(errors[i+1], errors[i], lr=lr)

        # Meta-predictor online adaptation
        self.meta_w += 0.05 * (actual_top_error - meta_pred) * self.layers[-1].state

        return {
            "initial_free_energy": error_history[0],
            "final_free_energy": error_history[-1],
            "vfe_reduction_pct": ((error_history[0] - error_history[-1]) / (error_history[0] + 1e-9)) * 100.0,
            "meta_prediction_accuracy": float(np.mean(meta_accuracy_history[-3:])),
            "layers_count": len(self.layers)
        }


def benchmark_pph():
    print("=" * 60)
    print("BENCHMARKING THESIS 2: PREDICTIVE PROCESSING HIERARCHY")
    print("=" * 60)

    pph = PredictiveProcessingHierarchy(dims=[128, 256, 128, 64], seed=2026)
    rng = np.random.default_rng(42)

    # Stream of structured sensory observations
    base_pattern = rng.standard_normal(128)
    bench_results = []
    t0 = time.time()

    for trial in range(20):
        # Structured input with periodic drift
        noise = rng.standard_normal(128) * 0.2
        sensory = np.tanh(base_pattern * np.sin(trial * 0.3) + noise)
        metrics = pph.step(sensory, num_inference_steps=12, lr=0.1)
        bench_results.append(metrics)
        if (trial + 1) % 5 == 0:
            print(f"   Trial {trial+1:02d} | VFE: {metrics['initial_free_energy']:.3f} -> {metrics['final_free_energy']:.3f} (Red: {metrics['vfe_reduction_pct']:.1f}%) | Meta-Acc: {metrics['meta_prediction_accuracy']*100:.1f}%")

    elapsed = time.time() - t0
    avg_reduction = np.mean([m['vfe_reduction_pct'] for m in bench_results])
    avg_meta_acc = np.mean([m['meta_prediction_accuracy'] for m in bench_results])

    print(f"\nSummary Thesis 2 (PPH):")
    print(f"   Mean Free Energy Reduction: {avg_reduction:.2f}%")
    print(f"   Meta-Prediction (Self-Surprise Awareness): {avg_meta_acc*100:.2f}%")
    print(f"   Throughput: {20/elapsed:.2f} inferences/sec")

    data = {
        "thesis": "Thesis_02_PPH",
        "avg_vfe_reduction_pct": float(avg_reduction),
        "meta_self_prediction_accuracy": float(avg_meta_acc),
        "throughput_inf_sec": float(20/elapsed)
    }
    with open("benchmark_results/thesis_02_pph.json", "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    benchmark_pph()
