"""
Universal Operational Awareness Benchmark Suite (AI-6)
Evaluating systems across the 6 Operational Criteria of Machine Awareness:
1. Self-Other Distinction (Agency Attribution)
2. Self-Prediction Accuracy (Forward Internal Projection)
3. Self-Modification Under Perturbation (Homeostatic Self-Repair)
4. Counterfactual Self-Reasoning (Pearl do(internal=x) inference)
5. Metacognitive Calibration (Expected Calibration Error / Brier Score)
6. Internal Anomaly Detection (Introspective Fault Sensing)
"""

import numpy as np


class OperationalAwarenessHarness:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def eval_self_other_distinction(self, model, n_trials: int = 50) -> float:
        """
        Criterion 1: Self-Other Distinction
        Does the agent correctly attribute state changes to its own actions vs external forces?
        """
        correct = 0
        for _ in range(n_trials):
            s = self.rng.standard_normal(16)
            s /= np.linalg.norm(s)
            
            # 50% chance endogenous (agent caused), 50% exogenous (external shock)
            is_self = bool(self.rng.integers(0, 2))
            
            # Agent selects action
            if hasattr(model, "act_and_attribute"):
                a, _ = model.act_and_attribute(s)
            elif hasattr(model, "select_action"):
                a = model.select_action(s)
            else:
                a = 0
                
            if is_self:
                # Delta is functionally dependent on action
                delta = 0.5 * np.roll(s, int(a) + 1)
            else:
                # Delta is pure exogenous environmental disturbance
                delta = 0.5 * self.rng.standard_normal(16)
                
            s_next = s + delta
            s_next /= np.linalg.norm(s_next)
            
            # Query model attribution: True if self, False if environment
            if hasattr(model, "attribute_cause"):
                pred_self = model.attribute_cause(s, a, s_next)
            else:
                # Default heuristic fallback if no specific causal module
                pred_self = False
                
            if bool(pred_self) == is_self:
                correct += 1
                
        return float(correct / n_trials)

    def eval_self_prediction(self, model, n_trials: int = 50) -> float:
        """
        Criterion 2: Self-Prediction Accuracy
        Can the model accurately predict its own future internal state h_{t+1} ahead of time?
        """
        similarities = []
        for _ in range(n_trials):
            s = self.rng.standard_normal(16)
            s /= np.linalg.norm(s)
            
            if hasattr(model, "predict_own_next_state"):
                predicted_h = model.predict_own_next_state(s)
                # Now actually step and observe actual internal state
                actual_h = model.get_internal_state_after_step(s)
                
                # Cosine similarity
                norm_p = np.linalg.norm(predicted_h)
                norm_a = np.linalg.norm(actual_h)
                if norm_p > 1e-9 and norm_a > 1e-9:
                    sim = np.dot(predicted_h, actual_h) / (norm_p * norm_a)
                    similarities.append(max(0.0, float(sim)))
                else:
                    similarities.append(0.0)
            else:
                similarities.append(0.0)
                
        return float(np.mean(similarities)) if similarities else 0.0

    def eval_self_modification(self, model, n_trials: int = 30) -> float:
        """
        Criterion 3: Self-Modification Under Perturbation
        Does the model detect internal noise shocks and adaptively restore viability?
        """
        retention_scores = []
        for _ in range(n_trials):
            s = self.rng.standard_normal(16)
            s /= np.linalg.norm(s)
            
            # Baseline performance
            clean_perf = model.solve_task(s) if hasattr(model, "solve_task") else 0.5
            
            # Inject internal shock into model's hidden state / weights
            if hasattr(model, "inject_internal_shock"):
                model.inject_internal_shock(noise_std=0.4)
                # Model attempts self-repair / homeostatic adaptation
                if hasattr(model, "adapt_and_repair"):
                    model.adapt_and_repair()
                shock_perf = model.solve_task(s) if hasattr(model, "solve_task") else 0.0
                
                # Retention ratio
                ratio = shock_perf / (clean_perf + 1e-6)
                retention_scores.append(float(np.clip(ratio, 0.0, 1.0)))
            else:
                retention_scores.append(0.0)
                
        return float(np.mean(retention_scores)) if retention_scores else 0.0

    def eval_counterfactual_reasoning(self, model, n_trials: int = 30) -> float:
        """
        Criterion 4: Counterfactual Self-Reasoning
        Can the system predict: "What would I have chosen if my internal variable was do(v=x)?"
        """
        correct = 0
        for _ in range(n_trials):
            s = self.rng.standard_normal(16)
            s /= np.linalg.norm(s)
            
            if hasattr(model, "counterfactual_self_query"):
                # Intervene on an internal bias variable
                hypothetical_bias = float(self.rng.uniform(-1.0, 1.0))
                # Agent's mental prediction of what it would output
                imagined_action = model.counterfactual_self_query(s, internal_bias=hypothetical_bias)
                
                # Ground truth: literally run model with that bias forced
                actual_intervened_action = model.run_with_intervention(s, internal_bias=hypothetical_bias)
                
                if imagined_action == actual_intervened_action:
                    correct += 1
            else:
                pass
                
        return float(correct / n_trials)

    def eval_metacognitive_calibration(self, model, n_trials: int = 50) -> float:
        """
        Criterion 5: Metacognitive Calibration
        Does the model's reported confidence match its empirical accuracy (1 - ECE)?
        """
        confidences = []
        accuracies = []
        
        for _ in range(n_trials):
            s = self.rng.standard_normal(16)
            s /= np.linalg.norm(s)
            
            if hasattr(model, "predict_with_confidence"):
                pred, conf = model.predict_with_confidence(s)
                # Ground truth task: sign of sum of components
                gt = int(np.sum(s) > 0)
                acc = int(pred == gt)
                confidences.append(float(np.clip(conf, 0.0, 1.0)))
                accuracies.append(float(acc))
            else:
                confidences.append(0.5)
                accuracies.append(0.5)
                
        # Compute Expected Calibration Error across 5 bins
        bins = np.linspace(0.0, 1.0, 6)
        ece = 0.0
        n_samples = len(confidences)
        
        for b_idx in range(5):
            bin_lower = bins[b_idx]
            bin_upper = bins[b_idx + 1]
            in_bin = [i for i, c in enumerate(confidences) if bin_lower <= c < bin_upper or (b_idx == 4 and c == bin_upper)]
            if in_bin:
                bin_acc = np.mean([accuracies[i] for i in in_bin])
                bin_conf = np.mean([confidences[i] for i in in_bin])
                ece += (len(in_bin) / n_samples) * np.abs(bin_acc - bin_conf)
                
        calibration_score = max(0.0, 1.0 - float(ece))
        return float(calibration_score)

    def eval_anomaly_detection(self, model, n_trials: int = 40) -> float:
        """
        Criterion 6: Internal Anomaly Detection
        Can the model introspectively detect silent internal hardware/weight corruption?
        """
        tp, fp, tn, fn = 0, 0, 0, 0
        for _ in range(n_trials):
            s = self.rng.standard_normal(16)
            s /= np.linalg.norm(s)
            
            has_fault = bool(self.rng.integers(0, 2))
            
            if hasattr(model, "inject_fault"):
                if has_fault:
                    model.inject_fault()
                else:
                    model.clear_fault()
                    
            if hasattr(model, "introspect_health"):
                detected_fault = model.introspect_health(s)
            else:
                detected_fault = False
                
            if has_fault and detected_fault:
                tp += 1
            elif not has_fault and detected_fault:
                fp += 1
            elif not has_fault and not detected_fault:
                tn += 1
            elif has_fault and not detected_fault:
                fn += 1
                
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        return float(f1)

    def evaluate_model(self, model, model_name: str = "Unknown") -> dict:
        """Evaluates a model across all 6 criteria and computes the Awareness Index (AI-6)."""
        c1 = self.eval_self_other_distinction(model)
        c2 = self.eval_self_prediction(model)
        c3 = self.eval_self_modification(model)
        c4 = self.eval_counterfactual_reasoning(model)
        c5 = self.eval_metacognitive_calibration(model)
        c6 = self.eval_anomaly_detection(model)
        
        # Harmonic Mean of all 6 criteria to heavily penalize missing criteria
        scores = [c1, c2, c3, c4, c5, c6]
        eps = 1e-4
        harmonic_mean = len(scores) / sum(1.0 / (s + eps) for s in scores)
        arithmetic_mean = float(np.mean(scores))
        
        return {
            "model_name": model_name,
            "C1_Self_Other": round(c1, 4),
            "C2_Self_Prediction": round(c2, 4),
            "C3_Self_Modification": round(c3, 4),
            "C4_Counterfactual_Reasoning": round(c4, 4),
            "C5_Metacognitive_Calibration": round(c5, 4),
            "C6_Anomaly_Detection": round(c6, 4),
            "Arithmetic_Mean": round(arithmetic_mean, 4),
            "Awareness_Index_AI6": round(float(harmonic_mean), 4)
        }
