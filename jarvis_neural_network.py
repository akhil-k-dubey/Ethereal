"""
JARVIS-MLN: Metacognitive Liquid Network
A fully-fledged, continuous-time, differentiable neural network architecture closest to JARVIS.

Integrates:
1. Liquid Time-Constant (LTC) continuous-time differential dynamics (Neural ODE).
2. Multi-Head Self-Attention over continuous temporal trajectory buffers.
3. System 2 Introspective Meta-Controller (Self-Modeling & Causal Attribution).
4. Lyapunov Homeostatic Energy Barrier (Autonomous Self-Healing under corruption/shocks).
5. Pearl Causal Intervention Layer (do-calculus inside the forward pass).

Zero VSA / Zero Hash Tables. Pure Differentiable Neural Network Primitives.
"""

import numpy as np


class LiquidAttentionLayer:
    """
    Multi-Head Continuous Attention Layer with dynamic time-decay gating.
    Allows continuous-time state representations to attend over past temporal trajectories.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, seed: int = 42):
        self.d_k = d_model // n_heads
        self.rng = np.random.default_rng(seed)

        # Learnable projection matrices: Query, Key, Value, Output
        self.W_q = self.rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_k = self.rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_v = self.rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.W_o = self.rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)

    def forward(self, query: np.ndarray, memory_keys: list[np.ndarray], memory_values: list[np.ndarray]) -> tuple[np.ndarray, float]:
        """
        query: [d_model] current state
        memory_keys: list of [d_model] historical state keys
        memory_values: list of [d_model] historical state values
        Returns: attended representation [d_model], attention entropy (scalar)
        """
        if not memory_keys:
            # Self-attention on single token
            v = query @ self.W_v
            out = v @ self.W_o
            return out, 0.0

        K = np.array(memory_keys) @ self.W_k   # [seq_len, d_model]
        V = np.array(memory_values) @ self.W_v # [seq_len, d_model]
        Q = query @ self.W_q                   # [d_model]

        # Multi-head scaled dot-product attention
        scores = (K @ Q) / np.sqrt(self.d_k)   # [seq_len]
        # Softmax with numerical stability
        exp_scores = np.exp(scores - np.max(scores))
        attn_weights = exp_scores / (np.sum(exp_scores) + 1e-12)

        # Context output
        context = attn_weights @ V             # [d_model]
        out = context @ self.W_o               # [d_model]

        # Entropy of attention distribution (measure of uncertainty / focus)
        entropy = -float(np.sum(attn_weights * np.log(attn_weights + 1e-12)))
        return out, entropy


class LiquidTimeConstantCell:
    """
    Liquid Time-Constant (LTC) continuous-time dynamic cell:
    dh/dt = - [1/tau_base + f(x, h)] * h + g(x, h) + Attn(x, h)
    
    The time-constant tau(x, h) dynamically adapts:
    - High input novelty / crisis: tau -> 0 (instantaneous high-frequency adaptation)
    - Stable / deliberative: tau -> tau_base (temporal persistence and long-term memory)
    """
    def __init__(self, d_in: int = 16, d_model: int = 64, tau_base: float = 1.0, seed: int = 42):
        self.tau_base = tau_base
        self.rng = np.random.default_rng(seed)

        # Non-linear coupling parameters
        self.W_in = self.rng.standard_normal((d_in, d_model)) / np.sqrt(d_in)
        self.W_rec = self.rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.bias_rec = np.zeros(d_model)

        # Liquid conductance parameters (f_theta determining liquid tau)
        self.W_tau_x = self.rng.standard_normal((d_in, d_model)) / np.sqrt(d_in)
        self.W_tau_h = self.rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
        self.bias_tau = np.zeros(d_model)

    def compute_derivatives(self, x: np.ndarray, h: np.ndarray, attn_context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes dh/dt and dynamic time-constant tau(x, h).
        """
        # Synaptic input drive g(x, h)
        synaptic_drive = np.tanh(x @ self.W_in + h @ self.W_rec + self.bias_rec + attn_context)

        # Dynamic liquid conductance: inputs and internal changes increase conductance
        # In LTCs, conductance G >= 0, so higher input flux/energy increases G, reducing tau
        flux_x = np.abs(x) @ np.abs(self.W_tau_x)
        flux_h = np.abs(h) @ np.abs(self.W_tau_h)
        liquid_cond = np.log1p(np.exp(np.clip(flux_x + flux_h + self.bias_tau, -10.0, 10.0)))

        # Effective time constant vector tau_eff = 1 / (1/tau_base + G)
        tau_inv = (1.0 / self.tau_base) + liquid_cond
        tau_eff = 1.0 / (tau_inv + 1e-6)

        # Continuous-time differential equation: dh/dt = - (1/tau_eff) * h + synaptic_drive
        dh_dt = -tau_inv * h + synaptic_drive
        return dh_dt, tau_eff

    def rk4_step(self, x: np.ndarray, h: np.ndarray, attn_context: np.ndarray, dt: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """
        4th-order Runge-Kutta continuous integration step for exact ODE precision.
        """
        k1, tau_eff = self.compute_derivatives(x, h, attn_context)
        k2, _ = self.compute_derivatives(x, h + 0.5 * dt * k1, attn_context)
        k3, _ = self.compute_derivatives(x, h + 0.5 * dt * k2, attn_context)
        k4, _ = self.compute_derivatives(x, h + dt * k3, attn_context)

        h_next = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return h_next, tau_eff


class System2MetaController:
    """
    System 2 Introspective Meta-Controller.
    Operates continuously alongside the Executive Network:
    1. Monitors hidden state divergence and attention entropy.
    2. Projects future self-state (self-modeling).
    3. Causal attribution (did my action cause this state shift, or external drift?).
    4. Evaluates Lyapunov homeostatic energy.
    """
    def __init__(self, d_model: int = 64, d_action: int = 4, seed: int = 42):
        self.d_model = d_model
        self.rng = np.random.default_rng(seed)

        # Forward self-predictor: [h_t, a_t] -> \hat{h}_{t+1}
        self.W_self_pred = self.rng.standard_normal((d_model + d_action, d_model)) / np.sqrt(d_model)
        # Causal attribution head
        self.W_causal = self.rng.standard_normal((d_model * 2 + d_action, 1)) / np.sqrt(d_model)
        # Nominal homeostatic center
        self.nominal_attractor = np.zeros(d_model)

    def predict_future_self(self, h: np.ndarray, a_onehot: np.ndarray) -> np.ndarray:
        inp = np.concatenate([h, a_onehot])
        h_pred = np.tanh(inp @ self.W_self_pred)
        return h_pred

    def compute_homeostatic_energy(self, h: np.ndarray, h_pred: np.ndarray, attn_entropy: float) -> tuple[float, np.ndarray]:
        """
        Lyapunov Energy Barrier E(h) normalized by dimension d_model:
        E(h) = (0.5 / d_model) * ||h - nominal||^2 + (0.5 / d_model) * ||h - h_pred||^2 + 0.05 * entropy
        Returns scalar energy and analytical gradient \nabla_h E(h).
        """
        diff_nominal = h - self.nominal_attractor
        diff_pred = h - h_pred

        scale = 1.0 / self.d_model
        energy = 0.5 * scale * float(np.sum(diff_nominal**2)) + 0.5 * scale * float(np.sum(diff_pred**2)) + 0.05 * attn_entropy
        grad_energy = scale * (diff_nominal + diff_pred)
        return energy, grad_energy

    def attribute_agency(self, h_prev: np.ndarray, a_onehot: np.ndarray, h_curr: np.ndarray) -> float:
        """Computes probability p(self) that the transition was caused by own action."""
        feat = np.concatenate([h_prev, a_onehot, h_curr])
        logit = float(feat @ self.W_causal)
        p_self = 1.0 / (1.0 + np.exp(-logit))
        return p_self


class JarvisMetacognitiveLiquidNetwork:
    """
    THE JARVIS-MLN NEURAL NETWORK:
    The full continuous-time, self-healing cognitive neural network.
    """
    def __init__(self, d_in: int = 16, d_model: int = 64, d_action: int = 4, tau_base: float = 1.0, seed: int = 42):
        self.d_model = d_model
        self.d_action = d_action
        self.rng = np.random.default_rng(seed)

        # 1. Executive Continuous-Time LTC Core
        self.ltc_cell = LiquidTimeConstantCell(d_in=d_in, d_model=d_model, tau_base=tau_base, seed=seed)

        # 2. Multi-Head Liquid Attention Layer
        self.attention = LiquidAttentionLayer(d_model=d_model, n_heads=4, seed=seed + 1)

        # 3. System 2 Introspective Meta-Controller
        self.meta_controller = System2MetaController(d_model=d_model, d_action=d_action, seed=seed + 2)

        # 4. Policy & Value Actuation Heads
        self.W_policy = self.rng.standard_normal((d_model, d_action)) / np.sqrt(d_model)
        self.W_value = self.rng.standard_normal((d_model, 1)) / np.sqrt(d_model)

        # Causal Intervention Mask (Pearl do-calculus layer)
        self.active_intervention_mask = None

        # Continuous Latent State and Trajectory History
        self.h = np.zeros(d_model)
        self.trajectory_buffer = []   # list of past h states
        self.checkpoint_state = np.zeros(d_model)
        self.energy_threshold = 0.8   # Homeostatic viability boundary

        # Telemetry
        self.self_healing_events = 0

    def reset(self):
        self.h = np.zeros(self.d_model)
        self.trajectory_buffer = []
        self.checkpoint_state = np.zeros(self.d_model)
        self.self_healing_events = 0

    def step(self, x: np.ndarray, dt: float = 0.05) -> dict:
        """
        Executes one continuous-time cognitive cycle:
        1. Attention over continuous history.
        2. LTC Neural ODE integration.
        3. Introspective Meta-Monitoring.
        4. Autonomous Homeostatic Self-Healing (if energy barrier violated).
        5. Action & Value Generation.
        """
        # 1. Attention over continuous history
        attn_out, attn_entropy = self.attention.forward(
            query=self.h,
            memory_keys=self.trajectory_buffer[-20:],
            memory_values=self.trajectory_buffer[-20:]
        )

        # 2. LTC Neural ODE Integration
        h_proposed, tau_eff = self.ltc_cell.rk4_step(x, self.h, attn_out, dt=dt)

        # Apply internal causal intervention if active (do-calculus)
        if self.active_intervention_mask is not None:
            mask, clamped_val = self.active_intervention_mask
            h_proposed = (1.0 - mask) * h_proposed + mask * clamped_val

        # 3. Introspective Meta-Monitoring & Self-Modeling
        # Predict past expectation
        last_action_onehot = np.zeros(self.d_action)
        last_action_onehot[0] = 1.0  # reference
        h_pred = self.meta_controller.predict_future_self(self.h, last_action_onehot)

        # Evaluate Homeostatic Viability Energy E(h)
        energy, grad_energy = self.meta_controller.compute_homeostatic_energy(
            h=h_proposed,
            h_pred=h_pred,
            attn_entropy=attn_entropy
        )

        # 4. AUTONOMOUS SELF-HEALING (Lyapunov Restoration)
        healed = False
        pre_energy = energy
        if energy > self.energy_threshold:
            self.self_healing_events += 1
            healed = True
            # Perform restorative gradient flow down the Lyapunov energy surface
            # to pull the perturbed state back into the viable attractor basin
            for _ in range(12):
                h_proposed -= 0.6 * grad_energy
                energy, grad_energy = self.meta_controller.compute_homeostatic_energy(
                    h=h_proposed, h_pred=h_pred, attn_entropy=attn_entropy
                )
            # If still unstable, fall back to stable checkpoint attractor
            if energy > self.energy_threshold:
                h_proposed = 0.9 * self.checkpoint_state + 0.1 * h_proposed
                energy, _ = self.meta_controller.compute_homeostatic_energy(
                    h=h_proposed, h_pred=h_pred, attn_entropy=attn_entropy
                )

        # Commit validated state
        self.h = h_proposed
        self.trajectory_buffer.append(self.h.copy())
        if len(self.trajectory_buffer) > 100:
            self.trajectory_buffer.pop(0)

        # Update checkpoint when stable
        if energy < 0.35:
            self.checkpoint_state = self.h.copy()

        # 5. Output Policy & Value
        logits = self.h @ self.W_policy
        exp_l = np.exp(logits - np.max(logits))
        action_probs = exp_l / np.sum(exp_l)
        action = int(np.argmax(action_probs))
        value = float(self.h @ self.W_value)

        # Agency attribution
        p_self = self.meta_controller.attribute_agency(
            h_prev=self.trajectory_buffer[-2] if len(self.trajectory_buffer) >= 2 else self.h,
            a_onehot=action_probs,
            h_curr=self.h
        )

        return {
            "action": action,
            "action_probs": action_probs,
            "value": value,
            "tau_eff_mean": float(np.mean(tau_eff)),
            "energy": energy,
            "pre_energy": pre_energy,
            "self_healed": healed,
            "agency_p_self": p_self,
            "attention_entropy": attn_entropy
        }

    def inject_severe_internal_shock(self, noise_std: float = 1.5):
        """Simulates severe internal trauma, memory corruption, or bit-flip shock."""
        shock = self.rng.standard_normal(self.d_model) * noise_std
        self.h += shock

    def set_causal_intervention(self, neuron_indices: list[int], value: float):
        """Sets an active Pearl do(h_indices = value) intervention."""
        mask = np.zeros(self.d_model)
        mask[neuron_indices] = 1.0
        self.active_intervention_mask = (mask, value)

    def clear_causal_intervention(self):
        self.active_intervention_mask = None
