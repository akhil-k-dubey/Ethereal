"""Proactive Anticipatory Intelligence Engine (The J.A.R.V.I.S. Loop) for Project Jane.

Transforms Jane from a purely reactive assistant into a proactive intelligence:
1. Continuous Ingestion & Telemetry Monitoring:
   Monitors streaming background data (system health, thermal metrics, sensor feeds, news, preprints).
2. Manifold Projection:
   Projects incoming streams into the 256D continuous concept space.
3. Latent Anomaly Detection:
   Identifies critical threshold violations, high energy states, and epistemic discrepancies.
4. Proactive Initiation (Speaking First):
   Without waiting to be addressed, Jane initiates conversation using her Ego Node (personality),
   alerting Akhil to anomalies and proposing concrete SMT-verified mitigation actions.
"""

import time
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Callable, Union

from src.utils.logger import get_logger
from src.core.api_gateway import ActionASTNode, APIGateway

logger = get_logger("anticipatory_engine")


class EgoNode:
    """
    Personality-grounded Ego Node for Jane's autonomous proactivity.
    Embodies Jane's self-awareness, identity, and relational stance toward Akhil.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        user_name: str = "Akhil",
        personality_vector: Optional[torch.Tensor] = None,
        traits: Optional[Dict[str, float]] = None,
        tom_manifold: Optional[Any] = None
    ):
        self.latent_dim = latent_dim
        self.user_name = user_name
        self.tom_manifold = tom_manifold
        self.traits = traits or {
            "proactivity": 0.95,
            "scientific_rigor": 0.98,
            "protectiveness": 0.92,
            "loyalty": 1.0,
            "calmness": 0.88
        }
        if personality_vector is not None:
            self.personality_vector = personality_vector
        else:
            torch.manual_seed(1337)
            vec = torch.randn(latent_dim)
            self.personality_vector = vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)
        self.state_label: str = "Self_Observing"

    def format_proactive_alert(self, anomaly_summary: str, recommendation: Optional[str] = None) -> str:
        """Phrases a proactive alert addressed directly to Akhil."""
        rec = f" {recommendation}" if recommendation else " Shall I take corrective measures?"
        return f"{self.user_name}, I am detecting an anomaly: {anomaly_summary}.{rec}"



class PipelineTranslatorAdapter:
    """Adapts JanePipeline translator (Ear / Mouth / Encoder) for AnticipatoryEngine."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline

    def encode(self, data_stream: Any) -> torch.Tensor:
        """Projects incoming stream text or dict into a 256D latent state."""
        if isinstance(data_stream, dict):
            text_rep = data_stream.get("text") or data_stream.get("summary") or str(data_stream)
        else:
            text_rep = str(data_stream)

        if hasattr(self.pipeline, "ear") and self.pipeline.ear is not None:
            try:
                return self.pipeline.ear(text_rep)
            except Exception:
                pass

        if hasattr(self.pipeline, "encoder") and self.pipeline.encoder is not None:
            try:
                tokens = self.pipeline.tokenizer.encode(text_rep)
                token_ids = [tok.token_id for tok in tokens]
                t = torch.tensor([token_ids], dtype=torch.long, device=self.pipeline.device)
                seq = self.pipeline.encoder(t)
                return self.pipeline.encoder.pool_state(seq)
            except Exception:
                pass

        # Fallback orthonormal projection
        torch.manual_seed(abs(hash(text_rep)) % (2**31))
        vec = torch.randn(self.pipeline.latent_dim, device=self.pipeline.device)
        return vec / (torch.norm(vec, p=2, dim=-1, keepdim=True) + 1e-8)

    def generate(self, alert_vector: torch.Tensor, personality_vector: Optional[torch.Tensor] = None) -> str:
        """Generates proactive verbalization addressing Akhil."""
        user_name = "Akhil"
        if hasattr(self.pipeline, "self_model") and self.pipeline.self_model:
            user_name = self.pipeline.self_model.user_profile.get("name", "Akhil")

        return f"{user_name}, I am detecting an anomaly in the monitored systems. Shall I vent the cooling systems?"


class PipelineReasonerAdapter:
    """Adapts JanePipeline reasoning and energy landscape for AnticipatoryEngine."""

    def __init__(self, pipeline: Any, energy_threshold: float = 0.85):
        self.pipeline = pipeline
        self.energy_threshold = energy_threshold
        from src.reasoning_engine.intent_manifold import TrueIntentManifold
        self.intent_manifold = TrueIntentManifold.get_instance()

    def detect_anomaly(self, latent_state: torch.Tensor, stream_data: Optional[Any] = None) -> bool:
        """Evaluates whether the latent state or stream data indicates an anomaly."""
        # 1. Explicit telemetry anomaly check
        if isinstance(stream_data, dict):
            if stream_data.get("is_anomaly") or stream_data.get("alert"):
                return True

            # Schema with {"metric": "...", "value": ...}
            metric = str(stream_data.get("metric", "")).lower()
            val = stream_data.get("value")
            if isinstance(val, (int, float)):
                fval = float(val)
                if ("temp" in metric) and fval > 85.0:
                    return True
                if ("press" in metric) and fval > 10.0:
                    return True
                if ("cpu" in metric or "mem" in metric) and fval > 90.0:
                    return True
                if ("power" in metric) and fval > 100.0:
                    return True
                if fval > 100.0:
                    return True

            # Direct key schema
            temp = stream_data.get("temperature") or stream_data.get("temp")
            if isinstance(temp, (int, float)) and float(temp) > 85.0:
                return True
            mem = stream_data.get("memory_percent") or stream_data.get("memory")
            if isinstance(mem, (int, float)) and float(mem) > 90.0:
                return True
            pressure = stream_data.get("pressure")
            if isinstance(pressure, (int, float)) and float(pressure) > 10.0:
                return True
            power = stream_data.get("power")
            if isinstance(power, (int, float)) and float(power) > 100.0:
                return True

        # 2. Continuous vector anomaly check
        text_str = str(stream_data).lower()
        if self.intent_manifold.similarity(text_str, "emergency") >= 0.45:
            return True

        # 3. Energy landscape check in 256D manifold
        if hasattr(self.pipeline, "energy_landscape") and self.pipeline.energy_landscape is not None:
            try:
                energy = self.pipeline.energy_landscape.compute_energy(latent_state)
                if float(energy.mean().item()) > self.energy_threshold:
                    return True
            except Exception:
                pass

        return False

    def resolve_anomaly(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Relaxes the anomalous latent state towards an equilibrium recovery vector."""
        if hasattr(self.pipeline, "energy_landscape") and self.pipeline.energy_landscape is not None:
            try:
                settled, _ = self.pipeline.energy_landscape.settle_equilibrium(latent_state, steps=5)
                return settled
            except Exception:
                pass
        return F.normalize(latent_state * 0.5, p=2, dim=-1)


class PipelineIngestionAdapter:
    """Adapts ingestion queues and telemetry feeds for AnticipatoryEngine."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.queue: List[Any] = []


    def get_latest_stream(self) -> Optional[Any]:
        if self.queue:
            return self.queue.pop(0)
        return None


class LiquidAnticipatoryCore(nn.Module):
    """
    Continuous-Time Liquid Neural ODE & Lyapunov Homeostatic Self-Healing Core.
    
    1. Liquid Time-Constant (LTC) continuous dynamics:
       dh/dt = - [1/tau_base + G(x, h)] * h + tanh(W_in x + W_rec h + b + attn_context)
       tau_eff = 1 / (1/tau_base + G(x, h))
       Integrated using 4th-order Runge-Kutta (RK4) continuous ODE stepping.
       
    2. Multi-Head Continuous Attention over rolling temporal trajectory buffers.
    
    3. System 2 Introspective Meta-Controller & Lyapunov Energy Barrier:
       E(h) = (0.5 / d) * ||h - nominal||^2 + (0.5 / d) * ||h - h_pred||^2 + 0.05 * entropy
       
    4. Autonomous Self-Healing (Lyapunov Restoration):
       When E(h) > energy_threshold, performs gradient descent along -\nabla_h E(h)
       to autonomously pull the disturbed hidden state back into the nominal attractor basin.
       If corruption is extreme, engages stable checkpoint attractor fallback.
    """
    def __init__(
        self,
        latent_dim: int = 256,
        tau_base: float = 1.0,
        energy_threshold: float = 0.80,
        n_heads: int = 4,
        device: Union[str, torch.device] = "cpu"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.tau_base = tau_base
        self.energy_threshold = energy_threshold
        self.device = torch.device(device)
        self.n_heads = n_heads
        self.d_k = max(1, latent_dim // n_heads)

        # Continuous LTC Cell Parameters
        torch.manual_seed(42)
        scale = 1.0 / (latent_dim ** 0.5)
        self.W_in = nn.Parameter(torch.randn(latent_dim, latent_dim) * scale)
        self.W_rec = nn.Parameter(torch.randn(latent_dim, latent_dim) * scale)
        self.bias_rec = nn.Parameter(torch.zeros(latent_dim))

        self.W_tau_x = nn.Parameter(torch.randn(latent_dim, latent_dim) * scale)
        self.W_tau_h = nn.Parameter(torch.randn(latent_dim, latent_dim) * scale)
        self.bias_tau = nn.Parameter(torch.zeros(latent_dim))

        # Multi-Head Attention Parameters
        self.W_q = nn.Parameter(torch.randn(latent_dim, latent_dim) * scale)
        self.W_k = nn.Parameter(torch.randn(latent_dim, latent_dim) * scale)
        self.W_v = nn.Parameter(torch.randn(latent_dim, latent_dim) * scale)
        self.W_o = nn.Parameter(torch.randn(latent_dim, latent_dim) * scale)

        # Meta-Controller Predictive Head: [h, h] -> h_pred
        self.W_self_pred = nn.Parameter(torch.randn(latent_dim * 2, latent_dim) * scale)
        self.nominal_attractor = nn.Parameter(torch.zeros(latent_dim), requires_grad=False)

        # Dynamic State Buffers
        self.h = torch.zeros(latent_dim, device=self.device)
        self.checkpoint_state = torch.zeros(latent_dim, device=self.device)
        self.trajectory_buffer: List[torch.Tensor] = []
        self.self_healing_events: int = 0
        self.step_counter: int = 0

        self.to(self.device)

    def _compute_attention(self, query: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Multi-head attention over rolling trajectory buffer."""
        if not self.trajectory_buffer:
            return torch.zeros(self.latent_dim, device=self.device), 0.0

        keys_stacked = torch.stack(self.trajectory_buffer[-20:]).to(self.device)
        values_stacked = keys_stacked

        K = keys_stacked @ self.W_k
        V = values_stacked @ self.W_v
        Q = query @ self.W_q

        scores = (K @ Q) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=0)

        context = attn_weights @ V
        out = context @ self.W_o
        entropy = -float(torch.sum(attn_weights * torch.log(attn_weights + 1e-12)).item())
        return out, entropy

    def compute_derivatives(self, x: torch.Tensor, h: torch.Tensor, attn_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates continuous dh/dt and dynamic time-constant vector tau_eff."""
        synaptic_drive = torch.tanh(x @ self.W_in + h @ self.W_rec + self.bias_rec + attn_context)

        flux_x = torch.abs(x) @ torch.abs(self.W_tau_x)
        flux_h = torch.abs(h) @ torch.abs(self.W_tau_h)
        liquid_cond = torch.log1p(torch.exp(torch.clamp(flux_x + flux_h + self.bias_tau, -10.0, 10.0)))

        tau_inv = (1.0 / self.tau_base) + liquid_cond
        tau_eff = 1.0 / (tau_inv + 1e-6)

        dh_dt = -tau_inv * h + synaptic_drive
        return dh_dt, tau_eff

    def rk4_step(self, x: torch.Tensor, h: torch.Tensor, attn_context: torch.Tensor, dt: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """4th-order Runge-Kutta numerical integration for continuous Neural ODE dynamics."""
        k1, tau_eff = self.compute_derivatives(x, h, attn_context)
        k2, _ = self.compute_derivatives(x, h + 0.5 * dt * k1, attn_context)
        k3, _ = self.compute_derivatives(x, h + 0.5 * dt * k2, attn_context)
        k4, _ = self.compute_derivatives(x, h + dt * k3, attn_context)

        h_next = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return h_next, tau_eff

    def compute_homeostatic_energy(self, h: torch.Tensor, h_pred: torch.Tensor, entropy: float) -> Tuple[float, torch.Tensor]:
        """Lyapunov Energy Barrier E(h) and exact analytical gradient \\nabla_h E(h)."""
        diff_nominal = h - self.nominal_attractor
        diff_pred = h - h_pred
        scale = 1.0 / self.latent_dim

        energy = 0.5 * scale * float(torch.sum(diff_nominal ** 2).item()) + \
                 0.5 * scale * float(torch.sum(diff_pred ** 2).item()) + \
                 0.05 * entropy

        grad_energy = scale * (diff_nominal + diff_pred)
        return energy, grad_energy

    @torch.no_grad()
    def step(self, x: torch.Tensor, dt: float = 0.05) -> Dict[str, Any]:
        """Executes one continuous cognitive cycle with autonomous Lyapunov self-healing."""
        self.step_counter += 1
        x = x.to(self.device)
        if x.dim() > 1:
            x = x.squeeze(0)

        # 1. Temporal attention
        attn_out, entropy = self._compute_attention(self.h)

        # 2. Continuous LTC RK4 integration
        h_proposed, tau_eff = self.rk4_step(x, self.h, attn_out, dt=dt)

        # 3. System 2 Self-prediction & Lyapunov homeostatic evaluation
        pair = torch.cat([self.h, self.h])
        h_pred = torch.tanh(pair @ self.W_self_pred)
        energy, grad_energy = self.compute_homeostatic_energy(h_proposed, h_pred, entropy)

        # 4. Autonomous Self-Healing (Lyapunov Restoration)
        healed = False
        pre_energy = energy
        if energy > self.energy_threshold:
            healed = True
            self.self_healing_events += 1
            # Gradient flow down the Lyapunov energy surface
            for _ in range(12):
                h_proposed = h_proposed - 0.6 * grad_energy
                energy, grad_energy = self.compute_homeostatic_energy(h_proposed, h_pred, entropy)

            # Fall back to stable checkpoint attractor if needed
            if energy > self.energy_threshold:
                h_proposed = 0.85 * self.checkpoint_state + 0.15 * h_proposed
                energy, _ = self.compute_homeostatic_energy(h_proposed, h_pred, entropy)

        # Commit validated state
        self.h = h_proposed.detach()
        self.trajectory_buffer.append(self.h.clone())
        if len(self.trajectory_buffer) > 50:
            self.trajectory_buffer.pop(0)

        # Update stable checkpoint attractor
        if energy < 0.35:
            self.checkpoint_state = self.h.clone()

        return {
            "h": self.h,
            "energy": energy,
            "pre_energy": pre_energy,
            "tau_eff": float(torch.mean(tau_eff).item()),
            "self_healed": healed,
            "entropy": entropy
        }

    def inject_severe_shock(self, noise_std: float = 2.0):
        """Simulates internal perturbation, thermal shock, or memory noise."""
        with torch.no_grad():
            shock = torch.randn(self.latent_dim, device=self.device) * noise_std
            self.h = self.h + shock

    def reset(self):
        self.h = torch.zeros(self.latent_dim, device=self.device)
        self.checkpoint_state = torch.zeros(self.latent_dim, device=self.device)
        self.trajectory_buffer.clear()
        self.self_healing_events = 0
        self.step_counter = 0


class AnticipatoryEngine:
    """
    Proactive Anticipatory Intelligence Loop for Project Jane.
    Runs constantly in the background, monitoring data streams, projecting them
    into the 256D manifold, detecting anomalies, and initiating conversation with Akhil.
    """

    def __init__(
        self,
        pipeline: Any,
        ego_node: Optional[Any] = None,
        check_interval_sec: float = 10.0,
        alert_callback: Optional[Callable[[str], None]] = None,
        api_gateway: Optional[Any] = None
    ):
        self.pipeline = pipeline
        self.check_interval_sec = check_interval_sec
        self.alert_callback = alert_callback
        self.api_gateway = api_gateway or getattr(pipeline, "api_gateway", None)
        self.latent_dim = getattr(pipeline, "latent_dim", 256)
        self.device = getattr(pipeline, "device", torch.device("cpu"))

        # 0. Ego Node Personality
        if ego_node is not None:
            self.ego_node = ego_node
        elif hasattr(pipeline, "ego_node") and pipeline.ego_node is not None:
            self.ego_node = pipeline.ego_node
        else:
            user_name = "Akhil"
            if hasattr(pipeline, "self_model") and pipeline.self_model:
                user_name = pipeline.self_model.user_profile.get("name", "Akhil")
            self.ego_node = EgoNode(
                latent_dim=self.latent_dim,
                user_name=user_name
            )

        # Continuous-Time Liquid Neural ODE & Self-Healing Core
        self.liquid_core = LiquidAnticipatoryCore(
            latent_dim=self.latent_dim,
            tau_base=1.0,
            energy_threshold=0.80,
            device=self.device
        )

        # 1. Adapters for pipeline components if not natively provided
        if not hasattr(self.pipeline, "translator"):
            self.pipeline.translator = PipelineTranslatorAdapter(self.pipeline)
        if not hasattr(self.pipeline, "reasoner"):
            self.pipeline.reasoner = PipelineReasonerAdapter(self.pipeline)
        if not hasattr(self.pipeline, "ingestion_daemon"):
            self.pipeline.ingestion_daemon = PipelineIngestionAdapter(self.pipeline)

        # Thread management
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._pending_streams: List[Any] = []
        self._detected_anomalies: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def enqueue_stream(self, data: Any):
        """Allows external systems, background tasks, or tests to inject real-time data."""
        with self._lock:
            self._pending_streams.append(data)

    def get_latest_stream(self) -> Optional[Any]:
        """Retrieves the latest available stream event."""
        with self._lock:
            if self._pending_streams:
                return self._pending_streams.pop(0)

        if hasattr(self.pipeline, "ingestion_daemon") and self.pipeline.ingestion_daemon is not None:
            if hasattr(self.pipeline.ingestion_daemon, "get_latest_stream"):
                return self.pipeline.ingestion_daemon.get_latest_stream()

        return None

    def _encode_stream(self, data_stream: Any) -> torch.Tensor:
        """Encodes stream data via pipeline translator into 256D continuous space."""
        if hasattr(self.pipeline, "translator") and hasattr(self.pipeline.translator, "encode"):
            return self.pipeline.translator.encode(data_stream)
        elif hasattr(self.pipeline, "ear"):
            return self.pipeline.ear(str(data_stream))
        torch.manual_seed(42)
        return torch.randn(getattr(self.pipeline, "latent_dim", 256))

    def _detect_anomaly(self, latent_state: torch.Tensor, stream_data: Optional[Any] = None) -> bool:
        """Evaluates whether the latent state or stream data indicates an anomaly."""
        if hasattr(self.pipeline, "reasoner") and hasattr(self.pipeline.reasoner, "detect_anomaly"):
            try:
                # Support both detect_anomaly(latent_state) and detect_anomaly(latent_state, stream_data)
                return self.pipeline.reasoner.detect_anomaly(latent_state, stream_data)
            except TypeError:
                return self.pipeline.reasoner.detect_anomaly(latent_state)
        return False

    def _resolve_anomaly(self, latent_state: torch.Tensor) -> torch.Tensor:
        """Resolves the anomaly into a target alert/recovery vector."""
        if hasattr(self.pipeline, "reasoner") and hasattr(self.pipeline.reasoner, "resolve_anomaly"):
            return self.pipeline.reasoner.resolve_anomaly(latent_state)
        return F.normalize(latent_state, p=2, dim=-1)

    def _generate_alert(
        self,
        alert_vector: torch.Tensor,
        personality_vector: Optional[torch.Tensor] = None,
        data_stream: Optional[Any] = None
    ) -> str:
        """Generates proactive verbalization addressing Akhil."""
        # 1. Custom personality phrasing via EgoNode
        summary = ""
        if isinstance(data_stream, dict):
            metric = data_stream.get("metric") or "system telemetry"
            val = data_stream.get("value", "")
            unit = data_stream.get("unit", "")
            summary = f"{metric} reaching {val} {unit}".strip()
            if not val:
                summary = data_stream.get("summary") or data_stream.get("text") or "unexpected threshold breach"
        elif isinstance(data_stream, str):
            summary = data_stream
        else:
            summary = "monitored system parameters"

        # Check if pipeline translator generate is provided
        if hasattr(self.pipeline, "translator") and hasattr(self.pipeline.translator, "generate"):
            try:
                gen_text = self.pipeline.translator.generate(alert_vector, personality_vector)
                if gen_text and len(gen_text) > 5:
                    return gen_text
            except Exception as e:
                logger.debug("Translator generate fallback: %s", e)

        # Standard EgoNode verbalization
        user_name = getattr(self.ego_node, "user_name", "Akhil")
        if hasattr(self.ego_node, "format_proactive_alert"):
            return self.ego_node.format_proactive_alert(
                anomaly_summary=summary,
                recommendation="Shall I vent the cooling systems or execute containment?"
            )

        return f"{user_name}, I am detecting an anomaly in {summary}. Shall I vent the cooling systems?"

    def _update_pipeline_interoception(self, liquid_res: Dict[str, Any], is_anomaly: bool):
        """Pushes live neural ODE and homeostatic metrics into pipeline.self_model."""
        if hasattr(self.pipeline, "self_model") and self.pipeline.self_model:
            strain = 0.85 if is_anomaly else min(0.35, liquid_res.get("energy", 0.1) * 0.4)
            if hasattr(self.pipeline.self_model, "update_interoception"):
                self.pipeline.self_model.update_interoception(
                    cognitive_strain=strain,
                    epistemic_entropy=liquid_res.get("entropy", 0.1),
                    homeostatic_energy=liquid_res.get("energy", 0.15),
                    liquid_tau=liquid_res.get("tau_eff", 0.95),
                    self_healed=liquid_res.get("self_healed", False)
                )

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Executes a continuous-time anticipatory monitoring step:
        1. Evolve continuous Liquid Neural ODE state (LTC cell with RK4 integration).
        2. Evaluate Lyapunov homeostatic energy barrier and execute autonomous self-healing.
        3. Project incoming data stream into the 256D manifold.
        4. Detect anomalies via manifold energy or telemetry thresholds.
        5. Push living interoceptive metrics into pipeline.self_model.
        6. PROACTIVE INITIATION: Jane speaks first when anomalies or high energy are detected.
        """
        data_stream = self.get_latest_stream()
        if data_stream is None:
            # Idle continuous ambient drift
            idle_x = torch.zeros(self.latent_dim, device=self.device)
            liquid_res = self.liquid_core.step(idle_x, dt=0.05)
            self._update_pipeline_interoception(liquid_res, is_anomaly=False)
            return None

        # 2. Map data into 256D manifold
        latent_state = self._encode_stream(data_stream)
        if latent_state.dim() > 1:
            latent_state_flat = latent_state.squeeze(0)
        else:
            latent_state_flat = latent_state

        # Evolve continuous LTC dynamics
        liquid_res = self.liquid_core.step(latent_state_flat, dt=0.05)

        # 3. Check if the data violates any thresholds in the manifold or high Lyapunov energy
        is_anomaly = self._detect_anomaly(latent_state, data_stream) or (liquid_res["pre_energy"] > 1.25)

        # Feed living interoceptive telemetry to self_model
        self._update_pipeline_interoception(liquid_res, is_anomaly=is_anomaly)

        if is_anomaly:
            # 4. Proactive initiation
            alert_vector = self._resolve_anomaly(latent_state)
            personality_vec = getattr(self.ego_node, "personality_vector", None)
            response = self._generate_alert(alert_vector, personality_vec, data_stream)

            # Announce proactively
            print(f"[JANE PROACTIVE]: {response}")
            if self.alert_callback:
                try:
                    self.alert_callback(response)
                except Exception as e:
                    logger.error("Error invoking alert_callback: %s", e)

            # Record in episodic memory ledger if present
            if hasattr(self.pipeline, "episodic_ledger") and self.pipeline.episodic_ledger:
                self.pipeline.episodic_ledger.record_event(
                    action_type="PROACTIVE_ALERT",
                    concept_name="AnticipatoryAnomaly",
                    source_name="AnticipatoryEngine",
                    formula_or_axiom="manifold_energy > threshold",
                    status="COMMITTED",
                    details={
                        "alert": response,
                        "data_stream": str(data_stream)[:200],
                        "lyapunov_energy": liquid_res["energy"],
                        "self_healed": liquid_res["self_healed"],
                        "liquid_tau": liquid_res["tau_eff"]
                    }
                )

            record = {
                "timestamp": time.time(),
                "data_stream": data_stream,
                "response": response,
                "alert_vector_norm": float(torch.norm(alert_vector).item()),
                "liquid_telemetry": {
                    "energy": liquid_res["energy"],
                    "pre_energy": liquid_res["pre_energy"],
                    "tau_eff": liquid_res["tau_eff"],
                    "self_healed": liquid_res["self_healed"],
                    "entropy": liquid_res["entropy"]
                }
            }
            with self._lock:
                self._detected_anomalies.append(record)
            return record

        return None

    def propose_mitigation_action(self, anomaly_record: Dict[str, Any]) -> ActionASTNode:
        """
        Formulates a formal, Z3-verifiable action to mitigate the detected anomaly.
        Ready to be passed directly to APIGateway.execute_verified_action.
        """
        stream = anomaly_record.get("data_stream", {})
        if isinstance(stream, dict) and "temp" in str(stream).lower():
            return ActionASTNode(
                action_type="HARDWARE_CONTROL",
                payload={"subsystem": "cooling", "valve": "vent_1", "state": "OPEN", "power": 40.0},
                safety_contract="energy_out <= energy_in",
                metadata={"reason": "thermal_mitigation"}
            )
        elif isinstance(stream, dict) and "memory" in str(stream).lower():
            return ActionASTNode(
                action_type="CODE_EXEC",
                payload="import gc; gc.collect(); cleaned = True",
                safety_contract="energy_out <= energy_in",
                metadata={"reason": "memory_cleanup"}
            )
        else:
            return ActionASTNode(
                action_type="HARDWARE_CONTROL",
                payload={"command": "EMERGENCY_VENT", "threshold": 0.5},
                safety_contract="energy_out <= energy_in",
                metadata={"reason": "generic_anomaly_mitigation"}
            )

    def run_background_monitor(self):
        """Runs constantly, looking for things to bring to Akhil's attention."""
        while not self._stop_event.is_set():
            try:
                self.step()
            except Exception as e:
                logger.error("Error in anticipatory background loop: %s", e, exc_info=True)
            self._stop_event.wait(self.check_interval_sec)

    def start(self):
        """Starts the background monitor daemon thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self.run_background_monitor,
            name="JaneAnticipatoryMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("AnticipatoryEngine background monitor started (Interval: %.1fs).", self.check_interval_sec)

    def stop(self, timeout: float = 2.0):
        """Stops the background monitor thread gracefully."""
        self._stop_event.set()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=timeout)
        self._monitor_thread = None
        logger.info("AnticipatoryEngine background monitor stopped.")

    @property
    def is_running(self) -> bool:
        """Returns True if the background thread is currently active."""
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def get_detected_anomalies(self) -> List[Dict[str, Any]]:
        """Returns history of detected anomalies."""
        with self._lock:
            return list(self._detected_anomalies)


    def get_liquid_state(self) -> Dict[str, Any]:
        """Returns the current continuous-time latent state and Lyapunov telemetry."""
        return {
            "h": self.liquid_core.h,
            "self_healing_events": self.liquid_core.self_healing_events,
            "step_counter": self.liquid_core.step_counter,
            "trajectory_buffer_len": len(self.liquid_core.trajectory_buffer)
        }

    def inject_internal_shock(self, noise_std: float = 2.0):
        """Injects a severe perturbation into the continuous-time cognitive state."""
        self.liquid_core.inject_severe_shock(noise_std=noise_std)
