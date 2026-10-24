"""Comprehensive Verification Suite for Upgrade 2 & Upgrade 5:
Living Interoceptive State & Continuous-Time Liquid Self-Healing Engine.

Verifies:
1. InteroceptiveStateTracker:
   - Dynamic tracking of cognitive strain, epistemic entropy, Lyapunov energy, and liquid tau.
   - Viability degradation under stress and restoration under self-healing.
   - Grounded first-person qualitative qualia narrative generation.
2. SelfModelEngine Living Interoception Integration:
   - Live state manifest, system prompt context, and answer_self_query enrichment.
3. LiquidAnticipatoryCore Continuous-Time Neural ODE:
   - RK4 continuous integration dynamics over 256D latent space.
   - Dynamic time-constant adaptation (tau_eff accelerates under high input novelty).
   - Trajectory buffer multi-head continuous attention.
4. Autonomous Lyapunov Homeostatic Self-Healing:
   - Energy barrier breach under severe perturbations (shocks).
   - Restorative gradient descent down -\\nabla_h E(h) pulling state back to attractor basin.
   - Automatic self_healing_events accounting.
5. Full JanePipeline Interconnection:
   - Reasoning convergence updates strain & energy.
   - Global workspace ignition updates epistemic entropy.
   - Anticipatory engine continuous stepping updates self-model interoception.
"""

import torch

from src.reasoning_engine.self_model import SelfModelEngine, InteroceptiveStateTracker
from src.core.anticipatory_engine import AnticipatoryEngine, LiquidAnticipatoryCore, EgoNode
from src.core.pipeline import JanePipeline


# =============================================================================
# 1. InteroceptiveStateTracker Unit Tests
# =============================================================================

def test_interoceptive_state_tracker_baseline():
    """Verify default initial condition of living interoceptive state."""
    tracker = InteroceptiveStateTracker()
    summary = tracker.get_summary()

    assert summary["status"] == "stable_equilibrium"
    assert summary["cognitive_strain"] < 0.15
    assert summary["epistemic_entropy"] < 0.20
    assert summary["homeostatic_energy"] < 0.20
    assert summary["liquid_tau"] > 0.80
    assert summary["self_healing_events"] == 0
    assert summary["internal_viability"] >= 0.85

    narrative = tracker.get_narrative()
    assert "stable homeostatic equilibrium" in narrative
    assert "viability" in narrative


def test_interoceptive_state_tracker_dynamic_adaptation():
    """Verify smooth adaptation under cognitive strain and stress."""
    tracker = InteroceptiveStateTracker()

    # Apply heavy cognitive strain and elevated Lyapunov energy
    tracker.update(cognitive_strain=0.90, homeostatic_energy=1.20, liquid_tau=0.25)
    summary = tracker.get_summary()

    assert summary["cognitive_strain"] > 0.30
    assert summary["homeostatic_energy"] > 0.40
    assert summary["liquid_tau"] < 0.80
    assert summary["internal_viability"] < 0.85

    # Push further into critical perturbation
    for _ in range(5):
        tracker.update(cognitive_strain=0.95, homeostatic_energy=1.40, liquid_tau=0.15)

    crit_summary = tracker.get_summary()
    assert crit_summary["status"] in ("stressed", "critical_rebalancing")
    assert crit_summary["internal_viability"] < 0.60

    # Self-healing restorative event
    tracker.record_healing()
    healed_summary = tracker.get_summary()
    assert healed_summary["self_healing_events"] == 1
    assert healed_summary["internal_viability"] >= 0.70


# =============================================================================
# 2. SelfModelEngine Living Interoception Tests
# =============================================================================

def test_self_model_manifest_and_system_context():
    """Verify live state manifest and system context include living interoception."""
    engine = SelfModelEngine()
    engine.update_interoception(cognitive_strain=0.10, homeostatic_energy=0.15, liquid_tau=0.92)

    manifest = engine.get_live_state_manifest()
    assert "interoception" in manifest
    assert "interoceptive_narrative" in manifest
    assert manifest["interoception"]["status"] == "stable_equilibrium"

    sys_context = engine.get_system_context()
    assert "Current Living Interoceptive State" in sys_context
    assert "homeostatic equilibrium" in sys_context


def test_self_model_answer_self_query_system_status():
    """Verify answer_self_query returns living interoceptive metrics in system_status."""
    engine = SelfModelEngine()
    engine.update_interoception(
        cognitive_strain=0.12,
        epistemic_entropy=0.18,
        homeostatic_energy=0.22,
        liquid_tau=0.88,
        self_healed=False
    )

    state = engine.answer_self_query("What is your current system status and health?")
    assert "interoception" in state
    assert "system_status" in state

    status = state["system_status"]
    assert status["equilibrium"] is True
    assert status["status"] == "operating in stable equilibrium"
    assert "cognitive_strain" in status
    assert "epistemic_entropy" in status
    assert "homeostatic_energy" in status
    assert "liquid_tau" in status
    assert "self_healing_events" in status
    assert "interoceptive_narrative" in status


# =============================================================================
# 3. LiquidAnticipatoryCore Continuous Neural ODE & LTC Tests
# =============================================================================

def test_liquid_anticipatory_core_rk4_continuous_integration():
    """Verify 4th-order Runge-Kutta continuous integration steps latent state."""
    core = LiquidAnticipatoryCore(latent_dim=256, tau_base=1.0, device="cpu")
    initial_h = core.h.clone()

    x_ambient = torch.randn(256) * 0.1
    res = core.step(x_ambient, dt=0.05)

    assert res["h"].shape == (256,)
    assert not torch.allclose(res["h"], initial_h)
    assert res["energy"] >= 0.0
    assert res["tau_eff"] > 0.0
    assert len(core.trajectory_buffer) == 1


def test_liquid_tau_accelerates_on_novel_crisis():
    """Verify liquid time-constant tau_eff contracts under high input flux."""
    core = LiquidAnticipatoryCore(latent_dim=256, tau_base=1.0, device="cpu")

    # Calm ambient input -> tau_eff is close to tau_base (deliberative)
    x_calm = torch.randn(256) * 0.05
    res_calm = core.step(x_calm, dt=0.05)
    tau_calm = res_calm["tau_eff"]

    # Sudden high-amplitude crisis flux -> tau_eff contracts sharply for rapid adaptation
    x_crisis = torch.randn(256) * 3.5
    res_crisis = core.step(x_crisis, dt=0.05)
    tau_crisis = res_crisis["tau_eff"]

    assert tau_crisis < tau_calm
    assert tau_crisis < 0.60  # Rapid accelerated adaptation


# =============================================================================
# 4. Autonomous Lyapunov Homeostatic Self-Healing Tests
# =============================================================================

def test_autonomous_lyapunov_self_healing_under_severe_shock():
    """Verify Lyapunov gradient restoration autonomously heals internal state perturbations."""
    core = LiquidAnticipatoryCore(latent_dim=256, tau_base=1.0, energy_threshold=0.75, device="cpu")

    # Establish initial baseline
    for _ in range(5):
        core.step(torch.zeros(256), dt=0.05)

    assert core.self_healing_events == 0

    # Inject severe internal trauma / noise shock
    core.inject_severe_shock(noise_std=2.8)

    # Execute cognitive cycle: should detect energy violation and execute Lyapunov restoration
    res = core.step(torch.zeros(256), dt=0.05)

    assert res["self_healed"] is True
    assert core.self_healing_events >= 1
    # Restored energy must be substantially lower than perturbed pre-energy
    assert res["energy"] < res["pre_energy"]
    assert res["energy"] < 0.85  # Pulled back towards attractor basin


# =============================================================================
# 5. Full JanePipeline Interconnection Tests
# =============================================================================

def test_anticipatory_engine_updates_self_model_interoception():
    """Verify AnticipatoryEngine steps feed live telemetry into pipeline.self_model."""
    class MockPipeline:
        def __init__(self):
            self.latent_dim = 256
            self.device = torch.device("cpu")
            self.self_model = SelfModelEngine()
            self.episodic_ledger = None
            self.api_gateway = None

    pipe = MockPipeline()
    engine = AnticipatoryEngine(pipeline=pipe, ego_node=EgoNode(user_name="Akhil"))

    # Enqueue nominal stream
    engine.enqueue_stream({"text": "Sensor reading nominal", "metric": "temperature", "value": 40.0})
    engine.step()

    # Verify self_model received interoception update
    summary = pipe.self_model.interoception.get_summary()
    assert summary["internal_viability"] >= 0.80
    assert summary["status"] == "stable_equilibrium"


def test_jane_pipeline_end_to_end_living_interoception():
    """End-to-end test with full JanePipeline: reasoning and introspection."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=False, device="cpu")

    # 1. Execute conversational self-query
    res = pipeline.process("Jane, what is your current internal system status and energy?")
    assert "response_text" in res
    assert "interoception" in res
    assert res["interoception"]["status"] == "stable_equilibrium"

    # 2. Check anticipatory engine liquid core is active and healthy
    liquid_state = pipeline.anticipatory_engine.get_liquid_state()
    assert liquid_state["h"].shape == (256,)
    assert liquid_state["self_healing_events"] == 0

    # 3. Inject internal shock to pipeline anticipatory engine
    pipeline.anticipatory_engine.inject_internal_shock(noise_std=3.0)
    # Step anticipatory engine
    pipeline.anticipatory_engine.enqueue_stream({"text": "Telemetry ping", "metric": "tick", "value": 1.0})
    pipeline.anticipatory_engine.step()

    # Verify self-healing engaged
    after_shock_state = pipeline.anticipatory_engine.get_liquid_state()
    assert after_shock_state["self_healing_events"] >= 1

    # 4. Check that self_model reflected the self-healing event
    updated_manifest = pipeline.self_model.get_live_state_manifest()
    assert updated_manifest["interoception"]["self_healing_events"] >= 1
