"""Comprehensive Verification Suite for Upgrade 3 & Upgrade 4:
Strange-Loop Self-Intervention & Subjective Autobiographical Continuity.

Verifies:
1. StrangeLoopController:
   - DNC memory tape checkpointing at reasoning steps.
   - Contradiction and paradox detection (energy divergence, Z3 falsification, critic violations).
   - Counterfactual DNC memory tape rollback and causal intervention vector repair.
   - Commitment to EpisodicLedger and interoceptive strain updates.
2. AutobiographicalNarrativeEngine:
   - Autobiographical moments recording (turn index, interoceptive snapshot, project anchor).
   - Dynamic first-person autobiographical narrative synthesis.
   - Chronesthesia (mental time travel): past vs. present internal state comparison.
3. End-to-End JanePipeline Integration:
   - Multi-step reasoning with strange-loop intervention.
   - Conversational turns automatically weaved into the narrative stream.
   - Mental time travel query accessible via pipeline.mental_time_travel.
"""

import time
import torch

from src.reasoning_engine.strange_loop import StrangeLoopController, StrangeLoopInterventionRecord
from src.reasoning_engine.autobiographical_stream import AutobiographicalNarrativeEngine
from src.reasoning_engine.self_model import SelfModelEngine
from src.reasoning_engine.episodic_ledger import EpisodicLedger
from src.core.pipeline import JanePipeline


# =============================================================================
# 1. StrangeLoopController Unit Tests
# =============================================================================

def test_strange_loop_tape_checkpointing():
    """Verify DNC tape checkpointing saves and retrieves snapshots accurately."""
    controller = StrangeLoopController(latent_dim=256)
    controller.clear_checkpoints()

    z0 = torch.randn(256)
    m0 = torch.zeros(1, 128, 64)
    controller.record_checkpoint(step=0, latent_state=z0, memory_tape=m0, energy=0.10)

    z1 = torch.randn(256)
    m1 = torch.ones(1, 128, 64) * 0.5
    controller.record_checkpoint(step=1, latent_state=z1, memory_tape=m1, energy=0.25)

    assert len(controller.checkpoints) == 2
    assert controller.checkpoints[0].step == 0
    assert controller.checkpoints[1].step == 1
    assert torch.allclose(controller.checkpoints[0].latent_state, z0)
    assert torch.allclose(controller.checkpoints[1].memory_tape, m1)


def test_strange_loop_contradiction_detection():
    """Verify detection of energy spikes, barrier violations, and Z3 contradictions."""
    controller = StrangeLoopController(latent_dim=256, energy_divergence_threshold=1.80)

    # 1. Nominal energy -> No contradiction
    assert controller.detect_contradiction(current_energy=0.35, prev_energy=0.30) is None

    # 2. Extreme energy barrier violation
    contra_energy = controller.detect_contradiction(current_energy=2.10, prev_energy=0.50)
    assert contra_energy == "ENERGY_BARRIER_VIOLATION"

    # 3. Sudden Lyapunov energy spike
    contra_spike = controller.detect_contradiction(current_energy=1.60, prev_energy=0.40)
    assert contra_spike == "LYAPUNOV_ENERGY_SPIKE"

    # 4. Formal Z3 mathematical contradiction
    contra_z3 = controller.detect_contradiction(current_energy=0.30, prev_energy=0.30, z3_is_valid=False)
    assert contra_z3 == "Z3_SMT_CONTRADICTION"

    # 5. Critic identity denial
    contra_critic = controller.detect_contradiction(
        current_energy=0.30,
        prev_energy=0.30,
        critic_violations=["Stateless identity denial: claimed to have no memory"]
    )
    assert contra_critic is not None
    assert "CRITIC_VIOLATION" in contra_critic


def test_strange_loop_counterfactual_rollback_and_repair(tmp_path):
    """Verify counterfactual rollback restores pre-fault tape and directs latent state away from paradox."""
    log_file = tmp_path / "test_ledger.jsonl"
    ledger = EpisodicLedger(log_path=str(log_file))
    self_model = SelfModelEngine()

    controller = StrangeLoopController(
        latent_dim=256,
        episodic_ledger=ledger,
        self_model=self_model
    )

    # Establish stable checkpoint at step 1
    z_stable = torch.ones(256) * 0.1
    m_stable = torch.ones(1, 128, 64) * 0.2
    controller.record_checkpoint(step=1, latent_state=z_stable, memory_tape=m_stable, energy=0.20)

    # Introduce corrupted state at step 2 that causes paradox
    z_corrupt = z_stable + torch.randn(256) * 1.5

    # Execute counterfactual rollback
    repaired_z, restored_m, target_ckpt = controller.execute_counterfactual_rollback(
        fault_step=2,
        contradiction_type="ENERGY_BARRIER_VIOLATION",
        current_state=z_corrupt
    )

    assert target_ckpt.step == 1
    assert restored_m is not None
    assert torch.allclose(restored_m, m_stable)
    # Repaired latent state should be normalized and distinct from corrupt state
    assert abs(torch.norm(repaired_z).item() - 1.0) < 1e-4
    assert not torch.allclose(repaired_z, z_corrupt)

    # Record intervention and check ledger + self_model updates
    rec = StrangeLoopInterventionRecord(
        step_fault=2,
        contradiction_type="ENERGY_BARRIER_VIOLATION",
        pre_repair_energy=2.10,
        post_repair_energy=0.25,
        repaired_steps=2,
        success=True
    )
    controller.record_intervention_event(rec, prompt="Verify energy equilibrium")

    # Check episodic ledger audit
    events = ledger.get_recent_events(limit=1)
    assert len(events) == 1
    assert events[0].action_type == "STRANGE_LOOP_INTERVENTION"
    assert events[0].status == "COMMITTED"

    # Check self_model interoception reflects strange-loop intervention
    summary = self_model.interoception.get_summary()
    assert summary["strange_loop_interventions"] >= 1


# =============================================================================
# 2. AutobiographicalNarrativeEngine Unit Tests
# =============================================================================

def test_autobiographical_narrative_engine_recording_and_synthesis(tmp_path):
    """Verify recording cognitive turns and synthesizing first-person narrative continuity."""
    stream_file = tmp_path / "autobio_test.jsonl"
    self_model = SelfModelEngine()

    engine = AutobiographicalNarrativeEngine(
        self_model=self_model,
        stream_path=str(stream_file)
    )

    # Record turn 1
    engine.record_cognitive_turn(
        user_text="Jane, can you explain the conservation of energy?",
        response_text="Energy cannot be created or destroyed, only transformed between states.",
        route_type="CONVERSATIONAL",
        internal_telemetry={"internal_viability": 0.94, "cognitive_strain": 0.08},
        new_concepts=["conservation_of_energy"]
    )

    # Record turn 2
    engine.record_cognitive_turn(
        user_text="What happens when a severe thermal anomaly occurs?",
        response_text="My anticipatory engine activates and triggers emergency containment.",
        route_type="CONVERSATIONAL",
        internal_telemetry={"internal_viability": 0.65, "cognitive_strain": 0.72},
        new_concepts=["thermal_containment"]
    )

    assert len(engine.stream) == 2

    narrative = engine.get_autobiographical_narrative()
    assert "Akhil" in narrative
    assert "autobiographical continuity" in narrative
    assert "Turn 1" in narrative or "Turn 2" in narrative


def test_chronesthesia_mental_time_travel(tmp_path):
    """Verify Endel Tulving's chronesthesia (mental time travel) compares past vs present states."""
    stream_file = tmp_path / "autobio_chronesthesia.jsonl"
    self_model = SelfModelEngine()
    self_model.update_interoception(cognitive_strain=0.10, homeostatic_energy=0.15)

    engine = AutobiographicalNarrativeEngine(
        self_model=self_model,
        stream_path=str(stream_file)
    )

    # Record past turn during high-strain anomaly
    engine.record_cognitive_turn(
        user_text="Critical coolant breach detected in reactor sector 4",
        response_text="Initiating coolant bypass valve opening to mitigate pressure spike.",
        route_type="PROACTIVE_ALERT",
        internal_telemetry={"internal_viability": 0.55, "cognitive_strain": 0.85, "homeostatic_energy": 0.90},
        new_concepts=["coolant_bypass"]
    )

    # Mental time travel query reflecting on the past incident
    reflection = engine.mental_time_travel("Remember when we had the coolant breach?")
    assert reflection["found"] is True
    assert "coolant breach" in reflection["narrative"].lower()
    assert "delta" in reflection
    delta = reflection["delta"]
    assert "strain_delta" in delta
    assert "viability_delta" in delta
    # Strain should have dropped compared to the past crisis
    assert delta["strain_delta"] < 0.0


# =============================================================================
# 3. Full JanePipeline End-to-End Integration Tests
# =============================================================================

def test_jane_pipeline_strange_loop_and_autobiographical_stream_end_to_end():
    """End-to-end verification of JanePipeline with StrangeLoop and AutobiographicalStream."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=False, device="cpu")

    # 1. Verify components initialized on pipeline
    assert hasattr(pipeline, "strange_loop")
    assert hasattr(pipeline, "autobiographical_engine")
    assert pipeline.self_model.autobiographical_engine is not None

    # 2. Execute standard conversational turn
    res = pipeline.process("Jane, let us analyze the relation between thermodynamics and entropy.")
    assert "response_text" in res
    assert "autobiographical_narrative" in res
    assert len(pipeline.autobiographical_engine.stream) >= 1

    # 3. Test mental time travel through pipeline
    travel_res = pipeline.mental_time_travel("thermodynamics and entropy")
    assert travel_res["found"] is True
    assert "thermodynamics" in travel_res["narrative"].lower() or "entropy" in travel_res["narrative"].lower()

    # 4. Check strange loop controller attributes
    assert pipeline.strange_loop.total_interventions_count >= 0

    # 5. Query Jane about her past evolution
    self_state = pipeline.self_model.answer_self_query("How have you changed earlier in our timeline?")
    assert "autobiographical_narrative" in self_state
    assert "chronesthesia" in self_state
    assert self_state["chronesthesia"] is not None
    assert self_state["chronesthesia"]["found"] is True
