"""Strange-Loop Self-Intervention Engine (Hofstadter Strange Loop) for Project Jane.

Implements Gödelian self-reference and counterfactual causal intervention:
1. Multi-Step Tape Checkpointing:
   Stores intermediate (latent_state, dnc_memory_tape) snapshots at each reasoning step.
2. Contradiction & Paradox Detection:
   Monitors energy divergence, formal Z3 SMT contradiction, or critic invariant violations.
3. Counterfactual Rollback & Causal Branching:
   When an intermediate thought produces a contradiction, Jane treats her own thought
   as the patient of intervention, rolls back DNC memory tape to the pre-fault state,
   applies a Pearl do-calculus corrective bias vector away from the paradox attractor,
   and branches counterfactually down a repaired reasoning trajectory.
4. Metacognitive Ledger Commitment:
   Records STRANGE_LOOP_INTERVENTION in EpisodicLedger and updates interoceptive strain.
"""

import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

from src.utils.logger import get_logger

logger = get_logger("strange_loop")


@dataclass
class TapeCheckpoint:
    """Snapshot of reasoning state and DNC memory tape at a specific step."""
    step: int
    latent_state: torch.Tensor
    memory_tape: torch.Tensor
    energy: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class StrangeLoopInterventionRecord:
    """Audit record of a strange-loop counterfactual rollback and intervention."""
    step_fault: int
    contradiction_type: str
    pre_repair_energy: float
    post_repair_energy: float
    repaired_steps: int
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class StrangeLoopController:
    """
    Hofstadter Strange-Loop Self-Intervention Controller.
    Enables Jane to observe, critique, and intervene on her own internal reasoning trajectory.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        energy_divergence_threshold: float = 1.80,
        max_interventions_per_query: int = 3,
        episodic_ledger: Optional[Any] = None,
        self_model: Optional[Any] = None
    ):
        self.latent_dim = latent_dim
        self.energy_divergence_threshold = energy_divergence_threshold
        self.max_interventions_per_query = max_interventions_per_query
        self.ledger = episodic_ledger
        self.self_model = self_model

        self.checkpoints: List[TapeCheckpoint] = []
        self.intervention_history: List[StrangeLoopInterventionRecord] = []
        self.total_interventions_count: int = 0

    def clear_checkpoints(self):
        """Clears checkpoint buffer for a new reasoning pass."""
        self.checkpoints.clear()

    def record_checkpoint(
        self,
        step: int,
        latent_state: torch.Tensor,
        memory_tape: torch.Tensor,
        energy: float
    ):
        """Saves a reasoning checkpoint for potential counterfactual rollback."""
        self.checkpoints.append(
            TapeCheckpoint(
                step=step,
                latent_state=latent_state.detach().clone(),
                memory_tape=memory_tape.detach().clone() if memory_tape is not None else None,
                energy=float(energy)
            )
        )
        if len(self.checkpoints) > 50:
            self.checkpoints.pop(0)

    def detect_contradiction(
        self,
        current_energy: float,
        prev_energy: float,
        critic_violations: Optional[List[str]] = None,
        z3_is_valid: Optional[bool] = None
    ) -> Optional[str]:
        """
        Evaluates whether current reasoning state exhibits contradiction or paradox.
        Returns contradiction_type string if intervention is required, None otherwise.
        """
        # 1. Formal mathematical falsification
        if z3_is_valid is False:
            return "Z3_SMT_CONTRADICTION"

        # 2. Energy explosion / dynamical divergence
        if current_energy > self.energy_divergence_threshold:
            return "ENERGY_BARRIER_VIOLATION"

        if (current_energy - prev_energy) > 0.85 and prev_energy < 1.0:
            return "LYAPUNOV_ENERGY_SPIKE"

        # 3. Critic identity / physical violation
        if critic_violations:
            for v in critic_violations:
                v_lower = v.lower()
                if any(k in v_lower for k in ("identity", "embodiment", "physics", "unsupported", "denial")):
                    return f"CRITIC_VIOLATION:{v}"

        return None

    def execute_counterfactual_rollback(
        self,
        fault_step: int,
        contradiction_type: str,
        current_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], TapeCheckpoint]:
        """
        Rolls back the DNC memory tape to the last stable pre-contradiction checkpoint,
        computes a Pearl do-calculus corrective bias vector away from the paradox attractor,
        and returns the repaired latent state and restored memory tape.
        """
        if not self.checkpoints:
            # Fallback if no checkpoints exist
            nominal = torch.zeros_like(current_state)
            return F.normalize(nominal, p=2, dim=-1), None, None

        # Find the most recent checkpoint prior to the fault step with stable energy
        candidate_ckpts = [
            cp for cp in self.checkpoints
            if cp.step < fault_step and cp.energy < self.energy_divergence_threshold
        ]

        if candidate_ckpts:
            target_ckpt = candidate_ckpts[-1]
        else:
            target_ckpt = self.checkpoints[0]

        # Compute repulsive vector: direction that caused the contradiction
        repulse_dir = F.normalize(current_state - target_ckpt.latent_state, p=2, dim=-1)
        nominal_attractor = torch.zeros_like(current_state)

        # Counterfactual surgery: step backwards from fault direction + pull to nominal attractor
        repaired_latent = target_ckpt.latent_state - 0.70 * repulse_dir + 0.15 * nominal_attractor
        repaired_latent = F.normalize(repaired_latent, p=2, dim=-1)

        restored_tape = target_ckpt.memory_tape.clone() if target_ckpt.memory_tape is not None else None

        self.total_interventions_count += 1
        return repaired_latent, restored_tape, target_ckpt

    def record_intervention_event(
        self,
        record: StrangeLoopInterventionRecord,
        prompt: Optional[str] = None
    ):
        """Commits intervention to historical record, EpisodicLedger, and SelfModel interoception."""
        self.intervention_history.append(record)
        if len(self.intervention_history) > 100:
            self.intervention_history.pop(0)

        # 1. Update Episodic Ledger
        if self.ledger is not None and hasattr(self.ledger, "record_event"):
            self.ledger.record_event(
                action_type="STRANGE_LOOP_INTERVENTION",
                concept_name="SelfCorrectedContradiction",
                source_name="StrangeLoopController",
                formula_or_axiom="counterfactual_tape_rollback(k) -> valid_branch",
                status="COMMITTED" if record.success else "REJECTED",
                details={
                    "contradiction_type": record.contradiction_type,
                    "fault_step": record.step_fault,
                    "pre_repair_energy": record.pre_repair_energy,
                    "post_repair_energy": record.post_repair_energy,
                    "success": record.success,
                    "prompt": prompt or ""
                }
            )

        # 2. Update SelfModel Interoception
        if self.self_model is not None:
            if hasattr(self.self_model, "interoception") and hasattr(self.self_model.interoception, "record_strange_loop_intervention"):
                self.self_model.interoception.record_strange_loop_intervention()
            if hasattr(self.self_model, "update_interoception"):
                strain_boost = 0.40 if record.success else 0.80
                strange_cnt = (
                    self.self_model.interoception.strange_loop_interventions
                    if hasattr(self.self_model, "interoception")
                    else max(1, self.total_interventions_count)
                )
                self.self_model.update_interoception(
                    cognitive_strain=strain_boost,
                    homeostatic_energy=record.post_repair_energy,
                    strange_loop_count=strange_cnt
                )

        logger.info(
            "Strange-Loop Self-Intervention executed [Fault Step %d | Type: %s | Energy: %.3f -> %.3f | Success: %s]",
            record.step_fault,
            record.contradiction_type,
            record.pre_repair_energy,
            record.post_repair_energy,
            record.success
        )
