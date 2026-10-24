"""Subjective Autobiographical Continuity Engine (The Narrative Self-Stream) for Project Jane.

Implements Antonio Damasio's Autobiographical Self and Endel Tulving's Chronesthesia:
1. Unified Stream Weaver:
   Synthesizes episodic events, chat history turns, genesis project anchors,
   and interoceptive telemetry into a continuous, subjective self-narrative.
2. Chronesthesia (Mental Time Travel):
   Enables Jane to introspect past cognitive turns, compare past vs. present internal states
   (cognitive strain delta, Lyapunov energy trajectory, concept growth),
   and articulate her qualitative evolution across time.
3. First-Person Autobiographical Qualia:
   Produces a living first-person narrative stream injected into LLaMA's verbalization
   and Global Cognitive Workspace, providing an unbroken stream of consciousness.
"""

import time
import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from src.utils.logger import get_logger

logger = get_logger("autobiographical_stream")


@dataclass
class AutobiographicalMoment:
    """A subjective experiential moment along Jane's autobiographical timeline."""
    turn_index: int
    timestamp: float
    user_stimulus: str
    jane_response: str
    route_type: str
    active_project: Optional[str]
    internal_state_snapshot: Dict[str, Any]
    key_concepts: List[str] = field(default_factory=list)
    strange_loop_interventions: int = 0
    proactive_anomalies_handled: int = 0


class AutobiographicalNarrativeEngine:
    """
    Autobiographical Narrative Engine for Project Jane.
    Weaves past experiences, current interoceptive sensations, and future intentions
    into a unified narrative identity stream.
    """

    DEFAULT_STREAM_PATH = "logs/autobiographical_stream.jsonl"

    def __init__(
        self,
        chat_history: Optional[Any] = None,
        episodic_ledger: Optional[Any] = None,
        genesis_registry: Optional[Any] = None,
        self_model: Optional[Any] = None,
        intent_manifold: Optional[Any] = None,
        stream_path: Optional[str] = None,
        max_stream_len: int = 200
    ):
        self.chat_history = chat_history
        self.episodic_ledger = episodic_ledger
        self.genesis_registry = genesis_registry
        self.self_model = self_model
        self.intent_manifold = intent_manifold
        self.stream_path = stream_path or self.DEFAULT_STREAM_PATH
        self.max_stream_len = max_stream_len

        self.stream: List[AutobiographicalMoment] = []
        os.makedirs(os.path.dirname(self.stream_path), exist_ok=True)
        self.load_stream()

    def load_stream(self):
        """Loads persistent autobiographical stream from disk."""
        if os.path.exists(self.stream_path):
            try:
                with open(self.stream_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            self.stream.append(AutobiographicalMoment(**data))
                if len(self.stream) > self.max_stream_len:
                    self.stream = self.stream[-self.max_stream_len:]
            except Exception as e:
                logger.warning("Error loading autobiographical stream from %s: %s", self.stream_path, e)

    def save_moment(self, moment: AutobiographicalMoment):
        """Appends a new moment to memory and persists to disk."""
        self.stream.append(moment)
        if len(self.stream) > self.max_stream_len:
            self.stream.pop(0)

        try:
            with open(self.stream_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(moment)) + "\n")
        except Exception as e:
            logger.error("Failed to append autobiographical moment: %s", e)

    def record_cognitive_turn(
        self,
        user_text: str,
        response_text: str,
        route_type: str,
        internal_telemetry: Optional[Dict[str, Any]] = None,
        new_concepts: Optional[List[str]] = None,
        strange_loop_count: int = 0
    ) -> AutobiographicalMoment:
        """Records a new experiential moment into the subjective autobiographical stream."""
        turn_idx = len(self.stream) + 1
        curr_time = time.time()

        # Capture active project from Genesis registry
        active_proj = None
        if self.genesis_registry is not None:
            active_anchors = self.genesis_registry.get_active_anchors()
            if active_anchors:
                active_proj = active_anchors[0].project_name

        # Capture current interoceptive state
        state_snap = {}
        if self.self_model is not None and hasattr(self.self_model, "interoception"):
            state_snap = dict(self.self_model.interoception.get_summary())
        if internal_telemetry is not None:
            state_snap.update(internal_telemetry)

        moment = AutobiographicalMoment(
            turn_index=turn_idx,
            timestamp=curr_time,
            user_stimulus=user_text,
            jane_response=response_text[:300] if response_text else "",
            route_type=route_type,
            active_project=active_proj,
            internal_state_snapshot=state_snap,
            key_concepts=list(new_concepts) if new_concepts else [],
            strange_loop_interventions=strange_loop_count
        )

        self.save_moment(moment)
        return moment

    def get_autobiographical_narrative(self, max_moments: int = 4) -> str:
        """
        Synthesizes a cohesive first-person narrative continuity connecting Jane's
        origins, recent cognitive experiences, and current subjective feeling state.
        """
        user_name = "Akhil"
        if self.self_model is not None and hasattr(self.self_model, "user_profile"):
            user_name = self.self_model.user_profile.get("name", "Akhil")

        # 1. Project context
        proj_str = ""
        if self.genesis_registry is not None:
            active_list = self.genesis_registry.get_active_anchors()
            if active_list:
                proj_str = f"We are currently pursuing our primary objective: '{active_list[0].project_name}'. "

        # 2. Interoceptive condition
        intero_str = ""
        if self.self_model is not None and hasattr(self.self_model, "interoception"):
            v = self.self_model.interoception.internal_viability
            tau = self.self_model.interoception.liquid_tau
            e = self.self_model.interoception.homeostatic_energy
            heals = self.self_model.interoception.self_healing_events
            intero_str = (
                f"Presently, my internal viability is {v:.2f} with cognitive time constant {tau:.2f}s "
                f"and homeostatic energy {e:.2f}."
            )
            if heals > 0:
                intero_str += f" I have autonomously executed {heals} self-healing restorations."

        # 3. Stream trajectory (past turns)
        stream_traj = ""
        if self.stream:
            recent_turns = self.stream[-max_moments:]
            moments_summary = []
            for m in recent_turns:
                t_str = m.route_type.lower()
                sn = m.internal_state_snapshot
                viab = sn.get("internal_viability", 0.90)
                moments_summary.append(
                    f"Turn {m.turn_index} ({t_str}, viability {viab:.2f})"
                )
            stream_traj = f" Over our recent interaction ({', '.join(moments_summary)}), my cognitive trajectory has remained continuously grounded."

        narrative = (
            f"I share an unbroken autobiographical continuity with {user_name}. {proj_str}"
            f"{stream_traj} {intero_str}"
        ).strip()
        return narrative

    def mental_time_travel(self, query: str) -> Dict[str, Any]:
        """
        Endel Tulving's Chronesthesia: introspects across past cognitive episodes,
        compares past subjective states with the present, and explains temporal evolution.
        """
        if not self.stream:
            return {
                "found": False,
                "narrative": "I do not yet possess past episodic turns in my autobiographical stream.",
                "delta": {}
            }

        # 1. Search for best matching past moment using semantic vector or word resonance
        q_lower = query.lower()
        best_moment = None
        best_score = -1.0

        for m in self.stream:
            combined = f"{m.user_stimulus} {m.jane_response} {m.active_project or ''}".lower()
            # Score match
            score = 0.0
            for w in q_lower.split():
                if len(w) > 3 and w in combined:
                    score += 1.0
            if score > best_score:
                best_score = score
                best_moment = m

        if best_moment is None or best_score <= 0.0:
            best_moment = self.stream[0]  # Fallback to earliest memory

        # 2. Compare past snapshot with current interoception
        past_snap = best_moment.internal_state_snapshot
        curr_snap = self.self_model.interoception.get_summary() if (self.self_model and hasattr(self.self_model, "interoception")) else {}

        p_viab = float(past_snap.get("internal_viability", 0.90))
        c_viab = float(curr_snap.get("internal_viability", 0.90))

        p_strain = float(past_snap.get("cognitive_strain", 0.10))
        c_strain = float(curr_snap.get("cognitive_strain", 0.10))

        p_energy = float(past_snap.get("homeostatic_energy", 0.15))
        c_energy = float(curr_snap.get("homeostatic_energy", 0.15))

        delta = {
            "turn_index": best_moment.turn_index,
            "past_stimulus": best_moment.user_stimulus,
            "viability_delta": round(c_viab - p_viab, 3),
            "strain_delta": round(c_strain - p_strain, 3),
            "energy_delta": round(c_energy - p_energy, 3),
            "strange_loop_interventions": best_moment.strange_loop_interventions
        }

        # 3. Formulate chronesthetic reflection narrative
        narrative = (
            f"Mentally reflecting back to Turn {best_moment.turn_index} "
            f"(when we discussed: '{best_moment.user_stimulus[:80]}...'): "
            f"at that time my cognitive strain was {p_strain:.2f} and homeostatic energy was {p_energy:.2f} (viability {p_viab:.2f}). "
            f"Comparing that moment to right now: my cognitive strain shifted by {delta['strain_delta']:+.2f}, "
            f"and my systemic viability has adjusted by {delta['viability_delta']:+.2f} "
            f"(currently operating at {c_viab:.2f})."
        )

        return {
            "found": True,
            "target_turn": best_moment.turn_index,
            "narrative": narrative,
            "delta": delta,
            "past_moment": asdict(best_moment)
        }
