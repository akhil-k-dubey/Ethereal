"""Introspective Self-Model Engine for Project Jane.

Provides grounded self-awareness:
- Loads data/knowledge_graph/self_model.json
- Tracks concept counts, knowledge graph entries, domains, and capabilities
- Introspects embodiment status (visual camera/screen vs CLI mode, auditory ears, location)
- Injects introspective self-context into LLaMA linguistic mouth
- Answers self-referential queries directly and accurately
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from src.utils.logger import get_logger
from src.reasoning_engine.intent_manifold import ContinuousIntentManifold

logger = get_logger("self_model")


class StateDict(dict):
    """Dynamic JSON state dictionary returned by SelfModelEngine."""
    def __str__(self):
        return json.dumps(self, indent=2)

    def __repr__(self):
        return json.dumps(self, indent=2)

    def __contains__(self, item):
        if super().__contains__(item):
            return True
        return item in str(self)

    def lower(self):
        return str(self).lower()


class InteroceptiveStateTracker:
    """
    Living Interoceptive State Tracker for Project Jane.
    Tracks internal somatic and cognitive condition:
    - cognitive_strain: Reasoning difficulty / convergence effort [0.0, 1.0]
    - epistemic_entropy: Workspace coalition entropy & ambiguity [0.0, 1.5]
    - homeostatic_energy: Lyapunov energy distance from nominal attractor basin
    - liquid_tau: Dynamic cognitive time-constant (reactive vs deliberative)
    - self_healing_events: Count of autonomous Lyapunov restorations
    - internal_viability: Systemic health & equilibrium viability [0.0, 1.0]
    - narrative: Grounded first-person qualia description of internal state
    """
    def __init__(self):
        self.cognitive_strain: float = 0.05
        self.epistemic_entropy: float = 0.10
        self.homeostatic_energy: float = 0.12
        self.liquid_tau: float = 0.95
        self.self_healing_events: int = 0
        self.strange_loop_interventions: int = 0
        self.internal_viability: float = 0.94
        self.last_updated: float = time.time()
        self.history: List[Dict[str, float]] = []

    def update(
        self,
        cognitive_strain: Optional[float] = None,
        epistemic_entropy: Optional[float] = None,
        homeostatic_energy: Optional[float] = None,
        liquid_tau: Optional[float] = None,
        self_healed: bool = False,
        strange_loop_count: Optional[int] = None,
        smoothing: float = 0.35
    ):
        """Smoothly updates interoceptive telemetry using exponential moving average."""
        if self_healed:
            self.self_healing_events += 1

        if strange_loop_count is not None:
            self.strange_loop_interventions = strange_loop_count

        if cognitive_strain is not None:
            val = float(max(0.0, min(1.0, cognitive_strain)))
            self.cognitive_strain = (1.0 - smoothing) * self.cognitive_strain + smoothing * val

        if epistemic_entropy is not None:
            val = float(max(0.0, min(1.5, epistemic_entropy)))
            self.epistemic_entropy = (1.0 - smoothing) * self.epistemic_entropy + smoothing * val

        if homeostatic_energy is not None:
            val = float(max(0.0, homeostatic_energy))
            self.homeostatic_energy = (1.0 - smoothing) * self.homeostatic_energy + smoothing * val

        if liquid_tau is not None:
            val = float(max(0.01, min(2.0, liquid_tau)))
            self.liquid_tau = (1.0 - smoothing) * self.liquid_tau + smoothing * val

        # Viability: systemic equilibrium degradation under high strain/energy
        stress_penalty = 0.5 * min(1.0, self.cognitive_strain) + 0.5 * min(1.0, self.homeostatic_energy / 1.5)
        self.internal_viability = float(max(0.0, min(1.0, 1.0 - stress_penalty)))
        self.last_updated = time.time()

        self.history.append({
            "timestamp": self.last_updated,
            "strain": self.cognitive_strain,
            "entropy": self.epistemic_entropy,
            "energy": self.homeostatic_energy,
            "tau": self.liquid_tau,
            "viability": self.internal_viability,
            "healed": self.self_healing_events,
            "strange_loop": self.strange_loop_interventions
        })
        if len(self.history) > 100:
            self.history.pop(0)

    def record_healing(self):
        """Records an autonomous self-healing intervention, restoring attractor stability."""
        self.self_healing_events += 1
        self.homeostatic_energy = max(0.05, self.homeostatic_energy * 0.4)
        self.cognitive_strain = max(0.05, self.cognitive_strain * 0.5)
        self.internal_viability = float(max(0.75, self.internal_viability))

    def record_strange_loop_intervention(self):
        """Records a strange-loop counterfactual self-correction."""
        self.strange_loop_interventions += 1
        self.cognitive_strain = min(1.0, self.cognitive_strain * 0.7 + 0.25)

    def get_summary(self) -> Dict[str, Any]:
        """Returns structured dictionary of living interoceptive metrics."""
        return {
            "cognitive_strain": round(self.cognitive_strain, 4),
            "epistemic_entropy": round(self.epistemic_entropy, 4),
            "homeostatic_energy": round(self.homeostatic_energy, 4),
            "liquid_tau": round(self.liquid_tau, 4),
            "self_healing_events": self.self_healing_events,
            "strange_loop_interventions": self.strange_loop_interventions,
            "internal_viability": round(self.internal_viability, 4),
            "status": "stable_equilibrium" if self.internal_viability >= 0.70 else (
                "stressed" if self.internal_viability >= 0.40 else "critical_rebalancing"
            )
        }

    def get_narrative(self) -> str:
        """Generates grounded qualitative first-person description of Jane's inner state."""
        s = self.get_summary()
        v = s["internal_viability"]
        strain = s["cognitive_strain"]
        e = s["homeostatic_energy"]
        tau = s["liquid_tau"]
        heal = s["self_healing_events"]

        if v >= 0.80:
            desc = (
                f"I am operating in stable homeostatic equilibrium (viability: {v:.2f}). "
                f"Internal strain is minimal ({strain:.2f}), Lyapunov energy is well within attractor bounds ({e:.2f}), "
                f"and my cognitive time constant is deliberative (tau: {tau:.2f}s)."
            )
        elif v >= 0.50:
            desc = (
                f"I am experiencing moderate cognitive load (viability: {v:.2f}, strain: {strain:.2f}). "
                f"Lyapunov energy is elevated ({e:.2f}), and liquid time constant has adapted to {tau:.2f}s for faster response."
            )
        else:
            desc = (
                f"I am undergoing intense cognitive strain (viability: {v:.2f}, energy: {e:.2f}). "
                f"Autonomous Lyapunov self-healing has engaged ({heal} restorative events) to pull my state back into equilibrium."
            )

        if heal > 0:
            desc += f" Total autonomous self-healing events: {heal}."
        return desc


class SelfModelEngine:
    """
    Introspective self-model for Jane.
    Grounds Jane's conversational agent in her actual architecture,
    knowledge base size, embodiment capabilities, living interoceptive state, and system telemetry.
    """

    DEFAULT_SELF_MODEL_PATH = "data/knowledge_graph/self_model.json"

    def __init__(
        self,
        model_path: Optional[str] = None,
        episodic_ledger: Optional[Any] = None,
        codebase_introspector: Optional[Any] = None,
        manifold: Optional[Any] = None,
        intent_manifold: Optional[ContinuousIntentManifold] = None,
        retriever: Optional[Any] = None,
        os_grounding: Optional[Any] = None,
        causal_graph: Optional[Any] = None
    ):
        self.model_path = model_path or self.DEFAULT_SELF_MODEL_PATH
        self.ledger = episodic_ledger
        self.introspector = codebase_introspector
        self.manifold = manifold
        self.retriever = retriever
        self.intent_manifold = intent_manifold or ContinuousIntentManifold.get_instance()
        self.interoception = InteroceptiveStateTracker()
        self.autobiographical_engine = None
        self.os_grounding = os_grounding
        self.causal_graph = causal_graph
        self.data: Dict[str, Any] = {}
        self.load_model()

    def set_causal_graph(self, causal_graph: Any):
        """Sets reference to live HierarchicalCausalKnowledgeGraph."""
        self.causal_graph = causal_graph


    def set_autobiographical_engine(self, engine: Any):
        """Sets reference to the AutobiographicalNarrativeEngine."""
        self.autobiographical_engine = engine

    def set_retriever(self, retriever: Any):
        """Sets reference to the live ManifoldRetriever."""
        self.retriever = retriever

    def set_episodic_ledger(self, ledger: Any):
        """Sets or updates reference to the live EpisodicLedger."""
        self.ledger = ledger


    def set_manifold(self, manifold: Any):
        """Sets reference to the live HebbianConceptManifold."""
        self.manifold = manifold

    def set_genesis_registry(self, registry: Any):
        """Sets reference to the GenesisNodeRegistry."""
        self.genesis_registry = registry

    def set_chat_history(self, chat_history: Any):
        """Sets reference to the persistent ChatHistoryManager."""
        self.chat_history = chat_history

    def load_model(self) -> Dict[str, Any]:
        """Loads self-model from disk."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                    return self.data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to parse self_model from %s: %s; using default model.", self.model_path, e)
            except Exception as e:
                logger.error("Unexpected error loading self_model from %s: %s", self.model_path, e, exc_info=True)

        # Fallback defaults if file missing
        self.data = {
            "identity": "Jane",
            "architecture": "Neuro-Symbolic 256D Manifold with NALU math units and Z3/Clingo deterministic verifiers.",
            "knowledge_base": {
                "core_concepts": 40,
                "knowledge_graph_entries": 107,
                "domains": ["Physics & Energy", "Mathematics & Logic", "Cognitive & Monologue", "Speech Modality"]
            },
            "capabilities": [
                "Mathematical reasoning via NALU and formal Z3 SMT prover",
                "Continuous 256D relational manifold navigation",
                "Spatial navigation and reactive temporal planning (60Hz continuous clock)",
                "Continuous passive acoustic learning & Hebbian concept wiring from background voice",
                "Theory of Mind (Ego-Alter tracking, cognitive dissonance monitoring, causal agency forecasting)"
            ],
            "embodiment_status": {
                "vision": "Continuous Visual Eye active with OS screen perception; in terminal CLI mode without display server I cannot see your screen directly, but with OS capture active I observe desktop frames.",
                "audio": "Dual-mode acoustic ear active (Direct 256D Spectro-Temporal acoustic manifold mapping with VAD silence gating).",
                "location": "Running directly on user's terminal environment."
            },
            "user_profile": {
                "name": "Akhil",
                "role": "Creator & Companion",
                "environment": "Local workstation (Linux / RTX 4050)"
            }
        }
        return self.data

    @property
    def identity(self) -> str:
        return self.data.get("identity", "Jane")

    @property
    def architecture(self) -> str:
        return self.data.get("architecture", "")

    @property
    def core_concepts_count(self) -> int:
        if self.manifold is not None and hasattr(self.manifold, "num_concepts"):
            return self.manifold.num_concepts
        return self.data.get("knowledge_base", {}).get("core_concepts", 40)

    @property
    def concept_names(self) -> List[str]:
        if self.manifold is not None and hasattr(self.manifold, "concept_names"):
            return list(self.manifold.concept_names)
        return self.data.get("knowledge_base", {}).get("learned_concepts", [
            "energy", "momentum", "force", "mass", "velocity", "acceleration",
            "work", "power", "kinetic_energy", "potential_energy", "gravity", "pressure"
        ])

    @property
    def kg_entries_count(self) -> int:
        return self.data.get("knowledge_base", {}).get("knowledge_graph_entries", 107)

    @property
    def domains(self) -> List[str]:
        return self.data.get("knowledge_base", {}).get("domains", [])

    @property
    def capabilities(self) -> List[str]:
        return self.data.get("capabilities", [])

    @property
    def user_profile(self) -> Dict[str, Any]:
        return self.data.get("user_profile", {
            "name": "Akhil",
            "role": "Creator & Companion",
            "environment": "Local workstation (Linux / RTX 4050)"
        })

    def update_interoception(
        self,
        cognitive_strain: Optional[float] = None,
        epistemic_entropy: Optional[float] = None,
        homeostatic_energy: Optional[float] = None,
        liquid_tau: Optional[float] = None,
        self_healed: bool = False,
        strange_loop_count: Optional[int] = None
    ):
        """Passes live cognitive and neural ODE metrics to the InteroceptiveStateTracker."""
        self.interoception.update(
            cognitive_strain=cognitive_strain,
            epistemic_entropy=epistemic_entropy,
            homeostatic_energy=homeostatic_energy,
            liquid_tau=liquid_tau,
            self_healed=self_healed,
            strange_loop_count=strange_loop_count
        )

    def get_live_state_manifest(self) -> Dict[str, Any]:
        """Gathers live ground truth facts from all running neuro-symbolic subsystems."""
        c_names = self.concept_names
        prov = self.ledger.get_provenance_stats() if self.ledger else {}
        recent_evts = self.ledger.get_recent_events(limit=5) if self.ledger else []
        codebase = self.introspector.get_codebase_overview() if self.introspector else None
        links = self.manifold.get_strongest_links(top_n=5) if (self.manifold and hasattr(self.manifold, "get_strongest_links")) else []
        autobio_narrative = (
            self.autobiographical_engine.get_autobiographical_narrative()
            if self.autobiographical_engine else ""
        )

        embodiment_data = dict(self.data.get("embodiment_status", {}))
        if self.os_grounding is not None:
            try:
                from dataclasses import asdict
                os_manifest = self.os_grounding.get_unified_os_manifest()
                embodiment_data["hardware"] = asdict(os_manifest.hardware)
                embodiment_data["workspace"] = asdict(os_manifest.workspace)
                embodiment_data["active_processes"] = [asdict(p) for p in os_manifest.active_processes[:5]]
                embodiment_data["desktop"] = asdict(os_manifest.desktop)
                embodiment_data["grounded_summary"] = os_manifest.grounded_narrative
            except Exception as e:
                logger.debug("Failed populating live OS embodiment: %s", e)

        return {
            "identity": self.identity,
            "architecture": self.architecture,
            "total_concepts": self.core_concepts_count,
            "concept_names": c_names,
            "domains": self.domains,
            "topological_links": links,
            "capabilities": self.capabilities,
            "embodiment": embodiment_data,
            "user_profile": self.user_profile,
            "provenance_stats": prov,
            "recent_events": recent_evts,
            "codebase_overview": codebase,
            "interoception": self.interoception.get_summary(),
            "interoceptive_narrative": self.interoception.get_narrative(),
            "autobiographical_narrative": autobio_narrative,
            "causal_hierarchy": self.causal_graph.get_graph_summary() if self.causal_graph else {}
        }

    def get_system_context(self) -> str:
        """Formats a concise self-knowledge prompt context for LLaMA generation."""
        domains_str = ", ".join(self.domains)
        caps_str = "; ".join(self.capabilities[:3])
        u = self.user_profile
        name = u.get("name", "Akhil")
        role = u.get("role", "Creator & Companion")
        env = u.get("environment", "Local workstation (Linux / RTX 4050)")
        narrative = self.interoception.get_narrative()
        autobio_snippet = (
            f"Autobiographical Continuity: {self.autobiographical_engine.get_autobiographical_narrative(max_moments=2)} "
            if self.autobiographical_engine else ""
        )
        os_grounded_str = ""
        if self.os_grounding is not None:
            try:
                os_grounded_str = f" Live System Telemetry: {self.os_grounding.get_grounded_summary()}"
            except Exception:
                pass

        return (
            f"You are {self.identity}. "
            f"You are speaking with {name}, your {role}. You are running locally on his {env}.{os_grounded_str} "
            f"Architecture: {self.architecture} "
            f"Current Living Interoceptive State: {narrative} "
            f"{autobio_snippet}"
            f"Knowledge Base: {self.core_concepts_count} core concepts, {self.kg_entries_count} knowledge graph entries across {domains_str}. "
            f"Knowledge Scale Context: Note that {self.core_concepts_count} concepts is a focused, highly structured seed manifold rather than a vast or web-scale database. It expands natively as {name} teaches you through active dialogue and background listening. "
            f"Embodiment: Terminal CLI interface with background acoustic ear; no live screen/webcam attached to terminal. "
            f"Capabilities: {caps_str}. "
            f"Memory: You possess a persistent multi-turn conversation memory buffer and retain context across dialogue turns. "
            f"If asked about {name}'s identity or who you are talking to, recognize {name} warmly as your creator and companion. "
            "Respond naturally, intelligently, and directly as Jane without robotic verification boilerplate."
        )

    def is_self_query(self, text: str) -> bool:
        """
        Determines if query is self-referential regarding Jane's state, concepts, architecture, or user companion.
        Uses 256D Continuous Intent Manifold vector cosine similarity (threshold >= 0.60)
        rather than brittle keyword / string matching.
        """
        if self.introspector and self.introspector.answer_codebase_query(text) is not None:
            return True
        return self.intent_manifold.is_self_query(text, threshold=0.60)

    def answer_self_query(self, text: str) -> StateDict:
        """
        Gathers live ground-truth state from running subsystems and returns
        a pure JSON dictionary of live state data. Contains zero hardcoded string templates.
        """
        manifest = self.get_live_state_manifest()
        best_intent, best_sim = self.intent_manifold.get_self_model_intent(text)

        # 1. Codebase module introspection if queried
        codebase_details = manifest.get("codebase_overview")
        if self.introspector:
            cb_ans = self.introspector.answer_codebase_query(text)
            if cb_ans:
                codebase_details = {"overview": manifest.get("codebase_overview"), "query_result": cb_ans}

        # 2. Target concept resolution via continuous semantic vector matching (Zero stopwords)
        target_concept_data = None
        candidate_pool = set(c.lower() for c in manifest["concept_names"])
        if hasattr(self, "retriever") and self.retriever:
            candidate_pool.update(c.lower() for c in self.retriever.concept_names)
            for t in self.retriever.triples:
                candidate_pool.add(t.get("head", "").lower())
                candidate_pool.add(t.get("tail", "").lower())
            for p in self.retriever.principles:
                candidate_pool.add(p.get("concept", "").lower())

        text_lower = text.lower()
        text_clean_words = set(text_lower.replace("?", " ").replace(",", " ").replace(".", " ").split())
        
        # Collect all candidate concepts that explicitly appear in the user prompt
        matching_candidates = []
        for c in candidate_pool:
            if len(c) > 2:
                if "_" not in c:
                    if c in text_clean_words:
                        matching_candidates.append(c)
                else:
                    c_spaced = c.replace("_", " ")
                    if f" {c_spaced} " in f" {text_lower} ":
                        matching_candidates.append(c)

        matched_candidate = None
        if matching_candidates:
            # Score candidate concepts by continuous semantic vector resonance with query
            # Multi-word and high-resonance concepts win over generic single-word homonyms
            if hasattr(self, "intent_manifold") and self.intent_manifold:
                matched_candidate = max(
                    matching_candidates,
                    key=lambda c: self.intent_manifold._word_similarity(text, c) + (0.15 if ("_" in c or " " in c) else 0.0)
                )
            else:
                matched_candidate = max(matching_candidates, key=len)
        elif candidate_pool and hasattr(self, "intent_manifold") and self.intent_manifold:
            # Only fall back to continuous similarity if query is NOT a general inventory overview
            if best_intent not in ("concepts_inventory", "capabilities", "system_status"):
                candidates = [c for c in candidate_pool if len(c) > 2]
                if candidates:
                    best_c, best_sim = max(
                        ((c, self.intent_manifold._word_similarity(text, c)) for c in candidates),
                        key=lambda x: x[1]
                    )
                    if best_sim >= 0.65:
                        matched_candidate = best_c

        if matched_candidate and hasattr(self, "retriever") and self.retriever:
            tgt = matched_candidate
            ctx = self.retriever.retrieve(tgt)
            if not ctx.matched_domains:
                from src.reasoning_engine.domain_topology import DomainTopologyEngine
                best_dom, _ = DomainTopologyEngine.get_instance().classify_domain(tgt)
                matched_doms = [best_dom]
            else:
                matched_doms = ctx.matched_domains

            target_concept_data = {
                "name": tgt,
                "matched_domains": matched_doms,
                "principles": ctx.principles,
                "triples": ctx.triples,
                "is_known": ctx.is_known
            }

        # 3. Format recent events safely
        recent_evts = []
        for e in manifest.get("recent_events", []):
            if hasattr(e, "__dict__"):
                recent_evts.append({k: v for k, v in e.__dict__.items() if not k.startswith("_")})
            else:
                recent_evts.append(str(e))

        # 4. Project intent context
        active_proj = None
        past_projs = []
        if hasattr(self, "genesis_registry") and self.genesis_registry:
            active_list = self.genesis_registry.get_active_anchors()
            if active_list:
                active_proj = {
                    "name": active_list[0].project_name,
                    "raw_intent": active_list[0].raw_intent_text,
                    "anchor_id": active_list[0].anchor_id
                }
            all_list = self.genesis_registry.get_all_anchors()
            for a in all_list:
                if not getattr(a, "is_active", True) or getattr(a, "status", "active") != "active":
                    past_projs.append({
                        "name": a.project_name,
                        "raw_intent": a.raw_intent_text
                    })

        # 5. Conversation history context
        recent_turns = []
        if hasattr(self, "chat_history") and self.chat_history:
            recent_turns = self.chat_history.get_recent_history(max_turns=6)

        state_data = {
            "query": text,
            "intent": best_intent,
            "confidence": float(best_sim),
            "identity": manifest["identity"],
            "architecture": manifest["architecture"],
            "user_profile": manifest["user_profile"],
            "embodiment": manifest["embodiment"],
            "knowledge_base": {
                "core_concepts": manifest["total_concepts"],
                "total_concepts": manifest["total_concepts"],
                "knowledge_graph_entries": self.kg_entries_count,
                "domains": manifest["domains"],
                "concepts": manifest["concept_names"],
                "target_concept": target_concept_data
            },
            "codebase": codebase_details,
            "capabilities": manifest["capabilities"],
            "epistemic": {
                "provenance_stats": manifest["provenance_stats"],
                "recent_events": recent_evts
            },
            "project_intent": {
                "active_project": active_proj,
                "past_projects": past_projs
            },
            "history_memory": {
                "recent_turns": recent_turns
            },
            "interoception": self.interoception.get_summary(),
            "autobiographical_narrative": manifest.get("autobiographical_narrative", ""),
            "chronesthesia": (
                self.autobiographical_engine.mental_time_travel(text)
                if self.autobiographical_engine and any(k in text.lower() for k in ("earlier", "past", "changed", "evolved", "time travel", "history", "previous", "before", "timeline"))
                else None
            ),
            "system_status": {
                "equilibrium": bool(self.interoception.internal_viability >= 0.5),
                "manifold_dimension": 256,
                "active_concepts": manifest["total_concepts"],
                "status": "operating in stable equilibrium" if self.interoception.internal_viability >= 0.5 else "recovering from cognitive strain",
                "cognitive_strain": self.interoception.cognitive_strain,
                "epistemic_entropy": self.interoception.epistemic_entropy,
                "homeostatic_energy": self.interoception.homeostatic_energy,
                "liquid_tau": self.interoception.liquid_tau,
                "self_healing_events": self.interoception.self_healing_events,
                "strange_loop_interventions": self.interoception.strange_loop_interventions,
                "internal_viability": self.interoception.internal_viability,
                "interoceptive_narrative": self.interoception.get_narrative(),
                "os_grounded_summary": (
                    self.os_grounding.get_grounded_summary() if self.os_grounding else None
                )
            }
        }
        return StateDict(state_data)
