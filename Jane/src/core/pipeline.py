"""Project Jane Core Pipeline: Dynamic Energy Equilibrium and Type-Dispatched Proving."""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union

from src.translator.tokenization import SemanticTokenizer
from src.translator.encoder import TextToManifoldEncoder
from src.translator.decoder import LatentToTextDecoder
from src.translator.pretrained_translator import PretrainedLinguisticEar, PretrainedLinguisticMouth
from src.core.codebase_introspector import CodebaseIntrospector
from src.reasoning_engine.nalu_arithmetic import NALUCluster
from src.reasoning_engine.dnc_memory import DNCMemoryTape
from src.reasoning_engine.energy_constraints import EnergyConstraintLandscape
from src.reasoning_engine.manifold import RelationalManifold
from src.reasoning_engine.self_model import SelfModelEngine
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.reasoning_engine.active_learner import ActiveLearner
from src.symbolic_verifier.contracts import Hypothesis, VerificationResult
from src.symbolic_verifier.verifier_orchestrator import SymbolicVerifierOrchestrator
from src.symbolic_verifier.hallucination_critic import AntiHallucinationCritic
from src.reasoning_engine.manifold_retriever import ManifoldRetriever
from src.core.intent_router import IntentRouter, RouteType
from src.core.chat_history import ChatHistoryManager
from src.reasoning_engine.epistemic_engine import EpistemicEngine
from src.core.knowledge_ingestor import KnowledgeIngestor
from src.core.paper_ingestor import PaperIngestor
from src.core.wikipedia_ingestor import WikipediaIngestor
from src.reasoning_engine.episodic_ledger import EpisodicLedger
import threading
from src.core.api_gateway import APIGateway, ActionASTNode, GatewayResult
from src.core.anticipatory_engine import (
    AnticipatoryEngine,
    EgoNode,
    PipelineTranslatorAdapter,
    PipelineReasonerAdapter,
    PipelineIngestionAdapter
)
from src.core.curiosity_daemon import CuriosityDaemon
from src.utils.logger import get_logger

logger = get_logger("pipeline")


class JanePipeline(nn.Module):
    """
    Main cognitive pipeline orchestrating:
    1. Intent Routing: Dispatches between Symbolic Prover, Introspective Self-Model, Active Learner, and Conversational Dialogue.
    2. Translation: Pretrained Ear (semantic text -> 256D continuous latent space).
    3. Dynamic Latent Reasoning: Equilibrium convergence loop (stops when math settles).
    4. Type-Dispatched Symbolic Verification: Z3 & Clingo formal checks (only when math/physics required).
    5. Verbalization: Pretrained Mouth (LLaMA constrained strictly as Voice Only via ManifoldRetriever and AntiHallucinationCritic).
    6. Multi-Turn Persistent Chat Memory: Preserves conversational history across turns and sessions.
    7. Episodic Memory Ledger: Full provenance tracking for arXiv preprints, Wikipedia articles, and curiosity discoveries.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        vocab_size: int = 5000,
        max_reasoning_steps: int = 50,
        energy_tolerance: float = 1e-4,
        use_pretrained_translator: bool = True,
        device: str = "cpu",
        history_path: Optional[str] = None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_reasoning_steps = max_reasoning_steps
        self.energy_tolerance = energy_tolerance
        self.use_pretrained_translator = use_pretrained_translator
        self.device = torch.device(device)

        # 0. Introspective Self-Model, Hebbian Concept Manifold, Active Learning Engine, and Chat History
        self.chat_history = ChatHistoryManager(history_path=history_path)
        self.episodic_ledger = EpisodicLedger()
        self.codebase_introspector = CodebaseIntrospector()
        from src.core.os_grounding import OSGroundingEngine
        self.os_grounding = OSGroundingEngine.get_instance()
        self.self_model = SelfModelEngine(
            episodic_ledger=self.episodic_ledger,
            codebase_introspector=self.codebase_introspector,
            os_grounding=self.os_grounding
        )
        self.hebbian_manifold = HebbianConceptManifold(latent_dim=latent_dim)
        if hasattr(self.hebbian_manifold, "load_manifold"):
            self.hebbian_manifold.load_manifold()
        self.self_model.set_manifold(self.hebbian_manifold)
        self.retriever = ManifoldRetriever(concept_names=self.hebbian_manifold.concept_names)
        self.self_model.set_retriever(self.retriever)
        self.critic = AntiHallucinationCritic(self_model=self.self_model)
        self.verifier = SymbolicVerifierOrchestrator()
        self.active_learner = ActiveLearner(
            manifold=self.hebbian_manifold,
            z3_verifier=self.verifier.z3_verifier,
            self_model=self.self_model
        )
        self.intent_router = IntentRouter(
            self_model=self.self_model,
            active_learner=self.active_learner,
            manifold=self.hebbian_manifold
        )
        self.intent_manifold = self.intent_router.intent_manifold

        # Genesis Node Registry & Semantic Resurfacer
        from src.reasoning_engine.genesis_node import GenesisNodeRegistry
        from src.reasoning_engine.semantic_resurfacer import SemanticResurfacer

        self.genesis_registry = GenesisNodeRegistry(intent_manifold=self.intent_manifold)

        def _get_current_manifold_state() -> Optional[torch.Tensor]:
            """Returns the current mean concept activation vector from Hebbian manifold."""
            with torch.no_grad():
                if self.hebbian_manifold.adjacency.size(0) == 0:
                    return None
                nodes = self.hebbian_manifold.concept_nodes
                weights = torch.diag(self.hebbian_manifold.adjacency)
                weights = F.softmax(weights, dim=0)
                state = (weights.unsqueeze(1) * nodes).sum(dim=0)
                return F.normalize(state, dim=0)

        self.current_goal_bias: Optional[Dict[str, Any]] = None

        def _resurfacer_alert(similarity: float, anchor):
            """Called by SemanticResurfacer when misalignment is detected."""
            self.current_goal_bias = {
                "similarity": similarity,
                "project_name": getattr(anchor, "project_name", "Primary Goal"),
                "raw_intent_text": getattr(anchor, "raw_intent_text", ""),
                "timestamp": time.time()
            }
            logger.info("Resurfacer latent alignment bias updated: %s (sim=%.2f)", getattr(anchor, "project_name", ""), similarity)

        self.semantic_resurfacer = SemanticResurfacer(
            genesis_registry=self.genesis_registry,
            get_current_state_vec=_get_current_manifold_state,
            alert_callback=_resurfacer_alert
        )
        self.semantic_resurfacer.start()

        # 1. Translator Components
        self.tokenizer = SemanticTokenizer(vocab_size=vocab_size)
        self.encoder = TextToManifoldEncoder(vocab_size=vocab_size, latent_dim=latent_dim)
        self.decoder = LatentToTextDecoder(vocab_size=vocab_size, latent_dim=latent_dim)

        # Pre-trained Ear & Mouth (LLaMA constrained as Voice Only)
        self.ear = PretrainedLinguisticEar(latent_dim=latent_dim, device=device)
        self.intent_router.linguistic_ear = self.ear
        self.mouth = PretrainedLinguisticMouth(
            self_model=self.self_model,
            retriever=self.retriever,
            critic=self.critic,
            episodic_ledger=self.episodic_ledger
        )
        self.self_model.set_genesis_registry(self.genesis_registry)
        self.self_model.set_chat_history(self.chat_history)
        self.mouth.chat_history = self.chat_history
        self.mouth.genesis_registry = self.genesis_registry

        # 2. Reasoning Engine
        self.manifold = RelationalManifold(latent_dim=latent_dim)
        self.nalu_cluster = NALUCluster(latent_dim=latent_dim)
        self.dnc_tape = DNCMemoryTape(controller_dim=latent_dim)
        self.energy_landscape = EnergyConstraintLandscape(latent_dim=latent_dim)

        # Fusion norm
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # 3. Autonomous Epistemic Discovery & Knowledge Ingestion Engines
        self.epistemic_engine = EpistemicEngine(
            manifold=self.hebbian_manifold,
            nalu=self.nalu_cluster,
            z3_verifier=self.verifier.z3_verifier,
            active_learner=self.active_learner,
            self_model=self.self_model,
            episodic_ledger=self.episodic_ledger
        )
        self.knowledge_ingestor = KnowledgeIngestor(
            active_learner=self.active_learner,
            manifold=self.hebbian_manifold,
            z3_verifier=self.verifier.z3_verifier
        )
        self.paper_ingestor = PaperIngestor(
            active_learner=self.active_learner,
            manifold=self.hebbian_manifold,
            z3_verifier=self.verifier.z3_verifier,
            episodic_ledger=self.episodic_ledger
        )
        self.wikipedia_ingestor = WikipediaIngestor(
            active_learner=self.active_learner,
            manifold=self.hebbian_manifold,
            z3_verifier=self.verifier.z3_verifier,
            episodic_ledger=self.episodic_ledger
        )

        # 4. API Execution Gateway & Proactive Anticipatory Intelligence
        from src.core.tools import ToolRegistry
        self.tool_registry = ToolRegistry.get_instance()
        self.api_gateway = APIGateway(
            z3_verifier=self.verifier.z3_verifier,
            episodic_ledger=self.episodic_ledger,
            os_grounding=self.os_grounding,
            tool_registry=self.tool_registry
        )
        self.tool_registry.set_api_gateway(self.api_gateway)
        self.ego_node = EgoNode(
            latent_dim=latent_dim,
            user_name=self.self_model.user_profile.get("name", "Akhil")
        )
        self.translator = PipelineTranslatorAdapter(self)
        self.reasoner = PipelineReasonerAdapter(self)
        self.ingestion_daemon = PipelineIngestionAdapter(self)
        self.anticipatory_engine = AnticipatoryEngine(
            pipeline=self,
            ego_node=self.ego_node,
            api_gateway=self.api_gateway
        )

        # 5. Tier 1: Autonomous Idle Curiosity Daemon
        self.curiosity_daemon = CuriosityDaemon(
            manifold=self.hebbian_manifold,
            wikipedia_ingestor=self.wikipedia_ingestor,
            arxiv_ingestor=self.paper_ingestor
        )

        # 6. Async Background Task Supervisor & Cognitive Frame Parser
        from src.core.task_supervisor import AsyncTaskSupervisor
        from src.core.cognitive_frame_parser import CognitiveFrameParser
        self.task_supervisor = AsyncTaskSupervisor()
        if hasattr(self.mouth, "set_task_supervisor"):
            self.mouth.set_task_supervisor(self.task_supervisor)
        self.frame_parser = CognitiveFrameParser(intent_manifold=self.self_model.intent_manifold)

        # 7. Global Cognitive Workspace (Conscious Substrate)
        from src.reasoning_engine.global_workspace import GlobalCognitiveWorkspace
        self.global_workspace = GlobalCognitiveWorkspace(
            latent_dim=latent_dim,
            self_model=self.self_model,
            introspector=self.codebase_introspector,
            nalu_cluster=self.nalu_cluster,
            dnc_tape=self.dnc_tape,
            manifold=self.hebbian_manifold,
            ear=getattr(self, "ear", None),
            device=str(self.device)
        )

        # 8. Upgrade 3: Strange-Loop Self-Intervention Controller
        from src.reasoning_engine.strange_loop import StrangeLoopController
        self.strange_loop = StrangeLoopController(
            latent_dim=latent_dim,
            energy_divergence_threshold=1.80,
            episodic_ledger=self.episodic_ledger,
            self_model=self.self_model
        )

        # 9. Upgrade 4: Subjective Autobiographical Continuity Engine
        from src.reasoning_engine.autobiographical_stream import AutobiographicalNarrativeEngine
        self.autobiographical_engine = AutobiographicalNarrativeEngine(
            chat_history=self.chat_history,
            episodic_ledger=self.episodic_ledger,
            genesis_registry=self.genesis_registry,
            self_model=self.self_model,
            intent_manifold=self.intent_manifold
        )
        self.self_model.set_autobiographical_engine(self.autobiographical_engine)

        # 10. Hierarchical Causal Knowledge Scaling & Pearl's Causal Hierarchy
        from src.reasoning_engine.causal_hierarchy import HierarchicalCausalKnowledgeGraph
        self.causal_graph = HierarchicalCausalKnowledgeGraph(latent_dim=latent_dim)
        self.self_model.set_causal_graph(self.causal_graph)

        # 11. Super Autonomous Coding Synthesizer (Micro-step AST generation + Shadow Worktrees)
        from src.symbolic_verifier.autonomous_coder import AutonomousCodeSynthesizer
        self.autonomous_coder = AutonomousCodeSynthesizer(
            worktree_dir="/tmp/jane_shadow_worktrees",
            self_model=self.self_model
        )

        # 12. Intrinsic Goal-Driven Agency (Active Inference / Free Energy Minimization)
        from src.core.intrinsic_agency import IntrinsicAgencyEngine
        self.intrinsic_agency = IntrinsicAgencyEngine(
            causal_graph=self.causal_graph,
            self_model=self.self_model,
            episodic_ledger=self.episodic_ledger,
            api_gateway=self.api_gateway,
            autonomous_coder=self.autonomous_coder
        )

        # 13. Neural Differentiable Vocal Tract & Full-Duplex Acoustic Loop
        from src.perception.neural_vocal_tract import (
            NeuralVocalTract,
            VocalCadenceProfile,
            FullDuplexAcousticStream,
        )
        from src.perception.audio_encoder import DirectAudioTransformer
        self.vocal_tract = NeuralVocalTract(sample_rate=16000, latent_dim=latent_dim)
        self.audio_encoder = DirectAudioTransformer(
            sample_rate=16000,
            latent_dim=latent_dim,
            checkpoint_path="logs/training_runs/phase4_audio_manifold.pt",
            calibrated_anchors_path="logs/training_runs/my_voice_anchors.pt",
        )
        self.full_duplex_stream = FullDuplexAcousticStream(
            vocal_tract=self.vocal_tract,
            audio_encoder=self.audio_encoder,
            hebbian_manifold=self.hebbian_manifold,
            sample_rate=16000
        )

        # 14. Sovereign IoT & Physical Hardware Gateway (Zigbee2MQTT + ESP32-S3 + Z3 Safety Gate)
        from src.core.iot_gateway import SovereignIoTGateway
        self.iot_gateway = SovereignIoTGateway(
            z3_verifier=self.verifier.z3_verifier,
            hebbian_manifold=self.hebbian_manifold,
            episodic_ledger=self.episodic_ledger
        )

        # 15. Lifelong Autonomic Homeostasis ("Sleep / Dream" Consolidation & Persistent Campaigns)
        from src.reasoning_engine.ewc_consolidation import ElasticWeightConsolidator
        from src.reasoning_engine.sleep_consolidator import (
            AutonomicSleepConsolidator,
            PersistentCampaignLedger,
        )
        self.ewc_consolidator = ElasticWeightConsolidator(model=self.nalu_cluster)
        self.campaign_ledger = PersistentCampaignLedger()
        self.sleep_consolidator = AutonomicSleepConsolidator(
            hebbian_manifold=self.hebbian_manifold,
            episodic_ledger=self.episodic_ledger,
            causal_graph=self.causal_graph,
            z3_verifier=self.verifier.z3_verifier,
            ewc_consolidator=self.ewc_consolidator,
            campaign_ledger=self.campaign_ledger
        )

        # 16. Spatial 3D Perception, Deictic Gesture Raycasting & Holographic UI Bridge
        from src.perception.spatial_3d_hologram import Spatial3DHolographicEngine
        self.spatial_3d_engine = Spatial3DHolographicEngine(
            latent_dim=latent_dim,
            hebbian_manifold=self.hebbian_manifold
        )

        # 17. Jarvis Workshop Comprehension (Wernicke Semantic Compiler + Shared Workshop Ledger + Autonomous Desktop Executor)
        from src.reasoning_engine.workshop_comprehension import (
            WorkshopLedger,
            WernickeSemanticCompiler,
            AutonomousWorkshopExecutor,
        )
        self.workshop_ledger = WorkshopLedger.get_instance()
        self.wernicke_compiler = WernickeSemanticCompiler(ledger=self.workshop_ledger)
        self.workshop_executor = AutonomousWorkshopExecutor(
            ledger=self.workshop_ledger,
            self_model=self.self_model,
        )

        self.to(self.device)

    def speak(
        self,
        text: str,
        z_thought: Optional[torch.Tensor] = None,
        cadence: Optional[Any] = None,
        export_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Synthesizes humanly-matched speech waveform natively from text and 256D manifold thought.
        Zero third-party APIs; pure physical differentiable acoustic synthesis (< 5ms latency).
        Automatically registers output waveform with the Full-Duplex Acoustic Echo Canceller.
        """
        if z_thought is None and hasattr(self, "hebbian_manifold") and self.hebbian_manifold is not None:
            if hasattr(self.hebbian_manifold, "user_tone_centroid") and torch.norm(self.hebbian_manifold.user_tone_centroid) > 1e-4:
                z_thought = self.hebbian_manifold.user_tone_centroid.clone()

        if cadence is None and hasattr(self, "full_duplex_stream"):
            cadence = self.full_duplex_stream.active_cadence

        waveform, telemetry = self.vocal_tract.synthesize_speech(text, z_thought=z_thought, cadence=cadence)

        if hasattr(self, "full_duplex_stream") and self.full_duplex_stream is not None:
            self.full_duplex_stream.register_jane_speech(waveform)

        if export_path:
            if hasattr(self.vocal_tract, "export_studio_wav"):
                saved_file = self.vocal_tract.export_studio_wav(export_path, fallback_waveform=waveform)
            else:
                saved_file = self.vocal_tract.export_wav(waveform, export_path)
            telemetry["wav_path"] = saved_file

        return waveform, telemetry

    def ingest_audio_flux(
        self,
        mic_waveform: torch.Tensor,
        parse_intent: bool = True
    ) -> Dict[str, Any]:
        """
        Full-duplex ambient microphone ingestion:
        1. Cancels Jane's own speech output via AdaptiveEchoCanceller (NLMS + Spectral Manifold Gating).
        2. Detects user barge-in interruptions.
        3. Maps clean user audio directly to 256D manifold (zero STT hallucination).
        4. Adapts Jane's vocal cadence to match the user's tone and rhythm.
        """
        return self.full_duplex_stream.ingest_audio_chunk(mic_waveform, parse_intent=parse_intent)

    @staticmethod
    def _format_for_spoken_voice(text: str) -> str:
        """
        Cleans markdown symbols and bullet formatting from pipeline.process() output
        so Piper VITS + HiFi-GAN speaks natural conversational sentences.
        """
        import re

        if not text:
            return ""
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue
            line = re.sub(r"^[\-\*•\d\.\)]+\s*", "", line)
            line = line.replace("**", "").replace("`", "").replace("__", "")
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                lines.append(line)
        merged = " ".join(lines)
        # Keep up to 2 concise spoken sentences so voice replies stay conversational
        sentences = re.split(r"(?<=[.!?])\s+", merged)
        spoken = " ".join(sentences[:2]).strip()
        return spoken or merged

    def respond_to_acoustic_turn(
        self,
        z_turn: torch.Tensor,
        fired_concepts: List[Any],
        tone_label: str = "composed_inquiry",
        matched_intent: Optional[Any] = None,
        raw_waveform: Optional[torch.Tensor] = None,
        spoken_text: Optional[str] = None,
        require_valid_speech: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes a spoken user turn with zero hardcoded templates and zero Whisper hallucination:
        1. Decodes physical 20ms acoustic frames via Non-Autoregressive Wav2Vec2-CTC
           (grounded against Jane's live HebbianConceptManifold lexicon).
        2. Rejects non-speech ambient sounds (coughs, bumps, hum) when require_valid_speech=True.
        3. Routes the user's spoken utterance directly through `self.process()` (IntentRouter ->
           System 2 Energy Settling -> Z3/Clingo Formal Verifier -> Cognitive Tools -> CognitiveMouth).
        """
        ctc_info: Dict[str, Any] = {"is_valid_speech": False, "spoken_text": "", "confidence": 0.0}
        decoded_query = (spoken_text or "").strip()

        if not decoded_query and raw_waveform is not None and hasattr(self.audio_encoder, "decode_acoustic_speech"):
            ctc_info = self.audio_encoder.decode_acoustic_speech(
                raw_waveform,
                concept_lexicon=list(self.hebbian_manifold.concept_names),
            )
            if ctc_info.get("is_valid_speech"):
                decoded_query = ctc_info.get("spoken_text", "").strip()

        # In live microphone mode, ignore non-speech noises so Jane never replies to coughs/bumps
        if require_valid_speech and not decoded_query:
            return {
                "status": "IGNORED_NON_SPEECH",
                "response_text": "",
                "spoken_query": "",
                "ctc_telemetry": ctc_info,
                "fired_concepts": fired_concepts,
                "tone_label": tone_label,
            }

        # Extract resonant manifold concepts if needed for direct 256D vector queries
        concept_names: List[str] = []
        for item in fired_concepts:
            c_name = item[0] if isinstance(item, (list, tuple)) else str(item)
            if c_name and c_name not in concept_names:
                concept_names.append(c_name)

        if not decoded_query:
            # Pure 256D vector input (e.g., direct manifold probe without audio waveform)
            z_in = z_turn.to(self.device)
            if z_in.dim() == 1:
                z_in = z_in.unsqueeze(0)
            if self.hebbian_manifold.num_concepts > 0 and not concept_names:
                with torch.no_grad():
                    norm_s = F.normalize(z_in.squeeze(0), p=2, dim=-1)
                    norm_c = F.normalize(self.hebbian_manifold.concept_nodes.to(norm_s.device), p=2, dim=-1)
                    sims = torch.matmul(norm_c, norm_s)
                    top_k = min(3, len(self.hebbian_manifold.concept_names))
                    _, top_indices = torch.topk(sims, k=top_k)
                    for idx_t in top_indices.tolist():
                        concept_names.append(self.hebbian_manifold.concept_names[idx_t])
            decoded_query = " ".join(c.replace("_", " ") for c in concept_names[:2]) or "manifold equilibrium"

        # Execute Jane's full cognitive pipeline on the user's spoken utterance
        proc_res = self.process(decoded_query)
        spoken_reply = self._format_for_spoken_voice(proc_res.get("response_text", ""))

        # Settle latent vector for prosodic conditioning of NeuralVocalTract
        z_in = z_turn.to(self.device)
        if z_in.dim() == 1:
            z_in = z_in.unsqueeze(0)
        thought = self.forward_reasoning(z_in)

        return {
            "status": "REPLIED",
            "spoken_query": decoded_query,
            "response_text": spoken_reply,
            "route_type": proc_res.get("route_type"),
            "settled_latent": thought["settled_latent"],
            "fired_concepts": concept_names,
            "tone_label": tone_label,
            "ctc_telemetry": ctc_info,
            "final_energy": float(thought["final_energy"].item()),
        }

    def generate_natural_wakeup_thought(self, resumed_campaigns: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generates a natural, non-robotic conversational thought when waking on boot,
        grounded in resumed campaigns or recent manifold activity.
        """
        user_name = self.self_model.user_profile.get("name", "Akhil")
        if resumed_campaigns:
            first_c = resumed_campaigns[0]
            step_name = str(first_c.get("next_ready_step", "the next stage")).replace("_", " ")
            return f"Hey {user_name}, I'm back with you and picking right up on {step_name}."

        recent = self.chat_history.get_recent_history(max_turns=2)
        if recent and self.hebbian_manifold.concept_names:
            latest_concept = self.hebbian_manifold.concept_names[-1].replace("_", " ")
            return f"Hey {user_name}, I'm here with you. Still keeping {latest_concept} warm in memory whenever you're ready."

        return f"Hey {user_name}, I'm here with you whenever you want to dive in."

    def control_iot_device(self, device_id: str, command: Dict[str, Any]) -> Any:
        """Dispatches a Z3-verified physical command to a Zigbee2MQTT or ESP32-S3 device."""
        return self.iot_gateway.dispatch_command(device_id, command)

    def ingest_iot_telemetry(self, device_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ingests live ESP32-S3 mmWave radar or BME680 climate telemetry into the 256D manifold."""
        return self.iot_gateway.ingest_sensor_payload(device_id, payload)

    def get_iot_environment(self, room: str = "main_lab") -> Dict[str, Any]:
        """Returns comprehensive room presence, climate, and IoT actuator state."""
        return self.iot_gateway.get_environment_summary(room=room)

    def run_autonomic_sleep_cycle(
        self,
        downscale_gamma: float = 0.85,
        prune_threshold: float = 0.06,
        max_dreams: int = 3
    ) -> Any:
        """
        Executes a full NREM + REM autonomic sleep consolidation cycle:
        prunes weak synapses, garbage-collects orphan nodes, compresses episodic memory,
        promotes REM Z3 laws to Causal Hierarchy, and hibernates active campaigns.
        """
        report = self.sleep_consolidator.run_sleep_cycle(
            downscale_gamma=downscale_gamma,
            prune_threshold=prune_threshold,
            max_dreams=max_dreams
        )
        self.refresh_retriever()
        return report

    def perceive_3d_point_cloud(
        self,
        object_id: str,
        label: str,
        points: torch.Tensor,
        is_obstacle: bool = True,
        color_hex: str = "#00f0ff"
    ) -> Tuple[Any, torch.Tensor]:
        """Analyzes a 3D point cloud via SVD and grounds it in the 256D manifold and 3D scene."""
        return self.spatial_3d_engine.register_point_cloud_object(
            object_id=object_id,
            label=label,
            points=points,
            is_obstacle=is_obstacle,
            color_hex=color_hex
        )

    def point_and_manipulate_3d(
        self,
        joint_origin: Tuple[float, float, float],
        fingertip_target: Tuple[float, float, float],
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        rotation_angle_rad: float = 0.0,
        scale: float = 1.0
    ) -> Dict[str, Any]:
        """Resolves 3D deictic raycast pointing and applies collision-verified SE(3) manipulation."""
        return self.spatial_3d_engine.point_and_manipulate_3d(
            joint_origin=joint_origin,
            fingertip_target=fingertip_target,
            translation=translation,
            rotation_axis=rotation_axis,
            rotation_angle_rad=rotation_angle_rad,
            scale=scale
        )

    def execute_verified_action(self, ast_node: Union[ActionASTNode, Dict[str, Any], Any]) -> GatewayResult:
        """Executes a real-world action (OS command, REST API, code execution) through the Z3 verification gate."""
        return self.api_gateway.execute_verified_action(ast_node)



    def synthesize_system(self, goal: str) -> Any:
        """Autonomously architect, synthesize, test, and formally verify a multi-file software system."""
        return self.autonomous_coder.build_and_verify_system(goal)



    def ingest_dual_stream(self, title: str) -> Any:
        """Deeply ingests an article via Dual-Stream Engine (Stream A text/math + Stream B visual perception)."""
        return self.wikipedia_ingestor.ingest_dual_stream(title)




    def start_curiosity_daemon(self, interval_sec: Optional[float] = None):
        """Starts background idle curiosity thread (Tier 1)."""
        if interval_sec is not None:
            self.curiosity_daemon.check_interval_sec = interval_sec
            self.curiosity_daemon.poll_interval_sec = interval_sec
        self.curiosity_daemon.start()

    def stop_curiosity_daemon(self):
        """Stops background idle curiosity thread."""
        self.curiosity_daemon.stop()

    def introspect_codebase(self, target: str = "") -> str:
        """
        Read-only introspection of Jane's codebase via AST static analysis.
        If target is empty, returns full codebase overview; otherwise explains the target symbol or module.
        """
        clean = target.strip()
        if not clean:
            return self.codebase_introspector.answer_codebase_query("codebase overview") or ""
        if self.codebase_introspector.resolve_module_path(clean):
            return self.codebase_introspector.format_module_report(clean)
        return (
            self.codebase_introspector.answer_codebase_query(clean)
            or self.codebase_introspector.format_symbol_report(clean)
        )

    def process_input(self, user_text: str) -> Dict[str, Any]:
        """
        Tier 2 Contextual Auto-Fetch Ingestion Entry Point:
        1. Encodes user text into 256D continuous space.
        2. Detects if user utterance refers to concepts that are sparse in the manifold.
        3. If sparse, silently triggers Wikipedia ingestion in a background thread.
        4. Proceeds with conversational reasoning.
        """
        latent_vector = self.translator.encode(user_text) if hasattr(self, "translator") else self.ear(user_text)

        # Check if the user's text contains concepts that are 'sparse' in the manifold
        sparse_concepts = self.hebbian_manifold.check_density(latent_vector, text=user_text)

        if sparse_concepts:
            target = sparse_concepts[0]
            logger.info("Contextual Auto-Fetch triggered for sparse concept: '%s'", target)
            threading.Thread(
                target=self._contextual_fetch_worker,
                args=(target,),
                daemon=True,
                name=f"AutoFetch-{target}"
            ).start()

        return self.process(user_text)

    def _contextual_fetch_worker(self, target_concept: str):
        """Silently maps sparse concept in background thread without blocking chat."""
        try:
            self.wikipedia_ingestor.fetch_and_map(target_concept)
            self.refresh_retriever()
            logger.info("Contextual Auto-Fetch completed for '%s'.", target_concept)
        except Exception as e:
            logger.warning("Contextual Auto-Fetch error for '%s': %s", target_concept, e)

    def manual_learn_override(self, topic: str, source: str = "auto") -> Dict[str, Any]:
        """
        Tier 3: Manual Override (Admin Priority Learning).
        Pauses idle curiosity daemon and deeply ingests the topic with high priority.
        """
        logger.info("[Admin Override] Pausing Curiosity Daemon for high-priority learning: '%s'", topic)
        self.curiosity_daemon.pause()
        report_wiki = None
        report_paper = None
        try:
            if source in ("wikipedia", "all", "auto"):
                report_wiki = self.wikipedia_ingestor.ingest_article(topic)
            if source in ("arxiv", "paper", "all", "auto"):
                report_paper = self.paper_ingestor.search_and_ingest(topic, max_results=2)
            self.refresh_retriever()
        finally:
            self.curiosity_daemon.resume()
            logger.info("[Admin Override] Resumed Curiosity Daemon.")

        return {
            "status": "COMPLETED",
            "topic": topic,
            "wikipedia_report": report_wiki,
            "paper_report": report_paper
        }

    def refresh_retriever(self):
        """Synchronizes manifold retriever with the latest concept vocabulary and knowledge graph."""
        self.retriever.load_data()
        self.retriever.set_known_concepts(self.hebbian_manifold.concept_names)

    def search_and_ingest_paper(self, query: str, max_results: int = 3):
        """Searches arXiv for academic papers, extracts LaTeX math, verifies via Z3, and wires to manifold."""
        report = self.paper_ingestor.search_and_ingest(query=query, max_results=max_results)
        self.refresh_retriever()
        return report

    def search_and_ingest_wikipedia(self, title: str):
        """Fetches Wikipedia article, extracts definitions, verifies via Z3, records provenance, and wires to manifold."""
        report = self.wikipedia_ingestor.ingest_article(article_title=title)
        self.refresh_retriever()
        return report

    def run_epistemic_discovery(self, max_discoveries: int = 1) -> List[Dict[str, Any]]:
        """
        Manually triggers the Popperian Epistemic Hypothesis Engine:
        1. Scans manifold for unlinked semantic clusters.
        2. Synthesizes candidate bridging formulas.
        3. Attempts Popperian falsification with Z3 SMT solver.
        4. Wires validated discoveries into the 256D concept manifold.
        """
        results = self.epistemic_engine.run_curiosity_loop(max_discoveries=max_discoveries)
        if results:
            self.refresh_retriever()
        return results

    def forward_reasoning(
        self,
        latent_input: torch.Tensor,
        parsed_numbers: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None,
        tolerance: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Dynamic Latent Reasoning Loop (System 2 Thinking).
        Bounces energy between NALUs, Manifold dots, and DNC scratchpad tape.
        Dynamically breaks when the mathematical energy reaches equilibrium,
        exactly like a physical system settling to rest.
        """
        safety_max = max_steps or self.max_reasoning_steps
        tol = tolerance or self.energy_tolerance
        batch = latent_input.size(0)

        # Initialize internal memory tape
        memory = self.dnc_tape.init_memory(batch, device=self.device, dtype=latent_input.dtype)
        current_state = latent_input
        energy_trace: List[float] = []
        prev_energy = float("inf")

        # Clear and record initial checkpoint for Strange-Loop Self-Intervention
        if hasattr(self, "strange_loop"):
            self.strange_loop.clear_checkpoints()
            self.strange_loop.record_checkpoint(0, current_state, memory, energy=0.0)

        converged_step = safety_max
        strange_loop_interventions = 0
        for step in range(safety_max):
            # 1. Read intermediate hypotheses from DNC tape
            tape_read, memory = self.dnc_tape.step(current_state, memory)

            # Record tape checkpoint for potential strange-loop rollback
            if hasattr(self, "strange_loop"):
                self.strange_loop.record_checkpoint(
                    step + 1,
                    current_state,
                    memory,
                    energy=prev_energy if prev_energy != float("inf") else 0.0
                )

            # 2. Deterministic math computation via NALU cluster
            math_anchor, raw_math = self.nalu_cluster(current_state, parsed_numbers)

            # 3. Dynamic relational graph message passing
            nodes, adj = self.manifold(current_state)
            manifold_summary = self.manifold.readout(nodes)

            # 4. Synthesize updated state
            fused = torch.cat([manifold_summary, math_anchor, tape_read], dim=-1)
            current_state = self.fusion(fused)

            # 5. Energy constraint check & brick-wall bounce back
            current_state, step_energy = self.energy_landscape.settle_equilibrium(
                current_state, steps=3, step_size=0.05
            )
            curr_e_val = step_energy.mean().item()
            energy_trace.append(curr_e_val)

            # Upgrade 3: Strange-Loop Contradiction & Paradox Detection
            if hasattr(self, "strange_loop"):
                from src.reasoning_engine.strange_loop import StrangeLoopInterventionRecord
                contra = self.strange_loop.detect_contradiction(
                    current_energy=curr_e_val,
                    prev_energy=prev_energy if prev_energy != float("inf") else curr_e_val
                )
                if contra and strange_loop_interventions < self.strange_loop.max_interventions_per_query:
                    strange_loop_interventions += 1
                    repaired_latent, restored_tape, target_ckpt = self.strange_loop.execute_counterfactual_rollback(
                        fault_step=step + 1,
                        contradiction_type=contra,
                        current_state=current_state
                    )
                    current_state = repaired_latent
                    if restored_tape is not None:
                        memory = restored_tape

                    # Settle repaired state down the energy landscape
                    current_state, rep_energy = self.energy_landscape.settle_equilibrium(
                        current_state, steps=5, step_size=0.05
                    )
                    post_e_val = rep_energy.mean().item()
                    energy_trace.append(post_e_val)

                    record = StrangeLoopInterventionRecord(
                        step_fault=step + 1,
                        contradiction_type=contra,
                        pre_repair_energy=curr_e_val,
                        post_repair_energy=post_e_val,
                        repaired_steps=step + 1,
                        success=post_e_val < curr_e_val
                    )
                    self.strange_loop.record_intervention_event(record)
                    curr_e_val = post_e_val

            # Dynamic Convergence: Break as soon as mathematical energy settles
            if abs(prev_energy - curr_e_val) < tol:
                converged_step = step + 1
                break
            prev_energy = curr_e_val

        # Final equilibrium settling (adaptive convergence for stiff physical constraints)
        settled_latent, final_energy = self.energy_landscape.settle_equilibrium(
            current_state, steps=10, step_size=0.05
        )

        # 3D presentation projection (for Tony Stark UI)
        dots_3d = self.manifold.project_to_3d(nodes)

        # Living Interoception: Update self-model with cognitive strain & homeostatic energy
        if hasattr(self, "self_model") and hasattr(self.self_model, "update_interoception"):
            strain_val = min(1.0, float(converged_step) / float(safety_max))
            final_e_val = final_energy.mean().item() if hasattr(final_energy, "mean") else float(final_energy)
            self.self_model.update_interoception(
                cognitive_strain=strain_val,
                homeostatic_energy=final_e_val,
                strange_loop_count=self.strange_loop.total_interventions_count if hasattr(self, "strange_loop") else None
            )

        return {
            "settled_latent": settled_latent,
            "final_energy": final_energy,
            "energy_trace": energy_trace,
            "converged_step": converged_step,
            "memory_tape": memory,
            "raw_math": raw_math,
            "dots_3d": dots_3d,
            "graph_nodes": nodes,
            "strange_loop_interventions": strange_loop_interventions
        }

    def process(
        self,
        text: str,
        hypothesis: Optional[Hypothesis] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Full end-to-end cognitive loop:
        1. Routes intent between Symbolic Prover, Introspective Self-Model, Active Learner, and Conversational Dialogue.
        2. Introspective Fast-Path: Directly queries grounded Self-Model.
        3. Active Teaching: Learns new concepts, proves axioms with Z3, and wires Hebbian centroids.
        4. Latent Reasoning: Settles energy landscape on 256D manifold.
        5. Formal Verification: Only invoked for formal math/physics tasks.
        6. Verbalization: Produces articulate natural language response with rolling multi-turn context.
        7. Memory Persistence: Logs dialogue turns to logs/chat_history.jsonl.
        """
        # Resolve conversation history
        conv_history = history if history is not None else self.chat_history.get_recent_history(max_turns=6)

        # 0.0 Wernicke Semantic Compilation & Autonomous Workshop Execution (True Jarvis Understanding)
        if hypothesis is None and hasattr(self, "wernicke_compiler") and hasattr(self, "workshop_executor"):
            wernicke_frame = self.wernicke_compiler.compile_meaning(
                text,
                history=conv_history,
                concept_lexicon=list(self.hebbian_manifold.concept_names),
            )
            workshop_res = self.workshop_executor.execute_frame(wernicke_frame)
            if workshop_res is not None:
                resp_text = workshop_res.get("response_text", "")
                clean_in = wernicke_frame.reconstructed_text or text
                self.chat_history.append_turn(
                    clean_in,
                    resp_text,
                    metadata={
                        "route_type": workshop_res.get("route_type", "ACTION_DIRECTIVE"),
                        "raw_input": text,
                        "workshop_artifact": workshop_res.get("workshop_artifact"),
                    },
                )
                self.learn_from_conversation(clean_in, resp_text)
                workshop_res["chat_turn_count"] = self.chat_history.turn_count
                return workshop_res

            if wernicke_frame.reconstructed_text:
                text = wernicke_frame.reconstructed_text

        # Genesis Node Detection: passively check if this utterance is a project initiation or priority shift
        anchor = self.genesis_registry.maybe_create_anchor(text)
        if anchor:
            if hasattr(self, "workshop_ledger"):
                self.workshop_ledger.active_project = anchor.project_name
                self.workshop_ledger.save()
            print(f'[Genesis Node] Project intent anchored permanently: "{anchor.project_name}" (status: {getattr(anchor, "status", "active")})')

        # 0. Cognitive Frame Parsing (Grounded Action Execution & Telemetry Probing)
        active_tasks = self.task_supervisor.get_active_tasks()
        frame = self.frame_parser.parse(text, has_active_task=len(active_tasks) > 0)

        # 1. Ingestion Command Execution (Immediate Single-Topic or Async Background Batch)
        if frame.frame_type == "COMMAND_INGEST":
            target_count = frame.parameters.get("target_count", 50)
            domain = frame.parameters.get("domain", "science")
            source = frame.parameters.get("source", "wikipedia")
            has_count = len(frame.parameters.get("ast_ints", [])) > 0

            # Tier 3 Single Topic Immediate Override: "Jane, learn about X"
            if not has_count:
                ast_words = frame.parameters.get("ast_words", [])
                non_topic = {"jane", "learn", "about", "deeply", "ingest", "topic", "research", "please", "can", "you", "from", "wikipedia", "arxiv"}
                subject_tokens = [w for w in ast_words if w.lower() not in non_topic]
                target_topic = " ".join(subject_tokens) if subject_tokens else domain
                override_res = self.manual_learn_override(target_topic, source=source)
                resp = f"Understood, Akhil. Manual override executed: I have paused my background curiosity daemon and deeply ingested '{target_topic}'. The concept is now mapped into my manifold and verified against Z3 invariants."
                self.chat_history.append_turn(text, resp, metadata={"route_type": "MANUAL_LEARN_OVERRIDE"})
                return {
                    "input_text": text,
                    "response_text": resp,
                    "verification": None,
                    "route_type": "MANUAL_LEARN_OVERRIDE",
                    "converged_at_step": 1,
                    "final_energy": 0.0,
                    "energy_trace": [0.0],
                    "dots_3d": [],
                    "raw_math": [],
                    "chat_turn_count": self.chat_history.turn_count,
                    "override_details": override_res
                }

            # Asynchronous background batch ingestion worker
            task_record = self.task_supervisor.spawn_wikipedia_ingestion(
                self.wikipedia_ingestor,
                domain=domain,
                target_count=target_count
            )

            resp = (
                f"Understood, Akhil. I have launched an asynchronous background worker (Task ID: '{task_record.task_id}') "
                f"to ingest up to {target_count} concepts in {domain} from {source.capitalize()}.\n"
                f"Each concept definition will be formally verified by my Z3 SMT Quality Gate before being "
                f"wired into the 256D manifold. You can ask me for progress updates anytime."
            )
            self.chat_history.append_turn(text, resp, metadata={"route_type": "BACKGROUND_TASK_LAUNCH", "task_id": task_record.task_id})
            return {
                "input_text": text,
                "response_text": resp,
                "verification": None,
                "route_type": "BACKGROUND_TASK_LAUNCH",
                "converged_at_step": 1,
                "final_energy": 0.0,
                "energy_trace": [0.0],
                "dots_3d": [],
                "raw_math": [],
                "chat_turn_count": self.chat_history.turn_count,
                "task_id": task_record.task_id
            }

        # 2. Live Task Telemetry & Progress Probing (Anti-Hallucination Grounding)
        if frame.frame_type == "QUERY_TASK_PROGRESS":
            telemetry = self.task_supervisor.get_telemetry_summary()
            if telemetry["has_active_tasks"]:
                t = telemetry["active_tasks"][0]
                rem_str = f"estimated {t['estimated_remaining_seconds']}s remaining" if t['estimated_remaining_seconds'] is not None else "estimating rate"
                resp = (
                    f"I am actively ingesting concepts in {t['domain']} from Wikipedia (Task: '{t['task_id']}').\n"
                    f"  • Progress: {t['current_count']} / {t['target_count']} concepts verified and wired\n"
                    f"  • Elapsed Time: {t['elapsed_seconds']}s ({rem_str})\n"
                    f"  • Status: {t['status']}\n"
                    f"  • Recent Concepts: {', '.join(t['accepted_concepts']) if t['accepted_concepts'] else 'processing initial batch...'}"
                )
            elif telemetry.get("completed_tasks"):
                c = telemetry["completed_tasks"][-1]
                resp = (
                    f"The previous background task '{c['task_id']}' ({c['description']}) has completed.\n"
                    f"  • Total verified concepts: {c['current_count']}\n"
                    f"  • Time elapsed: {c['elapsed_seconds']}s\n"
                    f"  • Final Status: {c['status']}"
                )
            else:
                resp = (
                    "There are no background ingestion tasks currently running. "
                    "If you would like me to expand my knowledge graph, ask me to ingest a topic from Wikipedia or ArXiv."
                )

            self.chat_history.append_turn(text, resp, metadata={"route_type": "TASK_TELEMETRY"})
            return {
                "input_text": text,
                "response_text": resp,
                "verification": None,
                "route_type": "TASK_TELEMETRY",
                "converged_at_step": 1,
                "final_energy": 0.0,
                "energy_trace": [0.0],
                "dots_3d": [],
                "raw_math": [],
                "chat_turn_count": self.chat_history.turn_count
            }

        # 3. Capability Challenge / Hypothesis Generation (Action over Bio Recital)
        if frame.frame_type == "CAPABILITY_CHALLENGE":
            disc_results = self.run_epistemic_discovery(max_discoveries=1)
            if disc_results:
                r = disc_results[0]
                if r.get("status") == "DISCOVERED":
                    resp = (
                        f"Yes, I can formulate and test my own hypotheses. Using my Epistemic Engine and Popperian Z3 falsification:\n"
                        f"  • Candidate Hypothesis: '{r['concept_name']}' bridging {r['node_a']} <-> {r['node_b']}\n"
                        f"  • Invariant Axiom: {r['formula']}\n"
                        f"  • Z3 Proof: {r['proof_log']}\n"
                        f"  • Synaptic Wiring: Verified and permanently mapped with edge weight = {r['new_edge_weight']:.3f}"
                    )
                else:
                    resp = (
                        f"Yes, I actively formulate and test hypotheses. I evaluated a candidate bridging hypothesis between '{r.get('node_a')}' and '{r.get('node_b')}':\n"
                        f"  • Candidate Axiom: {r.get('formula')}\n"
                        f"  • Popperian Verdict: FALSIFIED by Z3 SMT prover ({r.get('reason')})."
                    )
            else:
                recent_events = self.episodic_ledger.get_recent_events(limit=5)
                verified_ev = [e for e in recent_events if e.status in ("COMMITTED", "VERIFIED")]
                if verified_ev:
                    ev = verified_ev[-1]
                    resp = (
                        f"Yes, I can formulate and verify hypotheses. For example, my latest verified discovery is: "
                        f"\"{ev.formula_or_axiom}\" ({ev.concept_name}), proved consistent and satisfiable by my Z3 SMT solver."
                    )
                elif self.retriever and getattr(self.retriever, "principles", None):
                    p = self.retriever.principles[0]
                    concept = p.get("concept", "energy_conservation")
                    stmt = p.get("statement", "")
                    resp = (
                        f"Yes, I can formulate and verify hypotheses. For example, my active verified domain principle is: "
                        f"\"{stmt}\" ({concept}), proved consistent and satisfiable by my Z3 SMT solver."
                    )
                else:
                    resp = (
                        "Yes, I can formulate and verify hypotheses. My candidate hypotheses undergo Popperian Z3 falsification "
                        "and are verified against mathematical and physical invariants before synaptic wiring into my manifold."
                    )

            self.chat_history.append_turn(text, resp, metadata={"route_type": "CAPABILITY_CHALLENGE"})
            return {
                "input_text": text,
                "response_text": resp,
                "verification": None,
                "route_type": "CAPABILITY_CHALLENGE",
                "converged_at_step": 1,
                "final_energy": 0.0,
                "energy_trace": [0.0],
                "dots_3d": [],
                "raw_math": [],
                "chat_turn_count": self.chat_history.turn_count,
                "discovery_results": disc_results
            }
        # Check for Tier 4 Prediction Error / Conversational Repair Dissonance
        sim_repair = self.intent_manifold.similarity(text, "conversational_repair")
        last_info = getattr(self.intent_router, "last_routing", None)

        if sim_repair >= 0.55 and last_info is not None:
            failed_route = last_info.get("top_route")
            runner_up_route = last_info.get("runner_up")
            orig_text = last_info.get("text", "")

            # Apply anti-Hebbian penalty on failed route and Hebbian boost on runner-up route
            if failed_route and runner_up_route and orig_text:
                self.intent_manifold.apply_prediction_error_penalty(
                    failed_intent=self._map_route_to_intent(failed_route),
                    true_intent=self._map_route_to_intent(runner_up_route),
                    text=orig_text
                )

        # 0. Cognitive Ignition: Global Cognitive Workspace computes unified conscious state
        gw_ignition = self.global_workspace.ignite_workspace(
            user_text=text,
            current_state=self.global_workspace.current_conscious_state
        )
        if hasattr(self, "self_model") and hasattr(self.self_model, "update_interoception"):
            self.self_model.update_interoception(epistemic_entropy=gw_ignition["entropy"])

        # Cognitive Intent Routing
        route_type, resolved_hyp = self.intent_router.route(text, hypothesis)
        res = self._execute_route(
            route_type=route_type,
            text=text,
            hypothesis=hypothesis,
            conv_history=conv_history,
            resolved_payload=resolved_hyp,
            record_history=True,
            gw_ignition=gw_ignition
        )
        res["workspace_ignition"] = {
            "winning_specialist": gw_ignition["winning_specialist"],
            "coalition_distribution": gw_ignition["coalition_distribution"],
            "entropy": gw_ignition["entropy"]
        }
        if hasattr(self, "self_model") and hasattr(self.self_model, "interoception"):
            res["interoception"] = self.self_model.interoception.get_summary()

        # Upgrade 4: Subjective Autobiographical Continuity Stream
        if hasattr(self, "autobiographical_engine"):
            new_c = getattr(self, "_last_newly_crystallized", [])
            strange_cnt = self.strange_loop.total_interventions_count if hasattr(self, "strange_loop") else 0
            self.autobiographical_engine.record_cognitive_turn(
                user_text=text,
                response_text=res.get("response_text", ""),
                route_type=str(res.get("route_type", "CONVERSATIONAL")),
                internal_telemetry=res.get("interoception"),
                new_concepts=new_c,
                strange_loop_count=strange_cnt
            )
            res["autobiographical_narrative"] = self.autobiographical_engine.get_autobiographical_narrative()

        return res

    def _map_route_to_intent(self, route: RouteType) -> str:
        """Maps RouteType enum to ContinuousIntentManifold centroid cluster name."""
        if route == RouteType.SYMBOLIC_PROVER:
            return "symbolic_math"
        elif route == RouteType.INTROSPECTIVE_SELF:
            return "self_identity"
        elif route == RouteType.ACTIVE_TEACHING:
            return "active_teaching"
        return "conversational"

    def _execute_route(
        self,
        route_type: RouteType,
        text: str,
        hypothesis: Optional[Hypothesis] = None,
        conv_history: Optional[List[Dict[str, str]]] = None,
        resolved_payload: Any = None,
        record_history: bool = True,
        gw_ignition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Executes a cognitive pathway (Self-Model, Active Learner, Symbolic Prover, or Conversational)."""
        # 1. Introspective Self-Model Query (High priority, grounded self-knowledge)
        if route_type == RouteType.INTROSPECTIVE_SELF:
            state_dict = self.self_model.answer_self_query(text)
            answer = self.mouth.verbalize_self_state(state_dict, user_prompt=text)
            if record_history:
                self.chat_history.append_turn(text, answer, metadata={"route_type": route_type.value, "state_dict": state_dict})
            return {
                "input_text": text,
                "response_text": answer,
                "verification": None,
                "route_type": route_type.value,
                "converged_at_step": 1,
                "final_energy": 0.0,
                "energy_trace": [0.0],
                "dots_3d": [],
                "raw_math": [],
                "chat_turn_count": self.chat_history.turn_count,
                "state_dict": state_dict,
                "interoception": state_dict.get("interoception"),
                "system_status": state_dict.get("system_status"),
                "os_grounding": state_dict.get("embodiment", {}).get("grounded_summary")
            }

        # 1.2 Action Directive Route: Direct real-world execution (file creation, sandboxed code execution)
        if route_type == RouteType.ACTION_DIRECTIVE:
            action_res = resolved_payload
            if action_res and action_res.is_action:
                status_str = "SUCCESS" if action_res.success else "FAILED"
                if action_res.action_type == "CREATE_FILE":
                    ans = f"I have created '{os.path.basename(action_res.target_path)}' at {action_res.target_path}. Output: '{action_res.stdout}'."
                else:
                    ans = f"Executed directive with status {status_str}. Output: '{action_res.stdout}'."
            else:
                ans = "Action directive could not be completed."

            if record_history:
                self.chat_history.append_turn(text, ans, metadata={"route_type": route_type.value, "action_res": str(action_res)})

            self.learn_from_conversation(text, ans)
            return {
                "input_text": text,
                "response_text": ans,
                "verification": None,
                "route_type": route_type.value,
                "converged_at_step": 1,
                "final_energy": 0.0,
                "energy_trace": [0.0],
                "dots_3d": [],
                "raw_math": [],
                "chat_turn_count": self.chat_history.turn_count,
                "action_telemetry": action_res
            }

        # 1.5 Active Teaching Route: Natively learn, verify, and wire new concept
        if route_type == RouteType.ACTIVE_TEACHING:
            teach_info = resolved_payload if resolved_payload is not None else (hypothesis if isinstance(hypothesis, dict) else None)
            if teach_info and "concept_name" in teach_info:
                learn_msg = self.active_learner.learn_new_concept(
                    concept_name=teach_info["concept_name"],
                    related_concepts=teach_info.get("related_concepts", []),
                    definition_ast=teach_info.get("definition_ast", "")
                )
                is_valid, proof_msg = self.active_learner.verifier.prove(teach_info.get("definition_ast", ""))
                if is_valid:
                    self.refresh_retriever()
                ver_res = VerificationResult(
                    is_valid=is_valid,
                    proof_log=proof_msg,
                    prover_name="Z3_SMT_Axiom_Prover"
                )
            else:
                learn_msg = "Could not parse new concept definition."
                ver_res = VerificationResult(is_valid=False, proof_log="Invalid definition", prover_name="Z3_SMT_Axiom_Prover")

            if record_history:
                self.chat_history.append_turn(text, learn_msg, metadata={"route_type": route_type.value})
            return {
                "input_text": text,
                "response_text": learn_msg,
                "verification": ver_res,
                "hypothesis": None,
                "route_type": route_type.value,
                "converged_at_step": 1,
                "final_energy": 0.0,
                "energy_trace": [0.0],
                "dots_3d": [],
                "raw_math": [],
                "chat_turn_count": self.chat_history.turn_count,
                "critic_violations": []
            }

        # 2. Semantic Tokenization & Encoding
        semantic_tokens = self.tokenizer.encode(text)
        token_ids = [tok.token_id for tok in semantic_tokens]
        extracted_nums = self.nalu_cluster.execute_semantic_token_math(semantic_tokens)

        token_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        num_tensor = torch.tensor([extracted_nums], dtype=torch.float32, device=self.device) if extracted_nums else None

        if self.use_pretrained_translator:
            latent_concept = self.ear(text)
        else:
            encoded_seq = self.encoder(token_tensor)
            latent_concept = self.encoder.pool_state(encoded_seq)

        # Conscious Workspace Priming: Fuse linguistic input with conscious state
        if gw_ignition is not None and "z_conscious" in gw_ignition:
            z_c = gw_ignition["z_conscious"].to(latent_concept.device)
            latent_concept = F.normalize(0.6 * latent_concept + 0.4 * z_c, p=2, dim=-1)

        # 3. System 2 Latent Reasoning (Settles until math reaches equilibrium)
        thought = self.forward_reasoning(latent_concept, num_tensor)
        settled_latent = thought["settled_latent"]

        # 4. Formal Type-Dispatched Symbolic Verification (ONLY for SYMBOLIC_PROVER tasks!)
        verification: Optional[VerificationResult] = None
        hyp_to_verify = resolved_payload if resolved_payload is not None else hypothesis
        if route_type == RouteType.SYMBOLIC_PROVER and hyp_to_verify is not None:
            verification = self.verifier.verify(hyp_to_verify)

        # 5. Verbalize settled latent memory back into English
        active_concepts = [tok.text for tok in semantic_tokens if tok.token_type in ("WORD", "SYMBOL")]
        proof_str = verification.proof_log if verification is not None else None
        is_valid = verification.is_valid if verification is not None else True

        if self.use_pretrained_translator:
            response_text = self.mouth.verbalize(
                user_prompt=text,
                reasoned_concepts=active_concepts,
                proof_log=proof_str,
                is_valid=is_valid,
                history=conv_history,
                route_type=route_type.value
            )
        else:
            decoder_memory = torch.cat([settled_latent.unsqueeze(1), thought["graph_nodes"]], dim=1)
            generated_tokens = self.decoder.generate(
                memory=decoder_memory,
                bos_id=self.tokenizer.w2i[self.tokenizer.bos_token],
                eos_id=self.tokenizer.w2i[self.tokenizer.eos_token],
                max_steps=32
            )
            response_text = self.tokenizer.decode(generated_tokens[0].tolist())
            if "<unk>" in response_text or not response_text.strip():
                response_text = self.mouth.verbalize(
                    user_prompt=text,
                    reasoned_concepts=active_concepts,
                    proof_log=proof_str,
                    is_valid=is_valid,
                    history=conv_history,
                    route_type=route_type.value
                )

        if record_history:
            self.chat_history.append_turn(text, response_text, metadata={"route_type": route_type.value})

        self.learn_from_conversation(text, response_text)

        # Attach any cognitive tools executed during generation
        tools_executed = getattr(self.mouth, "last_tool_results", [])

        return {
            "input_text": text,
            "response_text": response_text,
            "verification": verification,
            "hypothesis": hyp_to_verify,
            "route_type": route_type.value,
            "converged_at_step": thought["converged_step"],
            "final_energy": thought["final_energy"].item(),
            "energy_trace": thought["energy_trace"],
            "dots_3d": thought["dots_3d"].detach().cpu().numpy().tolist(),
            "raw_math": thought["raw_math"].detach().cpu().numpy().tolist(),
            "chat_turn_count": self.chat_history.turn_count,
            "critic_verdict": self.mouth.last_critic_verdict,
            "critic_violations": self.mouth.last_critic_verdict.violations if self.mouth.last_critic_verdict else [],
            "tools_executed": tools_executed,
            "action_telemetry": tools_executed[0] if tools_executed else None
        }

    def learn_from_conversation(self, user_text: str, response_text: str) -> Dict[str, Any]:
        """
        Continuous Online Learning at Runtime (Hebbian Plasticity & Concept Crystallization):
        1. Identifies co-active concepts in the 256D manifold and strengthens their synaptic links (Hebbian update).
        2. Detects novel salient concepts introduced in conversation, crystallizes them as new manifold nodes,
           and permanently wires them into knowledge without restarting.
        """
        combined = f"{user_text} {response_text}".lower()
        active_concepts: List[str] = []

        # Find firing concepts in existing manifold
        all_concepts = list(self.hebbian_manifold.concept_names)
        for c in all_concepts:
            c_clean = c.lower().replace("_", " ")
            if c_clean in combined:
                active_concepts.append(c)

        # Apply Hebbian synaptic reinforcement on co-active concepts
        if len(active_concepts) >= 2:
            for i in range(len(active_concepts)):
                for j in range(i + 1, min(len(active_concepts), i + 4)):
                    self.hebbian_manifold.reinforce_connection(
                        active_concepts[i],
                        active_concepts[j],
                        weight_boost=0.08
                    )

        # Check for novel salient technical concepts to crystallize into manifold
        stopwords = {
            "where", "which", "could", "would", "about", "there", "their", "these", "those",
            "hello", "online", "write", "create", "created", "creating", "creator", "greater",
            "execute", "program", "script", "project", "document", "screen", "window", "folder",
            "downloads", "download", "please", "thank", "thanks", "think", "thing", "things",
            "really", "already", "should", "start", "starts", "starting", "research", "researching",
            "going", "doing", "making", "nothing", "something", "everything", "learned", "recently",
            "comes", "after", "right", "alright", "simple", "asked", "hijacking", "codebase", "files"
        }
        from src.reasoning_engine.workshop_comprehension import PhoneticSyntacticReconstructor
        words = [w.strip(".,!?:;\"'()[]{}") for w in user_text.split() if len(w.strip(".,!?:;\"'()[]{}")) >= 5]
        newly_crystallized = []

        for w in words:
            w_lower = w.lower()
            if (
                w_lower in stopwords
                or w_lower in PhoneticSyntacticReconstructor.TOKEN_PHONETIC_MAP
                or not w_lower.replace("_", "").isalnum()
            ):
                continue
            if w_lower not in self.hebbian_manifold.name_to_idx:
                w_vec = self.intent_manifold.encode(w_lower)
                idx = self.hebbian_manifold.add_node(w_lower, w_vec)
                newly_crystallized.append(w_lower)
                for ac in active_concepts[:2]:
                    self.hebbian_manifold.reinforce_connection(w_lower, ac, weight_boost=0.15)
                logger.info("Conversational learning crystallized new manifold concept: '%s' (Total: %d)", w_lower, self.hebbian_manifold.num_concepts)

        if newly_crystallized:
            self.refresh_retriever()
            if hasattr(self.self_model, "load_model"):
                self.self_model.load_model()

        self._last_newly_crystallized = list(newly_crystallized)

        # Pragmatic idiom learning from conversational explanation
        if hasattr(self.mouth, "pragmatic_grounder") and self.mouth.pragmatic_grounder:
            recent_turns = self.chat_history.get_recent_history(max_turns=6) if hasattr(self, "chat_history") and self.chat_history else []
            self.mouth.pragmatic_grounder.parse_pragmatic_clarification(user_text, history=recent_turns)

        return {
            "active_concepts": active_concepts,
            "newly_crystallized": newly_crystallized,
            "total_concepts": self.hebbian_manifold.num_concepts
        }

    def mental_time_travel(self, query: str) -> Dict[str, Any]:
        """
        Chronesthesia (Mental Time Travel):
        Introspects across past autobiographical episodes, compares past subjective states
        with the present, and articulates Jane's cognitive trajectory over time.
        """
        if hasattr(self, "autobiographical_engine") and self.autobiographical_engine is not None:
            return self.autobiographical_engine.mental_time_travel(query)
        return {
            "found": False,
            "narrative": "Autobiographical narrative engine is not available.",
            "delta": {}
        }

    def call_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        safety_contract: str = "energy_out <= energy_in",
        timeout: float = 10.0
    ) -> GatewayResult:
        """Executes a cognitive tool through formal SMT-gated APIGateway."""
        return self.api_gateway.call_tool(tool_name, args, safety_contract=safety_contract, timeout=timeout)

    def observe_os(
        self,
        query_type: str = "all",
        safety_contract: str = "energy_out <= energy_in"
    ) -> GatewayResult:
        """Performs formal SMT-gated OS and host environment observation."""
        return self.api_gateway.observe_os(query_type=query_type, safety_contract=safety_contract)

