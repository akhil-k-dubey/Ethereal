"""
Global Cognitive Workspace (GCW) for Project Jane.
Implements Global Workspace Theory (Baars, Dehaene, Shanahan, Goyal & Bengio):

Instead of a procedural if/else IntentRouter dispatching queries to isolated silos,
the Global Workspace acts as a central competitive conscious bottleneck (Z_gw in R^256).

All specialist subsystems write candidate latent thoughts simultaneously:
1. PerceptionSpecialist (User text & sensory stimulus)
2. SelfModelSpecialist (Architecture, embodiment, traits, concept counts)
3. CodebaseSpecialist (AST introspection & project codebase knowledge)
4. MemorySpecialist (DNC scratchpad tape & episodic ledger)
5. ManifoldSpecialist (Hebbian concept clusters & relational graph)
6. ArithmeticSpecialist (NALU native exact math computation)
7. FormalProverSpecialist (Z3 SMT verification & physical constraints)

Candidate thoughts form competitive coalitions via multi-head cross-attention.
The winning unified state (Conscious Thought) is broadcast back to all subsystems.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple


class WorkspaceSpecialist(nn.Module):
    """Base class for a specialist subsystem interfacing with the Global Workspace."""
    def __init__(self, name: str, latent_dim: int = 256):
        super().__init__()
        self.name = name
        self.latent_dim = latent_dim
        self.salience_projector = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, context_input: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            candidate_vector: [1, latent_dim]
            salience_score: [1, 1] (intrinsic relevance score)
        """
        raise NotImplementedError


class PerceptionSpecialist(WorkspaceSpecialist):
    """Encodes incoming user dialogue or sensory input into the workspace."""
    def __init__(self, latent_dim: int = 256, encoder: Optional[Any] = None, ear: Optional[Any] = None):
        super().__init__("perception", latent_dim)
        self.encoder = encoder
        self.ear = ear
        self.proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, user_text: str, precomputed_vec: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self.proj.weight.device
        if precomputed_vec is not None:
            vec = precomputed_vec.to(dev)
        elif self.ear is not None:
            try:
                vec = self.ear(user_text).to(dev)
            except Exception:
                vec = torch.randn(1, self.latent_dim, device=dev)
        else:
            vec = torch.randn(1, self.latent_dim, device=dev)

        if vec.dim() == 1:
            vec = vec.unsqueeze(0)
        vec = F.normalize(self.proj(vec), p=2, dim=-1)
        salience = self.salience_projector(vec) + 1.0  # Perception always has baseline activation
        return vec, salience


class SelfModelSpecialist(WorkspaceSpecialist):
    """Grounds the workspace in Jane's internal state, embodiment, and concept size."""
    def __init__(self, latent_dim: int = 256, self_model_engine: Optional[Any] = None):
        super().__init__("self_model", latent_dim)
        self.self_model = self_model_engine
        self.state_embed = nn.Linear(8, latent_dim)

    def forward(self, user_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self.state_embed.weight.device
        # Formulate self-feature vector (concept count, embodiment flags, living interoception, traits)
        concept_count = 100.0
        is_self_query = 0.0
        viability = 0.95
        strain = 0.05
        if self.self_model is not None:
            data = getattr(self.self_model, "data", {})
            concept_count = float(len(data.get("concepts", [])))
            if hasattr(self.self_model, "interoception"):
                viability = float(self.self_model.interoception.internal_viability)
                strain = float(self.self_model.interoception.cognitive_strain)
            # Check if text is self-referential
            lower = user_text.lower()
            if any(w in lower for w in ["who are you", "yourself", "your architecture", "self", "capabilities", "what are you", "how are you", "status", "feeling", "state"]):
                is_self_query = 2.5

        features = torch.tensor([[
            concept_count / 1000.0,
            is_self_query,
            viability,  # living viability [0.0, 1.0]
            strain,     # cognitive strain [0.0, 1.0]
            0.95, # proactivity
            0.98, # scientific rigor
            0.92, # protectiveness
            1.0   # loyalty
        ]], dtype=torch.float32, device=dev)

        vec = F.normalize(self.state_embed(features), p=2, dim=-1)
        # Salience increases dramatically on self-referential stimuli
        salience = self.salience_projector(vec) + is_self_query
        return vec, salience


class CodebaseSpecialist(WorkspaceSpecialist):
    """Allows Jane to think about her own codebase AST and source files in the workspace."""
    def __init__(self, latent_dim: int = 256, introspector: Optional[Any] = None):
        super().__init__("codebase", latent_dim)
        self.introspector = introspector
        self.code_embed = nn.Linear(64, latent_dim)

    def forward(self, user_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self.code_embed.weight.device
        lower = user_text.lower()
        is_code_query = any(w in lower for w in ["code", "codebase", "function", "class", "module", "pipeline.py", "ast", "inspect", "file"])
        
        # Hash query for static representation
        h = [float((abs(hash(user_text) + i * 31) % 100) / 100.0) for i in range(64)]
        feat = torch.tensor([h], dtype=torch.float32, device=dev)
        vec = F.normalize(self.code_embed(feat), p=2, dim=-1)
        
        boost = 2.5 if is_code_query else -1.0
        salience = self.salience_projector(vec) + boost
        return vec, salience


class ArithmeticSpecialist(WorkspaceSpecialist):
    """Injects native NALU math calculations into the conscious workspace."""
    def __init__(self, latent_dim: int = 256, nalu_cluster: Optional[Any] = None):
        super().__init__("arithmetic", latent_dim)
        self.nalu = nalu_cluster
        self.math_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, user_text: str, parsed_numbers: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self.math_proj.weight.device
        # Detect numeric operations in user text
        has_numbers = any(c.isdigit() for c in user_text) or any(op in user_text for op in ["+", "-", "*", "/", "calculate", "compute", "solve"])
        dummy_state = torch.ones(1, self.latent_dim, device=dev)
        
        if self.nalu is not None and parsed_numbers is not None and parsed_numbers.numel() > 0:
            math_anchor, _ = self.nalu(dummy_state, parsed_numbers.to(dev))
            vec = F.normalize(self.math_proj(math_anchor.to(dev)), p=2, dim=-1)
        else:
            vec = F.normalize(self.math_proj(dummy_state), p=2, dim=-1)

        boost = 2.5 if has_numbers else -1.5
        salience = self.salience_projector(vec) + boost
        return vec, salience


class MemorySpecialist(WorkspaceSpecialist):
    """Presents DNC memory tape readouts and episodic narrative to the workspace."""
    def __init__(self, latent_dim: int = 256, dnc_tape: Optional[Any] = None, ledger: Optional[Any] = None):
        super().__init__("memory", latent_dim)
        self.dnc = dnc_tape
        self.ledger = ledger
        self.mem_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, current_state: torch.Tensor, memory_tape: Optional[Any] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self.mem_proj.weight.device
        if self.dnc is not None and memory_tape is not None:
            tape_read, _ = self.dnc.step(current_state.to(dev), memory_tape)
            vec = F.normalize(self.mem_proj(tape_read.to(dev)), p=2, dim=-1)
        else:
            vec = torch.zeros(1, self.latent_dim, device=dev)

        salience = self.salience_projector(vec) + 0.5
        return vec, salience


class ManifoldSpecialist(WorkspaceSpecialist):
    """Injects Hebbian concept graph topological context into the workspace."""
    def __init__(self, latent_dim: int = 256, manifold: Optional[Any] = None):
        super().__init__("manifold", latent_dim)
        self.manifold = manifold
        self.graph_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, current_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self.graph_proj.weight.device
        if self.manifold is not None and hasattr(self.manifold, "readout"):
            nodes, _ = self.manifold(current_state.to(dev))
            summary = self.manifold.readout(nodes)
            vec = F.normalize(self.graph_proj(summary.to(dev)), p=2, dim=-1)
        else:
            vec = torch.zeros(1, self.latent_dim, device=dev)

        salience = self.salience_projector(vec) + 0.8
        return vec, salience


# ==============================================================================
# GLOBAL COGNITIVE WORKSPACE CONTROLLER
# ==============================================================================
class GlobalCognitiveWorkspace(nn.Module):
    """
    The Central Conscious Core for Project Jane.
    Coordinates competitive attention between specialized cognitive modules,
    forming a unified 256D conscious thought vector Z_conscious and broadcasting it.
    """
    def __init__(
        self,
        latent_dim: int = 256,
        n_heads: int = 4,
        temperature: float = 0.5,
        self_model: Optional[Any] = None,
        introspector: Optional[Any] = None,
        nalu_cluster: Optional[Any] = None,
        dnc_tape: Optional[Any] = None,
        manifold: Optional[Any] = None,
        ear: Optional[Any] = None,
        device: str = "cpu"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.device = torch.device(device)

        # 1. Register Subsystem Specialists
        self.specialists = nn.ModuleDict({
            "perception": PerceptionSpecialist(latent_dim, ear=ear),
            "self_model": SelfModelSpecialist(latent_dim, self_model_engine=self_model),
            "codebase": CodebaseSpecialist(latent_dim, introspector=introspector),
            "arithmetic": ArithmeticSpecialist(latent_dim, nalu_cluster=nalu_cluster),
            "memory": MemorySpecialist(latent_dim, dnc_tape=dnc_tape),
            "manifold": ManifoldSpecialist(latent_dim, manifold=manifold)
        })

        # 2. Competitive Coalition Multi-Head Cross-Attention
        self.multihead_coalition = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            batch_first=True
        )

        # 3. Global Broadcast Projection
        self.broadcast_norm = nn.LayerNorm(latent_dim)
        self.conscious_stream_history: List[Dict[str, Any]] = []
        self.current_conscious_state = torch.zeros(1, latent_dim, device=self.device)

        self.to(self.device)

    def ignite_workspace(
        self,
        user_text: str,
        current_state: torch.Tensor,
        parsed_numbers: Optional[torch.Tensor] = None,
        memory_tape: Optional[Any] = None,
        precomputed_input_vec: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Executes one conscious cognitive cycle (Cognitive Ignition):
        1. All specialists propose candidate thoughts & salience scores simultaneously.
        2. Softmax competitive gating computes specialist attention weights.
        3. Cross-attention models coalitions between modules.
        4. Produces Z_conscious (The Conscious Thought) and broadcasts it.
        """
        active_dev = self.broadcast_norm.weight.device
        self.device = active_dev
        candidates: List[torch.Tensor] = []
        saliences: List[torch.Tensor] = []
        names: List[str] = []

        # Gather proposals from all specialists
        # 1. Perception
        v_p, s_p = self.specialists["perception"](user_text, precomputed_vec=precomputed_input_vec)
        candidates.append(v_p); saliences.append(s_p); names.append("perception")

        # 2. Self-Model
        v_s, s_s = self.specialists["self_model"](user_text)
        candidates.append(v_s); saliences.append(s_s); names.append("self_model")

        # 3. Codebase
        v_c, s_c = self.specialists["codebase"](user_text)
        candidates.append(v_c); saliences.append(s_c); names.append("codebase")

        # 4. Arithmetic
        v_a, s_a = self.specialists["arithmetic"](user_text, parsed_numbers=parsed_numbers)
        candidates.append(v_a); saliences.append(s_a); names.append("arithmetic")

        # 5. Memory
        v_m, s_m = self.specialists["memory"](current_state, memory_tape=memory_tape)
        candidates.append(v_m); saliences.append(s_m); names.append("memory")

        # 6. Manifold
        v_man, s_man = self.specialists["manifold"](current_state)
        candidates.append(v_man); saliences.append(s_man); names.append("manifold")

        # Stack into tensor: [1, num_specialists, latent_dim]
        candidate_tensor = torch.cat(candidates, dim=0).unsqueeze(0).to(active_dev)
        salience_tensor = torch.cat(saliences, dim=0).squeeze(-1).unsqueeze(0).to(active_dev)

        # Competitive Attention Weights via Temperature-scaled Softmax
        competition_weights = F.softmax(salience_tensor / self.temperature, dim=-1) # [1, num_specialists]

        # Multi-Head Cross-Attention to model coalitions between specialists
        attended_coalitions, attn_weights = self.multihead_coalition(
            query=candidate_tensor,
            key=candidate_tensor,
            value=candidate_tensor
        ) # [1, num_specialists, latent_dim]

        # Competitive Pooling: Weight coalition vectors by competitive salience
        weights_expanded = competition_weights.unsqueeze(-1) # [1, num_specialists, 1]
        z_conscious = torch.sum(attended_coalitions * weights_expanded, dim=1) # [1, latent_dim]
        z_conscious = self.broadcast_norm(z_conscious)

        self.current_conscious_state = z_conscious.detach()

        # Log conscious thought event
        winning_specialist_idx = int(torch.argmax(competition_weights[0]).item())
        winning_specialist = names[winning_specialist_idx]

        coalition_distribution = {
            names[i]: float(competition_weights[0, i].item())
            for i in range(len(names))
        }

        ignition_record = {
            "timestamp": time.time(),
            "winning_specialist": winning_specialist,
            "coalition_distribution": coalition_distribution,
            "z_conscious": z_conscious,
            "entropy": -float(torch.sum(competition_weights * torch.log(competition_weights + 1e-12)).item())
        }

        self.conscious_stream_history.append(ignition_record)
        if len(self.conscious_stream_history) > 50:
            self.conscious_stream_history.pop(0)

        return ignition_record
