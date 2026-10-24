"""Unit and Integration Tests for Hierarchical Causal Knowledge Scaling,
Intrinsic Goal-Driven Agency, and Super Autonomous Code Synthesizer.
"""

import os
import shutil
import tempfile

from src.reasoning_engine.causal_hierarchy import HierarchicalCausalKnowledgeGraph, CausalInterventionResult
from src.core.intrinsic_agency import IntrinsicAgencyEngine, AutonomousGoal, GoalStatus, DriveType, EpistemicDrive, CompetenceDrive
from src.symbolic_verifier.autonomous_coder import AutonomousCodeSynthesizer, SystemBlueprint
from src.core.api_gateway import APIGateway
from src.core.tools import ToolRegistry
from src.reasoning_engine.self_model import SelfModelEngine
from src.symbolic_verifier.verifier_orchestrator import SymbolicVerifierOrchestrator


class TestHierarchicalCausalKnowledgeGraph:
    """Tests for Judea Pearl's 4-tier causal abstraction hierarchy and do-calculus."""

    def test_graph_initialization_and_tiers(self):
        graph = HierarchicalCausalKnowledgeGraph(latent_dim=256)
        summary = graph.get_graph_summary()
        assert summary["total_nodes"] >= 8
        assert 0 in graph.tier_index
        assert 1 in graph.tier_index
        assert 2 in graph.tier_index
        assert 3 in graph.tier_index
        assert len(graph.tier_index[3]) >= 1  # Meta-Architectures
        assert len(graph.tier_index[2]) >= 1  # Governing Laws
        assert len(graph.tier_index[1]) >= 1  # Mechanisms
        assert len(graph.tier_index[0]) >= 1  # Observables

    def test_top_down_coarse_to_fine_search(self):
        graph = HierarchicalCausalKnowledgeGraph(latent_dim=256)
        # Query for stability / lyapunov
        results = graph.hierarchical_top_down_search("lyapunov stability", top_k=3)
        assert len(results) > 0
        node_ids = [n.node_id for n in results]
        assert any("lyapunov" in nid for nid in node_ids)

        # Query for neural ODE / liquid network
        results_ode = graph.hierarchical_top_down_search("liquid neural network", top_k=3)
        assert len(results_ode) > 0
        assert any("liquid" in n.node_id or "rk4" in n.node_id for n in results_ode)

    def test_pearl_do_calculus_intervention(self):
        graph = HierarchicalCausalKnowledgeGraph(latent_dim=256)
        # Intervene on lyapunov_stability
        res = graph.simulate_do_intervention("lyapunov_stability", intervention_val={"V_dot": -0.5})
        assert isinstance(res, CausalInterventionResult)
        assert res.target_node == "lyapunov_stability"
        assert res.is_causally_valid is True
        assert len(res.affected_nodes) > 0
        # Should propagate to attractor_restoration_mechanism and homeostatic_lyapunov_energy
        assert any("attractor" in aff or "energy" in aff for aff in res.affected_nodes)
        assert len(res.predicted_state_deltas) > 0

    def test_causal_intervention_nonexistent_node(self):
        graph = HierarchicalCausalKnowledgeGraph(latent_dim=256)
        res = graph.simulate_do_intervention("non_existent_node_xyz", intervention_val=None)
        assert res.is_causally_valid is False
        assert len(res.violated_invariants) > 0

    def test_singleton_accessor(self):
        g1 = HierarchicalCausalKnowledgeGraph.get_instance()
        g2 = HierarchicalCausalKnowledgeGraph.get_instance()
        assert g1 is g2


class TestIntrinsicGoalDrivenAgency:
    """Tests for active inference / free energy minimization and autonomous goal formation."""

    def test_drive_instantiation_and_free_energy(self):
        causal_graph = HierarchicalCausalKnowledgeGraph.get_instance()
        self_model = SelfModelEngine()
        agency = IntrinsicAgencyEngine(
            causal_graph=causal_graph,
            self_model=self_model
        )
        fe = agency.compute_free_energy()
        assert isinstance(fe, float)
        assert fe >= 0.0

    def test_epistemic_goal_formulation(self):
        causal_graph = HierarchicalCausalKnowledgeGraph.get_instance()
        self_model = SelfModelEngine()
        agency = IntrinsicAgencyEngine(
            causal_graph=causal_graph,
            self_model=self_model
        )
        # High entropy should stimulate epistemic drive
        self_model.interoception.epistemic_entropy = 0.85
        goal = agency.formulate_goal_from_drive("entropy_spike", drive=EpistemicDrive())
        assert isinstance(goal, AutonomousGoal)
        assert goal.drive_type == DriveType.EPISTEMIC
        assert len(goal.milestones) >= 2
        assert goal.status == GoalStatus.ACTIVE
        assert goal.priority > 0.0

    def test_competence_goal_formulation_and_step(self):
        causal_graph = HierarchicalCausalKnowledgeGraph.get_instance()
        self_model = SelfModelEngine()
        agency = IntrinsicAgencyEngine(
            causal_graph=causal_graph,
            self_model=self_model
        )
        goal = agency.formulate_goal_from_drive("develop_attention_module", drive=CompetenceDrive())
        assert goal.drive_type == DriveType.COMPETENCE
        agency.submit_autonomous_goal(goal)

        # Step agency loop
        step_res = agency.step_autonomous_loop()
        assert step_res["executed"] is True
        assert step_res["goal_id"] == goal.goal_id
        assert step_res["progress"] > 0.0

    def test_agency_summary_metrics(self):
        agency = IntrinsicAgencyEngine()
        summary = agency.get_agency_summary()
        assert "free_energy" in summary
        assert "drive_pressures" in summary
        assert "goal_queue_length" in summary
        assert "active_goal" in summary


class TestAutonomousCodeSynthesizer:
    """Tests for closed-loop micro-step AST synthesis and shadow worktree verification."""

    def test_blueprinting_multi_file_ai_subsystem(self):
        tmp_dir = tempfile.mkdtemp(prefix="jane_test_worktree_")
        try:
            coder = AutonomousCodeSynthesizer(worktree_dir=tmp_dir)
            blueprint = coder.architect_system_blueprint("Implement Rotary Position Embedding and Scaled Attention")
            assert isinstance(blueprint, SystemBlueprint)
            assert len(blueprint.modules) >= 2
            # Module names
            names = [m.module_name for m in blueprint.modules]
            assert "rope_embedding" in names
            assert "multihead_attention" in names

            # Check micro-step function contracts
            for m in blueprint.modules:
                assert len(m.functions) > 0
                for fn in m.functions:
                    assert fn.name != ""
                    assert len(fn.formal_contracts) > 0
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_synthesis_and_verification_convergence(self):
        tmp_dir = tempfile.mkdtemp(prefix="jane_test_worktree_")
        try:
            coder = AutonomousCodeSynthesizer(worktree_dir=tmp_dir)
            blueprint = coder.build_and_verify_system("Build a numerical RK4 integrator and dynamical system")
            assert blueprint.converged is True
            assert blueprint.total_loc > 20
            # All modules must be formally verified (Lyapunov test energy = 0.0)
            for m in blueprint.modules:
                assert m.is_verified is True
                assert m.verification_energy == 0.0
                full_path = os.path.join(tmp_dir, blueprint.system_name, m.rel_path)
                assert os.path.exists(full_path)
                with open(full_path, "r", encoding="utf-8") as f:
                    code_content = f.read()
                    assert len(code_content) > 50
                    # Zero regex verification
                    assert "def " in code_content
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestAPIGatewayAndToolRegistryIntegration:
    """Tests for APIGateway action dispatch and ToolRegistry affordances."""

    def test_api_gateway_synthesize_action(self):
        verifier = SymbolicVerifierOrchestrator()
        gateway = APIGateway(z3_verifier=verifier.z3_verifier)
        res = gateway.synthesize_software_system(goal="Create a continuous attractor network")
        assert res.success is True
        assert res.action_type == "SYNTHESIZE_SYSTEM"
        assert res.output["converged"] is True
        assert len(res.output["modules"]) >= 2

    def test_tool_registry_synthesize_tool(self):
        registry = ToolRegistry.get_instance()
        tool_res = registry.execute_tool(
            "synthesize_software_system",
            {"goal": "Build an attention subsystem"}
        )
        assert tool_res.success is True
        assert "System:" in tool_res.output
        assert "Converged: True" in tool_res.output

    def test_tool_registry_query_causal_hierarchy(self):
        registry = ToolRegistry.get_instance()
        tool_res = registry.execute_tool(
            "query_causal_hierarchy",
            {"query": "lyapunov stability laws"}
        )
        assert tool_res.success is True
        assert "Tier" in tool_res.output
        assert len(tool_res.artifacts["matched_nodes"]) > 0


class TestPipelineAndSelfModelIntegration:
    """Tests for full pipeline and self-model integration."""

    def test_self_model_includes_causal_hierarchy(self):
        graph = HierarchicalCausalKnowledgeGraph.get_instance()
        self_model = SelfModelEngine(causal_graph=graph)
        manifest = self_model.get_live_state_manifest()
        assert "causal_hierarchy" in manifest
        causal_data = manifest["causal_hierarchy"]
        assert causal_data["total_nodes"] >= 8
        assert "tier_distribution" in causal_data
