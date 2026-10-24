"""Unit and Integration Tests for Dual-Stream Cognitive Ingestion Engine.

Verifies:
1. Stream A: Zero-regex HTML section parsing and LaTeX math extraction.
2. Stream B: Visual diagram perception and VisionPatchTransformer projection.
3. Cross-modal manifold alignment and Hebbian plasticity.
4. Integration with WikipediaIngestor and JanePipeline.
"""

import pytest
import torch

from src.perception.dual_stream_ingestor import DualStreamCognitiveIngestor, WikipediaHTMLStructuralParser, DualStreamIngestionResult, MathematicalBlock, VisualDiagramNode
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.reasoning_engine.causal_hierarchy import HierarchicalCausalKnowledgeGraph
from src.reasoning_engine.episodic_ledger import EpisodicLedger
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.perception.vision_encoder import VisionPatchTransformer
from src.core.wikipedia_ingestor import WikipediaIngestor


class TestStructuralAndMathParser:
    """Tests for Stream A zero-regex HTML section tree and LaTeX extraction."""

    def test_html_section_and_math_extraction(self):
        sample_html = """
        <html>
        <body>
            <h1>Quantum Information</h1>
            <p>Introductory paragraph describing quantum coherence and superposition.</p>
            <h2>Mathematical Formalism</h2>
            <p>The state of the system is given by the density matrix:</p>
            <math>\\rho = \\sum p_i |\\psi_i\\rangle\\langle\\psi_i|</math>
            <p>The Von Neumann entropy satisfies:</p>
            <img class="mwe-math-fallback-image-inline" alt="S = -\\text{Tr}(\\rho \\log \\rho)" />
            <h3>Pure State Criterion</h3>
            <p>For a pure state the purity equals one:</p>
            <math>\\text{Tr}(\\rho^2) = 1</math>
        </body>
        </html>
        """
        parser = WikipediaHTMLStructuralParser()
        parser.feed(sample_html)
        sections = parser.finish()

        assert len(sections) >= 2
        # Verify headings
        titles = [s.title for s in sections]
        assert any("Mathematical Formalism" in t for t in titles)

        # Verify math blocks
        all_eqs = [eq for s in sections for eq in s.equations]
        assert len(all_eqs) >= 2
        for eq in all_eqs:
            assert isinstance(eq, MathematicalBlock)
            assert len(eq.canonical_expression) > 0


class TestVisualDiagramPerception:
    """Tests for Stream B visual perception and VisionPatchTransformer encoding."""

    def test_diagram_synthesis_and_vit_projection(self):
        manifold = HebbianConceptManifold(latent_dim=256)
        verifier = Z3LogicVerifier()
        vision_encoder = VisionPatchTransformer()

        engine = DualStreamCognitiveIngestor(
            manifold=manifold,
            z3_verifier=verifier,
            vision_encoder=vision_encoder
        )

        for diag_type in ("bloch_sphere", "field_flux", "phase_portrait", "attention_map"):
            node = engine._create_visual_node(
                artifact_id=f"test_{diag_type}",
                caption=f"Test diagram for {diag_type}",
                diagram_type=diag_type
            )
            assert isinstance(node, VisualDiagramNode)
            assert node.frame_tensor.shape == (1, 3, 128, 128)
            assert node.visual_latent.shape == (256,)
            assert node.mean_luminance > 0.0
            assert node.contrast_metric > 0.0
            # Test norm of embedding
            assert pytest.approx(float(torch.norm(node.visual_latent).item()), rel=1e-3) == 1.0


class TestCrossModalDualStreamIngestion:
    """Tests for end-to-end Dual-Stream Ingestion and Cross-Modal Alignment."""

    def test_ingest_quantum_entanglement_dual_stream(self, tmp_path):
        ledger = EpisodicLedger(log_path=str(tmp_path / "dual_ledger.jsonl"))
        manifold = HebbianConceptManifold(latent_dim=256)
        verifier = Z3LogicVerifier()
        causal_graph = HierarchicalCausalKnowledgeGraph()
        vision_encoder = VisionPatchTransformer()

        ingestor = DualStreamCognitiveIngestor(
            manifold=manifold,
            z3_verifier=verifier,
            causal_graph=causal_graph,
            vision_encoder=vision_encoder,
            episodic_ledger=ledger
        )

        result = ingestor.ingest_dual_stream("Quantum_entanglement")
        assert isinstance(result, DualStreamIngestionResult)
        assert result.article_title == "Quantum_entanglement"
        assert result.stream_a_sections_parsed >= 3
        assert result.stream_a_equations_proved >= 3
        assert result.stream_b_diagrams_perceived >= 1
        assert len(result.cross_modal_aligned_concepts) >= 1

        # Check manifold nodes
        root = "quantum_entanglement"
        assert root in manifold.concept_names
        # Check diagram node
        assert any("bloch_sphere" in c for c in manifold.concept_names)

        # Check causal graph nodes
        assert causal_graph.get_node(root) is not None
        assert causal_graph.get_node(root).tier == 3

        # Check episodic ledger
        events = ledger.get_recent_events(limit=5)
        assert any(e.action_type == "DUAL_STREAM_INGESTION" for e in events)

    def test_ingest_runge_kutta_dual_stream(self, tmp_path):
        ledger = EpisodicLedger(log_path=str(tmp_path / "rk4_ledger.jsonl"))
        manifold = HebbianConceptManifold(latent_dim=256)
        verifier = Z3LogicVerifier()
        causal_graph = HierarchicalCausalKnowledgeGraph()

        ingestor = DualStreamCognitiveIngestor(
            manifold=manifold,
            z3_verifier=verifier,
            causal_graph=causal_graph,
            episodic_ledger=ledger
        )

        res = ingestor.ingest_dual_stream("Runge_Kutta_methods")
        assert res.stream_a_sections_parsed >= 2
        assert res.stream_b_diagrams_perceived >= 1
        assert "runge_kutta_methods" in manifold.concept_names


class TestWikipediaIngestorIntegration:
    """Tests verifying backward compatibility and upgraded dual-stream capability in WikipediaIngestor."""

    def test_wikipedia_ingestor_dual_stream_methods(self, tmp_path):
        ledger = EpisodicLedger(log_path=str(tmp_path / "wiki_compat.jsonl"))
        manifold = HebbianConceptManifold(latent_dim=256)
        verifier = Z3LogicVerifier()

        wiki_ingestor = WikipediaIngestor(
            active_learner=None,
            manifold=manifold,
            z3_verifier=verifier,
            episodic_ledger=ledger
        )

        # 1. Dual-stream direct call
        dual_res = wiki_ingestor.ingest_dual_stream("Attention_mechanism")
        assert dual_res.stream_a_sections_parsed >= 2
        assert dual_res.stream_b_diagrams_perceived >= 1

        # 2. Standard ingest_article call (backward-compatible)
        report = wiki_ingestor.ingest_article("Superconductivity")
        assert report.articles_processed == 1
        assert len(report.accepted_concepts) > 0
        assert report.query == "Superconductivity"
