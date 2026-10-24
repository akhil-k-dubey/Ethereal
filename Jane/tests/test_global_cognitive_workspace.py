"""
Unit and Integration Tests for Upgrade 1: Global Cognitive Workspace (GCW) in Project Jane.
Verifies:
1. Multi-specialist candidate thought proposal and salience calculation.
2. Competitive coalition formation across distinct cognitive domains (Self, Code, Math, Language).
3. Mixed-domain co-equal coalitions (e.g. Codebase + Arithmetic).
4. End-to-end integration into JanePipeline with conscious ignition tracking.
"""

import torch
from src.reasoning_engine.global_workspace import (
    GlobalCognitiveWorkspace,
    PerceptionSpecialist,
    SelfModelSpecialist,
    CodebaseSpecialist,
    ArithmeticSpecialist,
    MemorySpecialist,
    ManifoldSpecialist
)
from src.core.pipeline import JanePipeline


def test_specialists_standalone_proposals():
    """Verifies that all 6 specialists produce valid normalized vectors and salience scores."""
    latent_dim = 256
    
    p = PerceptionSpecialist(latent_dim)
    s = SelfModelSpecialist(latent_dim)
    c = CodebaseSpecialist(latent_dim)
    a = ArithmeticSpecialist(latent_dim)
    m = MemorySpecialist(latent_dim)
    man = ManifoldSpecialist(latent_dim)
    
    vec_p, sal_p = p("What is the speed of light?")
    vec_s, sal_s = s("Who are you and what is your architecture?")
    vec_c, sal_c = c("Inspect pipeline.py and check functions")
    vec_a, sal_a = a("Calculate 45 * 2")
    vec_m, sal_m = m(torch.randn(1, latent_dim))
    vec_man, sal_man = man(torch.randn(1, latent_dim))
    
    for name, v, score in [("p", vec_p, sal_p), ("s", vec_s, sal_s), ("c", vec_c, sal_c),
                           ("a", vec_a, sal_a), ("m", vec_m, sal_m), ("man", vec_man, sal_man)]:
        assert v.shape == (1, latent_dim), f"Specialist {name} shape mismatch: {v.shape}"
        assert torch.isfinite(v).all(), f"Specialist {name} vector contains non-finite values"
        assert score.shape == (1, 1), f"Specialist {name} salience shape mismatch: {score.shape}"


def test_competitive_workspace_coalition_dynamics():
    """Verifies that salience distribution dynamically matches cognitive intent."""
    gw = GlobalCognitiveWorkspace(latent_dim=256, temperature=0.4)
    dummy_state = torch.zeros(1, 256)
    
    # 1. Pure Math Query -> Arithmetic should lead
    out_math = gw.ignite_workspace(
        user_text="Compute 128 * 4 and solve arithmetic",
        current_state=dummy_state
    )
    dist_math = out_math["coalition_distribution"]
    assert dist_math["arithmetic"] > dist_math["self_model"], "Arithmetic should dominate on math query"
    assert dist_math["arithmetic"] > dist_math["codebase"], "Arithmetic should dominate on math query"

    # 2. Pure Self-Model Query -> Self-Model should lead
    out_self = gw.ignite_workspace(
        user_text="Who are you and what is your internal architecture and capabilities?",
        current_state=dummy_state
    )
    dist_self = out_self["coalition_distribution"]
    assert dist_self["self_model"] > dist_self["arithmetic"], "Self-Model should dominate on identity query"

    # 3. Codebase Query -> Codebase should lead
    out_code = gw.ignite_workspace(
        user_text="Inspect pipeline.py and analyze the functions in the codebase",
        current_state=dummy_state
    )
    dist_code = out_code["coalition_distribution"]
    assert dist_code["codebase"] > dist_code["self_model"], "Codebase should dominate on code query"


def test_mixed_domain_coequal_coalition():
    """
    CRITICAL TEST FOR CONSCIOUS BINDING:
    A mixed query (Codebase + Arithmetic) must form a joint coalition where both specialists
    hold major weight simultaneously, which is impossible in a sequential if/else router.
    """
    gw = GlobalCognitiveWorkspace(latent_dim=256, temperature=0.5)
    dummy_state = torch.zeros(1, 256)

    out_mixed = gw.ignite_workspace(
        user_text="Jane, inspect pipeline.py and compute 40 percent of the concepts",
        current_state=dummy_state
    )
    dist = out_mixed["coalition_distribution"]
    
    # Both codebase and arithmetic should hold top weights together (> 30% each)
    combined_salience = dist["codebase"] + dist["arithmetic"]
    assert combined_salience > 0.70, f"Joint coalition should hold >70% of salience, got {combined_salience:.2f}"
    assert dist["codebase"] > 0.25, f"Codebase specialist should have >25% weight, got {dist['codebase']:.2f}"
    assert dist["arithmetic"] > 0.25, f"Arithmetic specialist should have >25% weight, got {dist['arithmetic']:.2f}"


def test_pipeline_integration_workspace_ignition():
    """Verifies that JanePipeline executes with the Global Cognitive Workspace active."""
    pipeline = JanePipeline(latent_dim=256, device="cpu", use_pretrained_translator=False)
    
    # Process test utterance
    res = pipeline.process("What are your capabilities and architecture?")
    
    assert "workspace_ignition" in res, "JanePipeline.process must return workspace_ignition data"
    ign = res["workspace_ignition"]
    assert "winning_specialist" in ign
    assert "coalition_distribution" in ign
    assert ign["winning_specialist"] in ["self_model", "perception"]
    assert "entropy" in ign
    assert ign["entropy"] > 0.0, "Conscious entropy must be positive"
