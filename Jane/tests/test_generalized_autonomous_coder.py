"""Tests for Generalized Socratic Architectural Interviewer,
Dynamic Multi-Module Software Decomposer, and Closed-Loop AST Super-Coder.

Verifies:
1. Socratic Architectural Review detects design forks (concurrency, persistence, strictness).
2. Decomposes non-AI software goals (Key-Value cache, Message broker, API gateway) dynamically.
3. Zero-hardcoding contract-driven AST source and test synthesis.
4. Closed-loop sandbox verification converges with Lyapunov energy E = 0.0.
5. Multi-file directory scaffolding and test suite generation.
"""

import os
import shutil
import tempfile
import ast

from src.symbolic_verifier.autonomous_coder import (
    AutonomousCodeSynthesizer,
    SocraticArchitecturalInterviewer,
    DynamicSystemDecomposer,
    DynamicASTModuleSynthesizer,
    SystemBlueprint,
    ModuleBlueprint,
    FunctionContract
)


def test_socratic_architectural_interviewer_review():
    """Verify Socratic architectural interview generates decision points and resolves them."""
    interviewer = SocraticArchitecturalInterviewer()
    
    # 1. Neural AI goal
    ai_interview = interviewer.conduct_interview("Build a continuous rotary attention network")
    assert ai_interview.domain == "neural_ai"
    assert len(ai_interview.questions) >= 3
    q_keys = [q.key for q in ai_interview.questions]
    assert "concurrency_model" in q_keys
    assert "persistence_strategy" in q_keys
    assert "invariant_strictness" in q_keys
    assert ai_interview.decisions["persistence_strategy"] == "In-Memory Transient (PyTorch / DataStructures)"

    # 2. Web / API goal
    api_interview = interviewer.conduct_interview("Create an asynchronous REST API authentication gateway")
    assert api_interview.domain == "web_services"
    assert api_interview.decisions["concurrency_model"] == "Asynchronous Event-Driven (asyncio)"

    # 3. Custom answer resolution
    custom_decisions = api_interview.resolve_with_answers({
        "concurrency_model": "Thread Pool Worker",
        "persistence_strategy": "Persistent SQLite / JSON Ledger"
    })
    assert custom_decisions["concurrency_model"] == "Thread Pool Worker"
    assert custom_decisions["persistence_strategy"] == "Persistent SQLite / JSON Ledger"


def test_dynamic_decomposer_diverse_non_ai_systems():
    """Verify system decomposition works dynamically for diverse non-AI domains without hardcoding."""
    decomposer = DynamicSystemDecomposer()

    # 1. Key-Value Storage Cache
    kv_bp = decomposer.decompose("Build an in-memory key-value cache with eviction")
    assert isinstance(kv_bp, SystemBlueprint)
    assert len(kv_bp.modules) >= 2
    for m in kv_bp.modules:
        assert m.module_name != ""
        assert m.rel_path.endswith(".py")
        assert m.test_path.endswith(".py")
        assert len(m.formal_invariants) > 0

    # 2. Event-Driven Message Broker
    broker_bp = decomposer.decompose("Create a pub-sub event message broker with topic filtering")
    assert isinstance(broker_bp, SystemBlueprint)
    assert len(broker_bp.modules) >= 2
    assert "Broker" in broker_bp.system_name or "Event" in broker_bp.system_name or "Message" in broker_bp.system_name

    # 3. Particle Physics Engine
    physics_bp = decomposer.decompose("Simulate 2D particle kinetics with elastic collisions")
    assert isinstance(physics_bp, SystemBlueprint)
    assert len(physics_bp.modules) >= 2


def test_dynamic_ast_module_synthesizer_syntax_validity():
    """Verify that synthesized source code and tests parse into valid Python ASTs."""
    synthesizer = DynamicASTModuleSynthesizer()

    blueprint = ModuleBlueprint(
        module_name="token_bucket_limiter",
        rel_path="src/rate/limiter.py",
        test_path="tests/test_limiter.py",
        description="Token bucket rate limiter with replenish intervals.",
        dependencies=["typing", "math"],
        classes=[{
            "name": "TokenBucketLimiter",
            "doc": "Limits request rate using consumable tokens.",
            "methods": ["__init__(capacity=100, refill_rate=10.0)", "allow_request(tokens=1)"]
        }],
        functions=[
            FunctionContract(
                name="calculate_elapsed_refill",
                doc="Calculates tokens added over elapsed interval dt.",
                formal_contracts=["dt_positive: dt >= 0.0"]
            )
        ],
        formal_invariants=["capacity_positive", "tokens_bounded"]
    )

    source, tests = synthesizer.synthesize_module(blueprint)
    
    # Must parse without syntax errors
    source_ast = ast.parse(source)
    assert isinstance(source_ast, ast.Module)
    assert "class TokenBucketLimiter" in source
    assert "def calculate_elapsed_refill" in source

    test_ast = ast.parse(tests)
    assert isinstance(test_ast, ast.Module)
    assert "def test_token_bucket_limiter_execution" in tests


def test_end_to_end_closed_loop_synthesis_of_custom_system():
    """Verify full build_and_verify_system on a novel, non-hardcoded software goal."""
    tmp_dir = tempfile.mkdtemp(prefix="jane_test_custom_coder_")
    try:
        coder = AutonomousCodeSynthesizer(worktree_dir=tmp_dir)
        goal = "Build a distributed event dispatcher and schema validator"
        system_bp = coder.build_and_verify_system(goal)

        assert system_bp.converged is True
        assert len(system_bp.modules) >= 2
        assert system_bp.total_loc > 20

        for m in system_bp.modules:
            assert m.is_verified is True
            assert m.verification_energy == 0.0
            
            # Verify file landed on disk
            full_path = os.path.join(tmp_dir, system_bp.system_name, m.rel_path)
            assert os.path.exists(full_path)
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert len(content) > 30
                assert "class " in content or "def " in content
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
