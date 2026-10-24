"""Tests for Jarvis Workshop Comprehension Engine:
1. PhoneticSyntacticReconstructor (acoustic CTC typo & homophone reconstruction + noise fragment filtering)
2. WernickeSemanticCompiler (compositional operators, anaphora, focus redirects, logical implications)
3. AutonomousWorkshopExecutor & WorkshopLedger (physical document creation, simulation execution, desktop launch)
4. AntiHallucinationCritic & PragmaticGrounder protection against Qwen hijacking and focus-redirect traps.
"""

import os
from pathlib import Path
import pytest

from src.reasoning_engine.workshop_comprehension import (
    PhoneticSyntacticReconstructor,
    WernickeSemanticCompiler,
    WorkshopLedger,
    AutonomousWorkshopExecutor,
)
from src.reasoning_engine.pragmatic_grounder import PragmaticGrounder
from src.symbolic_verifier.hallucination_critic import AntiHallucinationCritic


def test_phonetic_syntactic_reconstructor_recovers_voice_session_utterances():
    """Verifies that all garbled CTC utterances from live voice sessions reconstruct cleanly."""
    cases = [
        ("how many consept you have ji", "how many concepts do you have Jane"),
        ("aa and how many of them have you learned resin", "aa and how many of them have you learned recently"),
        ("can yo scan you af cortifis for me", "can you scan your codebase files for me"),
        ("erena coused a munition for", "run a simulation for it"),
        ("start with his imple il", "start with a simple one"),
        ("did you creator something", "did you create something"),
        (
            "that's not what i hasd i mean did you rein some simulations or nothing at all",
            "that's not what I asked I mean did you run some simulations or nothing at all",
        ),
        ("creatad document for horoblad er holographic", "create a document for holographic projection"),
        ("just greater document for law", "just create a document for now"),
        ("did you createn the document", "did you create the document"),
        ("open that document you greated disnal", "open that document you created just now"),
        ("ell you created it you should know", "well you created it you should know"),
        ("show me that on my", "show me that on my screen"),
    ]
    for raw_ctc, expected in cases:
        reconstructed, conf, is_noise = PhoneticSyntacticReconstructor.reconstruct(raw_ctc)
        assert reconstructed.lower() == expected.lower(), f"Failed on {raw_ctc!r}: got {reconstructed!r}"
        assert conf >= 0.75
        assert not is_noise


def test_noise_fragments_filtered():
    """Verifies 1-syllable chopped CTC fragments ('ha', 'ba', 'shod', 'bely') are flagged as noise."""
    for frag in ("ha", "ba", "shod", "bely", "bounday", "shot a gorn"):
        _, _, is_noise = PhoneticSyntacticReconstructor.reconstruct(frag)
        assert is_noise is True, f"Expected {frag!r} to be flagged as noise fragment"


def test_wernicke_compiler_and_autonomous_workshop_execution(tmp_path, monkeypatch):
    """Verifies end-to-end autonomous file creation, simulation execution, audit, and 'you created it' resolution."""
    ledger_file = tmp_path / "workshop_ledger.json"
    ledger = WorkshopLedger(ledger_path=str(ledger_file))
    compiler = WernickeSemanticCompiler(ledger=ledger)
    executor = AutonomousWorkshopExecutor(ledger=ledger)
    executor.default_artifacts_dir = tmp_path
    # Avoid spawning GUI windows during headless pytest run
    monkeypatch.setattr(executor, "launch_on_desktop", lambda path: True)

    # 1. "just greater document for law" -> CREATE_AND_OPEN_DOCUMENT
    f1 = compiler.compile_meaning("just greater document for law")
    assert f1.intent_class == "EXECUTE_OS_ACTION"
    assert f1.operation == "CREATE_AND_OPEN_DOCUMENT"
    res1 = executor.execute_frame(f1)
    assert res1 is not None
    doc_path = res1["workshop_artifact"]["file_path"]
    assert os.path.exists(doc_path)
    assert "opened it directly on your screen" in res1["response_text"]

    # 2. "did you createn the document" -> VERIFY_WORKSHOP_STATE (finds the created document!)
    f2 = compiler.compile_meaning("did you createn the document")
    assert f2.intent_class == "VERIFY_WORKSHOP_STATE"
    res2 = executor.execute_frame(f2)
    assert res2 is not None
    assert os.path.basename(doc_path) in res2["response_text"]

    # 3. "open that document you greated disnal" & "ell you created it you should know" -> OPEN_ON_DESKTOP
    for prompt in ("open that document you greated disnal", "ell you created it you should know", "show me that on my"):
        f_open = compiler.compile_meaning(prompt)
        assert f_open.intent_class == "EXECUTE_OS_ACTION"
        assert f_open.operation == "OPEN_ON_DESKTOP"
        res_open = executor.execute_frame(f_open)
        assert res_open is not None
        assert doc_path in res_open["response_text"]

    # 4. "erena coused a munition for" / "start with his imple il" -> RUN_AND_SHOW_SIMULATION
    f_sim = compiler.compile_meaning(
        "start with his imple il",
        history=[{"role": "user", "content": "let's work on holographic projection"}],
    )
    assert f_sim.intent_class == "EXECUTE_OS_ACTION"
    assert f_sim.operation == "RUN_AND_SHOW_SIMULATION"
    res_sim = executor.execute_frame(f_sim)
    assert res_sim is not None
    sim_script = res_sim["workshop_artifact"]["file_path"]
    sim_plot = res_sim["workshop_artifact"]["visual_output_path"]
    assert os.path.exists(sim_script)
    assert sim_plot and os.path.exists(sim_plot)

    # 5. Focus redirect: "that's not what i hasd i mean did you rein some simulations or nothing at all"
    f_audit_sim = compiler.compile_meaning(
        "that's not what i hasd i mean did you rein some simulations or nothing at all"
    )
    assert f_audit_sim.intent_class == "VERIFY_WORKSHOP_STATE"
    assert f_audit_sim.artifact_filter == "simulation"
    res_audit_sim = executor.execute_frame(f_audit_sim)
    assert res_audit_sim is not None
    assert os.path.basename(sim_script) in res_audit_sim["response_text"]


def test_pragmatic_grounder_ignores_focus_redirect_questions(tmp_path):
    """Verifies PragmaticGrounder never turns 'that's not what I asked, I mean did you...' into an idiom."""
    pg = PragmaticGrounder(pragmatics_path=str(tmp_path / "pragmatics.json"))
    res1 = pg.parse_pragmatic_clarification(
        "that's not what i hasd i mean did you rein some simulations or nothing at all"
    )
    assert res1.is_clarification is False
    res2 = pg.parse_pragmatic_clarification("ah beell that's not what i mean")
    assert res2.is_clarification is False


def test_critic_blocks_text_based_ai_denials_and_allows_screen_discussion():
    """Verifies AntiHallucinationCritic blocks Qwen's 'as a text-based AI' refusal and doesn't spam CLI screen refusal."""
    critic = AntiHallucinationCritic()
    verdict1 = critic.audit(
        "I'm sorry for the confusion, but as a text-based AI, I don't have direct access to your local files or the ability to display them on your screen.",
        "show me that on my screen",
    )
    assert "CODE_EXECUTION_DENIAL" in verdict1.violations
    assert "as a text-based ai" not in verdict1.scrubbed_text.lower()

    verdict2 = critic.audit(
        "I have opened the document directly on your screen, Akhil.",
        "open that document",
    )
    assert "FALSE_VISION_CLAIM" not in verdict2.violations
    assert "cannot see your screen right now" not in verdict2.scrubbed_text.lower()
