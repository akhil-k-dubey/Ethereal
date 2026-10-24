"""Action Directive Dispatcher for Project Jane (Neuro-Symbolic & Zero Regex).

Bridges continuous conversational intent with real-world sandboxed execution:
1. Intent Classification: ContinuousIntentManifold vector projection (Zero string prefix checks).
2. Neural Action Synthesis: LLaMA/Qwen model synthesizes structured JSON action node (filename, directory, language, dynamic code).
3. Symbolic Verification: Formal path safety containment proof & StaticSecurityAuditor AST check.
4. Sandboxed Execution: POSIX resource isolation (RLIMIT_AS, RLIMIT_CPU) via SandboxedCompiler.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from src.core.api_gateway import APIGateway
from src.symbolic_verifier.sandboxed_compiler import SandboxedCompiler, StaticSecurityAuditor
from src.reasoning_engine.intent_manifold import ContinuousIntentManifold
from src.utils.logger import get_logger

logger = get_logger("action_dispatcher")


@dataclass
class ActionDirectiveResult:
    """Telemetry returned after executing a conversational action directive."""
    is_action: bool
    action_type: str = ""
    target_path: Optional[str] = None
    language: str = "python"
    code: str = ""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    message: str = ""


class ActionDirectiveDispatcher:
    """
    Translates natural language human directives into verified real-world actions.
    Uses continuous vector manifold for intent detection, neural synthesis for
    structured action parameters, and deterministic Z3/POSIX sandboxing for execution.
    """

    def __init__(
        self,
        api_gateway: Optional[APIGateway] = None,
        compiler: Optional[SandboxedCompiler] = None,
        llm_backend: Optional[Any] = None,
        intent_manifold: Optional[ContinuousIntentManifold] = None
    ):
        self.api_gateway = api_gateway or APIGateway(z3_verifier=None)
        self.compiler = compiler or SandboxedCompiler()
        self.llm_backend = llm_backend
        self.intent_manifold = intent_manifold or ContinuousIntentManifold.get_instance()


    def is_action_directive(self, text: str) -> bool:
        """
        Determines if user input is an actionable directive using the 256D continuous intent manifold
        and native NeuroSymbolicProgramSynthesizer domain projections.
        Zero hardcoded prefix matching.
        """
        if not text or not text.strip():
            return False

        sim_action = self.intent_manifold.similarity(text, "action_directive")
        sim_cap = self.intent_manifold.similarity(text, "capabilities")
        sim_lang = self.intent_manifold.similarity(text, "language_capability")
        sim_epistemic = self.intent_manifold.similarity(text, "epistemic_discovery")
        sim_inquiry = max(sim_cap, sim_lang, sim_epistemic)

        # Check domain resonance from native NeuroSymbolicProgramSynthesizer
        from src.symbolic_verifier.neuro_symbolic_synthesizer import NeuroSymbolicProgramSynthesizer
        synth = NeuroSymbolicProgramSynthesizer.get_instance()
        domain, sim_domain = synth.route_domain(text)

        # 0. Conversational capability or inquiry projection (e.g. "do you know how to", "can you explain")
        v_inquiry = self.intent_manifold.encode("do you know can you tell me explain describe how to")
        v_action_cmd = self.intent_manifold.encode("create write make build generate implement synthesize run")
        text_v = self.intent_manifold.encode(text)
        is_inquiry_query = torch.dot(text_v, v_inquiry).item() > torch.dot(text_v, v_action_cmd).item()

        if is_inquiry_query and sim_action < 0.35:
            return False

        # 1. Epistemic capability or knowledge inquiry suppression
        if sim_inquiry >= 0.30 and sim_inquiry >= sim_action and sim_domain < 0.65:
            return False

        # 2. High-confidence software or web synthesis directive
        if sim_domain >= 0.60 and sim_domain > (sim_inquiry + 0.15):
            return True

        # 3. General action directive resonance
        if sim_action >= 0.40 and sim_action > sim_inquiry:
            return True

        return False

    def parse_and_dispatch(self, user_text: str) -> Optional[ActionDirectiveResult]:
        """
        Synthesizes structured action from natural language, verifies safety, executes,
        and returns deterministic execution telemetry. Returns None if not an action.
        """
        if not self.is_action_directive(user_text):
            return None

        from src.symbolic_verifier.neuro_symbolic_synthesizer import NeuroSymbolicProgramSynthesizer
        synth = NeuroSymbolicProgramSynthesizer.get_instance()

        domain, _ = synth.route_domain(user_text)
        directory = synth.resolve_directory(user_text)
        filename = synth.resolve_filename(user_text, domain)

        synth_res = synth.synthesize_and_verify(
            task_description=user_text,
            target_directory=directory,
            target_filename=filename
        )

        output_str = synth_res.execution_result.clean_output if synth_res.execution_result else synth_res.message
        error_str = synth_res.execution_result.stderr if (synth_res.execution_result and not synth_res.success) else ""
        exit_code = synth_res.execution_result.exit_code if synth_res.execution_result else (0 if synth_res.success else 1)

        return ActionDirectiveResult(
            is_action=True,
            action_type="CREATE_FILE",
            target_path=synth_res.target_path,
            language=synth_res.language,
            code=synth_res.code,
            success=synth_res.success,
            stdout=output_str,
            stderr=error_str,
            exit_code=exit_code,
            message=synth_res.message
        )



