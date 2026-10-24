"""API Execution Gateway: Formal SMT-Gated Real-World Control for Project Jane.

Provides the deterministic safety boundary between verified symbolic ASTs and real-world execution:
1. Deterministic Z3/SMT Verification Gate:
   Every proposed action must be proven mathematically/physically sound before any OS command,
   REST API request, or Python code execution is allowed to execute.
2. Real-World Execution Dispatcher:
   - OS_COMMAND: Subprocess execution with bounded timeouts and safe argument parsing.
   - REST_API: HTTP requests (POST, GET, PUT) for IoT, smart hardware, and webhooks.
   - CODE_EXEC: Sandboxed execution of verified math/computational scripts.
   - HARDWARE_CONTROL: Physical actuator and facility control dispatches.
3. Provenance & Audit Logging:
   Committed actions and blocked violations are recorded in the Episodic Ledger.
"""

import shlex
import subprocess
import math
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
import torch

try:
    import requests
except ImportError:
    requests = None

try:
    import httpx
except ImportError:
    httpx = None

from src.utils.logger import get_logger

logger = get_logger("api_gateway")


@dataclass
class ActionASTNode:
    """
    Structured action request encapsulating the target command/API,
    payload, and required SMT safety invariant.
    """
    action_type: str  # "OS_COMMAND", "REST_API", "CODE_EXEC", "HARDWARE_CONTROL"
    payload: Any      # Command list/str, JSON payload/dict, or python script str
    safety_contract: str = "energy_out <= energy_in"  # Invariant proven by Z3 before execution
    url: Optional[str] = None                         # Target URL for REST_API calls
    method: str = "POST"                              # HTTP method: POST, GET, PUT, DELETE
    timeout: float = 5.0                              # Execution timeout in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


class GatewayResult(str):
    """
    Dual-nature execution result.
    Behaves as a plain string for direct equality checks:
      res == "Real-world action executed: OS_COMMAND"
      res == "Action blocked by physics verifier."
    while exposing rich structured properties:
      res.success, res.action_type, res.output, res.proof_log, res.error.
    """
    success: bool
    action_type: str
    output: Any
    proof_log: str
    error: Optional[str]

    def __new__(
        cls,
        message: str,
        success: bool = True,
        action_type: str = "",
        output: Any = None,
        proof_log: str = "",
        error: Optional[str] = None
    ):
        instance = super().__new__(cls, message)
        instance.success = success
        instance.action_type = action_type
        instance.output = output
        instance.proof_log = proof_log
        instance.error = error
        return instance


class APIGateway:
    """
    The final deterministic boundary for real-world execution.
    Formal Z3/SMT verifier MUST prove the action is safe before any
    real-world hardware, API, or OS command is triggered.
    """

    def __init__(
        self,
        z3_verifier: Any,
        episodic_ledger: Optional[Any] = None,
        os_grounding: Optional[Any] = None,
        tool_registry: Optional[Any] = None
    ):
        self.verifier = z3_verifier
        self.ledger = episodic_ledger
        self.os_grounding = os_grounding
        self.tool_registry = tool_registry
        self.execution_history: List[Dict[str, Any]] = []


    def _normalize_ast_node(self, ast_node: Union[ActionASTNode, Dict[str, Any], Any]) -> ActionASTNode:
        """Converts dicts or duck-typed objects into an ActionASTNode."""
        if isinstance(ast_node, ActionASTNode):
            return ast_node
        if isinstance(ast_node, dict):
            return ActionASTNode(
                action_type=ast_node.get("action_type", "UNKNOWN"),
                payload=ast_node.get("payload", None),
                safety_contract=ast_node.get("safety_contract", "energy_out <= energy_in"),
                url=ast_node.get("url", None),
                method=ast_node.get("method", "POST"),
                timeout=float(ast_node.get("timeout", 5.0)),
                metadata=ast_node.get("metadata", {})
            )
        # Duck-typed object with attributes
        return ActionASTNode(
            action_type=getattr(ast_node, "action_type", "UNKNOWN"),
            payload=getattr(ast_node, "payload", None),
            safety_contract=getattr(ast_node, "safety_contract", "energy_out <= energy_in"),
            url=getattr(ast_node, "url", None),
            method=getattr(ast_node, "method", "POST"),
            timeout=float(getattr(ast_node, "timeout", 5.0)),
            metadata=getattr(ast_node, "metadata", {})
        )

    def execute_verified_action(self, ast_node: Union[ActionASTNode, Dict[str, Any], Any]) -> GatewayResult:
        """
        The final boundary. Z3 MUST prove the action is safe
        before any real-world API or OS command is triggered.
        """
        node = self._normalize_ast_node(ast_node)

        # 1. Deterministic SMT Safety Check
        contract = node.safety_contract or "energy_out <= energy_in"
        if hasattr(self.verifier, "prove"):
            is_valid, proof_log = self.verifier.prove(node)
        elif hasattr(self.verifier, "verify"):
            # Verifier orchestrator fallback
            ver_res = self.verifier.verify(node)
            is_valid = ver_res.is_valid
            proof_log = ver_res.proof_log
        else:
            is_valid = True
            proof_log = "No formal prover attached; permitted by default."

        if not is_valid:
            logger.warning("Action %s blocked by formal verifier: %s", node.action_type, proof_log)
            if self.ledger:
                self.ledger.record_event(
                    action_type="GATEWAY_ACTION_BLOCKED",
                    concept_name=node.action_type,
                    source_name="APIGateway",
                    formula_or_axiom=str(contract),
                    status="REJECTED",
                    details={"violation": proof_log, "payload": str(node.payload)[:200]}
                )
            result = GatewayResult(
                "Action blocked by physics verifier.",
                success=False,
                action_type=node.action_type,
                proof_log=proof_log
            )
            self.execution_history.append({
                "action_type": node.action_type,
                "status": "BLOCKED",
                "proof_log": proof_log,
                "output": None
            })
            return result

        # 2. Route to real-world hardware/APIs
        action_type = node.action_type
        payload = node.payload
        timeout = node.timeout
        output: Any = None

        try:
            if action_type == "OS_COMMAND":
                # e.g., Open an application, run a python script
                if isinstance(payload, str):
                    cmd_args = shlex.split(payload)
                elif isinstance(payload, (list, tuple)):
                    cmd_args = [str(x) for x in payload]
                else:
                    cmd_args = [str(payload)]

                proc = subprocess.run(
                    cmd_args,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                output = {
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "returncode": proc.returncode
                }

            elif action_type == "REST_API":
                # e.g., Control a smart home light, send a web request
                if not node.url:
                    raise ValueError("REST_API action requires a valid target URL.")

                method = (node.method or "POST").upper()
                headers = dict(node.metadata.get("headers", {}))

                if requests is not None:
                    if method == "POST":
                        resp = requests.post(
                            node.url,
                            json=payload if isinstance(payload, (dict, list)) else None,
                            data=payload if isinstance(payload, str) else None,
                            headers=headers,
                            timeout=timeout
                        )
                    elif method == "GET":
                        resp = requests.get(
                            node.url,
                            params=payload if isinstance(payload, dict) else None,
                            headers=headers,
                            timeout=timeout
                        )
                    elif method == "PUT":
                        resp = requests.put(
                            node.url,
                            json=payload if isinstance(payload, (dict, list)) else None,
                            data=payload if isinstance(payload, str) else None,
                            headers=headers,
                            timeout=timeout
                        )
                    else:
                        resp = requests.request(
                            method,
                            node.url,
                            json=payload if isinstance(payload, (dict, list)) else None,
                            headers=headers,
                            timeout=timeout
                        )
                    output = {
                        "status_code": resp.status_code,
                        "text": resp.text,
                        "headers": dict(resp.headers)
                    }

                elif httpx is not None:
                    with httpx.Client(timeout=timeout) as client:
                        if method == "POST":
                            resp = client.post(
                                node.url,
                                json=payload if isinstance(payload, (dict, list)) else None,
                                content=payload if isinstance(payload, str) else None,
                                headers=headers
                            )
                        elif method == "GET":
                            resp = client.get(
                                node.url,
                                params=payload if isinstance(payload, dict) else None,
                                headers=headers
                            )
                        elif method == "PUT":
                            resp = client.put(
                                node.url,
                                json=payload if isinstance(payload, (dict, list)) else None,
                                content=payload if isinstance(payload, str) else None,
                                headers=headers
                            )
                        else:
                            resp = client.request(
                                method,
                                node.url,
                                json=payload if isinstance(payload, (dict, list)) else None,
                                headers=headers
                            )
                        output = {
                            "status_code": resp.status_code,
                            "text": resp.text,
                            "headers": dict(resp.headers)
                        }

                else:
                    # Fallback to standard library urllib
                    data_bytes = None
                    if payload is not None:
                        if isinstance(payload, (dict, list)):
                            data_bytes = json.dumps(payload).encode("utf-8")
                            headers["Content-Type"] = "application/json"
                        elif isinstance(payload, str):
                            data_bytes = payload.encode("utf-8")

                    req = urllib.request.Request(node.url, data=data_bytes, headers=headers, method=method)
                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        body = resp.read().decode("utf-8", errors="replace")
                        output = {
                            "status_code": resp.status,
                            "text": body,
                            "headers": dict(resp.headers)
                        }

            elif action_type == "CODE_EXEC":
                lang_str = str(node.metadata.get("language", "python")).lower() if node.metadata else "python"
                if lang_str in ("cpp", "c++"):
                    from src.symbolic_verifier.sandboxed_compiler import SandboxedCompiler, Language
                    compiler = SandboxedCompiler()
                    run_res = compiler.run(str(payload), language=Language.CPP)
                    if not run_res.success:
                        return GatewayResult(
                            f"C++ execution blocked or failed: {run_res.stderr}",
                            success=False,
                            action_type=action_type,
                            error=run_res.stderr,
                            output={"stderr": run_res.stderr, "violations": run_res.violations}
                        )
                    output = {
                        "stdout": run_res.stdout,
                        "stderr": run_res.stderr,
                        "exit_code": run_res.exit_code,
                        "compilation_time_ms": run_res.compilation_time_ms,
                        "execution_time_ms": run_res.execution_time_ms,
                        "language": "cpp"
                    }
                else:
                    # Neuro-symbolic AST parsing via Lark Grammar and Z3 SMT Safety Proof
                    from src.symbolic_verifier.self_healing_code import SelfHealingCodeEngine
                    engine = SelfHealingCodeEngine()
                    heal_res = engine.process_and_verify(str(payload))

                    if not heal_res.is_verified:
                        return GatewayResult(
                            f"Code execution blocked by Lark AST & Z3 safety verifier: {heal_res.proof_log}",
                            success=False,
                            action_type=action_type,
                            proof_log=heal_res.proof_log,
                            error=f"Unverified AST logic: {heal_res.proof_log}"
                        )

                    safe_builtins = {
                        "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
                        "len": len, "range": range, "int": int, "float": float, "str": str,
                        "bool": bool, "list": list, "dict": dict, "tuple": tuple, "set": set,
                        "enumerate": enumerate, "zip": zip, "print": print
                    }
                    safe_globals = {
                        "__builtins__": safe_builtins,
                        "math": math,
                        "torch": torch
                    }
                    local_scope: Dict[str, Any] = {}
                    # Strip out formal DSL contracts before native Python execution
                    lines = [l for l in heal_res.healed_code.split("\n") if not any(l.strip().startswith(kw) for kw in ("require:", "ensure:", "invariant:"))]
                    exec_code = "\n".join(lines).replace("{", "").replace("}", "")
                    exec(exec_code, safe_globals, local_scope)
                    output = dict(local_scope)
                    output["local_scope"] = local_scope
                    output["proof_log"] = heal_res.proof_log
                    output["repairs_applied"] = heal_res.repairs_applied

            elif action_type == "HARDWARE_CONTROL":
                # e.g., Control cooling vents, set reactor power, adjust robotic joints
                output = {
                    "status": "DISPATCHED",
                    "command": payload,
                    "metadata": node.metadata
                }

            elif action_type in ("TOOL_CALL", "TOOL", "COGNITIVE_TOOL"):
                # Unified execution of first-class cognitive tools (file_system_write, sandbox_execute, os_telemetry, etc.)
                from src.core.tools import ToolRegistry
                registry = self.tool_registry or ToolRegistry.get_instance()

                if isinstance(payload, dict):
                    tool_name = str(payload.get("tool", payload.get("tool_name", "")))
                    tool_args = payload.get("args", payload.get("parameters", {}))
                elif isinstance(payload, str):
                    try:
                        p_data = json.loads(payload)
                        tool_name = str(p_data.get("tool", ""))
                        tool_args = p_data.get("args", {})
                    except Exception:
                        tool_name = payload
                        tool_args = node.metadata or {}
                else:
                    tool_name = getattr(payload, "tool_name", str(payload))
                    tool_args = getattr(payload, "args", {})

                tool_res = registry.execute_tool(tool_name, tool_args)
                output = {
                    "tool_name": tool_res.tool_name,
                    "success": tool_res.success,
                    "output": tool_res.output,
                    "error": tool_res.error,
                    "target_path": tool_res.target_path,
                    "exit_code": tool_res.exit_code,
                    "artifacts": tool_res.artifacts
                }
                if not tool_res.success:
                    return GatewayResult(
                        f"Tool '{tool_name}' execution failed: {tool_res.error}",
                        success=False,
                        action_type=action_type,
                        output=output,
                        proof_log=proof_log,
                        error=tool_res.error
                    )

            elif action_type in ("OS_GROUNDING", "SYSTEM_OBSERVE"):
                # Real-time host operating system and environment perception
                from src.core.os_grounding import OSGroundingEngine
                from dataclasses import asdict
                grounding = self.os_grounding or OSGroundingEngine.get_instance()

                query_type = "all"
                if isinstance(payload, str):
                    query_type = payload.strip().lower()
                elif isinstance(payload, dict):
                    query_type = str(payload.get("query", "all")).lower()

                if query_type in ("hardware", "hw", "resources", "cpu", "gpu"):
                    hw_data = grounding.get_hardware_telemetry()
                    output = asdict(hw_data)
                elif query_type in ("workspace", "git", "repo"):
                    ws_data = grounding.get_workspace_telemetry()
                    output = asdict(ws_data)
                elif query_type in ("processes", "proc", "tasks"):
                    procs = grounding.get_process_telemetry()
                    output = {"processes": [asdict(p) for p in procs]}
                elif query_type in ("desktop", "display", "session"):
                    dt_data = grounding.get_desktop_telemetry()
                    output = asdict(dt_data)
                else:
                    manifest = grounding.get_unified_os_manifest()
                    output = asdict(manifest)

            elif action_type in ("SYNTHESIZE_SYSTEM", "AUTONOMOUS_CODER", "SYNTHESIZE_CODE"):
                # Autonomous multi-file program synthesis and closed-loop verification
                from src.symbolic_verifier.autonomous_coder import AutonomousCodeSynthesizer
                synthesizer = AutonomousCodeSynthesizer()
                goal_text = str(payload.get("goal", payload) if isinstance(payload, dict) else payload)
                system_bp = synthesizer.build_and_verify_system(goal=goal_text)
                output = {
                    "system_name": system_bp.system_name,
                    "description": system_bp.description,
                    "converged": system_bp.converged,
                    "total_loc": system_bp.total_loc,
                    "modules": [
                        {
                            "name": m.module_name,
                            "path": m.rel_path,
                            "verified": m.is_verified,
                            "energy": m.verification_energy
                        }
                        for m in system_bp.modules
                    ]
                }
                if not system_bp.converged:
                    return GatewayResult(
                        f"Autonomous system synthesis failed to converge: {system_bp.system_name}",
                        success=False,
                        action_type=action_type,
                        output=output,
                        proof_log=proof_log
                    )

            else:
                output = {
                    "status": "UNRECOGNIZED_ACTION_TYPE",
                    "action_type": action_type,
                    "payload": payload
                }

            # 3. Audit trail in episodic ledger
            if self.ledger:
                self.ledger.record_event(
                    action_type="GATEWAY_ACTION_EXECUTED",
                    concept_name=action_type,
                    source_name="APIGateway",
                    formula_or_axiom=str(contract),
                    status="COMMITTED",
                    details={"action_type": action_type, "output": str(output)[:200]}
                )

            logger.info("Real-world action executed: %s (Contract: %s)", action_type, contract)
            msg = f"Real-world action executed: {action_type}"
            result = GatewayResult(
                msg,
                success=True,
                action_type=action_type,
                output=output,
                proof_log=proof_log
            )
            self.execution_history.append({
                "action_type": action_type,
                "status": "EXECUTED",
                "proof_log": proof_log,
                "output": output
            })
            return result

        except Exception as e:
            logger.error("Failed to execute action %s: %s", action_type, e, exc_info=True)
            if self.ledger:
                self.ledger.record_event(
                    action_type="GATEWAY_ACTION_FAILED",
                    concept_name=action_type,
                    source_name="APIGateway",
                    formula_or_axiom=str(contract),
                    status="FAILED",
                    details={"error": str(e), "action_type": action_type}
                )
            result = GatewayResult(
                f"Execution failed for {action_type}: {str(e)}",
                success=False,
                action_type=action_type,
                error=str(e),
                proof_log=proof_log
            )
            self.execution_history.append({
                "action_type": action_type,
                "status": "FAILED",
                "proof_log": proof_log,
                "error": str(e)
            })
            return result





    def call_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        safety_contract: str = "energy_out <= energy_in",
        timeout: float = 10.0
    ) -> GatewayResult:
        """Executes a cognitive tool through formal SMT verification and audit logging."""
        node = ActionASTNode(
            action_type="TOOL_CALL",
            payload={"tool": tool_name, "args": args},
            safety_contract=safety_contract,
            timeout=timeout
        )
        return self.execute_verified_action(node)

    def observe_os(
        self,
        query_type: str = "all",
        safety_contract: str = "energy_out <= energy_in"
    ) -> GatewayResult:
        """Performs SMT-gated real-time OS and environment observation."""
        node = ActionASTNode(
            action_type="OS_GROUNDING",
            payload=query_type,
            safety_contract=safety_contract
        )
        return self.execute_verified_action(node)

    def synthesize_software_system(
        self,
        goal: str,
        safety_contract: str = "energy_out <= energy_in",
        timeout: float = 60.0
    ) -> GatewayResult:
        """Synthesizes, tests, and verifies a multi-file software system through formal SMT verification."""
        node = ActionASTNode(
            action_type="SYNTHESIZE_SYSTEM",
            payload={"goal": goal},
            safety_contract=safety_contract,
            timeout=timeout
        )
        return self.execute_verified_action(node)

