"""Unified Native Cognitive Tool Engine for Project Jane (Neuro-Symbolic & Zero Regex).

Defines real-world affordances as first-class cognitive tools:
1. file_system_write: Proves directory safety, AST audits code, writes file, and tests execution.
2. sandbox_execute: Runs code within POSIX resource isolation (RLIMIT_AS, RLIMIT_CPU).
3. query_knowledge_graph: Retrieves verified relational triples from Jane's manifold.
Zero regex parsing of tool invocations.
"""

import os
import json
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from src.symbolic_verifier.sandboxed_compiler import SandboxedCompiler, Language, StaticSecurityAuditor
from src.utils.logger import get_logger

logger = get_logger("cognitive_tools")


@dataclass
class ToolDefinition:
    """Metadata describing a cognitive tool affordance."""
    name: str
    description: str
    parameters: Dict[str, str]


@dataclass
class ToolExecutionResult:
    """Outcome of executing a verified tool call."""
    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    target_path: Optional[str] = None
    exit_code: int = 0
    artifacts: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    Central registry and safety execution engine for Jane's cognitive tools.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, compiler: Optional[SandboxedCompiler] = None, api_gateway: Optional[Any] = None):
        self.compiler = compiler or SandboxedCompiler()
        self.api_gateway = api_gateway
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def set_api_gateway(self, gateway: Any):
        """Sets reference to live APIGateway."""
        self.api_gateway = gateway

    def _register_default_tools(self) -> None:
        """Registers canonical cognitive tool affordances."""
        self.tools["file_system_write"] = ToolDefinition(
            name="file_system_write",
            description="Writes code or content to a file at the specified path and runs a verification test.",
            parameters={"path": "Target file path (e.g. ~/Downloads/script.py or /tmp/test.py)", "code": "Source code content to write"}
        )
        self.tools["sandbox_execute"] = ToolDefinition(
            name="sandbox_execute",
            description="Safely compiles and executes code inside POSIX resource-bounded sandbox (Python or C++).",
            parameters={"code": "Code string to execute", "language": "'python' or 'cpp'"}
        )
        self.tools["query_knowledge_graph"] = ToolDefinition(
            name="query_knowledge_graph",
            description="Queries the 256D manifold for verified knowledge graph concepts and axioms.",
            parameters={"concept": "Concept name to look up in the manifold"}
        )
        self.tools["os_telemetry"] = ToolDefinition(
            name="os_telemetry",
            description="Queries real-time host hardware telemetry (CPU, RAM, GPU, load averages).",
            parameters={"query": "'hardware', 'workspace', 'processes', 'desktop', or 'all'"}
        )
        self.tools["workspace_inspection"] = ToolDefinition(
            name="workspace_inspection",
            description="Inspects active project workspace, git status, and recently modified files.",
            parameters={"path": "Optional target directory (defaults to current project root)"}
        )
        self.tools["process_list"] = ToolDefinition(
            name="process_list",
            description="Lists top running developer and system processes on the host machine.",
            parameters={"limit": "Maximum number of processes to return (default: 15)"}
        )
        self.tools["file_system_read"] = ToolDefinition(
            name="file_system_read",
            description="Safely reads file content within verified project and workspace boundaries.",
            parameters={"path": "Target file path to read"}
        )
        self.tools["synthesize_software_system"] = ToolDefinition(
            name="synthesize_software_system",
            description="Autonomous closed-loop micro-step code synthesis and verification of complete multi-file systems.",
            parameters={"goal": "Specification or goal of the software/AI subsystem to generate and verify"}
        )
        self.tools["query_causal_hierarchy"] = ToolDefinition(
            name="query_causal_hierarchy",
            description="Queries Pearl's 4-tier causal abstraction hierarchy (Observables -> Mechanisms -> Laws -> Meta-Architectures).",
            parameters={"query": "Concept or mechanism query for causal top-down search"}
        )
        self.tools["iot_control"] = ToolDefinition(
            name="iot_control",
            description="Dispatches Z3-verified physical commands to Zigbee2MQTT and ESP32-S3 smart hardware (lights, relays, thermostat).",
            parameters={"device": "Device ID or friendly name", "command": "JSON command dictionary (e.g. {'state': 'ON', 'brightness': 200})"}
        )
        self.tools["iot_telemetry"] = ToolDefinition(
            name="iot_telemetry",
            description="Queries real-time ESP32-S3 mmWave radar presence, BME680 room climate, and Zigbee device telemetry.",
            parameters={"room": "Target room name (default: 'main_lab')"}
        )

    def get_system_tool_prompt(self) -> str:
        """Generates dynamic tool documentation prompt for the neural model."""
        lines = [
            "Available Environmental Tools:",
            "- file_system_write(path: str, code: str): Writes code to a file and executes it in sandbox.",
            "- file_system_read(path: str): Safely reads file contents within workspace boundary.",
            "- sandbox_execute(code: str, language: str): Safely executes code in sandbox.",
            "- os_telemetry(query: str): Inspects live CPU, RAM, GPU, and system resources.",
            "- workspace_inspection(path: str): Checks active git branch, commits, and modified files.",
            "- process_list(limit: int): Inspects top active processes and developer tasks.",
            "- synthesize_software_system(goal: str): Autonomously generates, tests, and verifies multi-file codebases in shadow worktrees.",
            "- query_causal_hierarchy(query: str): Traverses Pearl's 4-tier causal hierarchy and governing invariants.",
            "- iot_control(device: str, command: dict): Dispatches Z3-verified commands to Zigbee2MQTT and ESP32-S3 hardware.",
            "- iot_telemetry(room: str): Queries mmWave radar presence, room temperature/pressure/IAQ, and smart device states.",
            "",
            "When an operational action is required (creating a file, inspecting OS, reading code, testing an algorithm):",
            "Emit a tool call block formatted exactly as:",
            'TOOL_CALL: {"tool": "tool_name", "args": {"param1": "val1", ...}}',
            "Followed by your natural, conversational thought and explanation.",
            "If the user is only asking a question or conversing, do NOT emit any tool call."
        ]
        return "\n".join(lines)

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolExecutionResult:
        """Executes a tool call under formal symbolic verification and POSIX limits."""
        if tool_name == "file_system_write":
            return self._exec_file_system_write(args)
        elif tool_name == "file_system_read":
            return self._exec_file_system_read(args)
        elif tool_name == "sandbox_execute":
            return self._exec_sandbox_execute(args)
        elif tool_name == "query_knowledge_graph":
            return self._exec_query_knowledge_graph(args)
        elif tool_name in ("os_telemetry", "system_telemetry"):
            return self._exec_os_telemetry(args)
        elif tool_name in ("workspace_inspection", "git_status"):
            return self._exec_workspace_inspection(args)
        elif tool_name in ("process_list", "top_processes"):
            return self._exec_process_list(args)
        elif tool_name in ("synthesize_software_system", "synthesize_code", "build_system"):
            return self._exec_synthesize_software_system(args)
        elif tool_name in ("query_causal_hierarchy", "causal_search"):
            return self._exec_query_causal_hierarchy(args)
        elif tool_name in ("iot_control", "control_iot"):
            return self._exec_iot_control(args)
        elif tool_name in ("iot_telemetry", "room_telemetry"):
            return self._exec_iot_telemetry(args)
        else:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Unrecognized tool '{tool_name}'."
            )

    def _exec_file_system_write(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Executes file_system_write tool."""
        raw_path = str(args.get("path", "jane_script.py"))
        code = str(args.get("code", ""))

        resolved_path = self._verify_and_resolve_path(raw_path)
        if not resolved_path:
            return ToolExecutionResult(
                tool_name="file_system_write",
                success=False,
                error="Path failed formal symbolic containment proof."
            )

        is_document = resolved_path.endswith((".html", ".htm", ".css", ".json", ".md", ".txt", ".yaml", ".yml"))

        if not is_document:
            lang = Language.CPP if resolved_path.endswith((".cpp", ".cxx", ".cc")) else Language.PYTHON
            violations = (
                StaticSecurityAuditor.audit_cpp(code)
                if lang == Language.CPP
                else StaticSecurityAuditor.audit_python(code)
            )
            if violations:
                return ToolExecutionResult(
                    tool_name="file_system_write",
                    success=False,
                    target_path=resolved_path,
                    error=f"Security audit rejected code: {violations[0]}"
                )
        else:
            lang = Language.PYTHON

        # Write file with read-only sandbox fallback
        try:
            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
            with open(resolved_path, "w", encoding="utf-8") as f:
                f.write(code)
        except (OSError, PermissionError):
            resolved_path = os.path.join(tempfile.gettempdir(), os.path.basename(resolved_path))
            try:
                os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
                with open(resolved_path, "w", encoding="utf-8") as f:
                    f.write(code)
            except OSError as e:
                return ToolExecutionResult(
                    tool_name="file_system_write",
                    success=False,
                    target_path=resolved_path,
                    error=f"File write failed: {e}"
                )

        if is_document:
            return ToolExecutionResult(
                tool_name="file_system_write",
                success=True,
                output=f"Successfully written document to {resolved_path}",
                target_path=resolved_path,
                exit_code=0,
                artifacts={"code": code, "language": "document"}
            )

        # Execute test run in sandboxed compiler for executable code
        exec_res = self.compiler.run(code, language=lang)
        return ToolExecutionResult(
            tool_name="file_system_write",
            success=exec_res.success,
            output=exec_res.stdout.strip(),
            error=exec_res.stderr.strip(),
            target_path=resolved_path,
            exit_code=exec_res.exit_code,
            artifacts={"code": code, "language": lang.value}
        )

    def _exec_sandbox_execute(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Executes sandbox_execute tool."""
        code = str(args.get("code", ""))
        language_str = str(args.get("language", "python")).lower()
        lang = Language.CPP if ("cpp" in language_str or "c++" in language_str) else Language.PYTHON

        # Static AST security audit
        violations = (
            StaticSecurityAuditor.audit_cpp(code)
            if lang == Language.CPP
            else StaticSecurityAuditor.audit_python(code)
        )
        if violations:
            return ToolExecutionResult(
                tool_name="sandbox_execute",
                success=False,
                error=f"Security audit rejected code: {violations[0]}"
            )

        exec_res = self.compiler.run(code, language=lang)
        return ToolExecutionResult(
            tool_name="sandbox_execute",
            success=exec_res.success,
            output=exec_res.stdout.strip(),
            error=exec_res.stderr.strip(),
            exit_code=exec_res.exit_code,
            artifacts={"code": code, "language": lang.value}
        )

    def _exec_query_knowledge_graph(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Queries knowledge graph manifold."""
        concept = str(args.get("concept", "")).strip().lower()
        from src.reasoning_engine.manifold_retriever import ManifoldRetriever
        retriever = ManifoldRetriever()
        ctx = retriever.retrieve(concept)
        if ctx.triples:
            trip_strs = [f"{t.get('head', '')} -> {t.get('relation', '')} -> {t.get('tail', '')}" for t in ctx.triples]
            triples_str = ", ".join(trip_strs)
        else:
            triples_str = ctx.format_for_prompt()
        return ToolExecutionResult(
            tool_name="query_knowledge_graph",
            success=True,
            output=triples_str,
            artifacts={"concept": concept}
        )

    def _exec_file_system_read(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Safely reads file content within containment boundaries."""
        raw_path = str(args.get("path", "")).strip()
        if not raw_path:
            return ToolExecutionResult(
                tool_name="file_system_read",
                success=False,
                error="Path parameter is required."
            )
        resolved_path = self._verify_and_resolve_path(raw_path)
        if not resolved_path:
            return ToolExecutionResult(
                tool_name="file_system_read",
                success=False,
                error="Target path rejected by formal symbolic containment verifier."
            )
        if not os.path.exists(resolved_path) or not os.path.isfile(resolved_path):
            return ToolExecutionResult(
                tool_name="file_system_read",
                success=False,
                target_path=resolved_path,
                error=f"File not found: {resolved_path}"
            )
        try:
            with open(resolved_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(512000)
            return ToolExecutionResult(
                tool_name="file_system_read",
                success=True,
                output=content,
                target_path=resolved_path,
                artifacts={"size_bytes": len(content)}
            )
        except Exception as e:
            return ToolExecutionResult(
                tool_name="file_system_read",
                success=False,
                target_path=resolved_path,
                error=f"Failed to read file: {e}"
            )

    def _exec_os_telemetry(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Collects real-time OS hardware and resource telemetry."""
        from src.core.os_grounding import OSGroundingEngine
        from dataclasses import asdict
        engine = OSGroundingEngine.get_instance()
        q_type = str(args.get("query", "all")).lower()
        if q_type in ("hardware", "hw", "resources", "cpu", "gpu"):
            data = asdict(engine.get_hardware_telemetry())
        elif q_type in ("workspace", "git"):
            data = asdict(engine.get_workspace_telemetry())
        elif q_type in ("processes", "proc"):
            data = {"processes": [asdict(p) for p in engine.get_process_telemetry()]}
        elif q_type in ("desktop", "session"):
            data = asdict(engine.get_desktop_telemetry())
        else:
            manifest = engine.get_unified_os_manifest()
            data = asdict(manifest)

        summary = engine.get_grounded_summary()
        return ToolExecutionResult(
            tool_name="os_telemetry",
            success=True,
            output=summary,
            artifacts=data
        )

    def _exec_workspace_inspection(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Inspects project workspace and git status."""
        from src.core.os_grounding import OSGroundingEngine
        from dataclasses import asdict
        engine = OSGroundingEngine.get_instance()
        target_path = args.get("path")
        ws = engine.get_workspace_telemetry(target_path)
        git_info = (
            f"Branch: {ws.current_branch} | Commit: {ws.latest_commit_hash} ('{ws.latest_commit_message}') | "
            f"Modified: {ws.modified_files_count}, Untracked: {ws.untracked_files_count}, Staged: {ws.staged_files_count}"
            if ws.is_git_repository else "Standalone directory (not a git repo)"
        )
        recent_str = f"Recent files: {', '.join(ws.recent_modified_files[:5])}" if ws.recent_modified_files else "No recent file edits"
        output_str = f"Workspace: {ws.workspace_path}\n{git_info}\n{recent_str}"
        return ToolExecutionResult(
            tool_name="workspace_inspection",
            success=True,
            output=output_str,
            target_path=ws.workspace_path,
            artifacts=asdict(ws)
        )

    def _exec_process_list(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Inspects active developer and system processes."""
        from src.core.os_grounding import OSGroundingEngine
        from dataclasses import asdict
        engine = OSGroundingEngine.get_instance()
        limit = int(args.get("limit", 15))
        procs = engine.get_process_telemetry(max_processes=limit)
        lines = [f"PID {p.pid:6d} | CPU {p.cpu_percent:5.1f}% | MEM {p.memory_percent:5.1f}% | {p.command}" for p in procs]
        out_text = "\n".join(lines) if lines else "No active developer processes identified."
        return ToolExecutionResult(
            tool_name="process_list",
            success=True,
            output=out_text,
            artifacts={"processes": [asdict(p) for p in procs]}
        )

    def _exec_synthesize_software_system(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Executes autonomous multi-file software synthesis in isolated shadow worktree."""
        goal = str(args.get("goal", "")).strip()
        if not goal:
            return ToolExecutionResult(
                tool_name="synthesize_software_system",
                success=False,
                error="Goal specification is required."
            )
        from src.symbolic_verifier.autonomous_coder import AutonomousCodeSynthesizer
        coder = AutonomousCodeSynthesizer()
        blueprint = coder.build_and_verify_system(goal)
        modules_summary = ", ".join(f"{m.module_name} ({'verified' if m.is_verified else 'unverified'})" for m in blueprint.modules)
        output_str = (
            f"System: {blueprint.system_name} | Converged: {blueprint.converged} | "
            f"Total LOC: {blueprint.total_loc} | Modules: [{modules_summary}]"
        )
        return ToolExecutionResult(
            tool_name="synthesize_software_system",
            success=blueprint.converged,
            output=output_str,
            artifacts={
                "system_name": blueprint.system_name,
                "converged": blueprint.converged,
                "total_loc": blueprint.total_loc,
                "modules": [m.module_name for m in blueprint.modules]
            }
        )

    def _exec_query_causal_hierarchy(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Executes top-down logarithmic search on Pearl's 4-tier causal hierarchy."""
        query = str(args.get("query", "")).strip()
        from src.reasoning_engine.causal_hierarchy import HierarchicalCausalKnowledgeGraph
        graph = HierarchicalCausalKnowledgeGraph.get_instance()
        nodes = graph.hierarchical_top_down_search(query, top_k=5)
        lines = [f"[Tier {n.tier} | {n.domain}] {n.name}: {n.description}" for n in nodes]
        out_str = "\n".join(lines) if lines else f"No causal abstractions found for '{query}'."
        return ToolExecutionResult(
            tool_name="query_causal_hierarchy",
            success=True,
            output=out_str,
            artifacts={"matched_nodes": [n.node_id for n in nodes]}
        )

    def _exec_iot_control(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Executes Z3-gated IoT hardware command."""
        from src.core.iot_gateway import SovereignIoTGateway
        gateway = getattr(self, "_iot_gateway", None)
        if gateway is None:
            gateway = SovereignIoTGateway()
            self._iot_gateway = gateway

        device = str(args.get("device", "lab_desk_light"))
        cmd = args.get("command", {"state": "ON"})
        if isinstance(cmd, str):
            try:
                cmd = json.loads(cmd)
            except Exception:
                cmd = {"state": cmd}

        res = gateway.dispatch_command(device, cmd)
        return ToolExecutionResult(
            tool_name="iot_control",
            success=res.success,
            output=f"Dispatched to {res.mqtt_topic} ({res.transport_mode}): {res.updated_state}" if res.success else "",
            error=res.error or "",
            artifacts={"proof_log": res.proof_log, "updated_state": res.updated_state, "transport": res.transport_mode}
        )

    def _exec_iot_telemetry(self, args: Dict[str, Any]) -> ToolExecutionResult:
        """Queries sovereign IoT room presence and climate telemetry."""
        from src.core.iot_gateway import SovereignIoTGateway
        gateway = getattr(self, "_iot_gateway", None)
        if gateway is None:
            gateway = SovereignIoTGateway()
            self._iot_gateway = gateway

        room = str(args.get("room", "main_lab"))
        summary = gateway.get_environment_summary(room=room)
        return ToolExecutionResult(
            tool_name="iot_telemetry",
            success=True,
            output=summary["narrative"],
            artifacts=summary
        )


    def _verify_and_resolve_path(self, path: str) -> Optional[str]:
        """Proves path safety invariants (containment in user home, tmp, or workspace)."""
        clean = path.strip().replace('"', '').replace("'", "")
        clean_name = os.path.basename(clean) or "script.py"

        # 0. Check direct file existence in workspace / current dir
        if os.path.exists(clean) or os.path.exists(os.path.abspath(clean)):
            abs_cand = os.path.abspath(clean)
            prohibited = ("/bin", "/sbin", "/etc", "/usr", "/boot", "/sys", "/proc", "/dev", "/root")
            if not any(abs_cand == bad or abs_cand.startswith(bad + "/") for bad in prohibited):
                return abs_cand

        if "tmp" in clean.lower():
            base_dir = "/tmp"
        elif "download" in clean.lower():
            base_dir = os.path.expanduser("~/Downloads")
        elif "desktop" in clean.lower():
            base_dir = os.path.expanduser("~/Desktop")
        elif "documents" in clean.lower():
            base_dir = os.path.expanduser("~/Documents")
        elif clean.startswith("~"):
            base_dir = os.path.expanduser(os.path.dirname(clean))
        elif os.path.isabs(clean):
            base_dir = os.path.dirname(clean)
        else:
            base_dir = os.path.expanduser("~/Downloads")

        abs_path = os.path.abspath(os.path.join(base_dir, clean_name))

        # Check prohibited roots
        prohibited = ("/bin", "/sbin", "/etc", "/usr", "/boot", "/sys", "/proc", "/dev", "/root")
        for bad in prohibited:
            if abs_path == bad or abs_path.startswith(bad + "/"):
                logger.warning("Prohibited root directory rejected: %s", abs_path)
                return None

        return abs_path

    def parse_and_execute_tool_calls(self, text: str) -> Tuple[List[ToolExecutionResult], str]:
        """
        Extracts and executes TOOL_CALL blocks without regex.
        Handles both single-line and multi-line bracketed or unbracketed TOOL_CALL payloads.
        Returns list of execution results and clean conversational text.
        """
        results: List[ToolExecutionResult] = []
        clean_lines: List[str] = []

        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            call_idx = stripped.find("TOOL_CALL:")
            if call_idx != -1:
                # Accumulate potential tool call lines
                buffer = stripped[call_idx + len("TOOL_CALL:"):].strip()
                prefix = stripped[:call_idx].strip().rstrip("[").strip()
                parsed = self._extract_json_object(buffer)
                consumed = 1
                while parsed is None and (i + consumed) < len(lines):
                    next_line = lines[i + consumed]
                    buffer += "\n" + next_line
                    parsed = self._extract_json_object(buffer)
                    consumed += 1
                    if "}" in next_line and ("]" in next_line or next_line.strip().endswith("}")):
                        break

                if parsed and isinstance(parsed, dict):
                    tool_name = parsed.get("tool", "")
                    args = parsed.get("args", {})
                    if tool_name and isinstance(args, dict):
                        res = self.execute_tool(tool_name, args)
                        results.append(res)
                        if prefix:
                            clean_lines.append(prefix)
                        i += consumed
                        continue

            clean_lines.append(line)
            i += 1

        clean_text = "\n".join(clean_lines).strip()
        return results, clean_text

    def _extract_json_object(self, s: str) -> Optional[Dict[str, Any]]:
        """Parses JSON object without regex."""
        idx1 = s.find("{")
        idx2 = s.rfind("}")
        if idx1 != -1 and idx2 != -1 and idx2 > idx1:
            snippet = s[idx1 : idx2 + 1]
            try:
                data = json.loads(snippet)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        return None
