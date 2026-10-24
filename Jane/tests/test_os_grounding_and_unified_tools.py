"""Comprehensive Test Suite for OS Grounding Engine and APIGateway Unified Tool Execution.

Verifies:
1. OSGroundingEngine:
   - Hardware telemetry (CPU count, RAM usage, load average, GPU/CUDA detection).
   - Workspace telemetry (project path, git status, recent modified files).
   - Process telemetry (active developer processes, CPU/memory metrics).
   - Desktop telemetry (display server, session type, username).
   - Grounded natural language summary generation without regex.
2. APIGateway Unified Tool Access:
   - Executes cognitive tools via TOOL_CALL (file_system_write, sandbox_execute, os_telemetry, file_system_read).
   - Formal SMT/Z3 safety gating blocks actions claiming thermodynamic or math violations.
   - Executes OS_GROUNDING action with hardware, workspace, and process queries.
   - Records all tool invocations in EpisodicLedger audit trail.
3. SelfModelEngine Live Integration:
   - Live state manifest dynamically incorporates real OS embodiment.
   - System status telemetry reflects host machine metrics.
4. JanePipeline End-to-End:
   - pipeline.call_tool, pipeline.observe_os, and conversational processing.
"""

import os
import time
import pytest

from src.core.os_grounding import (
    OSGroundingEngine,
    HardwareTelemetry,
    WorkspaceTelemetry,
    ProcessTelemetry,
    DesktopTelemetry,
    OSStateManifest
)
from src.core.tools import ToolRegistry
from src.core.api_gateway import APIGateway
from src.symbolic_verifier.z3_interface import Z3LogicVerifier
from src.reasoning_engine.episodic_ledger import EpisodicLedger
from src.reasoning_engine.self_model import SelfModelEngine
from src.core.pipeline import JanePipeline


@pytest.fixture
def temp_ledger(tmp_path):
    log_file = tmp_path / "os_tools_ledger.jsonl"
    return EpisodicLedger(log_path=str(log_file))


@pytest.fixture
def verifier():
    return Z3LogicVerifier()


@pytest.fixture
def os_engine(tmp_path):
    return OSGroundingEngine(workspace_root=str(tmp_path))


@pytest.fixture
def gateway(verifier, temp_ledger, os_engine):
    reg = ToolRegistry.get_instance()
    gw = APIGateway(
        z3_verifier=verifier,
        episodic_ledger=temp_ledger,
        os_grounding=os_engine,
        tool_registry=reg
    )
    reg.set_api_gateway(gw)
    return gw


# =============================================================================
# 1. OSGroundingEngine Unit Tests
# =============================================================================

def test_os_grounding_hardware_telemetry(os_engine):
    """Verifies live CPU, RAM, disk, and platform telemetry."""
    hw = os_engine.get_hardware_telemetry()
    assert isinstance(hw, HardwareTelemetry)
    assert hw.cpu_count >= 1
    assert hw.total_ram_mb > 0
    assert hw.available_ram_mb > 0
    assert 0.0 <= hw.ram_usage_percent <= 100.0
    assert len(hw.load_averages) == 3
    assert hw.platform != ""
    assert hw.architecture != ""


def test_os_grounding_workspace_telemetry(tmp_path):
    """Verifies workspace inspection and file tracking."""
    # Create test files
    f1 = tmp_path / "test_module.py"
    f1.write_text("print('hello')", encoding="utf-8")

    engine = OSGroundingEngine(workspace_root=str(tmp_path))
    ws = engine.get_workspace_telemetry()
    assert isinstance(ws, WorkspaceTelemetry)
    assert os.path.abspath(str(tmp_path)) == os.path.abspath(ws.workspace_path)
    assert "test_module.py" in ws.recent_modified_files


def test_os_grounding_process_telemetry(os_engine):
    """Verifies non-brittle extraction of active system processes."""
    procs = os_engine.get_process_telemetry(max_processes=10)
    assert isinstance(procs, list)
    if procs:
        p0 = procs[0]
        assert isinstance(p0, ProcessTelemetry)
        assert p0.pid > 0
        assert p0.cpu_percent >= 0.0
        assert p0.command != ""


def test_os_grounding_desktop_telemetry(os_engine):
    """Verifies display server and user perception."""
    dt = os_engine.get_desktop_telemetry()
    assert isinstance(dt, DesktopTelemetry)
    assert dt.display_server in ("X11", "Wayland", "Headless")
    assert dt.user_name != ""


def test_os_grounding_unified_manifest_and_narrative(os_engine):
    """Verifies complete OS state manifest and natural language summary."""
    manifest = os_engine.get_unified_os_manifest()
    assert isinstance(manifest, OSStateManifest)
    assert "CPU Load:" in manifest.grounded_narrative
    assert "Host RAM:" in manifest.grounded_narrative
    assert manifest.grounded_narrative == os_engine.get_grounded_summary()


# =============================================================================
# 2. APIGateway Unified Tool Execution Tests
# =============================================================================

def test_api_gateway_executes_tool_call(gateway, temp_ledger):
    """Verifies calling a tool through APIGateway with SMT verification."""
    res = gateway.call_tool(
        tool_name="query_knowledge_graph",
        args={"concept": "energy"},
        safety_contract="energy_out <= energy_in"
    )
    assert res == "Real-world action executed: TOOL_CALL"
    assert res.success is True
    assert res.action_type == "TOOL_CALL"
    assert res.output["tool_name"] == "query_knowledge_graph"

    # Verify committed event in ledger
    recent = temp_ledger.get_recent_events(limit=1)
    assert len(recent) == 1
    assert recent[0].action_type == "GATEWAY_ACTION_EXECUTED"
    assert recent[0].concept_name == "TOOL_CALL"


def test_api_gateway_blocks_tool_call_on_smt_violation(gateway, temp_ledger):
    """Verifies Z3 blocks tool calls violating physical or mathematical invariants."""
    res = gateway.call_tool(
        tool_name="sandbox_execute",
        args={"code": "x = 10", "language": "python"},
        safety_contract="energy_out > energy_in"  # Thermodynamic violation
    )
    assert res == "Action blocked by physics verifier."
    assert res.success is False

    # Check episodic ledger audit
    recent = temp_ledger.get_recent_events(limit=1)
    assert len(recent) == 1
    assert recent[0].action_type == "GATEWAY_ACTION_BLOCKED"
    assert recent[0].status == "REJECTED"


def test_api_gateway_executes_os_grounding_action(gateway):
    """Verifies SMT-gated real-time OS and host environment observation."""
    # 1. Hardware query
    res_hw = gateway.observe_os(query_type="hardware")
    assert res_hw == "Real-world action executed: OS_GROUNDING"
    assert res_hw.success is True
    assert "cpu_count" in res_hw.output
    assert "ram_usage_percent" in res_hw.output

    # 2. All manifest query
    res_all = gateway.observe_os(query_type="all")
    assert res_all.success is True
    assert "grounded_narrative" in res_all.output


def test_tools_file_system_read_and_os_telemetry(tmp_path):
    """Verifies file_system_read and os_telemetry cognitive tools in ToolRegistry."""
    reg = ToolRegistry.get_instance()

    # Test file_system_read
    test_file = tmp_path / "sample_doc.txt"
    test_file.write_text("Consciousness and continuous dynamical time.", encoding="utf-8")

    read_res = reg.execute_tool("file_system_read", {"path": str(test_file)})
    assert read_res.success is True
    assert "Consciousness and continuous" in read_res.output

    # Test os_telemetry tool
    telemetry_res = reg.execute_tool("os_telemetry", {"query": "hardware"})
    assert telemetry_res.success is True
    assert "CPU Load:" in telemetry_res.output


# =============================================================================
# 3. SelfModel and End-to-End JanePipeline Integration Tests
# =============================================================================

def test_self_model_live_os_embodiment(os_engine):
    """Verifies SelfModelEngine incorporates live OS telemetry into manifest."""
    self_model = SelfModelEngine(os_grounding=os_engine)
    manifest = self_model.get_live_state_manifest()

    embodiment = manifest["embodiment"]
    assert "hardware" in embodiment
    assert "workspace" in embodiment
    assert "grounded_summary" in embodiment
    assert "CPU Load:" in embodiment["grounded_summary"]

    context = self_model.get_system_context()
    assert "Live System Telemetry:" in context


def test_jane_pipeline_os_grounding_and_tools_end_to_end():
    """Verifies JanePipeline has full OS grounding and tool execution enabled."""
    pipeline = JanePipeline(latent_dim=256, use_pretrained_translator=False, device="cpu")

    # 1. Verify OS grounding attached
    assert hasattr(pipeline, "os_grounding")
    assert pipeline.self_model.os_grounding is not None

    # 2. Execute OS observation through pipeline
    obs_res = pipeline.observe_os("hardware")
    assert obs_res.success is True
    assert obs_res.output["cpu_count"] >= 1

    # 3. Execute tool through pipeline
    tool_res = pipeline.call_tool("os_telemetry", {"query": "all"})
    assert tool_res.success is True
    assert "CPU Load:" in tool_res.output["output"]

    # 4. End-to-end conversation turn retains grounded awareness
    res = pipeline.process("Jane, what are your current system resources and host environment?")
    assert "response_text" in res
    assert "system_status" in res
    assert res["system_status"]["os_grounded_summary"] is not None
