"""Real-World End-to-End Lifecycle Crucible for Project Jane.

Instead of isolated synthetic unit tests, this crucible executes Jane as a single
unified organism across a complete, unbroken Waking -> Spatial 3D -> Engineering Campaign ->
Full-Duplex Voice/IoT -> Nighttime NREM/REM Sleep Consolidation -> Cold Reboot ->
Morning Wake Resumption lifecycle, and renders the live state to the Holographic UI Page.
"""

import os
import math
import torch

from src.core.pipeline import JanePipeline
from src.reasoning_engine.sleep_consolidator import PersistentCampaignLedger


def test_real_world_continuous_lifecycle_crucible():
    """
    Executes a single continuous real-world operational cycle without mocks or hardcoding:
    1. Perceives 3D volumetric point clouds via SVD and wires them into the 256D manifold.
    2. Resolves a 3D human pointing gesture ray, plans a 3D collision-free bypass trajectory
       around a 3D obstacle, and applies an SE(3) Rodrigues rotation + translation.
    3. Launches a 3-stage engineering campaign DAG, completes Stage 1 via AutonomousCodeSynthesizer,
       processes full-duplex audio + IoT telemetry, and accumulates daytime synaptic noise.
    4. Runs an Autonomic NREM + REM Sleep Cycle: hibernates the active campaign to disk,
       prunes weak synapses & transient orphan nodes, compresses episodic logs, and promotes
       REM Z3 dream laws into the Causal Hierarchy.
    5. Simulates a cold machine reboot by instantiating a fresh pipeline / ledger from disk,
       resuming the interrupted campaign from Stage 2, and exporting the 3D Holographic UI page.
    """
    campaign_file = "/tmp/jane_crucible_campaigns.json"
    archive_file = "/tmp/jane_crucible_archive.jsonl"
    if os.path.exists(campaign_file):
        os.remove(campaign_file)
    if os.path.exists(archive_file):
        os.remove(archive_file)

    # =========================================================================
    # PHASE 1: Waking Boot & 3D Volumetric Point-Cloud Scene Perception
    # =========================================================================
    jane = JanePipeline(use_pretrained_translator=False)
    jane.campaign_ledger = PersistentCampaignLedger(storage_path=campaign_file)
    jane.sleep_consolidator.campaign_ledger = jane.campaign_ledger
    jane.sleep_consolidator.archive_path = archive_file

    # Generate 3D point cloud 1: Prototype Turbine Assembly centered at (0.0, 0.0, 1.0)
    theta = torch.linspace(0, 2 * math.pi, 60)
    z_cyl = torch.linspace(0.6, 1.4, 60)
    turbine_pts = torch.stack([
        0.35 * torch.cos(theta),
        0.35 * torch.sin(theta),
        z_cyl
    ], dim=-1)
    turbine_obj, z_turbine = jane.perceive_3d_point_cloud(
        object_id="turbine_assembly",
        label="Prototype Turbine Core",
        points=turbine_pts,
        is_obstacle=False,
        color_hex="#00f0ff"
    )
    assert turbine_obj.volume > 0.0
    assert z_turbine.shape == (1, 256)

    # Generate 3D point cloud 2: Central Containment Pillar Obstacle at (0.0, 1.5, 0.0 -> 1.8)
    obs_corners = torch.tensor([
        [-0.5, 1.1, 0.0], [0.5, 1.1, 0.0], [0.5, 1.9, 0.0], [-0.5, 1.9, 0.0],
        [-0.5, 1.1, 1.8], [0.5, 1.1, 1.8], [0.5, 1.9, 1.8], [-0.5, 1.9, 1.8],
    ], dtype=torch.float32)
    pillar_obj, _ = jane.perceive_3d_point_cloud(
        object_id="containment_pillar",
        label="Central Containment Pillar",
        points=obs_corners,
        is_obstacle=True,
        color_hex="#ff4d6d"
    )
    assert pillar_obj.is_obstacle is True

    # =========================================================================
    # PHASE 2: 3D Deictic Hand Pointing Raycast, SE(3) Transform & 3D Bypass
    # =========================================================================
    # User stands at (0.0, -2.5, 1.0) and points hand through (0.0, -1.5, 1.0) -> hits turbine_assembly at (0, 0, 1)
    # Requests moving it to (0.0, 3.0, 1.0) (which must bypass containment_pillar in 3D!) and rotating 45 deg
    manip_res = jane.point_and_manipulate_3d(
        joint_origin=(0.0, -2.5, 1.0),
        fingertip_target=(0.0, -1.5, 1.0),
        translation=(0.0, 3.0, 0.0),
        rotation_axis=(0.0, 0.0, 1.0),
        rotation_angle_rad=math.pi / 4.0,
        scale=1.0
    )
    assert manip_res["hit"] is True
    assert manip_res["object_id"] == "turbine_assembly"
    assert manip_res["trajectory_verified"] is True
    # Because the pillar blocks the direct line Y: 0 -> 3, the 3D planner must synthesize >= 3 waypoints
    assert len(manip_res["planned_trajectory_3d"]) >= 3
    assert abs(manip_res["updated_centroid"][1] - 3.0) < 0.05

    # =========================================================================
    # PHASE 3: Daytime Engineering Campaign + Voice/IoT + Synaptic Clutter
    # =========================================================================
    campaign = jane.campaign_ledger.create_campaign(
        campaign_id="camp_telemetry_engine",
        goal="Build a distributed telemetry ring buffer with deterministic bounds checking",
        steps=[
            {
                "step_id": "step_1_core_buffer",
                "title": "Synthesize & Verify Ring Buffer Core",
                "action_type": "SYNTHESIZE_MODULE",
                "depends_on": []
            },
            {
                "step_id": "step_2_stream_router",
                "title": "Synthesize & Verify Telemetry Router",
                "action_type": "SYNTHESIZE_MODULE",
                "depends_on": ["step_1_core_buffer"]
            },
            {
                "step_id": "step_3_integration_seal",
                "title": "Verify End-to-End Invariant Seal",
                "action_type": "VERIFY_SYSTEM",
                "depends_on": ["step_2_stream_router"]
            }
        ]
    )

    # Execute Step 1 in real life using AutonomousCodeSynthesizer
    ready_step = jane.campaign_ledger.get_next_ready_step("camp_telemetry_engine")
    assert ready_step is not None and ready_step.step_id == "step_1_core_buffer"
    bp = jane.synthesize_system(campaign.goal)
    assert bp.converged is True
    jane.campaign_ledger.mark_step_completed(
        "camp_telemetry_engine",
        "step_1_core_buffer",
        artifact={"system_name": bp.system_name, "total_loc": bp.total_loc},
        verification_energy=0.0
    )

    # Full-duplex voice output + echo suppression + IoT sensor ingestion during the day
    speech_wave, _ = jane.speak("Stage one of the telemetry engine is verified. Preparing for evening consolidation.")
    echo_check = jane.ingest_audio_flux(speech_wave * 0.45)
    assert echo_check["is_self_echo"] is True

    jane.ingest_iot_telemetry(
        "lab_omnipresence_sensor",
        {"presence": True, "stationary_breathing": True, "distance_cm": 140.0, "temperature_c": 21.5}
    )

    # Simulate daytime transient orphan concepts & unreinforced weak synapses
    for i in range(4):
        noise_vec = torch.randn(256)
        idx = jane.hebbian_manifold.add_node(f"transient_chatter_{i}", noise_vec)
        # Zero out its off-diagonal links so it is a true unreinforced transient orphan
        jane.hebbian_manifold.adjacency.data[idx, :] = 0.0
        jane.hebbian_manifold.adjacency.data[:, idx] = 0.0
        jane.hebbian_manifold.adjacency.data[idx, idx] = 0.55

    for i in range(12):
        jane.episodic_ledger.record_event(
            action_type="DAYTIME_OBSERVATION",
            concept_name="thermodynamics" if i % 2 == 0 else "equilibrium",
            source_name="LiveCrucible",
            formula_or_axiom="energy_out <= energy_in",
            status="COMMITTED"
        )

    pre_nodes = jane.hebbian_manifold.num_concepts
    pre_causal_laws = len(jane.causal_graph.tier_index.get(2, []))

    # =========================================================================
    # PHASE 4: Nighttime Autonomic Sleep / Dream Cycle (NREM + REM)
    # =========================================================================
    sleep_report = jane.run_autonomic_sleep_cycle(
        downscale_gamma=0.85,
        prune_threshold=0.06,
        max_dreams=3
    )

    assert sleep_report.synapses_pruned > 0
    assert sleep_report.orphan_nodes_pruned >= 4
    assert jane.hebbian_manifold.num_concepts <= pre_nodes - 4
    assert sleep_report.episodic_events_compressed > 0
    assert len(sleep_report.rem_axioms_promoted) == 3
    assert len(jane.causal_graph.tier_index.get(2, [])) > pre_causal_laws
    assert "camp_telemetry_engine" in sleep_report.hibernated_campaigns

    # =========================================================================
    # PHASE 5: Cold Reboot & Morning Wake Campaign Resumption + Holographic UI
    # =========================================================================
    rebooted_ledger = PersistentCampaignLedger(storage_path=campaign_file)
    resumed_list = rebooted_ledger.wake_and_resume_campaigns()
    assert len(resumed_list) == 1
    resumed_info = resumed_list[0]
    assert resumed_info["campaign_id"] == "camp_telemetry_engine"
    assert resumed_info["completed_steps"] == 1
    assert resumed_info["next_ready_step"] == "step_2_stream_router"

    # Complete remaining campaign steps after wake
    rebooted_ledger.mark_step_completed("camp_telemetry_engine", "step_2_stream_router", verification_energy=0.0)
    next_final = rebooted_ledger.get_next_ready_step("camp_telemetry_engine")
    assert next_final is not None and next_final.step_id == "step_3_integration_seal"
    rebooted_ledger.mark_step_completed("camp_telemetry_engine", "step_3_integration_seal", verification_energy=0.0)
    assert rebooted_ledger.campaigns["camp_telemetry_engine"].status == "COMPLETED"

    # Export interactive 3D Holographic Crucible Page with full real-world telemetry
    html_path = jane.spatial_3d_engine.export_holographic_crucible_page(
        output_html_path="logs/holographic_ui/jane_holographic_crucible.html",
        lifecycle_telemetry={
            "pre_sleep_entropy": sleep_report.pre_sleep_synaptic_entropy,
            "post_sleep_entropy": sleep_report.post_sleep_synaptic_entropy,
            "synapses_pruned": sleep_report.synapses_pruned,
            "orphan_nodes_pruned": sleep_report.orphan_nodes_pruned,
            "episodic_events_compressed": sleep_report.episodic_events_compressed,
            "rem_laws_promoted": len(sleep_report.rem_axioms_promoted),
            "campaign_resumed_step": resumed_info["next_ready_step"],
            "campaign_final_status": rebooted_ledger.campaigns["camp_telemetry_engine"].status
        }
    )
    assert os.path.exists(html_path)


def test_24x7_daemon_boot_full_duplex_and_sigterm_reboot_resumption():
    """
    Tests the 24/7 SovereignJaneDaemon & ModeRouter lifecycle across laptop power-off and reboot:
    1. Boot 1: Starts daemon, launches a multi-day campaign, completes Step 1, speaks via
       ModeRouter, suppresses room self-echo via FullDuplexAcousticStream, and processes
       real user speech into the 256D manifold.
    2. Laptop Shutdown (SIGTERM): Daemon intercepts shutdown, hibernates active campaigns,
       and persists the manifold and status telemetry to disk.
    3. Boot 2 (Laptop Turned Back On): Fresh daemon instance boots, automatically wakes
       and resumes the hibernated campaign from Step 2.
    """
    from scripts.jane_24x7_daemon import SovereignJaneDaemon

    campaign_file = "/tmp/jane_daemon_boot_campaigns.json"
    status_file = "logs/test_daemon_status.json"
    if os.path.exists(campaign_file):
        os.remove(campaign_file)

    # --- BOOT 1: Laptop is on, Jane daemon is running ---
    daemon_boot1 = SovereignJaneDaemon(
        use_pretrained_translator=False,
        idle_sleep_threshold_sec=5.0,
        curiosity_interval_sec=0.0,
        status_file=status_file,
    )
    daemon_boot1.pipeline.campaign_ledger = PersistentCampaignLedger(storage_path=campaign_file)
    daemon_boot1.pipeline.sleep_consolidator.campaign_ledger = daemon_boot1.pipeline.campaign_ledger

    boot1_status = daemon_boot1.start(start_hardware_ear=False, start_curiosity=False)
    assert boot1_status["state"] == "RUNNING_24X7"

    # Start an engineering campaign during Boot 1 and finish Step 1
    daemon_boot1.pipeline.campaign_ledger.create_campaign(
        campaign_id="camp_24x7_autonomy",
        goal="Continuous 24/7 autonomous homeostasis verification",
        steps=[
            {"step_id": "s1_init", "title": "Initialize Manifold", "action_type": "INIT", "depends_on": []},
            {"step_id": "s2_verify", "title": "Verify Invariants After Reboot", "action_type": "VERIFY", "depends_on": ["s1_init"]},
        ],
    )
    daemon_boot1.pipeline.campaign_ledger.mark_step_completed("camp_24x7_autonomy", "s1_init", verification_energy=0.0)

    # Jane speaks through ModeRouter; verify room echo is cancelled in ModeRouter.process_audio_chunk_once
    speak_telemetry = daemon_boot1.router.speak_and_play(
        "Hey Akhil, I'm tracking the manifold.",
        play_hardware_audio=False,
    )
    spoken_wave = speak_telemetry["waveform"]
    assert spoken_wave is not None and spoken_wave.numel() > 1600

    echo_telemetry = daemon_boot1.router.process_audio_chunk_once(spoken_wave * 0.4)
    assert echo_telemetry["is_self_echo"] is True
    assert daemon_boot1.router.suppressed_echo_count >= 1

    # Clear echo cooldown and simulate user speaking a real English question followed by a pause -> verify Jane decodes intent and replies via pipeline.process()
    daemon_boot1.router._speaking_cooldown_until = 0.0
    daemon_boot1.router.full_duplex_stream.echo_canceller.clear_reference()
    user_voice_chunk, _ = daemon_boot1.pipeline.vocal_tract.synthesize_speech(
        "What is the derivative of kinetic energy with respect to velocity?"
    )
    daemon_boot1.router.process_audio_chunk_once(user_voice_chunk, auto_reply_on_pause=False)
    pause_chunk = torch.zeros(1, 24000)
    turn_res = daemon_boot1.router.process_audio_chunk_once(pause_chunk, auto_reply_on_pause=False)
    assert turn_res["status"] == "TURN_COMPLETED_AND_REPLIED"
    assert any(k in turn_res["spoken_query"].lower() for k in ("derivative", "kinetic", "energy", "velocity"))
    assert len(turn_res["spoken_reply"]) > 5
    assert "<unk>" not in turn_res["spoken_reply"].lower()
    assert "with you, akhil—connecting" not in turn_res["spoken_reply"].lower()
    assert "online and listening" not in turn_res["spoken_reply"].lower()

    # Simulate laptop shutdown (SIGTERM) -> clean hibernation
    shutdown_status = daemon_boot1.shutdown()
    assert shutdown_status["state"] == "HIBERNATED"
    assert "camp_24x7_autonomy" in shutdown_status["hibernation_report"]["hibernated_campaigns"]

    # --- BOOT 2: Laptop turns back on, systemd starts fresh daemon ---
    daemon_boot2 = SovereignJaneDaemon(
        use_pretrained_translator=False,
        idle_sleep_threshold_sec=5.0,
        curiosity_interval_sec=0.0,
        status_file=status_file,
    )
    daemon_boot2.pipeline.campaign_ledger = PersistentCampaignLedger(storage_path=campaign_file)
    daemon_boot2.pipeline.sleep_consolidator.campaign_ledger = daemon_boot2.pipeline.campaign_ledger

    boot2_status = daemon_boot2.start(start_hardware_ear=False, start_curiosity=False)
    assert boot2_status["resumed_campaigns_on_boot"] == 1
    assert daemon_boot2.resumed_campaigns[0]["campaign_id"] == "camp_24x7_autonomy"
    assert daemon_boot2.resumed_campaigns[0]["next_ready_step"] == "s2_verify"
    wakeup_thought = daemon_boot2.pipeline.generate_natural_wakeup_thought(daemon_boot2.resumed_campaigns)
    assert "s2 verify" in wakeup_thought.lower()
    assert "online and listening" not in wakeup_thought.lower()
    daemon_boot2.shutdown()

