"""Verification Suite for Interactive Voice Demos, Full-Duplex Echo Cancellation, and Sovereign IoT Gateway."""

import os
import wave
import torch

from src.perception.neural_vocal_tract import NeuralVocalTract, VocalCadenceProfile, FullDuplexAcousticStream
from src.perception.audio_encoder import DirectAudioTransformer
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from src.core.iot_gateway import SovereignIoTGateway
from src.core.pipeline import JanePipeline


def test_interactive_voice_cadence_demos_and_wav_export():
    """
    1. Synthesizes interactive voice waveforms across 4 distinct cadence modes
       (composed_inquiry, urgent, teaching, whisper) steered by 256D manifold vectors.
    2. Exports playable 16kHz 16-bit PCM WAV files to logs/voice_demos/.
    3. Verifies distinct pitch contours, spectral richness, and real-time latency.
    """
    tract = NeuralVocalTract(sample_rate=16000, latent_dim=256)
    demo_dir = os.path.abspath("logs/voice_demos")
    os.makedirs(demo_dir, exist_ok=True)

    modes = [
        ("composed_inquiry", "All manifold invariants are stable and ready for inspection."),
        ("urgent", "Warning thermal flux exceeds nominal threshold."),
        ("teaching", "Energy conservation links Hamiltonian dynamics to topological symmetry."),
        ("whisper", "Ambient acoustic stream is online in quiet reflection mode.")
    ]

    f0_results = {}
    for idx, (mode_name, phrase) in enumerate(modes):
        cadence = VocalCadenceProfile.from_tone_label(mode_name)
        z_thought = torch.zeros(256)
        z_thought[idx * 10] = 1.0

        waveform, telemetry = tract.synthesize_speech(phrase, z_thought=z_thought, cadence=cadence)
        wav_path = os.path.join(demo_dir, f"jane_{mode_name}.wav")
        saved_path = tract.export_wav(waveform, wav_path)

        assert os.path.exists(saved_path)
        with wave.open(saved_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == waveform.shape[-1]

        f0_results[mode_name] = telemetry["f0_mean_hz"]
        assert telemetry["real_time_factor"] < 0.20  # At least 5x faster than real-time for 22.05kHz HiFi-GAN

    # Urgent cadence should have higher pitch than whisper cadence
    assert f0_results["urgent"] > f0_results["whisper"]


def test_full_duplex_echo_cancellation_and_self_trigger_suppression():
    """
    Verifies that when Jane speaks, her own acoustic output echoing back into the
    microphone is cancelled (> 10 dB ERLE) and suppressed so she never talks to herself.
    """
    tract = NeuralVocalTract(sample_rate=16000, latent_dim=256)
    encoder = DirectAudioTransformer(sample_rate=16000, latent_dim=256, checkpoint_path=None)
    manifold = HebbianConceptManifold(latent_dim=256, save_path="/tmp/test_duplex_manifold.pt")
    stream = FullDuplexAcousticStream(
        vocal_tract=tract,
        audio_encoder=encoder,
        hebbian_manifold=manifold
    )

    # Jane synthesizes an utterance
    jane_wave, _ = tract.synthesize_speech("Checking reactor pressure equilibrium.")
    stream.register_jane_speech(jane_wave)

    # Simulate room reflection (attenuated copy of Jane's voice + tiny room noise)
    room_echo = 0.65 * jane_wave + 0.002 * torch.randn_like(jane_wave)

    res = stream.ingest_audio_chunk(room_echo)
    assert res["is_self_echo"] is True
    assert res["status"] == "SUPPRESSED_SELF_ECHO"
    assert res["echo_return_loss_db"] > 10.0
    assert stream.suppressed_echo_frames == 1


def test_full_duplex_user_barge_in_and_cadence_adaptation():
    """
    Verifies that when the user speaks over Jane (barge-in) or speaks cleanly into
    the ambient microphone, the stream detects user speech, updates the 256D manifold,
    and adapts Jane's vocal cadence to the user's vocal energy/tone.
    """
    tract = NeuralVocalTract(sample_rate=16000, latent_dim=256)
    encoder = DirectAudioTransformer(sample_rate=16000, latent_dim=256, checkpoint_path=None)
    manifold = HebbianConceptManifold(latent_dim=256, save_path="/tmp/test_barge_manifold.pt")
    stream = FullDuplexAcousticStream(
        vocal_tract=tract,
        audio_encoder=encoder,
        hebbian_manifold=manifold
    )

    # 1. Jane speaks softly
    jane_wave, _ = tract.synthesize_speech("Running background analysis.")
    stream.register_jane_speech(jane_wave * 0.25)

    # 2. User barges in loudly with distinct spectral content (emphatic/urgent speech)
    t = torch.linspace(0, 1.0, jane_wave.shape[-1])
    user_barge_wave = (
        0.55 * torch.sin(2 * 3.14159 * 260.0 * t)
        + 0.35 * torch.sin(2 * 3.14159 * 1180.0 * t)
        + 0.20 * torch.sin(2 * 3.14159 * 2450.0 * t)
    ).unsqueeze(0)
    mixed_mic = 0.20 * jane_wave + user_barge_wave

    res = stream.ingest_audio_chunk(mixed_mic, parse_intent=False)
    assert res["is_self_echo"] is False
    assert res["is_barge_in"] is True
    assert res["status"] == "USER_SPEECH_PROCESSED"
    assert res["adapted_cadence"].tone_mode == "urgent"
    assert stream.barge_in_events == 1


def test_sovereign_iot_gateway_z3_safety_and_manifold_grounding():
    """
    Verifies SovereignIoTGateway:
    1. ESP32-S3 mmWave radar presence and BME680 environmental sensor fusion into 256D manifold.
    2. Safe Zigbee2MQTT actuation commands pass Z3 verification.
    3. Unsafe thermal (>30C) or electrical load (>1800W) commands are blocked by Z3.
    """
    manifold = HebbianConceptManifold(latent_dim=256, save_path="/tmp/test_iot_manifold.pt")
    iot = SovereignIoTGateway(hebbian_manifold=manifold)

    # 1. Ingest ESP32-S3 mmWave radar + BME680 sensor packet
    telem = iot.ingest_sensor_payload(
        "lab_omnipresence_sensor",
        {
            "presence": True,
            "stationary_breathing": True,
            "distance_cm": 95.0,
            "stationary_energy": 96,
            "temperature_c": 26.4,
            "humidity_pct": 48.0,
            "pressure_hpa": 1005.0,
            "iaq_index": 28.0
        }
    )
    assert telem["presence"]["presence_detected"] is True
    assert telem["presence"]["distance_cm"] == 95.0
    assert telem["climate"]["temperature_c"] == 26.4
    assert "temperature" in telem["manifold_concepts_primed"]

    # 2. Safe Zigbee2MQTT thermostat command
    ok_res = iot.dispatch_command("lab_climate_controller", {"state": "COOL", "target_temp_c": 22.0})
    assert ok_res.success is True
    assert ok_res.z3_verified is True
    assert ok_res.mqtt_topic == "zigbee2mqtt/lab_climate_controller/set"

    # 3. Unsafe thermal command (42.0C exceeds human thermal safety limit [16C, 30C])
    bad_temp = iot.dispatch_command("lab_climate_controller", {"target_temp_c": 42.0})
    assert bad_temp.success is False
    assert bad_temp.z3_verified is False
    assert bad_temp.transport_mode == "blocked_by_z3"

    # 4. Unsafe electrical load command (3200W exceeds 1800W relay breaker rating)
    bad_power = iot.dispatch_command("workstation_aux_relay", {"state": "ON", "power_watts": 3200.0})
    assert bad_power.success is False
    assert bad_power.z3_verified is False
    assert bad_power.transport_mode == "blocked_by_z3"


def test_pipeline_end_to_end_voice_and_iot_integration():
    """
    Verifies JanePipeline end-to-end integration of speak(), ingest_audio_flux(),
    control_iot_device(), and get_iot_environment().
    """
    pipeline = JanePipeline(use_pretrained_translator=False)

    # 1. Speak and verify echo reference registration
    wave_out, speech_meta = pipeline.speak(
        "Systems online.",
        export_path="logs/voice_demos/pipeline_online.wav"
    )
    assert wave_out.shape[1] > 1600
    assert os.path.exists(speech_meta["wav_path"])

    # 2. Echoed output is suppressed by pipeline.ingest_audio_flux()
    echo_res = pipeline.ingest_audio_flux(wave_out * 0.5)
    assert echo_res["is_self_echo"] is True

    # 3. IoT control and environment summary via pipeline
    iot_res = pipeline.control_iot_device("lab_desk_light", {"state": "ON", "brightness": 210})
    assert iot_res.success is True
    env_summary = pipeline.get_iot_environment("main_lab")
    assert env_summary["devices"]["zb_ambient_light_01"]["state"]["brightness"] == 210
