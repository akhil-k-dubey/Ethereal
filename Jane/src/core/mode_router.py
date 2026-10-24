"""Mode Router: Orchestrating Active Deliberation, Passive Eavesdropping, and Hybrid Reality.

Coordinates Jane's cognitive states:
- ACTIVE: Interactive text/voice conversation, System 2 equilibrium deliberation, Z3 formal proofs.
- PASSIVE: Background continuous eavesdropping, Hebbian manifold wiring, user vocal tone learning.
- HYBRID: Terminal chat in foreground + background passive ear continuously learning.
"""

import time
import threading
from enum import Enum
from typing import Optional, Dict, Any, Callable
import torch

from src.perception.audio_encoder import DirectAudioTransformer
from src.reasoning_engine.hebbian_manifold import HebbianConceptManifold
from scripts.live_mic_driver import LiveMicDriver
from src.utils.logger import get_logger

logger = get_logger("mode_router")


class OperatingMode(Enum):
    ACTIVE = "ACTIVE"
    PASSIVE = "PASSIVE"
    HYBRID = "HYBRID"


class ModeRouter:
    """
    Mode Router & 24/7 Background Cognitive Manager.
    Manages full-duplex background audio eavesdropping (with self-echo cancellation),
    continuous Hebbian learning, NeuralVocalTract speaker output, autonomic NREM/REM
    sleep consolidation during idle periods, and boot/shutdown campaign persistence.
    """

    def __init__(
        self,
        mode: OperatingMode = OperatingMode.HYBRID,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        audio_threshold_rms: float = 0.0280,
        autosave_interval_sec: float = 45.0,
        sleep_idle_threshold_sec: float = 900.0,
        on_flux_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        pipeline: Optional[Any] = None
    ):
        self.mode = mode
        self.device = torch.device(device)
        self.audio_threshold_rms = audio_threshold_rms
        self.autosave_interval_sec = autosave_interval_sec
        self.sleep_idle_threshold_sec = sleep_idle_threshold_sec
        self.on_flux_callback = on_flux_callback
        self.pipeline = pipeline

        # Share perception, manifold, vocal tract, and sleep consolidator with pipeline if attached
        if pipeline is not None:
            self.hebbian_manifold = pipeline.hebbian_manifold
            self.audio_encoder = pipeline.audio_encoder
            self.vocal_tract = pipeline.vocal_tract
            self.full_duplex_stream = pipeline.full_duplex_stream
            self.sleep_consolidator = pipeline.sleep_consolidator
            self.campaign_ledger = pipeline.campaign_ledger
        else:
            from src.perception.neural_vocal_tract import NeuralVocalTract, FullDuplexAcousticStream
            from src.reasoning_engine.sleep_consolidator import (
                AutonomicSleepConsolidator,
                PersistentCampaignLedger,
            )
            from src.reasoning_engine.episodic_ledger import EpisodicLedger
            from src.reasoning_engine.causal_hierarchy import HierarchicalCausalKnowledgeGraph
            from src.symbolic_verifier.z3_interface import Z3LogicVerifier

            self.audio_encoder = DirectAudioTransformer(
                checkpoint_path="logs/training_runs/phase4_audio_manifold.pt",
                calibrated_anchors_path="logs/training_runs/my_voice_anchors.pt"
            ).to(self.device)
            self.audio_encoder.eval()
            self.hebbian_manifold = HebbianConceptManifold(latent_dim=256)
            self.vocal_tract = NeuralVocalTract(sample_rate=16000, latent_dim=256)
            self.full_duplex_stream = FullDuplexAcousticStream(
                vocal_tract=self.vocal_tract,
                audio_encoder=self.audio_encoder,
                hebbian_manifold=self.hebbian_manifold,
                sample_rate=16000
            )
            self.campaign_ledger = PersistentCampaignLedger()
            self.sleep_consolidator = AutonomicSleepConsolidator(
                hebbian_manifold=self.hebbian_manifold,
                episodic_ledger=EpisodicLedger(),
                causal_graph=HierarchicalCausalKnowledgeGraph(latent_dim=256),
                z3_verifier=Z3LogicVerifier(),
                campaign_ledger=self.campaign_ledger
            )

        # Background Thread Components
        self.mic_driver: Optional[LiveMicDriver] = None
        self._background_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_listening = False
        self._last_save_time = time.time()
        self._last_voice_activity_time = time.time()
        self._has_slept_in_current_idle_window = False

        # Adaptive Dynamic VAD & Spoken Turn-Taking State
        self._ambient_rms_ema: float = 0.0140
        self._recent_hop_rms: list = []
        self._current_turn_buffer: list = []
        self._current_turn_wave_segments: list = []
        self._current_turn_peak_rms: float = 0.0
        self._consecutive_quiet_hops: int = 0
        self._speaking_cooldown_until: float = 0.0
        self.auto_reply_voice: bool = True
        self.last_heard_query: str = ""

        # Telemetry metrics
        self.total_utterances_heard = 0
        self.suppressed_echo_count = 0
        self.barge_in_count = 0
        self.sleep_cycles_run = 0
        self.total_spoken_replies = 0
        self.last_spoken_reply: str = ""
        self.last_spoken_timestamp: float = 0.0
        self.last_flux_summary: Optional[Dict[str, Any]] = None

    @staticmethod
    def _speech_band_rms(waveform: torch.Tensor, sample_rate: int = 16000) -> float:
        """
        Computes RMS energy strictly within the human speech frequency band (140 Hz - 4200 Hz),
        filtering out low-frequency (< 140 Hz) laptop fan / AC motor rumble and high-frequency hiss.
        """
        w = waveform.detach().cpu().squeeze().float()
        if w.dim() != 1 or w.numel() < 64:
            return float(torch.sqrt(torch.mean(w ** 2) + 1e-12).item())
        w = w - torch.mean(w)
        spec = torch.fft.rfft(w)
        freqs = torch.fft.rfftfreq(w.numel(), d=1.0 / float(sample_rate))
        band_mask = torch.clamp((freqs - 120.0) / 40.0, 0.0, 1.0) * torch.clamp((4400.0 - freqs) / 400.0, 0.0, 1.0)
        w_speech = torch.fft.irfft(spec * band_mask, n=w.numel())
        return float(torch.sqrt(torch.mean(w_speech ** 2) + 1e-12).item())

    def _build_session_env(self) -> Dict[str, str]:
        """Ensures XDG_RUNTIME_DIR and DBUS_SESSION_BUS_ADDRESS are present for PipeWire and notify-send."""
        import os
        env = os.environ.copy()
        uid = os.getuid()
        runtime_dir = env.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}"
        if os.path.exists(runtime_dir):
            env["XDG_RUNTIME_DIR"] = runtime_dir
            bus_path = f"{runtime_dir}/bus"
            if "DBUS_SESSION_BUS_ADDRESS" not in env and os.path.exists(bus_path):
                env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"
        return env

    def _send_desktop_notification(self, title: str, message: str) -> None:
        """Sends a non-blocking desktop notification so the user can visually see Jane's spoken response."""
        import subprocess
        try:
            env = self._build_session_env()
            subprocess.Popen(
                ["notify-send", "-a", "Jane", "-t", "7000", title, message],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.debug("Desktop notification skipped: %s", e)

    def wake_on_boot(self) -> Dict[str, Any]:
        """
        Executed automatically when the laptop boots or the 24/7 daemon starts:
        1. Reloads the persisted 256D HebbianConceptManifold and user vocal tone centroid.
        2. Wakes and resumes any unfinished multi-day campaigns from PersistentCampaignLedger.
        """
        ledger = self.pipeline.campaign_ledger if self.pipeline is not None else self.campaign_ledger
        manifold_loaded = self.hebbian_manifold.load_manifold()
        resumed_campaigns = ledger.wake_and_resume_campaigns()
        logger.info(
            "Jane woke on boot: manifold_loaded=%s, total_concepts=%d, resumed_campaigns=%d",
            manifold_loaded,
            self.hebbian_manifold.num_concepts,
            len(resumed_campaigns)
        )
        return {
            "status": "AWAKE_AND_RESTORED",
            "manifold_loaded": manifold_loaded,
            "total_concepts": self.hebbian_manifold.num_concepts,
            "resumed_campaigns": resumed_campaigns
        }

    def speak_natural_wakeup(self, resumed_campaigns: Optional[list] = None, play_hardware_audio: bool = True) -> Dict[str, Any]:
        """
        Synthesizes and plays a natural, non-robotic wakeup thought grounded in Jane's
        live manifold and resumed campaigns when the daemon boots.
        """
        if self.pipeline is not None and hasattr(self.pipeline, "generate_natural_wakeup_thought"):
            wakeup_text = self.pipeline.generate_natural_wakeup_thought(resumed_campaigns=resumed_campaigns)
        else:
            wakeup_text = "Hey Akhil, I'm here with you whenever you want to dive in."

        return self.speak_and_play(
            wakeup_text,
            play_hardware_audio=play_hardware_audio,
            notify_desktop=play_hardware_audio,
        )

    def hibernate_on_shutdown(self) -> Dict[str, Any]:
        """
        Executed automatically when the laptop powers off, suspends, or receives SIGTERM:
        1. Stops live microphone capture cleanly.
        2. Hibernates all active multi-step campaigns to disk.
        3. Runs a fast NREM consolidation pass and saves the 256D manifold to disk.
        """
        self.stop_passive_listening()
        ledger = self.pipeline.campaign_ledger if self.pipeline is not None else self.campaign_ledger
        hibernated_ids = ledger.hibernate_active_campaigns()
        self.hebbian_manifold.save_manifold()
        logger.info(
            "Jane hibernated for shutdown: hibernated_campaigns=%s, saved_concepts=%d",
            hibernated_ids,
            self.hebbian_manifold.num_concepts
        )
        return {
            "status": "HIBERNATED_FOR_SHUTDOWN",
            "hibernated_campaigns": hibernated_ids,
            "saved_concepts": self.hebbian_manifold.num_concepts
        }

    def speak_and_play(
        self,
        text: str,
        z_thought: Optional[torch.Tensor] = None,
        play_hardware_audio: bool = True,
        notify_desktop: bool = False,
    ) -> Dict[str, Any]:
        """
        Synthesizes speech via NeuralVocalTract, registers it with the Full-Duplex
        Acoustic Echo Canceller so Jane never triggers on her own voice, exports the WAV
        to logs/voice_demos/latest_utterance.wav, and plays it out loud via PipeWire
        pw-play (with ALSA aplay fallback).
        """
        import os
        import subprocess

        wav_path = "logs/voice_demos/latest_utterance.wav"
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)

        if self.pipeline is not None:
            waveform, telemetry = self.pipeline.speak(text, z_thought=z_thought, export_path=wav_path)
        else:
            if z_thought is None:
                z_thought = (
                    self.hebbian_manifold.user_tone_centroid.clone()
                    if torch.norm(self.hebbian_manifold.user_tone_centroid) > 1e-4
                    else None
                )
            waveform, telemetry = self.vocal_tract.synthesize_speech(
                text,
                z_thought=z_thought,
                cadence=self.full_duplex_stream.active_cadence
            )
            self.full_duplex_stream.register_jane_speech(waveform)
            if hasattr(self.vocal_tract, "export_studio_wav"):
                self.vocal_tract.export_studio_wav(wav_path, fallback_waveform=waveform)
            else:
                self.vocal_tract.export_wav(waveform, wav_path)
            telemetry["wav_path"] = wav_path

        duration_sec = float(telemetry.get("duration_sec", 2.0))
        self.last_spoken_reply = text
        self.last_spoken_timestamp = time.time()
        self.total_spoken_replies += 1

        if notify_desktop:
            self._send_desktop_notification("Jane", text)

        played_hw = False
        if play_hardware_audio:
            self._speaking_cooldown_until = time.time() + duration_sec + 1.5
            env = self._build_session_env()
            # 1. Primary: PipeWire pw-play (mixes cleanly with live microphone capture)
            try:
                res = subprocess.run(
                    ["pw-play", wav_path],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=max(12.0, duration_sec + 4.0),
                )
                played_hw = (res.returncode == 0)
            except Exception as e:
                logger.debug("pw-play unavailable or failed (%s); falling back to aplay", e)

            # 2. Fallback: ALSA aplay
            if not played_hw:
                try:
                    res = subprocess.run(
                        ["aplay", "-q", wav_path],
                        env=env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=max(12.0, duration_sec + 4.0),
                    )
                    played_hw = (res.returncode == 0)
                except Exception as e:
                    logger.debug("Hardware speaker playback skipped or unavailable: %s", e)

            # Drain any microphone chunks captured while the speaker was actively playing
            if self.mic_driver is not None and hasattr(self.mic_driver, "audio_queue"):
                while not self.mic_driver.audio_queue.empty():
                    try:
                        self.mic_driver.audio_queue.get_nowait()
                    except Exception:
                        break
            self._speaking_cooldown_until = time.time() + 1.2
        else:
            self._speaking_cooldown_until = time.time() + duration_sec + 0.55

        telemetry["played_hardware_audio"] = played_hw
        telemetry["waveform"] = waveform
        telemetry["spoken_text"] = text
        return telemetry

    def _complete_spoken_turn_and_reply(self, play_hardware_audio: bool = True) -> Dict[str, Any]:
        """
        Concatenates the buffered raw waveform segments and 256D acoustic vectors from the
        user's completed spoken turn, decodes the spoken utterance via Non-Autoregressive
        Wav2Vec2-CTC, executes Jane's cognitive pipeline (`pipeline.process()`), and
        speaks the answer out loud through the speakers.
        """
        turn_items = list(self._current_turn_buffer)
        wave_segments = list(self._current_turn_wave_segments)
        self._current_turn_buffer.clear()
        self._current_turn_wave_segments.clear()
        self._current_turn_peak_rms = 0.0
        self._consecutive_quiet_hops = 0

        if not turn_items:
            return {}

        raw_turn_waveform = None
        if wave_segments:
            raw_turn_waveform = torch.cat(wave_segments, dim=-1)

        z_list = [item["z_user"].squeeze() for item in turn_items if item.get("z_user") is not None]
        if z_list:
            z_stack = torch.stack(z_list, dim=0)
            z_turn = torch.nn.functional.normalize(torch.mean(z_stack, dim=0), p=2, dim=-1)
        else:
            z_turn = torch.randn(256, device=self.device)

        fired_concepts = []
        for item in turn_items:
            for c in item.get("fired_concepts", []):
                c_name = c[0] if isinstance(c, (list, tuple)) else str(c)
                if c_name and c_name not in fired_concepts:
                    fired_concepts.append(c_name)

        dominant_tone = turn_items[-1].get("tone_label", "composed_inquiry")
        matched_intent = next((item.get("intent") for item in reversed(turn_items) if item.get("intent") is not None), None)

        spoken_query = ""
        if self.pipeline is not None and hasattr(self.pipeline, "respond_to_acoustic_turn"):
            turn_res = self.pipeline.respond_to_acoustic_turn(
                z_turn=z_turn,
                fired_concepts=fired_concepts,
                tone_label=dominant_tone,
                matched_intent=matched_intent,
                raw_waveform=raw_turn_waveform,
                require_valid_speech=play_hardware_audio,
            )
            if turn_res.get("status") == "IGNORED_NON_SPEECH":
                return {"status": "IGNORED_NON_SPEECH", "reply_text": ""}
            reply_text = turn_res.get("response_text", "").strip()
            spoken_query = turn_res.get("spoken_query", "").strip()
            z_thought = turn_res.get("settled_latent")
        else:
            ctc_res = self.audio_encoder.decode_acoustic_speech(
                raw_turn_waveform,
                concept_lexicon=list(self.hebbian_manifold.concept_names),
            ) if raw_turn_waveform is not None else {"is_valid_speech": False, "spoken_text": ""}
            if play_hardware_audio and not ctc_res.get("is_valid_speech"):
                return {"status": "IGNORED_NON_SPEECH", "reply_text": ""}
            spoken_query = ctc_res.get("spoken_text", "").strip()
            reply_text = spoken_query or (fired_concepts[0].replace("_", " ") if fired_concepts else "")
            z_thought = z_turn

        if not reply_text:
            return {"status": "IGNORED_NON_SPEECH", "reply_text": ""}

        if spoken_query:
            self.last_heard_query = spoken_query

        speak_telemetry = self.speak_and_play(
            reply_text,
            z_thought=z_thought,
            play_hardware_audio=play_hardware_audio,
            notify_desktop=play_hardware_audio,
        )
        return {
            "status": "REPLIED",
            "spoken_query": spoken_query,
            "reply_text": reply_text,
            "fired_concepts": fired_concepts,
            "tone_label": dominant_tone,
            "played_hardware_audio": speak_telemetry.get("played_hardware_audio", False),
            "wav_path": speak_telemetry.get("wav_path", "logs/voice_demos/latest_utterance.wav"),
        }

    def process_audio_chunk_once(
        self,
        chunk: torch.Tensor,
        auto_reply_on_pause: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes a single [1, T] audio chunk through Dynamic Self-Calibrating Voice-Band VAD
        + Full-Duplex Echo Cancellation + Hebbian Manifold adaptation + Fast Spoken Turn-Taking.
        """
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        full_rms = float(torch.sqrt(torch.mean(chunk ** 2)).item())
        hop_samples = min(chunk.shape[-1], 8000)
        hop_slice = chunk[:, -hop_samples:]
        hop_rms = float(torch.sqrt(torch.mean(hop_slice ** 2)).item())
        hop_voice_rms = self._speech_band_rms(hop_slice)

        now = time.time()

        # 0. If Jane is currently speaking out loud, suppress speaker room reflections unless user barges in
        if now < self._speaking_cooldown_until:
            duplex_res = self.full_duplex_stream.ingest_audio_chunk(chunk.to(self.device))
            if not duplex_res.get("is_barge_in"):
                self.suppressed_echo_count += 1
                duplex_res["is_self_echo"] = True
                return duplex_res
            # User barged in with high orthogonal energy!
            self._speaking_cooldown_until = 0.0

        # Continuous Dynamic Noise Floor Tracking (25th percentile of recent 12s speech-band hops)
        if full_rms > 1e-4:
            self._recent_hop_rms.append(hop_voice_rms)
            if len(self._recent_hop_rms) > 24:
                self._recent_hop_rms.pop(0)
            sorted_hops = sorted(self._recent_hop_rms)
            p25_floor = sorted_hops[max(0, len(sorted_hops) // 4)]
            if len(self._recent_hop_rms) == 1:
                self._ambient_rms_ema = max(0.005, min(0.045, p25_floor))
            else:
                # Fast downward tracking when room quiets, steady upward tracking if fan speeds up
                alpha = 0.35 if p25_floor < self._ambient_rms_ema else 0.15
                self._ambient_rms_ema = max(0.005, min(0.065, (1.0 - alpha) * self._ambient_rms_ema + alpha * p25_floor))

        if len(self._recent_hop_rms) >= 3:
            dynamic_voice_threshold = max(0.008, self._ambient_rms_ema * 1.24, self._ambient_rms_ema + 0.0035)
        else:
            dynamic_voice_threshold = min(self.audio_threshold_rms, 0.020)

        active_turn_threshold = (
            max(dynamic_voice_threshold, 0.32 * self._current_turn_peak_rms)
            if len(self._current_turn_buffer) >= 2
            else dynamic_voice_threshold
        )

        is_explicit_silence = (full_rms < 0.004) or (
            not auto_reply_on_pause and full_rms < self.audio_threshold_rms and hop_voice_rms < 0.015
        )
        is_voice_hop = (not is_explicit_silence) and (hop_voice_rms >= active_turn_threshold)

        # 1. Check if the latest 0.5s hop has returned to the dynamic noise floor (silence or pause)
        if is_explicit_silence or not is_voice_hop:
            if len(self._current_turn_buffer) > 0:
                self._consecutive_quiet_hops += 1
                # Complete turn immediately on explicit synthetic zero-silence, or after 2 consecutive
                # quiet hops (1.0s genuine trailing pause) on live microphone so mid-sentence breaths never chop speech.
                should_complete_turn = (
                    is_explicit_silence
                    or self._consecutive_quiet_hops >= 2
                )
                if should_complete_turn:
                    reply_info = self._complete_spoken_turn_and_reply(play_hardware_audio=auto_reply_on_pause)
                    if self.last_flux_summary is not None and reply_info.get("reply_text"):
                        self.last_flux_summary["spoken_reply"] = reply_info.get("reply_text", "")
                        self.last_flux_summary["heard_query"] = reply_info.get("spoken_query", "")
                        if self.on_flux_callback:
                            try:
                                self.on_flux_callback(self.last_flux_summary)
                            except Exception as e:
                                logger.error("Error executing on_flux_callback: %s", e, exc_info=True)
                    return {
                        "status": "TURN_COMPLETED_AND_REPLIED",
                        "audio_rms": full_rms,
                        "spoken_query": reply_info.get("spoken_query", ""),
                        "spoken_reply": reply_info.get("reply_text", ""),
                        "reply_info": reply_info,
                    }
                else:
                    # Keep the single brief mid-phrase breath hop in the waveform so words aren't clipped
                    self._current_turn_wave_segments.append(hop_slice.detach().cpu().clone())
                    return {"status": "BRIEF_BREATH_PAUSE", "audio_rms": full_rms}

            idle_sec = now - self._last_voice_activity_time
            if (
                idle_sec >= self.sleep_idle_threshold_sec
                and not self._has_slept_in_current_idle_window
            ):
                sleep_rep = self.sleep_consolidator.run_sleep_cycle()
                self._has_slept_in_current_idle_window = True
                self.sleep_cycles_run += 1
                return {
                    "status": "AUTONOMIC_SLEEP_COMPLETED",
                    "sleep_report": sleep_rep
                }
            return {"status": "SILENCE", "audio_rms": full_rms}

        # 2. Active user voice detected above dynamic noise floor
        self._last_voice_activity_time = now
        self._has_slept_in_current_idle_window = False
        self._consecutive_quiet_hops = 0

        duplex_res = self.full_duplex_stream.ingest_audio_chunk(chunk.to(self.device))
        if duplex_res.get("is_self_echo"):
            self.suppressed_echo_count += 1
            return duplex_res

        if duplex_res.get("is_barge_in"):
            self.barge_in_count += 1

        self.total_utterances_heard += 1
        # Accumulate contiguous raw audio waveform (first hop includes pre-roll so initial syllable is never clipped)
        if len(self._current_turn_wave_segments) == 0:
            self._current_turn_wave_segments.append(chunk.detach().cpu().clone())
        else:
            self._current_turn_wave_segments.append(hop_slice.detach().cpu().clone())

        self._current_turn_buffer.append(duplex_res)
        self._current_turn_peak_rms = max(self._current_turn_peak_rms, hop_voice_rms)

        # Allow up to ~14.0s (28 hops) of continuous speech before forcing a phrase boundary
        phrase_full = len(self._current_turn_buffer) >= 28

        spoken_reply_text = ""
        if auto_reply_on_pause and phrase_full:
            reply_info = self._complete_spoken_turn_and_reply(play_hardware_audio=True)
            spoken_reply_text = reply_info.get("reply_text", "")

        flux_summary = {
            "utterance_count": self.hebbian_manifold.total_utterances_observed,
            "audio_rms": duplex_res.get("residual_rms", full_rms),
            "rms": duplex_res.get("residual_rms", full_rms),
            "voice_band_rms": round(hop_voice_rms, 4),
            "dynamic_noise_floor": round(self._ambient_rms_ema, 4),
            "tone_label": duplex_res.get("tone_label", "ambient"),
            "fired_concepts": duplex_res.get("fired_concepts", []),
            "reinforced_edges": duplex_res.get("reinforced_edges", []),
            "is_barge_in": duplex_res.get("is_barge_in", False),
            "adapted_cadence": duplex_res.get("adapted_cadence"),
            "heard_query": self.last_heard_query,
            "spoken_reply": spoken_reply_text or self.last_spoken_reply,
        }
        self.last_flux_summary = flux_summary

        if self.on_flux_callback:
            try:
                self.on_flux_callback(flux_summary)
            except Exception as e:
                logger.error("Error executing on_flux_callback: %s", e, exc_info=True)

        if now - self._last_save_time > self.autosave_interval_sec:
            self.hebbian_manifold.save_manifold()
            self._last_save_time = now

        return flux_summary

    def start_passive_listening(self):
        """Launches non-blocking background microphone listener and Hebbian learning thread."""
        if self._is_listening:
            return

        self._stop_event.clear()
        self.mic_driver = LiveMicDriver(
            sample_rate=16000,
            chunk_duration=1.5,
            hop_duration=0.5
        )
        try:
            self.mic_driver.start()
        except Exception as e:
            print(f"[ModeRouter] Notice: Audio capture unavailable ({e}). Continuing in text-only mode.")
            return

        self._is_listening = True
        self._background_thread = threading.Thread(
            target=self._passive_listener_loop,
            daemon=True,
            name="Jane-PassiveEar"
        )
        self._background_thread.start()

    def stop_passive_listening(self):
        """Stops the background eavesdropping thread and persists the learned manifold."""
        if not self._is_listening:
            return

        self._stop_event.set()
        self._is_listening = False

        if self.mic_driver:
            self.mic_driver.stop()
            self.mic_driver = None

        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=2.0)
            self._background_thread = None

        # Persist weights
        self.hebbian_manifold.save_manifold()

    def _passive_listener_loop(self):
        """
        24/7 Background loop:
        - Listens to user speech with Adaptive VAD sleep gating.
        - Cancels Jane's own speech reflections via FullDuplexAcousticStream.
        - Replies out loud when the user finishes a spoken turn in ACTIVE/HYBRID mode.
        - Triggers NREM/REM autonomic sleep consolidation during extended idle windows.
        """
        while not self._stop_event.is_set():
            if not self.mic_driver:
                break

            chunk = self.mic_driver.get_chunk(timeout=0.2)
            if chunk is None:
                time.sleep(0.02)
                continue

            should_reply = self.auto_reply_voice and (self.mode in (OperatingMode.ACTIVE, OperatingMode.HYBRID))
            res = self.process_audio_chunk_once(chunk, auto_reply_on_pause=should_reply)
            if res.get("status") == "SILENCE":
                time.sleep(0.02)

    def set_mode(self, new_mode: OperatingMode):
        """Switches Jane's operational mode."""
        self.mode = new_mode
        if new_mode in (OperatingMode.PASSIVE, OperatingMode.HYBRID):
            self.start_passive_listening()
        else:
            self.stop_passive_listening()

    @property
    def is_listening(self) -> bool:
        return self._is_listening

