#!/usr/bin/env python3
"""Sovereign 24/7 Background Daemon for Project Jane.

Runs Jane continuously in the background while the laptop is powered on:
1. Wakes on boot (`router.wake_on_boot()`), automatically resuming any multi-day
   scientific campaigns hibernated before the previous shutdown.
2. Starts the Full-Duplex Passive/Hybrid Ear (`router.start_passive_listening()`)
   capturing live hardware microphone audio (`sounddevice` / ALSA `arecord`),
   filtering room self-echo via Hilbert-space orthogonal projection, and wiring
   user cadence + concepts directly into the 256D Hebbian manifold.
3. Runs the background Epistemic Curiosity Daemon (`pipeline.start_curiosity_daemon()`)
   and triggers autonomic NREM/REM sleep consolidation during extended idle windows.
4. Intercepts `SIGTERM` / `SIGINT` from Linux `systemd` on laptop shutdown or
   reboot, executing `router.hibernate_on_shutdown()` to checkpoint all active
   campaigns, manifold weights, and vocal profiles cleanly to disk.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.core.mode_router import ModeRouter, OperatingMode
from src.core.pipeline import JanePipeline
from src.utils.logger import get_logger

logger = get_logger("jane_24x7_daemon")


class SovereignJaneDaemon:
    """Manages Jane's 24/7 autonomous lifecycle across laptop boots and shutdowns."""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_pretrained_translator: bool = True,
        idle_sleep_threshold_sec: float = 1800.0,
        curiosity_interval_sec: float = 600.0,
        status_file: str = "logs/daemon_status.json",
    ):
        self.device = device
        self.status_file = Path(PROJECT_ROOT) / status_file
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self._shutdown_requested = False
        self.boot_timestamp = time.time()
        self.last_flux_telemetry: Dict[str, Any] = {}

        logger.info("Initializing SovereignJaneDaemon on device=%s...", self.device)
        self.pipeline = JanePipeline(
            latent_dim=256,
            use_pretrained_translator=use_pretrained_translator,
            device=self.device,
        )

        self.router = ModeRouter(
            mode=OperatingMode.HYBRID,
            device=self.device,
            on_flux_callback=self._on_acoustic_flux,
            pipeline=self.pipeline,
            sleep_idle_threshold_sec=idle_sleep_threshold_sec,
        )
        self.curiosity_interval_sec = curiosity_interval_sec
        self.resumed_campaigns = []

    def _on_acoustic_flux(self, telemetry: Dict[str, Any]) -> None:
        """Invoked whenever the passive ear processes an acoustic utterance."""
        self.last_flux_telemetry = {
            "timestamp": time.time(),
            "rms": round(float(telemetry.get("audio_rms", telemetry.get("rms", 0.0))), 4),
            "voice_band_rms": round(float(telemetry.get("voice_band_rms", 0.0)), 4),
            "dynamic_noise_floor": round(float(telemetry.get("dynamic_noise_floor", getattr(self.router, "_ambient_rms_ema", 0.0))), 4),
            "tone_label": telemetry.get("tone_label", "Normal"),
            "fired_concepts": [
                c[0] if isinstance(c, (list, tuple)) else str(c)
                for c in telemetry.get("fired_concepts", [])
            ],
            "echo_cancelled": telemetry.get("echo_cancelled", False),
            "barge_in_detected": telemetry.get("is_barge_in", telemetry.get("barge_in_detected", False)),
            "heard_query": telemetry.get("heard_query", telemetry.get("spoken_query", getattr(self.router, "last_heard_query", ""))),
            "spoken_reply": telemetry.get("spoken_reply", self.router.last_spoken_reply),
        }
        self._write_status(state="ACTIVE_LISTENING")

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        """Signal handler for SIGTERM (systemd shutdown) and SIGINT."""
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.info("Received %s — initiating graceful hibernation...", sig_name)
        self._shutdown_requested = True

    def start(self, start_hardware_ear: bool = True, start_curiosity: bool = True) -> Dict[str, Any]:
        """Wakes Jane on boot, resumes hibernated campaigns, greets naturally, and starts background threads."""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # 1. Wake on boot & resume hibernated multi-day campaigns
        wake_report = self.router.wake_on_boot()
        self.resumed_campaigns = wake_report.get("resumed_campaigns", [])
        logger.info(
            "Boot wakeup complete: resumed %d campaign(s).",
            len(self.resumed_campaigns),
        )

        # 2. Speak a natural, conversational wakeup thought before opening the mic stream
        if start_hardware_ear:
            self.router.speak_natural_wakeup(
                resumed_campaigns=self.resumed_campaigns,
                play_hardware_audio=True,
            )

        # 3. Start continuous hardware microphone listening & full-duplex turn-taking
        if start_hardware_ear:
            self.router.start_passive_listening()

        # 4. Start background epistemic curiosity daemon
        if start_curiosity and self.curiosity_interval_sec > 0:
            self.pipeline.start_curiosity_daemon(interval_sec=self.curiosity_interval_sec)

        status = self._write_status(state="RUNNING_24X7")
        return status

    def shutdown(self) -> Dict[str, Any]:
        """Gracefully stops daemons, hibernates active campaigns, and saves manifold."""
        logger.info("Executing SovereignJaneDaemon shutdown & campaign hibernation...")
        try:
            self.pipeline.stop_curiosity_daemon()
        except Exception as exc:
            logger.warning("Curiosity daemon stop warning: %s", exc)

        hibernation_report = self.router.hibernate_on_shutdown()
        status = self._write_status(
            state="HIBERNATED",
            extra={"hibernation_report": hibernation_report},
        )
        logger.info("SovereignJaneDaemon cleanly hibernated to disk.")
        return status

    def _write_status(
        self,
        state: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Writes live daemon health & telemetry to logs/daemon_status.json."""
        payload: Dict[str, Any] = {
            "pid": os.getpid(),
            "state": state,
            "boot_timestamp": self.boot_timestamp,
            "uptime_sec": round(time.time() - self.boot_timestamp, 2),
            "mode": self.router.mode.value,
            "is_listening": self.router.is_listening,
            "resumed_campaigns_on_boot": len(self.resumed_campaigns),
            "sleep_cycles_completed": self.router.sleep_cycles_run,
            "manifold_concepts": self.router.hebbian_manifold.num_concepts,
            "total_spoken_replies": self.router.total_spoken_replies,
            "last_heard_query": getattr(self.router, "last_heard_query", ""),
            "last_spoken_reply": self.router.last_spoken_reply,
            "last_spoken_timestamp": self.router.last_spoken_timestamp,
            "last_flux": self.last_flux_telemetry,
            "updated_at": time.time(),
        }
        if extra:
            payload.update(extra)

        try:
            tmp_path = self.status_file.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp_path.replace(self.status_file)
        except Exception as exc:
            logger.warning("Failed to write daemon status file: %s", exc)

        return payload

    def run_forever(
        self,
        max_duration_sec: Optional[float] = None,
        heartbeat_sec: float = 2.0,
        start_hardware_ear: bool = True,
        start_curiosity: bool = True,
    ) -> Dict[str, Any]:
        """Runs the main 24/7 loop until SIGTERM/SIGINT or max_duration_sec expires."""
        self.start(
            start_hardware_ear=start_hardware_ear,
            start_curiosity=start_curiosity,
        )
        start_t = time.time()
        try:
            while not self._shutdown_requested:
                if max_duration_sec is not None and (time.time() - start_t) >= max_duration_sec:
                    logger.info("Max duration %.2fs reached; initiating shutdown.", max_duration_sec)
                    break
                self._write_status(state="RUNNING_24X7")
                time.sleep(min(heartbeat_sec, 0.5 if max_duration_sec else heartbeat_sec))
        finally:
            final_status = self.shutdown()
        return final_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Jane 24/7 Sovereign Background Daemon")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional runtime duration in seconds before clean hibernation (default: infinite 24/7).",
    )
    parser.add_argument(
        "--idle-sleep-sec",
        type=float,
        default=1800.0,
        help="Seconds of acoustic silence before triggering an autonomic NREM/REM sleep cycle.",
    )
    parser.add_argument(
        "--curiosity-sec",
        type=float,
        default=600.0,
        help="Interval in seconds between autonomous epistemic curiosity cycles.",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        default=True,
        help="Load in-process Qwen translator weights in daemon mode (enabled by default).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable in-process LLM translator weights in daemon mode.",
    )
    args = parser.parse_args()

    daemon = SovereignJaneDaemon(
        use_pretrained_translator=(not args.no_llm),
        idle_sleep_threshold_sec=args.idle_sleep_sec,
        curiosity_interval_sec=args.curiosity_sec,
    )
    daemon.run_forever(max_duration_sec=args.duration)


if __name__ == "__main__":
    main()
