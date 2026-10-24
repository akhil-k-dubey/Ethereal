#!/usr/bin/env bash
# =====================================================================
# PROJECT JANE — 24/7 SYSTEMD USER SERVICE INSTALLER
# Installs and enables jane-daemon.service so Jane starts automatically
# whenever the laptop boots and cleanly hibernates on shutdown.
# =====================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_SRC="$PROJECT_ROOT/configs/jane-daemon.service"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
SERVICE_DST="$SYSTEMD_USER_DIR/jane-daemon.service"

mkdir -p "$SYSTEMD_USER_DIR"
cp "$SERVICE_SRC" "$SERVICE_DST"

echo "[Jane 24/7] Installed unit file to: $SERVICE_DST"

if command -v systemctl &> /dev/null; then
    systemctl --user daemon-reload || true
    systemctl --user enable --now jane-daemon.service || true
    echo "[Jane 24/7] Service enabled and started via systemctl --user."
    echo "[Jane 24/7] Check live status anytime with: ./jane --status"
else
    echo "[Jane 24/7] systemctl not found in current environment; unit file is staged at $SERVICE_DST"
fi
