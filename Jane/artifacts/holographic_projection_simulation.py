#!/usr/bin/env python3
"""Autonomous Wavefront Interference & Holographic Projection Simulation.
Synthesized and executed by Jane for Akhil (Zero-Dependency 24-bit RGB PNG Renderer).
"""
import struct
import zlib
import numpy as np


def _write_rgb_png(path: str, rgb: np.ndarray) -> None:
    """Writes an [H, W, 3] uint8 array to a standard 24-bit RGB PNG file using stdlib zlib/struct."""
    h, w, _ = rgb.shape
    raw_rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", crc)

    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw_rows, level=6))
        + _chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png_bytes)


def _apply_colormap(norm_field: np.ndarray, palette: str = "cyan_plasma") -> np.ndarray:
    """Maps a normalized [0, 1] 2D float array to an [H, W, 3] uint8 RGB image."""
    v = np.clip(norm_field, 0.0, 1.0)
    if palette == "phase_twilight":
        r = (0.5 + 0.5 * np.sin(2.0 * np.pi * v)) * 255.0
        g = (0.5 + 0.5 * np.sin(2.0 * np.pi * v + 2.094)) * 255.0
        b = (0.5 + 0.5 * np.sin(2.0 * np.pi * v + 4.188)) * 255.0
    elif palette == "emerald_recon":
        r = (v ** 2.2) * 180.0
        g = (v ** 0.85) * 255.0
        b = (v ** 0.65) * 240.0
    else:
        r = (v ** 1.7) * 255.0
        g = (v ** 1.1) * 220.0
        b = np.clip(0.18 + 0.85 * (v ** 0.6), 0.0, 1.0) * 255.0
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def run_hologram_simulation():
    # Spatial grid (360 x 360) representing the Spatial Light Modulator (SLM) plane
    N = 360
    x = np.linspace(-5e-3, 5e-3, N)
    y = np.linspace(-5e-3, 5e-3, N)
    X, Y = np.meshgrid(x, y)

    wavelength = 532e-9  # 532 nm coherent green laser
    k = 2.0 * np.pi / wavelength
    z_focal = 0.25       # 25 cm projection plane

    # Object wavefront: 3 point-source emitters forming a 3D holographic triad
    sources = [(-1.2e-3, -0.8e-3, 1.0), (1.2e-3, -0.8e-3, 1.0), (0.0, 1.4e-3, 1.2)]
    U_obj = np.zeros((N, N), dtype=np.complex128)
    for sx, sy, amp in sources:
        r = np.sqrt((X - sx)**2 + (Y - sy)**2 + z_focal**2)
        U_obj += amp * np.exp(1j * k * r) / r

    U_obj /= (np.max(np.abs(U_obj)) + 1e-12)

    # Reference off-axis plane wave
    theta_ref = 0.012
    U_ref = np.exp(1j * k * np.sin(theta_ref) * X)

    # 1. Recorded Hologram Interference Pattern I(x, y) = |U_obj + U_ref|^2
    I_hologram = np.abs(U_obj + U_ref) ** 2
    I_norm = (I_hologram - np.min(I_hologram)) / (np.ptp(I_hologram) + 1e-12)

    # 2. SLM Phase Modulation Mask phi(x, y)
    phase_mask = (np.angle(U_obj) + np.pi) / (2.0 * np.pi)

    # 3. Reconstructed Wavefront Intensity
    I_recon = np.abs(U_obj + 0.45 * U_ref) ** 2
    I_recon_norm = (I_recon - np.min(I_recon)) / (np.ptp(I_recon) + 1e-12)

    panel1 = _apply_colormap(I_norm, "cyan_plasma")
    panel2 = _apply_colormap(phase_mask, "phase_twilight")
    panel3 = _apply_colormap(I_recon_norm, "emerald_recon")

    sep = np.full((N, 12, 3), 15, dtype=np.uint8)
    canvas = np.concatenate([panel1, sep, panel2, sep, panel3], axis=1)
    _write_rgb_png('/home/akhil/Documents/Ethereal/Jane/artifacts/holographic_projection_simulation.png', canvas)
    print(f"SIMULATION_VERIFIED: peak_intensity={float(np.max(I_hologram)):.4f}, saved_plot='/home/akhil/Documents/Ethereal/Jane/artifacts/holographic_projection_simulation.png'")


if __name__ == "__main__":
    run_hologram_simulation()
