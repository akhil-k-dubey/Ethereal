"""GPU Training Pipeline for Jane's Piper VITS + HiFi-GAN Studio Voice & Prosody Adapter.

Trains `HiFiStudioVoiceAdapter` and `NeuralManifoldArticulator` on CUDA using
22.05 kHz Piper VITS + HiFi-GAN waveforms across manifold concept anchors and cadence modes:
1. Minimizes Nasal-to-Open-Mouth Energy Ratio (eliminates 290-385 Hz boxy/nasal coloration).
2. Optimizes 110-175 Hz vocal-cord chest fundamental warmth and 5.8-10.5 kHz studio air brilliance.
3. Aligns 256D manifold thought vectors with stable prosodic f0/effort/breathiness targets.
4. Saves converged weights to `logs/training_runs/jane_hifigan_voice_adapter.pt`.
"""

from __future__ import annotations

import os
import time
import torch
import torch.nn.functional as F

from src.perception.neural_vocal_tract import (
    NeuralVocalTract,
    VocalCadenceProfile,
)


def compute_spectral_band_ratios(
    waveform_22k: torch.Tensor,
    sample_rate: int = 22050,
) -> dict[str, torch.Tensor]:
    """Computes differentiable spectral energy ratios across key human vocal bands."""
    N = waveform_22k.shape[-1]
    spec_mag = torch.abs(torch.fft.rfft(waveform_22k)) + 1e-7
    power = spec_mag ** 2
    freqs = torch.fft.rfftfreq(N, d=1.0 / float(sample_rate), device=waveform_22k.device)

    total_power = torch.mean(power) + 1e-8
    chest_band = (freqs >= 110.0) & (freqs <= 175.0)
    nasal_band = (freqs >= 290.0) & (freqs <= 385.0)
    diction_band = (freqs >= 2000.0) & (freqs <= 4000.0)
    air_band = (freqs >= 5800.0) & (freqs <= 10500.0)

    return {
        "chest_ratio": torch.mean(power[chest_band]) / total_power,
        "nasal_ratio": torch.mean(power[nasal_band]) / total_power,
        "diction_ratio": torch.mean(power[diction_band]) / total_power,
        "air_ratio": torch.mean(power[air_band]) / total_power,
    }


def train_piper_hifigan_voice_adapter(
    epochs: int = 100,
    lr: float = 3e-3,
    save_path: str = "logs/training_runs/jane_hifigan_voice_adapter.pt",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[VoiceTrainer] Initializing Piper VITS + HiFi-GAN Studio Adapter training on {device}...")

    tract = NeuralVocalTract(sample_rate=16000, latent_dim=256)
    phrases = [
        ("composed_inquiry", "Following your train of thought on conservation and how it links to energy, Akhil."),
        ("teaching", "Hamiltonian dynamics preserve symplectic phase volume across continuous trajectories."),
        ("urgent", "Equilibrium drift detected across the primary manifold boundary."),
        ("whisper", "Quietly consolidating verified axioms while the ambient room is calm."),
    ]

    # Collect raw 22.05 kHz HiFi-GAN waveforms from Piper before adapter sculpting
    raw_samples = []
    for idx, (mode_name, text) in enumerate(phrases):
        cadence = VocalCadenceProfile.from_tone_label(mode_name)
        z = torch.zeros(256, dtype=torch.float32)
        z[idx * 17] = 1.0
        z[(idx * 17 + 5) % 256] = 0.6
        z = F.normalize(z, p=2, dim=-1)

        _, _ = tract.synthesize_speech(text, z_thought=z, cadence=cadence)
        assert tract.last_studio_waveform is not None
        raw_22k = tract.last_studio_waveform.clone().to(device)
        raw_samples.append((mode_name, cadence, z.to(device), raw_22k))

    adapter = tract.studio_adapter.to(device)
    articulator = tract.articulator.to(device)
    adapter.train()
    articulator.train()

    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(articulator.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    # Compute baseline spectral ratios on raw waveforms
    base_ratios = [compute_spectral_band_ratios(w) for _, _, _, w in raw_samples]

    t0 = time.perf_counter()
    initial_loss = None
    final_loss = 0.0

    for step in range(epochs):
        optimizer.zero_grad()
        step_loss = torch.tensor(0.0, device=device)

        for i, (mode_name, cadence, z_vec, raw_wave) in enumerate(raw_samples):
            sculpted_wave, _ = adapter(raw_wave, z_thought=z_vec, cadence=cadence, sample_rate=22050)
            cur_ratios = compute_spectral_band_ratios(sculpted_wave)
            b_r = base_ratios[i]

            # 1. Anti-nasal open-throat target: reduce 290-385 Hz nasal energy by ~32% (-1.8 to -2.2 dB)
            target_nasal = b_r["nasal_ratio"].detach() * 0.68
            loss_nasal = F.smooth_l1_loss(cur_ratios["nasal_ratio"], target_nasal) * 15.0

            # 2. Chest fundamental warmth target: boost 110-175 Hz fundamental by ~28% (+1.1 dB)
            target_chest = b_r["chest_ratio"].detach() * 1.28
            loss_chest = F.smooth_l1_loss(cur_ratios["chest_ratio"], target_chest) * 10.0

            # 3. Open-mouth diction & studio air target: boost 2-4 kHz diction (+22%) and 6-10.5 kHz air (+35%)
            target_diction = b_r["diction_ratio"].detach() * 1.22
            target_air = b_r["air_ratio"].detach() * 1.35
            loss_clarity = (
                F.smooth_l1_loss(cur_ratios["diction_ratio"], target_diction) * 20.0
                + F.smooth_l1_loss(cur_ratios["air_ratio"], target_air) * 25.0
            )

            # 4. Waveform phase-fidelity & symmetry regularization on FIR presence kernel
            w_conv = adapter.presence_conv.weight.squeeze()
            loss_sym = F.mse_loss(w_conv[:7], torch.flip(w_conv[8:], dims=[0])) * 5.0
            loss_identity = F.mse_loss(sculpted_wave, raw_wave) * 0.25

            # 5. Differentiable prosodic articulator calibration
            feats = articulator.thought_adapter(z_vec.unsqueeze(0))
            pred_df0 = articulator.pitch_head(feats).squeeze()
            target_df0 = torch.tensor(
                0.35 if mode_name == "urgent" else (-0.25 if mode_name == "whisper" else 0.05),
                device=device,
            )
            loss_prosody = F.mse_loss(pred_df0, target_df0) * 2.0

            step_loss = step_loss + loss_nasal + loss_chest + loss_clarity + loss_sym + loss_identity + loss_prosody

        step_loss.backward()
        optimizer.step()

        loss_val = float(step_loss.item())
        if initial_loss is None:
            initial_loss = loss_val
        final_loss = loss_val

        if (step + 1) % 25 == 0 or step == 0:
            print(f"  [Epoch {step+1:03d}/{epochs}] Studio Acoustic Loss: {loss_val:.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(
        {
            "studio_adapter": adapter.cpu().state_dict(),
            "articulator": articulator.cpu().state_dict(),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "sample_rate": 22050,
            "vocoder": "piper_vits_hifigan_en_US-hfc_female-medium",
        },
        save_path,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[VoiceTrainer] Saved trained HiFi-GAN studio adapter to {save_path} "
        f"in {elapsed:.2f}s (Loss: {initial_loss:.5f} -> {final_loss:.5f})"
    )
    return {
        "save_path": save_path,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "elapsed_sec": round(elapsed, 2),
    }


if __name__ == "__main__":
    train_piper_hifigan_voice_adapter()
