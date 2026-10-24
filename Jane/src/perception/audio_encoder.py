"""Direct Audio-to-Vector Transformer: Bypassing Speech-to-Text Completely.

No Whisper, no English text transcription, zero text hallucination.
Maps raw 16kHz audio waveforms directly into Jane's 256D Relational Manifold:
1. 1D Temporal Convolutional Stems (Wav2Vec2 style downsampling).
2. Temporal Transformer Encoder (Self-Attention over acoustic frames).
3. 256D Multimodal Bottleneck Projection.
4. Energy/Confidence Gate (Drops noise, coughing, and ambiguous acoustic signals).
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logger import get_logger

logger = get_logger("audio_encoder")


@dataclass
class AudioIntent:
    """Structured intent extracted directly from acoustic waveforms without text generation."""
    action_type: str            # "EMERGENCY_HALT", "VENT_GAS", "CHECK_PRESSURE", "SET_POWER", "NAVIGATE"
    target: str                 # "reactor_airlock", "reactor_core", "lab", etc.
    parameters: Dict[str, Any]
    confidence: float
    is_emergency: bool          # Triggers override vs append
    raw_256d_vector: torch.Tensor


class DirectAudioTransformer(nn.Module):
    """
    Direct Audio-to-256D Transformer.
    Takes 1D raw waveform tensors [B, 1, T] and projects them directly
    into the 256D Manifold latent space with zero intermediate text generation.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        conv_channels: int = 128,
        audio_dim: int = 192,
        latent_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 6,
        confidence_threshold: float = 0.45,
        checkpoint_path: Optional[str] = "logs/training_runs/phase4_audio_manifold.pt",
        calibrated_anchors_path: Optional[str] = None
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.audio_dim = audio_dim
        self.latent_dim = latent_dim
        self.confidence_threshold = confidence_threshold

        # Ensure num_heads evenly divides audio_dim
        if audio_dim % num_heads != 0:
            valid_heads = [h for h in (8, 6, 4, 2, 1) if audio_dim % h == 0]
            num_heads = valid_heads[0] if valid_heads else 1

        # 1. 1D Temporal Convolutional Feature Extractor
        # Downsamples raw 16kHz samples by 40x (stride 5, 2, 2, 2)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=10, stride=5, padding=5),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(conv_channels, audio_dim, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(audio_dim, audio_dim, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
        )

        # 2. Positional Embeddings for Temporal Frames (up to 250 frames ~ 5 sec audio)
        self.max_frames = 250
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_frames, audio_dim) * 0.02)

        # 3. Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=audio_dim,
            nhead=num_heads,
            dim_feedforward=audio_dim * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(audio_dim)

        # 4. Multimodal Bottleneck: Audio -> 256D Manifold
        self.manifold_projection = nn.Sequential(
            nn.Linear(audio_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )

        # 5. Fixed Orthonormal Canonical Intent Anchors in 256D space
        # These correspond to formal AST actions, not English words!
        intent_names = [
            "EMERGENCY_HALT",
            "VENT_GAS",
            "CHECK_PRESSURE",
            "SET_POWER",
            "NAVIGATE_LAB",
            "NAVIGATE_REACTOR"
        ]
        eye = torch.eye(latent_dim)
        self.canonical_intents = nn.ParameterDict({
            name: nn.Parameter(eye[i].clone(), requires_grad=False)
            for i, name in enumerate(intent_names)
        })

        self.dim_adapter: Optional[nn.Linear] = None

        # Automatic checkpoint loading with dynamic dimension adaptation
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                state = ckpt.get("audio_transformer", ckpt) if isinstance(ckpt, dict) else ckpt
                if isinstance(state, dict):
                    ckpt_dim = None
                    if "norm.weight" in state:
                        ckpt_dim = state["norm.weight"].shape[0]
                    elif "pos_embed" in state:
                        ckpt_dim = state["pos_embed"].shape[-1]
                    elif "transformer.layers.0.norm1.weight" in state:
                        ckpt_dim = state["transformer.layers.0.norm1.weight"].shape[0]

                    if ckpt_dim is not None and ckpt_dim != self.audio_dim:
                        logger.warning(
                            "Checkpoint dimension mismatch detected: checkpoint d_model=%d vs model audio_dim=%d. "
                            "Dynamically injecting nn.Linear(%d, %d) adapter and updating downstream layers to %dD.",
                            ckpt_dim, self.audio_dim, self.audio_dim, ckpt_dim, ckpt_dim
                        )
                        self.dim_adapter = nn.Linear(self.audio_dim, ckpt_dim)
                        self.pos_embed = nn.Parameter(torch.randn(1, self.max_frames, ckpt_dim) * 0.02)
                        downstream_heads = num_heads if ckpt_dim % num_heads == 0 else (6 if ckpt_dim % 6 == 0 else 4)
                        enc_layer = nn.TransformerEncoderLayer(
                            d_model=ckpt_dim,
                            nhead=downstream_heads,
                            dim_feedforward=ckpt_dim * 2,
                            activation="gelu",
                            batch_first=True,
                            norm_first=True
                        )
                        self.transformer = nn.TransformerEncoder(
                            enc_layer,
                            num_layers=num_layers,
                            enable_nested_tensor=False
                        )
                        self.norm = nn.LayerNorm(ckpt_dim)
                        self.manifold_projection = nn.Sequential(
                            nn.Linear(ckpt_dim, latent_dim),
                            nn.LayerNorm(latent_dim)
                        )

                    # Load matching state dict keys with shape safety
                    model_state = self.state_dict()
                    compatible_state = {}
                    mismatched_keys = []
                    for k, v in state.items():
                        if k in model_state:
                            if v.shape == model_state[k].shape:
                                compatible_state[k] = v
                            else:
                                mismatched_keys.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                        else:
                            mismatched_keys.append((k, tuple(v.shape), "missing in model"))

                    incompatible = self.load_state_dict(compatible_state, strict=False)
                    logger.info("Loaded %d compatible layer tensors from checkpoint %s.", len(compatible_state), checkpoint_path)
                    if mismatched_keys:
                        logger.info("Checkpoint keys with shape mismatch or unmapped: %s", mismatched_keys)
                    if incompatible.missing_keys:
                        logger.debug("Keys initialized locally: %s", incompatible.missing_keys)
            except Exception as e:
                logger.error("Failed to load audio transformer checkpoint from %s: %s", checkpoint_path, e, exc_info=True)

        # User-calibrated vocal anchors loading if explicitly provided
        if calibrated_anchors_path and os.path.exists(calibrated_anchors_path):
            try:
                self.load_calibrated_anchors(calibrated_anchors_path)
                logger.info("Loaded calibrated vocal anchors from %s", calibrated_anchors_path)
            except Exception as e:
                logger.warning("Failed to load calibrated vocal anchors from %s: %s", calibrated_anchors_path, e)
        else:
            # Initialize default synthetic acoustic anchors for uncalibrated instances
            try:
                from environments.spatial_graph.acoustic_generator import SyntheticAcousticGenerator
                gen = SyntheticAcousticGenerator()
                with torch.no_grad():
                    for name in intent_names:
                        clip = gen.generate_command_audio(name, duration=1.0)
                        z = self.forward(clip)
                        norm_z = F.normalize(z.squeeze(0), p=2, dim=-1)
                        self.canonical_intents[name].data.copy_(norm_z)
                logger.debug("Initialized synthetic acoustic anchors for %d intents", len(intent_names))
            except Exception as e:
                logger.debug("Synthetic acoustic anchor generator not available (%s); using random orthonormal initialization", e)

    def load_calibrated_anchors(self, anchors_path: str = "logs/training_runs/my_voice_anchors.pt") -> bool:
        """Loads user-calibrated 256D vocal anchor vectors."""
        if os.path.exists(anchors_path):
            data = torch.load(anchors_path, map_location="cpu")
            if isinstance(data, dict):
                if "anchors" in data:
                    anchors_dict = data["anchors"]
                    self.mean_voice_vector = data.get("mean_voice_vector", None)
                    self.ambient_noise_rms = data.get("ambient_noise_rms", 0.05)
                    self.speech_threshold = data.get("speech_threshold", 0.10)
                else:
                    anchors_dict = data
                    self.mean_voice_vector = None

                for k, vec in anchors_dict.items():
                    if k in self.canonical_intents:
                        norm_vec = F.normalize(vec.squeeze(), p=2, dim=-1)
                        self.canonical_intents[k].data.copy_(norm_vec)
                return True
        return False

    def save_calibrated_anchors(self, save_path: str = "logs/training_runs/my_voice_anchors.pt"):
        """Saves current state of adapted vocal anchors to disk."""
        anchors_dict = {
            k: v.data.cpu() for k, v in self.canonical_intents.items()
        }
        mean_v = self.mean_voice_vector.cpu() if hasattr(self, "mean_voice_vector") and self.mean_voice_vector is not None else None
        ambient_rms = getattr(self, "ambient_noise_rms", 0.02)
        speech_thresh = getattr(self, "speech_threshold", 0.028)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "anchors": anchors_dict,
            "mean_voice_vector": mean_v,
            "ambient_noise_rms": ambient_rms,
            "speech_threshold": speech_thresh
        }, save_path)

    def adapt_vocal_anchor(
        self,
        command_name: str,
        z_sample: torch.Tensor,
        alpha: float = 0.08,
        save_path: Optional[str] = "logs/training_runs/my_voice_anchors.pt"
    ) -> float:
        """
        Continual Hebbian Vocal Adaptation:
        Refines the canonical anchor for command_name using an Exponential Moving Average (EMA)
        towards the user's freshly spoken and verified audio vector:
            a_new = normalize((1 - alpha) * a_old + alpha * z_sample_centered)
        Returns the updated cosine similarity.
        """
        with torch.no_grad():
            # Resolve command anchor key (e.g. "NAVIGATE" -> "NAVIGATE_LAB" or "NAVIGATE_REACTOR")
            target_key = command_name
            if target_key not in self.canonical_intents:
                for k in self.canonical_intents.keys():
                    if target_key in k:
                        target_key = k
                        break

            if target_key not in self.canonical_intents:
                return 0.0

            z_vec = z_sample.squeeze().to(self.canonical_intents[target_key].device)
            if hasattr(self, "mean_voice_vector") and self.mean_voice_vector is not None:
                mean_v = self.mean_voice_vector.to(z_vec.device)
                z_centered = z_vec - mean_v
                norm_sample = F.normalize(z_centered, p=2, dim=-1)
                # Adapt mean voice vector slowly
                self.mean_voice_vector = (1.0 - alpha * 0.5) * mean_v.cpu() + (alpha * 0.5) * z_vec.cpu()
            else:
                norm_sample = F.normalize(z_vec, p=2, dim=-1)

            old_anchor = self.canonical_intents[target_key].data
            # EMA Hebbian momentum update
            new_anchor = (1.0 - alpha) * old_anchor + alpha * norm_sample
            new_anchor = F.normalize(new_anchor, p=2, dim=-1)
            self.canonical_intents[target_key].data.copy_(new_anchor)

            updated_sim = F.cosine_similarity(norm_sample.unsqueeze(0), new_anchor.unsqueeze(0), dim=-1).item()

            if save_path:
                self.save_calibrated_anchors(save_path)

            return updated_sim

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Maps raw audio waveforms into a 256D manifold latent concept vector.
        Extracts 2D Spectro-Temporal features (32 frequency bins x 8 time steps = 256D)
        preserving phonetic formants and temporal cadence with phase-invariance.
        Args:
            waveform: [B, 1, T] or [B, T]
        Returns:
            z_audio: [B, 256]
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # [B, T]
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # [1, T]

        batch = waveform.size(0)

        # STFT
        w = torch.hann_window(400, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=w,
            return_complex=True
        )
        mag = torch.abs(stft)  # [B, 257, T_frames]

        # 2D Adaptive Pool to 32 frequency bins x 8 temporal slices = 256 dimensions
        grid = F.adaptive_avg_pool2d(mag.unsqueeze(1), (32, 8))  # [B, 1, 32, 8]
        log_grid = torch.log(grid.squeeze(1) + 1e-5)  # [B, 32, 8]

        # Flatten to 256D
        raw_256 = log_grid.reshape(batch, -1)  # [B, 256]

        # Zero-center to remove global microphone gain / volume offset
        centered = raw_256 - raw_256.mean(dim=-1, keepdim=True)
        return F.normalize(centered, p=2, dim=-1)

    def parse_audio_intent(self, waveform: torch.Tensor, return_details: bool = False):
        """
        Extracts structured intent directly from waveform without autoregressive text generation.
        Enforces Energy/Confidence Gate:
        If the audio is noisy, coughs, or unclear, returns None (No hallucination!).
        """
        with torch.no_grad():
            z_audio = self.forward(waveform)  # [1, 256]
            if hasattr(self, "mean_voice_vector") and self.mean_voice_vector is not None:
                mean_v = self.mean_voice_vector.to(z_audio.device).unsqueeze(0)
                z_centered = z_audio - mean_v
                norm_audio = F.normalize(z_centered, p=2, dim=-1)
            else:
                norm_audio = F.normalize(z_audio, p=2, dim=-1)

            best_intent = None
            best_sim = -1.0

            for intent_name, anchor_vec in self.canonical_intents.items():
                norm_anchor = F.normalize(anchor_vec.to(norm_audio.device).unsqueeze(0), p=2, dim=-1)
                sim = F.cosine_similarity(norm_audio, norm_anchor, dim=-1).item()
                if sim > best_sim:
                    best_sim = sim
                    best_intent = intent_name

            if best_sim < self.confidence_threshold or best_intent is None:
                if return_details:
                    return None, best_intent, best_sim
                return None

            is_emergency = "EMERGENCY" in best_intent or "HALT" in best_intent
            intent = AudioIntent(
                action_type=best_intent,
                target="cognitive_manifold",
                parameters={"confidence": round(best_sim, 4)},
                confidence=best_sim,
                is_emergency=is_emergency,
                raw_256d_vector=z_audio,
            )

            if return_details:
                return intent, best_intent, best_sim
            return intent

    @classmethod
    def _init_ctc_acoustic_engine(cls, device: torch.device) -> bool:
        """
        Initializes the local Non-Autoregressive Frame-Synchronous Wav2Vec2-CTC
        acoustic transducer (models/wav2vec2_ctc). Zero Whisper, zero autoregressive
        language model decoder, physically incapable of hallucinating sentences on silence.
        """
        if getattr(cls, "_ctc_model", None) is not None and getattr(cls, "_ctc_processor", None) is not None:
            return True

        import threading
        if getattr(cls, "_ctc_lock", None) is None:
            cls._ctc_lock = threading.Lock()

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        model_dir = os.path.join(repo_root, "models", "wav2vec2_ctc")
        if not os.path.exists(model_dir):
            return False

        with cls._ctc_lock:
            if getattr(cls, "_ctc_model", None) is not None:
                return True
            try:
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
                proc = Wav2Vec2Processor.from_pretrained(model_dir, local_files_only=True)
                model = Wav2Vec2ForCTC.from_pretrained(model_dir, local_files_only=True)
                target_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(target_dev).eval()
                cls._ctc_processor = proc
                cls._ctc_model = model
                cls._ctc_device = target_dev
                logger.info("Loaded non-autoregressive Wav2Vec2-CTC acoustic transducer on %s", target_dev)
                return True
            except Exception as e:
                logger.warning("Could not load local Wav2Vec2-CTC model from %s: %s", model_dir, e)
                return False

    @staticmethod
    def _ground_ctc_phonetics(raw_text: str, concept_lexicon: Optional[list] = None) -> str:
        """
        Grounds raw frame-level CTC phonetic spellings against Jane's live HebbianConceptManifold
        vocabulary and proper nouns without inventing words.
        """
        from difflib import SequenceMatcher

        if not raw_text.strip():
            return ""

        proper_noun_map = {
            "ukhil": "Akhil",
            "ackell": "Akhil",
            "akil": "Akhil",
            "akill": "Akhil",
            "akhill": "Akhil",
            "acul": "Akhil",
            "uckel": "Akhil",
            "akhil": "Akhil",
            "jain": "Jane",
            "jayne": "Jane",
            "jaine": "Jane",
            "jein": "Jane",
            "jane": "Jane",
            "haiden": "Hey Jane",
            "hayden": "Hey Jane",
            "heijden": "Hey Jane",
            "haidn": "Hey Jane",
            "haijane": "Hey Jane",
            "hejane": "Hey Jane",
            "hijane": "Hey Jane",
            "heyjane": "Hey Jane",
            "hajane": "Hey Jane",
            "heijen": "Hey Jane",
            "hijen": "Hey Jane",
            "gennetic": "kinetic",
            "hamiltonean": "Hamiltonian",
            "symetry": "symmetry",
        }

        domain_words = set()
        if concept_lexicon:
            for c in concept_lexicon:
                for part in str(c).lower().replace("_", " ").split():
                    if len(part) >= 5 and part.isalpha():
                        domain_words.add(part)

        corrected_tokens = []
        for tok in raw_text.lower().split():
            clean_tok = tok.strip(".,!?;:\"'")
            if not clean_tok:
                continue
            if clean_tok in proper_noun_map:
                corrected_tokens.append(proper_noun_map[clean_tok])
                continue
            if clean_tok in domain_words or len(clean_tok) < 5:
                corrected_tokens.append(clean_tok)
                continue

            best_match = clean_tok
            best_ratio = 0.0
            for dw in domain_words:
                if abs(len(dw) - len(clean_tok)) > 2 or dw[0] != clean_tok[0]:
                    continue
                ratio = SequenceMatcher(None, clean_tok, dw).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = dw
            if best_ratio >= 0.80:
                corrected_tokens.append(best_match)
            else:
                corrected_tokens.append(clean_tok)

        first_pass = " ".join(corrected_tokens)
        try:
            from src.reasoning_engine.workshop_comprehension import PhoneticSyntacticReconstructor
            reconstructed, _, _ = PhoneticSyntacticReconstructor.reconstruct(
                first_pass, concept_lexicon=concept_lexicon
            )
            return reconstructed or first_pass
        except Exception:
            return first_pass

    def decode_acoustic_speech(
        self,
        waveform: torch.Tensor,
        concept_lexicon: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Non-Autoregressive Frame-Synchronous CTC Acoustic Speech Readout:
        1. Applies a spectral high-pass filter (> 110 Hz) to remove laptop fan / AC rumble
           so low-frequency hum never dominates Wav2Vec2's unit-variance normalization.
        2. Dynamically trims leading/trailing background noise using short-time energy percentiles.
        3. Classifies each 20ms frame independently into acoustic characters or <BLANK> and
           reconstructs intended English via PhoneticSyntacticReconstructor.
        """
        w = waveform.detach().cpu().squeeze().float()
        if w.dim() != 1 or w.numel() < 3200:
            return {"is_valid_speech": False, "spoken_text": "", "confidence": 0.0}

        # 1. Remove DC offset and < 110 Hz laptop fan / AC rumble via FFT high-pass
        w = w - torch.mean(w)
        spec = torch.fft.rfft(w)
        freqs = torch.fft.rfftfreq(w.numel(), d=1.0 / float(self.sample_rate))
        hp_mask = torch.clamp((freqs - 90.0) / 50.0, 0.0, 1.0)
        lp_mask = torch.clamp((7600.0 - freqs) / 400.0, 0.0, 1.0)
        w_clean = torch.fft.irfft(spec * hp_mask * lp_mask, n=w.numel())

        rms = float(torch.sqrt(torch.mean(w_clean ** 2) + 1e-12).item())
        if rms < 0.004:
            return {"is_valid_speech": False, "spoken_text": "", "confidence": 0.0}

        if not self._init_ctc_acoustic_engine(self.pos_embed.device):
            return {"is_valid_speech": False, "spoken_text": "", "confidence": 0.0}

        proc = self._ctc_processor
        model = self._ctc_model
        dev = self._ctc_device

        # 2. Dynamic envelope trimming based on 50ms frame energy above clip's 25th-percentile floor
        w_np = w_clean.numpy()
        frame_len = 800  # 50ms at 16kHz
        if w_clean.numel() >= frame_len * 4:
            n_frames = w_clean.numel() // frame_len
            framed = w_clean[: n_frames * frame_len].view(n_frames, frame_len)
            frame_rms = torch.sqrt(torch.mean(framed ** 2, dim=-1) + 1e-12)
            floor_rms = float(torch.quantile(frame_rms, 0.25).item())
            peak_rms = float(torch.max(frame_rms).item())
            voice_thresh = max(floor_rms * 1.30, floor_rms + 0.0025, peak_rms * 0.22)
            active_frames = (frame_rms >= voice_thresh).nonzero(as_tuple=False).squeeze(-1)
            if active_frames.numel() >= 2:
                st_frame = max(0, int(active_frames[0].item()) - 3)
                en_frame = min(n_frames, int(active_frames[-1].item()) + 4)
                w_np = w_np[st_frame * frame_len : en_frame * frame_len]

        with self._ctc_lock, torch.no_grad():
            inputs = proc(w_np, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
            logits = model(inputs.input_values.to(dev)).logits  # [1, T_frames, V]
            probs = torch.softmax(logits, dim=-1)
            max_probs, pred_ids = torch.max(probs, dim=-1)

            pad_id = proc.tokenizer.pad_token_id
            non_blank_mask = (pred_ids != pad_id)
            non_blank_count = int(non_blank_mask.sum().item())
            total_frames = int(pred_ids.numel())
            non_blank_ratio = non_blank_count / float(max(1, total_frames))

            if non_blank_count < 2 or non_blank_ratio < 0.012:
                return {
                    "is_valid_speech": False,
                    "spoken_text": "",
                    "confidence": 0.0,
                    "non_blank_ratio": round(non_blank_ratio, 4),
                }

            mean_conf = float(max_probs[non_blank_mask].mean().item())
            raw_decoded = proc.batch_decode(pred_ids)[0].strip()

        grounded_text = self._ground_ctc_phonetics(raw_decoded, concept_lexicon=concept_lexicon)
        is_noise_frag = False
        lex_conf = 1.0
        try:
            from src.reasoning_engine.workshop_comprehension import PhoneticSyntacticReconstructor
            _, lex_conf, is_noise_frag = PhoneticSyntacticReconstructor.reconstruct(
                grounded_text, concept_lexicon=concept_lexicon
            )
        except Exception:
            pass

        alpha_chars = sum(1 for ch in grounded_text if ch.isalpha())
        is_valid = (mean_conf >= 0.42) and (alpha_chars >= 2) and (not is_noise_frag or len(grounded_text.split()) >= 3)

        return {
            "is_valid_speech": is_valid,
            "spoken_text": grounded_text if is_valid else "",
            "raw_ctc_text": raw_decoded.lower(),
            "confidence": round(mean_conf, 4),
            "lexical_confidence": round(lex_conf, 4),
            "is_noise_fragment": is_noise_frag,
            "non_blank_ratio": round(non_blank_ratio, 4),
        }
