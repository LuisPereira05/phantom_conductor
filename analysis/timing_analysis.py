# analysis/timing_analysis.py
import collections
import time

import numpy as np
import torch

from buffers import SR, audio_buffer
from config import CFG
from logger import Logger
from state import PhantomState

from .feature_extractor import TimingFeatureExtractor
from .speed_controller import SpeedController
from .timing_network import LightweightTimingNet, TimingAlignerNet


def timing_analysis_thread(state: PhantomState, logger: Logger):
    """Replaces bpm_analysis_thread. Predicts Δt, not BPM."""

    # ── Load model ─────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use lightweight on CPU
    if device.type == "cpu":
        model = LightweightTimingNet(feature_dim=31, hidden_dim=64)
    else:
        model = TimingAlignerNet(feature_dim=31, hidden_dim=128, num_layers=2)

    model_path = CFG.get("timing_model_path", "./models/timing_aligner_v1.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # ── Init components ────────────────────────────────────────
    extractor = TimingFeatureExtractor(sr=SR)
    controller = SpeedController(
        k_p=CFG.get("speed_k_p", 2.0),
        k_i=CFG.get("speed_k_i", 0.1),
        k_d=CFG.get("speed_k_d", 0.5),
        median_window=CFG.get("delta_median_window", 8),
    )

    live_feature_buffer = collections.deque(maxlen=CFG.get("timing_seq_len", 32))
    last = 0.0
    analyze_every = CFG.get("analyze_every", 0.25)
    seq_len = CFG.get("timing_seq_len", 32)

    logger.info("[timing] Neural timing aligner started")

    while state.alive():
        now = time.time()

        if state.reference_features is None:
            time.sleep(analyze_every)
            continue

        if now - last >= analyze_every and len(audio_buffer) >= SR * 2:
            last = now
            y_live = np.array(audio_buffer, dtype=np.float32)

            # ── RMS gate (kept from your original) ─────────────────
            rms = float(np.sqrt(np.mean(y_live**2)))
            if rms < CFG.get("rms_threshold", 0.01):
                continue

            # ── Extract live features ──────────────────────────────
            live_features = extractor.extract(y_live)  # (n_beats_live, 31)
            for feat in live_features:
                live_feature_buffer.append(feat.numpy())

            if len(live_feature_buffer) < seq_len // 2:
                continue

            # ── Align with reference ───────────────────────────────
            # Find current position in reference
            current_time = state.get_current_playback_time()  # You implement this
            beat_idx = np.searchsorted(state.reference_beat_times, current_time)

            ref_start = max(0, beat_idx - seq_len // 2)
            ref_end = min(len(state.reference_features), ref_start + seq_len)
            ref_window = state.reference_features[ref_start:ref_end]

            # Pad if needed
            if len(ref_window) < seq_len:
                pad = seq_len - len(ref_window)
                ref_window = torch.cat([ref_window, torch.zeros(pad, 31)])

            live_list = list(live_feature_buffer)[-seq_len:]
            if len(live_list) < seq_len:
                pad = seq_len - len(live_list)
                live_list = [np.zeros(31)] * pad + live_list
            live_window = torch.tensor(np.stack(live_list), dtype=torch.float32)

            # ── Neural network prediction ──────────────────────────
            ref_batch = ref_window.unsqueeze(0).to(device)
            live_batch = live_window.unsqueeze(0).to(device)

            with torch.no_grad():
                delta_t_seq = model(ref_batch, live_batch)  # (1, seq)
                delta_t_current = delta_t_seq[0, -1].item()  # Latest beat

            # ── Speed control ──────────────────────────────────────
            playback_speed = controller.update(delta_t_current)
            state.apply_playback_speed(playback_speed, delta_t=delta_t_current)

            logger.debug(
                f"[timing] Δt={delta_t_current * 1000:+.1f}ms  "
                f"speed={playback_speed:.3f}  rms={rms:.4f}"
            )

        time.sleep(analyze_every / 4)
