"""
Phantom Conductor — Audio Analysis (BPM Detection)
===================================================
Reads raw microphone samples from the shared ring buffer and writes
a smoothed, octave-corrected BPM estimate back to PhantomState.

DSP chain  (CFG.use_onset_net = False, default)
---------
  raw samples
    → bandpass filter (40–3000 Hz)
    → HPSS (percussive component)
    → onset strength (median aggregate)
    → full autocorrelation
    → parabolic peak interpolation
    → octave correction  (½× / 1× / 2× closest to previous)
    → exponential smoothing (α = CFG.smooth_alpha)
    → STATE.apply_audio_bpm()

OnsetNet chain  (CFG.use_onset_net = True, requires models/onset_net_v1.pt)
-------------
  raw samples
    → log-mel spectrogram (n_mels=64, hop=512)
    → rolling frame buffer
    → OnsetNet CNN (causal, per-frame onset probability)
    → peak-pick  (threshold = CFG.onset_threshold,
                  min gap   = CFG.onset_min_gap_sec)
    → STATE.recent_beat_times  (PLL input)
    → BPM derived from inter-onset intervals
    → exponential smoothing (α = CFG.smooth_alpha)
    → STATE.apply_audio_bpm()

Changes from v0.5.0
--------------------
* MIN_BPM, MAX_BPM, SMOOTH_ALPHA, ANALYZE_EVERY now read from CFG so
  that the Settings panel can override them at runtime.

Changes from v0.5.2 (tempo tapper)
------------------------------------
* bpm_analysis_thread now checks CFG.use_tempo_tapper on every pass
  (not just at startup) and skips writing audio-derived BPM entirely
  while tapper mode is enabled. Previously the only thing preventing
  audio from overwriting a tap was the 4s override window in
  state.apply_audio_bpm(), so the mic would silently take back over
  a few seconds after every tap. Checking the live config flag here
  means flipping the Settings checkbox takes effect immediately,
  in both directions, with no thread restart needed.
* Still calls estimate_bpm() even while gated off, so bpm_analysis_dbg
  stays fresh for diagnostics — it just doesn't push the result into
  state when tapper mode owns the BPM.

Changes from v0.5.3 (OnsetNet)
--------------------------------
* bpm_analysis_thread gains a second mode toggled by CFG.use_onset_net:
  - False (default): existing DSP chain, unchanged.
  - True: OnsetNet CNN inference. Onset timestamps → state.recent_beat_times
    for the PLL; BPM is derived from inter-onset intervals and EMA-smoothed
    with the same smooth_alpha.
* ONSET_NET_AVAILABLE (module-level bool) is set once at import time.
  If models/onset_net_v1.pt is absent or PyTorch is missing, it stays
  False and CFG.use_onset_net has no effect (DSP always used).
* STATE.onset_net_available mirrors this flag so the UI can hide the
  checkbox when no checkpoint is present.
* While CFG.use_tempo_tapper is True, audio-derived BPM is suppressed
  in both modes (same gate as before). STATE.recent_beat_times is still
  updated by OnsetNet even in tapper mode so the PLL can run independently.

OnsetNet architecture (training/onset_network.py)
--------------------------------------------------
* Input:  (batch, n_mels, time) — log-mel spectrogram, no context window
* Output: (batch, time)          — per-frame onset logits (sigmoid outside)
* Stack:  4× CausalConv1d (dilations 1/2/4/8, kernel 5) + BN + ReLU,
          then a 1×1 conv head. Causality via left-padding — no future
          frames are ever seen, so the same forward() works at inference.
* Checkpoint format: dict with keys model_state_dict, epoch, f1, config
  (config carries n_mels and hidden so we reconstruct the right shape).
"""

import collections
import struct
import time
import wave
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, lfilter

from buffers import SR, audio_buffer
from config import CFG
from logger import Logger
from phase_lock import PhaseLock
from state import PhantomState

_recording_buffer: list[np.ndarray] = []
_recording_active: bool = False
_pll = PhaseLock()


def start_cnn_recording():
    global _recording_active, _recording_buffer
    _recording_buffer = []
    _recording_active = True


def stop_cnn_recording(path: str = "tmp.wav"):
    global _recording_active
    _recording_active = False
    if not _recording_buffer:
        return
    audio = np.concatenate(_recording_buffer)  # float32 at _NN_SR after resample
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_NN_SR)
        wf.writeframes(audio_int16.tobytes())
    print(f"[cnn_record] saved {len(audio) // _NN_SR}s to {path}")


# ── Constants ─────────────────────────────────────────────────────────────────
HOP_LENGTH = 256
BANDPASS = (40, 3000)

# OnsetNet spectrogram constants — must match training/onset_dataset.py
# n_mels and hidden are read back from the checkpoint config dict, so
# these are just the defaults used when the checkpoint predates the config key.
_N_MELS_DEFAULT = 80  # OnsetNet(n_mels=80, hidden=32) from train_onset.py
_NN_HOP_LENGTH = 256  # match onset_dataset.HOP_LENGTH exactly
_NN_SR = 22050  # training SR
_NN_N_FFT = 1024  # match onset_dataset n_fft

_CHECKPOINT_PATH = Path(__file__).parent / "models" / "onset_net_v1.pt"


# ═══════════════════════════════════════════════════════════════════════════════
#  DSP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _butter_bandpass(lo: float, hi: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, [max(1e-3, lo / nyq), min(0.999, hi / nyq)], btype="band")
    return b, a


def _apply_bandpass(y: np.ndarray, fs: int, lo: float, hi: float) -> np.ndarray:
    b, a = _butter_bandpass(lo, hi, fs)
    return lfilter(b, a, y)


def _pick_peak(ac: np.ndarray, lag_min: int, lag_max: int) -> float | None:
    """Parabolic interpolation around the strongest autocorrelation peak."""
    seg = ac[lag_min : lag_max + 1]
    if not seg.size:
        return None
    i = lag_min + int(np.argmax(seg))
    if 0 < i < len(ac) - 1:
        y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
        d = y0 - 2 * y1 + y2
        if d:
            return i + 0.5 * (y0 - y2) / d
    return float(i)


def _octave_correct(bpm: float, prev: float | None) -> float:
    if prev is None or not np.isfinite(prev):
        return bpm
    return min([bpm / 2, bpm, bpm * 2], key=lambda x: abs(x - prev))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN DSP ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════


def estimate_bpm(
    y: np.ndarray, sr: int = SR, bpm_prev: float | None = None
) -> tuple[float | None, dict]:
    """
    Estimate BPM from a buffer of audio samples.

    Returns
    -------
    (bpm_smoothed, debug_dict)
    bpm_smoothed is None when the buffer is too short or too quiet.
    """
    min_bpm = CFG.min_bpm
    max_bpm = CFG.max_bpm
    smooth_alpha = CFG.smooth_alpha

    if len(y) < sr * 2:
        return None, {"msg": "buffer too short"}

    y = librosa.util.normalize(y.astype(np.float32))
    y = _apply_bandpass(y, sr, *BANDPASS)

    _, y_p = librosa.effects.hpss(y)

    onset = librosa.onset.onset_strength(
        y=y_p, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median
    )

    if onset.size < 8 or np.max(onset) < 1e-3:
        return None, {"msg": "weak onset"}

    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1 :]
    lag_min = max(2, int(np.floor(60 * sr / (max_bpm * HOP_LENGTH))))
    lag_max = min(int(np.ceil(60 * sr / (min_bpm * HOP_LENGTH))), len(ac) - 1)

    if lag_min >= lag_max:
        return None, {"msg": "invalid range"}

    lag = _pick_peak(ac, lag_min, lag_max)
    if lag is None or not np.isfinite(lag) or lag <= 0:
        return None, {"msg": "invalid lag"}

    bpm_raw = 60.0 * sr / (HOP_LENGTH * lag)
    bpm_corr = float(np.clip(_octave_correct(bpm_raw, bpm_prev), min_bpm, max_bpm))
    if bpm_prev is None or not np.isfinite(bpm_prev):
        bpm_s = bpm_corr
    elif abs(bpm_corr - bpm_prev) < 5.0:
        bpm_s = smooth_alpha * bpm_corr + (1 - smooth_alpha) * bpm_prev
    else:
        bpm_s = bpm_corr

    return bpm_s, {
        "mode": "dsp",
        "bpm_raw": float(bpm_raw),
        "bpm_corr": bpm_corr,
        "onset_max": float(np.max(onset)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ONSET NET — model definition, load, inference
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture: training/onset_network.py
#   Input  (batch, n_mels, time) — log-mel spectrogram
#   Output (batch, time)          — per-frame onset logits (sigmoid outside)
#   Stack: 4× CausalConv1d (k=5, dilations 1/2/4/8) + BN + ReLU, 1×1 head
#
# Checkpoint format (train_onset.py):
#   { model_state_dict, epoch, f1, config: {n_mels, hidden} }


def _build_onset_net(n_mels: int = 80, hidden: int = 32):
    """
    Reconstruct OnsetNet from training/onset_network.py.
    n_mels and hidden are read from the checkpoint config dict so the
    shape always matches the saved weights.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
            super().__init__()
            self.pad = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_ch, out_ch, kernel_size, dilation=dilation, padding=0
            )

        def forward(self, x):  # x: (batch, ch, time)
            return self.conv(F.pad(x, (self.pad, 0)))

    class OnsetNet(nn.Module):
        def __init__(self, n_mels: int, hidden: int):
            super().__init__()
            self.net = nn.Sequential(
                CausalConv1d(n_mels, hidden, kernel_size=5, dilation=1),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                CausalConv1d(hidden, hidden, kernel_size=5, dilation=2),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                CausalConv1d(hidden, hidden, kernel_size=5, dilation=4),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                CausalConv1d(hidden, hidden, kernel_size=5, dilation=8),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            )
            self.head = nn.Conv1d(hidden, 1, kernel_size=1)

        def forward(self, mel):  # mel: (batch, n_mels, time)
            return self.head(self.net(mel)).squeeze(1)  # (batch, time) logits

    return OnsetNet(n_mels=n_mels, hidden=hidden)


# ── Load checkpoint once at import time ──────────────────────────────────────

_onset_net = None
_onset_n_mels = _N_MELS_DEFAULT
ONSET_NET_AVAILABLE = False

try:
    import torch as _torch

    if _CHECKPOINT_PATH.exists():
        _ckpt = _torch.load(_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        _cfg = _ckpt.get("config", {})
        _onset_n_mels = _cfg.get("n_mels", _N_MELS_DEFAULT)
        _hidden = _cfg.get("hidden", 32)

        _net = _build_onset_net(n_mels=_onset_n_mels, hidden=_hidden)
        _net.load_state_dict(_ckpt["model_state_dict"])
        _net.eval()
        _onset_net = _net
        ONSET_NET_AVAILABLE = True
        print(
            f"[audio_analysis] OnsetNet loaded — "
            f"n_mels={_onset_n_mels}, hidden={_hidden}, "
            f"epoch={_ckpt.get('epoch', '?')}, F1={_ckpt.get('f1', 0):.3f}"
        )
    else:
        print(f"[audio_analysis] {_CHECKPOINT_PATH} not found — DSP fallback active")
except ImportError:
    print("[audio_analysis] PyTorch not installed — DSP fallback active")
except Exception as _e:
    print(f"[audio_analysis] OnsetNet load failed ({_e}) — DSP fallback active")


# ── Rolling inference state ───────────────────────────────────────────────────
# Only written by bpm_analysis_thread (single daemon thread) — no locking needed.

_onset_prob_history: collections.deque = collections.deque(maxlen=256)


def _audio_to_mel(y: np.ndarray, sr: int, n_mels: int) -> np.ndarray:
    if sr != _NN_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_NN_SR)
    if _recording_active:
        _recording_buffer.append(y.copy())
    mel = librosa.feature.melspectrogram(
        y=y.astype(np.float32),
        sr=_NN_SR,
        n_mels=n_mels,
        hop_length=_NN_HOP_LENGTH,
        n_fft=_NN_N_FFT,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    mean, std = log_mel.mean(), log_mel.std() + 1e-8
    return ((log_mel - mean) / std).astype(np.float32)


def _run_onset_net(y, sr, now, logger=None) -> tuple[list[float], dict]:
    """
    Run OnsetNet on the current audio buffer and return
    (onset_timestamps, debug_dict).

    The model is fully causal (CausalConv1d with left-padding only), so we
    feed the whole buffer as a single sequence — (1, n_mels, T) — and read
    the per-frame logit curve out the other end.  No context-window batching
    needed; causality is enforced by the architecture.

    onset_timestamps are absolute monotonic wall-clock seconds.
    """
    rms = float(np.sqrt(np.mean(y**2)))
    silence_threshold = CFG.get("onset_silence_threshold", 0.01)
    if rms < silence_threshold:
        if logger:
            logger.info(f"[onset] silence detected (rms={rms:.4f}) — skipping")
        return [], {"mode": "onset_net", "msg": "silence", "rms": rms}
    import torch

    start_cnn_recording()

    mel = _audio_to_mel(y, sr, _onset_n_mels)  # (n_mels, T)
    T = mel.shape[1]
    if T == 0:
        return [], {"msg": "empty mel"}

    frame_dur = _NN_HOP_LENGTH / _NN_SR  # now 256/22050 ≈ 11.6ms

    # Forward pass — single sequence, batch size 1
    mel_t = torch.from_numpy(mel[np.newaxis])  # (1, n_mels, T)
    with torch.no_grad():
        logits = _onset_net(mel_t)  # (1, T)
        probs = torch.sigmoid(logits)[0].cpu().numpy()  # (T,)

    for p in probs:
        _onset_prob_history.append(float(p))

    # Peak-pick (mirrors train_onset.peak_pick logic)
    threshold = CFG.get("onset_threshold", 0.5)
    min_gap_sec = CFG.get("onset_min_gap_sec", 0.10)
    min_gap_frames = max(1, int(min_gap_sec / frame_dur))

    detected: list[float] = []
    last_peak = -min_gap_frames
    for i in range(1, T - 1):
        if (
            probs[i] > threshold
            and probs[i] >= probs[i - 1]
            and probs[i] >= probs[i + 1]
            and i - last_peak >= min_gap_frames
        ):
            # Map frame index to wall-clock time.
            # Frame 0 = start of this buffer; frame T-1 = now.
            frames_from_end = T - 1 - i
            detected.append(now - frames_from_end * frame_dur)
            last_peak = i

    prob_mean = (
        float(np.mean(list(_onset_prob_history)[-20:])) if _onset_prob_history else 0.0
    )

    if logger:
        logger.info(
            f"[onset] buf={len(y)} T={T} frames | "
            f"detected={len(detected)} prob_max={float(probs.max()):.3f} "
            f"prob_mean={prob_mean:.3f} threshold={CFG.get('onset_threshold', 0.5):.2f} "
            f"sr={sr} n_mels={_onset_n_mels} hop={_NN_HOP_LENGTH}"
        )
    return detected, {
        "mode": "onset_net",
        "onsets_this_pass": len(detected),
        "onset_prob_mean": prob_mean,
    }


def _bpm_from_onsets(
    timestamps: list[float], min_bpm: float, max_bpm: float
) -> float | None:
    if len(timestamps) < 4:
        return None
    recent = sorted(timestamps)[-32:]  # wider window
    iois = np.diff(recent)
    iois = iois[
        (iois > 60.0 / max_bpm) & (iois < 60.0 / min_bpm)
    ]  # filter impossible IOIs immediately
    if len(iois) < 3:
        return None
    # Also check double/half IOI candidates — beat vs subdivision problem
    bpm_candidates = 60.0 / iois
    # Histogram vote — bin into 1-BPM buckets and take the peak
    hist, edges = np.histogram(
        bpm_candidates, bins=np.arange(min_bpm, max_bpm + 1, 1.0)
    )
    if hist.max() == 0:
        return None
    return float(edges[np.argmax(hist)] + 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS THREAD
# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS THREAD
# ═══════════════════════════════════════════════════════════════════════════════


def bpm_analysis_thread(state: PhantomState, logger: Logger):
    """
    Runs in a daemon thread.  Every CFG.analyze_every seconds it copies the
    current contents of audio_buffer and runs either the DSP chain or the
    OnsetNet CNN, depending on CFG.use_onset_net.

    Both modes
    ----------
    * CFG.use_tempo_tapper is checked live every pass; while True, neither
      mode writes audio-derived BPM (tapper owns state.bpm_live).
    * state.last_bpm_analysis_dbg is always kept fresh for the HUD/debug
      panel, regardless of tapper mode or which detection mode is active.

    OnsetNet mode additionally
    --------------------------
    * Detected onset timestamps are appended to state.recent_beat_times
      for the PLL, even while tapper mode is active.
    * CFG.use_onset_net is ignored (DSP always used) when
      ONSET_NET_AVAILABLE is False.
    """
    # Expose checkpoint availability to the UI so it can hide the checkbox.
    state.onset_net_available = ONSET_NET_AVAILABLE

    last = 0.0
    tapper_was_active = False
    recent_onset_times: collections.deque = collections.deque(maxlen=16)
    bpm_history: collections.deque = collections.deque(
        maxlen=CFG.get("bpm_smooth_window", 5)
    )

    while state.alive():
        analyze_every = CFG.analyze_every
        now = time.time()
        # reset PLL on track change so the grid doesn't drift from a previous song
        if state.load_new_track is not None:
            _pll.reset()

        if now - last >= analyze_every and len(audio_buffer) >= SR * 3:
            last = now
            window_samples = int(3.0 * SR)
            y = np.array(audio_buffer, dtype=np.float32)[-window_samples:]

            tapper_active = CFG.get("use_tempo_tapper", False)
            use_nn = CFG.get("use_onset_net", False) and ONSET_NET_AVAILABLE

            # ── DIAGNOSTIC — remove once working ─────────────────────────────
            logger.info(
                f"[bpm] pass | buf={len(y)} sr={SR} nn={use_nn} tapper={tapper_active} "
                f"bpm_now={state.get_bpm()}"
            )
            # ─────────────────────────────────────────────────────────────────

            # ── tapper-mode transition logging ────────────────────────────────
            if tapper_active and not tapper_was_active:
                logger.info(
                    "bpm-analysis: tempo tapper enabled — audio BPM writes paused"
                )
            elif tapper_was_active and not tapper_active:
                logger.info(
                    "bpm-analysis: tempo tapper disabled — resuming audio BPM writes"
                )
            tapper_was_active = tapper_active

            if use_nn:
                # ── OnsetNet path ─────────────────────────────────────────────
                mono_now = time.monotonic()
                detected, dbg = _run_onset_net(y, SR, time.time(), logger)

                for ts in detected:
                    recent_onset_times.append(ts)
                    state.recent_beat_times.append(ts)
                    pll_correction = _pll.update(
                        beat_time=ts, bpm=state.get_bpm() or 120.0
                    )

                bpm_histogram = _bpm_from_onsets(
                    list(recent_onset_times), CFG.min_bpm, CFG.max_bpm
                )
                prev = state.get_bpm()

                if bpm_histogram is None:
                    state.last_bpm_analysis_dbg = dbg
                else:
                    # EMA smooth the histogram estimate
                    smooth_alpha = CFG.smooth_alpha
                    bpm_s = (
                        smooth_alpha * bpm_histogram + (1 - smooth_alpha) * prev
                        if prev and np.isfinite(prev)
                        else bpm_histogram
                    )
                    

                    bpm_history.append(bpm_s)
                    bpm_s = float(
                        np.median(list(bpm_history))
                    )  # median over recent estimates

                    dbg["bpm_smoothed"] = bpm_s
                    dbg["pll_correction"] = _pll.rate_correction
                    state.last_bpm_analysis_dbg = dbg
                    if not tapper_active:
                        state.apply_audio_bpm(bpm_s)
            else:
                # ── DSP path (original, unchanged) ────────────────────────────
                bpm_new, dbg = estimate_bpm(y, bpm_prev=state.get_bpm())

                if bpm_new and np.isfinite(bpm_new):
                    state.last_bpm_analysis_dbg = dbg  # always kept fresh
                    if not tapper_active:
                        wrote = state.apply_audio_bpm(
                            bpm_new,
                            raw=dbg.get("bpm_raw"),
                            corrected=dbg.get("bpm_corr"),
                            onset_max=dbg.get("onset_max", 0.0),
                        )
                        if wrote:
                            state.last_bpm_dbg = dbg

            with state._lock:
                state.buffer_fill = min(1.0, len(audio_buffer) / (SR * 10))

        time.sleep(analyze_every / 4)
        stop_cnn_recording()
