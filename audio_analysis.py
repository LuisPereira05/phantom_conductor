"""
Phantom Conductor — Audio Analysis (BPM Detection)
===================================================
Reads raw microphone samples from the shared ring buffer and writes
a smoothed, octave-corrected BPM estimate back to PhantomState.

DSP chain
---------
  raw samples
    → RMS gate  (skip if below CFG.rms_threshold, default 0.01)
    → bandpass filter (40–8000 Hz)
    → HPSS (percussive component)
    → onset strength (median aggregate)
    → full autocorrelation  (sliced from centre: ac_full[ac_full.size // 2:])
    → parabolic peak interpolation
    → octave correction  (½× / 1× / 2× closest to bpm_original)
    → median filter over rolling window of raw estimates (spike rejection)
    → STATE.apply_audio_bpm()

Smoothing strategy
------------------
The old exponential smoother blended every new estimate into the previous
value, so a single bad estimate (octave error, autocorrelation mis-peak)
would drag the output for several seconds.  The median filter keeps a
rolling window of the last N raw estimates and outputs their median.  One
bad estimate out of eight is simply outvoted and has zero effect on output.
Window size is CFG.bpm_median_window (default 8).  Larger = more stable
but slower to respond to genuine tempo changes.

Changes from v0.5.x
--------------------
* Bandpass widened 3 kHz → 8 kHz: preserves pick-attack transients
  (3–5 kHz) which carry the clearest rhythmic signal for rhythm guitar.
* Autocorrelation slice uses ac_full[ac_full.size // 2:] — correct for
  even-length onset arrays; the previous [len(onset)-1:] was off-by-one.
* _octave_correct anchors to bpm_original (stable ground truth) instead
  of the previous smoothed estimate (could drift and self-reinforce).
* Exponential smoother replaced by rolling median filter (see above).
* RMS gate: skip analysis entirely when signal is below threshold.
* PLL / beat timestamp machinery removed — tempo-only tracking.
* CFG.use_tempo_tapper gating retained unchanged.
"""

import collections
import time

import librosa
import numpy as np
from scipy.signal import butter, lfilter

from buffers import SR, audio_buffer
from config import CFG
from logger import Logger
from state import PhantomState

# ── Constants ─────────────────────────────────────────────────────────────────
HOP_LENGTH = 256
BANDPASS = (40, 8000)  # widened from 3000 — preserves pick-attack transients


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


def _octave_correct(bpm: float, bpm_original: float | None) -> float:
    """
    Pick the octave of bpm (½×, 1×, 2×) closest to bpm_original.
    bpm_original is the known tempo of the backing track — a stable ground
    truth that never drifts, unlike a previous smoothed estimate would.
    Falls back to returning bpm unchanged if bpm_original is unavailable.
    """
    if bpm_original is None or not np.isfinite(bpm_original) or bpm_original <= 0:
        return bpm
    candidates = [bpm / 2, bpm, bpm * 2]
    return min(candidates, key=lambda x: abs(x - bpm_original))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════


def estimate_bpm(
    y: np.ndarray,
    sr: int = SR,
    bpm_original: float | None = None,
    logger: Logger | None = None,
) -> tuple[float | None, dict]:
    """
    Estimate BPM from a buffer of audio samples.

    Returns
    -------
    (bpm_corr, debug_dict)
    bpm_corr is the single-window octave-corrected estimate before median
    filtering.  Returns None when the buffer is too short, too quiet, or
    the autocorrelation produces an invalid peak.
    Smoothing (median filter) is handled in bpm_analysis_thread so the
    rolling window persists across calls.
    """
    min_bpm = CFG.min_bpm
    max_bpm = CFG.max_bpm

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

    # Slice from centre of full correlogram — correct for even-length arrays.
    ac_full = np.correlate(onset, onset, mode="full")
    ac = ac_full[ac_full.size // 2 :]

    lag_min = max(2, int(np.floor(60.0 * sr / (max_bpm * HOP_LENGTH))))
    lag_max = min(int(np.ceil(60.0 * sr / (min_bpm * HOP_LENGTH))), len(ac) - 1)

    if lag_min >= lag_max:
        return None, {"msg": "invalid range"}

    lag = _pick_peak(ac, lag_min, lag_max)
    if lag is None or not np.isfinite(lag) or lag <= 0:
        return None, {"msg": "invalid lag"}

    bpm_raw = 60.0 * sr / (HOP_LENGTH * lag)
    bpm_corr = float(np.clip(_octave_correct(bpm_raw, bpm_original), min_bpm, max_bpm))

    if logger is not None:
        logger.debug(
            f"bpm_raw={bpm_raw:.1f}  bpm_corr={bpm_corr:.1f}  "
            f"onset_max={float(np.max(onset)):.3f}"
        )

    return bpm_corr, {
        "bpm_raw": float(bpm_raw),
        "bpm_corr": bpm_corr,
        "onset_max": float(np.max(onset)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS THREAD
# ═══════════════════════════════════════════════════════════════════════════════


def bpm_analysis_thread(state: PhantomState, logger: Logger):
    """
    Runs in a daemon thread.  Every CFG.analyze_every seconds it:
      1. Gates on RMS — skips silences entirely.
      2. Calls estimate_bpm() to get a single-window BPM estimate.
      3. Pushes it into a rolling deque and outputs the median — this
         rejects spikes without the lag that exponential smoothing causes.
      4. Writes the result to state via apply_audio_bpm(), unless
         CFG.use_tempo_tapper is True (tap input owns BPM in that mode).

    Median window
    -------------
    CFG.bpm_median_window (default 8) controls the trade-off:
      • Larger → more spike rejection, slower response to real tempo changes.
      • Smaller → faster response, more susceptible to single bad estimates.
    At CFG.analyze_every=0.25s, a window of 8 covers the last 2 seconds.
    """
    last = 0.0
    tapper_was_active = False

    # Rolling window of octave-corrected single-window estimates.
    # Median of this window is what gets written to state.
    bpm_window: collections.deque[float] = collections.deque(
        maxlen=CFG.get("bpm_median_window", 8)
    )

    while state.alive():
        analyze_every = CFG.analyze_every
        now = time.time()

        # Keep window size in sync with live config without recreating deque.
        new_window_size = CFG.get("bpm_median_window", 8)
        if new_window_size != bpm_window.maxlen:
            bpm_window = collections.deque(bpm_window, maxlen=new_window_size)

        if now - last >= analyze_every and len(audio_buffer) >= SR * 2:
            last = now
            y = np.array(audio_buffer, dtype=np.float32)

            # ── RMS gate — skip analysis in silence ───────────────────────────
            rms = float(np.sqrt(np.mean(y**2)))
            rms_threshold = CFG.get("rms_threshold", 0.01)
            if rms < rms_threshold:
                logger.debug(f"bpm-analysis: silent (rms={rms:.4f} < {rms_threshold})")
                with state._lock:
                    state.buffer_fill = min(1.0, len(audio_buffer) / (SR * 10))
                continue

            # ── Estimate ──────────────────────────────────────────────────────
            bpm_new, dbg = estimate_bpm(
                y,
                bpm_original=state.bpm_original,
                logger=logger,
            )

            tapper_active = CFG.get("use_tempo_tapper", False)

            if tapper_active and not tapper_was_active:
                logger.info(
                    "bpm-analysis: tempo tapper enabled — audio BPM writes paused"
                )
            elif tapper_was_active and not tapper_active:
                logger.info(
                    "bpm-analysis: tempo tapper disabled — resuming audio BPM writes"
                )
            tapper_was_active = tapper_active

            if bpm_new is not None and np.isfinite(bpm_new):
                state.last_bpm_analysis_dbg = dbg

                # ── Median filter — spike-resistant smoothing ──────────────────
                # One outlier in a window of 8 is outvoted and has zero effect.
                bpm_window.append(bpm_new)
                bpm_smooth = float(np.median(bpm_window))

                if not tapper_active:
                    wrote = state.apply_audio_bpm(
                        bpm_smooth,
                        raw=dbg.get("bpm_raw"),
                        corrected=dbg.get("bpm_corr"),
                        onset_max=dbg.get("onset_max", 0.0),
                    )
                    if wrote:
                        state.last_bpm_dbg = dbg

            with state._lock:
                state.buffer_fill = min(1.0, len(audio_buffer) / (SR * 10))

        time.sleep(analyze_every / 4)
