"""
Phantom Conductor — Audio Analysis (BPM Detection)
===================================================
Reads raw microphone samples from the shared ring buffer and writes
a smoothed, octave-corrected BPM estimate back to PhantomState.

DSP chain
---------
  raw samples
    → bandpass filter (40–3000 Hz)
    → HPSS (percussive component)
    → onset strength (median aggregate)
    → full autocorrelation
    → parabolic peak interpolation
    → octave correction  (½× / 1× / 2× closest to previous)
    → exponential smoothing (α = SMOOTH_ALPHA)
    → STATE.set_bpm()

Nothing in this file touches sounddevice, OpenCV, or the UI.
"""

import time
import numpy as np
import librosa
from scipy.signal import butter, lfilter

from buffers import audio_buffer, SR
from state   import PhantomState
from logger  import Logger

# ── Constants ─────────────────────────────────────────────────────────────────
HOP_LENGTH    = 256
MIN_BPM       = 60
MAX_BPM       = 200
SMOOTH_ALPHA  = 0.3
BANDPASS      = (40, 3000)
ANALYZE_EVERY = 0.25          # seconds between analysis passes


# ═══════════════════════════════════════════════════════════════════════════════
#  DSP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _butter_bandpass(lo: float, hi: float, fs: int, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order,
                  [max(1e-3, lo / nyq), min(0.999, hi / nyq)],
                  btype="band")
    return b, a


def _apply_bandpass(y: np.ndarray, fs: int, lo: float, hi: float) -> np.ndarray:
    b, a = _butter_bandpass(lo, hi, fs)
    return lfilter(b, a, y)


def _pick_peak(ac: np.ndarray, lag_min: int, lag_max: int) -> float | None:
    """Parabolic interpolation around the strongest autocorrelation peak."""
    seg = ac[lag_min:lag_max + 1]
    if not seg.size:
        return None
    i = lag_min + int(np.argmax(seg))
    if 0 < i < len(ac) - 1:
        y0, y1, y2 = ac[i-1], ac[i], ac[i+1]
        d = y0 - 2*y1 + y2
        if d:
            return i + 0.5 * (y0 - y2) / d
    return float(i)


def _octave_correct(bpm: float, prev: float | None) -> float:
    if prev is None or not np.isfinite(prev):
        return bpm
    return min([bpm/2, bpm, bpm*2], key=lambda x: abs(x - prev))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_bpm(y: np.ndarray, sr: int = SR,
                 bpm_prev: float | None = None) -> tuple[float | None, dict]:
    """
    Estimate BPM from a buffer of audio samples.

    Returns
    -------
    (bpm_smoothed, debug_dict)
    bpm_smoothed is None when the buffer is too short or too quiet.
    """
    if len(y) < sr * 2:
        return None, {"msg": "buffer too short"}

    y = librosa.util.normalize(y.astype(np.float32))
    y = _apply_bandpass(y, sr, *BANDPASS)

    _, y_p = librosa.effects.hpss(y)

    onset = librosa.onset.onset_strength(
        y=y_p, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median)

    if onset.size < 8 or np.max(onset) < 1e-3:
        return None, {"msg": "weak onset"}

    ac      = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    lag_min = max(2, int(np.floor(60 * sr / (MAX_BPM * HOP_LENGTH))))
    lag_max = min(int(np.ceil(60 * sr / (MIN_BPM * HOP_LENGTH))), len(ac) - 1)

    if lag_min >= lag_max:
        return None, {"msg": "invalid range"}

    lag = _pick_peak(ac, lag_min, lag_max)
    if lag is None or not np.isfinite(lag) or lag <= 0:
        return None, {"msg": "invalid lag"}

    bpm_raw  = 60.0 * sr / (HOP_LENGTH * lag)
    bpm_corr = float(np.clip(_octave_correct(bpm_raw, bpm_prev), MIN_BPM, MAX_BPM))

    if bpm_prev is None or not np.isfinite(bpm_prev):
        bpm_s = bpm_corr
    elif abs(bpm_corr - bpm_prev) < 5.0:
        bpm_s = SMOOTH_ALPHA * bpm_corr + (1 - SMOOTH_ALPHA) * bpm_prev
    else:
        bpm_s = bpm_corr

    return bpm_s, {
        "bpm_raw":   float(bpm_raw),
        "bpm_corr":  bpm_corr,
        "onset_max": float(np.max(onset)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS THREAD
# ═══════════════════════════════════════════════════════════════════════════════

def bpm_analysis_thread(state: PhantomState, logger: Logger):
    """
    Runs in a daemon thread.  Every ANALYZE_EVERY seconds it copies the
    current contents of audio_buffer and runs estimate_bpm(), writing the
    result to state.  Also updates state.buffer_fill.
    """
    last = 0.0
    while state.alive():
        now = time.time()
        if now - last >= ANALYZE_EVERY and len(audio_buffer) >= SR * 2:
            last = now
            y = np.array(audio_buffer, dtype=np.float32)
            bpm_new, dbg = estimate_bpm(y, bpm_prev=state.get_bpm())
            if bpm_new and np.isfinite(bpm_new):
                state.set_bpm(
                    bpm_new,
                    raw=dbg.get("bpm_raw"),
                    corrected=dbg.get("bpm_corr"),
                    onset_max=dbg.get("onset_max", 0.0),
                )
                state.last_bpm_dbg = dbg
            with state._lock:
                state.buffer_fill = min(
                    1.0, len(audio_buffer) / (SR * 10))   # 10 s == full
        time.sleep(ANALYZE_EVERY / 4)
