"""
Phantom Conductor — Audio Processing & Track Loader
====================================================
Responsibilities
----------------
* load_track          : loads any audio file via librosa; reads BPM from
                        metadata tag or falls back to beat_track estimation.
* backing_track_thread: beat-by-beat playback loop with live time-stretching
                        (pyrubberband).  Reads live BPM from PhantomState,
                        stretches each beat block to match, and pushes it to
                        the shared audio_queue for audio_playback to drain.

This module reads from PhantomState (BPM, gain, flags) and writes only
track-progress fields (position, duration, bpm_original).
It does NOT touch sounddevice directly — that is audio_input's job.

Changes from v0.5.2
---------------------
* backing_track_thread now actually consumes state.skip_to_next /
  state.skip_to_prev. Previously state.dispatch_command("next"/"prev")
  only set these flags — nothing downstream ever read them, so gesture-
  or tap-triggered next/prev silently did nothing even though the
  dispatch itself worked correctly (visible in the log as
  "gesture [...] POINT -> next" with no actual track change).
* The flag is read-and-cleared under the lock in one step, the same
  pattern already used for load_new_track, so a gesture firing twice
  in quick succession can't queue up two skips. Routed through
  state.queue.next_track() / prev_track() — the same calls the UI's
  Prev/Next buttons already use — so behavior stays consistent
  regardless of whether the skip came from a button, a gesture, or
  (in the future) a second tapper input.
"""

import os
import time

import librosa
import numpy as np

from buffers import SR, audio_queue
from logger import Logger
from state import PhantomState

try:
    from mutagen import File as MutagenFile

    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

try:
    import pyrubberband as pyrb

    HAS_PYRB = True
except ImportError:
    HAS_PYRB = False
    print("[warn] pyrubberband not found — time-stretch disabled")

MIN_BPM = 60
MAX_BPM = 200


# ═══════════════════════════════════════════════════════════════════════════════
#  BPM TAG READER
# ═══════════════════════════════════════════════════════════════════════════════


def _read_bpm_tag(path: str) -> float | None:
    """Try every known BPM tag field across formats (requires mutagen)."""
    if not HAS_MUTAGEN:
        return None
    try:
        af = MutagenFile(path)
        tags = af.tags if af else None
        if not tags:
            return None
        for key in ("TBPM", "bpm", "BPM", "TXXX:BPM", "----:com.apple.iTunes:BPM"):
            if key in tags:
                raw = tags[key]
                val = str(
                    raw[0]
                    if (hasattr(raw, "__iter__") and not isinstance(raw, str))
                    else raw
                )
                return float(val.strip())
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  BPM ESTIMATOR FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════


def _estimate_bpm_from_file(y: np.ndarray, sr: int, logger: Logger) -> float | None:
    """Use librosa beat tracker as a fallback for untagged files."""
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, "__len__"):
            tempo = float(tempo[0]) if len(tempo) else None
        else:
            tempo = float(tempo)
        if tempo and MIN_BPM <= tempo <= MAX_BPM:
            return tempo
        for candidate in (tempo * 2, tempo / 2):
            if candidate and MIN_BPM <= candidate <= MAX_BPM:
                return float(candidate)
    except Exception as e:
        logger.warn(f"librosa beat_track failed: {e}")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  TRACK LOADER
# ═══════════════════════════════════════════════════════════════════════════════


def load_track(
    path: str, state: PhantomState, logger: Logger
) -> tuple[np.ndarray, float, float]:
    """
    Load any supported audio file via librosa.

    BPM priority
    ------------
    1. UI-supplied BPM already stored in state.bpm_original
    2. Metadata tag in the file (TBPM / bpm / …)
    3. librosa beat_track estimation
    4. Fall back to state.bpm_original (default 120)

    Returns
    -------
    (samples_float32, bpm_original, duration_seconds)
    """
    logger.info(f"loading: {os.path.basename(path)}")
    y, _ = librosa.load(path, sr=SR, mono=True)
    dur = len(y) / SR

    with state._lock:
        ui_bpm = state.bpm_original

    tag_bpm = _read_bpm_tag(path)
    if tag_bpm:
        logger.ok(f"BPM from tag: {tag_bpm:.1f}")

    if tag_bpm is None:
        logger.info("no BPM tag — running beat tracker…")
        tag_bpm = _estimate_bpm_from_file(y, SR, logger)
        if tag_bpm:
            logger.ok(f"BPM from beat_track: {tag_bpm:.1f}")
        else:
            logger.warn(f"beat_track failed — using ref BPM {ui_bpm:.1f}")

    bpm_orig = tag_bpm if tag_bpm else ui_bpm
    logger.ok(f"track ready: {dur:.1f}s  bpm_ref={bpm_orig:.1f}")
    return y.astype(np.float32), bpm_orig, dur


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKING TRACK THREAD
# ═══════════════════════════════════════════════════════════════════════════════


def _consume_skip_flags(state: PhantomState) -> str | None:
    """
    Atomically read-and-clear state.skip_to_next / skip_to_prev.

    Returns "next", "prev", or None. If somehow both were set in the
    same tick (e.g. two gestures landed back-to-back), "next" wins and
    "prev" is dropped — an arbitrary but harmless tiebreak, since this
    should essentially never happen in practice given gesture hold
    times are much longer than one loop iteration.
    """
    with state._lock:
        nxt = state.skip_to_next
        prv = state.skip_to_prev
        state.skip_to_next = False
        state.skip_to_prev = False
    if nxt:
        return "next"
    if prv:
        return "prev"
    return None


def backing_track_thread(state: PhantomState, logger: Logger):
    """
    Waits for a track signal (state.load_new_track), loads it, then plays
    beat-by-beat with live time-stretching.  Respects PLAY/PAUSE.
    When a track ends it auto-advances the queue (or loops if is_looping).

    Also honours state.skip_to_next / state.skip_to_prev, set by
    state.dispatch_command("next"/"prev") — e.g. from a trained gesture
    or the UI's Prev/Next buttons.  Checked every iteration, even while
    paused or idle, so a skip request is never silently dropped.
    """
    y_full = None
    bpm_orig = 120.0
    pos = 0
    t_next = time.time()

    while state.alive():
        # ── Check for a manual skip request (gesture / tapper / UI) ───────────
        skip = _consume_skip_flags(state)
        if skip:
            next_t = (
                state.queue.next_track() if skip == "next" else state.queue.prev_track()
            )
            if next_t:
                with state._lock:
                    state.load_new_track = next_t
                logger.info(f"skip: {skip} → {next_t['name']}")
            else:
                logger.info(f"skip: {skip} requested but queue has no track")
            # Fall through — the load_new_track check below will pick
            # this up on the same iteration.

        # ── Check for a new track to load ─────────────────────────────────────
        with state._lock:
            new_track = state.load_new_track
            if new_track:
                state.load_new_track = None

        if new_track:
            if new_track.get("bpm"):
                state.set_bpm_original(float(new_track["bpm"]))

            try:
                y_full, bpm_orig, dur = load_track(new_track["path"], state, logger)

                if new_track.get("bpm"):
                    bpm_orig = float(new_track["bpm"])

                with state._lock:
                    state.bpm_original = bpm_orig
                    state.bpm_live = bpm_orig
                    state.stretch_ratio = 1.0
                    state.track_path = new_track["path"]
                    state.track_duration = dur
                    state.track_position = 0.0
                    state.markers = []

                state.queue.set_bpm(state.queue._index, bpm_orig)
                pos = 0
                t_next = time.time()
                state.play()
                logger.ok(f"playing: {new_track['name']}  BPM={bpm_orig:.1f}")

            except Exception as e:
                logger.err(f"failed to load track: {e}")
                y_full = None
                new_track = None

        # ── Idle ──────────────────────────────────────────────────────────────
        if y_full is None:
            time.sleep(0.05)
            continue

        # ── Paused ────────────────────────────────────────────────────────────
        if not state.playing():
            time.sleep(0.05)
            t_next = time.time()
            continue

        # ── Track finished ────────────────────────────────────────────────────
        if pos >= len(y_full):
            logger.ok("track finished")
            with state._lock:
                looping = state.is_looping
            if looping:
                pos = 0
                t_next = time.time()
                logger.info("loop: restarting")
            else:
                next_t = state.queue.next_track()
                if next_t:
                    with state._lock:
                        state.load_new_track = next_t
                    y_full = None
                    pos = 0
                else:
                    state.pause()
                    with state._lock:
                        state.track_position = state.track_duration
                    y_full = None
                    logger.info("queue empty — stopped")
            continue

        # ── Build one beat block ───────────────────────────────────────────────
        safe_orig = max(1.0, bpm_orig)
        beat_size = int(60.0 / safe_orig * SR)
        end = min(pos + beat_size, len(y_full))

        with state._lock:
            gain = state.gain

        block = y_full[pos:end] * gain
        bpm_live = state.get_bpm() or safe_orig
        rate = (state.bpm_live / state.bpm_original) * _pll.rate_correction

        # Time-stretch
        if HAS_PYRB and len(block) > 512 and abs(rate - 1.0) > 0.005:
            try:
                block = pyrb.time_stretch(block, SR, rate)
            except Exception as e:
                logger.warn(f"time-stretch: {e}")

        # Wait until scheduled beat time
        wait = t_next - time.time()
        if wait > 0:
            time.sleep(wait)

        try:
            audio_queue.put_nowait(block.astype(np.float32))
        except Exception:
            pass  # drop if full — stay in sync

        pos += beat_size
        t_next += 60.0 / max(1.0, bpm_live)
        state.set_position(pos / SR)
