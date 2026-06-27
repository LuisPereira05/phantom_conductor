"""
Phantom Conductor — Phase Lock Loop (Beat Sync)
================================================
Compares detected beat timestamps against an extrapolated beat grid and
produces a fractional rate nudge to keep the backing track phase-locked
to the live musician.

Algorithm
---------
  1. Extrapolate the next expected beat from the last confirmed beat + beat_period.
  2. When a new beat lands, compute delta = recorded − extrapolated.
  3. Gate: if |delta| > beat_period / 2 (an 8th note at current tempo), discard —
     it's almost certainly a mis-detection or a very late onset, not a phase error.
  4. Accepted deltas are stored in a short history (last N_HISTORY beats) and
     averaged to suppress single-beat jitter.
  5. The nudge applied this beat is avg_delta / NUDGE_STEPS, expressed as a
     fractional rate correction:
         rate_correction = 1 + (nudge_seconds / beat_period)
     Multiplying playback rate by this value shifts the next beat's arrival by
     nudge_seconds, converging to phase-lock in ~NUDGE_STEPS beats.

Parameters (all tunable via CFG at runtime)
-------------------------------------------
  pll_nudge_steps   int    10    beats to spread one full delta correction over
  pll_history       int     4    beats of delta history to average
  pll_gate_factor  float   0.5   gate = beat_period * pll_gate_factor
                                 (0.5 = 8th note, 0.25 = 16th note, etc.)

Usage
-----
    pll = PhaseLock()

    # call once per detected beat, passing the wall-clock timestamp and live BPM
    correction = pll.update(beat_time=time.time(), bpm=state.get_bpm())

    # multiply into whatever rate you're already sending to pyrubberband
    rate *= correction   # e.g. rate = bpm_live / bpm_original * correction
"""

from __future__ import annotations

import collections
import time

from config import CFG
from logger import Logger

# ── Fallback defaults (overridden by CFG when keys exist) ─────────────────────
_DEFAULT_NUDGE_STEPS = 10
_DEFAULT_HISTORY = 4
_DEFAULT_GATE_FACTOR = 0.5  # fraction of a beat; 0.5 = 8th note


class PhaseLock:
    """
    Stateful phase-lock loop.  One instance per playback session.

    Thread-safety: update() is called from the beat-detection thread;
    rate_correction is read from the playback thread.  Both accesses are
    simple float reads/writes which are atomic in CPython, so no explicit
    lock is needed.  If you move to a sub-interpreter or multiprocessing
    model, add a threading.Lock around _rate_correction.
    """

    def __init__(self, logger: Logger | None = None) -> None:
        self._logger = logger

        # Rolling history of accepted phase deltas (seconds)
        self._deltas: collections.deque[float] = collections.deque(
            maxlen=CFG.get("pll_history", _DEFAULT_HISTORY)
        )

        # Wall-clock time we expect the next beat to land
        self._next_expected: float | None = None

        # Most-recently computed rate multiplier — read by playback thread
        self._rate_correction: float = 1.0

        # How many beats ago we last accepted a delta (for logging)
        self._beats_since_accept: int = 0

    # ── Public API ────────────────────────────────────────────────────────────
    @property
    def next_expected(self) -> float | None:
        return self._next_expected

    @property
    def rate_correction(self) -> float:
        """Current rate multiplier.  Multiply into bpm_live/bpm_original."""
        return self._rate_correction

    def update(self, beat_time: float, bpm: float) -> float:
        """
        Call once per detected beat.

        Parameters
        ----------
        beat_time : float
            Wall-clock timestamp (time.time()) of the detected beat.
        bpm : float
            Current live BPM estimate.

        Returns
        -------
        float
            Rate correction multiplier for this beat (same as self.rate_correction).
        """
        if bpm <= 0:
            return self._rate_correction

        nudge_steps = CFG.get("pll_nudge_steps", _DEFAULT_NUDGE_STEPS)
        gate_factor = CFG.get("pll_gate_factor", _DEFAULT_GATE_FACTOR)
        history_len = CFG.get("pll_history", _DEFAULT_HISTORY)

        # Keep deque maxlen in sync with live config without recreating it
        self._deltas = collections.deque(self._deltas, maxlen=history_len)

        beat_period = 60.0 / bpm  # seconds per beat at current tempo
        gate = beat_period * gate_factor  # max credible phase error

        # ── 1. Seed the grid on the first beat ────────────────────────────────
        if self._next_expected is None:
            self._next_expected = beat_time + beat_period
            if self._logger:
                self._logger.debug(
                    f"pll: grid seeded  next_expected={self._next_expected:.3f}"
                )
            return self._rate_correction

        # ── 2. Compute raw delta ───────────────────────────────────────────────
        delta = beat_time - self._next_expected  # + = musician rushing

        # ── 3. Gate ───────────────────────────────────────────────────────────
        if abs(delta) > gate:
            self._beats_since_accept += 1
            if self._logger:
                self._logger.debug(
                    f"pll: beat GATED  delta={delta * 1000:+.1f}ms  "
                    f"gate=±{gate * 1000:.1f}ms  "
                    f"({self._beats_since_accept} beats since last accept)"
                )
            # Still advance the grid — the beat happened, just not where expected
            self._next_expected = beat_time + beat_period
            return self._rate_correction

        # ── 4. Accept delta and average history ───────────────────────────────
        self._deltas.append(delta)
        self._beats_since_accept = 0
        avg_delta = sum(self._deltas) / len(self._deltas)

        # ── 5. Compute nudge as a rate correction ─────────────────────────────
        # We want to shift the next beat's arrival by (avg_delta / nudge_steps).
        # Changing the rate by r for one beat_period seconds shifts arrival by:
        #   shift = beat_period * (r - 1)
        # Solving for r:
        #   r = 1 + shift / beat_period
        nudge_sec = avg_delta / nudge_steps
        self._rate_correction = 1.0 + (nudge_sec / beat_period)

        if self._logger:
            self._logger.debug(
                f"pll: accepted  delta={delta * 1000:+.1f}ms  "
                f"avg={avg_delta * 1000:+.1f}ms  "
                f"nudge={nudge_sec * 1000:+.1f}ms  "
                f"rate_corr={self._rate_correction:.5f}"
            )

        # Advance grid by one nominal beat from the expected position (not the
        # recorded one) so jitter in detection doesn't drift the grid.
        self._next_expected += beat_period

        return self._rate_correction

    def reset(self) -> None:
        """Call when BPM changes drastically or playback restarts."""
        self._deltas.clear()
        self._next_expected = None
        self._rate_correction = 1.0
        self._beats_since_accept = 0
        if self._logger:
            self._logger.info("pll: reset")
