"""
Phase Lock Loop (Beat Sync)

Compara timestamps de pulsos detectados contra una grilla extrapolada
y produce una corrección de velocidad para mantener la pista sincronizada
con el músico en vivo.

Algoritmo

1. Extrapolar el próximo pulso esperado: último pulso confirmado + beat_period.
2. Cuando llega un pulso nuevo, calcular delta = real - esperado.
3. Gate: si |delta| > beat_period / 2, descartar (detección errónea).
4. Los deltas aceptados se guardan en un historial corto (N_HISTORY pulsos)
   y se promedian para suavizar el jitter.
5. La corrección aplicada este pulso:
       rate_correction = 1 + (avg_delta / NUDGE_STEPS) / beat_period
   Multiplicar la velocidad de reproducción por este valor converge
   a la sincronía en ~NUDGE_STEPS pulsos.

Parámetros (configurables en CFG en tiempo de ejecución)
  pll_nudge_steps   int    10    pulsos para distribuir la corrección
  pll_history       int     4    pulsos de historial para el promedio
  pll_gate_factor  float   0.5   gate = beat_period * pll_gate_factor
                                 (0.5 = corchea, 0.25 = semicorchea)
"""

from __future__ import annotations

import collections
import time

from config import CFG
from logger import Logger

# Fallbacks por defecto
_DEFAULT_NUDGE_STEPS = 10
_DEFAULT_HISTORY = 4
_DEFAULT_GATE_FACTOR = 0.5  # fraction of a beat; 0.5 = 8th note


class PhaseLock:
    def __init__(self, logger: Logger | None = None) -> None:
        self._logger = logger

        # Historial rotativo de variaciones aceptadas
        self._deltas: collections.deque[float] = collections.deque(
            maxlen=CFG.get("pll_history", _DEFAULT_HISTORY)
        )

        # Momento esperado del próximo beat
        self._next_expected: float | None = None

        # El último multiplicador calculado
        self._rate_correction: float = 1.0

        # Hace cuantos beats fue aceptada la última variación (logging)
        self._beats_since_accept: int = 0

    # Público

    @property
    def rate_correction(self) -> float:
        return self._rate_correction

    def update(self, beat_time: float, bpm: float) -> float:
        if bpm <= 0:
            return self._rate_correction

        nudge_steps = CFG.get("pll_nudge_steps", _DEFAULT_NUDGE_STEPS)
        gate_factor = CFG.get("pll_gate_factor", _DEFAULT_GATE_FACTOR)
        history_len = CFG.get("pll_history", _DEFAULT_HISTORY)

        self._deltas = collections.deque(self._deltas, maxlen=history_len)

        beat_period = 60.0 / bpm  # segundos por beat a BPM actual
        gate = beat_period * gate_factor

        if self._next_expected is None:
            self._next_expected = beat_time + beat_period
            if self._logger:
                self._logger.debug(
                    f"pll: proximo beat esperado={self._next_expected:.3f}"
                )
            return self._rate_correction

        # Variación cruda
        delta = beat_time - self._next_expected

        if abs(delta) > gate:
            self._beats_since_accept += 1
            if self._logger:
                self._logger.debug(
                    f"pll: beat GATED  delta={delta * 1000:+.1f}ms  "
                    f"gate=±{gate * 1000:.1f}ms  "
                    f"({self._beats_since_accept} beats desde última aceptación)"
                )

            self._next_expected = beat_time + beat_period
            return self._rate_correction

        # Aceptar variación e historial de promediio
        self._deltas.append(delta)
        self._beats_since_accept = 0
        avg_delta = sum(self._deltas) / len(self._deltas)

        # Calcular "empujón" como multiplicador de corrección
        nudge_sec = avg_delta / nudge_steps
        self._rate_correction = 1.0 + (nudge_sec / beat_period)

        if self._logger:
            self._logger.debug(
                f"pll: variación aceptada={delta * 1000:+.1f}ms  "
                f"avg={avg_delta * 1000:+.1f}ms  "
                f"nudge={nudge_sec * 1000:+.1f}ms  "
                f"rate_corr={self._rate_correction:.5f}"
            )

        self._next_expected += beat_period

        return self._rate_correction

    def reset(self) -> None:

        self._deltas.clear()
        self._next_expected = None
        self._rate_correction = 1.0
        self._beats_since_accept = 0
        if self._logger:
            self._logger.info("pll: reset")
