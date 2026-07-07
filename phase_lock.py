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

  pll_catchup_beats            int    4     beats en que se intenta pagar
                                             la deuda de fase acumulada
  pll_catchup_max_rate_mult    float  1.5   techo de rate_correction
                                             durante el catch-up (1.5 = no
                                             pasa de 1.5x ni baja de 1/1.5x)
  pll_debt_exit_threshold_beats float 0.05  deuda restante (en beats) para
                                             considerar el catch-up completo
                                             y volver al nudge normal

Catch-up (overshoot controlado)
  Cuando el drift de un beat excede el gate normal, en vez de descartarlo
  como detección errónea, se asume que el músico cambió de tempo de
  verdad: el exceso se acumula como "deuda de fase" (segundos) y
  rate_correction se calcula para pagarla en ~pll_catchup_beats beats,
  corriendo más rápido (o más lento) que el target_rate hasta que la
  deuda se agota. Esto es lo que hace que, p.ej., al pasar de 120 a 130
  BPM el backing track corra brevemente por encima de 130 BPM hasta
  alcanzar la posición del músico, en vez de quedar permanentemente
  desfasado a la velocidad correcta.
"""

from __future__ import annotations

import collections
import time

import numpy as np

from config import CFG
from logger import Logger

# Fallbacks por defecto
_DEFAULT_NUDGE_STEPS = 1
_DEFAULT_HISTORY = 2
_DEFAULT_GATE_FACTOR = 1.0  # fraction of a beat; 0.5 = 8th note

_DEFAULT_CATCHUP_BEATS = 2  # en cuantos beats se intenta pagar la deuda
_DEFAULT_CATCHUP_MAX_RATE_MULT = 2.0  # techo: rate_correction no pasa de 1.5x
_DEFAULT_DEBT_EXIT_THRESHOLD_BEATS = (
    0.05  # deuda restante (en beats) para volver a modo nudge
)


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

        self._phase_debt: float = 0.0
        self._catching_up: bool = False

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
        catchup_beats = CFG.get("pll_catchup_beats", _DEFAULT_CATCHUP_BEATS)
        catchup_max_mult = CFG.get(
            "pll_catchup_max_rate_mult", _DEFAULT_CATCHUP_MAX_RATE_MULT
        )
        debt_exit_beats = CFG.get(
            "pll_debt_exit_threshold_beats", _DEFAULT_DEBT_EXIT_THRESHOLD_BEATS
        )

        self._deltas = collections.deque(self._deltas, maxlen=history_len)

        beat_period = 60.0 / bpm  # segundos por beat a BPM actual
        gate = beat_period * gate_factor
        debt_exit_sec = debt_exit_beats * beat_period

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

            # Antes esto se descartaba en silencio (detección errónea
            # asumida). Pero un drift sostenido más allá del gate también
            # es justo lo que se ve cuando el músico cambia de tempo de
            # verdad — el grid sigue extrapolando el tempo viejo mientras
            # el músico ya está en el nuevo. Tratamos el exceso como
            # deuda de fase a pagar con un overshoot de velocidad
            # controlado, en vez de perderlo.
            self._phase_debt += delta
            self._catching_up = True

            if self._logger:
                self._logger.debug(
                    f"pll: beat GATED  delta={delta * 1000:+.1f}ms  "
                    f"gate=±{gate * 1000:.1f}ms  "
                    f"({self._beats_since_accept} beats desde última aceptación)  "
                    f"phase_debt={self._phase_debt * 1000:+.1f}ms"
                )

            self._next_expected = beat_time + beat_period
            self._rate_correction = self._catchup_rate(
                beat_period, catchup_beats, catchup_max_mult
            )
            return self._rate_correction

        # Aceptar variación e historial de promediio
        self._deltas.append(delta)
        self._beats_since_accept = 0
        avg_delta = sum(self._deltas) / len(self._deltas)

        if self._catching_up:
            # Seguimos pagando deuda: un beat aceptado mientras hay deuda
            # pendiente cuenta como "un beat más de progreso" — se
            # descuenta proporcionalmente al tiempo de un beat, no se
            # asume pagada de un solo golpe (eso solo pasaría si delta
            # por si solo cerrara toda la deuda, lo cual ya estaría fuera
            # del gate normalmente).
            self._phase_debt -= np.sign(self._phase_debt) * min(
                abs(self._phase_debt), beat_period / catchup_beats
            )
            if abs(self._phase_debt) <= debt_exit_sec:
                self._phase_debt = 0.0
                self._catching_up = False
                if self._logger:
                    self._logger.info(
                        "pll: catch-up completo, volviendo a nudge normal"
                    )

        if self._catching_up:
            self._rate_correction = self._catchup_rate(
                beat_period, catchup_beats, catchup_max_mult
            )
        else:
            # Calcular "empujón" como multiplicador de corrección
            nudge_sec = avg_delta / nudge_steps
            self._rate_correction = 1.0 + (nudge_sec / beat_period)

        if self._logger:
            self._logger.debug(
                f"pll: variación aceptada={delta * 1000:+.1f}ms  "
                f"avg={avg_delta * 1000:+.1f}ms  "
                f"rate_corr={self._rate_correction:.5f}  "
                f"catching_up={self._catching_up}  "
                f"phase_debt={self._phase_debt * 1000:+.1f}ms"
            )

        self._next_expected += beat_period

        return self._rate_correction

    def _catchup_rate(
        self, beat_period: float, catchup_beats: float, max_mult: float
    ) -> float:
        """
        Rate de overshoot proporcional a la deuda pendiente, repartido
        para pagarse en `catchup_beats` beats — misma forma que el nudge
        normal (proporcional, sin termino integral), pero sin el techo
        pequeño que impone el gate. `max_mult` limita el caso extremo
        (ej. una detección espuria de un beat perdido) para que nunca
        ordene una velocidad absurda.
        """
        if beat_period <= 0:
            return 1.0
        # phase_debt > 0 significa que los beats reales llegan DESPUÉS de
        # lo que el grid esperaba (el músico se atrasó / el grid va
        # adelantado) -> hay que FRENAR el backing track para que la
        # posición lo alcance -> rate_correction < 1.
        # phase_debt < 0 significa que los beats reales llegan ANTES de
        # lo esperado (el músico se aceleró) -> hay que ACELERAR el
        # backing track para no quedarse atrás -> rate_correction > 1.
        # Por eso el boost lleva signo opuesto a phase_debt.
        boost = -self._phase_debt / (catchup_beats * beat_period)
        rate = 1.0 + boost
        return float(np.clip(rate, 1.0 / max_mult, max_mult))

    def reset(self) -> None:

        self._deltas.clear()
        self._next_expected = None
        self._rate_correction = 1.0
        self._beats_since_accept = 0
        self._phase_debt = 0.0
        self._catching_up = False
        if self._logger:
            self._logger.info("pll: reset")
