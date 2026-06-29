import re
import time

import serial
from pedal import PedalController
from serial.tools import list_ports

from config import CFG
from logger import Logger
from state import PhantomState

BAUD_DEFAULT = 115200
RECONNECT_DELAY = 2.0
READ_TIMEOUT = 0.5

_BPM_RE = re.compile(r"^BPM:([0-9.]+)\s*$")
_OOR_RE = re.compile(r"^TAP:out_of_range\s+([0-9.]+)\s*$")

# Pedal button lines, e.g. "B1p" (button 1 pressed) / "B2r" (button 2
# released). Shares this same serial connection/thread with the tempo
# tapper — see pedal.py for the tap/hold/combo state machine these
# events feed into.
_PEDAL_RE = re.compile(r"^B([1-9])(p|r)\s*$")


# PUERTOS


def list_serial_ports() -> list[str]:
    return [p.device for p in list_ports.comports()]


def _autodetect_port(logger: Logger) -> str | None:
    """
    Preferir puertos que su descripción mencione un chip Arduino.
    """
    candidates = list(list_ports.comports())
    if not candidates:
        return None
    keywords = ("arduino", "ch340", "usb-serial", "usb serial", "wchusbserial")
    for p in candidates:
        desc = (p.description or "").lower()
        if any(k in desc for k in keywords):
            logger.info(f"tap: autodetected port {p.device} ({p.description})")
            return p.device
    logger.warn(
        f"tap: ningun puerto obviamente Arduino — usando {candidates[0].device}"
    )
    return candidates[0].device


# Parseo de lineas


def _handle_line(
    line: str,
    state: PhantomState,
    logger: Logger,
    tapper_active: bool,
    pedal: PedalController | None = None,
):
    line = line.strip()
    if not line:
        return

    if line == "READY":
        logger.ok("tap: Arduino reporta READY")
        return

    if line == "TAP:first":
        logger.info("tap: primer tap registrado, esperando segundo...")
        return

    if line == "TAP:timeout":
        logger.info("tap: secuencia de tap time-out, reseteando")
        return

    m = _BPM_RE.match(line)
    if m:
        try:
            bpm = float(m.group(1))
        except ValueError:
            logger.warn(f"tap: línea malformada: {line!r}")
            return
        if tapper_active:
            state.apply_tap_bpm(bpm)
            logger.ok(f"tap: BPM {bpm:.1f}")
        else:
            logger.info(
                f"tap: BPM {bpm:.1f} leido pero ignorado (tempo tapper deshabilitado)"
            )
        return

    m = _OOR_RE.match(line)
    if m:
        logger.info(f"tap: tap out of range ({m.group(1)} bpm) — ignorado")
        return

    m = _PEDAL_RE.match(line)
    if m:
        if pedal is None:
            # Pedal controller not wired in for this run — line is
            # recognized but there's nowhere to send it.
            return
        button_id = int(m.group(1))
        is_press = m.group(2) == "p"
        pedal.handle_event(button_id, is_press)
        return

    logger.warn(f"tap: linea no reconocida: {line!r}")


# SERIAL THREAD


def tempo_tapper_thread(
    state: PhantomState, logger: Logger, pedal: PedalController | None = None
):
    """
    Same serial connection serves both the tempo tapper (BPM:/TAP:
    lines) and the foot pedal (B<n>p / B<n>r lines) — one Arduino, one
    port, two logical devices sharing the wire protocol. Pass a
    PedalController (see pedal.py / pedal.make_pedal_dispatch) to wire
    pedal lines somewhere; pass None to run tempo-tapper-only, e.g. on
    setups with no pedal attached, exactly as before this feature
    was added.
    """
    baud = CFG.get("tap_baud", BAUD_DEFAULT)
    logged_disabled_once = False
    ser = None

    while state.alive():
        if ser is None:
            port = CFG.get("tap_port") or _autodetect_port(logger)
            if port is None:
                if not logged_disabled_once:
                    logger.warn("tap: ningun puerto serial encontrado. Reintentando...")
                    logged_disabled_once = True
                time.sleep(RECONNECT_DELAY)
                continue
            try:
                ser = serial.Serial(port, baud, timeout=READ_TIMEOUT)
                logger.ok(f"tap: conectado en {port} @ {baud} baud")
                state.tap_connected = True
                logged_disabled_once = False
            except Exception as e:
                logger.err(f"tap: fallo al abrir puerto {port}: {e}")
                ser = None
                time.sleep(RECONNECT_DELAY)
                continue

        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            tapper_active = CFG.get("use_tempo_tapper", False)
            _handle_line(line, state, logger, tapper_active, pedal=pedal)
        except (serial.SerialException, OSError) as e:
            logger.err(f"tap: serial error, reconectando: {e}")
            try:
                ser.close()
            except Exception:
                pass
            ser = None
            state.tap_connected = False
            time.sleep(RECONNECT_DELAY)

    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass
    state.tap_connected = False
    logger.info("tap: thread exited")
