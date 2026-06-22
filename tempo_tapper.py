"""
Phantom Conductor — Foot Tempo Tapper (Serial Bridge)
======================================================
Reads line-based BPM events from the Arduino piezo tapper over USB
serial and writes them into PhantomState, using state.apply_tap_bpm()
— the same pairing audio_analysis.py uses via apply_audio_bpm().

Wire protocol (from tempo_tapper.ino)
-------------------------------------
  READY                  — sent once on boot
  TAP:first              — first tap of a new pair, no BPM yet
  BPM:<float>            — second tap landed in range, e.g. "BPM:128.4"
  TAP:out_of_range <bpm> — second tap outside 40-300 bpm, informational
  TAP:timeout            — waiting_second cleared after TIMEOUT_MS

Why a "source" flag
--------------------
bpm_analysis_thread (audio) and this thread both want to own
state.bpm_live. Coordination happens in two layers:

  1. CFG.use_tempo_tapper is checked LIVE by both threads, every pass.
     - audio_analysis_thread: skips apply_audio_bpm() entirely while
       this is True.
     - this thread: still keeps the serial port open and drains
       incoming lines while it's False (so the OS read buffer doesn't
       back up and nothing is lost when it's flipped back on), but
       does NOT call state.apply_tap_bpm() until it's True again.
  2. state.apply_tap_bpm() / apply_audio_bpm() still maintain the
     short override window as a second line of defence — useful if
     someone flips the checkbox mid-tap, or if a stray audio analysis
     pass lands in the same instant a tap comes in.

Both checks are necessary: (1) makes the checkbox actually mean
something continuously, not just at thread startup; (2) avoids a
race in the brief moment around the toggle.

This module does NOT decide PLAY/PAUSE — it only ever writes BPM.
If you wire a second piezo/button for transport later, route it
through state.dispatch_command() the same way gesture_recognition does.
"""

import re
import time

import serial
from serial.tools import list_ports

from config import CFG
from logger import Logger
from state import PhantomState

# ── Constants ─────────────────────────────────────────────────────────────────
BAUD_DEFAULT = 115200
RECONNECT_DELAY = 2.0  # seconds between reconnect attempts
READ_TIMEOUT = 0.5  # serial read timeout (keeps loop responsive to stop())

_BPM_RE = re.compile(r"^BPM:([0-9.]+)\s*$")
_OOR_RE = re.compile(r"^TAP:out_of_range\s+([0-9.]+)\s*$")


# ═══════════════════════════════════════════════════════════════════════════════
#  PORT DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════


def list_serial_ports() -> list[str]:
    """Return device paths for all currently visible serial ports."""
    return [p.device for p in list_ports.comports()]


def _autodetect_port(logger: Logger) -> str | None:
    """
    Best-effort guess when CFG.tap_port is unset: prefer ports whose
    description mentions a typical Arduino USB-serial chip.
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

    # Fall back to the first available port rather than refusing to run
    logger.warn(f"tap: no obvious Arduino port — defaulting to {candidates[0].device}")
    return candidates[0].device


# ═══════════════════════════════════════════════════════════════════════════════
#  LINE PARSING
# ═══════════════════════════════════════════════════════════════════════════════


def _handle_line(line: str, state: PhantomState, logger: Logger, tapper_active: bool):
    """
    Parse one serial line and, if it's a BPM reading and tapper mode is
    currently enabled, write it into state. Lines are still parsed and
    logged even when tapper_active is False, so the log stays useful
    and timeouts/first-taps don't go silent — only the actual state
    write is gated.
    """
    line = line.strip()
    if not line:
        return

    if line == "READY":
        logger.ok("tap: Arduino reports READY")
        return

    if line == "TAP:first":
        logger.info("tap: first tap registered, waiting for second…")
        return

    if line == "TAP:timeout":
        logger.info("tap: tap sequence timed out, resetting")
        return

    m = _BPM_RE.match(line)
    if m:
        try:
            bpm = float(m.group(1))
        except ValueError:
            logger.warn(f"tap: malformed BPM line: {line!r}")
            return
        if tapper_active:
            state.apply_tap_bpm(bpm)
            logger.ok(f"tap: BPM {bpm:.1f}")
        else:
            logger.info(
                f"tap: BPM {bpm:.1f} read but ignored "
                f"(tempo tapper disabled in Settings)"
            )
        return

    m = _OOR_RE.match(line)
    if m:
        logger.info(f"tap: tap out of range ({m.group(1)} bpm) — ignored")
        return

    logger.warn(f"tap: unrecognised line: {line!r}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SERIAL THREAD
# ═══════════════════════════════════════════════════════════════════════════════


def tempo_tapper_thread(state: PhantomState, logger: Logger):
    """
    Daemon thread: opens the configured serial port and feeds BPM/TAP
    events into PhantomState until state.alive() goes False.

    Unlike v0.5.2, this thread does NOT exit early when
    CFG.use_tempo_tapper is False at startup — it always opens the
    port (so the connection is warm and ready) and checks the flag
    live on every line, so flipping the Settings checkbox at runtime
    enables/disables tap input immediately without needing the app
    restarted or the thread relaunched.

    Reconnects automatically if the port disappears (e.g. USB unplug).

    Config keys read from CFG (add to config.py _DEFAULTS if missing):
      tap_port  : str | None   — e.g. "/dev/ttyUSB0" or "COM5"; None = autodetect
      tap_baud  : int          — defaults to 115200
      use_tempo_tapper : bool  — gates whether incoming taps are applied
    """
    baud = CFG.get("tap_baud", BAUD_DEFAULT)
    logged_disabled_once = False

    ser = None
    while state.alive():
        if ser is None:
            port = CFG.get("tap_port") or _autodetect_port(logger)
            if port is None:
                if not logged_disabled_once:
                    logger.warn("tap: no serial port found — retrying…")
                    logged_disabled_once = True
                time.sleep(RECONNECT_DELAY)
                continue
            try:
                ser = serial.Serial(port, baud, timeout=READ_TIMEOUT)
                logger.ok(f"tap: connected on {port} @ {baud} baud")
                state.tap_connected = True
                logged_disabled_once = False
            except Exception as e:
                logger.err(f"tap: failed to open {port}: {e}")
                ser = None
                time.sleep(RECONNECT_DELAY)
                continue

        try:
            raw = ser.readline()
            if not raw:
                continue  # just a read timeout, loop again and check state.alive()
            line = raw.decode("utf-8", errors="replace")
            tapper_active = CFG.get("use_tempo_tapper", False)
            _handle_line(line, state, logger, tapper_active)

        except (serial.SerialException, OSError) as e:
            logger.err(f"tap: serial error, reconnecting: {e}")
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
