"""
Phantom Conductor — Onset Training Data Recorder
=================================================
Records guitar audio and Arduino foot-tap timestamps simultaneously,
for training a learned onset detector to replace librosa.onset.onset_strength
on the live-guitar path (see phase_lock.py — this is what should be
feeding pll.update(beat_time=...) once trained).

Wire protocol expected from the Arduino (data-collection sketch variant):
    READY           — sent once on boot
    TAP:<millis>     — one line per detected tap, device-side millis()

Usage
-----
    python record_onset_data.py --out take_001
    python record_onset_data.py --out take_002 --port /dev/ttyUSB0 --no-beep

Produces, per take:
    <out>.wav            — mono mic recording
    <out>_onsets.json     — {"sample_rate", "host_onset_times", "device_millis",
                             "recording_start_epoch"}

Recording flow
---------------
  1. Press Enter to arm.
  2. A 3-2-1 countdown beep plays so you have time to get your hands
     in position (count-in beeps are NOT logged as onsets).
  3. Recording starts. Play + tap the pedal once per beat.
     Each registered tap plays a short, distinct beep — if you don't
     hear a beep when you tap, the serial line isn't being parsed
     (check wiring / port / threshold pot) and that take's labels
     will be wrong.
  4. Press Enter again to stop and save.

Sanity-check your data before training it on anything — see
plot_onsets.py (companion script) to visually verify tap timestamps
actually land on audio transients.
"""

import argparse
import json
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
import serial
import sounddevice as sd
from serial.tools import list_ports

SR = 44100
TAP_RE = re.compile(r"^TAP:(\d+)\s*$")

# ── Beep tones (Hz) — distinct from each other so count-in vs. registered
#    taps are unmistakable by ear while you're playing ─────────────────────
COUNT_IN_FREQ = 440.0  # A4 — count-in click
TAP_FREQ = 880.0  # A5 — registered tap (one octave up, clearly different)
BEEP_DURATION_S = 0.06
BEEP_SR = 44100


def _make_beep(
    freq: float, duration: float = BEEP_DURATION_S, sr: int = BEEP_SR
) -> np.ndarray:
    """Short sine burst with a fast fade in/out to avoid audible clicks."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    fade_len = max(1, int(sr * 0.005))
    envelope = np.ones_like(tone)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    return (tone * envelope).astype(np.float32)


_COUNT_IN_BEEP = _make_beep(COUNT_IN_FREQ)
_TAP_BEEP = _make_beep(TAP_FREQ)


def _play_beep(beep: np.ndarray):
    """Fire-and-forget playback on a separate stream from the recording input,
    so beep audio never touches the captured mic buffer."""
    try:
        sd.play(beep, samplerate=BEEP_SR, blocking=False)
    except Exception as e:
        print(f"[warn] beep playback failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  SERIAL PORT DISCOVERY (mirrors tempo_tapper.py conventions)
# ═══════════════════════════════════════════════════════════════════════════


def _autodetect_port() -> str | None:
    candidates = list(list_ports.comports())
    if not candidates:
        return None
    keywords = ("arduino", "ch340", "usb-serial", "usb serial", "wchusbserial")
    for p in candidates:
        desc = (p.description or "").lower()
        if any(k in desc for k in keywords):
            print(f"[info] autodetected port {p.device} ({p.description})")
            return p.device
    print(f"[warn] no obvious Arduino port — defaulting to {candidates[0].device}")
    return candidates[0].device


# ═══════════════════════════════════════════════════════════════════════════
#  RECORDER
# ═══════════════════════════════════════════════════════════════════════════


class OnsetRecorder:
    def __init__(self, port: str | None, baud: int, beep_enabled: bool):
        self.port = port or _autodetect_port()
        if self.port is None:
            raise RuntimeError("No serial port found — connect the Arduino and retry.")
        self.baud = baud
        self.beep_enabled = beep_enabled

        self._ser: serial.Serial | None = None
        self._stop_flag = threading.Event()
        self._serial_thread: threading.Thread | None = None

        self._audio_chunks: list[np.ndarray] = []
        self._host_onset_times: list[float] = []
        self._device_millis: list[int] = []

        self.start_time: float | None = None

    # ── Serial reader thread ────────────────────────────────────────────────
    def _serial_reader(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
        except Exception as e:
            print(f"[error] failed to open {self.port}: {e}")
            self._stop_flag.set()
            return

        # Drain the boot "READY" line if present
        time.sleep(0.3)
        self._ser.reset_input_buffer()

        while not self._stop_flag.is_set():
            try:
                raw = self._ser.readline()
            except Exception as e:
                print(f"[error] serial read failed: {e}")
                break
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue

            m = TAP_RE.match(line)
            if m and self.start_time is not None:
                host_t = time.time() - self.start_time
                self._host_onset_times.append(host_t)
                self._device_millis.append(int(m.group(1)))
                print(f"  tap @ {host_t:7.3f}s  (#{len(self._host_onset_times)})")
                # if self.beep_enabled:
                # _play_beep(_TAP_BEEP)
            elif line != "READY":
                print(f"[serial] {line!r}")

        try:
            self._ser.close()
        except Exception:
            pass

    # ── Audio callback ───────────────────────────────────────────────────────
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio warn] {status}")
        self._audio_chunks.append(indata.copy())

    # ── Main recording flow ─────────────────────────────────────────────────
    def run(self, out_prefix: str, countin_beats: int = 3, countin_bpm: float = 80.0):
        input("Press Enter to arm recording (count-in will follow)... ")

        beat_period = 60.0 / countin_bpm
        print(f"Count-in: {countin_beats} beats @ {countin_bpm:.0f} BPM")
        for i in range(countin_beats):
            # _play_beep(_COUNT_IN_BEEP)
            print(f"  {countin_beats - i}...")
            time.sleep(beat_period)

        self.start_time = time.time()
        self._serial_thread = threading.Thread(target=self._serial_reader, daemon=True)
        self._serial_thread.start()

        stream = sd.InputStream(
            samplerate=SR, channels=1, dtype="float32", callback=self._audio_callback
        )
        stream.start()
        print("\n>>> RECORDING — play and tap the pedal on every beat. <<<")
        print(">>> Press Enter to stop. <<<\n")
        input()

        stream.stop()
        stream.close()
        self._stop_flag.set()
        if self._serial_thread:
            self._serial_thread.join(timeout=2.0)

        self._save(out_prefix)

    def _save(self, out_prefix: str):
        if not self._audio_chunks:
            print("[error] no audio captured — nothing saved.")
            return

        # Ensure the parent directory exists before writing anything —
        # a missing directory here used to throw mid-save and silently
        # discard a take that had already been fully recorded.
        out_path = Path(out_prefix)
        if out_path.parent != Path(""):
            out_path.parent.mkdir(parents=True, exist_ok=True)

        audio = np.concatenate(self._audio_chunks, axis=0).flatten()
        duration_s = len(audio) / SR

        # Save labels FIRST and independently of the audio write. Labels are
        # tiny (just timestamps) and the audio buffer is already safely in
        # memory regardless of what happens next, but if anything below
        # does fail, having the JSON on disk means the take isn't a total
        # loss — the WAV can be re-dumped from the in-memory buffer in a
        # REPL if needed, but the precise tap timestamps can't be
        # reconstructed at all once the process exits.
        labels_path = f"{out_prefix}_onsets.json"
        try:
            with open(labels_path, "w") as f:
                json.dump(
                    {
                        "sample_rate": SR,
                        "duration_s": duration_s,
                        "host_onset_times": self._host_onset_times,
                        "device_millis": self._device_millis,
                        "recording_start_epoch": self.start_time,
                    },
                    f,
                    indent=2,
                )
            print(f"Saved {len(self._host_onset_times)} onset labels → {labels_path}")
        except Exception as e:
            print(f"[error] failed to save labels: {e}")
            print(
                f"[recovery] raw onset times (copy these down!): {self._host_onset_times}"
            )

        wav_path = f"{out_prefix}.wav"
        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SR)
                clipped = np.clip(audio, -1.0, 1.0)
                wf.writeframes((clipped * 32767).astype(np.int16).tobytes())
            print(f"Saved {duration_s:.1f}s of audio → {wav_path}")
        except Exception as e:
            print(f"[error] failed to save audio: {e}")
            print(
                "[recovery] audio is still in memory as a numpy array in "
                "this process — if running interactively, save it manually "
                "before exiting."
            )
            return

        if self._host_onset_times:
            implied_bpm = self._implied_tempo_summary()
            print(f"Implied tempo range from taps: {implied_bpm}")

    def _implied_tempo_summary(self) -> str:
        times = np.array(self._host_onset_times)
        if len(times) < 2:
            return "n/a (need >=2 taps)"
        intervals = np.diff(times)
        bpms = 60.0 / intervals[intervals > 0]
        if len(bpms) == 0:
            return "n/a"
        return f"{bpms.min():.0f}-{bpms.max():.0f} BPM (median {np.median(bpms):.0f})"


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(description="Record guitar audio + tap-onset labels")
    ap.add_argument(
        "--out", required=True, help="output filename prefix, e.g. take_001"
    )
    ap.add_argument("--port", default=None, help="serial port (default: autodetect)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--no-beep", action="store_true", help="disable tap-confirmation beep"
    )
    ap.add_argument("--countin-beats", type=int, default=3)
    ap.add_argument("--countin-bpm", type=float, default=80.0)
    args = ap.parse_args()

    recorder = OnsetRecorder(
        port=args.port, baud=args.baud, beep_enabled=not args.no_beep
    )
    recorder.run(
        out_prefix=args.out,
        countin_beats=args.countin_beats,
        countin_bpm=args.countin_bpm,
    )


if __name__ == "__main__":
    main()
