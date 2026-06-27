"""
Phantom Conductor — Onset Label Sanity Check
=============================================
Plots a recorded take's waveform with vertical lines at each logged tap
timestamp. Run this on every take BEFORE using it for training — a
take where the lines don't line up with visible transients has bad
labels (debounce too aggressive, threshold pot misadjusted, you tapped
late/early relative to your own playing, etc.) and should be redone
rather than fed into the dataset.

Usage
-----
    python plot_onsets.py take_001              # interactive window (default)
    python plot_onsets.py take_001 --start 10 --window 5   # zoom into 10s-15s
    python plot_onsets.py take_001 --save        # save PNG instead (headless use)

Looks for take_001.wav and take_001_onsets.json relative to the current
directory — pass the same prefix (including any subfolder, e.g.
training_data/take_001) that you gave record_onset_data.py's --out.
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np


def load_wav(path: str) -> tuple[np.ndarray, int]:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found — did the recording finish saving? "
            f"Check for a matching _onsets.json too."
        )
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        sampwidth = wf.getsampwidth()
        nchannels = wf.getnchannels()
        raw = wf.readframes(n)

    if sampwidth != 2:
        raise ValueError(
            f"expected 16-bit audio (sampwidth=2), got sampwidth={sampwidth} "
            f"— this script only decodes int16 WAVs produced by record_onset_data.py"
        )

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    if nchannels > 1:
        audio = audio.reshape(-1, nchannels).mean(axis=1)  # downmix, just in case
    return audio, sr


def load_labels(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found — the .wav exists but its onset labels don't. "
            f"Was this take's _save() interrupted partway through?"
        )
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "prefix", help="filename prefix, e.g. take_001 or training_data/take_001"
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="save a PNG instead of opening an interactive window "
        "(use this for headless/SSH setups with no display)",
    )
    ap.add_argument(
        "--window",
        type=float,
        default=None,
        help="seconds to plot, e.g. 10 — defaults to whole take",
    )
    ap.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    args = ap.parse_args()

    wav_path = f"{args.prefix}.wav"
    labels_path = f"{args.prefix}_onsets.json"

    audio, sr = load_wav(wav_path)
    labels = load_labels(labels_path)

    # matplotlib backend chosen AFTER parsing args, based on whether we're
    # actually going to show a window — Agg (headless) for --save, the
    # normal interactive backend otherwise.
    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    onset_times = np.array(labels["host_onset_times"])
    duration = len(audio) / sr

    start = args.start
    end = duration if args.window is None else min(duration, start + args.window)

    i0, i1 = int(start * sr), int(end * sr)
    t = np.arange(i0, i1) / sr
    seg = audio[i0:i1]
    seg_onsets = onset_times[(onset_times >= start) & (onset_times <= end)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, seg, linewidth=0.5, color="steelblue")
    for ot in seg_onsets:
        ax.axvline(ot, color="red", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(
        f"{args.prefix}  —  {len(seg_onsets)} taps shown  "
        f"({len(onset_times)} total in full take)"
    )
    ax.set_xlim(start, end)
    fig.tight_layout()

    if args.save:
        out_png = f"{args.prefix}_check.png"
        out_dir = Path(out_png).parent
        if str(out_dir) != "":
            out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"Saved {out_png}")
    else:
        print(
            "Inspect the window: red lines should sit right on top of "
            "visible transients/attacks in the blue waveform. Use the "
            "plot toolbar to zoom/pan into specific taps. If lines are "
            "consistently offset by a small fixed amount, that's likely "
            "serial/debounce latency and may be correctable with a "
            "constant bias. If they're scattered or clearly miss real "
            "attacks, redo the take."
        )
        plt.show()


if __name__ == "__main__":
    main()
