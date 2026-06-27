"""
Phantom Conductor — Onset Model Diagnostic Plot
=================================================
Runs a trained OnsetNet checkpoint on a take and overlays BOTH the
ground-truth tap timestamps and the model's predicted onset timestamps
on the waveform, in different colors, so you can see exactly where
predictions diverge from what you actually played.

IMPORTANT: for this to be a meaningful generalization check, run it on
a take that was NOT in the --data-dir used to train the checkpoint —
otherwise you're just looking at training performance, which the
val_F1 number during training already (optimistically) measures.

Usage
-----
    python training/diagnose_onsets.py \
        --checkpoint ./models/onset_net_v1.pt \
        --take held_out/take_010

Saves an interactive plot by default; --save writes a PNG instead,
matching plot_onsets.py's convention.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from onset_dataset import (
    FRAME_TIME,
    SR,
    _audio_to_mel,
    _load_take,
)
from onset_network import OnsetNet
from train_onset import peak_pick


def run_inference(
    checkpoint_path: str,
    wav_path: Path,
    json_path: Path,
    threshold: float,
    min_spacing_frames: int,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {"n_mels": 80, "hidden": 32})

    model = OnsetNet(n_mels=cfg["n_mels"], hidden=cfg["hidden"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    take = _load_take(wav_path, json_path)
    mel = _audio_to_mel(take["audio"])  # (n_mels, n_frames)
    mel_t = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, n_frames)

    with torch.no_grad():
        logits = model(mel_t)
        probs = torch.sigmoid(logits)[0]  # (n_frames,)

    pred_frames = peak_pick(
        probs, threshold=threshold, min_spacing_frames=min_spacing_frames
    )
    pred_times = np.array(pred_frames, dtype=np.float32) * FRAME_TIME

    return {
        "audio": take["audio"],
        "true_times": take["onset_times"],
        "pred_times": pred_times,
        "probs": probs.numpy(),
        "n_frames": mel.shape[1],
        "epoch": ckpt.get("epoch"),
        "f1": ckpt.get("f1"),
    }


def match_onsets(true_times: np.ndarray, pred_times: np.ndarray, tolerance_s: float):
    """
    Greedy nearest-match within tolerance. Returns:
      matched_true   — true onsets that have a nearby prediction (hits)
      missed_true    — true onsets with NO nearby prediction (misses — what
                        you most want to look at)
      matched_pred   — predictions that matched a true onset
      spurious_pred  — predictions with no nearby true onset (false alarms)
    """
    used_pred = set()
    matched_true, missed_true = [], []

    for t in true_times:
        candidates = [
            (abs(t - p), i) for i, p in enumerate(pred_times) if i not in used_pred
        ]
        candidates = [c for c in candidates if c[0] <= tolerance_s]
        if candidates:
            _, best_i = min(candidates)
            used_pred.add(best_i)
            matched_true.append(t)
        else:
            missed_true.append(t)

    matched_pred = pred_times[[i for i in range(len(pred_times)) if i in used_pred]]
    spurious_pred = pred_times[
        [i for i in range(len(pred_times)) if i not in used_pred]
    ]

    return (
        np.array(matched_true),
        np.array(missed_true),
        np.array(matched_pred),
        np.array(spurious_pred),
    )


def _write_click_wav(timestamps: np.ndarray, duration_s: float, out_path: str):
    """
    Write a mono WAV with a short sine click at each timestamp.
    Useful for auditioning hits/misses against the original recording.
    """
    import struct
    import wave

    n_samples = int(duration_s * SR)
    buf = np.zeros(n_samples, dtype=np.float32)

    click_dur = int(0.015 * SR)  # 15ms click
    click_t = np.arange(click_dur) / SR
    click = np.sin(2 * np.pi * 1000 * click_t)  # 1kHz sine
    click *= np.hanning(click_dur)  # smooth edges

    for ts in timestamps:
        i = int(ts * SR)
        end = min(i + click_dur, n_samples)
        length = end - i
        buf[i:end] += click[:length]

    # Normalize to prevent clipping if clicks overlap
    peak = np.abs(buf).max()
    if peak > 0:
        buf /= peak

    int16 = (np.clip(buf, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(out_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(int16.tobytes())
    print(f"  wrote {out_path}  ({len(timestamps)} clicks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="path to trained .pt file")
    ap.add_argument("--take", required=True, help="prefix, e.g. held_out/take_010")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-spacing-frames", type=int, default=4)
    ap.add_argument(
        "--tolerance-ms",
        type=float,
        default=35.0,
        help="how close a prediction must be to a true onset to count as a match",
    )
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    wav_path = Path(f"{args.take}.wav")
    json_path = Path(f"{args.take}_onsets.json")
    if not wav_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"missing {wav_path} or {json_path}")

    result = run_inference(
        args.checkpoint, wav_path, json_path, args.threshold, args.min_spacing_frames
    )

    matched_true, missed_true, matched_pred, spurious_pred = match_onsets(
        result["true_times"], result["pred_times"], args.tolerance_ms / 1000.0
    )

    n_true = len(result["true_times"])
    n_hit = len(matched_true)
    n_missed = len(missed_true)
    n_spurious = len(spurious_pred)
    print(f"Checkpoint: epoch={result['epoch']}  train val_F1={result['f1']}")
    print(f"True onsets:     {n_true}")
    print(f"  Hit:           {n_hit}  ({100 * n_hit / max(1, n_true):.1f}%)")
    print(f"  Missed:        {n_missed}  ({100 * n_missed / max(1, n_true):.1f}%)")
    print(f"Spurious (false-alarm) predictions: {n_spurious}")
    if n_missed:
        print(f"\nMissed onset timestamps (look at these in the plot):")
        print("  " + ", ".join(f"{t:.2f}s" for t in sorted(missed_true)))

    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audio = result["audio"]
    duration = len(audio) / SR
    start = args.start
    end = duration if args.window is None else min(duration, start + args.window)
    i0, i1 = int(start * SR), int(end * SR)
    t = np.arange(i0, i1) / SR
    seg = audio[i0:i1]

    def _in_range(arr):
        return arr[(arr >= start) & (arr <= end)] if len(arr) else arr

    fig, (ax_wave, ax_prob) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, height_ratios=[2, 1]
    )

    ax_wave.plot(t, seg, linewidth=0.4, color="steelblue", zorder=1)
    for x in _in_range(matched_true):
        ax_wave.axvline(x, color="green", linewidth=1.0, alpha=0.8, zorder=2)
    for x in _in_range(missed_true):
        ax_wave.axvline(x, color="red", linewidth=1.4, alpha=0.9, zorder=3)
    for x in _in_range(spurious_pred):
        ax_wave.axvline(
            x, color="orange", linewidth=1.0, alpha=0.7, linestyle="--", zorder=2
        )

    ax_wave.set_ylabel("amplitude")
    ax_wave.set_title(
        f"{args.take}  —  green=hit  red=MISSED  orange(dashed)=false-alarm  "
        f"({n_hit}/{n_true} hit, {n_spurious} false-alarms)"
    )

    frame_t = np.arange(result["n_frames"]) * FRAME_TIME
    mask = (frame_t >= start) & (frame_t <= end)
    ax_prob.plot(frame_t[mask], result["probs"][mask], color="purple", linewidth=0.8)
    ax_prob.axhline(args.threshold, color="gray", linewidth=0.6, linestyle=":")
    ax_prob.set_ylabel("P(onset)")
    ax_prob.set_xlabel("time (s)")
    ax_prob.set_ylim(0, 1.05)

    fig.tight_layout()

    if args.save:
        out_png = f"{args.take}_diagnosis.png"
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"\nSaved {out_png}")
    else:
        print(
            "\nInspect the window: look for whether RED (missed) lines "
            "cluster around quiet passages, specific techniques, or a "
            "particular section — that points to a data-coverage gap. "
            "Scattered/random misses point more toward model capacity "
            "or threshold tuning."
        )
        plt.show()

    audio_duration = len(result["audio"]) / SR

    print("\nWriting click tracks...")
    take_stem = args.take
    _write_click_wav(
        result["true_times"],
        audio_duration,
        f"{take_stem}_clicks_true.wav",
    )
    _write_click_wav(
        result["pred_times"],
        audio_duration,
        f"{take_stem}_clicks_pred.wav",
    )


if __name__ == "__main__":
    main()
