"""
convert_guitarset.py
=====================
Converts GuitarSet .jams annotations + mic audio into the
(name.wav, name_onsets.json) format expected by OnsetDataset.

GuitarSet layout (after download from Zenodo):
    guitarset/
      annotation/          ← .jams files
        00_BN1-129-Eb_comp.jams
        00_BN1-129-Eb_solo.jams
        ...
      audio/
        audio_mic/         ← use these (monophonic reference mic)
          00_BN1-129-Eb_comp_mic.wav
          ...
        audio_hex_orig/
        audio_hex_cln/

Usage
-----
    pip install jams
    python convert_guitarset.py \\
        --guitarset-dir  /path/to/guitarset \\
        --out-dir        ./training_data \\
        --split-sec      10              # optional, matches train_onset --split-sec

Output
------
    training_data/
        gs_00_BN1-129-Eb_comp.wav
        gs_00_BN1-129-Eb_comp_onsets.json
        gs_00_BN1-129-Eb_comp_c0.wav        # if --split-sec given
        gs_00_BN1-129-Eb_comp_c0_onsets.json
        ...
"""

import argparse
import json
import shutil
from pathlib import Path

import jams
import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────


def load_beat_times(jams_path: Path) -> np.ndarray:
    """
    Extract beat timestamps (seconds) from a GuitarSet .jams file.
    Uses the beat_position namespace — same approach as mirdata's load_beats().
    """
    jam = jams.load(str(jams_path))
    annos = jam.search(namespace="beat_position")
    if not annos:
        raise ValueError(f"No beat_position annotation in {jams_path.name}")
    times, _ = annos[0].to_event_values()
    return np.array(times, dtype=np.float32)


def write_onsets_json(path: Path, onset_times: np.ndarray):
    with open(path, "w") as f:
        json.dump({"host_onset_times": onset_times.tolist()}, f, indent=2)


def split_and_write(
    wav_src: Path,
    onset_times: np.ndarray,
    out_dir: Path,
    stem: str,
    split_sec: float,
):
    """
    Split wav + onsets into chunks of split_sec seconds and write each pair.
    Requires librosa/soundfile for audio splitting.
    """
    import librosa
    import soundfile as sf

    y, sr = librosa.load(str(wav_src), sr=None, mono=True)
    chunk_samples = int(split_sec * sr)
    n_chunks = int(np.ceil(len(y) / chunk_samples))

    written = 0
    for i in range(n_chunks):
        t_start = i * split_sec
        t_end = (i + 1) * split_sec

        chunk_audio = y[i * chunk_samples : (i + 1) * chunk_samples]
        if len(chunk_audio) < chunk_samples // 2:
            # skip very short tail chunks (< half the window)
            continue

        mask = (onset_times >= t_start) & (onset_times < t_end)
        chunk_onsets = onset_times[mask] - t_start

        chunk_stem = f"{stem}_c{i}"
        sf.write(str(out_dir / f"{chunk_stem}.wav"), chunk_audio, sr)
        write_onsets_json(out_dir / f"{chunk_stem}_onsets.json", chunk_onsets)
        written += 1

    return written


# ── main ──────────────────────────────────────────────────────────────────────


def convert(guitarset_dir: str, out_dir: str, split_sec: float | None = None):
    gs = Path(guitarset_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    anno_dir = gs / "annotation"
    mic_dir = gs / "audio" / "audio_mic"

    if not anno_dir.exists():
        raise RuntimeError(f"annotation/ not found under {gs}")
    if not mic_dir.exists():
        raise RuntimeError(f"audio/audio_mic/ not found under {gs}")

    jams_files = sorted(anno_dir.glob("*.jams"))
    if not jams_files:
        raise RuntimeError(f"No .jams files found in {anno_dir}")

    total_clips = 0
    skipped = 0

    for jams_path in jams_files:
        # e.g. 00_BN1-129-Eb_comp.jams → look for 00_BN1-129-Eb_comp_mic.wav
        base = jams_path.stem  # 00_BN1-129-Eb_comp
        mic_wav = mic_dir / f"{base}_mic.wav"

        if not mic_wav.exists():
            print(f"  [skip] no mic wav for {jams_path.name}")
            skipped += 1
            continue

        try:
            beat_times = load_beat_times(jams_path)
        except Exception as e:
            print(f"  [skip] {jams_path.name}: {e}")
            skipped += 1
            continue

        stem = f"gs_{base}"

        if split_sec:
            n = split_and_write(mic_wav, beat_times, out, stem, split_sec)
            print(f"  {base}: {len(beat_times)} beats → {n} chunks")
            total_clips += n
        else:
            # Copy wav as-is, just write the json
            dst_wav = out / f"{stem}.wav"
            shutil.copy2(mic_wav, dst_wav)
            write_onsets_json(out / f"{stem}_onsets.json", beat_times)
            print(f"  {base}: {len(beat_times)} beats → 1 clip")
            total_clips += 1

    print(f"\nDone. {total_clips} clips written to {out}, {skipped} skipped.")
    print(f"Now train with:")
    if split_sec:
        print(f"  python -m training.train_onset --data-dir {out} --epochs 60")
    else:
        print(
            f"  python -m training.train_onset --data-dir {out} "
            f"--epochs 60 --split-sec 10"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--guitarset-dir",
        required=True,
        help="Root of the GuitarSet download (contains annotation/ and audio/)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Where to write (name.wav, name_onsets.json) pairs",
    )
    ap.add_argument(
        "--split-sec",
        type=float,
        default=None,
        help="Split each excerpt into chunks of this length (e.g. 10). "
        "If omitted, each ~30s excerpt is written as one clip.",
    )
    args = ap.parse_args()
    convert(args.guitarset_dir, args.out_dir, args.split_sec)
