"""
training/onset_dataset.py
==========================
Builds (mel_spectrogram, onset_probability_curve) training pairs from
recorded (wav, onsets.json) takes produced by record_onset_data.py.

Label strategy
--------------
Each true tap timestamp becomes a soft Gaussian bump in the target
curve, not a hard single-frame spike. This matches standard practice
in onset-detection literature: tap timing always has a few ms of
inherent jitter (yours and the hardware's), so demanding the network
hit one exact frame is needlessly punishing and would mostly teach it
to hedge toward lower confidence everywhere instead of learning the
real acoustic pattern.
"""

import json
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

SR = 22050
N_MELS = 80
HOP_LENGTH = 256  # ~11.6ms per frame at SR=22050 — fine grain for onset timing
FRAME_TIME = HOP_LENGTH / SR

# Width (in frames) of the Gaussian label bump around each true onset.
LABEL_SIGMA_FRAMES = 2.5

# How many frames of context the model sees per training window.
WINDOW_FRAMES = 256  # ~3s of audio at this hop length


def _load_take(wav_path: Path, json_path: Path) -> dict:
    y, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    with open(json_path) as f:
        labels = json.load(f)
    onset_times = np.array(labels["host_onset_times"], dtype=np.float32)
    return {"audio": y, "onset_times": onset_times, "name": wav_path.stem}


def _audio_to_mel(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=1024
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Per-take normalization — zero mean, unit variance.
    mean, std = log_mel.mean(), log_mel.std() + 1e-8
    return ((log_mel - mean) / std).astype(np.float32)  # (n_mels, n_frames)


def _onsets_to_target(onset_times: np.ndarray, n_frames: int) -> np.ndarray:
    """Soft Gaussian-bump target curve, one value per frame."""
    target = np.zeros(n_frames, dtype=np.float32)
    frame_idx = np.arange(n_frames)
    for ot in onset_times:
        center = ot / FRAME_TIME
        bump = np.exp(-0.5 * ((frame_idx - center) / LABEL_SIGMA_FRAMES) ** 2)
        target = np.maximum(target, bump)  # overlapping bumps don't stack additively
    return target


def _split_audio(y: np.ndarray, split_sec: float) -> list[np.ndarray]:
    chunk_samples = int(split_sec * SR)
    return [y[i : i + chunk_samples] for i in range(0, len(y), chunk_samples)]


def _split_onsets(
    onset_times: np.ndarray, y: np.ndarray, split_sec: float
) -> list[np.ndarray]:
    """Re-zero onset timestamps relative to the start of each chunk."""
    chunks = []
    n_chunks = int(np.ceil(len(y) / (split_sec * SR)))
    for i in range(n_chunks):
        t_start = i * split_sec
        t_end = (i + 1) * split_sec
        mask = (onset_times >= t_start) & (onset_times < t_end)
        chunks.append(onset_times[mask] - t_start)
    return chunks


class OnsetDataset(Dataset):
    """
    Loads every (name.wav, name_onsets.json) pair found under `data_dir`,
    converts each to a (mel, target) pair, and serves fixed-length random
    crops for training.

    Directory layout expected:
        data_dir/
          take_001.wav
          take_001_onsets.json
          take_002.wav
          take_002_onsets.json
          ...
    """

    def __init__(
        self,
        data_dir: str,
        window_frames: int = WINDOW_FRAMES,
        crops_per_take: int = 20,
        split_sec: float | None = None,  # e.g. 10.0 to split takes into 10s chunks
    ):
        data_dir = Path(data_dir)
        wav_files = sorted(data_dir.glob("*.wav"))
        if not wav_files:
            raise RuntimeError(f"No .wav files found in {data_dir}")

        self.window_frames = window_frames
        self.crops_per_take = crops_per_take
        self.takes = []

        for wav_path in wav_files:
            json_path = wav_path.with_name(wav_path.stem + "_onsets.json")
            if not json_path.exists():
                print(
                    f"  [dataset] skipping {wav_path.name} — no matching _onsets.json"
                )
                continue

            take = _load_take(wav_path, json_path)
            audio_chunks = (
                _split_audio(take["audio"], split_sec) if split_sec else [take["audio"]]
            )
            onset_chunks = (
                _split_onsets(take["onset_times"], take["audio"], split_sec)
                if split_sec
                else [take["onset_times"]]
            )

            for i, (chunk_audio, chunk_onsets) in enumerate(
                zip(audio_chunks, onset_chunks)
            ):
                mel = _audio_to_mel(chunk_audio)
                target = _onsets_to_target(chunk_onsets, mel.shape[1])
                if mel.shape[1] < window_frames:
                    print(f"  [dataset] skipping {wav_path.name} chunk {i} — too short")
                    continue
                self.takes.append(
                    {"mel": mel, "target": target, "name": f"{take['name']}_c{i}"}
                )
            print(
                f"  [dataset] loaded {wav_path.name}: "
                f"{mel.shape[1]} frames, {len(take['onset_times'])} onsets"
            )

        if not self.takes:
            raise RuntimeError(
                f"No usable takes in {data_dir} — check that every .wav has "
                f"a matching _onsets.json and is longer than the crop window."
            )

        total = len(self.takes) * self.crops_per_take
        print(f"[dataset] {len(self.takes)} takes, {total} crops/epoch")

    def __len__(self):
        return len(self.takes) * self.crops_per_take

    def __getitem__(self, idx):
        take = self.takes[idx // self.crops_per_take]
        mel, target = take["mel"], take["target"]
        n_frames = mel.shape[1]

        max_start = n_frames - self.window_frames
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end = start + self.window_frames

        mel_crop = mel[:, start:end]
        target_crop = target[start:end]

        return {
            "mel": torch.from_numpy(mel_crop),  # (n_mels, window_frames)
            "target": torch.from_numpy(target_crop),  # (window_frames,)
        }
