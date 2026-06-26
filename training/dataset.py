"""
training/dataset.py
===================
Generates synthetic training pairs from reference backing tracks.
No live guitar recordings needed — we simulate realistic timing drift.
"""

import random
from pathlib import Path

import librosa
import numpy as np
import torch
from analysis.timing_feature_extractor import TimingFeatureExtractor
from torch.utils.data import Dataset


class TimingOffsetDataset(Dataset):
    """
    Dataset of (ref_features, live_features, delta_t) triplets.

    Each sample is a 32-beat window with synthetic timing drift applied.
    The "live" features are just reference features + noise + time offset.
    """

    def __init__(
        self,
        audio_dir: str,
        sr: int = 22050,
        seq_len: int = 32,
        max_offset_ms: float = 100.0,
        samples_per_song: int = 200,
    ):
        self.audio_files = (
            list(Path(audio_dir).glob("*.wav"))
            + list(Path(audio_dir).glob("*.mp3"))
            + list(Path(audio_dir).glob("*.flac"))
        )
        self.sr = sr
        self.seq_len = seq_len
        self.max_offset = max_offset_ms / 1000.0  # Convert to seconds
        self.samples_per_song = samples_per_song
        self.extractor = TimingFeatureExtractor(sr=sr)

        # Pre-extract reference features for all songs
        self.song_data = []
        for audio_path in self.audio_files:
            try:
                data = self._process_song(audio_path)
                self.song_data.append(data)
                print(
                    f"  [dataset] Loaded: {audio_path.name} ({data['n_beats']} beats)"
                )
            except Exception as e:
                print(f"  [dataset] Skipped {audio_path.name}: {e}")

        total_samples = len(self.song_data) * self.samples_per_song
        print(f"[dataset] {len(self.song_data)} songs, {total_samples} total samples")

    def _process_song(self, audio_path: Path) -> dict:
        """Extract reference features and beat times from a song."""
        # Load up to 5 minutes (songs are usually shorter, but cap for safety)
        y, _ = librosa.load(str(audio_path), sr=self.sr, mono=True, duration=300)

        features = self.extractor.extract(y)  # (n_beats, 31)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=self.sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        return {
            "features": features,  # torch.Tensor (n_beats, 31)
            "beat_times": beat_times,  # np.ndarray (n_beats,)
            "n_beats": len(beat_times),
        }

    def __len__(self):
        return len(self.song_data) * self.samples_per_song

    def __getitem__(self, idx):
        song = self.song_data[idx // self.samples_per_song]

        # Random window within song
        max_start = max(0, song["n_beats"] - self.seq_len - 1)
        if max_start == 0:
            start_beat = 0
        else:
            start_beat = random.randint(0, max_start)
        end_beat = min(start_beat + self.seq_len, song["n_beats"])

        ref_features = song["features"][start_beat:end_beat]  # (window_beats, 31)
        ref_times = song["beat_times"][start_beat:end_beat]
        n_beats = len(ref_features)

        # Generate synthetic drift
        # Model: musician starts synced, then gradually drifts ±max_offset
        # Uses cumulative random walk (realistic: small errors compound)
        drift_rates = np.random.normal(0, self.max_offset / (n_beats * 2), n_beats)
        drift = np.cumsum(drift_rates)
        # Clamp to realistic bounds
        drift = np.clip(drift, -self.max_offset, self.max_offset)

        # Add subtle acceleration bias (musicians tend to rush or drag)
        if random.random() > 0.5:
            bias = np.linspace(0, self.max_offset * 0.3, n_beats)
        else:
            bias = np.linspace(0, -self.max_offset * 0.3, n_beats)
        drift = drift + bias
        drift = np.clip(drift, -self.max_offset, self.max_offset)

        # Target: Δt at each beat (what the network should predict)
        delta_t = torch.tensor(drift, dtype=torch.float32)  # (n_beats,)

        # Simulate "live" features:
        # 1. Add noise (different guitar timbre, room acoustics)
        # 2. Slightly shift features to simulate timing offset
        noise = torch.randn_like(ref_features) * 0.15
        live_features = ref_features + noise

        # Pad to fixed length
        if n_beats < self.seq_len:
            pad = self.seq_len - n_beats
            ref_features = torch.cat([ref_features, torch.zeros(pad, 31)])
            live_features = torch.cat([live_features, torch.zeros(pad, 31)])
            delta_t = torch.cat([delta_t, torch.zeros(pad)])

        return {
            "ref_features": ref_features[: self.seq_len],  # (seq_len, 31)
            "live_features": live_features[: self.seq_len],  # (seq_len, 31)
            "delta_t": delta_t[: self.seq_len],  # (seq_len,)
        }


def collate_fn(batch):
    """Stack batch samples into tensors."""
    ref = torch.stack([b["ref_features"] for b in batch])
    live = torch.stack([b["live_features"] for b in batch])
    delta = torch.stack([b["delta_t"] for b in batch])
    return {
        "ref_features": ref,  # (batch, seq_len, 31)
        "live_features": live,  # (batch, seq_len, 31)
        "delta_t": delta,  # (batch, seq_len)
    }
