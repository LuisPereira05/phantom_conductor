# analysis/reference_cache.py
import json
import pickle
from pathlib import Path

import librosa
import numpy as np
import torch

from .feature_extractor import TimingFeatureExtractor


def cache_reference_features(
    audio_path: str,
    output_dir: str = "./reference_cache",
    sr: int = 22050,
) -> dict:
    """Extract and cache reference features. Run once per song."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    song_id = Path(audio_path).stem
    cache_path = output_dir / f"{song_id}.pkl"

    # Return cached if exists
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    extractor = TimingFeatureExtractor(sr=sr)
    features = extractor.extract(y)  # torch.Tensor (n_beats, 31)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    cache = {
        "features": features.numpy(),  # np.ndarray (n_beats, 31)
        "beat_times": beat_times,  # np.ndarray (n_beats,)
        "tempo": float(tempo),
        "sr": sr,
        "audio_path": audio_path,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    return cache


def load_reference_features(song_id: str, cache_dir: str = "./reference_cache") -> dict:
    """Load pre-computed reference features."""
    cache_path = Path(cache_dir) / f"{song_id}.pkl"
    with open(cache_path, "rb") as f:
        return pickle.load(f)
