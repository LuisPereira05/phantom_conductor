import librosa
import numpy as np
import torch


class TimingFeatureExtractor:
    """Extracts 31-D beat-synchronous features from audio."""

    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.feature_dim = 31  # 13 MFCC + 12 Chroma + 6 rhythm/timbre

    def extract(self, y: np.ndarray) -> torch.Tensor:
        if len(y) < self.sr * 2:
            y = np.pad(y, (0, self.sr * 2 - len(y)))

        y = y.astype(np.float32)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=128, hop_length=512
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=512)
        if len(beat_frames) < 2:
            beat_interval = int(self.sr * 60 / 120 / 512)
            beat_frames = np.arange(0, mel_spec.shape[1], beat_interval)

        features = []

        # 1. MFCC (13-D) — timbre
        mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=13)
        features.append(librosa.util.sync(mfcc, beat_frames, aggregate=np.mean))

        # 2. Chroma (12-D) — harmony
        chroma = librosa.feature.chroma_stft(S=log_mel, sr=self.sr)
        features.append(librosa.util.sync(chroma, beat_frames, aggregate=np.mean))

        # 3. Onset strength (1-D)
        onset = librosa.onset.onset_strength(S=log_mel, sr=self.sr)
        features.append(
            librosa.util.sync(onset.reshape(1, -1), beat_frames, aggregate=np.mean)
        )

        # 4. RMS energy (1-D)
        rms = librosa.feature.rms(S=mel_spec)
        features.append(librosa.util.sync(rms, beat_frames, aggregate=np.mean))

        # 5. Spectral flux (1-D)
        flux = np.abs(np.diff(mel_spec, axis=1, prepend=mel_spec[:, :1]))
        features.append(
            librosa.util.sync(
                flux.mean(axis=0).reshape(1, -1), beat_frames, aggregate=np.mean
            )
        )

        # 6. Zero-crossing rate (1-D)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)
        features.append(librosa.util.sync(zcr, beat_frames, aggregate=np.mean))

        # 7. Spectral centroid (1-D)
        cent = librosa.feature.spectral_centroid(S=log_mel, sr=self.sr)
        features.append(librosa.util.sync(cent, beat_frames, aggregate=np.mean))

        # 8. Spectral rolloff (1-D)
        rolloff = librosa.feature.spectral_rolloff(S=log_mel, sr=self.sr)
        features.append(librosa.util.sync(rolloff, beat_frames, aggregate=np.mean))

        # Stack and normalize
        combined = np.vstack(features).T  # (n_beats, 31)
        mean = combined.mean(axis=0, keepdims=True)
        std = combined.std(axis=0, keepdims=True) + 1e-8
        normalized = (combined - mean) / std

        return torch.from_numpy(normalized).float()
