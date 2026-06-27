"""
training/onset_network.py
===========================
Small causal CNN for per-frame onset-probability prediction.

Causal = each output frame only ever sees past + current input frames,
never future ones — required since at inference time we're processing
a live, ongoing audio stream and don't have "future" samples yet.
Achieved via left-padding before each conv instead of symmetric padding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D conv that only looks backward in time, via manual left-padding."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=0)

    def forward(self, x):  # x: (batch, ch, time)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class OnsetNet(nn.Module):
    """
    Input:  (batch, n_mels, time)        — log-mel spectrogram
    Output: (batch, time)                 — per-frame onset probability
    """

    def __init__(self, n_mels: int = 80, hidden: int = 32):
        super().__init__()
        # Dilated causal conv stack — widening receptive field without
        # needing a large kernel or a recurrent layer.
        self.net = nn.Sequential(
            CausalConv1d(n_mels, hidden, kernel_size=5, dilation=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            CausalConv1d(hidden, hidden, kernel_size=5, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            CausalConv1d(hidden, hidden, kernel_size=5, dilation=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            CausalConv1d(hidden, hidden, kernel_size=5, dilation=8),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, mel):  # mel: (batch, n_mels, time)
        x = self.net(mel)
        logits = self.head(x).squeeze(1)  # (batch, time)
        return logits  # raw logits — apply sigmoid outside (BCEWithLogits)
