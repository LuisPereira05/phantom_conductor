import torch
import torch.nn as nn


class TimingAlignerNet(nn.Module):
    """
    Input:  [ref_features || live_features || |ref - live|]  (batch, seq, 93)
    Output: Δt per beat (seconds)                            (batch, seq)
    """

    def __init__(
        self,
        feature_dim: int = 31,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        max_offset: float = 0.5,
    ):
        super().__init__()

        input_dim = feature_dim * 3  # ref + live + diff

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            batch_first=True,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # Bound to [-1, 1]
        )

        self.max_offset = max_offset

    def forward(self, ref_features, live_features):
        diff = torch.abs(ref_features - live_features)
        x = torch.cat([ref_features, live_features, diff], dim=-1)  # (batch, seq, 93)

        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        lstm_out = self.norm1(lstm_out)

        # Causal self-attention (can't look ahead)
        seq_len = x.shape[1]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, attn_mask=mask)

        x = self.norm2(lstm_out + attn_out)
        x = self.ffn(x)
        delta_t = self.delta_head(x).squeeze(-1)  # (batch, seq)

        return delta_t * self.max_offset  # Scale to [-max_offset, max_offset]


class LightweightTimingNet(nn.Module):
    """CPU-friendly variant (~10× fewer parameters)."""

    def __init__(self, feature_dim: int = 31, hidden_dim: int = 64):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=feature_dim * 3,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.max_offset = 0.5

    def forward(self, ref, live):
        diff = torch.abs(ref - live)
        x = torch.cat([ref, live, diff], dim=-1)
        x, _ = self.encoder(x)
        return self.head(x).squeeze(-1) * self.max_offset
