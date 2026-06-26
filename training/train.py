"""
training/train.py
=================
Train the TimingAlignerNet on synthetic data.

Usage:
    python -m training.train --audio-dir ./audio --epochs 100
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from analysis.timing_network import LightweightTimingNet, TimingAlignerNet
from training.dataset import TimingOffsetDataset, collate_fn


def train(
    audio_dir: str,
    output_path: str = "./models/timing_aligner_v1.pt",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_lightweight: bool = False,
    seq_len: int = 32,
    max_offset_ms: float = 100.0,
):
    print(f"=" * 60)
    print(f"Phantom Conductor — Timing Model Training")
    print(f"Device: {device}")
    print(f"Model: {'Lightweight' if use_lightweight else 'Full'}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"=" * 60)

    # Create dataset
    print("\n[1/4] Building dataset...")
    dataset = TimingOffsetDataset(
        audio_dir=audio_dir,
        seq_len=seq_len,
        max_offset_ms=max_offset_ms,
        samples_per_song=200,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No audio files found in {audio_dir}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set >0 if not on Windows/Mac with PyTorch issues
        pin_memory=True if device == "cuda" else False,
    )

    # Create model
    print("\n[2/4] Initializing model...")
    if use_lightweight:
        model = LightweightTimingNet(feature_dim=31, hidden_dim=64)
        print(f"  Parameters: ~{sum(p.numel() for p in model.parameters()):,}")
    else:
        model = TimingAlignerNet(
            feature_dim=31,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            max_offset=max_offset_ms / 1000.0,
        )
        print(f"  Parameters: ~{sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device)

    # Loss function: weighted MSE
    # Later beats in the sequence matter more for real-time prediction
    def weighted_mse(pred, target):
        weights = torch.linspace(0.5, 1.0, pred.shape[1], device=device)
        loss = ((pred - target) ** 2) * weights.unsqueeze(0)
        return loss.mean()

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=True
    )

    # Training loop
    print("\n[3/4] Training...")
    best_loss = float("inf")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mae = 0.0  # Mean absolute error in ms

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch in pbar:
            ref = batch["ref_features"].to(device)
            live = batch["live_features"].to(device)
            target = batch["delta_t"].to(device)

            optimizer.zero_grad()
            pred = model(ref, live)

            loss = weighted_mse(pred, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            mae = torch.abs(pred - target).mean().item() * 1000  # ms
            total_mae += mae

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "mae": f"{mae:.1f}ms",
                }
            )

        avg_loss = total_loss / len(loader)
        avg_mae = total_mae / len(loader)
        scheduler.step(avg_loss)

        print(
            f"  Epoch {epoch:3d}: loss={avg_loss:.4f}  MAE={avg_mae:.1f}ms  lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": {
                        "feature_dim": 31,
                        "hidden_dim": 128 if not use_lightweight else 64,
                        "seq_len": seq_len,
                        "max_offset_ms": max_offset_ms,
                    },
                },
                output_path,
            )
            print(f"  → Saved best model (loss={best_loss:.4f})")

    print(f"\n[4/4] Training complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phantom Conductor timing model")
    parser.add_argument(
        "--audio-dir", required=True, help="Directory with backing tracks"
    )
    parser.add_argument("--output", default="./models/timing_aligner_v1.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lightweight", action="store_true", help="Use smaller model")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--max-offset", type=float, default=100.0, help="Max synthetic offset in ms"
    )

    args = parser.parse_args()

    train(
        audio_dir=args.audio_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_lightweight=args.lightweight,
        seq_len=args.seq_len,
        max_offset_ms=args.max_offset,
    )
