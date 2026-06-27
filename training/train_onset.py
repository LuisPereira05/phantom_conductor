"""
training/train_onset.py
=========================
Trains OnsetNet on recorded (wav, onsets.json) takes.

Usage
-----
    python -m training.train_onset --data-dir ./training_data --epochs 60
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from onset_dataset import OnsetDataset
from onset_network import OnsetNet
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

train_losses = []
val_losses = []

precisions = []
recalls = []
f1s = []


def peak_pick(
    probs: torch.Tensor, threshold: float = 0.5, min_spacing_frames: int = 4
) -> list:
    """
    Extract discrete onset frame-indices from a probability curve:
    local maxima above `threshold`, at least `min_spacing_frames` apart.
    Used only for the eval metric below — the live system will use its
    own version of this at inference time.
    """
    probs = probs.detach().cpu().numpy()
    peaks = []
    last_peak = -min_spacing_frames
    for i in range(1, len(probs) - 1):
        if (
            probs[i] > threshold
            and probs[i] >= probs[i - 1]
            and probs[i] >= probs[i + 1]
        ):
            if i - last_peak >= min_spacing_frames:
                peaks.append(i)
                last_peak = i
    return peaks


def evaluate(
    model,
    loader,
    device,
    frame_time,
    criterion=None,
    tolerance_frames: int = 3,
):
    model.eval()

    tp, fp, fn = 0, 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(device)
            target = batch["target"].to(device)

            logits = model(mel)

            if criterion is not None:
                total_loss += criterion(logits, target).item()

            probs = torch.sigmoid(logits)

            for b in range(probs.shape[0]):
                pred_peaks = set(peak_pick(probs[b]))
                true_peaks = set(
                    i for i in range(target.shape[1]) if target[b, i].item() > 0.9
                )

                matched_true = set()

                for p in pred_peaks:
                    hit = False
                    for t in true_peaks:
                        if abs(p - t) <= tolerance_frames and t not in matched_true:
                            matched_true.add(t)
                            hit = True
                            break

                    if hit:
                        tp += 1
                    else:
                        fp += 1

                fn += len(true_peaks - matched_true)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    val_loss = total_loss / max(1, len(loader))

    return val_loss, precision, recall, f1


def train(
    data_dir: str,
    output_path: str = "./models/onset_net_v1.pt",
    epochs: int = 60,
    batch_size: int = 16,
    lr: float = 3e-4,
    val_fraction: float = 0.15,
    split_sec: float | None = None,
    finetune: bool = False,  # ← add this
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/4] Building dataset...")
    full_dataset = OnsetDataset(data_dir, split_sec=args.split_sec)

    val_size = max(1, int(len(full_dataset) * val_fraction))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print("\n[2/4] Initializing model...")
    model = OnsetNet(n_mels=80, hidden=32).to(device)

    if finetune and Path(output_path).exists():
        ckpt = torch.load(output_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_f1 = ckpt.get("f1", -1.0)
        print(
            f"  Finetuning from checkpoint — epoch={ckpt.get('epoch', '?')} F1={best_f1:.3f}"
        )
    elif finetune:
        print(
            f"  --finetune set but no checkpoint found at {output_path} — training from scratch"
        )
    else:
        print(f"  Training from scratch")

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Onsets are sparse — most frames are background (target ~0). Without
    # weighting, the easy win for the loss is "always predict near 0",
    # which would still report a low loss while being useless. Weight
    # positive-ish frames more heavily so the network actually has to
    # learn the attack pattern, not just the class prior.
    pos_weight = torch.tensor([4.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def lr_lambda(epoch):
        if epoch < 2:
            return (epoch + 1) / 2  # ramp from 50% to 100% over 2 epochs
        return 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    print("\n[3/4] Training...")
    best_f1 = -1.0
    patience_counter = 0
    early_stop_patience = 15  # epochs without improvement before stopping
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    from training.onset_dataset import FRAME_TIME

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            mel = batch["mel"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            logits = model(mel)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        val_loss, precision, recall, f1 = evaluate(
            model,
            val_loader,
            device,
            FRAME_TIME,
            criterion,
        )
        train_losses.append(avg_loss)
        val_losses.append(val_loss)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        if epoch <= 2:
            warmup.step()
        else:
            scheduler.step(avg_loss)

        print(
            f"  Epoch {epoch:3d}: "
            f"train={avg_loss:.4f} "
            f"val={val_loss:.4f} "
            f"val_P={precision:.3f}  val_R={recall:.3f}  val_F1={f1:.3f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "f1": best_f1,
                    "config": {"n_mels": 80, "hidden": 32},
                },
                output_path,
            )
            print(f"  → saved best model (F1={best_f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(
                    f"\n  Early stop at epoch {epoch} — no improvement for {early_stop_patience} epochs"
                )
                break

    print(f"\n[4/4] Done. Best val F1: {best_f1:.3f}")
    print(f"Model saved to: {output_path}")
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, precisions, label="Precision")
    plt.plot(epochs_range, recalls, label="Recall")
    plt.plot(epochs_range, f1s, label="F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    graph_path = Path(output_path).with_suffix(".png")
    plt.savefig(graph_path, dpi=200)
    plt.close()

    print(f"Training curves saved to: {graph_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", default="./models/onset_net_v1.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument(
        "--split-sec",
        type=float,
        default=None,
        help="Split each take into chunks of this length before training",
    )
    ap.add_argument(
        "--finetune",
        action="store_true",
        help="Load existing checkpoint and finetune instead of training from scratch",
    )
    args = ap.parse_args()

    train(
        data_dir=args.data_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        split_sec=args.split_sec,
        finetune=args.finetune,
    )
