"""Stratified 5-fold cross-validation training (PyTorch).

Loads the .npz produced by `dataset.prepare_data`, runs 5-fold CV, and saves:

    runs/<name>/fold_{k}.pt            # per-fold checkpoint (val-best)
    runs/<name>/fold_best.pt           # overall best fold (highest val acc)
    runs/<name>/scaler.pkl             # StandardScaler fit on the full dataset
    runs/<name>/labels.json            # class names in index order
    runs/<name>/metrics.json           # per-fold accuracy + mean ± std
    runs/<name>/training_curves.png    # train / val accuracy across folds

Each checkpoint is a dict: state_dict + the arch dims needed to rebuild the
model (time_steps, feature_dim, num_classes) + the label list — so inference
code can load it standalone.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainConfig
from .model import build_model


def scale_fold(scaler: StandardScaler, X: np.ndarray, fit: bool) -> np.ndarray:
    """StandardScaler over the per-frame feature axis.

    X has shape (samples, time, features). Each feature column is standardized,
    so we flatten over (samples, time) for fit/transform and restore the shape.
    """
    s, t, a = X.shape
    flat = X.reshape(s * t, a)
    out = scaler.fit_transform(flat) if fit else scaler.transform(flat)
    return out.reshape(s, t, a).astype(np.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            count += xb.size(0)
    return total_loss / count, correct / count


def plot_curves(fold_histories: List[dict], out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Pad ragged histories (early stopping ends folds at different epochs).
    max_len = max(len(h["val_acc"]) for h in fold_histories)

    def pad(seq):
        return np.array(seq + [seq[-1]] * (max_len - len(seq)))

    train = np.stack([pad(h["train_acc"]) for h in fold_histories])
    val = np.stack([pad(h["val_acc"]) for h in fold_histories])
    epochs = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train.mean(0), label="train acc", color="C0")
    ax.fill_between(epochs, train.min(0), train.max(0), color="C0", alpha=0.2)
    ax.plot(epochs, val.mean(0), label="val acc", color="C3", linestyle="--")
    ax.fill_between(epochs, val.min(0), val.max(0), color="C3", alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, help=".npz from prepare_data")
    ap.add_argument("--out", required=True, help="Output run directory")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--patience", type=int, default=30,
                    help="Early-stopping patience on val accuracy")
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is present")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    data = np.load(args.features, allow_pickle=False)
    X = data["X"].astype(np.float32)            # (S, T, A)
    y = data["y"].astype(np.int64)              # (S,)
    labels = [str(s) for s in data["labels"]]
    num_classes = len(labels)
    _, time_steps, feat_dim = X.shape
    print(f"Loaded X={X.shape}  y={y.shape}  classes={labels}  device={device}")

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.folds is not None:
        cfg.folds = args.folds

    torch.manual_seed(cfg.random_state)
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.random_state)

    fold_accs: List[float] = []
    histories: List[dict] = []
    best_fold = -1
    best_acc = -1.0

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n=== Fold {fold}/{cfg.folds}  train={len(train_idx)}  val={len(val_idx)} ===")
        scaler = StandardScaler()
        X_train = scale_fold(scaler, X[train_idx], fit=True)
        X_val = scale_fold(scaler, X[val_idx], fit=False)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y[train_idx])),
            batch_size=cfg.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y[val_idx])),
            batch_size=cfg.batch_size, shuffle=False,
        )

        torch.manual_seed(cfg.random_state + fold)
        model = build_model(time_steps, feat_dim, num_classes, cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        criterion = nn.CrossEntropyLoss()

        history = {"train_acc": [], "val_acc": []}
        fold_best_acc = -1.0
        fold_best_state = None
        epochs_since_improve = 0

        for epoch in range(1, cfg.epochs + 1):
            _, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)

            if va_acc > fold_best_acc:
                fold_best_acc = va_acc
                fold_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}  train_acc={tr_acc:.4f}  "
                      f"val_acc={va_acc:.4f}  best={fold_best_acc:.4f}")
            if epochs_since_improve >= args.patience:
                print(f"  early stop at epoch {epoch} (no val gain for {args.patience})")
                break

        histories.append(history)
        fold_accs.append(fold_best_acc)
        print(f"Fold {fold} best val_acc = {fold_best_acc:.4f}")

        checkpoint = {
            "state_dict": fold_best_state,
            "time_steps": time_steps,
            "feature_dim": feat_dim,
            "num_classes": num_classes,
            "labels": labels,
            "config": cfg.__dict__,
        }
        torch.save(checkpoint, out_dir / f"fold_{fold}.pt")
        if fold_best_acc > best_acc:
            best_acc = fold_best_acc
            best_fold = fold

    # Promote the winning fold and refit the scaler on the full dataset so the
    # deployed model sees consistent normalization regardless of best fold.
    best_ckpt = torch.load(out_dir / f"fold_{best_fold}.pt", map_location="cpu",
                           weights_only=False)
    torch.save(best_ckpt, out_dir / "fold_best.pt")

    final_scaler = StandardScaler()
    scale_fold(final_scaler, X, fit=True)
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(final_scaler, f)

    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    metrics = {
        "fold_val_acc": fold_accs,
        "mean_val_acc": float(np.mean(fold_accs)),
        "std_val_acc": float(np.std(fold_accs)),
        "best_fold": best_fold,
        "best_val_acc": best_acc,
        "config": cfg.__dict__,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_curves(histories, out_dir / "training_curves.png")

    print("\n=== Summary ===")
    print(f"Fold accs : {[f'{a:.4f}' for a in fold_accs]}")
    print(f"Mean ± Std: {metrics['mean_val_acc']:.4f} ± {metrics['std_val_acc']:.4f}")
    print(f"Best fold : {best_fold}  ({best_acc:.4f})")
    print(f"Artifacts → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
