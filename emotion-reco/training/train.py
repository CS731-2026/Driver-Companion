"""Stratified 5-fold cross-validation training (PyTorch).

The fold loop is exposed as importable functions (`train_one_fold`,
`cross_validate`, `evaluate`) so `benchmark.py` can reuse the exact same
training protocol across datasets.

`main()` trains one dataset cache and saves:

    runs/<name>/fold_{k}.pt            # per-fold checkpoint (val-best)
    runs/<name>/fold_best.pt           # overall best fold
    runs/<name>/scaler.pkl             # StandardScaler fit on the full dataset
    runs/<name>/labels.json            # class names in index order
    runs/<name>/metrics.json           # per-fold train/val accuracy + mean ± std
    runs/<name>/training_curves.png    # train / val accuracy across folds
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

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


def evaluate(
    model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device,
    batch_size: int = 32,
) -> float:
    """Classification accuracy of `model` on a (already-scaled) (X, y) set."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y)),
        batch_size=batch_size,
    )
    model.eval()
    correct = count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            count += xb.size(0)
    return correct / count


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    cfg: TrainConfig,
    device: torch.device,
    patience: int,
    verbose: bool = False,
) -> Dict:
    """Train on one (scaled) train/val split with early stopping on val acc.

    Returns the best val accuracy, the train accuracy *at that same epoch*, the
    best model state, and the full per-epoch history.
    """
    time_steps, feat_dim = X_train.shape[1], X_train.shape[2]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=cfg.batch_size, shuffle=False,
    )
    model = build_model(time_steps, feat_dim, num_classes, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "val_acc": []}
    best_val = -1.0
    train_at_best = 0.0
    best_state = None
    since_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        _, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        _, va_acc = run_epoch(model, val_loader, criterion, device)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_val:
            best_val = va_acc
            train_at_best = tr_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            since_improve = 0
        else:
            since_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}  train_acc={tr_acc:.4f}  "
                  f"val_acc={va_acc:.4f}  best={best_val:.4f}")
        if since_improve >= patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (no val gain for {patience})")
            break

    return {
        "best_val_acc": best_val,
        "train_acc_at_best": train_at_best,
        "best_state": best_state,
        "history": history,
        "time_steps": time_steps,
        "feature_dim": feat_dim,
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    labels: List[str],
    cfg: TrainConfig,
    device: torch.device,
    patience: int,
    verbose: bool = True,
) -> Dict:
    """Run stratified k-fold CV. StandardScaler is fit per fold on train only."""
    num_classes = len(labels)
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.random_state)

    folds: List[Dict] = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        if verbose:
            print(f"\n=== Fold {fold}/{cfg.folds}  train={len(tr)}  val={len(va)} ===")
        scaler = StandardScaler()
        X_tr = scale_fold(scaler, X[tr], fit=True)
        X_va = scale_fold(scaler, X[va], fit=False)
        torch.manual_seed(cfg.random_state + fold)
        res = train_one_fold(X_tr, y[tr], X_va, y[va], num_classes,
                             cfg, device, patience, verbose)
        res["fold"] = fold
        folds.append(res)
        if verbose:
            print(f"Fold {fold}: train={res['train_acc_at_best']:.4f}  "
                  f"val={res['best_val_acc']:.4f}")

    val_accs = [f["best_val_acc"] for f in folds]
    train_accs = [f["train_acc_at_best"] for f in folds]
    best_i = int(np.argmax(val_accs))
    return {
        "folds": folds,
        "fold_val_acc": val_accs,
        "fold_train_acc": train_accs,
        "mean_val_acc": float(np.mean(val_accs)),
        "std_val_acc": float(np.std(val_accs)),
        "mean_train_acc": float(np.mean(train_accs)),
        "best_fold": folds[best_i]["fold"],
        "best_val_acc": val_accs[best_i],
        "num_classes": num_classes,
        "time_steps": folds[0]["time_steps"],
        "feature_dim": folds[0]["feature_dim"],
    }


def plot_curves(histories: List[dict], out_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Pad ragged histories (early stopping ends folds at different epochs).
    max_len = max(len(h["val_acc"]) for h in histories)

    def pad(seq):
        return np.array(seq + [seq[-1]] * (max_len - len(seq)))

    train = np.stack([pad(h["train_acc"]) for h in histories])
    val = np.stack([pad(h["val_acc"]) for h in histories])
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


def load_cache(path: str) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Load an (X, y, labels) feature cache written by dataset.prepare_data."""
    data = np.load(path, allow_pickle=False)
    return (
        data["X"].astype(np.float32),
        data["y"].astype(np.int64),
        [str(s) for s in data["labels"]],
    )


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

    X, y, labels = load_cache(args.features)
    print(f"Loaded X={X.shape}  y={y.shape}  classes={labels}  device={device}")

    # Landmark config travels with the cache → into each checkpoint, so the
    # real-time demo rebuilds the exact same feature pipeline.
    meta = np.load(args.features, allow_pickle=False)
    selected_landmarks = (meta["selected_landmarks"].tolist()
                          if "selected_landmarks" in meta else None)
    pair_idx = meta["pair_idx"] if "pair_idx" in meta else None

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.folds is not None:
        cfg.folds = args.folds

    cv = cross_validate(X, y, labels, cfg, device, args.patience)

    # Save per-fold checkpoints.
    for f in cv["folds"]:
        torch.save(
            {
                "state_dict": f["best_state"],
                "time_steps": cv["time_steps"],
                "feature_dim": cv["feature_dim"],
                "num_classes": cv["num_classes"],
                "labels": labels,
                "config": cfg.__dict__,
                "selected_landmarks": selected_landmarks,
                "pair_idx": pair_idx,
            },
            out_dir / f"fold_{f['fold']}.pt",
        )
    best_ckpt = torch.load(out_dir / f"fold_{cv['best_fold']}.pt",
                           map_location="cpu", weights_only=False)
    torch.save(best_ckpt, out_dir / "fold_best.pt")

    # Refit the scaler on the full dataset for deployment.
    final_scaler = StandardScaler()
    scale_fold(final_scaler, X, fit=True)
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(final_scaler, f)

    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    metrics = {
        "fold_train_acc": cv["fold_train_acc"],
        "fold_val_acc": cv["fold_val_acc"],
        "mean_train_acc": cv["mean_train_acc"],
        "mean_val_acc": cv["mean_val_acc"],
        "std_val_acc": cv["std_val_acc"],
        "best_fold": cv["best_fold"],
        "best_val_acc": cv["best_val_acc"],
        "config": cfg.__dict__,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_curves([f["history"] for f in cv["folds"]], out_dir / "training_curves.png")

    print("\n=== Summary ===")
    print(f"Train acc : {[f'{a:.4f}' for a in cv['fold_train_acc']]}")
    print(f"Test  acc : {[f'{a:.4f}' for a in cv['fold_val_acc']]}")
    print(f"Mean test : {cv['mean_val_acc']:.4f} ± {cv['std_val_acc']:.4f}")
    print(f"Best fold : {cv['best_fold']}  ({cv['best_val_acc']:.4f})")
    print(f"Artifacts → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
