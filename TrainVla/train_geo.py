"""
train_geo.py
============
Trains an emotion classifier on FACS-grouped geometric features
(pairwise Euclidean distances + angles between facial landmarks),
adapting the method from:

  Köksal & Gumus, "Deep Learning-Based Real-Time Sequential Facial
  Expression Analysis Using Geometric Features", arXiv:2512.05669, 2025.

Original paper
--------------
  * MediaPipe FaceMesh → 478 landmarks.
  * Manually selected landmark subsets (61 / 122 / 250).
  * Each subset is partitioned into FACS-aligned groups (cat 1–5)
    to act as feature selection (paper, Table 2/3).
  * For every pair (i,j) inside a category, compute:
        d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        θ_ij = arctan( (x_i - x_j) / (y_i - y_j) )
  * Concatenate to form the feature vector.
  * Paper uses 5-frame sliding window + ConvLSTM1D for temporal dynamics.

Adaptation here
---------------
Our datasets (AffectNet, RAF-DB) are STATIC images, not sequences, so the
ConvLSTM1D / temporal-difference branch from the paper does not apply.
We keep the geometric feature extractor (the paper's principal contribution)
and feed the static feature vector into an MLP classifier — directly
comparable, on the same data, to the other methods under TrainVla/.

Dataset
-------
Expects .npz landmark files from Dataset/Dataset.py (mode=landmarks):
  landmarks   : (N, 478, 3)
  labels      : (N,)
  label_names : optional

Usage
-----
    python train_geo.py                          # default: 61 landmarks + AU grouping
    python train_geo.py --landmarks 122
    python train_geo.py --landmarks 250 --no-grouping
"""

from __future__ import annotations

import os
import json
import argparse
import logging
from pathlib import Path
from glob import glob
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from bench_logger import BenchLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNIFIED_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(UNIFIED_EMOTIONS)
LABEL_REMAP = {
    "anger": 0, "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3, "happiness": 3,
    "neutral": 4,
    "sad": 5, "sadness": 5,
    "surprise": 6,
}


# ─────────────────────────────────────────────────────────────────────────────
# Landmark subsets (paper Fig. 6 — selected from the 478 FaceMesh points,
# concentrated on eye, eyebrow, nose, mouth, and chin regions tied to AUs)
# ─────────────────────────────────────────────────────────────────────────────
# These are MediaPipe indices for landmark points biased toward facial action
# regions. They are approximations of the paper's manual selection — different
# enough from "all 478" to act as a feature-selection step.

# Region anchors (MediaPipe FaceLandmarker v2)
_LEFT_EYE  = [33, 133, 159, 145, 153, 154, 155, 173, 246, 161, 160, 158, 144, 163]
_RIGHT_EYE = [362, 263, 386, 374, 380, 381, 382, 398, 466, 388, 387, 385, 373, 390]
_LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
_NOSE      = [1, 2, 4, 5, 6, 19, 94, 168, 195, 197, 274, 44, 45, 220, 49]
_MOUTH     = [13, 14, 17, 78, 308, 0, 11, 12, 15, 16, 61, 291, 84, 314,
              81, 311, 178, 402, 87, 317]
_LOWER_JAW = [152, 175, 199, 200, 18, 83, 313, 421, 200, 396, 369, 396]


def _take(points: list[int], n: int) -> list[int]:
    """Deterministically pick n indices from a region list (ordered by paper layout)."""
    if n >= len(points):
        return points[:]
    step = max(1, len(points) // n)
    return points[::step][:n]


def landmark_subset(n: int) -> list[int]:
    """
    Build a subset of `n` landmark indices approximating Köksal & Gumus
    Fig. 6 (61 / 122 / 250). Distributes points proportionally across the
    eye/brow/nose/mouth/jaw regions tied to action units.
    """
    region_pool = [
        ("left_eye",  _LEFT_EYE,   0.20),
        ("right_eye", _RIGHT_EYE,  0.20),
        ("left_brow", _LEFT_BROW,  0.10),
        ("right_brow",_RIGHT_BROW, 0.10),
        ("nose",      _NOSE,       0.15),
        ("mouth",     _MOUTH,      0.20),
        ("lower_jaw", _LOWER_JAW,  0.05),
    ]
    picked: list[int] = []
    for _, pts, ratio in region_pool:
        picked.extend(_take(pts, max(1, int(n * ratio))))
    picked = sorted(set(picked))
    return picked[:n]


# ─────────────────────────────────────────────────────────────────────────────
# FACS-aligned category groups (paper Table 3)
# Pair generation is restricted to within each category, mirroring the paper's
# AU grouping — this is the feature-selection step.
# ─────────────────────────────────────────────────────────────────────────────

def build_categories(subset: list[int], grouping: bool) -> list[list[int]]:
    """Return a list of landmark-index categories used for pair generation."""
    if not grouping:
        return [subset]
    s = set(subset)
    return [
        sorted(s & (set(_LEFT_EYE) | set(_LEFT_BROW)
                    | set(_RIGHT_EYE) | set(_RIGHT_BROW))),       # cat 1
        sorted(s & (set(_LEFT_EYE) | set(_RIGHT_EYE) | set(_NOSE))),  # cat 2
        sorted(s & (set(_LEFT_EYE) | set(_LEFT_BROW)
                    | set(_RIGHT_EYE) | set(_RIGHT_BROW)
                    | set(_NOSE))),                                # cat 3
        sorted(s & (set(_NOSE) | set(_MOUTH) | set(_LOWER_JAW))),  # cat 4
        sorted(s & (set(_LEFT_EYE) | set(_RIGHT_EYE)
                    | set(_NOSE) | set(_MOUTH))),                  # cat 5
    ]


def build_pair_list(subset: list[int], grouping: bool) -> list[tuple[int, int]]:
    """Return the unique ordered pairs (i,j) used to compute (distance, angle)."""
    cats = build_categories(subset, grouping)
    seen = set()
    pairs: list[tuple[int, int]] = []
    for c in cats:
        for i, j in combinations(c, 2):
            key = (i, j) if i < j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Geometric feature extraction  (paper Eqs. 1 & 2)
# ─────────────────────────────────────────────────────────────────────────────

def geometric_features(
    landmarks: np.ndarray,
    pairs: list[tuple[int, int]],
) -> np.ndarray:
    """
    landmarks : (478, 3) — only (x, y) used per paper Eqs. 1 & 2
    pairs     : pair list from build_pair_list
    Returns   : float32 (2 * len(pairs),) — distances then angles, concatenated.
    """
    xy = landmarks[:, :2]
    out = np.empty(2 * len(pairs), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        dx = xy[i, 0] - xy[j, 0]
        dy = xy[i, 1] - xy[j, 1]
        out[k] = float(np.hypot(dx, dy))                       # Eq. 1
        out[len(pairs) + k] = float(np.arctan2(dx, dy + 1e-9)) # Eq. 2
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GeoFeatureDataset(Dataset):
    def __init__(
        self,
        npz_paths: list[str],
        pairs: list[tuple[int, int]],
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
        augment: bool = False,
    ):
        self.pairs = pairs
        self.augment = augment

        all_feats, all_labels = [], []
        for path in npz_paths:
            data = np.load(path, allow_pickle=True)
            if "landmarks" not in data:
                raise ValueError(
                    f"{path} has no 'landmarks' key — re-run the converter "
                    f"with mode=landmarks."
                )
            lm = data["landmarks"]
            lab = data["labels"]
            lnames = data["label_names"].tolist() if "label_names" in data else None

            for i in range(len(lm)):
                raw = int(lab[i])
                if lnames is not None:
                    name = lnames[raw].lower()
                    if name not in LABEL_REMAP:
                        continue
                    unified = LABEL_REMAP[name]
                else:
                    if raw >= NUM_CLASSES:
                        continue
                    unified = raw
                all_feats.append(geometric_features(lm[i], pairs))
                all_labels.append(unified)

        self.X = np.stack(all_feats, axis=0).astype(np.float32)
        self.y = np.array(all_labels, dtype=np.int64)

        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X).astype(np.float32)
        elif scaler is not None:
            self.scaler = scaler
            self.X = scaler.transform(self.X).astype(np.float32)
        else:
            self.scaler = None

        dist = {
            UNIFIED_EMOTIONS[c]: int((self.y == c).sum())
            for c in range(NUM_CLASSES) if (self.y == c).sum() > 0
        }
        log.info("Loaded %d samples  feat_dim=%d  dist=%s",
                 len(self.y), self.X.shape[1], dist)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            # Very mild Gaussian jitter on the geometric features
            x = x + np.random.normal(0, 0.005, x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.y, minlength=NUM_CLASSES).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts
        return torch.tensor(w / w.sum(), dtype=torch.float32)

    def sample_weights(self) -> np.ndarray:
        return self.class_weights().numpy()[self.y]


# ─────────────────────────────────────────────────────────────────────────────
# MLP classifier on geometric features
# (paper uses ConvLSTM1D for sequential frames; we use MLP because our data is
# static.  This is the spatial-only adaptation noted in the docstring.)
# ─────────────────────────────────────────────────────────────────────────────

class GeoMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),

            nn.Linear(128, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(data_dir: str, pairs, batch: int, workers: int):
    all_files = sorted(glob(os.path.join(data_dir, "*.npz")))
    if not all_files:
        raise FileNotFoundError(f"No .npz files in {data_dir}")

    splits = defaultdict(list)
    for f in all_files:
        n = Path(f).stem.lower()
        if "train" in n:
            splits["train"].append(f)
        elif "val" in n or "validation" in n:
            splits["val"].append(f)
        elif "test" in n:
            splits["test"].append(f)
        else:
            splits["train"].append(f)

    log.info("Files — train:%d val:%d test:%d",
             len(splits["train"]), len(splits["val"]), len(splits["test"]))

    train_ds = GeoFeatureDataset(splits["train"], pairs, fit_scaler=True, augment=True)
    scaler = train_ds.scaler
    val_ds  = GeoFeatureDataset(splits["val"],  pairs, scaler=scaler) if splits["val"]  else None
    test_ds = GeoFeatureDataset(splits["test"], pairs, scaler=scaler) if splits["test"] else None

    sw = train_ds.sample_weights()
    sampler = WeightedRandomSampler(torch.from_numpy(sw).float(), len(sw), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                              num_workers=workers, pin_memory=True)
    val_loader  = (DataLoader(val_ds,  batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=True) if val_ds  else None)
    test_loader = (DataLoader(test_ds, batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=True) if test_ds else None)
    return train_loader, val_loader, test_loader, scaler


def run_epoch(model, loader, criterion, optimizer, phase):
    is_train = phase == "train"
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, out_dir: Path):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="evaluate", leave=False):
            preds_all.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy().tolist())
            labels_all.extend(y.numpy().tolist())
    preds_all  = np.array(preds_all)
    labels_all = np.array(labels_all)
    acc = accuracy_score(labels_all, preds_all)
    f1  = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
    present = sorted(set(labels_all) | set(preds_all))
    names   = [UNIFIED_EMOTIONS[i] for i in present]
    report  = classification_report(labels_all, preds_all,
                                    labels=present, target_names=names,
                                    zero_division=0)
    log.info("\n%s", report)
    log.info("Accuracy=%.4f  Weighted-F1=%.4f", acc, f1)
    (out_dir / "evaluation_report.txt").write_text(
        f"Accuracy : {acc:.4f}\nWeighted F1 : {f1:.4f}\n\n{report}"
    )

    cm = confusion_matrix(labels_all, preds_all, labels=present)
    cm_norm = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, fmt in zip(
        axes, [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=names, yticklabels=names,
                    ax=ax, linewidths=0.4)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return acc, f1


def plot_training_curves(history: dict, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    if any(v is not None for v in history["val_loss"]):
        ax1.plot(epochs, history["val_loss"], label="Val", linewidth=2, linestyle="--")
    ax1.set_title("Loss", fontweight="bold"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], label="Train", linewidth=2)
    if any(v is not None for v in history["val_acc"]):
        ax2.plot(epochs, history["val_acc"], label="Val", linewidth=2, linestyle="--")
    ax2.set_title("Accuracy", fontweight="bold"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir  = Path(args.output)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    method_name = f"geo_{args.landmarks}{'_au' if not args.no_grouping else ''}"
    bench = BenchLogger(method=method_name, output_dir=out_dir, config=vars(args))

    log.info("Device : %s", DEVICE)
    log.info("Landmark subset : %d points, grouping=%s",
             args.landmarks, not args.no_grouping)

    subset = landmark_subset(args.landmarks)
    pairs  = build_pair_list(subset, grouping=not args.no_grouping)
    log.info("Selected %d landmark indices → %d unique pairs  → feature_dim=%d",
             len(subset), len(pairs), 2 * len(pairs))

    train_loader, val_loader, test_loader, scaler = make_loaders(
        args.data, pairs, args.batch, args.workers
    )

    in_dim = 2 * len(pairs)
    model  = GeoMLP(in_dim=in_dim, num_classes=NUM_CLASSES, dropout=args.dropout).to(DEVICE)
    bench.log_model(model)

    cw        = train_loader.dataset.class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, patience_count = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        bench.epoch_begin(epoch)
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, "train")
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)

        if val_loader:
            va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, "val")
            history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
            monitor, monitor_loss = va_acc, va_loss
        else:
            va_loss, va_acc = None, None
            history["val_loss"].append(None); history["val_acc"].append(None)
            monitor, monitor_loss = tr_acc, None

        scheduler.step()
        bench.epoch_end(epoch, tr_loss, tr_acc, va_loss, va_acc,
                        lr=optimizer.param_groups[0]["lr"])

        if monitor > best_acc:
            best_acc = monitor
            patience_count = 0
            torch.save({"model": model.state_dict(),
                        "scaler": scaler,
                        "pairs": pairs,
                        "config": vars(args)},
                       ckpt_dir / "best.pt")
            log.info("  ✓ New best=%.4f  saved → %s/best.pt", best_acc, ckpt_dir)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    torch.save({"model": model.state_dict(),
                "scaler": scaler,
                "pairs": pairs,
                "config": vars(args)},
               ckpt_dir / "final.pt")
    plot_training_curves(history, out_dir)

    log.info("Loading best checkpoint for evaluation…")
    state = torch.load(ckpt_dir / "best.pt", map_location=DEVICE)
    model.load_state_dict(state["model"])

    eval_loader = test_loader or val_loader
    acc, f1 = best_acc, 0.0
    if eval_loader:
        acc, f1 = evaluate(model, eval_loader, out_dir)

    dummy = torch.zeros(1, in_dim, device=DEVICE)
    bench.benchmark_inference(model, (dummy,), runs=200, warmup=20)
    bench.finalize(final_accuracy=acc, final_weighted_f1=f1, best_val_accuracy=best_acc)

    summary = {
        "method": method_name,
        "best_val_acc": best_acc,
        "final_test_acc": acc,
        "final_weighted_f1": f1,
        "feature_dim": in_dim,
        "pair_count": len(pairs),
        "landmark_subset_size": len(subset),
        "grouping": not args.no_grouping,
        "paper_accuracy_ck+": 0.93,   # Köksal & Gumus, CK+ 5-fold
        "hyperparams": vars(args),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Geometric-features facial emotion classifier "
                    "(Köksal & Gumus, arXiv 2512.05669, 2025)"
    )
    parser.add_argument("--data", default="../facemesh_dataset",
                        help="Folder with .npz landmark files "
                             "(mode=landmarks or mode=both; "
                             "default: ../facemesh_dataset)")
    parser.add_argument("--output", default="./geo_output",
                        help="Output folder (default: ./geo_output)")
    parser.add_argument("--landmarks", type=int, default=61, choices=[61, 122, 250],
                        help="Landmark subset size (paper Fig. 6): 61, 122, or 250")
    parser.add_argument("--no-grouping", action="store_true",
                        help="Disable FACS-based AU grouping (use all pairs)")
    parser.add_argument("--epochs",   type=int, default=80)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--dropout",  type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers",  type=int, default=4)
    args = parser.parse_args()
    main(args)
