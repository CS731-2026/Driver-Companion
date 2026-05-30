"""
train_blendshape_emonet.py
==========================
Trains and evaluates an emotion classifier on MediaPipe
Face Blendshape coefficients, implementing the approach from:

  Jakhete & Kulkarni, "A Comprehensive Survey and Evaluation of
  MediaPipe Face Mesh for Human Emotion Recognition", ICCUBEA 2024.

What the paper does
-------------------
  1. MediaPipe FaceLandmarker extracts 52 blendshape coefficients per frame
     (named scores like mouthSmileLeft, browDownLeft, jawOpen, etc.).
  2. These coefficients directly encode muscle-movement semantics that map
     onto Ekman's 6 basic emotions (Table I of the paper).
  3. A deep classifier is trained on these 52-dim vectors.

Paper's key results
-------------------
  MediaPipe blendshapes → 98.8% accuracy
  (vs DLIB 96.5%, OpenPose 94.7%)

Architecture
------------
The paper uses blendshapes as input features but leaves the exact
network unspecified (shows a code snippet with thresholding logic).
We implement a well-regularised MLP — the natural fit for a 52-dim
tabular input — and add an attention-weighted variant that learns which
blendshapes matter most for each emotion (consistent with the paper's
discussion of muscle movements per emotion in Table I).

  BlendshapeMLP:
    FC(52 → 256, BN, ReLU, Dropout)
    FC(256 → 128, BN, ReLU, Dropout)
    FC(128 → 64,  BN, ReLU, Dropout)
    FC(64  → num_classes)

  BlendshapeAttentionNet (optional --model attention):
    Attention layer learns per-blendshape importance weights
    → weighted 52-dim input → same MLP tower

Dataset
-------
Expects .npz files from convert_to_facemesh_dataset.py --mode blendshapes:
  blendshapes     : float32 (N, 52)
  blendshape_names: str array (52,)
  labels          : int64   (N,)
  label_names     : str array (optional)

Usage
-----
    pip install torch scikit-learn matplotlib seaborn tqdm numpy

    # Run converter first:
    python convert_to_facemesh_dataset.py --mode blendshapes

    # Then train:
    python train_blendshape_emonet.py
    python train_blendshape_emonet.py --model attention --epochs 60
    python train_blendshape_emonet.py --eval-only --checkpoint ./output/best.pt
"""

import os
import json
import argparse
import logging
from pathlib import Path
from glob import glob
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm

from .bench_logger import BenchLogger

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Emotion / label configuration
# ─────────────────────────────────────────────────────────────────────────────
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

# Canonical MediaPipe blendshape names (52 total)
BLENDSHAPE_NAMES = [
    "neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward",
    "jawLeft", "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
NUM_BLENDSHAPES = len(BLENDSHAPE_NAMES)   # 52

# Paper Table I: which blendshapes are most diagnostic per emotion
EMOTION_KEY_BLENDSHAPES = {
    "happy":    ["mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft",
                 "cheekSquintRight", "mouthDimpleLeft", "mouthDimpleRight"],
    "sad":      ["browInnerUp", "browDownLeft", "browDownRight",
                 "mouthFrownLeft", "mouthFrownRight"],
    "surprise": ["browOuterUpLeft", "browOuterUpRight", "browInnerUp",
                 "eyeWideLeft", "eyeWideRight", "jawOpen"],
    "fear":     ["browInnerUp", "browOuterUpLeft", "browOuterUpRight",
                 "jawOpen", "eyeSquintLeft", "eyeSquintRight",
                 "mouthStretchLeft", "mouthStretchRight"],
    "angry":    ["browDownLeft", "browDownRight", "eyeSquintLeft",
                 "eyeSquintRight", "noseSneerLeft", "noseSneerRight",
                 "mouthPressLeft", "mouthPressRight"],
    "disgust":  ["noseSneerLeft", "noseSneerRight", "mouthFrownLeft",
                 "mouthFrownRight", "mouthLowerDownLeft", "mouthLowerDownRight"],
    "neutral":  ["neutral"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BlendshapeDataset(Dataset):
    """
    Loads .npz blendshape files, remaps labels, optionally scales features.
    Returns (blendshape_vector, label).
    """

    def __init__(
        self,
        npz_paths: list[str],
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
        augment: bool = False,
    ):
        self.augment = augment
        all_bs, all_labels = [], []

        for path in npz_paths:
            data   = np.load(path, allow_pickle=True)
            if "blendshapes" not in data:
                raise ValueError(
                    f"{path} has no 'blendshapes' key. "
                    "Re-run the converter with --mode blendshapes."
                )
            bs     = data["blendshapes"]      # (N, 52)
            labels = data["labels"]           # (N,)
            lnames = data["label_names"].tolist() if "label_names" in data else None

            for i in range(len(bs)):
                raw = int(labels[i])
                if lnames is not None:
                    name = lnames[raw].lower()
                    if name not in LABEL_REMAP:
                        continue
                    unified = LABEL_REMAP[name]
                else:
                    if raw >= NUM_CLASSES:
                        continue
                    unified = raw

                all_bs.append(bs[i])
                all_labels.append(unified)

        self.X = np.stack(all_bs, axis=0).astype(np.float32)   # (N, 52)
        self.y = np.array(all_labels, dtype=np.int64)

        # StandardScaler
        if fit_scaler:
            self.scaler = StandardScaler()
            self.X = self.scaler.fit_transform(self.X).astype(np.float32)
        elif scaler is not None:
            self.scaler = scaler
            self.X = scaler.transform(self.X).astype(np.float32)
        else:
            self.scaler = None

        class_dist = {
            UNIFIED_EMOTIONS[c]: int((self.y == c).sum())
            for c in range(NUM_CLASSES) if (self.y == c).sum() > 0
        }
        log.info("Loaded %d samples  dist=%s", len(self.y), class_dist)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            # Small Gaussian jitter on blendshape scores (clipped to [0,1])
            x = np.clip(x + np.random.normal(0, 0.01, x.shape), 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.y, minlength=NUM_CLASSES).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts
        return torch.tensor(w / w.sum(), dtype=torch.float32)

    def sample_weights(self) -> np.ndarray:
        return self.class_weights().numpy()[self.y]


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

class BlendshapeMLP(nn.Module):
    """
    Simple deep MLP on 52-dim blendshape coefficients.
    Fastest and most interpretable baseline — matches the paper's spirit.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_BLENDSHAPES, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class BlendshapeAttentionNet(nn.Module):
    """
    Adds a learned per-blendshape attention gate before the MLP.
    The gate scores reveal which blendshapes the network relies on
    per emotion — directly interpretable against Table I of the paper.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        # Attention gate: learns a scalar weight per blendshape
        self.attention = nn.Sequential(
            nn.Linear(NUM_BLENDSHAPES, NUM_BLENDSHAPES),
            nn.Tanh(),
            nn.Linear(NUM_BLENDSHAPES, NUM_BLENDSHAPES),
            nn.Sigmoid(),         # output in (0,1) — soft feature mask
        )

        self.mlp = nn.Sequential(
            nn.Linear(NUM_BLENDSHAPES, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),

            nn.Linear(64, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_attention: bool = False):
        attn = self.attention(x)          # (B, 52)  soft per-blendshape weight
        x_weighted = x * attn             # element-wise gating
        logits = self.mlp(x_weighted)
        if return_attention:
            return logits, attn
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(data_dir: str, batch: int, workers: int):
    """Discover .npz blendshape files, split by filename suffix."""
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

    train_ds = BlendshapeDataset(splits["train"], fit_scaler=True, augment=True)
    scaler   = train_ds.scaler

    val_ds  = BlendshapeDataset(splits["val"],  scaler=scaler) if splits["val"]  else None
    test_ds = BlendshapeDataset(splits["test"], scaler=scaler) if splits["test"] else None

    sw = train_ds.sample_weights()
    sampler = WeightedRandomSampler(
        torch.from_numpy(sw).float(), len(sw), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                              num_workers=workers, pin_memory=True)
    val_loader  = (DataLoader(val_ds,  batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=True) if val_ds  else None)
    test_loader = (DataLoader(test_ds, batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=True) if test_ds else None)

    return train_loader, val_loader, test_loader, scaler


def run_epoch(model, loader, criterion, optimizer, phase: str):
    is_train = phase == "train"
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss   = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation & plots
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, out_dir: Path):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="evaluate", leave=False):
            preds = model(X.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    present = sorted(set(all_labels) | set(all_preds))
    names   = [UNIFIED_EMOTIONS[i] for i in present]
    report  = classification_report(all_labels, all_preds,
                                    labels=present, target_names=names,
                                    zero_division=0)
    log.info("\n%s", report)
    log.info("Accuracy=%.4f  Weighted-F1=%.4f", acc, f1)
    (out_dir / "evaluation_report.txt").write_text(
        f"Accuracy : {acc:.4f}\nWeighted F1 : {f1:.4f}\n\n{report}"
    )

    # Confusion matrix (mirrors Fig 8 / Fig 10 of the paper)
    cm      = confusion_matrix(all_labels, all_preds, labels=present)
    cm_norm = cm.astype(float) / cm.sum(1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
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
    log.info("Confusion matrix → %s", out_dir / "confusion_matrix.png")
    return acc, f1, all_labels, all_preds


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


def plot_blendshape_importance(model, out_dir: Path):
    """
    For BlendshapeAttentionNet: visualise mean attention weights per blendshape,
    mirroring the blendshape bar charts in Figures 6 & 7 of the paper.
    """
    if not isinstance(model, BlendshapeAttentionNet):
        return

    model.eval()
    # Extract the learned attention weights directly from the linear layers
    # (weight matrix of the final sigmoid layer gives per-blendshape importance)
    with torch.no_grad():
        # Pass a uniform input to read out the attention gate
        dummy = torch.ones(1, NUM_BLENDSHAPES, device=DEVICE)
        _, attn = model(dummy, return_attention=True)
        weights = attn.squeeze().cpu().numpy()   # (52,)

    # Sort descending
    order = np.argsort(weights)[::-1]
    sorted_names   = [BLENDSHAPE_NAMES[i] for i in order]
    sorted_weights = weights[order]

    # Colour bars by emotion relevance
    emotion_colors = {
        "happy": "#2ecc71", "sad": "#3498db", "angry": "#e74c3c",
        "fear": "#9b59b6", "surprise": "#f39c12", "disgust": "#e67e22",
        "neutral": "#95a5a6",
    }
    bar_colors = []
    for name in sorted_names:
        color = "#cccccc"
        for emo, bs_list in EMOTION_KEY_BLENDSHAPES.items():
            if name in bs_list:
                color = emotion_colors.get(emo, "#cccccc")
                break
        bar_colors.append(color)

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(sorted_names)), sorted_weights,
                   color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Attention Weight")
    ax.set_title("Blendshape Importance (Attention Gate)\n"
                 "Colours indicate primary emotion association",
                 fontweight="bold")

    patches = [mpatches.Patch(color=c, label=e)
               for e, c in emotion_colors.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "blendshape_importance.png", dpi=150)
    plt.close(fig)
    log.info("Blendshape importance → %s", out_dir / "blendshape_importance.png")


def plot_per_class_blendshape_heatmap(dataset: BlendshapeDataset, out_dir: Path):
    """
    Mean blendshape activation per emotion class — shows the paper's
    muscle-movement patterns (Table I) emerging from the data.
    """
    means = np.zeros((NUM_CLASSES, NUM_BLENDSHAPES), dtype=np.float32)
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for i in range(len(dataset)):
        x, y = dataset[i]
        means[y.item()] += x.numpy()
        counts[y.item()] += 1

    for c in range(NUM_CLASSES):
        if counts[c] > 0:
            means[c] /= counts[c]

    present = [c for c in range(NUM_CLASSES) if counts[c] > 0]
    present_names = [UNIFIED_EMOTIONS[c] for c in present]
    data = means[present]                     # (n_present, 52)

    fig, ax = plt.subplots(figsize=(22, max(4, len(present) * 0.8)))
    sns.heatmap(
        data,
        xticklabels=BLENDSHAPE_NAMES,
        yticklabels=present_names,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.2,
        annot=False,
    )
    ax.set_title("Mean Blendshape Activation per Emotion\n"
                 "(Validates muscle-movement patterns from Paper Table I)",
                 fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    fig.savefig(out_dir / "blendshape_heatmap.png", dpi=150)
    plt.close(fig)
    log.info("Blendshape heatmap → %s", out_dir / "blendshape_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir  = Path(args.output)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    method_name = f"blendshape_{args.model}"
    bench = BenchLogger(method=method_name, output_dir=out_dir, config=vars(args))

    log.info("Device : %s", DEVICE)
    log.info("Model  : %s", args.model)

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, scaler = make_loaders(
        args.data, args.batch, args.workers
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if args.model == "attention":
        model = BlendshapeAttentionNet(NUM_CLASSES, args.dropout).to(DEVICE)
    else:
        model = BlendshapeMLP(NUM_CLASSES, args.dropout).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parameters: %s (%.3f M)", f"{n_params:,}", n_params / 1e6)
    bench.log_model(model)

    # ── Eval-only ─────────────────────────────────────────────────────────
    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint required with --eval-only")
        # weights_only=False: our own checkpoint bundles the sklearn StandardScaler
        # next to the state_dict, which PyTorch 2.6+ refuses to unpickle by default.
        state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        loader = test_loader or val_loader
        if loader is None:
            raise RuntimeError("No test or val split found.")
        evaluate(model, loader, out_dir)
        plot_blendshape_importance(model, out_dir)
        return

    # ── Generate data-level blendshape heatmap ────────────────────────────
    plot_per_class_blendshape_heatmap(train_loader.dataset, out_dir)

    # Class imbalance is already handled by the WeightedRandomSampler in the
    # train loader — adding class weights here too would double-correct toward
    # rare classes. Keep the sampler; drop the loss weight.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        bench.epoch_begin(epoch)
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, "train")
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)

        if val_loader:
            va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, "val")
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            monitor = va_acc
        else:
            va_loss = None
            history["val_loss"].append(None)
            history["val_acc"].append(None)
            monitor = tr_acc

        scheduler.step()
        bench.epoch_end(epoch, tr_loss, tr_acc, va_loss, monitor if val_loader else None,
                        lr=optimizer.param_groups[0]["lr"])

        log.info(
            "Epoch %3d/%d  tr_loss=%.4f  tr_acc=%.4f  "
            "val_acc=%.4f  lr=%.2e",
            epoch, args.epochs, tr_loss, tr_acc, monitor,
            optimizer.param_groups[0]["lr"],
        )

        if monitor > best_acc:
            best_acc = monitor
            patience_count = 0
            torch.save(
                {"model": model.state_dict(), "scaler": scaler, "config": vars(args)},
                ckpt_dir / "best.pt",
            )
            log.info("  ✓ New best=%.4f  saved → %s/best.pt", best_acc, ckpt_dir)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info("Early stopping at epoch %d (patience=%d)",
                         epoch, args.patience)
                break

    torch.save(
        {"model": model.state_dict(), "scaler": scaler, "config": vars(args)},
        ckpt_dir / "final.pt",
    )
    plot_training_curves(history, out_dir)

    # ── Final evaluation ──────────────────────────────────────────────────
    log.info("Loading best checkpoint for evaluation…")
    # weights_only=False: our own checkpoint bundles the sklearn StandardScaler
    # next to the state_dict, which PyTorch 2.6+ refuses to unpickle by default.
    state = torch.load(ckpt_dir / "best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])

    eval_loader = test_loader or val_loader
    acc, f1 = best_acc, 0.0
    if eval_loader:
        acc, f1, _, _ = evaluate(model, eval_loader, out_dir)

    # Inference latency benchmark on best model
    dummy_bs = torch.zeros(1, NUM_BLENDSHAPES, device=DEVICE)
    bench.benchmark_inference(model, (dummy_bs,), runs=200, warmup=20)
    bench.finalize(final_accuracy=acc, final_weighted_f1=f1,
                   best_val_accuracy=best_acc)

    # Attention / importance plots
    plot_blendshape_importance(model, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        "best_val_acc":     best_acc,
        "final_test_acc":   acc,
        "final_weighted_f1": f1,
        "model_type":       args.model,
        "num_classes":      NUM_CLASSES,
        "emotions":         UNIFIED_EMOTIONS,
        "blendshape_count": NUM_BLENDSHAPES,
        "model_params_M":   round(n_params / 1e6, 4),
        "paper_accuracy":   0.988,   # MediaPipe result from ICCUBEA 2024
        "hyperparams":      vars(args),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Summary → %s/run_summary.json", out_dir)
    log.info("Done. Best accuracy: %.4f  (Paper: 98.8%%)", best_acc)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blendshape-based emotion classifier "
                    "(Jakhete & Kulkarni, ICCUBEA 2024)"
    )
    parser.add_argument("--data", default="cache/facemesh_dataset",
                        help="Folder with .npz blendshape files "
                             "(default: cache/facemesh_dataset)")
    parser.add_argument("--output", default="runs/blendshape",
                        help="Output folder (default: runs/blendshape)")
    parser.add_argument("--model", default="mlp",
                        choices=["mlp", "attention"],
                        help="mlp = plain MLP; attention = MLP with "
                             "per-blendshape attention gate (default: mlp)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch",  type=int, default=64)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="",
                        help="Checkpoint path for --eval-only")
    args = parser.parse_args()
    main(args)