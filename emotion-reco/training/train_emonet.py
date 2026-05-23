"""
train_emonet.py
===============
Trains and evaluates EmoNet — a dual-stream deep learning model for
facial emotion recognition — on the FaceMesh landmark dataset produced
by convert_to_facemesh_dataset.py.

Architecture (adapted from DriveEmo-FL, Section 3.4)
-----------------------------------------------------
The paper uses two input streams:
  Stream A (spatial)  — 2D micro-Doppler spectrogram  → CNN
  Stream B (temporal) — 1D VTP feature vector          → MLP

For face landmarks we map these as follows:
  Stream A (spatial)  — landmark (x,y,z) coords reshaped to a 2D grid → CNN
  Stream B (geometric)— hand-crafted geometric features (inter-landmark
                        distances, facial ratios, symmetry scores)     → MLP

Both embeddings are fused, then passed to a shared MLP head that
predicts the emotion class (softmax).

Dataset layout expected
-----------------------
./facemesh_dataset/
    <ds>_<split>.npz   {"landmarks": (N,478,3), "labels": (N,),
                         "label_names": (...)}   [from converter script]

The trainer auto-detects splits from the filename ("train" / "val" substrings),
so both data sources below drop in with no code change:

    Dataset/Dataset.py          AffectNet-HQ + RAF-DB (legacy)
    Dataset/papers_dataset.py   CK+ (Köksal & Gumus 2025) + FER-2013
                                (Goodfellow et al. 2013; cited in
                                Jakhete & Kulkarni 2024 and Luan et al. 2025)
                                — stratified 80/20 train/val split, fixed seed

Usage
-----
    # Train on all .npz files found under ./facemesh_dataset/
    python train_emonet.py

    # Custom paths / hyperparams
    python train_emonet.py --data ./facemesh_dataset --epochs 40 --batch 64

    # Evaluate only (skip training)
    python train_emonet.py --eval-only --checkpoint ./checkpoints/best.pt

Requirements
------------
    pip install torch torchvision scikit-learn matplotlib seaborn tqdm
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

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

# ── Unified emotion label map ─────────────────────────────────────────────────
# Different source datasets use different integer schemes and spellings:
#   AffectNet-HQ (legacy)   8 classes, "anger"/"happy"/...
#   RAF-DB        (legacy)  7 classes
#   CK+           (papers)  7 classes incl. "contempt" (dropped here)
#   FER-2013      (papers)  7 classes, "angry"/"sad"/...
# We normalise everything to the shared 7-class Ekman vocabulary below.
# Any label not listed in LABEL_REMAP is dropped at load time.
UNIFIED_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(UNIFIED_EMOTIONS)

LABEL_REMAP = {
    # Covers AffectNet / FER-2013 / CK+ spellings
    "anger":    0, "angry":   0,
    "disgust":  1,
    "fear":     2,
    "happy":    3, "happiness": 3,
    "neutral":  4,
    "sad":      5, "sadness":   5,
    "surprise": 6,
    # "contempt" intentionally absent → dropped to preserve 7-class parity
}

# ─────────────────────────────────────────────────────────────────────────────
# Geometric feature extraction  (Stream B)
# ─────────────────────────────────────────────────────────────────────────────
# Key landmark indices from MediaPipe FaceLandmarker v2 (478 points)
_LM = {
    # Mouth
    "mouth_left":   61,  "mouth_right":  291,
    "mouth_top":    13,  "mouth_bottom": 14,
    # Eyes
    "left_eye_inner":  133, "left_eye_outer":  33,
    "left_eye_top":    159, "left_eye_bottom": 145,
    "right_eye_inner": 362, "right_eye_outer": 263,
    "right_eye_top":   386, "right_eye_bottom":374,
    # Eyebrows
    "left_brow_inner":  107, "left_brow_outer":  46,
    "right_brow_inner": 336, "right_brow_outer": 276,
    # Nose
    "nose_tip":     4,   "nose_base":    2,
    # Face outline
    "chin":         152, "forehead":     10,
    "left_cheek":   234, "right_cheek":  454,
}

# Pairs whose Euclidean distance is used as a feature
_DISTANCE_PAIRS = [
    ("mouth_left",  "mouth_right"),    # mouth width
    ("mouth_top",   "mouth_bottom"),   # mouth openness
    ("left_eye_inner",  "left_eye_outer"),   # left eye width
    ("right_eye_inner", "right_eye_outer"),  # right eye width
    ("left_eye_top",    "left_eye_bottom"),  # left eye height
    ("right_eye_top",   "right_eye_bottom"), # right eye height
    ("left_brow_inner", "left_brow_outer"),  # left brow span
    ("right_brow_inner","right_brow_outer"), # right brow span
    ("nose_tip",    "mouth_top"),      # nose-to-lip
    ("chin",        "forehead"),       # face height
    ("left_cheek",  "right_cheek"),    # face width
    ("left_eye_top","left_brow_inner"),  # brow-eye gap left
    ("right_eye_top","right_brow_inner"),# brow-eye gap right
]

GEO_DIM = len(_DISTANCE_PAIRS) + 4   # distances + 4 ratio features


def extract_geometric_features(landmarks: np.ndarray) -> np.ndarray:
    """
    landmarks : (478, 3)  normalised (x, y, z)
    Returns   : (GEO_DIM,) float32 geometric feature vector
    """
    feats = []

    # Euclidean distances (xy only — z is less reliable)
    for a, b in _DISTANCE_PAIRS:
        pa = landmarks[_LM[a], :2]
        pb = landmarks[_LM[b], :2]
        feats.append(float(np.linalg.norm(pa - pb)))

    # Normalise all distances by face height so they are scale-invariant
    face_h = feats[9] if feats[9] > 1e-6 else 1.0
    feats = [f / face_h for f in feats]

    # Ratio features
    mouth_w = feats[0]
    mouth_h = feats[1]
    eye_w_l = feats[2]
    eye_h_l = feats[4]
    eye_w_r = feats[3]
    eye_h_r = feats[5]

    feats.append(mouth_h / (mouth_w + 1e-6))          # mouth aspect ratio
    feats.append(eye_h_l / (eye_w_l + 1e-6))          # left eye aspect ratio
    feats.append(eye_h_r / (eye_w_r + 1e-6))          # right eye aspect ratio
    # Facial symmetry score (lower = more symmetric)
    feats.append(abs(eye_w_l - eye_w_r) / (eye_w_l + eye_w_r + 1e-6))

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FaceMeshEmotionDataset(Dataset):
    """
    Loads one or more .npz files, remaps labels to the unified vocabulary,
    and returns (spatial_grid, geo_features, label) per sample.

    spatial_grid : (1, 22, 22)  — landmarks projected onto a 2-D spatial grid
                   (22*22 = 484 ≥ 478, padded to fit)
    geo_features : (GEO_DIM,)   — hand-crafted geometric ratios
    label        : int
    """

    def __init__(self, npz_paths: list[str], augment: bool = False):
        self.augment = augment
        all_landmarks = []
        all_labels    = []

        for path in npz_paths:
            data = np.load(path, allow_pickle=True)
            lm   = data["landmarks"]          # (N, 478, 3)
            lab  = data["labels"]             # (N,)
            lnames = (
                data["label_names"].tolist()
                if "label_names" in data else None
            )

            for i in range(len(lm)):
                raw_label = int(lab[i])
                # Resolve name → unified index
                if lnames is not None:
                    name = lnames[raw_label].lower()
                    if name not in LABEL_REMAP:
                        continue   # drop unrecognised classes (e.g. contempt)
                    unified = LABEL_REMAP[name]
                else:
                    if raw_label >= NUM_CLASSES:
                        continue
                    unified = raw_label

                all_landmarks.append(lm[i])
                all_labels.append(unified)

        self.landmarks = np.stack(all_landmarks, axis=0).astype(np.float32)
        self.labels    = np.array(all_labels, dtype=np.int64)

        log.info(
            "Loaded %d samples from %d file(s)  classes=%s",
            len(self.labels), len(npz_paths),
            {UNIFIED_EMOTIONS[c]: int((self.labels == c).sum())
             for c in range(NUM_CLASSES) if (self.labels == c).sum() > 0},
        )

    def __len__(self):
        return len(self.labels)

    def _to_spatial_grid(self, lm: np.ndarray) -> np.ndarray:
        """
        Reshape (478, 3) → (1, 22, 22) float32.
        The 478 landmark (x,y,z) triplets are flattened to 1434 values,
        then we keep the first 484 (22×22) — the remaining 50 values
        (484-478)*3 are zero-padded.  We use only xy for the grid cells
        to keep spatial meaning; z is injected through the geo stream.
        """
        # Use all 3 coords flattened — gives the CNN full 3-D info
        flat = lm.flatten()                      # 478*3 = 1434
        target = 22 * 22 * 3                     # 1452  (pad 18 zeros)
        padded = np.zeros(target, dtype=np.float32)
        padded[:len(flat)] = flat
        grid = padded.reshape(3, 22, 22)         # (3, 22, 22)
        return grid

    def _augment(self, lm: np.ndarray) -> np.ndarray:
        """
        Lightweight augmentations on the landmark coords:
          - small random jitter (noise)
          - horizontal flip (mirror face left↔right)
          - small random scale perturbation
        """
        # Jitter
        lm = lm + np.random.normal(0, 0.004, lm.shape).astype(np.float32)

        # Horizontal flip (mirror around x = 0.5)
        if np.random.rand() < 0.5:
            lm[:, 0] = 1.0 - lm[:, 0]

        # Scale perturbation (±5%)
        scale = np.random.uniform(0.95, 1.05)
        center = lm.mean(axis=0, keepdims=True)
        lm = (lm - center) * scale + center

        return lm.astype(np.float32)

    def __getitem__(self, idx):
        lm    = self.landmarks[idx].copy()
        label = self.labels[idx]

        if self.augment:
            lm = self._augment(lm)

        grid = self._to_spatial_grid(lm)           # (3, 22, 22)
        geo  = extract_geometric_features(lm)      # (GEO_DIM,)

        return (
            torch.from_numpy(grid),
            torch.from_numpy(geo),
            torch.tensor(label, dtype=torch.long),
        )

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for balanced sampling / loss."""
        counts = np.bincount(self.labels, minlength=NUM_CLASSES).astype(np.float32)
        counts = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def sample_weights(self) -> np.ndarray:
        """Per-sample weight for WeightedRandomSampler."""
        cw = self.class_weights().numpy()
        return cw[self.labels]


# ─────────────────────────────────────────────────────────────────────────────
# EmoNet architecture  (Section 3.4 of DriveEmo-FL)
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2D → BN → ReLU → MaxPool."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
        )

    def forward(self, x):
        return self.block(x)


class EmoNet(nn.Module):
    """
    Dual-stream EmoNet adapted for face landmark input.

    Stream A — CNN on the (3, 22, 22) spatial landmark grid.
      Mirrors the paper's Conv2D blocks: 32 → 64 → 128 → 256 channels.
      Output embedding: z_spatial ∈ R^256

    Stream B — MLP on the (GEO_DIM,) geometric feature vector.
      Dense 16 → Dense 32 with ReLU (as in paper's VTP stream).
      Output embedding: z_geo ∈ R^32

    Fusion — concatenate → 1×1 Conv mixing → shared Dense head → softmax.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()

        # ── Stream A: spatial CNN ──────────────────────────────────────────
        self.cnn = nn.Sequential(
            ConvBlock(3,  32),    # (3,22,22)  → (32,11,11)
            ConvBlock(32, 64),    # → (64,6,6)
            ConvBlock(64, 128),   # → (128,3,3)
            ConvBlock(128, 256),  # → (256,2,2)
            nn.AdaptiveAvgPool2d(1),              # → (256,1,1)
            nn.Flatten(),                         # → (256,)
        )

        # ── Stream B: geometric MLP ────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(GEO_DIM, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
        )

        # ── Fusion block ───────────────────────────────────────────────────
        fused_dim = 256 + 32   # 288  (matches paper's N×288)
        self.fusion = nn.Sequential(
            # 1×1 conv mixing (paper uses two 1×1 Conv2D layers)
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
        )

        # ── Shared prediction head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.emotion_out = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, grid, geo):
        z_spatial = self.cnn(grid)           # (B, 256)
        z_geo     = self.mlp(geo)            # (B, 32)
        z         = torch.cat([z_spatial, z_geo], dim=1)   # (B, 288)
        z         = self.fusion(z)           # (B, 288)
        h         = self.head(z)             # (B, 64)
        logits    = self.emotion_out(h)      # (B, num_classes)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
    """
    Discover all .npz files under data_dir, split into train/val/test by
    filename suffix (_train, _validation, _test).  Falls back to a random
    80/10/10 split if no suffix is found.
    """
    all_files = sorted(glob(os.path.join(data_dir, "*.npz")))
    if not all_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    splits = defaultdict(list)
    for f in all_files:
        name = Path(f).stem.lower()
        if "train" in name:
            splits["train"].append(f)
        elif "val" in name or "validation" in name:
            splits["val"].append(f)
        elif "test" in name:
            splits["test"].append(f)
        else:
            splits["train"].append(f)   # fallback

    log.info("Split file counts — train:%d  val:%d  test:%d",
             len(splits["train"]), len(splits["val"]), len(splits["test"]))

    train_ds = FaceMeshEmotionDataset(splits["train"], augment=True)
    val_ds   = FaceMeshEmotionDataset(splits["val"],   augment=False) if splits["val"]  else None
    test_ds  = FaceMeshEmotionDataset(splits["test"],  augment=False) if splits["test"] else None

    # Balanced sampler for training
    sw = train_ds.sample_weights()
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sw).float(),
        num_samples=len(sw),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader  = (DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
                   if val_ds else None)
    test_loader = (DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
                   if test_ds else None)

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer, phase: str):
    """One epoch of train or eval. Returns (avg_loss, accuracy)."""
    is_train = phase == "train"
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for grid, geo, labels in tqdm(loader, desc=phase, leave=False):
            grid   = grid.to(DEVICE)
            geo    = geo.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(grid, geo)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation & plots
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, label_names: list[str], out_dir: Path):
    """Full evaluation: classification report + confusion matrix plot."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for grid, geo, labels in tqdm(loader, desc="evaluate", leave=False):
            logits = model(grid.to(DEVICE), geo.to(DEVICE))
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── text report ───────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    present = sorted(set(all_labels) | set(all_preds))
    names   = [label_names[i] for i in present]

    report = classification_report(
        all_labels, all_preds,
        labels=present, target_names=names,
        zero_division=0,
    )

    log.info("\n%s", report)
    log.info("Overall  Accuracy=%.4f  Weighted-F1=%.4f", acc, f1)

    report_path = out_dir / "evaluation_report.txt"
    report_path.write_text(
        f"Accuracy : {acc:.4f}\nWeighted F1 : {f1:.4f}\n\n{report}"
    )
    log.info("Report saved → %s", report_path)

    # ── confusion matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=present)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=names, yticklabels=names,
            ax=ax, linewidths=0.4,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", cm_path)

    return acc, f1


def plot_training_curves(history: dict, out_dir: Path):
    """Save loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train loss", linewidth=2)
    if history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], label="Val loss", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train acc", linewidth=2)
    if history["val_acc"]:
        ax2.plot(epochs, history["val_acc"], label="Val acc", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy", fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    curve_path = out_dir / "training_curves.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    log.info("Training curves saved → %s", curve_path)


def plot_per_class_accuracy(model, loader, label_names: list[str], out_dir: Path):
    """Bar chart of per-class accuracy (mirrors Fig. 20 in the paper)."""
    model.eval()
    class_correct = defaultdict(int)
    class_total   = defaultdict(int)

    with torch.no_grad():
        for grid, geo, labels in loader:
            logits = model(grid.to(DEVICE), geo.to(DEVICE))
            preds  = logits.argmax(dim=1).cpu()
            for p, l in zip(preds.numpy(), labels.numpy()):
                class_total[l]   += 1
                class_correct[l] += int(p == l)

    classes = sorted(class_total.keys())
    accs    = [class_correct[c] / class_total[c] for c in classes]
    names   = [label_names[c] for c in classes]

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(classes)))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy"); ax.set_title("Per-Class Accuracy", fontweight="bold")
    ax.axhline(np.mean(accs), color="steelblue", linestyle="--",
               linewidth=1.2, label=f"Mean {np.mean(accs):.3f}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "per_class_accuracy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Per-class accuracy chart saved → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    out_dir  = Path(args.output)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bench = BenchLogger(method="emonet_landmark", output_dir=out_dir, config=vars(args))

    log.info("Device: %s", DEVICE)
    log.info("Emotions (%d): %s", NUM_CLASSES, UNIFIED_EMOTIONS)

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = make_dataloaders(
        args.data, args.batch, args.workers
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = EmoNet(num_classes=NUM_CLASSES, dropout=args.dropout).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("EmoNet  params: %s  (%.2f M)", f"{n_params:,}", n_params / 1e6)
    bench.log_model(model)

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint required with --eval-only")
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
        log.info("Loaded checkpoint: %s", args.checkpoint)
        loader = test_loader or val_loader
        if loader is None:
            raise RuntimeError("No test or val split found for evaluation.")
        evaluate(model, loader, UNIFIED_EMOTIONS, out_dir)
        plot_per_class_accuracy(model, loader, UNIFIED_EMOTIONS, out_dir)
        return

    # Class imbalance is already handled by the WeightedRandomSampler in the
    # train loader — adding class weights here too would double-correct toward
    # rare classes (sampler oversamples them AND each rare-class sample then
    # contributes a larger loss). That over-pull typically costs 2-5 % on
    # imbalanced sets like AffectNet. Keep the sampler; drop the loss weight.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ── Optimiser & scheduler (paper: Adam lr=0.001, epochs=20) ──────────
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}
    best_val_acc = 0.0
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
        else:
            va_loss, va_acc = None, tr_acc
            history["val_loss"].append(None)
            history["val_acc"].append(None)

        scheduler.step()

        bench.epoch_end(epoch, tr_loss, tr_acc, va_loss, va_acc,
                        lr=optimizer.param_groups[0]["lr"])
        log.info(
            "Epoch %3d/%d  train_loss=%.4f  train_acc=%.4f  val_acc=%.4f  lr=%.2e",
            epoch, args.epochs, tr_loss, tr_acc, va_acc,
            optimizer.param_groups[0]["lr"],
        )

        # Checkpoint best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            patience_count = 0
            best_path = ckpt_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            log.info("  ✓ New best val_acc=%.4f  saved → %s", best_val_acc, best_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info("Early stopping triggered (patience=%d)", args.patience)
                break

        # Save latest every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")

    # ── Save final + training curves ──────────────────────────────────────
    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    plot_training_curves(history, out_dir)

    # ── Final evaluation ──────────────────────────────────────────────────
    log.info("Loading best checkpoint for final evaluation…")
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=DEVICE))

    eval_loader = test_loader or val_loader
    if eval_loader:
        acc, f1 = evaluate(model, eval_loader, UNIFIED_EMOTIONS, out_dir)
        plot_per_class_accuracy(model, eval_loader, UNIFIED_EMOTIONS, out_dir)
    else:
        log.warning("No test or val split found — skipping final evaluation.")
        acc, f1 = best_val_acc, 0.0

    # ── Inference latency benchmark on best model ─────────────────────────
    dummy_grid = torch.zeros(1, 3, 22, 22, device=DEVICE)
    dummy_geo  = torch.zeros(1, GEO_DIM, device=DEVICE)
    bench.benchmark_inference(model, (dummy_grid, dummy_geo), runs=200, warmup=20)
    bench.finalize(final_accuracy=acc, final_weighted_f1=f1,
                   best_val_accuracy=best_val_acc)

    # ── Save run summary ──────────────────────────────────────────────────
    summary = {
        "best_val_acc": best_val_acc,
        "final_test_acc": acc,
        "final_weighted_f1": f1,
        "epochs_trained": epoch,
        "num_classes": NUM_CLASSES,
        "emotions": UNIFIED_EMOTIONS,
        "model_params_M": round(n_params / 1e6, 3),
        "hyperparams": vars(args),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Run summary → %s/run_summary.json", out_dir)
    log.info("Done. Best val acc: %.4f", best_val_acc)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EmoNet on the FaceMesh landmark dataset"
    )
    parser.add_argument(
        "--data", default="cache/facemesh_dataset",
        help="Folder containing the .npz files (default: cache/facemesh_dataset)",
    )
    parser.add_argument(
        "--output", default="runs/emonet",
        help="Output folder for checkpoints and plots (default: runs/emonet)",
    )
    parser.add_argument(
        "--epochs", type=int, default=40,
        help="Number of training epochs (default: 40)",
    )
    parser.add_argument(
        "--batch", type=int, default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Learning rate (default: 5e-4 — slightly under the paper's 1e-3, "
             "more stable on AffectNet's noisy labels)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.4,
        help="Dropout rate (default: 0.4, as per paper)",
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience in epochs (default: 15)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader worker processes (default: 4)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training and run evaluation only",
    )
    parser.add_argument(
        "--checkpoint", default="",
        help="Path to a .pt checkpoint (required with --eval-only)",
    )
    args = parser.parse_args()
    main(args)