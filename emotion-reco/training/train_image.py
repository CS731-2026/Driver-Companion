"""
train_image.py
==============
Image-based CNN baseline for facial emotion recognition.

The CalmWheel design presentation explicitly asks:

    "Train and Test the models both on facemesh dataset and original
     dataset to show the difference of accuracy and hardware utilization."

This script handles the *original-dataset* side of that comparison.
The three landmark-based scripts (train.py, train2.py, train_geo.py) handle
the facemesh side.

Architecture
------------
MobileNetV2 (Sandler et al., CVPR 2018), the same lightweight backbone
adopted by Luan et al. 2025 for driver emotion recognition (3D-MobileNetV2
in their MHLT model).  Here we use the 2D variant since AffectNet / RAF-DB
are static images, and we initialise from torchvision's ImageNet weights to
match the typical "fine-tune on AffectNet" pipeline in the FER literature
(e.g. Roka & Rawat 2023).

Dataset
-------
Loads the original AffectNet-HQ and RAF-DB-7emotions splits directly from
HuggingFace via the same configs used by Dataset/Dataset.py, so the
comparison against the facemesh runs is on the exact same source data.

Usage
-----
    pip install torch torchvision datasets scikit-learn matplotlib seaborn tqdm
    python train_image.py
    python train_image.py --backbone mobilenet_v3_small --epochs 30
"""

from __future__ import annotations

import io
import os
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from datasets import load_dataset, load_from_disk

from .bench_logger import BenchLogger

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
# Dataset (loads HuggingFace AffectNet-HQ + RAF-DB images directly)
# Matches Dataset/Dataset.py DATASETS_CONFIG so we train on the exact same
# source images that the facemesh pipeline uses, just without the landmark
# extraction step.
# ─────────────────────────────────────────────────────────────────────────────

DATASETS_CONFIG = {
    "affectnet": {
        "repo_id": "Piro17/affectnethq",
        "local_path": "./datasets/AffectnetHQ/",
        "image_col": "image",
        "label_col": "label",
    },
    "rafdb": {
        "repo_id": "deanngkl/raf-db-7emotions",
        "local_path": "./datasets/RAF-DB-7emotions/",
        "image_col": "image",
        "label_col": "label",
    },
}


def _load_hf(cfg: dict):
    local = cfg["local_path"]
    if os.path.exists(local):
        try:
            return load_from_disk(local)
        except Exception as e:
            log.warning("load_from_disk failed (%s) — re-downloading…", e)
    ds = load_dataset(cfg["repo_id"])
    ds.save_to_disk(local)
    return ds


class HFImageEmotionDataset(Dataset):
    """
    Iterate HuggingFace splits and yield (img_tensor, unified_label).
    Performs the same label remap as the landmark trainers so the test
    sets line up across methods.
    """

    def __init__(self, hf_split, image_col: str, label_col: str,
                 label_names: list | None, transform=None):
        self.split = hf_split
        self.image_col = image_col
        self.label_col = label_col
        self.label_names = label_names
        self.transform = transform

        # Pre-resolve indices we can keep (skip classes not in unified map)
        self.indices: list[int] = []
        self.labels:  list[int] = []
        for i in range(len(hf_split)):
            raw = hf_split[i][label_col]
            if raw is None:
                continue
            if label_names is not None:
                name = label_names[int(raw)].lower()
                if name not in LABEL_REMAP:
                    continue
                unified = LABEL_REMAP[name]
            else:
                if int(raw) >= NUM_CLASSES:
                    continue
                unified = int(raw)
            self.indices.append(i)
            self.labels.append(unified)

        self.labels_np = np.array(self.labels, dtype=np.int64)
        dist = {
            UNIFIED_EMOTIONS[c]: int((self.labels_np == c).sum())
            for c in range(NUM_CLASSES) if (self.labels_np == c).sum() > 0
        }
        log.info("Kept %d / %d samples  dist=%s",
                 len(self.indices), len(hf_split), dist)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ex = self.split[self.indices[idx]]
        raw = ex[self.image_col]

        if isinstance(raw, dict) and "bytes" in raw:
            img = Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        elif isinstance(raw, Image.Image):
            img = raw.convert("RGB")
        else:
            img = Image.fromarray(np.array(raw)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(self.labels[idx], dtype=torch.long)

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels_np, minlength=NUM_CLASSES).astype(np.float32)
        counts = np.where(counts == 0, 1.0, counts)
        w = 1.0 / counts
        return torch.tensor(w / w.sum(), dtype=torch.float32)

    def sample_weights(self) -> np.ndarray:
        return self.class_weights().numpy()[self.labels_np]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_backbone(name: str, num_classes: int) -> nn.Module:
    """ImageNet-pretrained lightweight backbones tailored for FER."""
    name = name.lower()
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown backbone: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(image_size: int, batch: int, workers: int, datasets_selected: list[str]):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_parts, val_parts, test_parts = [], [], []

    for ds_name in datasets_selected:
        cfg = DATASETS_CONFIG[ds_name]
        log.info("══ %s ══", ds_name)
        ds = _load_hf(cfg)

        try:
            label_names = ds[list(ds.keys())[0]].features[cfg["label_col"]].names
        except Exception:
            label_names = None

        for split_name, split_data in ds.items():
            tf = train_tf if "train" in split_name.lower() else eval_tf
            sub = HFImageEmotionDataset(
                split_data, cfg["image_col"], cfg["label_col"],
                label_names=label_names, transform=tf,
            )
            n = split_name.lower()
            if "train" in n:
                train_parts.append(sub)
            elif "val" in n or "validation" in n:
                val_parts.append(sub)
            elif "test" in n:
                test_parts.append(sub)
            else:
                train_parts.append(sub)

    if not train_parts:
        raise RuntimeError("No training data was loaded.")

    train_ds = torch.utils.data.ConcatDataset(train_parts)
    val_ds   = torch.utils.data.ConcatDataset(val_parts)  if val_parts  else None
    test_ds  = torch.utils.data.ConcatDataset(test_parts) if test_parts else None

    # Balanced sampler over the train concat
    all_labels = np.concatenate([p.labels_np for p in train_parts])
    counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(np.float32)
    counts = np.where(counts == 0, 1.0, counts)
    cw = 1.0 / counts
    cw = cw / cw.sum()
    sw = cw[all_labels]
    sampler = WeightedRandomSampler(torch.from_numpy(sw).float(),
                                    len(sw), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler,
                              num_workers=workers, pin_memory=True)
    val_loader  = (DataLoader(val_ds,  batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=True) if val_ds  else None)
    test_loader = (DataLoader(test_ds, batch_size=batch, shuffle=False,
                              num_workers=workers, pin_memory=True) if test_ds else None)

    return train_loader, val_loader, test_loader, torch.tensor(cw, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, phase):
    is_train = phase == "train"
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for x, y in tqdm(loader, desc=phase, leave=False):
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

    method_name = f"image_{args.backbone}"
    bench = BenchLogger(method=method_name, output_dir=out_dir, config=vars(args))
    log.info("Device : %s", DEVICE)
    log.info("Backbone : %s", args.backbone)

    datasets_selected = args.datasets.split(",") if args.datasets else list(DATASETS_CONFIG.keys())

    train_loader, val_loader, test_loader, _ = make_loaders(
        args.image_size, args.batch, args.workers, datasets_selected,
    )

    model = build_backbone(args.backbone, NUM_CLASSES).to(DEVICE)
    bench.log_model(model)

    # Class imbalance is already handled by the WeightedRandomSampler in the
    # train loader — adding class weights here too would double-correct toward
    # rare classes. Keep the sampler; drop the loss weight.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
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
            monitor = va_acc
        else:
            va_loss, va_acc = None, None
            history["val_loss"].append(None); history["val_acc"].append(None)
            monitor = tr_acc

        scheduler.step()
        bench.epoch_end(epoch, tr_loss, tr_acc, va_loss, va_acc,
                        lr=optimizer.param_groups[0]["lr"])

        if monitor > best_acc:
            best_acc = monitor
            patience_count = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            log.info("  ✓ New best=%.4f  saved → %s/best.pt", best_acc, ckpt_dir)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    plot_training_curves(history, out_dir)

    log.info("Loading best checkpoint for evaluation…")
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=DEVICE))

    eval_loader = test_loader or val_loader
    acc, f1 = best_acc, 0.0
    if eval_loader:
        acc, f1 = evaluate(model, eval_loader, out_dir)

    dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=DEVICE)
    bench.benchmark_inference(model, (dummy,), runs=200, warmup=20)
    bench.finalize(final_accuracy=acc, final_weighted_f1=f1, best_val_accuracy=best_acc)

    summary = {
        "method": method_name,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "best_val_acc": best_acc,
        "final_test_acc": acc,
        "final_weighted_f1": f1,
        "datasets": datasets_selected,
        "hyperparams": vars(args),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image-based CNN baseline for facial emotion recognition "
                    "(the 'original dataset' side of the design's comparison)"
    )
    parser.add_argument("--datasets", default="",
                        help="Comma-separated subset of {affectnet,rafdb} (default: all)")
    parser.add_argument("--output",   default="runs/image")
    parser.add_argument("--backbone", default="mobilenet_v2",
                        choices=["mobilenet_v2", "mobilenet_v3_small", "resnet18"],
                        help="Lightweight ImageNet-pretrained backbone")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs",   type=int, default=35,
                        help="MobileNet fine-tuning benefits from a longer "
                             "schedule on AffectNet than the paper default")
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--workers",  type=int, default=4)
    args = parser.parse_args()
    main(args)
