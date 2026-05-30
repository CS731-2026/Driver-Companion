"""
plot_all_models.py
==================
Runs every training method and produces a single figure showing train and
val/test accuracy per epoch for all models on the same axes.

Data layout (CK+ processed by this project):
  cache/ckplus_static/    — static landmark + blendshape .npz files
                            used by emonet, blendshape, geo_static
  cache/ckplus.npz        — temporal (ConvLSTM1D) feature cache
  cache/ckplus_baseline.npz — flat features for baseline MLP models

BenchLogger-based models (emonet, blendshape, geo_static) are launched via
subprocess; their per-epoch curves come from the ``*_metrics.json`` each
trainer writes.  If that file already exists the training step is skipped
(pass ``--force`` to retrain).

ConvLSTM1D is run in-process via the ``cross_validate`` Python API; the
per-fold histories are averaged and plotted as a mean curve.

Baseline models (BasicNN / OptimizedNN) are run in-process with a thin
history-tracking wrapper; they appear as full accuracy curves too.

The image-CNN model (train_image.py) requires a HuggingFace download and
is excluded by default — pass ``--include-image`` to enable it.

Run from emotion-reco/:

    python3 -m training.plot_all_models
    python3 -m training.plot_all_models --quick           # 3 epochs each
    python3 -m training.plot_all_models --force           # always retrain
    python3 -m training.plot_all_models --plot-only       # read existing metrics
    python3 -m training.plot_all_models --skip emonet     # skip a method
    python3 -m training.plot_all_models --only convlstm   # one method only
    python3 -m training.plot_all_models --include-image   # also run MobileNetV2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

# BenchLogger-based models (launched via subprocess).
# data_dir is forwarded as --data; the trainer writes *_metrics.json.
SUBPROCESS_METHODS: list[dict] = [
    {
        "name":         "emonet",
        "label":        "EmoNet (dual-stream)",
        "module":       "training.train_emonet",
        "output":       "runs/emonet",
        "metrics_file": "emonet_landmark_metrics.json",
    },
    {
        "name":         "blendshape_mlp",
        "label":        "Blendshape MLP",
        "module":       "training.train_blendshape",
        "output":       "runs/blendshape_mlp",
        "args":         ["--model", "mlp"],
        "metrics_file": "blendshape_mlp_metrics.json",
    },
    {
        "name":         "blendshape_attention",
        "label":        "Blendshape Attention",
        "module":       "training.train_blendshape",
        "output":       "runs/blendshape_attn",
        "args":         ["--model", "attention"],
        "metrics_file": "blendshape_attention_metrics.json",
    },
    {
        "name":         "geo_static",
        "label":        "Geo MLP (122 lm, AU groups)",
        "module":       "training.train_geo_static",
        "output":       "runs/geo_static",
        "args":         ["--landmarks", "122"],
        "metrics_file": "geo_122_au_metrics.json",
    },
]

# Image CNN — excluded by default (needs HuggingFace download).
IMAGE_METHOD: dict = {
    "name":         "image_cnn",
    "label":        "MobileNetV2 (raw images)",
    "module":       "training.train_image",
    "output":       "runs/image",
    "args":         ["--backbone", "mobilenet_v2"],
    "metrics_file": "image_mobilenet_v2_metrics.json",
    "no_data":      True,
}

# Colour palette: each model gets its own hue.
# Train = solid line, val = dashed line, same colour.
_COLORS = [
    "#1f77b4",   # blue
    "#ff7f0e",   # orange
    "#2ca02c",   # green
    "#d62728",   # red
    "#9467bd",   # purple
    "#8c564b",   # brown
    "#e377c2",   # pink
    "#17becf",   # cyan
]


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess runner (BenchLogger models)
# ──────────────────────────────────────────────────────────────────────────────

def _run_subprocess(method: dict, root: Path, data_dir: str,
                    quick: bool, force: bool) -> bool:
    metrics_path = root / method["output"] / method["metrics_file"]
    if metrics_path.exists() and not force:
        print(f"[skip]  {method['name']:25s} — metrics exist at {metrics_path}")
        return True

    cmd = [sys.executable, "-m", method["module"],
           "--output", method["output"]]
    if not method.get("no_data", False):
        cmd += ["--data", data_dir]
    cmd += method.get("args", [])
    if quick:
        cmd += ["--epochs", "3"]

    sep = "═" * 72
    print(f"\n{sep}\n  {method['name']}\n  {' '.join(cmd)}\n{sep}")
    return subprocess.run(cmd, cwd=root).returncode == 0


def _load_benchlogger(method: dict, root: Path) -> dict | None:
    """Read per-epoch {train_acc, val_acc} from a BenchLogger *_metrics.json."""
    path = root / method["output"] / method["metrics_file"]
    if not path.exists():
        print(f"[warn]  No metrics for {method['name']}: {path}")
        return None
    try:
        data = json.loads(path.read_text())
        epochs = data.get("epochs", [])
        if not epochs:
            print(f"[warn]  Empty epoch list in {path}")
            return None
        train_acc = [e["train_acc"] for e in epochs]
        val_acc   = [e["val_acc"]   for e in epochs if e.get("val_acc") is not None]
        return {"train_acc": train_acc, "val_acc": val_acc}
    except Exception as exc:
        print(f"[warn]  Could not parse {path}: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# ConvLSTM1D in-process runner
# ──────────────────────────────────────────────────────────────────────────────

def _run_convlstm(cache_path: str, epochs_override: int | None,
                  patience: int = 30) -> dict | None:
    """
    Run 5-fold cross-validation of the ConvLSTM1D model in-process.
    Returns averaged per-epoch {train_acc, val_acc} across folds.
    """
    try:
        import torch
        from training.train import cross_validate, load_cache
        from training.config import TrainConfig
    except ImportError as exc:
        print(f"[warn]  Could not import training.train: {exc}")
        return None

    if not Path(cache_path).exists():
        print(f"[warn]  ConvLSTM1D cache not found: {cache_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═' * 72}\n  convlstm  (in-process, device={device})\n{'═' * 72}")

    X, y, labels = load_cache(cache_path)
    cfg = TrainConfig()
    if epochs_override is not None:
        cfg.epochs = epochs_override

    cv = cross_validate(X, y, labels, cfg, device, patience=patience, verbose=True)

    # Average per-epoch histories across folds (pad shorter folds with last value).
    all_train = [f["history"]["train_acc"] for f in cv["folds"]]
    all_val   = [f["history"]["val_acc"]   for f in cv["folds"]]
    max_len   = max(len(t) for t in all_train)

    def _pad(seq):
        return seq + [seq[-1]] * (max_len - len(seq))

    mean_train = np.mean([_pad(t) for t in all_train], axis=0).tolist()
    mean_val   = np.mean([_pad(v) for v in all_val],   axis=0).tolist()

    print(f"ConvLSTM1D: mean_val={cv['mean_val_acc']:.4f} ± {cv['std_val_acc']:.4f}")
    return {"train_acc": mean_train, "val_acc": mean_val}


# ──────────────────────────────────────────────────────────────────────────────
# Baseline in-process runners (BasicNN / OptimizedNN with history tracking)
# ──────────────────────────────────────────────────────────────────────────────

def _run_baseline_nn(name: str, cache_path: str,
                     quick: bool = False) -> dict | None:
    """
    Run one baseline NN (basic_nn or optimized_nn) with per-epoch history.
    Returns mean {train_acc, val_acc} across folds.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from training.baseline import (
            BasicNN, OptimizedNN,
            BASIC_EPOCHS, OPT_EPOCHS, LEARNING_RATE, BATCH_SIZE, FOLDS,
            RANDOM_STATE, load_baseline_cache,
        )
        from training.train import run_epoch
    except ImportError as exc:
        print(f"[warn]  Could not import baseline: {exc}")
        return None

    if not Path(cache_path).exists():
        print(f"[warn]  Baseline cache not found: {cache_path}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═' * 72}\n  {name}  (in-process, device={device})\n{'═' * 72}")

    X, y, labels = load_baseline_cache(cache_path, "displacement")
    num_classes = len(labels)
    in_dim = X.shape[1]
    epochs = 3 if quick else BASIC_EPOCHS
    use_sched = (name == "optimized_nn")

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_train_histories: list[list[float]] = []
    fold_val_histories:   list[list[float]] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        torch.manual_seed(RANDOM_STATE + fold)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx]).astype(np.float32)
        X_va = scaler.transform(X[va_idx]).astype(np.float32)
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        model = (BasicNN(in_dim, num_classes) if name == "basic_nn"
                 else OptimizedNN(in_dim, num_classes)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        scheduler = (torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
                     if use_sched else None)

        tr_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=BATCH_SIZE, shuffle=True,
            drop_last=len(X_tr) > BATCH_SIZE,
        )
        va_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
            batch_size=BATCH_SIZE, shuffle=False,
        )

        train_hist, val_hist = [], []
        for ep in range(epochs):
            _, tr_acc = run_epoch(model, tr_loader, criterion, device, optimizer)
            _, va_acc = run_epoch(model, va_loader, criterion, device)
            train_hist.append(tr_acc)
            val_hist.append(va_acc)
            if scheduler:
                scheduler.step()
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  {name} fold {fold}/{FOLDS}  ep {ep+1}/{epochs}  "
                      f"train={tr_acc:.4f}  val={va_acc:.4f}")

        fold_train_histories.append(train_hist)
        fold_val_histories.append(val_hist)

    max_len = max(len(h) for h in fold_train_histories)

    def _pad(seq):
        return seq + [seq[-1]] * (max_len - len(seq))

    mean_train = np.mean([_pad(h) for h in fold_train_histories], axis=0).tolist()
    mean_val   = np.mean([_pad(h) for h in fold_val_histories],   axis=0).tolist()
    print(f"{name}: best mean val = {max(mean_val):.4f}")
    return {"train_acc": mean_train, "val_acc": mean_val}


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _plot(histories: list[dict], out_path: Path) -> None:
    """
    histories: list of {label, train_acc, val_acc}
    Each model = one colour; solid = train, dashed = val/test.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, h in enumerate(histories):
        color = _COLORS[i % len(_COLORS)]
        label = h["label"]
        train = h["train_acc"]
        val   = h["val_acc"]

        ax.plot(range(1, len(train) + 1), train,
                color=color, linewidth=1.8,
                label=f"{label} — train")
        if val:
            ax.plot(range(1, len(val) + 1), val,
                    color=color, linewidth=1.8, linestyle="--",
                    label=f"{label} — val/test")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Train / Validation Accuracy — All Models",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, loc="lower right", ncol=2,
              framealpha=0.9, edgecolor="grey")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {out_path.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

ALL_NAMES = (
    [m["name"] for m in SUBPROCESS_METHODS]
    + ["convlstm", "basic_nn", "optimized_nn", IMAGE_METHOD["name"]]
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data", default="cache/ckplus_static",
        help="Directory with landmark/blendshape .npz files for emonet / "
             "blendshape / geo_static  (default: cache/ckplus_static)",
    )
    ap.add_argument(
        "--convlstm-cache", default="cache/ckplus.npz",
        help="Feature-cache .npz for ConvLSTM1D  (default: cache/ckplus.npz)",
    )
    ap.add_argument(
        "--baseline-cache", default="cache/ckplus_baseline.npz",
        help="Feature-cache .npz for baseline NNs  (default: cache/ckplus_baseline.npz)",
    )
    ap.add_argument(
        "--out", default="runs/all_models_accuracy.png",
        help="Output plot file  (default: runs/all_models_accuracy.png)",
    )
    ap.add_argument("--quick",     action="store_true",
                    help="Override epochs=3 for a quick smoke test")
    ap.add_argument("--force",     action="store_true",
                    help="Retrain even when metrics files already exist")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip training; plot from existing metrics files only")
    ap.add_argument("--include-image", action="store_true",
                    help="Also run the MobileNetV2 image-CNN (needs HuggingFace)")
    ap.add_argument(
        "--skip",  action="append", default=[], metavar="NAME",
        help="Skip a method by name (repeatable).  Names: " + ", ".join(ALL_NAMES),
    )
    ap.add_argument(
        "--only",  action="append", default=[], metavar="NAME",
        help="Run only these method names (repeatable)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent   # emotion-reco/

    def _wanted(name: str) -> bool:
        if args.only and name not in args.only:
            return False
        return name not in args.skip

    histories: list[dict] = []

    # ── BenchLogger subprocess models ────────────────────────────────────────
    sub_methods = SUBPROCESS_METHODS[:]
    if args.include_image:
        sub_methods.append(IMAGE_METHOD)

    for m in sub_methods:
        if not _wanted(m["name"]):
            continue
        data_dir = "." if m.get("no_data") else args.data
        if not args.plot_only:
            _run_subprocess(m, root, data_dir, args.quick, args.force)
        h = _load_benchlogger(m, root)
        if h:
            histories.append({"label": m["label"], **h})

    # ── ConvLSTM1D (in-process) ───────────────────────────────────────────────
    if _wanted("convlstm"):
        cache = str(root / args.convlstm_cache)
        if args.plot_only:
            print("[skip]  convlstm — --plot-only: no in-process run")
        else:
            h = _run_convlstm(
                cache,
                epochs_override=3 if args.quick else None,
                patience=5 if args.quick else 30,
            )
            if h:
                histories.append({"label": "ConvLSTM1D (mean 5-fold)", **h})

    # ── Baseline NNs (in-process) ─────────────────────────────────────────────
    baseline_specs = [
        ("basic_nn",      "BasicNN (256→128)"),
        ("optimized_nn",  "OptimizedNN (512→256→128)"),
    ]
    for bname, blabel in baseline_specs:
        if not _wanted(bname):
            continue
        cache = str(root / args.baseline_cache)
        if args.plot_only:
            print(f"[skip]  {bname} — --plot-only: no in-process run")
        else:
            h = _run_baseline_nn(bname, cache, quick=args.quick)
            if h:
                histories.append({"label": blabel, **h})

    # ── Plot ─────────────────────────────────────────────────────────────────
    if not histories:
        print("\nNo histories found — nothing to plot.")
        print("Check that --data / --convlstm-cache / --baseline-cache paths exist,")
        print("or remove --plot-only to train first.")
        return 1

    _plot(histories, root / args.out)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Accuracy summary ───────────────────────────────────────────────")
    print(f"{'Model':<35}  {'Best train':>10}  {'Best val':>10}")
    print("-" * 60)
    for h in histories:
        best_tr  = max(h["train_acc"]) if h["train_acc"] else float("nan")
        best_val = max(h["val_acc"])   if h["val_acc"]   else float("nan")
        print(f"{h['label']:<35}  {best_tr:>10.4f}  {best_val:>10.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
