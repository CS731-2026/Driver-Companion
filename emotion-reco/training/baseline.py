"""Paper baseline - Juarez-Jimenez et al. 2025 (arXiv:2506.17191).

PyTorch reimplementation of "Facial Landmark Visualization and Emotion
Recognition Through Neural Networks". No official code was released, so this is
a from-scratch port of the paper's Section 3.4 classifiers, kept as a baseline
to compare against the ConvLSTM1D model.

The paper feeds normalized landmark features (see dataset.prepare_baseline) into
three classifiers and reports 4-fold cross-validation accuracy:

    Decision Tree      ~80%    interpretable reference (hyperparameters tuned)
    Basic NN           ~97%    256 -> 128, ReLU, 50% dropout, 50 epochs
    Optimized NN      98.48%   512 -> 256(+residual) -> 128, GELU, BatchNorm,
                               30% dropout, 200 epochs, Adam + LR scheduler

Run:
    python -m training.baseline --features cache/ckplus_baseline.npz \
        --out runs/baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .train import evaluate, run_epoch

# Paper hyperparameters (Section 3.4).
BASIC_EPOCHS = 50
OPT_EPOCHS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
FOLDS = 4
RANDOM_STATE = 42

# Published accuracies, used only for the side-by-side comparison table.
PAPER_BASELINE = {
    "decision_tree": 0.80,
    "basic_nn": 0.9697,
    "optimized_nn": 0.9848,
}


class BasicNN(nn.Module):
    """Paper's basic network: two FC layers, ReLU, 50% dropout."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OptimizedNN(nn.Module):
    """Paper's optimized network: 512 -> 256 -> 128 with GELU, BatchNorm and
    30% dropout, plus a residual connection around the 256-unit layer.

    The 512->256 dimension change needs a projection shortcut for the residual.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.proj = nn.Linear(512, 256)        # residual shortcut (512 -> 256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.drop(self.act(self.bn1(self.fc1(x))))
        h2 = self.drop(self.act(self.bn2(self.fc2(h1)) + self.proj(h1)))
        h3 = self.drop(self.act(self.bn3(self.fc3(h2))))
        return self.out(h3)


def train_nn(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    epochs: int,
    device: torch.device,
    use_scheduler: bool,
) -> nn.Module:
    """Train an MLP for a fixed number of epochs (the paper uses no early stop)."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
        drop_last=len(X_tr) > BATCH_SIZE,   # keep BatchNorm happy, never empty
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        if use_scheduler else None
    )
    for _ in range(epochs):
        run_epoch(model, loader, criterion, device, optimizer)
        if scheduler is not None:
            scheduler.step()
    return model


def fit_tree(X_tr: np.ndarray, y_tr: np.ndarray,
             X_te: np.ndarray, y_te: np.ndarray) -> float:
    """Fit a hyperparameter-tuned decision tree, return test accuracy.

    GridSearchCV tunes on inner CV folds of the training data only, so the test
    fold stays unseen.
    """
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"criterion": ["gini", "entropy"],
         "max_depth": [None, 5, 10, 20],
         "min_samples_leaf": [1, 2, 4]},
        cv=3,
    )
    grid.fit(X_tr, y_tr)
    return float((grid.predict(X_te) == y_te).mean())


def cross_validate_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> Dict:
    """Stratified 4-fold CV for one classifier."""
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs: List[float] = []
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        if name == "decision_tree":
            acc = fit_tree(X[tr], y[tr], X[te], y[te])
        else:
            torch.manual_seed(RANDOM_STATE + fold)
            in_dim = X.shape[1]
            if name == "basic_nn":
                model = train_nn(BasicNN(in_dim, num_classes), X[tr], y[tr],
                                 BASIC_EPOCHS, device, use_scheduler=False)
            else:  # optimized_nn
                model = train_nn(OptimizedNN(in_dim, num_classes), X[tr], y[tr],
                                 OPT_EPOCHS, device, use_scheduler=True)
            acc = evaluate(model, X[te], y[te], device)
        accs.append(acc)
        print(f"  {name:14s} fold {fold}/{FOLDS}: {acc:.4f}")
    return {
        "model": name,
        "fold_acc": accs,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "paper_acc": PAPER_BASELINE[name],
    }


def load_baseline_cache(path: str, feature: str):
    """Load (X, y, labels) from a prepare_baseline .npz for the chosen feature."""
    data = np.load(path, allow_pickle=False)
    key = "X_disp" if feature == "displacement" else "X_abs"
    return (
        data[key].astype(np.float32),
        data["y"].astype(np.int64),
        [str(s) for s in data["labels"]],
    )


def render_table(rows: List[Dict]) -> str:
    lines = [
        "Baseline vs Juarez-Jimenez et al. 2025 (arXiv:2506.17191)",
        "-" * 60,
        f"{'Model':<16}{'Test%':>9}{'+-std':>8}{'Paper%':>9}{'Delta':>9}",
        "-" * 60,
    ]
    for r in rows:
        delta = f"{100 * (r['mean_acc'] - r['paper_acc']):+.1f}"
        lines.append(
            f"{r['model']:<16}{100 * r['mean_acc']:>9.1f}"
            f"{100 * r['std_acc']:>8.1f}{100 * r['paper_acc']:>9.1f}{delta:>9}"
        )
    lines.append("-" * 60)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--features", required=True, help=".npz from prepare_baseline")
    ap.add_argument("--out", default="runs/baseline", help="Output directory")
    ap.add_argument("--feature", choices=["displacement", "absolute"],
                    default="displacement",
                    help="Which paper feature set to classify on")
    ap.add_argument("--models", nargs="+",
                    default=["decision_tree", "basic_nn", "optimized_nn"],
                    choices=["decision_tree", "basic_nn", "optimized_nn"])
    ap.add_argument("--cpu", action="store_true", help="Force CPU even with CUDA")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    X, y, labels = load_baseline_cache(args.features, args.feature)
    print(f"Loaded X={X.shape}  classes={labels}")
    print(f"Feature set: {args.feature}   device: {device}")

    rows = []
    for name in args.models:
        print(f"\n=== {name} ===")
        rows.append(cross_validate_model(name, X, y, len(labels), device))

    table = render_table(rows)
    print("\n" + "=" * 60)
    print(table)

    (out_dir / "results.json").write_text(json.dumps(
        {"feature": args.feature, "labels": labels, "results": rows}, indent=2
    ))
    (out_dir / "results.md").write_text("```\n" + table + "\n```\n")
    print(f"\nSaved -> {out_dir / 'results.json'}  and  {out_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
