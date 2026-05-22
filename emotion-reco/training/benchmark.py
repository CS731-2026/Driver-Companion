"""Reproduce the paper's results table across datasets.

Runs the paper's evaluation protocol (Köksal & Gumus 2025, Sections 4.1-4.2)
and prints a train/test results table compared against the published numbers.

Per-dataset experiment (Section 4.1) — 5-fold cross-validation:
    python -m training.benchmark \
        --datasets ckplus=cache/ckplus.npz mmi=cache/mmi.npz \
        --out runs/benchmark

Composite experiments (Section 4.2), needs >=2 datasets:
    --composite        adds (a) merged 5-fold CV and
                       (b) leave-one-dataset-out hold-out tests.

Dataset keys that match the paper get a side-by-side comparison; the recognised
keys are: ckplus, oulu_casia_vis, oulu_casia_nir, mmi (hyphens/spaces ok).

NOTE: the paper's CK+/Oulu-CASIA/MMI datasets are access-gated. With only
RAVDESS cached you still get a valid results table — just without a paper
baseline for that row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import TrainConfig
from .model import build_model
from .train import cross_validate, evaluate, load_cache, scale_fold, train_one_fold


# Published accuracies (Table 5 / Section 4.2). Used only for side-by-side
# display; our config is 61-landmark AU grouping, so small deviations are
# expected versus the paper's best (250-landmark) numbers.
PAPER_RESULTS: Dict[str, float] = {
    "ckplus": 0.93,
    "oulu_casia_vis": 0.79,
    "oulu_casia_nir": 0.77,
    "mmi": 0.68,
}
PAPER_COMPOSITE = {
    "holdout_mmi": 0.6970,   # train CK+/Oulu-NIR/Oulu-VIS, test MMI
    "merged_5fold": 0.8210,  # all datasets merged, 5-fold CV
}


def normalize_key(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def parse_datasets(items: List[str]) -> List[Tuple[str, str]]:
    """Parse `name=path.npz` or bare `path.npz` (name inferred from filename)."""
    out = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(item).stem
        out.append((name, path))
    return out


def align_to_common_labels(
    datasets: List[Tuple[str, np.ndarray, np.ndarray, List[str]]]
) -> Tuple[List[str], List[Tuple[str, np.ndarray, np.ndarray]]]:
    """Restrict every dataset to the emotions common to all of them.

    The paper merges datasets on their shared emotions (e.g. dropping contempt).
    Labels are remapped into a single shared index space.
    """
    common = sorted(set.intersection(*[set(lbls) for _, _, _, lbls in datasets]))
    if not common:
        raise ValueError("Datasets share no common emotion labels — cannot merge.")
    new_index = {name: i for i, name in enumerate(common)}

    aligned = []
    for name, X, y, lbls in datasets:
        keep_mask = np.array([lbls[c] in new_index for c in y])
        remapped = np.array([new_index[lbls[c]] for c in y[keep_mask]], dtype=np.int64)
        aligned.append((name, X[keep_mask], remapped))
    return common, aligned


def fmt_pct(x: float | None) -> str:
    return "  —  " if x is None else f"{100 * x:5.1f}"


def run_per_dataset(
    datasets: List[Tuple[str, str]],
    cfg: TrainConfig,
    device: torch.device,
    patience: int,
) -> List[Dict]:
    rows = []
    for name, path in datasets:
        print(f"\n{'#' * 70}\n# Dataset: {name}  ({path})\n{'#' * 70}")
        X, y, labels = load_cache(path)
        print(f"X={X.shape}  classes={labels}")
        cv = cross_validate(X, y, labels, cfg, device, patience, verbose=True)
        paper = PAPER_RESULTS.get(normalize_key(name))
        rows.append({
            "dataset": name,
            "n_sequences": int(len(y)),
            "n_classes": len(labels),
            "train_acc": cv["mean_train_acc"],
            "test_acc": cv["mean_val_acc"],
            "test_std": cv["std_val_acc"],
            "fold_test_acc": cv["fold_val_acc"],
            "paper_acc": paper,
        })
    return rows


def run_composite(
    datasets: List[Tuple[str, str]],
    cfg: TrainConfig,
    device: torch.device,
    patience: int,
) -> Dict:
    """Section 4.2: merged 5-fold CV + leave-one-dataset-out hold-out tests."""
    loaded = [(name, *load_cache(path)) for name, path in datasets]
    common, aligned = align_to_common_labels(loaded)
    print(f"\nComposite common emotions ({len(common)}): {common}")

    # Experiment 2 — merge everything, 5-fold CV.
    X_all = np.concatenate([X for _, X, _ in aligned])
    y_all = np.concatenate([y for _, _, y in aligned])
    print(f"\n{'=' * 70}\n# Composite exp 2 — merged 5-fold CV  (X={X_all.shape})\n{'=' * 70}")
    cv = cross_validate(X_all, y_all, common, cfg, device, patience, verbose=True)

    # Experiment 1 — leave-one-dataset-out: train on the rest, test on the held-out.
    holdouts = []
    for i, (name, X_te, y_te) in enumerate(aligned):
        X_tr = np.concatenate([X for j, (_, X, _) in enumerate(aligned) if j != i])
        y_tr = np.concatenate([y for j, (_, _, y) in enumerate(aligned) if j != i])
        print(f"\n{'=' * 70}\n# Composite exp 1 — hold out '{name}' "
              f"(train={len(y_tr)}, test={len(y_te)})\n{'=' * 70}")
        scaler = StandardScaler()
        # Carve a small stratified val split out of train for early stopping.
        X_t, X_v, y_t, y_v = train_test_split(
            X_tr, y_tr, test_size=0.1, stratify=y_tr,
            random_state=cfg.random_state,
        )
        X_t = scale_fold(scaler, X_t, fit=True)
        X_v = scale_fold(scaler, X_v, fit=False)
        torch.manual_seed(cfg.random_state)
        res = train_one_fold(X_t, y_t, X_v, y_v, len(common),
                             cfg, device, patience, verbose=True)
        model = build_model(res["time_steps"], res["feature_dim"],
                            len(common), cfg).to(device)
        model.load_state_dict(res["best_state"])
        test_acc = evaluate(model, scale_fold(scaler, X_te, fit=False), y_te, device)
        print(f"Hold-out '{name}' test acc = {test_acc:.4f}")
        holdouts.append({
            "holdout": name,
            "train_acc": res["train_acc_at_best"],
            "test_acc": test_acc,
            "paper_acc": PAPER_COMPOSITE.get(f"holdout_{normalize_key(name)}"),
        })

    return {
        "common_labels": common,
        "merged_5fold": {
            "train_acc": cv["mean_train_acc"],
            "test_acc": cv["mean_val_acc"],
            "test_std": cv["std_val_acc"],
            "paper_acc": PAPER_COMPOSITE["merged_5fold"],
        },
        "holdouts": holdouts,
    }


def render_table(per_dataset: List[Dict], composite: Dict | None) -> str:
    lines = []
    lines.append("Per-dataset 5-fold cross-validation (Section 4.1)")
    lines.append("-" * 70)
    lines.append(f"{'Dataset':<18}{'Seqs':>6}{'Train%':>9}"
                 f"{'Test%':>9}{'±std':>7}{'Paper%':>9}{'Δ':>8}")
    lines.append("-" * 70)
    for r in per_dataset:
        delta = ("" if r["paper_acc"] is None
                 else f"{100 * (r['test_acc'] - r['paper_acc']):+.1f}")
        lines.append(
            f"{r['dataset']:<18}{r['n_sequences']:>6}"
            f"{fmt_pct(r['train_acc']):>9}{fmt_pct(r['test_acc']):>9}"
            f"{100 * r['test_std']:>7.1f}{fmt_pct(r['paper_acc']):>9}{delta:>8}"
        )
    lines.append("-" * 70)

    if composite:
        lines.append("")
        lines.append("Composite experiments (Section 4.2)")
        lines.append("-" * 70)
        m = composite["merged_5fold"]
        lines.append(f"{'merged 5-fold CV':<30}"
                     f"test {fmt_pct(m['test_acc'])}  "
                     f"paper {fmt_pct(m['paper_acc'])}")
        for h in composite["holdouts"]:
            lines.append(f"{'hold out ' + h['holdout']:<30}"
                         f"test {fmt_pct(h['test_acc'])}  "
                         f"paper {fmt_pct(h['paper_acc'])}")
        lines.append("-" * 70)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="One or more `name=path.npz` (or bare paths)")
    ap.add_argument("--out", default="runs/benchmark", help="Output directory")
    ap.add_argument("--composite", action="store_true",
                    help="Also run the Section 4.2 merged / hold-out experiments")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.folds is not None:
        cfg.folds = args.folds

    datasets = parse_datasets(args.datasets)
    per_dataset = run_per_dataset(datasets, cfg, device, args.patience)

    composite = None
    if args.composite:
        if len(datasets) < 2:
            print("\n--composite needs >=2 datasets; skipping composite experiments.")
        else:
            composite = run_composite(datasets, cfg, device, args.patience)

    table = render_table(per_dataset, composite)
    print("\n\n" + "=" * 70)
    print("RESULTS  vs  Köksal & Gumus 2025")
    print("=" * 70)
    print(table)

    (out_dir / "results.json").write_text(json.dumps(
        {"per_dataset": per_dataset, "composite": composite, "config": cfg.__dict__},
        indent=2,
    ))
    (out_dir / "results.md").write_text("```\n" + table + "\n```\n")
    print(f"\nSaved → {out_dir / 'results.json'}  and  {out_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
