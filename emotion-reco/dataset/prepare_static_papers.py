"""
papers_dataset.py
=================
Builds the FaceMesh dataset from the datasets referenced in the project's
papers, instead of the generic AffectNet-HQ / RAF-DB used by Dataset.py.

Datasets pulled (all from HuggingFace, both 7-class Ekman):
  - CK+      — Köksal & Gumus 2025, arXiv:2512.05669
               (apex frames give a clean static FER set; MediaPipe FaceMesh
               is the exact landmark detector used in that paper)
  - FER-2013 — Goodfellow et al. 2013
               (the standard FER benchmark referenced in Jakhete & Kulkarni's
               ICCUBEA-2024 survey and Luan et al. 2025)

Pipeline
--------
For each dataset:
  1. Download from HuggingFace (multiple repo IDs tried as fall-back).
  2. Run MediaPipe FaceLandmarker per image (same helpers as Dataset.py).
  3. Remap dataset-specific class names to the project's 7-class vocabulary.
  4. Stratified 80/20 train/val split with a fixed seed.

Output layout
-------------
./facemesh_dataset/
    papers_<ds>_train.npz   {landmarks, labels, label_names, [blendshapes]}
    papers_<ds>_val.npz     same keys
    papers_metadata.json    counts, HF IDs, paper attribution, split seed

The filenames contain "train"/"val", so TrainVla/train.py picks them up
automatically through its existing glob-and-route logic.

Usage
-----
    cd Dataset/
    python papers_dataset.py                            # CK+ + FER-2013
    python papers_dataset.py --datasets ckplus
    python papers_dataset.py --mode both                # also blendshapes
    python papers_dataset.py --val-frac 0.15 --seed 7
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Reuse the MediaPipe + cleaning helpers from the sibling prepare_static module.
from . import prepare_static as _base

_THIS_DIR = Path(__file__).resolve().parent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared 7-class Ekman vocabulary — must match TrainVla/train.py:UNIFIED_EMOTIONS
# ─────────────────────────────────────────────────────────────────────────────
UNIFIED_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Maps every per-dataset class name we expect to see → unified index.
# CK+ "contempt" is intentionally absent: those samples are dropped to keep
# 7-class parity with the existing trainer.
NAME_TO_UNIFIED = {
    "anger":     0, "angry":     0,
    "disgust":   1,
    "fear":      2,
    "happy":     3, "happiness": 3,
    "neutral":   4,
    "sad":       5, "sadness":   5,
    "surprise":  6,
}


# ─────────────────────────────────────────────────────────────────────────────
# Paper-referenced datasets on HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
# Each entry lists candidate repo IDs; the first one that loads is used.
# Edit the `candidates` list if you have a preferred / verified mirror.
PAPERS_DATASETS = {
    "ckplus": {
        # AlirezaF138/ckplus-dataset: 981 imgs, single 'train' split,
        # `label` column is Value('string') ("anger","disgust",...,"contempt").
        "candidates": [
            "AlirezaF138/ckplus-dataset",
        ],
        "image_col": "image",
        "label_col": "label",
        "paper": "Köksal & Gumus 2025 (arXiv:2512.05669)",
    },
    "fer2013": {
        # AutumnQiu/fer2013: full FER-2013 (28709/3589/3589), ClassLabel int,
        # names ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'].
        "candidates": [
            "AutumnQiu/fer2013",
            "3una/Fer2013",
        ],
        "image_col": "image",
        "label_col": "label",
        "paper": "Goodfellow et al. 2013 — benchmark in Jakhete & Kulkarni 2024 / Luan et al. 2025",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_from_candidates(candidates: list, cache_root: str):
    """Try each candidate HuggingFace ID in order; return (DatasetDict, used_id)."""
    from datasets import load_dataset, load_from_disk

    last_err = None
    for cand in candidates:
        local_path = os.path.join(cache_root, cand.replace("/", "__"))

        if os.path.exists(local_path):
            try:
                log.info("Loading cached: %s", local_path)
                return load_from_disk(local_path), cand
            except Exception as e:
                log.warning("Cache load failed for %s (%s); re-downloading…", cand, e)

        log.info("Downloading %s from HuggingFace…", cand)
        try:
            ds = load_dataset(cand)
            ds.save_to_disk(local_path)
            log.info("  cached → %s", local_path)
            return ds, cand
        except Exception as e:
            log.warning("  %s failed (%s)", cand, e)
            last_err = e

    raise RuntimeError(
        f"All HuggingFace candidates failed.\n"
        f"  Tried: {candidates}\n"
        f"  Last error: {last_err}"
    )


def resolve_label_names(ds, label_col: str):
    """Return list[str] of class names if HF features expose them, else None."""
    try:
        first_split = list(ds.keys())[0]
        return list(ds[first_split].features[label_col].names)
    except Exception:
        return None


def normalize_label_column(ds, label_col: str):
    """
    Ensure the label column is integer-encoded so process_split (which calls
    int(label)) works. Returns (ds, label_names).

    Three cases:
      ClassLabel(int)  → no change; names come from feature.
      Value('string')  → enumerate unique strings → int IDs, build name list.
      Value('int*')    → no change; names come from resolve_label_names (None).
    """
    from datasets import ClassLabel

    first_split = list(ds.keys())[0]
    feat = ds[first_split].features[label_col]

    if isinstance(feat, ClassLabel):
        return ds, list(feat.names)

    is_string = getattr(feat, "dtype", None) == "string"
    if not is_string:
        return ds, resolve_label_names(ds, label_col)

    # Collect every unique string label across splits, sort for determinism.
    unique = sorted({
        s for split in ds.keys() for s in ds[split][label_col]
    })
    name_to_idx = {n: i for i, n in enumerate(unique)}
    log.info("  encoding string labels → ints: %s", unique)

    def _encode(ex):
        ex[label_col] = name_to_idx[ex[label_col]]
        return ex

    ds = ds.map(_encode)
    return ds, unique


# ─────────────────────────────────────────────────────────────────────────────
# Label remap + stratified split
# ─────────────────────────────────────────────────────────────────────────────

def remap_to_unified(labels_arr: np.ndarray, label_names: list | None):
    """
    Map dataset-native labels → unified 7-class indices.

    Returns
    -------
    mapped : (M,) int64 — unified-vocab labels for kept samples
    keep   : (N,) bool  — mask over the input array
    """
    if label_names is None:
        # Assume the dataset already uses the unified scheme; drop OOB labels.
        keep = (labels_arr >= 0) & (labels_arr < len(UNIFIED_EMOTIONS))
        return labels_arr[keep].astype(np.int64), keep

    mapped_full = np.full(len(labels_arr), -1, dtype=np.int64)
    for i, raw in enumerate(labels_arr):
        name = label_names[int(raw)].lower().strip()
        if name in NAME_TO_UNIFIED:
            mapped_full[i] = NAME_TO_UNIFIED[name]

    keep = mapped_full >= 0
    return mapped_full[keep], keep


def stratified_split(labels: np.ndarray, val_frac: float, seed: int):
    """Return (train_idx, val_idx) for a stratified shuffle split."""
    idx = np.arange(len(labels))

    # train_test_split needs at least 2 samples per class for stratify.
    counts = np.bincount(labels, minlength=len(UNIFIED_EMOTIONS))
    rare = [c for c, n in enumerate(counts) if 0 < n < 2]
    stratify = labels if not rare else None
    if rare:
        log.warning(
            "  Stratify disabled — classes with <2 samples: %s",
            [UNIFIED_EMOTIONS[c] for c in rare],
        )

    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_frac,
        stratify=stratify,
        random_state=seed,
        shuffle=True,
    )
    return train_idx, val_idx


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset processing (calls _base.process_split for each HF split)
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(ds_key: str, cfg: dict, landmarker, mode: str, cache_root: str):
    """
    Process every HF split for a dataset, concatenate, and remap labels.

    Returns
    -------
    features : dict   — {"landmarks": (N,478,3), ["blendshapes": (N,52)]}
    labels   : (N,) int64 in unified vocab
    stats    : dict with totals, skip reasons, per-class counts, used HF id
    """
    ds, used_id = load_from_candidates(cfg["candidates"], cache_root)
    log.info("  using HF id: %s", used_id)
    ds, label_names = normalize_label_column(ds, cfg["label_col"])
    log.info("  native classes: %s", label_names)

    feats_landmarks = []
    feats_blend = []
    raw_labels_chunks = []
    combined_stats = {"total": 0, "kept": 0, "skipped": 0, "skip_reasons": {}}

    for split_name, split_data in ds.items():
        log.info("  [%s] processing native split '%s' (%d examples)",
                 ds_key, split_name, len(split_data))
        features, labels_arr, stats = _base.process_split(
            split_data,
            landmarker,
            image_col=cfg["image_col"],
            label_col=cfg["label_col"],
            split_name=f"{ds_key}/{split_name}",
            label_names=label_names,
            preview=False,
            preview_scale=1.0,
            mode=mode,
        )
        if "landmarks" in features:
            feats_landmarks.append(features["landmarks"])
        if "blendshapes" in features:
            feats_blend.append(features["blendshapes"])
        raw_labels_chunks.append(labels_arr)

        combined_stats["total"]   += stats["total"]
        combined_stats["kept"]    += stats["kept"]
        combined_stats["skipped"] += stats["skipped"]
        for r, c in stats["skip_reasons"].items():
            combined_stats["skip_reasons"][r] = (
                combined_stats["skip_reasons"].get(r, 0) + c
            )

    raw_labels = (
        np.concatenate(raw_labels_chunks, axis=0)
        if raw_labels_chunks else np.zeros(0, dtype=np.int64)
    )

    out: dict = {}
    if feats_landmarks:
        out["landmarks"] = np.concatenate(feats_landmarks, axis=0)
    if feats_blend:
        out["blendshapes"] = np.concatenate(feats_blend, axis=0)

    mapped, keep = remap_to_unified(raw_labels, label_names)
    dropped = int((~keep).sum())
    if dropped:
        log.info("  [%s] dropped %d sample(s) with unmapped labels "
                 "(e.g. CK+ 'contempt')", ds_key, dropped)

    for k in list(out.keys()):
        out[k] = out[k][keep]

    combined_stats["used_hf_id"] = used_id
    combined_stats["paper"]      = cfg["paper"]
    combined_stats["dropped_unmapped"] = dropped
    combined_stats["per_class_kept"]   = {
        UNIFIED_EMOTIONS[c]: int((mapped == c).sum())
        for c in range(len(UNIFIED_EMOTIONS)) if (mapped == c).sum() > 0
    }
    return out, mapped, combined_stats


# ─────────────────────────────────────────────────────────────────────────────
# .npz writer
# ─────────────────────────────────────────────────────────────────────────────

def save_split_npz(path: Path, features: dict, labels: np.ndarray):
    """Save one train/val split in the same key layout as Dataset.py outputs."""
    kwargs: dict = {
        "labels":      labels.astype(np.int64),
        "label_names": np.array(UNIFIED_EMOTIONS),
    }
    if "landmarks" in features:
        kwargs["landmarks"] = features["landmarks"].astype(np.float32)
    if "blendshapes" in features:
        kwargs["blendshapes"]      = features["blendshapes"].astype(np.float32)
        kwargs["blendshape_names"] = np.array(_base.BLENDSHAPE_NAMES)
    np.savez_compressed(path, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.hf_cache
    Path(cache_root).mkdir(parents=True, exist_ok=True)

    selected_keys = (
        [k.strip() for k in args.datasets.split(",") if k.strip()]
        if args.datasets else list(PAPERS_DATASETS.keys())
    )
    unknown = [k for k in selected_keys if k not in PAPERS_DATASETS]
    if unknown:
        raise SystemExit(
            f"Unknown dataset key(s): {unknown}. "
            f"Available: {list(PAPERS_DATASETS.keys())}"
        )
    selected = {k: PAPERS_DATASETS[k] for k in selected_keys}

    log.info("Building MediaPipe FaceLandmarker from %s", args.model)
    landmarker = _base.build_landmarker(
        args.model, blendshapes=(args.mode in ("blendshapes", "both"))
    )
    log.info("Feature mode  : %s", args.mode)
    log.info("Datasets      : %s", list(selected.keys()))
    log.info("Unified vocab : %s", UNIFIED_EMOTIONS)
    log.info("Split         : %.0f%% train / %.0f%% val  (seed=%d)",
             (1 - args.val_frac) * 100, args.val_frac * 100, args.seed)

    metadata = {
        "split_seed":   args.seed,
        "val_fraction": args.val_frac,
        "label_names":  UNIFIED_EMOTIONS,
        "datasets":     {},
    }

    for ds_key, cfg in selected.items():
        log.info("═══ Dataset: %s  (%s) ═══", ds_key, cfg["paper"])
        try:
            feats, labels, stats = process_dataset(
                ds_key, cfg, landmarker, args.mode, cache_root
            )
        except Exception as e:
            log.error("[%s] aborted: %s: %s — skipping this dataset.",
                      ds_key, type(e).__name__, e)
            metadata["datasets"][ds_key] = {
                "error": f"{type(e).__name__}: {e}",
                "paper": cfg["paper"],
            }
            continue

        n = len(labels)
        if n == 0:
            log.warning("[%s] no usable samples after MediaPipe + label remap.", ds_key)
            metadata["datasets"][ds_key] = stats
            continue

        train_idx, val_idx = stratified_split(labels, args.val_frac, args.seed)
        log.info("  [%s] kept=%d  →  train=%d  val=%d",
                 ds_key, n, len(train_idx), len(val_idx))

        for split_name, idx in (("train", train_idx), ("val", val_idx)):
            split_feats = {k: v[idx] for k, v in feats.items()}
            split_labels = labels[idx]
            out_path = output_dir / f"papers_{ds_key}_{split_name}.npz"
            save_split_npz(out_path, split_feats, split_labels)
            log.info(
                "    %-32s n=%-6d  classes=%s",
                out_path.name, len(split_labels),
                {UNIFIED_EMOTIONS[c]: int((split_labels == c).sum())
                 for c in range(len(UNIFIED_EMOTIONS))
                 if (split_labels == c).sum() > 0},
            )

        stats["train_count"] = int(len(train_idx))
        stats["val_count"]   = int(len(val_idx))
        metadata["datasets"][ds_key] = stats

    meta_path = output_dir / "papers_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("Metadata → %s", meta_path)
    log.info("Done. Output: %s", output_dir.resolve())


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build a FaceMesh dataset from the papers' datasets "
            "(CK+, FER-2013) with stratified 80/20 train/val split."
        )
    )
    parser.add_argument(
        "--datasets", default="",
        help="Comma-separated subset (ckplus,fer2013). Default: all.",
    )
    parser.add_argument(
        "--output", default="cache/facemesh_dataset",
        help="Output dir for .npz files (default: cache/facemesh_dataset).",
    )
    parser.add_argument(
        "--mode", default="landmarks",
        choices=["landmarks", "blendshapes", "both"],
        help="Feature mode (see Dataset.py).",
    )
    parser.add_argument(
        "--model", default=_base.MODEL_PATH,
        help="Path to face_landmarker.task (auto-downloaded if missing).",
    )
    parser.add_argument(
        "--hf-cache", default="./datasets_papers",
        help="Local cache dir for HuggingFace downloads.",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.20,
        help="Fraction held out for validation (default: 0.20).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/val split (default: 42).",
    )
    args = parser.parse_args()
    main(args)
