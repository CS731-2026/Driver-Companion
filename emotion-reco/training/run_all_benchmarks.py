"""
run_all_benchmarks.py
=====================
Runs every static-image emotion training script in sequence and prints a
side-by-side comparison of accuracy, latency, and hardware utilization.

  1. train_emonet.py      — EmoNet (dual-stream CNN+MLP on 478 landmarks)
                            Adapted from DriveEmo-FL (Sci Rep 2026).
  2. train_blendshape.py  — Blendshape MLP / AttentionNet
                            (Jakhete & Kulkarni 2024).
  3. train_geo_static.py  — Geometric features + MLP, static-image
                            adaptation of Köksal & Gumus 2025.
  4. train_image.py       — MobileNetV2 on raw AffectNet+RAF-DB images
                            ("original-dataset" arm of the CalmWheel design).

Run from the emotion-reco/ project root:

    python -m training.run_all_benchmarks                  # everything
    python -m training.run_all_benchmarks --quick          # 3 epochs each
    python -m training.run_all_benchmarks --skip image     # skip a method
    python -m training.run_all_benchmarks --only emonet_landmark
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .bench_logger import compare


METHODS = [
    {
        "name":   "emonet_landmark",
        "module": "training.train_emonet",
        "output": "runs/emonet",
        "metrics_glob": "emonet_landmark_metrics.json",
    },
    {
        "name":   "blendshape_mlp",
        "module": "training.train_blendshape",
        "output": "runs/blendshape_mlp",
        "args":   ["--model", "mlp"],
        "metrics_glob": "blendshape_mlp_metrics.json",
    },
    {
        "name":   "blendshape_attention",
        "module": "training.train_blendshape",
        "output": "runs/blendshape_attn",
        "args":   ["--model", "attention"],
        "metrics_glob": "blendshape_attention_metrics.json",
    },
    {
        "name":   "geo_61_au",
        "module": "training.train_geo_static",
        "output": "runs/geo_static",
        "args":   ["--landmarks", "61"],
        "metrics_glob": "geo_61_au_metrics.json",
    },
    {
        "name":   "image_mobilenetv2",
        "module": "training.train_image",
        "output": "runs/image",
        "args":   ["--backbone", "mobilenet_v2"],
        "metrics_glob": "image_mobilenet_v2_metrics.json",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Override epochs=3 for smoke testing")
    parser.add_argument("--skip", action="append", default=[],
                        help="Skip a method by name (repeatable)")
    parser.add_argument("--only", action="append", default=[],
                        help="Run only this method (repeatable)")
    parser.add_argument("--data", default="cache/facemesh_dataset",
                        help="Shared --data path for the facemesh-based trainers "
                             "(default: cache/facemesh_dataset)")
    args = parser.parse_args()

    # This file lives at training/run_all_benchmarks.py — go up one level to
    # the emotion-reco/ project root, which is the right cwd for `python -m`.
    emotion_reco = Path(__file__).resolve().parent.parent

    selected = METHODS
    if args.only:
        selected = [m for m in METHODS if m["name"] in args.only]
    if args.skip:
        selected = [m for m in selected if m["name"] not in args.skip]

    print(f"Running {len(selected)} methods…  (cwd: {emotion_reco}  data: {args.data})")

    for m in selected:
        cmd = [sys.executable, "-m", m["module"], "--output", m["output"]]
        # train_image.py loads images directly from HuggingFace, ignores --data.
        if m["module"] != "training.train_image":
            cmd += ["--data", args.data]
        cmd += m.get("args", [])
        if args.quick:
            cmd += ["--epochs", "3"]
        print("\n" + "═" * 78)
        print(" ".join(cmd))
        print("═" * 78)
        subprocess.run(cmd, check=False, cwd=emotion_reco)

    # ── Gather metrics ──────────────────────────────────────────────────────
    metrics_paths = []
    for m in selected:
        path = emotion_reco / m["output"] / m["metrics_glob"]
        if path.exists():
            metrics_paths.append(path)
        else:
            print(f"⚠  Missing: {path}")

    if metrics_paths:
        print("\n\n══ Comparison ══")
        print(compare(metrics_paths))
    else:
        print("No metrics files produced — see per-method logs.")


if __name__ == "__main__":
    main()
