"""
run_all_benchmarks.py
=====================
Runs every facial emotion training script in sequence and prints a
side-by-side comparison of accuracy, latency, and hardware utilization.

  1. train.py        — EmoNet (dual-stream CNN+MLP on 478 landmarks)
                       Adapted from DriveEmo-FL (Sci Rep 2026).
  2. train2.py       — Blendshape MLP / AttentionNet (Jakhete & Kulkarni 2024).
  3. train_geo.py    — Geometric features + MLP (Köksal & Gumus 2025).
  4. train_image.py  — MobileNetV2 on raw AffectNet+RAF-DB images.
                       This is the "original dataset" arm required by the
                       CalmWheel design slide.

Usage
-----
    python run_all_benchmarks.py                    # everything, defaults
    python run_all_benchmarks.py --quick            # 3 epochs each (smoke test)
    python run_all_benchmarks.py --skip image       # skip the image baseline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bench_logger import compare


METHODS = [
    {
        "name":   "emonet_landmark",
        "script": "train.py",
        "output": "../emonet_output",
        "metrics_glob": "emonet_landmark_metrics.json",
    },
    {
        "name":   "blendshape_mlp",
        "script": "train2.py",
        "output": "../blendshape_output",
        "args":   ["--model", "mlp"],
        "metrics_glob": "blendshape_mlp_metrics.json",
    },
    {
        "name":   "blendshape_attention",
        "script": "train2.py",
        "output": "../blendshape_attn_output",
        "args":   ["--model", "attention"],
        "metrics_glob": "blendshape_attention_metrics.json",
    },
    {
        "name":   "geo_61_au",
        "script": "train_geo.py",
        "output": "../geo_output",
        "args":   ["--landmarks", "61"],
        "metrics_glob": "geo_61_au_metrics.json",
    },
    {
        "name":   "image_mobilenetv2",
        "script": "train_image.py",
        "output": "../image_output",
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
    args = parser.parse_args()

    here = Path(__file__).resolve().parent

    selected = METHODS
    if args.only:
        selected = [m for m in METHODS if m["name"] in args.only]
    if args.skip:
        selected = [m for m in selected if m["name"] not in args.skip]

    # All facemesh-based trainers read .npz files from this folder.
    # Image baseline (train_image.py) ignores --data; harmless to pass.
    data_dir = (here / ".." / "facemesh_dataset").resolve()

    print(f"Running {len(selected)} methods…  (data: {data_dir})")

    for m in selected:
        cmd = [sys.executable, str(here / m["script"]),
               "--output", str(here / m["output"])]
        if m["script"] != "train_image.py":
            cmd += ["--data", str(data_dir)]
        cmd += m.get("args", [])
        if args.quick:
            cmd += ["--epochs", "3"]
        print("\n" + "═" * 78)
        print(" ".join(cmd))
        print("═" * 78)
        subprocess.run(cmd, check=False, cwd=here)

    # ── Gather metrics ──────────────────────────────────────────────────────
    metrics_paths = []
    for m in selected:
        path = here / m["output"] / m["metrics_glob"]
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
