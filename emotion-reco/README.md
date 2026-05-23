# Driver-Companion Emotion Recognition

PyTorch implementation of **six** facial emotion recognition methods for the
**CalmWheel** driver-intervention project (COMPSYS 731, Group 10). Methods are
split across two data modalities — *sequential* (CK+, RAVDESS) and
*static-image* (AffectNet, RAF-DB) — so the design's "train and test on
facemesh dataset *and* original dataset" comparison can be run side-by-side.

## Methods

| # | Trainer | Modality | Paper | Idea | Paper acc |
|---|---|---|---|---|---:|
| 1 | `training/train.py` (ConvLSTM1D) | sequential | Köksal & Gumus 2025 ([arXiv:2512.05669](https://arxiv.org/abs/2512.05669)) | 5-keyframe landmark distance+angle → ConvLSTM1D | CK+ 93 % |
| 2 | `training/baseline.py` | sequential | Juárez-Jiménez et al. 2025 ([arXiv:2506.17191](https://arxiv.org/abs/2506.17191)) | 2-keyframe landmark displacement → MLP / decision tree | CK+ 98.5 % |
| 3 | `training/train_emonet.py` | static-image | DriveEmo-FL (Sci Rep 2026) | dual-stream landmark CNN + geometric MLP fusion | — |
| 4 | `training/train_blendshape.py` | static-image | Jakhete & Kulkarni 2024 (ICCUBEA) | 52 MediaPipe blendshapes → MLP / attention | CK+ 98.8 % |
| 5 | `training/train_geo_static.py` | static-image | Köksal & Gumus 2025 (static adaptation) | landmark distance+angle → MLP | — |
| 6 | `training/train_image.py` | static-image | — (CalmWheel baseline) | MobileNetV2 fine-tuned on raw images | — |

Methods 1-2 reach 90 %+ on CK+; methods 3-6 typically cap around **55-65 %**
on AffectNet/RAF-DB — that is the right number for those datasets, not a bug
(see [About the 60 % ceiling](#about-the-60-ceiling)).

## Setup

```bash
pip install -r requirements.txt
```

Run every command below from inside `emotion-reco/` so the `dataset`,
`training` and `implementation` packages import correctly. The MediaPipe model
file (`models/face_landmarker.task`, ~3.8 MB) downloads automatically on first
use.

## Layout

```
emotion-reco/
├── dataset/
│   ├── landmarks.py            # FaceLandmarker wrapper, 61/122/250 presets
│   ├── features.py             # distance + angle features
│   ├── dataset.py              # sequence folder walker + keyframe sampler
│   ├── prepare_data.py         # sequential → ConvLSTM feature cache
│   ├── prepare_baseline.py     # sequential → displacement feature cache
│   ├── prepare_static.py       # AffectNet/RAF-DB → landmark/blendshape cache
│   ├── prepare_static_papers.py# CK+/FER-2013 → landmark/blendshape cache
│   ├── fetch_ravdess.py        # download RAVDESS → frame folders
│   ├── organize_ckplus.py      # rearrange a full CK+ archive
│   ├── organize_ckplus_apex.py # rearrange a flattened apex-only CK+
│   └── make_synthetic.py       # fake feature cache for smoke tests
├── training/
│   ├── config.py               # ConvLSTM hyperparameters
│   ├── model.py                # ConvLSTM1D model
│   ├── train.py                # 5-fold CV trainer (method 1)
│   ├── baseline.py             # 4-fold CV baseline (method 2)
│   ├── benchmark.py            # multi-dataset sequential benchmark
│   ├── train_emonet.py         # static-image method 3
│   ├── train_blendshape.py     # static-image method 4
│   ├── train_geo_static.py     # static-image method 5
│   ├── train_image.py          # static-image method 6
│   ├── bench_logger.py         # shared hardware/timing logger
│   └── run_all_benchmarks.py   # runs methods 3-6 side-by-side
└── implementation/
    └── realtime.py             # real-time webcam inference (sequential model)
```

`cache/`, `runs/`, `data/`, `datasets/` and `models/` are created at runtime
and are git-ignored.

## Train and test — quick reference

Every method has three commands: prepare features, train, and test. Each
section below has the details; this table is the cheat sheet.

| # | Train | Test (held-out eval) | Real-time |
|---|---|---|---|
| 1 | `python -m training.train --features cache/ckplus.npz --out runs/ckplus` | (5-fold CV is the test; see `runs/ckplus/metrics.json`) | `python -m implementation.realtime --method sequential --model runs/ckplus/fold_best.pt --scaler runs/ckplus/scaler.pkl --labels runs/ckplus/labels.json` |
| 2 | `python -m training.baseline --features cache/ckplus_baseline.npz --out runs/baseline` | (4-fold CV is the test; see `runs/baseline/results.md`) | n/a (baseline needs neutral+peak — no webcam adapter) |
| 3 | `python -m training.train_emonet --data cache/ckplus_static --output runs/emonet_ckplus` | append `--eval-only --checkpoint runs/emonet_ckplus/checkpoints/best.pt` | `python -m implementation.realtime --method emonet --model runs/emonet_ckplus/checkpoints/best.pt` |
| 4 | `python -m training.train_blendshape --data cache/ckplus_static --output runs/blendshape_ckplus --model attention` | append `--eval-only --checkpoint runs/blendshape_ckplus/checkpoints/best.pt` | `python -m implementation.realtime --method blendshape --model runs/blendshape_ckplus/checkpoints/best.pt` |
| 5 | `python -m training.train_geo_static --data cache/ckplus_static --output runs/geo_static_ckplus` | append `--eval-only --checkpoint runs/geo_static_ckplus/checkpoints/best.pt` | `python -m implementation.realtime --method geo_static --model runs/geo_static_ckplus/checkpoints/best.pt` |
| 6 | `python -m training.train_image --output runs/image --backbone mobilenet_v2` | append `--eval-only --checkpoint runs/image/checkpoints/best.pt` | `python -m implementation.realtime --method image --model runs/image/checkpoints/best.pt --backbone mobilenet_v2` |

Substitute the dataset cache (`cache/ckplus.npz` → `cache/ravdess.npz`, etc.)
and the output directory (`runs/ckplus_*`) to switch datasets.

---

# Sequential pipeline (CK+, RAVDESS)

Each sample is a folder of chronologically-ordered frames showing one
expression from neutral → apex. Both methods feed those keyframes into a
MediaPipe FaceLandmarker (478 landmarks) and turn the result into geometric
features.

## Quick start

```bash
# 1. Get a sequential dataset — pick one:
#    a) RAVDESS (free, downloads from Zenodo; ~1.7 GB for actors 1-3)
python -m dataset.fetch_ravdess --out_root data/ravdess --actors 1-3
#    b) CK+ — full multi-frame archive (gated)
python -m dataset.organize_ckplus --ckplus_root /path/to/ckplus --out_root data/ckplus
#    c) CK+ — flattened apex-only redistribution (e.g. CS229 mirror)
python -m dataset.organize_ckplus_apex --ck_root /path/to/CK+ --out_root data/ckplus
#    d) Synthetic — no real data, smoke test only
python -m dataset.make_synthetic --output cache/synth.npz --sequences_per_class 60

# 2. Extract geometric features into a cache (.npz)
python -m dataset.prepare_data --data_root data/ckplus \
    --output cache/ckplus.npz --landmarks 61

# 3. Train with 5-fold cross-validation
python -m training.train --features cache/ckplus.npz --out runs/ckplus

# 4. Real-time webcam inference
python -m implementation.realtime --model runs/ckplus/fold_best.pt \
    --scaler runs/ckplus/scaler.pkl --labels runs/ckplus/labels.json
```

Training artifacts land in `runs/<name>/`: `fold_best.pt` (checkpoint — state
dict + arch dims + labels + landmark config), per-fold `fold_*.pt`,
`scaler.pkl`, `labels.json`, `metrics.json`, `training_curves.png`.

### Epochs and early stopping

`train.py` defaults to 200 epochs with early stopping (`--patience 30`). To
run **all** epochs, set patience ≥ epochs:

```bash
python -m training.train --features cache/ckplus.npz --out runs/ckplus \
    --epochs 200 --patience 200
```

The checkpoint always stores the best-validation epoch, not the last one.

## Landmark presets

Method 1's paper experiments with three landmark counts. Pick one at
`prepare_data` time with `--landmarks`:

| `--landmarks` | Landmarks | Pairs | Feature dim `A` | Notes |
|---|---:|---:|---:|---|
| `61` (default) | 61 | 1 456 | 2 912 | paper's best speed/accuracy point |
| `122` | 121 | 6 110 | 12 220 | ~200 M-param model |
| `250` | 253 | 26 866 | 53 732 | ~880 M-param model — heavy, slow |

The preset is stored in the `.npz` and copied into every checkpoint, so
`train.py` and `realtime.py` pick it up automatically. The same preset flag
exists on `dataset.prepare_baseline` so method 2 stays comparable.

## Reproducing the paper's results (method 1)

`training/benchmark.py` runs the paper's evaluation protocol:

```bash
# Per-dataset 5-fold CV (paper Section 4.1)
python -m training.benchmark --datasets ckplus=cache/ckplus.npz --out runs/benchmark

# Multiple datasets + composite experiments (Section 4.2 — needs >=2 caches)
python -m training.benchmark \
    --datasets ckplus=cache/ckplus.npz oulu_casia_vis=cache/oulu_vis.npz \
               oulu_casia_nir=cache/oulu_nir.npz mmi=cache/mmi.npz \
    --composite --out runs/benchmark
```

Dataset keys `ckplus`, `oulu_casia_vis`, `oulu_casia_nir`, `mmi` are matched to
the paper's accuracies (93 / 79 / 77 / 68 %) for a side-by-side `Δ` column.

## Baseline classifier (method 2)

Reimplementation of Juárez-Jiménez et al. 2025 — 2-keyframe landmark
displacement + plain MLP / decision tree. No official code was released.

```bash
# Extract baseline features (neutral + peak displacement)
python -m dataset.prepare_baseline --data_root data/ckplus \
    --output cache/ckplus_baseline.npz --landmarks 61

# 4-fold CV: tuned decision tree, basic MLP, and the paper's "optimized" MLP
python -m training.baseline --features cache/ckplus_baseline.npz \
    --out runs/baseline
```

Pass `--feature absolute` to classify raw peak positions instead of
displacements — the paper compares both.

Example on the apex-only CK+ (327 sequences, 61-landmark preset):

```
Model               Test%   ±std   Paper%      Δ
decision_tree        81.4     5.8     80.0   +1.4
basic_nn             85.6     4.6     97.0  -11.3
optimized_nn         91.4     3.1     98.5   -7.0
```

Decision tree reproduces the paper closely; the MLPs land lower because the
apex-only redistribution has 327 sequences (vs 593) and uses MediaPipe rather
than Dlib landmarks. Displacement beats raw positions by ~7 points, confirming
the paper's main result.

---

# Static-image pipeline (AffectNet, RAF-DB)

Each sample is a single in-the-wild face image with one of seven Ekman
emotions: `angry, disgust, fear, happy, neutral, sad, surprise`. All four
trainers share a common protocol — stratified split, balanced sampling,
cosine LR schedule, label-smoothed cross-entropy, early stopping, and a
hardware/timing log via `bench_logger.py`.

## Dataset prep

The four static-image trainers all consume `.npz` files written by
`dataset.prepare_static`. One MediaPipe pass per image extracts either the 478
landmarks, the 52 blendshape coefficients, or both, and bundles each
HuggingFace split into a single `.npz`.

```bash
# Pull AffectNet-HQ + RAF-DB from HuggingFace, run MediaPipe, save .npz files.
# Pick --mode for the methods you want to train downstream:
#   landmarks   → train_emonet, train_geo_static
#   blendshapes → train_blendshape
#   both        → everything (one MediaPipe pass, larger files)
python -m dataset.prepare_static --output cache/facemesh_dataset --mode both

# Variant — use the datasets the papers actually use (CK+ + FER-2013)
python -m dataset.prepare_static_papers --output cache/facemesh_dataset --mode both

# Variant — convert an existing sequential folder (e.g. your local CK+) into
# a static cache by taking the apex (last) frame of each sequence. Use this
# when the HuggingFace `ckplus-dataset` mirror is too small (48 x 48) for
# FaceMesh and you already organized full-res CK+ via organize_ckplus_apex.
python -m dataset.prepare_static_from_sequences \
    --data_root data/ckplus --output cache/ckplus_static --mode both
```

`prepare_static.py` filters images with no detected face, multi-face images,
blurry frames (Laplacian variance < 20) and <48 px sources. Each output file
contains:

| Key | Shape | Description |
|---|---|---|
| `landmarks` | `(N, 478, 3)` | per-image 3D MediaPipe landmarks (modes `landmarks`/`both`) |
| `blendshapes` | `(N, 52)` | MediaPipe blendshape coefficients (modes `blendshapes`/`both`) |
| `labels` | `(N,)` | unified emotion class index 0-6 (Ekman) |
| `label_names` | `(num_classes,)` | original dataset label strings |
| `blendshape_names` | `(52,)` | canonical MediaPipe blendshape names |

## Training the four methods

```bash
# 3. Dual-stream landmark CNN + geometric-features MLP (DriveEmo-FL adaptation)
python -m training.train_emonet --data cache/facemesh_dataset --output runs/emonet

# 4a. Blendshape MLP (Jakhete & Kulkarni 2024)
python -m training.train_blendshape --data cache/facemesh_dataset \
    --output runs/blendshape_mlp --model mlp

# 4b. Blendshape with per-coefficient attention gate (interpretable variant)
python -m training.train_blendshape --data cache/facemesh_dataset \
    --output runs/blendshape_attn --model attention

# 5. Static-image adaptation of Köksal & Gumus 2025
python -m training.train_geo_static --data cache/facemesh_dataset \
    --output runs/geo_static --landmarks 61

# 6. MobileNetV2 fine-tuned on raw AffectNet+RAF-DB images
#    (ignores --data; loads from HuggingFace directly)
python -m training.train_image --output runs/image --backbone mobilenet_v2
```

Each run writes to `runs/<name>/`:
- `checkpoints/best.pt`, `final.pt`
- `evaluation_report.txt`, `confusion_matrix.png`, `training_curves.png`
- `<method>_metrics.json`, `<method>_log.txt` (bench logger output)
- `run_summary.json`

The blendshape attention variant additionally produces
`blendshape_importance.png` (per-coefficient attention weights, colour-coded by
emotion association) and `blendshape_heatmap.png` (mean activation per
emotion).

## Testing trained static models

Each static trainer accepts `--eval-only --checkpoint <path>` to skip training
and run just the held-out evaluation against an existing best checkpoint. The
generated `evaluation_report.txt` + `confusion_matrix.png` overwrite the ones
in `--output`.

```bash
# Method 3 — EmoNet
python -m training.train_emonet --data cache/ckplus_static \
    --output runs/emonet_ckplus \
    --eval-only --checkpoint runs/emonet_ckplus/checkpoints/best.pt

# Method 4 — Blendshape (works for both --model mlp and attention)
python -m training.train_blendshape --data cache/ckplus_static \
    --output runs/blendshape_ckplus --model attention \
    --eval-only --checkpoint runs/blendshape_ckplus/checkpoints/best.pt

# Method 5 — Geo static
python -m training.train_geo_static --data cache/ckplus_static \
    --output runs/geo_static_ckplus \
    --eval-only --checkpoint runs/geo_static_ckplus/checkpoints/best.pt

# Method 6 — Image MobileNetV2
python -m training.train_image --output runs/image \
    --backbone mobilenet_v2 \
    --eval-only --checkpoint runs/image/checkpoints/best.pt
```

## Cross-method benchmark

`run_all_benchmarks.py` runs methods 3-6 sequentially and prints an
accuracy / latency / hardware-utilisation comparison:

```bash
# Everything — several hours on one GPU
python -m training.run_all_benchmarks

# Smoke test — 3 epochs each, ~10 min
python -m training.run_all_benchmarks --quick

# Skip or pick methods by name
python -m training.run_all_benchmarks --skip blendshape_attention
python -m training.run_all_benchmarks --only image_mobilenetv2
```

The comparison table is built from each method's `*_metrics.json` and printed
as a markdown table — copy/paste-able into the project report.

## Bug fixes vs the original implementation

Several common-in-FER bugs were fixed during the integration; runs should
improve by **2-5 %** versus the original scripts:

- **Double-weighting** — every trainer was using both a
  `WeightedRandomSampler` (already produces balanced batches) *and* a
  class-weighted `CrossEntropyLoss` (weights each sample again). Together they
  over-correct toward rare classes and suppress test accuracy on imbalanced
  sets like AffectNet. Fix: keep the sampler, drop `weight=cw` from the loss.
- **Duplicate landmark index** in `train_geo_static.py`'s `_LOWER_JAW` list
  (`200` and `396` appeared twice).
- **Hyperparameter tuning**: lower LR for EmoNet (`5e-4`), more epochs for
  MobileNetV2 (35), higher dropout for the static GeoMLP (0.5), longer
  early-stopping patience across the board.

---

## About the 60 % ceiling

The static-image methods consistently score **55-65 % on AffectNet/RAF-DB
7-class**, and that is the right number — not a bug.

- **AffectNet 7-class SOTA is ~65 %**. Labels are crowdsourced from in-the-
  wild images and have ~30 % inter-annotator disagreement, which caps any
  classifier.
- The two papers cited (Jakhete & Kulkarni 2024 at 98.8 %, Köksal & Gumus 2025
  at 93 %) were both evaluated on **CK+** — a lab dataset of frontal-pose
  apex frames so close to perfect that even simple models saturate. Those
  numbers are not directly comparable to AffectNet.
- The sequential methods 1 and 2 hit 90 %+ here because they *do* run on CK+.

To get those high numbers from the static-image trainers, point them at CK+
via `prepare_static_papers`:

```bash
python -m dataset.prepare_static_papers --datasets ckplus --mode both \
    --output cache/facemesh_papers
python -m training.train_blendshape --data cache/facemesh_papers \
    --output runs/blendshape_ckplus --model attention
```

---

## Real-time inference (webcam)

`implementation/realtime.py` handles all five inference paths from one
script — pick the model with `--method`. The sequential predictor buffers
frames; the four static predictors classify each frame independently.

```bash
# Method 1 — ConvLSTM1D (needs scaler + labels from training)
python -m implementation.realtime --method sequential \
    --model runs/ckplus/fold_best.pt \
    --scaler runs/ckplus/scaler.pkl \
    --labels runs/ckplus/labels.json

# Method 3 — EmoNet (landmark CNN + geo MLP)
python -m implementation.realtime --method emonet \
    --model runs/emonet_ckplus/checkpoints/best.pt

# Method 4 — Blendshape (MLP vs attention is auto-detected from the checkpoint)
python -m implementation.realtime --method blendshape \
    --model runs/blendshape_ckplus/checkpoints/best.pt

# Method 5 — Geo static (landmark distance+angle MLP)
python -m implementation.realtime --method geo_static \
    --model runs/geo_static_ckplus/checkpoints/best.pt

# Method 6 — MobileNetV2 on raw RGB
python -m implementation.realtime --method image \
    --model runs/image/checkpoints/best.pt --backbone mobilenet_v2
```

All static methods predict from the seven-class Ekman vocabulary
(`angry, disgust, fear, happy, neutral, sad, surprise`), so they don't need a
`labels.json`. The sequential method does — it preserves the dataset's own
class set (CK+ keeps `contempt`, RAVDESS uses `calm`, etc.).

Press `q` to close the window. Pass `--cpu` to force CPU inference.

Method 2 (baseline) is not wired into the webcam demo — it needs an explicit
neutral reference frame to compute displacement, which doesn't naturally fit a
single-stream webcam UX.

---

## Dataset access

| Dataset | Modality | Access |
|---|---|---|
| RAVDESS | sequential | free (Zenodo, CC BY-NC-SA 4.0) — `dataset.fetch_ravdess` |
| CK+ | sequential / static | gated — https://www.jeffcohn.net/Resources/ |
| Oulu-CASIA | sequential | gated — https://www.oulu.fi/cmvs/node/41319 |
| MMI | sequential | gated — https://mmifacedb.eu/ |
| AffectNet-HQ | static | HuggingFace `Piro17/affectnethq` (auto) |
| RAF-DB | static | HuggingFace `deanngkl/raf-db-7emotions` (auto) |
| FER-2013 | static | HuggingFace (used by `prepare_static_papers`) |

Each RAVDESS actor zip is ~553 MB; interrupted downloads resume.

## Expected sequential layout

`prepare_data.py` and `prepare_baseline.py` expect one folder per class, each
holding one sub-folder per sequence of chronologically-ordered frames:

```
data_root/
├── anger/
│   ├── S005_001/
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   └── ...
│   └── S010_002/...
├── happy/
│   └── ...
```

Both `fetch_ravdess.py` and `organize_ckplus*.py` produce exactly this layout.
