# Emotion Recognition (Köksal & Gumus 2025)

PyTorch reference implementation of *"Deep Learning-Based Real-Time Sequential
Facial Expression Analysis Using Geometric Features"* (arXiv:2512.05669v1).

## Method

1. MediaPipe FaceLandmarker (Tasks API) → 478 landmarks per frame.
2. Select a landmark preset (61 / 122 / 250) across 5 FACS-based groups.
3. For 5 key frames (neutral → apex), compute Euclidean distance and angle for
   every same-group landmark pair, then subtract the previous frame's features.
   Output: an `(N-1, A) = (4, A)` tensor per sequence.
4. StandardScaler → ConvLSTM1D(filters=8, kernel=1) → Flatten → MLP
   (2048 → 1024 → num_classes).
5. Trained with Adam + cross-entropy loss, stratified 5-fold cross-validation.

PyTorch has no built-in ConvLSTM, so `training/model.py` implements a
ConvLSTM1D cell (standard ConvLSTM, no peephole terms — matching the Keras
`ConvLSTM1D` the paper actually trained).

## Layout

```
emotion-reco/
├── dataset/
│   ├── landmarks.py            # FaceLandmarker wrapper, 61/122/250 presets
│   ├── features.py             # distance + angle features, frame differencing
│   ├── dataset.py              # sequence folder walker + keyframe sampler
│   ├── prepare_data.py         # raw frames → ConvLSTM feature cache (.npz)
│   ├── prepare_baseline.py     # raw frames → baseline feature cache (.npz)
│   ├── fetch_ravdess.py        # download RAVDESS → frame folders
│   ├── organize_ckplus.py      # rearrange a full CK+ archive
│   ├── organize_ckplus_apex.py # rearrange a flattened apex-only CK+
│   └── make_synthetic.py       # fake feature cache for pipeline smoke tests
├── training/
│   ├── config.py               # hyperparameters
│   ├── model.py                # PyTorch ConvLSTM1D + MLP
│   ├── train.py                # single-dataset 5-fold CV training
│   ├── benchmark.py            # multi-dataset results table vs the paper
│   └── baseline.py             # paper-2 baseline classifiers (4-fold CV)
└── implementation/
    └── realtime.py             # real-time webcam inference (5-frame buffer)
```

`cache/`, `runs/`, `data/` and `models/` are created at runtime and are
git-ignored.

## Setup

```bash
pip install -r requirements.txt
```

Run every command below from inside `emotion-reco/` so the `dataset`,
`training` and `implementation` packages import correctly. The MediaPipe model
file (`models/face_landmarker.task`) downloads automatically on first use.

## Quick start

```bash
# 1. Get a dataset — pick one:
#    a) RAVDESS (free, downloads from Zenodo; ~1.7 GB for actors 1-3)
python -m dataset.fetch_ravdess --out_root data/ravdess --actors 1-3
#    b) CK+ (after you obtain access — see "Dataset access" below)
python -m dataset.organize_ckplus --ckplus_root /path/to/ckplus --out_root data/ckplus
#    c) Synthetic (no real data — for pipeline smoke tests only)
python -m dataset.make_synthetic --output cache/synth.npz --sequences_per_class 60

# 2. Extract geometric features into a cache (.npz).
#    Skip for the synthetic path — make_synthetic already writes the cache.
python -m dataset.prepare_data --data_root data/ravdess \
    --output cache/ravdess.npz --landmarks 61

# 3. Train with 5-fold cross-validation
python -m training.train --features cache/ravdess.npz --out runs/ravdess

# 4. Real-time webcam inference
python -m implementation.realtime --model runs/ravdess/fold_best.pt \
    --scaler runs/ravdess/scaler.pkl --labels runs/ravdess/labels.json
```

Training artifacts land in `runs/<name>/`: `fold_best.pt` (checkpoint — state
dict + arch dims + labels + landmark config), per-fold `fold_*.pt`,
`scaler.pkl`, `labels.json`, `metrics.json`, `training_curves.png`.

### Epochs and early stopping

`train.py` defaults to 200 epochs with early stopping (`--patience 30` — stops
if validation accuracy stalls for 30 epochs). To run **all** epochs, set
patience ≥ epochs:

```bash
python -m training.train --features cache/ravdess.npz --out runs/ravdess \
    --epochs 200 --patience 200
```

The checkpoint always stores the best-validation epoch, not the last one.

## Landmark presets

The paper experiments with three landmark counts. Pick one at `prepare_data`
time with `--landmarks` (also available on `make_synthetic`):

| `--landmarks` | Landmarks | Pairs | Feature dim `A` | Notes |
|---|---:|---:|---:|---|
| `61` (default) | 61 | 1 456 | 2 912 | paper's best speed/accuracy point |
| `122` | 121 | 6 110 | 12 220 | ~200 M-param model |
| `250` | 253 | 26 866 | 53 732 | ~880 M-param model — heavy, slow |

The preset is stored in the `.npz` and copied into every checkpoint, so
`train.py` and `realtime.py` pick it up automatically — no flag needed
downstream. Changing the preset means re-running `prepare_data` (the feature
dimension changes). To customise *which* landmarks each preset uses, edit the
region lists in `dataset/landmarks.py`.

## Reproducing the paper's results

`training/benchmark.py` runs the paper's evaluation protocol and prints a
train/test table compared to the published numbers:

```bash
# Per-dataset 5-fold CV (Section 4.1)
python -m training.benchmark --datasets ravdess=cache/ravdess.npz --out runs/benchmark

# Multiple datasets + composite experiments (Section 4.2):
# merged 5-fold CV and leave-one-dataset-out hold-out tests
python -m training.benchmark \
    --datasets ckplus=cache/ckplus.npz oulu_casia_vis=cache/oulu_vis.npz \
               oulu_casia_nir=cache/oulu_nir.npz mmi=cache/mmi.npz \
    --composite --out runs/benchmark
```

Dataset keys `ckplus`, `oulu_casia_vis`, `oulu_casia_nir`, `mmi` are matched to
the paper's published accuracies (93 / 79 / 77 / 68 %) for a side-by-side `Δ`
column. Output: `runs/benchmark/results.json` and `results.md`.

To match the paper's *numbers* you need the paper's *datasets* — all gated (see
below). The benchmark still runs on any cache you have (e.g. RAVDESS); rows
without a paper baseline just show `—`.

## Paper baseline (Juárez-Jiménez et al. 2025)

A second paper — *"Facial Landmark Visualization and Emotion Recognition
Through Neural Networks"* ([arXiv:2506.17191](https://arxiv.org/abs/2506.17191))
— is reimplemented as a comparison baseline. No official code was released, so
`training/baseline.py` is a from-scratch PyTorch port of its Section 3.4
classifiers.

How it differs from the ConvLSTM1D model above:

| | ConvLSTM1D model | Baseline (paper 2) |
|---|---|---|
| Key frames | 5 (neutral → apex) | 2 (neutral, peak) |
| Feature | per-pair distance + angle, frame-differenced | per-landmark displacement (peak − neutral) |
| Classifier | ConvLSTM1D → MLP | plain MLP / decision tree |
| Cross-validation | 5-fold | 4-fold |

Both run on the same MediaPipe landmark preset, so a baseline run is directly
comparable — only the method changes. (The paper itself used Dlib's 68-point
detector.)

```bash
# 1. Extract baseline features (neutral + peak landmark displacement)
python -m dataset.prepare_baseline --data_root data/ckplus \
    --output cache/ckplus_baseline.npz --landmarks 61

# 2. Run the three baseline classifiers with 4-fold CV
python -m training.baseline --features cache/ckplus_baseline.npz \
    --out runs/baseline
```

`baseline.py` trains a tuned decision tree, the paper's *basic* MLP (256 → 128,
ReLU, 50 % dropout) and *optimized* MLP (512 → 256 → 128, GELU, BatchNorm,
residual connection, 30 % dropout), then prints a table against the paper's
published accuracies. Pass `--feature absolute` to classify raw peak positions
instead of displacements — the paper compares both feature sets.

Example run on the apex-only CK+ (327 sequences, 61-landmark preset):

```
Model               Test%   ±std   Paper%      Δ
decision_tree        81.4     5.8     80.0   +1.4
basic_nn             85.6     4.6     97.0  -11.3
optimized_nn         91.4     3.1     98.5   -7.0
```

The decision tree reproduces the paper closely. The neural nets land lower
because this redistribution of CK+ has 327 sequences (vs the paper's 593) and
uses MediaPipe landmarks rather than Dlib. The displacement feature beats raw
absolute positions (`--feature absolute` scores ~7 points lower), confirming
the paper's main comparison.

## Dataset access

The paper benchmarks on CK+, Oulu-CASIA, and MMI. All three are gated — you
must sign an academic-use form and wait for the maintainer to email a link:

| Dataset | Request page |
|---|---|
| CK+ | https://www.jeffcohn.net/Resources/ |
| Oulu-CASIA | https://www.oulu.fi/cmvs/node/41319 |
| MMI | https://mmifacedb.eu/ |

`fetch_ravdess.py` is a freely-licensed substitute (RAVDESS, CC BY-NC-SA 4.0)
so you can train end-to-end without waiting for access. Each actor zip is
~553 MB; interrupted downloads resume automatically.

## Expected dataset layout

`prepare_data.py` expects one folder per class, each holding one sub-folder per
sequence of chronologically-ordered frames:

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

Both `fetch_ravdess.py` and `organize_ckplus.py` produce exactly this layout.
