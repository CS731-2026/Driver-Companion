# Emotion Recognition (Köksal & Gumus 2025)

Reference implementation of *"Deep Learning-Based Real-Time Sequential Facial
Expression Analysis Using Geometric Features"* (arXiv:2512.05669v1).

Implemented in **PyTorch**.

Pipeline:
1. MediaPipe FaceMesh (Tasks API) → 478 landmarks per frame.
2. Select 61 landmarks across 5 FACS-based groups (cat1..cat5).
3. For 5 key frames (neutral → apex), compute Euclidean distance and angle for
   every same-group landmark pair, then subtract previous-frame features.
   Output: `(N-1, A) = (4, A)` tensor per sequence.
4. StandardScaler → ConvLSTM1D(filters=8, kernel=1) → Flatten → MLP
   (2048 → 1024 → num_classes).
5. Trained with Adam + cross-entropy loss, stratified 5-fold CV.

PyTorch has no built-in ConvLSTM, so `training/model.py` implements a
ConvLSTM1D cell (standard ConvLSTM, no peepholes — matching the Keras
`ConvLSTM1D` the paper actually trained).

## Layout

```
emotion-reco/
├── dataset/         # landmark detection, feature creation, sequence loader
│   ├── landmarks.py        # FaceMesh wrapper, 61 selected indices, 5 FACS groups
│   ├── features.py         # distance + angle features, frame differencing
│   ├── dataset.py          # CK+-style folder walker, 5-keyframe sampler
│   ├── prepare_data.py     # raw frames → feature cache (.npz)
│   ├── fetch_ravdess.py    # download RAVDESS, frame-extract to CK+ layout
│   ├── organize_ckplus.py  # rearrange an obtained CK+ archive
│   └── make_synthetic.py   # fake feature cache for pipeline smoke tests
├── training/        # ConvLSTM1D model + 5-fold CV training
└── implementation/  # real-time webcam inference with 5-frame buffer
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Get a dataset — pick one:
#    a) RAVDESS (free, downloads from Zenodo, ~1.6 GB for actors 1-3)
python -m dataset.fetch_ravdess --out_root data/ravdess --actors 1-3
#    b) CK+ (after you obtain access from the Cohn lab)
python -m dataset.organize_ckplus --ckplus_root /path/to/ckplus \
    --out_root data/ckplus
#    c) Synthetic (no real data — for pipeline smoke tests)
python -m dataset.make_synthetic --output cache/synth.npz --sequences_per_class 60

# 2. Preprocess raw sequence folders into a feature cache (.npz).
#    Skip this step for the synthetic path (it already writes the cache).
python -m dataset.prepare_data \
    --data_root data/ravdess \
    --output    cache/ravdess.npz

# 3. Train with 5-fold cross-validation
python -m training.train --features cache/ravdess.npz --out runs/ravdess

# 4. Run real-time inference
python -m implementation.realtime --model runs/ravdess/fold_best.pt \
    --scaler runs/ravdess/scaler.pkl --labels runs/ravdess/labels.json
```

Training artifacts land in `runs/<name>/`: `fold_best.pt` (PyTorch checkpoint —
state dict + arch dims + labels), `scaler.pkl`, `labels.json`, `metrics.json`,
`training_curves.png`.

Run all commands from inside `emotion-reco/` so the `dataset` / `training` /
`implementation` packages import correctly.

## Dataset access

The paper benchmarks on CK+, Oulu-CASIA, and MMI. All three are gated — you
must sign an academic-use form and wait for the maintainer to email a link:

| Dataset | Request page |
|---|---|
| CK+ | https://www.jeffcohn.net/Resources/ |
| Oulu-CASIA | https://www.oulu.fi/cmvs/node/41319 |
| MMI | https://mmifacedb.eu/ |

`fetch_ravdess.py` is a freely-licensed substitute (RAVDESS, CC BY-NC-SA 4.0)
so you can train end-to-end without waiting for access.

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
