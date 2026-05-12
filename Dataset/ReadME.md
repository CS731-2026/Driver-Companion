# FaceMesh Dataset Conversion Tool

A Python utility to convert facial emotion datasets (AffectNet and RAF-DB) to MediaPipe FaceMesh landmarks format. This tool extracts 478 facial landmarks from each image and saves them as compressed `.npz` files for efficient training.

---

## Overview

This script processes emotion recognition datasets by:
- Downloading or loading datasets from Hugging Face
- Extracting facial landmarks using MediaPipe's FaceLandmarker
- Filtering invalid images (too small, too blurry, no face detected, etc.)
- Saving landmarks and labels as compressed NumPy arrays
- Generating metadata for downstream training tasks

---

## Requirements

### System Dependencies
- Python 3.X
- OpenCV 

### Python Packages
Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key packages:
- `numpy` - Array operations
- `pillow` - Image handling
- `opencv-python` - Image processing and preview
- `tqdm` - Progress bars
- `mediapipe` - Face landmark detection
- `datasets` - Hugging Face dataset loading

### MediaPipe Model
Download the face landmarker model:
- File: `face_landmarker.task`
- Place in the same directory as `Dataset.py`
- [Download from MediaPipe Google Ai Webpage](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/index#models)

---

## Usage

### Basic Usage

Convert all available datasets (AffectNet + RAF-DB):
```bash
python Dataset.py
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | (all) | Comma-separated subset: `affectnet`, `rafdb`, or both |
| `--output` | `./facemesh_dataset` | Output directory for `.npz` files |
| `--model` | `face_landmarker.task` | Path to MediaPipe model file |
| `--preview` | (off) | Show live preview window during processing |
| `--preview-scale` | `1.0` | Scale factor for preview window (e.g., `1.5` for 50% larger) |

### Examples

**Convert only AffectNet dataset:**
```bash
python Dataset.py --datasets affectnet
```

**Convert RAF-DB only to custom output directory:**
```bash
python Dataset.py --datasets rafdb --output ./my_output
```

**Enable live preview with 1.5x scaling:**
```bash
python Dataset.py --preview --preview-scale 1.5
```

**Use custom model path:**
```bash
python Dataset.py --model ./models/my_face_landmarker.task
```

**Convert all datasets to custom output with preview:**
```bash
python Dataset.py --output ./dataset_output --preview
```

---

## Output Format

### Directory Structure
```
facemesh_dataset/
├── metadata.json
├── affectnet_train.npz
├── affectnet_validation.npz
└── rafdb_train.npz
```

### NPZ File Contents

Each `.npz` file contains:
- **`landmarks`**: Shape `(N, 478, 3)` - 3D coordinates (x, y, z) for 478 facial landmarks
- **`labels`**: Shape `(N,)` - Emotion class indices (0-6)
- **`label_names`**: List of emotion names (e.g., `['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']`)

---

## License & Attribution

- **MediaPipe**: Google LLC
- **AffectNet**: [affectnet.org](http://affectnet.org/)
- **RAF-DB**: RAF Database

Ensure you have appropriate permissions before using these datasets.

---

