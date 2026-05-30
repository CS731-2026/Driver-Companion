# CalmWheel — Driver Companion

**COMPSYS 731, Group 10** — Real-time in-car AI companion that detects driver
emotion via facial expression recognition (FER) and generates empathetic,
emotion-aware responses through an LLM.

The repo is split into four layers:

| Layer | Folder | Purpose |
|---|---|---|
| **App entry** | `run.py` (root) | Wires all modules, starts threads |
| **FER** | `emotionrec/` | Geo-122 inference, Kinect source |
| **LLM** | `llm/` | Client, prompt builder, conversation memory |
| **Main** | `main/` | CalmWheelLLM orchestrator, UI, STT |
| **Training** | `emotion-reco/` | Six FER model implementations, dataset prep |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         CalmWheel App                            │
│                                                                  │
│  emotionrec/              llm/                  main/            │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │realtime_fer │──▶│trigger_control   │──▶│ main.py          │  │
│  │(MediaPipe   │   │prompt_builder    │   │ (CalmWheelLLM)   │  │
│  │ 478 lm →   │   │emotion_strategy  │   │ → OpenRouter LLM │  │
│  │ Geo-122 MLP)│   │llm_client        │   │ → edge-tts TTS   │  │
│  └─────────────┘   │conversation      │   └──────────────────┘  │
│         │          └──────────────────┘           │             │
│  ┌──────▼────────────────────────────────────────▼───────────┐  │
│  │                    main/ui.py  (Tkinter 3-panel UI)        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                            ▲                                     │
│                       main/stt.py (Whisper STT)                  │
└──────────────────────────────────────────────────────────────────┘
                              ▲
               trained by  emotion-reco/
```

**FER model:** 478 MediaPipe landmarks → 3 228 pairwise geometric features
(distance + angle, Köksal & Gumus 2025) → GeoMLP.
**93.7 % validation accuracy on CK+**, 0.31 ms inference, 3 206 FPS.

**LLM backend:** OpenRouter — Claude Haiku 4.5 (default), GPT-4o Mini, Llama 3.3
70B, Gemma 4 26B. Switchable at runtime from the UI.

**TTS/STT:** Microsoft Edge TTS (`edge-tts`) for speech; Whisper `base` via
`faster-whisper` for microphone input.

---

## Quick start — App

### 1. Train the Geo-122 model

The FER checkpoint is not stored in git (14 MB). Run this once from inside
`emotion-reco/`:

```bash
cd emotion-reco
pip install -r requirements.txt

# Build static-image cache from CK+ (replace path)
python3 -m dataset.organize_ckplus_apex --ck_root /path/to/CK+ --out_root data/ckplus
python3 -m dataset.prepare_static_from_sequences \
    --data_root data/ckplus --output cache/ckplus_static --mode landmarks

# Train Geo-122
python3 -m training.train_geo_static \
    --data cache/ckplus_static --output runs/geo_static --landmarks 122
```

Checkpoint lands at `emotion-reco/runs/geo_static/checkpoints/best.pt`.

### 2. Install app dependencies

```bash
pip install -r requirements.txt
```

> First run downloads Whisper `base` (~150 MB); allow ~30 s.

### 3. Configure API keys

Edit `.env` at the project root:

```
OPENROUTER_API_KEY=your_key_here
```

### 4. Run

```bash
python run.py
```

---

## App CLI flags

```
python run.py [options]

  --checkpoint PATH    FER checkpoint  (default: emotion-reco/runs/geo_static/checkpoints/best.pt)
  --camera     INT     Camera index    (default: 0)
  --conf       FLOAT   Confidence threshold for label display (default: 0.3)
  --no-fer             Disable camera; use CLI simulation instead
  --no-bar             Hide probability bars on the camera feed
  --landmarker PATH    Path to face_landmarker.task (default: ./face_landmarker.task)
  --kinect             Use Kinect v2 with automatic RGB/IR switching
  --kinect-index INT   Kinect device index (default: 0)
```

---

## No camera? CLI mode

```bash
python run.py --no-fer
```

```
>> angry 前面那个车怎么开的！   # emotion + speech → LLM
>> sad                           # emotion only, no LLM
>> reset                         # clear conversation memory
>> mute                          # toggle TTS
>> quit
```

---

## Standalone FER test (no LLM)

```bash
python -m emotionrec.test_fer
python -m emotionrec.test_fer --checkpoint emotion-reco/runs/geo_static/checkpoints/best.pt
```

---

## LLM benchmark

```bash
python benchmark.py
# results → benchmark_results.json
```

---

## Repo layout

```
Driver-Companion/
├── run.py                       App entry point
├── benchmark.py                 Multi-LLM scenario benchmark
├── face_landmarker.task         MediaPipe face model (~3.6 MB)
├── gifs/                        Emotion avatar GIFs (7 emotions)
│
├── emotionrec/                  Emotion recognition layer
│   ├── realtime_fer.py          Geo-122 FER + MediaPipe camera loop
│   ├── test_fer.py              Standalone webcam test (no LLM)
│   └── kinect_source.py         Kinect v2 IR ↔ webcam hybrid source
│
├── llm/                         LLM layer
│   ├── interface.py             LLMInput / LLMOutput dataclasses
│   ├── llm_client.py            OpenRouter / Ollama client
│   ├── prompt_builder.py        LLM message assembly
│   ├── emotion_strategy.py      Per-emotion response strategies
│   ├── conversation.py          Multi-turn memory (last 6 turns)
│   └── trigger_control.py       Cooldown / intervention logic
│
├── main/                        Main / UI layer
│   ├── main.py                  CalmWheelLLM orchestrator + TTS
│   ├── ui.py                    Tkinter 3-panel UI + GIF avatar
│   └── stt.py                   Microphone VAD + Whisper STT
│
└── emotion-reco/                FER training code
    ├── dataset/                 Data prep scripts
    ├── training/                Six model trainers
    └── implementation/          Multi-method webcam inference
```

---

## LLM interface (programmatic)

```python
from llm.interface import LLMInput
from main.main import CalmWheelLLM

llm = CalmWheelLLM()
output = llm.process(LLMInput(emotion="angry", speech_text="Why is there so much traffic!"))

if output:
    print(output.response_text)
    print(output.emotion_timeline)
    # [{"t": 0, "emotion": "angry"}, {"t": 2500, "emotion": "neutral"}]

llm.new_trip()  # clear memory for a new session
```

---

## Kinect v2 support

Pass `--kinect` to switch between system webcam (bright conditions) and Kinect IR
(low-light / night driving) based on ambient luminance.

```bash
pip install pylibfreenect2
python run.py --kinect
```

---

## emotion-reco — Training layer

### Models

| # | Trainer | Modality | Paper | CK+ Accuracy |
|---|---|---|---|---:|
| 1 | `training/train.py` | sequential | Köksal & Gumus 2025 | 93 % |
| 2 | `training/baseline.py` | sequential | Juárez-Jiménez et al. 2025 | 98.5 % |
| 3 | `training/train_emonet.py` | static-image | DriveEmo-FL 2026 | 61.9 % |
| 4 | `training/train_blendshape.py` | static-image | Jakhete & Kulkarni 2024 | 87.3 % |
| **5** | **`training/train_geo_static.py`** | **static-image** | **Köksal & Gumus 2025** | **93.7 %** ← app |
| 6 | `training/train_image.py` | static-image | — (MobileNetV2) | — |

### End-to-end: compare all six models on CK+

```bash
cd emotion-reco

# Step 1 — get CK+ (one of)
python3 -m dataset.organize_ckplus_apex --ck_root /path/to/CK+ --out_root data/ckplus

# Step 2 — build caches
python3 -m dataset.prepare_static_from_sequences \
    --data_root data/ckplus --output cache/ckplus_static --mode both
python3 -m dataset.prepare_data \
    --data_root data/ckplus --output cache/ckplus.npz --landmarks 61
python3 -m dataset.prepare_baseline \
    --data_root data/ckplus --output cache/ckplus_baseline.npz --landmarks 61

# Step 3 — train and plot
python3 -m training.plot_all_models
# → runs/all_models_accuracy.png
```

### Results (CK+)

| Model | Val acc | Weighted F1 | Infer (ms) | FPS |
|---|---:|---:|---:|---:|
| EmoNet — CNN+MLP | 61.9 % | 0.593 | 0.77 | 1 307 |
| Blendshape MLP | 84.1 % | 0.836 | 0.38 | 2 625 |
| Blendshape Attention | 87.3 % | 0.868 | 0.45 | 2 213 |
| **Geo MLP (122 lm)** | **93.7 %** | **0.936** | **0.31** | **3 206** |

---

## Environment variables (`.env`)

| Key | Used by | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | `llm/llm_client.py` | Required for online LLM calls |
| `DEEPGRAM_API_KEY` | — | Alternative STT (not active by default) |
| `HF_TOKEN` | `emotion-reco/dataset/prepare_static.py` | HuggingFace dataset download |

---

## Troubleshooting

**LLM not responding**
Check `OPENROUTER_API_KEY` in `.env`, or test with:
```bash
python run.py --no-fer
# then type: neutral hello
```

**STT not picking up sound**
Edit `main/stt.py` and change the `sounddevice` device index at the top of
`listen_and_transcribe()`.

**FER checkpoint not found**
Train the Geo-122 model first (see Quick Start step 1) or point to a different
checkpoint:
```bash
python run.py --checkpoint emotion-reco/runs/geo_static/checkpoints/best.pt
```
