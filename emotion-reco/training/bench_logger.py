"""
bench_logger.py
================
Shared benchmarking logger used by every training script under TrainVla/.

Each training run gets its own log file that records:
  * Accuracy (per-epoch train/val and final test)
  * Process delay (per-epoch wall time, per-sample inference latency, FPS)
  * Hardware utilization (CPU %, RAM GB, GPU %, GPU memory GB)

Two outputs per run:
  <output_dir>/<method>_log.txt      human-readable line-by-line log
  <output_dir>/<method>_metrics.json structured metrics for plots/comparison

The logger is intentionally framework-agnostic: it works for any PyTorch model
and for the image / landmark / blendshape / geometric pipelines side-by-side.

CPU + RAM telemetry uses psutil. GPU telemetry uses the `nvitop` library
(the Python equivalent of `nvtop`, `pip install nvitop`) — same data nvtop
shows in its TUI, exposed through a clean Python API.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from nvitop import Device as _NvDevice
    _HAS_NVITOP = True
except ImportError:
    _NvDevice = None
    _HAS_NVITOP = False


# ─────────────────────────────────────────────────────────────────────────────
# Hardware probes
# ─────────────────────────────────────────────────────────────────────────────

def _gpu_index() -> int | None:
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return None


_NVITOP_CACHE: dict[int, Any] = {}


def _nvitop_device(idx: int):
    """Cache one nvitop Device per GPU index — querying it has overhead."""
    if not _HAS_NVITOP:
        return None
    if idx not in _NVITOP_CACHE:
        _NVITOP_CACHE[idx] = _NvDevice(idx)
    return _NVITOP_CACHE[idx]


def sample_hw() -> dict[str, float]:
    """One-shot hardware snapshot: CPU %, RAM GB, GPU %, GPU mem GB."""
    out: dict[str, float] = {}

    if _HAS_PSUTIL:
        out["cpu_pct"] = float(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        out["ram_used_gb"] = round(vm.used / 1024**3, 3)
        out["ram_pct"] = float(vm.percent)
    else:
        out["cpu_pct"] = -1.0
        out["ram_used_gb"] = -1.0
        out["ram_pct"] = -1.0

    idx = _gpu_index()
    dev = _nvitop_device(idx) if idx is not None else None
    if dev is not None:
        # nvitop returns NaN for unsupported sensors; coerce to a float
        gpu_util = dev.gpu_utilization()
        mem_used = dev.memory_used()
        mem_total = dev.memory_total()
        out["gpu_pct"] = float(gpu_util) if gpu_util is not None else -1.0
        out["gpu_mem_used_gb"] = round(float(mem_used) / 1024**3, 3) if mem_used else 0.0
        out["gpu_mem_total_gb"] = round(float(mem_total) / 1024**3, 3) if mem_total else 0.0
        # Bonus metrics nvtop also shows
        try:
            out["gpu_mem_pct"] = float(dev.memory_utilization())
        except Exception:
            pass
        try:
            out["gpu_temp_c"] = float(dev.temperature())
        except Exception:
            pass
        try:
            out["gpu_power_w"] = round(float(dev.power_usage()) / 1000.0, 1)
        except Exception:
            pass
    else:
        out["gpu_pct"] = 0.0
        out["gpu_mem_used_gb"] = 0.0
        out["gpu_mem_total_gb"] = 0.0

    return out


def hw_static() -> dict[str, Any]:
    """Static hardware description recorded once per run."""
    info: dict[str, Any] = {}
    if _HAS_PSUTIL:
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    idx = _gpu_index()
    if idx is not None:
        dev = _nvitop_device(idx)
        if dev is not None:
            info["gpu_name"] = dev.name()
            info["gpu_count"] = len(_NvDevice.all())
            try:
                info["gpu_driver_version"] = dev.driver_version()
            except Exception:
                pass
        else:
            info["gpu_name"] = "unknown_gpu_nvitop_missing"
            info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda
    else:
        info["gpu_name"] = "cpu_only"
    info["torch_version"] = torch.__version__
    info["nvitop_available"] = _HAS_NVITOP
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float | None
    val_acc: float | None
    epoch_seconds: float
    hw: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchRecord:
    method: str
    output_dir: str
    started_at: float = field(default_factory=time.time)
    hardware: dict[str, Any] = field(default_factory=hw_static)
    config: dict[str, Any] = field(default_factory=dict)

    num_params: int = 0
    num_params_million: float = 0.0

    epochs: list[EpochRecord] = field(default_factory=list)

    # Final inference benchmark (single sample, taken on best checkpoint)
    inference_latency_ms_mean: float = 0.0
    inference_latency_ms_p50: float = 0.0
    inference_latency_ms_p95: float = 0.0
    inference_fps: float = 0.0

    # Final metrics on test (or val) split
    final_accuracy: float = 0.0
    final_weighted_f1: float = 0.0
    best_val_accuracy: float = 0.0
    total_train_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["epochs"] = [asdict(e) for e in self.epochs]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# BenchLogger
# ─────────────────────────────────────────────────────────────────────────────

class BenchLogger:
    """
    One instance per training run. Wraps a Python logger that writes both to
    stdout and to <output_dir>/<method>_log.txt, plus accumulates structured
    metrics in <output_dir>/<method>_metrics.json.
    """

    def __init__(self, method: str, output_dir: str | Path, config: dict | None = None):
        self.method = method
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.record = BenchRecord(
            method=method,
            output_dir=str(self.output_dir),
            config=config or {},
        )

        # ── file + console handlers ─────────────────────────────────────────
        log_path = self.output_dir / f"{method}_log.txt"
        self.metrics_path = self.output_dir / f"{method}_metrics.json"

        self.logger = logging.getLogger(f"bench.{method}")
        self.logger.setLevel(logging.INFO)
        # Reset handlers so re-runs don't duplicate
        for h in list(self.logger.handlers):
            self.logger.removeHandler(h)
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S"
        )
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.propagate = False

        self.info("═══ Benchmark started: %s ═══", method)
        self.info("Output dir: %s", self.output_dir.resolve())
        self.info("Hardware: %s", json.dumps(self.record.hardware))

        self._epoch_start: float | None = None

    # passthrough conveniences
    def info(self, msg, *args):
        self.logger.info(msg, *args)

    def warning(self, msg, *args):
        self.logger.warning(msg, *args)

    # ── config / model recording ────────────────────────────────────────────

    def log_model(self, model: torch.nn.Module):
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.record.num_params = n
        self.record.num_params_million = round(n / 1e6, 4)
        self.info("Model parameters: %s  (%.3f M)", f"{n:,}", n / 1e6)

    def log_config(self, **kwargs):
        self.record.config.update(kwargs)
        self.info("Config: %s", json.dumps(kwargs, default=str))

    # ── epoch lifecycle ─────────────────────────────────────────────────────

    def epoch_begin(self, epoch: int):
        self._epoch_start = time.time()
        self.info("Epoch %d started", epoch)

    def epoch_end(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float | None = None,
        val_acc: float | None = None,
        lr: float | None = None,
    ):
        secs = time.time() - (self._epoch_start or time.time())
        hw = sample_hw()
        rec = EpochRecord(
            epoch=epoch,
            train_loss=float(train_loss),
            train_acc=float(train_acc),
            val_loss=float(val_loss) if val_loss is not None else None,
            val_acc=float(val_acc) if val_acc is not None else None,
            epoch_seconds=round(secs, 3),
            hw=hw,
        )
        self.record.epochs.append(rec)

        lr_str = f"  lr={lr:.2e}" if lr is not None else ""
        val_str = (
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            if val_acc is not None
            else "  (no val)"
        )
        self.info(
            "Epoch %3d done in %5.1fs  train_loss=%.4f  train_acc=%.4f%s%s  "
            "[cpu=%.0f%% ram=%.1fGB gpu=%.0f%% gpumem=%.2fGB]",
            epoch,
            secs,
            train_loss,
            train_acc,
            val_str,
            lr_str,
            hw["cpu_pct"],
            hw["ram_used_gb"],
            hw["gpu_pct"],
            hw["gpu_mem_used_gb"],
        )

    # ── inference benchmark ─────────────────────────────────────────────────

    @torch.no_grad()
    def benchmark_inference(
        self,
        model: torch.nn.Module,
        sample_inputs: tuple,
        runs: int = 200,
        warmup: int = 20,
    ):
        """
        Measure single-sample inference latency on `sample_inputs`.
        sample_inputs is a tuple of tensors, each already on the correct device,
        with batch dim = 1. Returned tensor is unused; we only time the forward.
        """
        device = next(model.parameters()).device
        model.eval()

        # Warmup
        for _ in range(warmup):
            _ = model(*sample_inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()

        latencies = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(*sample_inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

        latencies.sort()
        mean = sum(latencies) / len(latencies)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(0.95 * len(latencies))]
        fps = 1000.0 / mean if mean > 0 else 0.0

        self.record.inference_latency_ms_mean = round(mean, 3)
        self.record.inference_latency_ms_p50 = round(p50, 3)
        self.record.inference_latency_ms_p95 = round(p95, 3)
        self.record.inference_fps = round(fps, 2)

        self.info(
            "Inference latency over %d runs (warmup %d) — "
            "mean=%.2fms  p50=%.2fms  p95=%.2fms  → %.1f FPS",
            runs, warmup, mean, p50, p95, fps,
        )
        return mean

    # ── final results ───────────────────────────────────────────────────────

    def finalize(
        self,
        final_accuracy: float,
        final_weighted_f1: float,
        best_val_accuracy: float,
    ):
        self.record.final_accuracy = float(final_accuracy)
        self.record.final_weighted_f1 = float(final_weighted_f1)
        self.record.best_val_accuracy = float(best_val_accuracy)
        if self.record.epochs:
            self.record.total_train_seconds = round(
                sum(e.epoch_seconds for e in self.record.epochs), 2
            )
        self.metrics_path.write_text(json.dumps(self.record.to_dict(), indent=2))
        self.info(
            "Finished — final_acc=%.4f  f1=%.4f  best_val=%.4f  "
            "infer=%.2fms (%.1f FPS)  train_time=%ss",
            final_accuracy, final_weighted_f1, best_val_accuracy,
            self.record.inference_latency_ms_mean,
            self.record.inference_fps,
            self.record.total_train_seconds,
        )
        self.info("Metrics JSON: %s", self.metrics_path)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-method comparison helper (read the metrics JSONs back and tabulate)
# ─────────────────────────────────────────────────────────────────────────────

def compare(metrics_paths: list[str | Path]) -> str:
    """Render a markdown table comparing every method's JSON metrics file."""
    rows = []
    for p in metrics_paths:
        d = json.loads(Path(p).read_text())
        rows.append({
            "method": d["method"],
            "params_M": d["num_params_million"],
            "best_val_acc": d["best_val_accuracy"],
            "final_acc": d["final_accuracy"],
            "f1": d["final_weighted_f1"],
            "infer_ms": d["inference_latency_ms_mean"],
            "fps": d["inference_fps"],
            "train_sec": d["total_train_seconds"],
            "gpu_name": d["hardware"].get("gpu_name", "?"),
        })

    if not rows:
        return "(no metrics files)"

    headers = list(rows[0].keys())
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python bench_logger.py <metrics1.json> [<metrics2.json> ...]")
    else:
        print(compare(paths))
