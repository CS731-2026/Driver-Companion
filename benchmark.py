# benchmark.py
# 对比多个LLM在不同情绪场景下的表现
# 对应PPT里的 Multi-LLM Comparison 部分

import time
import json
from datetime import datetime
from typing import List, Dict

from llm.prompt_builder import build_messages
from llm.llm_client import call_online, call_local, ONLINE_MODELS, LOCAL_MODELS


# ── 测试场景：覆盖所有7种情绪 + 有/无语音两种情况 ──────────
TEST_CASES = [
    {"emotion": "angry",    "speech": "前面那个车怎么开的！",     "trigger": "speech"},
    {"emotion": "angry",    "speech": None,                       "trigger": "emotion_intervention"},
    {"emotion": "sad",      "speech": "今天发生了很糟糕的事",     "trigger": "speech"},
    {"emotion": "sad",      "speech": None,                       "trigger": "emotion_intervention"},
    {"emotion": "fear",     "speech": "刚才那个弯太急了我很慌",   "trigger": "speech"},
    {"emotion": "fear",     "speech": None,                       "trigger": "emotion_intervention"},
    {"emotion": "happy",    "speech": "今天心情超好！",           "trigger": "speech"},
    {"emotion": "happy",    "speech": None,                       "trigger": "companionship"},
    {"emotion": "neutral",  "speech": "现在几点了？",             "trigger": "speech"},
    {"emotion": "neutral",  "speech": None,                       "trigger": "companionship"},
    {"emotion": "disgust",  "speech": "这条路修了多久了！",       "trigger": "speech"},
    {"emotion": "surprise", "speech": None,                       "trigger": "emotion_intervention"},
]

# ── 要测试的模型 ────────────────────────────────────────────
BENCHMARK_ONLINE_MODELS = list(ONLINE_MODELS.keys())
# 本地模型需要Ollama，默认不跑，设为True才跑
INCLUDE_LOCAL = False


def run_benchmark(
    save_to: str = "benchmark_results.json",
    models: List[str] = None,
    cases: List[Dict] = None,
) -> List[Dict]:
    """
    运行完整benchmark，返回结果列表并保存JSON。

    Args:
        save_to: 结果保存路径
        models:  要测试的模型key列表，默认测所有online模型
        cases:   测试场景，默认用内置TEST_CASES
    """
    models  = models or BENCHMARK_ONLINE_MODELS
    cases   = cases  or TEST_CASES
    results = []

    total = len(models) * len(cases)
    done  = 0

    print(f"\n{'='*50}")
    print(f"Benchmark 开始：{len(models)} 个模型 × {len(cases)} 个场景 = {total} 次调用")
    print(f"{'='*50}\n")

    for model_key in models:
        is_local = model_key in LOCAL_MODELS

        if is_local and not INCLUDE_LOCAL:
            print(f"[跳过] {model_key}（本地模型，INCLUDE_LOCAL=False）")
            continue

        for case in cases:
            done += 1
            print(f"[{done}/{total}] {model_key} | emotion={case['emotion']} | speech={'有' if case['speech'] else '无'}")

            messages = build_messages(
                emotion=case["emotion"],
                speech_text=case["speech"],
                trigger_reason=case["trigger"],
            )

            start = time.time()
            error = None
            response = None

            try:
                if is_local:
                    response = call_local(messages, model_key)
                else:
                    response = call_online(messages, model_key)
            except Exception as e:
                error = str(e)
                print(f"  ✗ 错误: {error}")

            latency_ms = (time.time() - start) * 1000

            result = {
                "model":        model_key,
                "emotion":      case["emotion"],
                "has_speech":   case["speech"] is not None,
                "speech_text":  case["speech"],
                "trigger":      case["trigger"],
                "response":     response,
                "latency_ms":   round(latency_ms, 1),
                "char_count":   len(response) if response else 0,
                "error":        error,
            }
            results.append(result)

            if response:
                print(f"  → {response[:60]}{'...' if len(response)>60 else ''}")
                print(f"  ⏱ {latency_ms:.0f}ms\n")

    # ── 保存结果 ──
    output = {
        "run_at":  datetime.now().isoformat(),
        "models":  models,
        "results": results,
        "summary": _summarize(results),
    }
    with open(save_to, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到 {save_to}")
    _print_summary(output["summary"])
    return results


def _summarize(results: List[Dict]) -> Dict:
    """计算每个模型的平均latency和成功率"""
    from collections import defaultdict
    stats = defaultdict(lambda: {"latencies": [], "errors": 0, "total": 0})

    for r in results:
        m = r["model"]
        stats[m]["total"] += 1
        if r["error"]:
            stats[m]["errors"] += 1
        else:
            stats[m]["latencies"].append(r["latency_ms"])

    summary = {}
    for model, s in stats.items():
        lats = s["latencies"]
        summary[model] = {
            "total":        s["total"],
            "success":      s["total"] - s["errors"],
            "error_count":  s["errors"],
            "avg_latency":  round(sum(lats) / len(lats), 1) if lats else None,
            "min_latency":  round(min(lats), 1) if lats else None,
            "max_latency":  round(max(lats), 1) if lats else None,
        }
    return summary


def _print_summary(summary: Dict):
    print(f"\n{'='*50}")
    print("Benchmark 总结")
    print(f"{'='*50}")
    print(f"{'模型':<20} {'成功':<6} {'平均延迟':>10} {'最快':>10} {'最慢':>10}")
    print("-" * 58)
    for model, s in summary.items():
        avg = f"{s['avg_latency']}ms" if s['avg_latency'] else "N/A"
        mn  = f"{s['min_latency']}ms" if s['min_latency'] else "N/A"
        mx  = f"{s['max_latency']}ms" if s['max_latency'] else "N/A"
        print(f"{model:<20} {s['success']}/{s['total']:<4} {avg:>10} {mn:>10} {mx:>10}")
