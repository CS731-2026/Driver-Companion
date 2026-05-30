# llm_client.py
import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()  # loads OPENROUTER_API_KEY from .env

ONLINE_MODELS = {
    "claude-haiku": "anthropic/claude-haiku-4-5",
    "gpt-4o-mini":  "openai/gpt-4o-mini",
    "llama-70b":    "meta-llama/llama-3.3-70b-instruct:free",
    "gemma-26b":    "google/gemma-4-26b-a4b-it:free",
}

LOCAL_MODELS = {
    "gemma-local": "gemma3:4b",
    "llama-local":  "llama3",
}

DEFAULT_ONLINE_MODEL = "claude-haiku"
DEFAULT_LOCAL_MODEL  = "gemma-local"

OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"
OLLAMA_BASE     = "http://localhost:11434/api/chat"


def call_online(messages: List[Dict], model_key: str = DEFAULT_ONLINE_MODEL) -> str:
    api_key = os.environ["OPENROUTER_API_KEY"]
    model_id = ONLINE_MODELS.get(model_key, ONLINE_MODELS[DEFAULT_ONLINE_MODEL])

    resp = requests.post(
        OPENROUTER_BASE,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.7,
        },
        timeout=20
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def call_local(messages: List[Dict], model_key: str = DEFAULT_LOCAL_MODEL) -> str:
    model_name = LOCAL_MODELS.get(model_key, LOCAL_MODELS[DEFAULT_LOCAL_MODEL])

    resp = requests.post(
        OLLAMA_BASE,
        json={
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": 150, "temperature": 0.7}
        },
        timeout=120
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def warmup_local(model_key: str = DEFAULT_LOCAL_MODEL) -> bool:
    model_name = LOCAL_MODELS.get(model_key, LOCAL_MODELS[DEFAULT_LOCAL_MODEL])
    try:
        resp = requests.post(
            OLLAMA_BASE,
            json={"model": model_name, "messages": [{"role": "user", "content": "hi"}],
                  "stream": False, "options": {"num_predict": 1}},
            timeout=120,
        )
        return resp.ok
    except Exception:
        return False


def call_llm(
    messages: List[Dict],
    model_key: Optional[str] = None,
    mode: str = "online"
) -> tuple[str, str]:
    if mode == "offline":
        key = model_key or DEFAULT_LOCAL_MODEL
        return call_local(messages, key), key

    if mode == "auto":
        try:
            key = model_key or DEFAULT_ONLINE_MODEL
            return call_online(messages, key), key
        except Exception as e:
            print(f"[LLM] Online失败，切换本地: {e}")
            key = DEFAULT_LOCAL_MODEL
            return call_local(messages, key), key

    key = model_key or DEFAULT_ONLINE_MODEL
    return call_online(messages, key), key
