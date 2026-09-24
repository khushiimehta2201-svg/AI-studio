"""Provider-neutral local AI adapters.

The rest of the application depends only on TextModel/VisionModel behavior. Today
Ollama is the implementation; a different provider/model can be added behind the
same interfaces without changing planning, targeting, rendering, or the portal.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests


@dataclass(frozen=True)
class ModelSpec:
    role: str
    provider: str
    model: str
    url: str
    timeout: int
    num_ctx: int
    temperature: float
    retries: int = 1


class TextModel:
    def generate_json(self, prompt: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError


class VisionModel:
    def generate_json(self, prompt: str, image_b64: str) -> Optional[dict[str, Any]]:
        raise NotImplementedError


class _TelemetryMixin:
    def __init__(self, spec: ModelSpec, telemetry_dir: Optional[str | Path] = None):
        self.spec = spec
        self.telemetry_dir = Path(telemetry_dir) if telemetry_dir else None

    def _log(self, response: Optional[dict[str, Any]], latency_ms: int, error: Optional[str] = None) -> None:
        if not self.telemetry_dir:
            return
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.time(),
            "role": self.spec.role,
            "provider": self.spec.provider,
            "model": self.spec.model,
            "latency_ms": latency_ms,
            "prompt_tokens": (response or {}).get("prompt_eval_count"),
            "output_tokens": (response or {}).get("eval_count"),
            "total_tokens": (response or {}).get("prompt_eval_count", 0) + (response or {}).get("eval_count", 0)
            if response else None,
            "load_duration_ns": (response or {}).get("load_duration"),
            "error": error,
        }
        with (self.telemetry_dir / "ai_usage.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class OllamaTextClient(_TelemetryMixin, TextModel):
    def generate_json(self, prompt: str) -> Optional[dict[str, Any]]:
        payload = {
            "model": self.spec.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.spec.temperature, "num_ctx": self.spec.num_ctx},
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.spec.retries + 1):
            started = time.perf_counter()
            try:
                res = requests.post(self.spec.url, json=payload, timeout=self.spec.timeout)
                latency = int((time.perf_counter() - started) * 1000)
                res.raise_for_status()
                body = res.json()
                self._log(body if isinstance(body, dict) else None, latency)
                raw = body.get("response", "") if isinstance(body, dict) else ""
                data = _parse_json(raw)
                if data is None:
                    raise ValueError("Model returned non-JSON output")
                return data
            except Exception as exc:
                last_error = exc
                latency = int((time.perf_counter() - started) * 1000)
                self._log(None, latency, str(exc))
                if attempt >= self.spec.retries:
                    raise
        raise RuntimeError(str(last_error or "model failure"))


class OllamaVisionClient(_TelemetryMixin, VisionModel):
    def generate_json(self, prompt: str, image_b64: str) -> Optional[dict[str, Any]]:
        payload = {
            "model": self.spec.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.spec.temperature, "num_ctx": self.spec.num_ctx},
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.spec.retries + 1):
            started = time.perf_counter()
            try:
                res = requests.post(self.spec.url, json=payload, timeout=self.spec.timeout)
                latency = int((time.perf_counter() - started) * 1000)
                res.raise_for_status()
                body = res.json()
                self._log(body if isinstance(body, dict) else None, latency)
                raw = body.get("response", "") if isinstance(body, dict) else ""
                data = _parse_json(raw)
                if data is None:
                    raise ValueError("Vision model returned non-JSON output")
                return data
            except Exception as exc:
                last_error = exc
                latency = int((time.perf_counter() - started) * 1000)
                self._log(None, latency, str(exc))
                if attempt >= self.spec.retries:
                    raise
        raise RuntimeError(str(last_error or "vision failure"))


def _normalise_url(host: str) -> str:
    value = (host or "").strip().rstrip("/")
    if value.endswith("/api/generate"):
        return value
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value + "/api/generate"


def _parse_json(raw: Any) -> Optional[dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start:end + 1])
                return value if isinstance(value, dict) else None
            except Exception:
                return None
    return None


def create_clients(telemetry_dir: Optional[str | Path] = None) -> tuple[TextModel, VisionModel]:
    provider = os.getenv("AI_PROVIDER", "ollama").strip().lower()
    if provider != "ollama":
        raise ValueError("Only Ollama is implemented in this local build; add another provider behind this adapter interface.")
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11435")
    common = os.getenv("OLLAMA_URL", "").strip()
    text_url = os.getenv("OLLAMA_TEXT_URL", common) or _normalise_url(host)
    vision_url = os.getenv("OLLAMA_VISION_URL", common) or _normalise_url(host)
    ctx = max(1024, int(os.getenv("AI_NUM_CTX", "4096")))
    text = ModelSpec(
        "text_reasoning", "ollama",
        os.getenv("AI_TEXT_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2")).strip() or "llama3.2",
        text_url, max(5, int(os.getenv("AI_TEXT_TIMEOUT", "30"))), ctx, 0.1,
        max(0, int(os.getenv("AI_TEXT_RETRIES", "1"))),
    )
    vision = ModelSpec(
        "vision_grounding", "ollama",
        os.getenv("AI_VISION_MODEL", os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")).strip() or "qwen2.5vl:7b",
        vision_url, max(10, int(os.getenv("AI_VISION_TIMEOUT", "45"))), ctx, 0.0,
        max(0, int(os.getenv("AI_VISION_RETRIES", "1"))),
    )
    return OllamaTextClient(text, telemetry_dir), OllamaVisionClient(vision, telemetry_dir)
