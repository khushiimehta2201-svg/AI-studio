from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def _ollama_generate_url() -> str:
    explicit = os.getenv("OLLAMA_VISION_URL", "").strip().rstrip("/")
    if explicit:
        return explicit
    host = os.getenv("OLLAMA_HOST", "").strip().rstrip("/")
    if host:
        if host.endswith("/api/generate"):
            return host
        if not host.startswith(("http://", "https://")):
            host = "http://" + host
        return host + "/api/generate"
    base = os.getenv("OLLAMA_URL", "").strip().rstrip("/")
    if base:
        if base.endswith("/api/generate"):
            return base
        return base + "/api/generate"
    return "http://127.0.0.1:11435/api/generate"


OLLAMA_URL = _ollama_generate_url()
VISION_MODEL = os.getenv(
    "VISION_MODEL",
    os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b"),
).strip() or "qwen2.5vl:7b"
VISION_TIMEOUT = max(10, int(os.getenv("VISION_TIMEOUT", "60")))
VISION_MIN_CONFIDENCE = float(os.getenv("VISION_MIN_CONFIDENCE", "0.82"))


def _normalise_point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except Exception:
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return [x, y]


def _normalise_box(value):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x0, y0, x1, y1 = [float(v) for v in value[:4]]
    except Exception:
        return None
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        return None
    return [x0, y0, x1, y1]


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start:end + 1])
            return value if isinstance(value, dict) else {}
        except Exception:
            pass
    return {}


def _ocr_fallback(
    query: str,
    vision_words: Optional[List[Dict[str, Any]]],
    candidates: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    candidates = candidates or []
    query_low = (query or "").lower().strip()

    exact = [
        candidate for candidate in candidates
        if str(candidate.get("match_type", "")).lower() == "exact"
        and candidate.get("point") is not None
    ]
    if len(exact) == 1:
        candidate = exact[0]
        return {
            "found": True,
            "confidence": 0.92,
            "click_point": candidate.get("point"),
            "source": "ocr",
            "target_name": candidate.get("matched", query),
            "bounding_box": candidate.get("box"),
            "annotation": None,
        }

    query_tokens = [
        token.lower() for token in (query or "").split() if token.strip()
    ]
    if query_tokens and vision_words:
        matches = []
        for word in vision_words:
            value = str(word.get("text", "")).lower()
            if value and any(token in value for token in query_tokens):
                matches.append(word)
        if len(matches) == 1:
            word = matches[0]
            try:
                x = (float(word["x0"]) + float(word["x1"])) / 2.0
                y = (float(word["y0"]) + float(word["y1"])) / 2.0
            except Exception:
                return {"found": False, "confidence": 0.0, "click_point": None, "source": "none", "target_name": query, "bounding_box": None, "annotation": None}
            return {
                "found": True,
                "confidence": 0.72,
                "click_point": [x, y],
                "source": "ocr_word",
                "target_name": str(word.get("text", query)),
                "bounding_box": [word.get("x0"), word.get("y0"), word.get("x1"), word.get("y1")],
                "annotation": None,
            }

    return {
        "found": False,
        "confidence": 0.0,
        "click_point": None,
        "source": "none",
        "target_name": query,
        "bounding_box": None,
        "annotation": None,
    }


def analyze_image(
    image_path: str,
    query: str,
    vision_words: Optional[List[Dict[str, Any]]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    target_type: str = "",
    action: str = "",
) -> Dict[str, Any]:
    """Locate a UI target with local vision; return no point when uncertain."""
    image_path = Path(image_path)
    if not image_path.exists():
        return _ocr_fallback(query, vision_words, candidates)

    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        max_side = 1600
        if max(image.size) > max_side:
            scale = max_side / float(max(image.size))
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=88, optimize=True)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"[VISION] Could not read image: {exc}")
        return _ocr_fallback(query, vision_words, candidates)

    prompt = f"""
Locate the exact software UI control described below in this screenshot.

TARGET: {query}
TARGET TYPE: {target_type or "unknown"}
ACTION: {action or ""}

Return JSON only:
{{
  "found": true,
  "confidence": 0.0,
  "click_point": [0.0, 0.0],
  "target_name": "exact visible label when available",
  "bounding_box": [0.0, 0.0, 0.0, 0.0],
  "annotation": "brief reason"
}}

Rules:
- Coordinates are normalized 0..1 relative to the image.
- The click point must be inside the bounding box.
- Prefer the visible UI control matching the TARGET, not nearby text with a similar word.
- If there are multiple plausible matches or the control is not clearly visible, set found=false.
- Do not guess or invent a coordinate.
- Confidence should reflect certainty, not optimism.
""".strip()

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=VISION_TIMEOUT)
        response.raise_for_status()
        body = response.json()
        result = _extract_json(body.get("response", "") if isinstance(body, dict) else "")
        point = _normalise_point(result.get("click_point"))
        box = _normalise_box(result.get("bounding_box"))
        confidence = float(result.get("confidence", 0.0))
        found = bool(result.get("found")) and point is not None and confidence >= VISION_MIN_CONFIDENCE
        if found and box is not None:
            found = box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
        if found:
            return {
                "found": True,
                "confidence": confidence,
                "click_point": point,
                "source": "vision",
                "target_name": result.get("target_name") or query,
                "bounding_box": box,
                "annotation": result.get("annotation"),
            }
    except Exception as exc:
        print(f"[VISION] Ollama vision unavailable for '{query}': {exc}")

    return _ocr_fallback(query, vision_words, candidates)
