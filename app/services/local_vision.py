import base64
import json
import os
from typing import Any, Dict, Optional
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11435").rstrip("/")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"{OLLAMA_HOST}/api/generate")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
VISION_TIMEOUT = int(os.getenv("OLLAMA_VISION_TIMEOUT", "30"))


def _empty_result(confidence: float = 0.0) -> Dict[str, Any]:
    return {
        "found": False,
        "target_name": "",
        "bounding_box": None,
        "click_point": None,
        "confidence": confidence,
    }


def _normalize_coordinate(val: Any) -> float:
    try:
        v = float(val)
        if v > 100.0:
            v = v / 1000.0
        elif v > 1.0:
            v = v / 100.0
        return max(0.06, min(0.92, v))
    except Exception:
        return 0.50


def analyze_image(image_path: Optional[str], action_text: str) -> Dict[str, Any]:
    if not image_path or not isinstance(image_path, (str, os.PathLike)):
        return _empty_result()

    image_str = str(image_path)
    if not os.path.exists(image_str):
        return _empty_result()

    try:
        with open(image_str, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[VISION] Failed to read image '{image_str}': {e}")
        return _empty_result()

    prompt = f"""
Locate the ONE ACTUAL INTERACTIVE UI COMPONENT (e.g., button, search bar, dropdown, input text field) for the action: "{action_text}".

CRITICAL INSTRUCTIONS:
1. If the screenshot contains a red/orange numbered callout circle (1, 2, 3), DO NOT target the circle or the number itself!
2. Target the actual UI button or field indicated by or adjacent to the callout badge.
3. Return click_point as [x, y] in range 0 to 1000 relative to image width and height.

Return ONLY valid JSON:
{{
  "found": true,
  "target_name": "name of UI component",
  "click_point": [x, y],
  "confidence": 0.85
}}
"""

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=VISION_TIMEOUT)
        res.raise_for_status()
        body = res.json()
        raw_text = body.get("response", "")
        data = json.loads(raw_text)

        if not data.get("found"):
            return _empty_result()

        pt = data.get("click_point")
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            norm_x = _normalize_coordinate(pt[0])
            norm_y = _normalize_coordinate(pt[1])
            return {
                "found": True,
                "target_name": str(data.get("target_name", "")).strip(),
                "click_point": [norm_x, norm_y],
                "bounding_box": data.get("bounding_box"),
                "confidence": float(data.get("confidence", 0.75)),
            }
    except Exception as e:
        print(f"[VISION] Qwen-VL notice: {e}")

    return _empty_result()
