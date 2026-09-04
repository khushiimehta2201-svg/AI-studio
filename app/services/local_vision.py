import base64
import json
import os
from typing import Any, Dict, Optional
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
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


def analyze_image(image_path: Optional[str], action_text: str) -> Dict[str, Any]:
    # 1. Null-safe check: return immediately if no image path exists (e.g. text-only slides)
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
Identify the bounding box and center click coordinates of the UI control for the action: "{action_text}".
Coordinates must be normalized from 0.0 to 1.0 relative to image width and height.

Return ONLY JSON:
{{
  "found": true,
  "target_name": "button name",
  "bounding_box": [ymin, xmin, ymax, xmax],
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
        raw = body.get("response", "")
        data = json.loads(raw)

        if not data.get("found"):
            return _empty_result()

        pt = data.get("click_point")
        if isinstance(pt, (list, tuple)) and len(pt) == 2:
            x = max(0.0, min(1.0, float(pt[0])))
            y = max(0.0, min(1.0, float(pt[1])))
            return {
                "found": True,
                "target_name": data.get("target_name", ""),
                "click_point": [x, y],
                "bounding_box": data.get("bounding_box"),
                "confidence": float(data.get("confidence", 0.7)),
            }
    except Exception as e:
        print(f"[VISION] Qwen inference skipped: {e}")

    return _empty_result()
