import base64
import json
import os
from typing import Any, Dict, Optional

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
VISION_TIMEOUT = int(os.getenv("OLLAMA_VISION_TIMEOUT", "30"))

# A bounding box wider/taller than this fraction of the image is treated as
# "the model pointed at the whole screen", not a specific control -- this is
# the single biggest source of wrong-looking cursors from VLM grounding, so
# we reject it outright rather than trusting a returned click_point blindly.
MAX_BOX_FRACTION = 0.55
MIN_BOX_FRACTION = 0.0002


def _empty_result(confidence: float = 0.0) -> Dict[str, Any]:
    return {"found": False, "target_name": "", "bounding_box": None, "click_point": None, "confidence": confidence}


def analyze_image(image_path: Optional[str], action_text: str) -> Dict[str, Any]:
    if not image_path or not isinstance(image_path, (str, os.PathLike)) or not os.path.exists(str(image_path)):
        return _empty_result()

    try:
        with open(str(image_path), "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[VISION] Failed to read image '{image_path}': {e}")
        return _empty_result()

    prompt = f"""
You are locating a SINGLE specific interactive UI control (button, field, menu
item, checkbox, link, icon or tab) in a software screenshot for the action:
"{action_text}"

Rules:
- Pick the smallest element that satisfies the action, never a whole panel,
  toolbar, or the entire window.
- If you cannot find a control that clearly matches, set "found" to false.
- All coordinates are normalized 0.0-1.0 relative to image width/height,
  origin top-left.

Return ONLY JSON, no prose:
{{
  "found": true,
  "target_name": "short name of the control",
  "bounding_box": [x0, y0, x1, y1],
  "confidence": 0.0
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
        raw = res.json().get("response", "")
        data = json.loads(raw)

        if not data.get("found"):
            return _empty_result()

        box = data.get("bounding_box")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            return _empty_result()

        x0, y0, x1, y1 = (max(0.0, min(1.0, float(v))) for v in box)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area < MIN_BOX_FRACTION:
            return _empty_result()
        if area > MAX_BOX_FRACTION:
            # The model likely grounded the whole screen/panel rather than a
            # specific control -- discard instead of drawing a misleading
            # cursor in the middle of everything.
            return _empty_result()

        confidence = float(data.get("confidence", 0.6) or 0.6)
        # Penalize confidence for large-ish boxes even under the hard cutoff,
        # since bigger boxes are inherently less precise about "the" point.
        confidence *= max(0.4, 1.0 - area)

        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        return {
            "found": True,
            "target_name": str(data.get("target_name", "") or ""),
            "bounding_box": [x0, y0, x1, y1],
            "click_point": [cx, cy],
            "confidence": round(confidence, 3),
        }
    except Exception as e:
        print(f"[VISION] Qwen inference skipped: {e}")

    return _empty_result()
