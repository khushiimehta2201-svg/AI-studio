import base64
import json
import os
import re
from typing import Any, Dict, Optional

import cv2
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
VISION_TIMEOUT = int(os.getenv("OLLAMA_VISION_TIMEOUT", "90"))

# A bounding box wider/taller than this fraction of the image is treated as
# "the model pointed at the whole screen", not a specific control -- this is
# one of the biggest sources of wrong-looking cursors from VLM grounding, so
# we reject it outright rather than trusting a returned click_point blindly.
MAX_BOX_FRACTION = 0.55
MIN_BOX_FRACTION = 0.0002

# How far outside [0, 1] a *normalized* coordinate is allowed to drift
# before we treat it as the model ignoring the coordinate convention
# entirely (Qwen-family models are known to sometimes report pixel
# coordinates in their own internal resized frame instead of 0.0-1.0,
# regardless of what the prompt asks for). Clamping such values into range
# used to silently collapse them into a degenerate corner box that then
# failed area checks on every call -- rejecting outright is safer and lets
# other grounding sources (OCR, annotations) take over instead.
OUT_OF_RANGE_TOLERANCE = 0.08


def _empty_result(confidence: float = 0.0) -> Dict[str, Any]:
    return {"found": False, "target_name": "", "bounding_box": None, "click_point": None, "confidence": confidence}


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def analyze_image(image_path: Optional[str], action_text: str) -> Dict[str, Any]:
    if not image_path or not isinstance(image_path, (str, os.PathLike)) or not os.path.exists(str(image_path)):
        return _empty_result()

    img = cv2.imread(str(image_path))
    if img is None:
        return _empty_result()
    img_h, img_w = img.shape[:2]

    try:
        with open(str(image_path), "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"[VISION] Failed to read image '{image_path}': {e}")
        return _empty_result()

    # Ask for PIXEL coordinates against dimensions we state explicitly, and
    # normalize ourselves afterward -- Qwen-family grounding is documented
    # to work in actual pixel space relative to the image it was given, and
    # asking it to self-normalize to 0.0-1.0 was the likely cause of
    # coordinates coming back in the wrong space entirely.
    prompt = f"""
This screenshot is exactly {img_w} pixels wide and {img_h} pixels tall.

Locate the SINGLE specific interactive UI control (button, field, menu item,
checkbox, link, icon, or tab) needed for this action: "{action_text}"

Rules:
- Pick the smallest element that satisfies the action -- never a whole
  panel, toolbar, or the entire window.
- Give the bounding box in PIXEL coordinates within the {img_w}x{img_h}
  image described above (not a 0-1 fraction).
- If no control clearly matches, set "found" to false.

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
        data = _extract_json(raw)
        if not data:
            print(f"[VISION] Could not parse a JSON response for '{action_text}': {raw[:200]!r}")
            return _empty_result()

        if not data.get("found"):
            return _empty_result()

        box = data.get("bounding_box")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            return _empty_result()

        try:
            raw_x0, raw_y0, raw_x1, raw_y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            return _empty_result()

        # Normalize against the pixel dimensions we supplied. If the model
        # ignored that and returned values already in 0-1, this division
        # produces near-zero coordinates clustered in a corner -- detect
        # that case and treat the raw values as already-normalized instead.
        if max(raw_x0, raw_y0, raw_x1, raw_y1) <= 1.5:
            x0, y0, x1, y1 = raw_x0, raw_y0, raw_x1, raw_y1
        else:
            x0, y0, x1, y1 = raw_x0 / img_w, raw_y0 / img_h, raw_x1 / img_w, raw_y1 / img_h

        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        # Reject rather than clamp -- a value meaningfully outside [0, 1]
        # after normalization means the coordinate convention didn't match
        # what we assumed, and a clamped/guessed point is worse than
        # falling through to the next grounding source.
        if min(x0, y0) < -OUT_OF_RANGE_TOLERANCE or max(x1, y1) > 1.0 + OUT_OF_RANGE_TOLERANCE:
            print(f"[VISION] Discarding out-of-range box {[raw_x0, raw_y0, raw_x1, raw_y1]} for '{action_text}'")
            return _empty_result()

        x0, y0, x1, y1 = (max(0.0, min(1.0, v)) for v in (x0, y0, x1, y1))
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area < MIN_BOX_FRACTION:
            return _empty_result()
        if area > MAX_BOX_FRACTION:
            # The model likely grounded the whole screen/panel rather than a
            # specific control -- discard instead of drawing a misleading
            # cursor in the middle of everything.
            return _empty_result()

        confidence = float(data.get("confidence", 0.6) or 0.6)
        confidence *= max(0.4, 1.0 - area)

        return {
            "found": True,
            "target_name": str(data.get("target_name", "") or ""),
            "bounding_box": [x0, y0, x1, y1],
            "click_point": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
            "confidence": round(confidence, 3),
        }
    except requests.exceptions.Timeout:
        print(f"[VISION] Timed out after {VISION_TIMEOUT}s grounding '{action_text}' -- "
              f"consider raising OLLAMA_VISION_TIMEOUT if this happens often.")
    except Exception as e:
        print(f"[VISION] Qwen inference skipped: {e}")

    return _empty_result()
