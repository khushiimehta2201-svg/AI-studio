import base64
import json
import os
import re
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:7b")
VISION_TIMEOUT = int(os.getenv("OLLAMA_VISION_TIMEOUT", "90"))

# Grid dimensions for cell-based grounding. Larger grids are more precise
# but harder for the model to read correctly off a small/compressed image;
# 6x4 is a reasonable balance for typical widescreen UI screenshots.
GRID_COLS = int(os.getenv("OLLAMA_VISION_GRID_COLS", "6"))
GRID_ROWS = int(os.getenv("OLLAMA_VISION_GRID_ROWS", "4"))


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


def _draw_grid_overlay(img: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """Draws a labeled grid on a COPY of the image for the model to read
    cell numbers off of. The original screenshot used for the actual
    rendered video is never touched -- this overlay exists only inside the
    base64 payload sent to the vision model."""
    overlay = img.copy()
    h, w = overlay.shape[:2]
    cell_w, cell_h = w / cols, h / rows

    for c in range(1, cols):
        x = int(c * cell_w)
        cv2.line(overlay, (x, 0), (x, h), (0, 200, 255), 1, cv2.LINE_AA)
    for r in range(1, rows):
        y = int(r * cell_h)
        cv2.line(overlay, (0, y), (w, y), (0, 200, 255), 1, cv2.LINE_AA)

    cell = 1
    for r in range(rows):
        for c in range(cols):
            cx = int(c * cell_w + 6)
            cy = int(r * cell_h + 18)
            label = str(cell)
            cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
            cell += 1
    return overlay


def _cell_to_point(cell: int, cols: int, rows: int):
    if cell < 1 or cell > cols * rows:
        return None
    idx = cell - 1
    row, col = idx // cols, idx % cols
    return (col + 0.5) / cols, (row + 0.5) / rows


def analyze_image(image_path: Optional[str], action_text: str) -> Dict[str, Any]:
    if not image_path or not isinstance(image_path, (str, os.PathLike)) or not os.path.exists(str(image_path)):
        return _empty_result()

    img = cv2.imread(str(image_path))
    if img is None:
        return _empty_result()

    grid_img = _draw_grid_overlay(img, GRID_COLS, GRID_ROWS)
    ok, buf = cv2.imencode(".png", grid_img)
    if not ok:
        return _empty_result()
    image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    total_cells = GRID_COLS * GRID_ROWS

    # Asking the model to name a labeled grid cell -- rather than regress a
    # raw pixel/normalized bounding box -- sidesteps coordinate-convention
    # ambiguity entirely. We drew the grid ourselves, so mapping a cell
    # number back to a screen position has zero guesswork, unlike raw
    # coordinates which some VLMs report in their own internal resized
    # image space regardless of what units the prompt asks for.
    prompt = f"""
This screenshot has a numbered grid overlaid on it: {GRID_COLS} columns x
{GRID_ROWS} rows, numbered 1 to {total_cells} left-to-right, top-to-bottom
(cell 1 is top-left, cell {total_cells} is bottom-right).

Find the SINGLE specific interactive UI control (button, field, menu item,
checkbox, link, icon, or tab) needed for this action: "{action_text}"

Reply with the number of the grid cell that contains that control. If the
control spans more than one cell, give the cell containing its center. If
no control clearly matches this action anywhere in the image, set "found"
to false.

Return ONLY JSON, no prose:
{{
  "found": true,
  "target_name": "short name of the control",
  "cell": 0,
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

        try:
            cell = int(data.get("cell"))
        except (TypeError, ValueError):
            print(f"[VISION] No usable cell number in response for '{action_text}': {data!r}")
            return _empty_result()

        point = _cell_to_point(cell, GRID_COLS, GRID_ROWS)
        if point is None:
            print(f"[VISION] Cell {cell} is out of the {GRID_COLS}x{GRID_ROWS} grid range for '{action_text}'")
            return _empty_result()

        confidence = float(data.get("confidence", 0.55) or 0.55)
        # Grid-cell precision is inherently coarser than true pixel
        # grounding, so cap confidence below OCR/text-match results even
        # when the model reports high certainty about which cell is right.
        confidence = min(confidence, 0.65)

        return {
            "found": True,
            "target_name": str(data.get("target_name", "") or ""),
            "bounding_box": None,
            "click_point": list(point),
            "confidence": round(confidence, 3),
        }
    except requests.exceptions.Timeout:
        print(f"[VISION] Timed out after {VISION_TIMEOUT}s grounding '{action_text}' -- "
              f"consider raising OLLAMA_VISION_TIMEOUT if this happens often.")
    except Exception as e:
        print(f"[VISION] Qwen inference skipped: {e}")

    return _empty_result()
