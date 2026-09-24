"""Compatibility wrapper for vision calls.

The planner uses model_adapters directly. This module keeps the old import path
usable for integrations that still call analyze_image().
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from app.services.model_adapters import create_clients


def analyze_image(image_path: str, action_text: str, target_type: str = "", action: str = "") -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        return {"found": False, "confidence": 0.0}
    _, vision = create_clients()
    image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    prompt = f"""
Ground the software-training action to the visible application UI in this screenshot.
ACTION: {action or action_text}
PRIMARY TARGET: {action_text}
TARGET TYPE: {target_type or 'infer from action'}

Return ONLY JSON:
{{
  "found": true|false,
  "target_name": "exact visible UI label or concise description",
  "bounding_box": [x0,y0,x1,y1],
  "click_point": [x,y],
  "confidence": 0.0,
  "evidence": "brief visible evidence"
}}
Coordinates are normalized 0..1. Do not target document text, headings, captions, logos or decorative content.
""".strip()
    try:
        return vision.generate_json(prompt, image_b64) or {"found": False, "confidence": 0.0}
    except Exception:
        return {"found": False, "confidence": 0.0}
