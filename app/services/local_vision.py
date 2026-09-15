import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


OLLAMA_URL = os.getenv(
    "OLLAMA_VISION_URL",
    "http://localhost:11434/api/generate",
)

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    os.getenv("OLLAMA_MODEL", "qwen3-vl:2b-instruct"),
)

VISION_TIMEOUT = int(
    os.getenv("VISION_TIMEOUT", "60")
)


def _normalise_point(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None

    try:
        x = float(value[0])
        y = float(value[1])
    except Exception:
        return None

    # Model should return normalized coordinates.
    # Clamp them to the image bounds.
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))

    return [x, y]


def _normalise_box(value):
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None

    try:
        values = [float(v) for v in value[:4]]
    except Exception:
        return None

    return [
        max(0.0, min(1.0, values[0])),
        max(0.0, min(1.0, values[1])),
        max(0.0, min(1.0, values[2])),
        max(0.0, min(1.0, values[3])),
    ]


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

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
    """
    Use existing OCR candidates if vision is unavailable.
    """

    candidates = candidates or []

    exact = [
        candidate
        for candidate in candidates
        if candidate.get("match_type") == "exact"
        and candidate.get("point") is not None
    ]

    if len(exact) == 1:
        candidate = exact[0]

        return {
            "found": True,
            "confidence": 0.90,
            "click_point": candidate.get("point"),
            "source": "ocr",
            "target_name": candidate.get("matched", query),
            "bounding_box": candidate.get("box"),
            "annotation": None,
        }

    # Basic word-level fallback.
    query_tokens = [
        token.lower()
        for token in (query or "").split()
        if token.strip()
    ]

    if query_tokens and vision_words:
        for word in vision_words:
            value = str(word.get("text", "")).lower()

            if any(token in value for token in query_tokens):
                try:
                    x = (
                        float(word["x0"]) + float(word["x1"])
                    ) / 2.0

                    y = (
                        float(word["y0"]) + float(word["y1"])
                    ) / 2.0
                except Exception:
                    continue

                return {
                    "found": True,
                    "confidence": 0.65,
                    "click_point": [x, y],
                    "source": "ocr_word",
                    "target_name": str(word.get("text", query)),
                    "bounding_box": [
                        word.get("x0"),
                        word.get("y0"),
                        word.get("x1"),
                        word.get("y1"),
                    ],
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
) -> Dict[str, Any]:
    """
    Ask the local Ollama vision model to locate the UI element described by
    query.

    Coordinates returned by the model must be normalized:
        x = 0..1 from left to right
        y = 0..1 from top to bottom
    """

    image_path = Path(image_path)

    if not image_path.exists():
        return _ocr_fallback(query, vision_words, candidates)

    try:
        # Resize large PDF screenshots before sending them to the local model.
        # This keeps visual grounding responsive while retaining enough detail
        # for normal Teamcenter controls.
        from io import BytesIO
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        max_side = 1200
        if max(image.size) > max_side:
            scale = max_side / float(max(image.size))
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=84, optimize=True)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        print(f"[VISION] Could not read image: {exc}")
        return _ocr_fallback(query, vision_words, candidates)

    prompt = f"""
You are locating a user-interface control in a software screenshot.

TARGET:
{query}

Find the center point of the UI control that best matches the TARGET.

Return JSON only:

{{
  "found": true,
  "confidence": 0.0,
  "click_point": [0.0, 0.0],
  "target_name": "short name",
  "bounding_box": [0.0, 0.0, 0.0, 0.0],
  "annotation": "brief explanation"
}}

Coordinate rules:
- click_point uses normalized [x, y]
- x=0 is the left edge and x=1 is the right edge
- y=0 is the top edge and y=1 is the bottom edge
- bounding_box is [x0, y0, x1, y1], also normalized
- If the target cannot be located reliably, set found=false and click_point=null.
- Do not invent a target.
"""

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=VISION_TIMEOUT,
        )

        response.raise_for_status()

        body = response.json()

        raw = body.get("response", "")
        result = _extract_json(raw)

        point = _normalise_point(
            result.get("click_point")
        )

        box = _normalise_box(
            result.get("bounding_box")
        )

        found = bool(result.get("found")) and point is not None

        if found:
            return {
                "found": True,
                "confidence": float(
                    result.get("confidence", 0.0)
                ),
                "click_point": point,
                "source": "vision",
                "target_name": result.get(
                    "target_name",
                    query,
                ),
                "bounding_box": box,
                "annotation": result.get("annotation"),
            }

        print(
            f"[VISION] No confident target found for '{query}'"
        )

    except Exception as exc:
        print(
            f"[VISION] Ollama vision unavailable for "
            f"'{query}': {exc}"
        )

    return _ocr_fallback(
        query,
        vision_words,
        candidates,
    )
