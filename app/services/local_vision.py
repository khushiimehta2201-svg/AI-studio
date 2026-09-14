import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

VISION_MODEL = os.getenv(
    "OLLAMA_VISION_MODEL",
    os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b"),
)

VISION_TIMEOUT = int(
    os.getenv("OLLAMA_VISION_TIMEOUT", "90")
)

MIN_TARGET_AREA = 18
MAX_CANDIDATES = 20


def _empty_result() -> Dict[str, Any]:
    return {
        "found": False,
        "click_point": None,
        "target_name": "",
        "confidence": 0.0,
        "source": "none",
        "bounding_box": None,
        "annotation": None,
    }


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _iou(box_a, box_b) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)

    intersection = iw * ih
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)

    denominator = area_a + area_b - intersection

    if denominator <= 0:
        return 0.0

    return intersection / denominator


def _detect_red_annotations(
    image: np.ndarray,
) -> List[Dict[str, Any]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 90, 70], dtype=np.uint8)
    upper1 = np.array([12, 255, 255], dtype=np.uint8)

    lower2 = np.array([165, 90, 70], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = image.shape[:2]
    annotations = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < max(MIN_TARGET_AREA, width * height * 0.00002):
            continue

        if w > width * 0.35 or h > height * 0.35:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        circularity = (
            4.0 * np.pi * cv2.contourArea(contour)
            / (perimeter * perimeter)
        )

        aspect = w / max(1.0, h)

        # Red numbered circles are normally compact and roughly circular.
        # Long red regions are treated as highlights, not number markers.
        if 0.55 <= aspect <= 1.8 and circularity >= 0.30:
            annotations.append(
                {
                    "type": "number_marker",
                    "box": [
                        x / width,
                        y / height,
                        (x + w) / width,
                        (y + h) / height,
                    ],
                    "center": [
                        (x + w / 2.0) / width,
                        (y + h / 2.0) / height,
                    ],
                    "area": area,
                }
            )

    # Sort top-to-bottom, then left-to-right. This gives a deterministic
    # visual order without allowing the red marker itself to become a target.
    annotations.sort(
        key=lambda item: (
            item["center"][1],
            item["center"][0],
        )
    )

    for index, item in enumerate(annotations, start=1):
        item["order"] = index

    return annotations


def _detect_highlight_regions(
    image: np.ndarray,
) -> List[Dict[str, Any]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Bright/high-saturation pixels are used only to identify possible
    # annotation regions. They are never themselves click targets.
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = cv2.inRange(
        saturation,
        90,
        255,
    )

    bright = cv2.inRange(
        value,
        130,
        255,
    )

    mask = cv2.bitwise_and(mask, bright)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = image.shape[:2]
    regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < max(20, width * 0.015):
            continue

        if h < max(8, height * 0.006):
            continue

        if w > width * 0.95 and h > height * 0.95:
            continue

        area = w * h

        if area > width * height * 0.30:
            continue

        regions.append(
            {
                "type": "highlight",
                "box": [
                    x / width,
                    y / height,
                    (x + w) / width,
                    (y + h) / height,
                ],
                "area": area,
            }
        )

    regions.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    return regions[:MAX_CANDIDATES]


def _build_candidate_regions(
    image: np.ndarray,
    ocr_words: Optional[List[Dict[str, Any]]],
    query: str,
) -> Dict[str, Any]:
    annotations = _detect_red_annotations(image)
    highlights = _detect_highlight_regions(image)

    candidates = []

    for word in ocr_words or []:
        text = str(word.get("text", "")).strip()
        box = word.get("box")

        if not text or not box or len(box) != 4:
            continue

        candidates.append(
            {
                "type": "ocr",
                "text": text,
                "box": box,
                "center": [
                    (box[0] + box[2]) / 2.0,
                    (box[1] + box[3]) / 2.0,
            ],
            }
        )

    return {
        "ocr_candidates": candidates[:MAX_CANDIDATES],
        "number_markers": annotations,
        "highlights": highlights,
    }


def _annotation_near_box(
    box,
    annotations: List[Dict[str, Any]],
):
    for annotation in annotations:
        if _iou(box, annotation["box"]) > 0.10:
            return annotation

        ax, ay = annotation["center"]
        x0, y0, x1, y1 = box

        if (
            x0 - 0.03 <= ax <= x1 + 0.03
            and y0 - 0.03 <= ay <= y1 + 0.03
        ):
            return annotation

    return None


def _nearest_ocr_candidate(
    point,
    candidates,
):
    if not candidates:
        return None

    px, py = point

    ranked = []

    for candidate in candidates:
        cx, cy = candidate["center"]
        distance = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        ranked.append((distance, candidate))

    ranked.sort(key=lambda item: item[0])

    return ranked[0][1]


def _validate_model_target(
    result: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict):
        return None

    point = result.get("click_point")

    if not isinstance(point, list) or len(point) != 2:
        return None

    try:
        x = _clamp(float(point[0]))
        y = _clamp(float(point[1]))
    except Exception:
        return None

    # A model must never return the center of a detected red number marker.
    for annotation in metadata["number_markers"]:
        ax, ay = annotation["center"]

        distance = (
            (x - ax) ** 2
            + (y - ay) ** 2
        ) ** 0.5

        marker_box = annotation["box"]
        marker_w = marker_box[2] - marker_box[0]
        marker_h = marker_box[3] - marker_box[1]

        radius = max(marker_w, marker_h) * 0.75

        if distance <= max(0.012, radius):
            print(
                "[VISION] Rejected target because it points at "
                f"number marker {annotation['order']}."
            )
            return None

    target_name = str(
        result.get("target_name")
        or result.get("target")
        or ""
    ).strip()

    confidence = _clamp(
        float(result.get("confidence", 0.0))
    )

    annotation = None

    for marker in metadata["number_markers"]:
        if (
            target_name
            and target_name.isdigit()
            and int(target_name) == marker["order"]
        ):
            # The number can be used as evidence, but not as the click target.
            annotation = {
                "type": "number_marker",
                "order": marker["order"],
                "box": marker["box"],
            }

    # If the model gives a point with no useful target name, prefer the nearest
    # OCR candidate rather than trusting a random location.
    nearest = _nearest_ocr_candidate(
        [x, y],
        metadata["ocr_candidates"],
    )

    if nearest:
        nearest_annotation = _annotation_near_box(
            nearest["box"],
            metadata["number_markers"],
        )

        if nearest_annotation:
            # OCR text overlapping the number marker is not a valid target.
            nearest = None

    if nearest and not target_name:
        target_name = nearest["text"]

    if not target_name:
        target_name = "UI control"

    return {
        "found": True,
        "click_point": [x, y],
        "target_name": target_name,
        "confidence": confidence,
        "source": "vision",
        "bounding_box": result.get("bounding_box"),
        "annotation": annotation,
    }


def analyze_image(
    image_path: str,
    query: str,
    ocr_words: Optional[List[Dict[str, Any]]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    result = _empty_result()

    image = cv2.imread(str(image_path))

    if image is None:
        return result

    metadata = _build_candidate_regions(
        image,
        ocr_words,
        query,
    )

    if candidates:
        metadata["ocr_candidates"] = [
            {
                "type": "ocr",
                "text": candidate.get("matched", ""),
                "box": candidate.get("box"),
                "center": candidate.get("point"),
            }
            for candidate in candidates
            if candidate.get("box") and candidate.get("point")
        ]

    number_markers = metadata["number_markers"]
    highlights = metadata["highlights"]

    marker_description = "\n".join(
        (
            f"- marker {item['order']}: "
            f"center={item['center']} box={item['box']}"
        )
        for item in number_markers
    )

    highlight_description = "\n".join(
        (
            f"- highlight {index}: box={item['box']}"
        )
        for index, item in enumerate(highlights, start=1)
    )

    ocr_description = "\n".join(
        (
            f"- text='{item['text']}' "
            f"box={item['box']} "
            f"center={item['center']}"
        )
        for item in metadata["ocr_candidates"]
    )

    prompt = f"""
You are grounding a software tutorial action to a screenshot.

ACTION:
{query}

IMPORTANT RULES:

1. Return the location of the ACTUAL UI CONTROL that the user should interact
   with, such as a button, menu item, field, tab, checkbox, link, tree item,
   icon, or other application control.

2. Red numbered circles are annotations only.
   NEVER click a red numbered circle.
   NEVER return the center of a red numbered circle.

3. Highlights are annotations only.
   NEVER click a highlight border or highlight region merely because it is
   highlighted.

4. If a numbered marker points toward an actual UI element, use the marker as
   EVIDENCE to identify that UI element, but return the UI element's location.

5. If several numbered markers exist, preserve their visual order. Do not
   select marker 7 when the requested action corresponds to marker 5.

6. Prefer OCR text matching the requested action when available.

7. If multiple possible UI targets exist, choose the one best supported by the
   action wording, OCR, nearby annotation, and UI context.

8. The click point must be inside the actual UI element, not on an annotation.

Detected numbered markers:
{marker_description or "none"}

Detected highlight regions:
{highlight_description or "none"}

OCR candidates:
{ocr_description or "none"}

Return JSON only:
{{
  "found": true,
  "target_name": "actual UI control name",
  "click_point": [x, y],
  "bounding_box": [x0, y0, x1, y1],
  "confidence": 0.0
}}

Coordinates must be normalized from 0.0 to 1.0.
"""

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "images": [],
        "options": {
            "temperature": 0.05,
            "num_ctx": 4096,
        },
    }

    try:
        with open(image_path, "rb") as handle:
            payload["images"] = [
                base64.b64encode(handle.read()).decode("utf-8")
            ]

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=VISION_TIMEOUT,
        )
        response.raise_for_status()

        raw = response.json().get("response", "")
        parsed = _extract_json(raw)

        validated = _validate_model_target(
            parsed or {},
            metadata,
        )

        if validated:
            return validated

    except Exception as exc:
        print(f"[VISION] Ollama vision error: {exc}")

    # Do not invent a click location if vision failed.
    return result
