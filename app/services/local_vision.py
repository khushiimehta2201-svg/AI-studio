import base64
import json
import os
from typing import Any, Dict, Optional

import requests


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

VISION_MODEL = os.getenv(
    "OLLAMA_VISION_MODEL",
    "qwen2.5vl:7b",
)

VISION_TIMEOUT = int(
    os.getenv(
        "OLLAMA_VISION_TIMEOUT",
        "120",
    )
)


def _empty_result(
    confidence: float = 0.0,
) -> Dict[str, Any]:
    return {
        "found": False,
        "target_name": "",
        "bounding_box": None,
        "click_point": None,
        "approximate_position": None,
        "confidence": confidence,
    }


def _clamp(
    value: float,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            float(value),
        ),
    )


def _valid_point(
    point: Any,
) -> bool:
    return (
        isinstance(
            point,
            (list, tuple),
        )
        and len(point) == 2
        and all(
            isinstance(
                value,
                (int, float),
            )
            for value in point
        )
    )


def _valid_bbox(
    box: Any,
) -> bool:
    return (
        isinstance(
            box,
            (list, tuple),
        )
        and len(box) == 4
        and all(
            isinstance(
                value,
                (int, float),
            )
            for value in box
        )
    )


def _point_inside_bbox(
    point,
    bbox,
) -> bool:
    if not _valid_point(
        point
    ):
        return False

    if not _valid_bbox(
        bbox
    ):
        return False

    x, y = point

    x1, y1, x2, y2 = bbox

    left = min(
        float(x1),
        float(x2),
    )

    right = max(
        float(x1),
        float(x2),
    )

    top = min(
        float(y1),
        float(y2),
    )

    bottom = max(
        float(y1),
        float(y2),
    )

    return (
        left <= x <= right
        and top <= y <= bottom
    )


def _bbox_center(
    bbox,
):
    x1, y1, x2, y2 = bbox

    return [
        (
            float(x1)
            + float(x2)
        )
        / 2.0,

        (
            float(y1)
            + float(y2)
        )
        / 2.0,
    ]


def _load_image_base64(
    path: str,
) -> str:
    with open(
        path,
        "rb",
    ) as f:
        return base64.b64encode(
            f.read()
        ).decode(
            "utf-8"
        )


def _extract_json(
    text: str,
) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    try:
        return json.loads(
            text
        )
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if (
        start >= 0
        and end > start
    ):
        try:
            return json.loads(
                text[
                    start:end + 1
                ]
            )
        except json.JSONDecodeError:
            pass

    return None


def analyze_image(
    image_path: str,
    action: str,
) -> Dict[str, Any]:
    """
    Find the actual UI control corresponding to one action.

    Visual evidence priority:

    1. Red/orange rectangle around the control.
    2. Red/orange arrow pointing at the control.
    3. Other explicit instructional highlight.
    4. Clearly visible matching UI control.

    The model must NOT target:
    - annotation itself
    - arrow itself
    - annotation text
    - mouse pointer
    - caption
    - screenshot center
    - arbitrary location

    If the requested control cannot be reliably identified,
    found=false is returned.
    """

    if not os.path.exists(
        image_path
    ):
        return _empty_result()

    try:
        image_b64 = _load_image_base64(
            image_path
        )
    except Exception as exc:
        print(
            f"[VISION] image read failed: {exc}"
        )
        return _empty_result()

    prompt = f"""
You are a high-precision UI grounding system for a software
training video.

SCREENSHOT:
The supplied image is an actual screenshot from a software
training manual.

ACTION TO GROUND:
{action}

Your task is to identify the ONE ACTUAL UI CONTROL that the
user must interact with for this action.

============================================================
IMPORTANT: TRAINING-MANUAL ANNOTATIONS
============================================================

The screenshot may contain instructional annotations.

Common annotations include:

- red rectangles
- orange rectangles
- red outlines
- orange outlines
- red arrows
- orange arrows
- circles
- callouts
- highlighted regions

These annotations are strong evidence.

PRIORITY:

1. RED OR ORANGE BOX:
   If a red/orange rectangle surrounds the requested UI control,
   target the actual UI control INSIDE the rectangle.

2. RED OR ORANGE ARROW:
   If a red/orange arrow points to the requested control,
   target the actual UI control at the arrow endpoint.

3. OTHER HIGHLIGHT:
   If another obvious annotation identifies the control,
   use that evidence.

4. DIRECT UI MATCH:
   If there is no annotation, identify the visible UI control
   whose label/function matches the action.

============================================================
DO NOT TARGET
============================================================

Never target:

- the red/orange rectangle itself
- the arrow itself
- annotation text
- explanatory text
- captions
- the mouse pointer
- decorative elements
- blank areas
- the center of the screenshot merely because the target
  cannot be identified

============================================================
ACTION-SPECIFIC RULES
============================================================

For:

"Click Filter"

target the actual Filter button.

For:

"Click Add"

target the actual Add button.

For:

"Click Attachment"

target the actual Attachment tab.

For:

"Enter Username"

target the actual Username input field.

For:

"Enter Password"

target the actual Password input field.

For:

"Type abc"

target the actual input field where abc is entered.

For:

"Select ITL Design Part"

target the actual visible dropdown item only if it is visible.

If a menu/dropdown item is NOT visible, do not invent its location.

============================================================
COORDINATES
============================================================

Coordinates refer to the COMPLETE IMAGE.

x:
0 = left
1 = right

y:
0 = top
1 = bottom

bounding_box:

[x1, y1, x2, y2]

click_point:

[x, y]

The click_point MUST be inside the actual UI control's bounding box.

Do not use screenshot center as a fallback.

If the control cannot be reliably located:

{{
  "found": false,
  "target_name": "",
  "bounding_box": null,
  "click_point": null,
  "confidence": 0.0
}}

Return ONLY JSON.

Successful response:

{{
  "found": true,
  "target_name": "actual UI control",
  "bounding_box": [x1, y1, x2, y2],
  "click_point": [x, y],
  "confidence": 0.0
}}

The confidence should represent your confidence that the
identified control is the correct target.
"""

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [
            image_b64
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=VISION_TIMEOUT,
        )

        response.raise_for_status()

        body = response.json()

        raw = body.get(
            "response",
            "",
        )

        result = _extract_json(
            raw
        )

        if not result:
            print(
                "[VISION] Qwen returned invalid JSON"
            )
            return _empty_result()

        try:
            confidence = float(
                result.get(
                    "confidence",
                    0.0,
                )
                or 0.0
            )
        except Exception:
            confidence = 0.0

        found = bool(
            result.get(
                "found"
            )
        )

        if not found:
            return _empty_result(
                confidence
            )

        bbox = result.get(
            "bounding_box"
        )

        point = result.get(
            "click_point"
        )

        target_name = str(
            result.get(
                "target_name",
                "",
            )
            or ""
        ).strip()

        if not target_name:
            return _empty_result(
                confidence
            )

        if not _valid_bbox(
            bbox
        ):
            print(
                "[VISION] invalid bounding box"
            )
            return _empty_result(
                confidence
            )

        bbox = [
            _clamp(bbox[0]),
            _clamp(bbox[1]),
            _clamp(bbox[2]),
            _clamp(bbox[3]),
        ]

        if not _valid_point(
            point
        ):
            # This is NOT a screenshot-center fallback.
            # It is the center of the model-identified target.
            point = _bbox_center(
                bbox
            )

        point = [
            _clamp(point[0]),
            _clamp(point[1]),
        ]

        # Ensure click is actually within target.
        if not _point_inside_bbox(
            point,
            bbox,
        ):
            point = _bbox_center(
                bbox
            )

        return {
            "found": True,
            "target_name": target_name,
            "bounding_box": bbox,
            "click_point": point,
            "approximate_position": point,
            "confidence": confidence,
        }

    except Exception as exc:
        print(
            f"[VISION] failed for "
            f"{os.path.basename(image_path)}: "
            f"{exc}"
        )

        return _empty_result()


def analyze_frames_with_ollama(
    screenshots,
    action: str,
):
    results = []

    for screenshot in screenshots:
        if isinstance(
            screenshot,
            dict,
        ):
            path = (
                screenshot.get("path")
                or screenshot.get("file")
                or screenshot.get("screenshot")
            )
        else:
            path = screenshot

        if not path:
            continue

        results.append(
            analyze_image(
                path,
                action,
            )
        )

    return results