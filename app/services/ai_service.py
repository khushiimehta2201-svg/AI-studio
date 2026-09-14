import difflib
import functools
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import pytesseract
import requests
from pytesseract import Output

from app.services.local_vision import analyze_image


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5vl:7b",
)

OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))

VISION_ENABLED = os.getenv(
    "OLLAMA_VISION_ENABLED",
    "true",
).lower() == "true"

VISION_MIN_CONFIDENCE = float(
    os.getenv("OLLAMA_VISION_MIN_CONFIDENCE", "0.60")
)

OCR_MIN_CONFIDENCE = float(
    os.getenv("OCR_MIN_CONFIDENCE", "35")
)

OCR_FUZZY_MIN_RATIO = float(
    os.getenv("OCR_FUZZY_MIN_RATIO", "0.72")
)


_tesseract_cmd = os.getenv("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

try:
    version = pytesseract.get_tesseract_version()
    print(f"[OCR] Tesseract found (v{version}) -- OCR grounding ACTIVE.")
except Exception as exc:
    print(
        "[OCR] WARNING: Tesseract executable was not found. "
        "OCR grounding will be weaker.\n"
        f"[OCR] {exc}"
    )


ACTION_VERBS = (
    "double click",
    "double-click",
    "right click",
    "right-click",
    "navigate to",
    "go to",
    "log in",
    "sign in",
    "open",
    "click",
    "select",
    "enter",
    "type",
    "browse",
    "launch",
    "choose",
    "apply",
    "create",
    "save",
    "run",
    "fill",
    "search",
    "locate",
    "expand",
    "collapse",
    "drag",
    "drop",
    "press",
    "check",
    "uncheck",
    "set",
    "add",
    "remove",
    "delete",
    "submit",
    "upload",
    "download",
    "attach",
    "confirm",
    "cancel",
)

_ACTION_START = re.compile(
    r"^(?:" + "|".join(
        re.escape(v) for v in sorted(ACTION_VERBS, key=len, reverse=True)
    ) + r")\b",
    re.I,
)

_LEAD_VERB = re.compile(
    r"^(?:" + "|".join(
        re.escape(v) for v in sorted(ACTION_VERBS, key=len, reverse=True)
    ) + r")\s+(?:the\s+|on\s+the\s+|to\s+)?",
    re.I,
)

_GENERIC_NOUNS = re.compile(
    r"\b(button|field|menu|tab|link|icon|option|box|dropdown|"
    r"drop-down|area|screen|control|checkbox|panel)\b",
    re.I,
)

_LEADING_NUMBER = re.compile(
    r"^\s*\d+(?:\.\d+)*[\).:\-]?\s*"
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[ \t]+", " ", str(value).replace("\r", " ")).strip()


def _normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _is_action_line(line: str) -> bool:
    line = _LEADING_NUMBER.sub("", line.strip()).strip("-•* ")
    return (
        bool(line)
        and 2 <= len(line.split())
        and len(line) <= 220
        and bool(_ACTION_START.match(line))
    )


def _target_query(line: str) -> str:
    line = _LEADING_NUMBER.sub("", line.strip()).strip("-•* ")
    query = _LEAD_VERB.sub("", line, count=1)
    query = _GENERIC_NOUNS.sub("", query)
    query = re.split(r"[.,;:]", query)[0]
    return _clean_text(query).strip(" .,:;'\"")


def _ollama(
    prompt: str,
    image_path: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
        },
    }

    if image_path:
        try:
            import base64

            with open(image_path, "rb") as handle:
                payload["images"] = [
                    base64.b64encode(handle.read()).decode("utf-8")
                ]
        except Exception as exc:
            print(f"[OLLAMA] image encoding failed: {exc}")
            return None

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout or OLLAMA_TIMEOUT,
        )
        response.raise_for_status()

        raw = response.json().get("response", "")
        if not raw:
            return None

        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group(0))
    except Exception as exc:
        print(f"[OLLAMA] error: {exc}")

    return None


@functools.lru_cache(maxsize=512)
def _ocr_words(
    image_path: str,
) -> Tuple[Tuple[str, float, float, float, float, float], ...]:
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            return tuple()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_h, original_w = gray.shape[:2]

        scale = 2 if max(original_h, original_w) < 1800 else 1

        if scale > 1:
            gray = cv2.resize(
                gray,
                (original_w * scale, original_h * scale),
                interpolation=cv2.INTER_CUBIC,
            )

        data = pytesseract.image_to_data(
            gray,
            output_type=Output.DICT,
            config="--psm 11",
        )

        words = []

        for i in range(len(data.get("text", []))):
            text = str(data["text"][i] or "").strip()
            if not text:
                continue

            try:
                confidence = float(data["conf"][i])
            except Exception:
                confidence = -1.0

            if confidence < OCR_MIN_CONFIDENCE:
                continue

            x = float(data["left"][i])
            y = float(data["top"][i])
            w = float(data["width"][i])
            h = float(data["height"][i])

            words.append(
                (
                    text,
                    x / scale,
                    y / scale,
                    (x + w) / scale,
                    (y + h) / scale,
                    confidence,
                )
            )

        return tuple(words)

    except Exception as exc:
        print(f"[OCR] failed for {image_path}: {exc}")
        return tuple()


def _prepare_vision_ocr_words(image_path: str) -> List[Dict[str, Any]]:
    image = cv2.imread(str(image_path))
    if image is None:
        return []

    height, width = image.shape[:2]
    result = []

    for text, x0, y0, x1, y1, confidence in _ocr_words(str(image_path)):
        result.append(
            {
                "text": text,
                "box": [
                    x0 / max(1, width),
                    y0 / max(1, height),
                    x1 / max(1, width),
                    y1 / max(1, height),
                ],
                "confidence": confidence,
            }
        )

    return result


def _query_tokens(query: str) -> List[str]:
    result = []
    for token in re.findall(r"[A-Za-z0-9]+", query):
        normalized = _normalize_token(token)
        if normalized:
            result.append(normalized)
    return result


def _find_ocr_candidates(
    query: str,
    image_path: Optional[str],
) -> List[Dict[str, Any]]:
    if not query or not image_path or not os.path.exists(str(image_path)):
        return []

    image = cv2.imread(str(image_path))
    if image is None:
        return []

    height, width = image.shape[:2]
    words = list(_ocr_words(str(image_path)))
    if not words:
        return []

    target_tokens = _query_tokens(query)
    if not target_tokens:
        return []

    normalized = [_normalize_token(word[0]) for word in words]
    candidates = []

    max_tokens = min(len(target_tokens), 5)

    for n in range(max_tokens, 0, -1):
        target = target_tokens[:n]

        for start in range(0, len(words) - n + 1):
            if normalized[start:start + n] != target:
                continue

            chunk = words[start:start + n]

            x0 = min(item[1] for item in chunk)
            y0 = min(item[2] for item in chunk)
            x1 = max(item[3] for item in chunk)
            y1 = max(item[4] for item in chunk)

            candidates.append(
                {
                    "point": [
                        (x0 + x1) / 2.0 / max(1, width),
                        (y0 + y1) / 2.0 / max(1, height),
                    ],
                    "box": [
                        x0 / max(1, width),
                        y0 / max(1, height),
                        x1 / max(1, width),
                        y1 / max(1, height),
                    ],
                    "matched": " ".join(item[0] for item in chunk),
                    "match_type": "exact",
                    "ratio": 1.0,
                    "ocr_confidence": min(item[5] for item in chunk),
                }
            )

    target_phrase = "".join(target_tokens)

    for window in range(1, min(4, len(words)) + 1):
        for start in range(0, len(words) - window + 1):
            phrase = "".join(normalized[start:start + window])
            if not phrase:
                continue

            ratio = difflib.SequenceMatcher(
                None,
                phrase,
                target_phrase,
            ).ratio()

            if ratio < OCR_FUZZY_MIN_RATIO:
                continue

            chunk = words[start:start + window]
            x0 = min(item[1] for item in chunk)
            y0 = min(item[2] for item in chunk)
            x1 = max(item[3] for item in chunk)
            y1 = max(item[4] for item in chunk)

            candidates.append(
                {
                    "point": [
                        (x0 + x1) / 2.0 / max(1, width),
                        (y0 + y1) / 2.0 / max(1, height),
                    ],
                    "box": [
                        x0 / max(1, width),
                        y0 / max(1, height),
                        x1 / max(1, width),
                        y1 / max(1, height),
                    ],
                    "matched": " ".join(item[0] for item in chunk),
                    "match_type": "fuzzy",
                    "ratio": ratio,
                    "ocr_confidence": min(item[5] for item in chunk),
                }
            )

    candidates.sort(
        key=lambda item: (
            item["match_type"] == "exact",
            item["ratio"],
            item["ocr_confidence"],
        ),
        reverse=True,
    )

    selected = []

    for candidate in candidates:
        duplicate = False

        for existing in selected:
            dx = candidate["point"][0] - existing["point"][0]
            dy = candidate["point"][1] - existing["point"][1]

            if (dx * dx + dy * dy) ** 0.5 < 0.008:
                duplicate = True
                break

        if not duplicate:
            selected.append(candidate)

        if len(selected) >= 20:
            break

    return selected


def _find_ocr_target(query: str, image_path: Optional[str]):
    candidates = _find_ocr_candidates(query, image_path)
    if not candidates:
        return None

    candidate = candidates[0]
    return (
        candidate["point"][0],
        candidate["point"][1],
        candidate["matched"],
    )


def _find_word_targets(
    query: str,
    words: List[Dict[str, Any]],
) -> List[Tuple[float, float, str]]:
    if not query or not words:
        return []

    tokens = [
        _normalize_token(token)
        for token in query.split()
        if _normalize_token(token)
    ]

    normalized = [
        _normalize_token(word.get("text", ""))
        for word in words
    ]

    results = []

    for n in range(min(len(tokens), 5), 0, -1):
        target = tokens[:n]

        for i in range(0, len(words) - n + 1):
            if normalized[i:i + n] != target:
                continue

            chunk = words[i:i + n]

            x0 = min(float(item["x0"]) for item in chunk)
            y0 = min(float(item["y0"]) for item in chunk)
            x1 = max(float(item["x1"]) for item in chunk)
            y1 = max(float(item["y1"]) for item in chunk)

            results.append(
                (
                    (x0 + x1) / 2.0,
                    (y0 + y1) / 2.0,
                    " ".join(str(item["text"]) for item in chunk),
                )
            )

        if results:
            return results

    return results


def _map_page_point_to_screenshot(
    nx: float,
    ny: float,
    page_w: float,
    page_h: float,
    screenshot: Dict[str, Any],
):
    rect = screenshot.get("page_rect")
    if not rect:
        return None

    x0, y0, x1, y1 = rect
    x = nx * page_w
    y = ny * page_h

    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return None

    return (
        (x - x0) / max(1e-6, x1 - x0),
        (y - y0) / max(1e-6, y1 - y0),
    )


def _fallback_cursor(shot_path: str):
    try:
        image = cv2.imread(str(shot_path))
        if image is None:
            return None

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best = None
        best_area = 0

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch

            if area < w * h * 0.008:
                continue
            if cw < 40 or ch < 16:
                continue
            if cw > w * 0.9 and ch > h * 0.4:
                continue

            if area > best_area:
                best_area = area
                best = (x + cw / 2.0, y + ch / 2.0)

        if not best:
            return 0.5, 0.5

        return best[0] / w, best[1] / h

    except Exception:
        return None


def _ground_action(
    query: str,
    page: Optional[Dict[str, Any]],
    screenshot: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not screenshot:
        return None

    shot_path = screenshot.get("path")
    if not shot_path:
        return None

    candidates = _find_ocr_candidates(query, shot_path) if query else []

    if VISION_ENABLED:
        vision_words = _prepare_vision_ocr_words(shot_path)

        vision = analyze_image(
            shot_path,
            query or "the requested UI control",
            vision_words,
            candidates=candidates,
        )

        if (
            vision.get("found")
            and vision.get("confidence", 0.0) >= VISION_MIN_CONFIDENCE
            and vision.get("click_point") is not None
        ):
            return {
                "point": vision["click_point"],
                "source": vision.get("source", "vision"),
                "confidence": vision.get("confidence", 0.0),
                "target_name": vision.get("target_name", query),
                "bounding_box": vision.get("bounding_box"),
                "annotation": vision.get("annotation"),
            }

    exact = [
        candidate
        for candidate in candidates
        if candidate["match_type"] == "exact"
    ]

    if len(exact) == 1:
        candidate = exact[0]
        return {
            "point": candidate["point"],
            "source": "ocr",
            "confidence": 0.90,
            "target_name": candidate["matched"],
            "bounding_box": candidate["box"],
            "annotation": None,
        }

    if (
        page
        and query
        and screenshot.get("page_rect")
        and page.get("width")
        and page.get("height")
    ):
        page_matches = _find_word_targets(
            query,
            page.get("words", []),
        )

        mapped = []

        for x, y, text in page_matches:
            point = _map_page_point_to_screenshot(
                x / page["width"],
                y / page["height"],
                page["width"],
                page["height"],
                screenshot,
            )

            if point:
                mapped.append((point, text))

        if len(mapped) == 1:
            point, text = mapped[0]
            return {
                "point": list(point),
                "source": "text",
                "confidence": 0.78,
                "target_name": text,
                "bounding_box": None,
                "annotation": None,
            }

    diagnostic = _fallback_cursor(shot_path)
    print(
        f"[GROUND] No confident target for '{query}' on {shot_path} "
        f"(heuristic-only guess would have been {diagnostic}, not used)"
    )

    return {
        "point": None,
        "source": "none",
        "confidence": 0.0,
        "target_name": query,
        "bounding_box": None,
        "annotation": None,
    }


def _page_has_action(page: Dict[str, Any]) -> bool:
    return any(
        _is_action_line(line)
        for line in page.get("text", "").splitlines()
    )


def _fallback_split_by_activity(
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sections = []
    current = None

    for page in pages:
        if current is None:
            current = {
                "title": f"Section {len(sections) + 1}",
                "pages": [],
            }
            sections.append(current)

        current["pages"].append(page)

        if _page_has_action(page):
            current = None

    return sections


def _build_sections(
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    content_pages = [
        page for page in pages if not page.get("skip_reason")
    ]

    if not content_pages:
        content_pages = list(pages)

    sections = []
    current = None

    for page in content_pages:
        heading = page.get("heading")

        if heading or current is None:
            current = {
                "title": heading or f"Section {len(sections) + 1}",
                "pages": [],
            }
            sections.append(current)

        current["pages"].append(page)

    if len(sections) <= 1 and len(content_pages) > 6:
        return _fallback_split_by_activity(content_pages)

    return sections


def _generate_screenshot_narration(text: str) -> str:
    text = _clean_text(text)

    if not text:
        return "Continue with the next step."

    if len(text.split()) <= 24:
        return text.rstrip(".") + "."

    prompt = f"""
Rewrite this software tutorial action into one concise spoken sentence.

ACTION:
{text}

Return JSON only:
{{"narration":"..."}}
"""

    result = _ollama(prompt, timeout=20)

    if isinstance(result, dict) and result.get("narration"):
        return _clean_text(result["narration"])

    return text[:220].rstrip(" ,;") + "."


def _summarize_theory(text: str) -> List[str]:
    text = _clean_text(text)

    if not text:
        return []

    prompt = f"""
Summarize the following software training explanation into 3 to 6 concise study points.

Do not invent information.

SOURCE:
{text[:3000]}

Return JSON:
{{"bullets":["...","..."]}}
"""

    result = _ollama(prompt, timeout=25)

    if isinstance(result, dict) and isinstance(result.get("bullets"), list):
        return [
            _clean_text(item)
            for item in result["bullets"]
            if _clean_text(item)
        ][:6]

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if len(sentence.strip()) > 12
    ]

    return sentences[:6]


def build_tutorial_plan(
    data: Dict[str, Any],
    narration_language: str = "en-us",
    progress_callback=None,
) -> Dict[str, Any]:
    pages = data.get("pages", [])
    screenshots = data.get("screenshots", [])

    shots_by_page: Dict[Any, List[Dict[str, Any]]] = {}

    for screenshot in screenshots:
        shots_by_page.setdefault(
            screenshot.get("page"),
            [],
        ).append(screenshot)

    for shots in shots_by_page.values():
        shots.sort(
            key=lambda item: item.get("screenshot_index", 0)
        )

    sections_raw = _build_sections(pages)
    sections_out = []
    steps = []
    step_id = 0

    for section_index, section in enumerate(sections_raw):
        explanation_chunks = []
        section_step_ids = []
        section_shots = []
        section_actions = []

        for page in section["pages"]:
            lines = [
                line.strip()
                for line in page.get("text", "").splitlines()
                if line.strip()
            ]

            heading = page.get("heading")

            content_lines = [
                line for line in lines if line != heading
            ]

            action_lines = [
                line for line in content_lines
                if _is_action_line(line)
            ]

            explanation_lines = [
                line for line in content_lines
                if not _is_action_line(line)
            ]

            if explanation_lines:
                explanation_chunks.append(" ".join(explanation_lines))

            page_shots = [
                shot
                for shot in shots_by_page.get(page.get("page"), [])
                if not shot.get("is_full_page")
            ]

            section_shots.extend(
                (page, shot) for shot in page_shots
            )

            section_actions.extend(
                (page, line) for line in action_lines
            )

        for action_index, (page, line) in enumerate(section_actions):
            query = _target_query(line)

            chosen_page = page
            chosen_shot = None
            grounding = None

            if section_shots:
                initial_index = min(
                    len(section_shots) - 1,
                    (
                        action_index * len(section_shots)
                    )
                    // max(1, len(section_actions)),
                )

                chosen_page, chosen_shot = section_shots[initial_index]

                grounding = _ground_action(
                    query,
                    chosen_page,
                    chosen_shot,
                )

                if not grounding or grounding.get("point") is None:
                    for candidate_page, candidate_shot in section_shots:
                        if candidate_shot is chosen_shot:
                            continue

                        alternative = _ground_action(
                            query,
                            candidate_page,
                            candidate_shot,
                        )

                        if alternative and alternative.get("point") is not None:
                            grounding = alternative
                            chosen_page = candidate_page
                            chosen_shot = candidate_shot
                            break

            narration = _generate_screenshot_narration(line)
            step_id += 1

            point = grounding.get("point") if grounding else None
            source = grounding.get("source", "none") if grounding else "none"
            confidence = (
                grounding.get("confidence", 0.0)
                if grounding
                else 0.0
            )

            step = {
                "id": step_id,
                "section_id": section_index + 1,
                "page_num": chosen_page.get("page"),
                "kind": "action",
                "title": _clean_text(line)[:80],
                "narration": narration,
                "tts_narration": narration,
                "caption": narration,
                "screenshot": (
                    chosen_shot.get("path")
                    if chosen_shot
                    else None
                ),
                "is_full_page": chosen_shot is None,
                "cursor": point,
                "targets": [point] if point else [],
                "cursor_source": source,
                "cursor_confidence": confidence,
                "cursor_target_name": (
                    grounding.get("target_name", "")
                    if grounding
                    else ""
                ),
                "cursor_annotation": (
                    grounding.get("annotation")
                    if grounding
                    else None
                ),
                "cursor_bounding_box": (
                    grounding.get("bounding_box")
                    if grounding
                    else None
                ),
            }

            steps.append(step)
            section_step_ids.append(step_id)

        sections_out.append(
            {
                "id": section_index + 1,
                "title": _clean_text(section["title"])[:90],
                "theory_bullets": _summarize_theory(
                    " ".join(explanation_chunks)
                ),
                "step_ids": section_step_ids,
            }
        )

        if progress_callback:
            progress = int(
                10
                + 80
                * (
                    (section_index + 1)
                    / max(1, len(sections_raw))
                )
            )
            progress_callback(
                progress,
                f"Prepared section {section_index + 1} "
                f"of {len(sections_raw)}",
            )

    title = next(
        (
            page.get("heading")
            for page in pages
            if page.get("heading")
        ),
        None,
    ) or "Software Training Tutorial"

    return {
        "title": title,
        "description": "Enterprise tutorial generated from PDF screenshots.",
        "narration_language": narration_language,
        "sections": sections_out,
        "steps": steps,
        "scene_count": len(steps),
    }
