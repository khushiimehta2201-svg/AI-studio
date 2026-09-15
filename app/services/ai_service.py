from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple



# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Keep Ollama disabled for the first successful end-to-end test.
#
# Your previous run was repeatedly waiting for Ollama and timing
# out. We will add optional AI enhancement later.
ENABLE_OLLAMA = os.getenv(
    "ENABLE_OLLAMA",
    "true",
).lower() in {"1", "true", "yes", "on"}

# Vision results are cached in-memory during a generation run so the same
# screenshot is not repeatedly analyzed for identical actions.
_VISION_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}

ACTION_VERBS = (
    "click",
    "select",
    "choose",
    "open",
    "enter",
    "type",
    "press",
    "add",
    "create",
    "delete",
    "save",
    "submit",
    "upload",
    "download",
    "drag",
    "drop",
    "navigate",
    "login",
    "log in",
    "search",
    "filter",
    "expand",
    "collapse",
    "double-click",
    "right-click",
)


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _shorten(
    text: str,
    limit: int = 220,
) -> str:

    text = _clean_text(text)

    if len(text) <= limit:
        return text

    result = text[:limit]

    if " " in result:
        result = result.rsplit(" ", 1)[0]

    return result.rstrip(" ,;:.") + "..."


def _progress(
    callback,
    value: int,
    message: str,
) -> None:

    if callback is None:
        return

    value = max(
        0,
        min(
            100,
            int(value),
        ),
    )

    try:
        callback(
            value,
            message,
        )
    except TypeError:
        callback(value)


# ============================================================
# ACTION DETECTION
# ============================================================

def _is_action_line(
    line: str,
) -> bool:

    text = _clean_text(
        line
    ).lower()

    if not text:
        return False

    # Explicit numbered instructions such as:
    # 1. Click...
    # 2) Select...
    if re.match(
        r"^\s*\d+[\.\):\-]\s*",
        text,
    ):
        remainder = re.sub(
            r"^\s*\d+[\.\):\-]\s*",
            "",
            text,
        )

        if any(
            re.search(
                rf"\b{re.escape(verb)}\b",
                remainder,
            )
            for verb in ACTION_VERBS
        ):
            return True

    # Bullet action instructions.
    if re.match(
        r"^\s*[-*•]\s*",
        text,
    ):
        if any(
            re.search(
                rf"\b{re.escape(verb)}\b",
                text,
            )
            for verb in ACTION_VERBS
        ):
            return True

    # Normal action sentence.
    return any(
        re.search(
            rf"\b{re.escape(verb)}\b",
            text,
        )
        for verb in ACTION_VERBS
    )


def _extract_action_lines(
    page: Dict[str, Any],
) -> Tuple[List[str], List[str]]:

    raw_text = page.get(
        "text",
        "",
    )

    if not raw_text:
        return [], []

    lines = [
        _clean_text(line)
        for line in str(raw_text).splitlines()
        if _clean_text(line)
    ]

    heading = _clean_text(
        page.get("heading")
    )

    if heading:
        content_lines = [
            line
            for line in lines
            if line != heading
        ]
    else:
        content_lines = lines

    actions = []
    explanations = []

    for line in content_lines:

        words = line.split()
        title_like_heading = (
            1 < len(words) <= 8
            and all(word[:1].isupper() for word in words if word[:1].isalpha())
            and not re.match(
                r"^(click|double-click|right-click|select|choose|open|enter|type|press|add|create|delete|save|submit|upload|download|drag|drop|navigate|login|log in|search|filter|expand|collapse)\b",
                line,
                flags=re.IGNORECASE,
            )
        )

        if _is_action_line(line) and not title_like_heading:
            actions.append(line)
        else:
            explanations.append(line)

    return actions, explanations


# ============================================================
# TARGET QUERY
# ============================================================

def _target_query(action: str) -> str:
    text = _clean_text(action)
    if not text:
        return ""
    text = re.sub(r"^\s*\d+[\.\):\-]\s*", "", text).strip()

    # Prefer the UI object named immediately after an interaction verb.
    # This gives the vision model a much cleaner target than the whole sentence.
    m = re.search(
        r"\b(?:double-click|right-click|click|select|choose|press)\s+(?:on\s+)?(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+(?:to|for|so|and|before|then)\b|[.,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        candidate = _clean_text(m.group(1))
        if candidate:
            return _shorten(candidate, 100)

    # For typing/entering, preserve the field context as it is often more
    # useful than the entered value itself.
    m = re.search(
        r"\b(?:enter|type)\s+(.+?)\s+in(?:to)?\s+(?:the\s+)?(.+?)(?:[.,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return _shorten(_clean_text(m.group(2)), 100)

    m = re.search(r"\b(?:open|navigate)\s+(?:to\s+)?(?:the\s+)?(.+?)(?:\s+and\s+|[.,;]|$)", text, flags=re.IGNORECASE)
    if m:
        candidate = _clean_text(m.group(1))
        if candidate:
            return _shorten(candidate, 100)

    # Login actions usually mean the Login control, not the surrounding page.
    if re.search(r"\blog[ -]?in\b", text, re.IGNORECASE):
        return "Login button"

    return _shorten(text, 100)


# ============================================================
# PAGE / SCREENSHOT HELPERS
# ============================================================

def _page_number(
    page: Dict[str, Any],
    fallback: int,
) -> int:

    for key in (
        "page",
        "page_num",
        "page_number",
    ):

        if page.get(key) is not None:
            value = _safe_int(
                page.get(key),
                -1,
            )

            if value >= 0:
                return value

    return fallback


def _screenshot_page(
    screenshot: Dict[str, Any],
) -> Optional[int]:

    for key in (
        "page",
        "page_num",
        "page_number",
    ):

        if screenshot.get(key) is not None:

            value = _safe_int(
                screenshot.get(key),
                -1,
            )

            if value >= 0:
                return value

    return None


def _screenshot_path(
    screenshot: Dict[str, Any],
) -> Optional[str]:

    for key in (
        "path",
        "image_path",
        "image",
        "file",
        "filename",
        "screenshot",
    ):

        value = screenshot.get(key)

        if value:
            return str(value)

    return None


def _build_screenshot_map(
    screenshots: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:

    result: Dict[
        int,
        List[Dict[str, Any]]
    ] = {}

    for screenshot in screenshots:

        if not isinstance(
            screenshot,
            dict,
        ):
            continue

        page = _screenshot_page(
            screenshot
        )

        if page is None:
            continue

        result.setdefault(
            page,
            [],
        ).append(
            screenshot
        )

    for page_screenshots in result.values():

        page_screenshots.sort(
            key=lambda item: _safe_int(
                item.get(
                    "screenshot_index",
                    item.get(
                        "index",
                        0,
                    ),
                ),
                0,
            )
        )

    return result


# ============================================================
# PDF TEXT TARGET FALLBACK
# ============================================================

def _normalise_words(
    page: Dict[str, Any],
) -> List[Dict[str, Any]]:

    words = page.get(
        "words",
        [],
    )

    if not isinstance(
        words,
        list,
    ):
        return []

    result = []

    for word in words:

        if not isinstance(
            word,
            (list, tuple, dict),
        ):
            continue

        if isinstance(
            word,
            dict,
        ):

            text = _clean_text(
                word.get(
                    "text",
                    word.get(
                        "word",
                        "",
                    ),
                )
            )

            if not text:
                continue

            try:
                x0 = float(
                    word.get(
                        "x0",
                        word.get(
                            "x",
                            0,
                        ),
                    )
                )

                y0 = float(
                    word.get(
                        "y0",
                        word.get(
                            "y",
                            0,
                        ),
                    )
                )

                x1 = float(
                    word.get(
                        "x1",
                        x0,
                    )
                )

                y1 = float(
                    word.get(
                        "y1",
                        y0,
                    )
                )

            except Exception:
                continue

        else:

            # PyMuPDF word format:
            #
            # x0, y0, x1, y1, word,
            # block_no, line_no, word_no

            if len(word) < 5:
                continue

            try:

                x0 = float(word[0])
                y0 = float(word[1])
                x1 = float(word[2])
                y1 = float(word[3])

            except Exception:
                continue

            text = _clean_text(
                word[4]
            )

            if not text:
                continue

        result.append(
            {
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            }
        )

    return result


def _find_text_target(
    query: str,
    page: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    query = _clean_text(
        query
    ).lower()

    if not query:
        return None

    words = _normalise_words(
        page
    )

    if not words:
        return None

    query_tokens = [
        token
        for token in re.findall(
            r"[a-zA-Z0-9_]+",
            query,
        )
        if len(token) > 1
    ]

    if not query_tokens:
        return None

    best = None
    best_score = 0

    for word in words:

        word_text = word[
            "text"
        ].lower()

        score = 0

        if query == word_text:
            score += 10

        if query in word_text:
            score += 7

        for token in query_tokens:

            if token == word_text:
                score += 5

            elif token in word_text:
                score += 2

        if score > best_score:

            x0 = word["x0"]
            y0 = word["y0"]
            x1 = word["x1"]
            y1 = word["y1"]

            best = {
                "point": [
                    (x0 + x1) / 2.0,
                    (y0 + y1) / 2.0,
                ],
                "source": "pdf-text",
                "confidence": min(
                    0.90,
                    0.50 + score * 0.04,
                ),
                "target_name": word["text"],
                "bounding_box": [
                    x0,
                    y0,
                    x1,
                    y1,
                ],
                "annotation": word["text"],
            }

            best_score = score

    return best


# ============================================================
# SCREENSHOT TARGET FALLBACK
# ============================================================

def _candidate_target(
    query: str,
    screenshot: Dict[str, Any],
) -> Optional[Dict[str, Any]]:

    candidates = []

    for key in (
        "candidates",
        "targets",
        "vision_words",
        "words",
        "text_regions",
    ):

        value = screenshot.get(
            key
        )

        if isinstance(
            value,
            list,
        ):
            candidates.extend(
                item
                for item in value
                if isinstance(
                    item,
                    dict,
                )
            )

    if not candidates:
        return None

    query_tokens = set(
        token
        for token in re.findall(
            r"[a-zA-Z0-9_]+",
            query.lower(),
        )
        if len(token) > 1
    )

    best = None
    best_score = 0

    for candidate in candidates:

        text = _clean_text(
            candidate.get(
                "text",
                candidate.get(
                    "label",
                    candidate.get(
                        "matched",
                        candidate.get(
                            "word",
                            "",
                        ),
                    ),
                ),
            )
        )

        if not text:
            continue

        candidate_tokens = set(
            re.findall(
                r"[a-zA-Z0-9_]+",
                text.lower(),
            )
        )

        score = len(
            query_tokens
            & candidate_tokens
        )

        if text.lower() == query.lower():
            score += 10

        box = candidate.get(
            "box"
        )

        point = candidate.get(
            "point"
        )

        if point is None and isinstance(
            box,
            dict,
        ):

            try:

                point = [
                    float(box["x"])
                    + float(box["width"]) / 2,
                    float(box["y"])
                    + float(box["height"]) / 2,
                ]

            except Exception:
                point = None

        if point is None:
            continue

        if score > best_score:

            best = {
                "point": list(point),
                "source": "screenshot-text",
                "confidence": min(
                    0.95,
                    0.55 + score * 0.08,
                ),
                "target_name": text,
                "bounding_box": box,
                "annotation": text,
            }

            best_score = score

    return best


def _ocr_screenshot_target(query: str, screenshot: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Best-effort OCR directly on the embedded UI screenshot. This is a
    local fallback and is only used when structured screenshot candidates do
    not already provide a target."""
    if not screenshot:
        return None
    path = _screenshot_path(screenshot)
    if not path or not os.path.exists(path):
        return None
    try:
        import pytesseract
        from PIL import Image
        image = Image.open(path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception as exc:
        # Tesseract is optional. Do not make generation fail when it is absent.
        return None

    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(t) > 1]
    if not tokens:
        return None
    best = None
    best_score = 0.0
    n = len(data.get("text", []))
    for i in range(n):
        text = _clean_text(data["text"][i])
        if not text:
            continue
        low = text.lower()
        score = 10.0 if low == query.lower() else 0.0
        score += sum(4.0 if tok == low else 1.5 if tok in low else 0 for tok in tokens)
        try:
            conf = float(data.get("conf", [0])[i])
            x, y, w, h = [float(data[k][i]) for k in ("left", "top", "width", "height")]
        except Exception:
            continue
        if conf < 20:
            continue
        if score > best_score:
            iw, ih = image.size
            best = {
                "point": [(x + w / 2) / max(1, iw), (y + h / 2) / max(1, ih)],
                "source": "screenshot-ocr",
                "confidence": min(0.92, 0.45 + score * 0.035),
                "target_name": text,
                "bounding_box": [x / max(1, iw), y / max(1, ih), (x+w) / max(1, iw), (y+h) / max(1, ih)],
                "annotation": text,
            }
            best_score = score
    return best


def _ground_action(
    action: str,
    page: Dict[str, Any],
    screenshot: Optional[Dict[str, Any]],
    allow_vision: bool = True,
) -> Dict[str, Any]:
    query = _target_query(action)
    if not query:
        return {
            "point": None, "source": "none", "confidence": 0.0,
            "target_name": "", "bounding_box": None, "annotation": None,
        }

    # 1) Structured screenshot candidates, when available.
    if screenshot:
        candidate = _candidate_target(query, screenshot)
        if candidate:
            return candidate

    # 2) Local OCR on the actual embedded UI screenshot.
    ocr_target = _ocr_screenshot_target(query, screenshot)
    if ocr_target:
        return ocr_target

    # 3) Optional local vision model. Disabled by default so generation stays
    # fast/offline; enable only when visual grounding is needed.
    if ENABLE_OLLAMA and allow_vision and screenshot:
        try:
            from app.services.local_vision import analyze_image
            path = _screenshot_path(screenshot)
            if path:
                cache_key = (os.path.abspath(path), query.lower())
                visual = _VISION_CACHE.get(cache_key)
                if visual is None:
                    visual = analyze_image(path, query)
                    _VISION_CACHE[cache_key] = visual
                if visual.get("found") and visual.get("click_point"):
                    return {
                        "point": visual.get("click_point"),
                        "source": visual.get("source", "vision"),
                        "confidence": float(visual.get("confidence", 0.0)),
                        "target_name": visual.get("target_name", query),
                        "bounding_box": visual.get("bounding_box"),
                        "annotation": visual.get("annotation"),
                    }
        except Exception as exc:
            print(f"[VISION] Target grounding skipped for '{query}': {exc}")

    # 4) PDF text coordinates are useful for full-page screenshots. For an
    # embedded UI screenshot they do NOT describe coordinates inside the
    # extracted image, so never pretend they are a valid UI target there.
    if screenshot is None or bool(screenshot.get("is_full_page")):
        text_target = _find_text_target(query, page)
        if text_target:
            return text_target

    # An embedded screenshot can still be grounded from PDF text when the
    # matching text is physically inside that image on the page. Convert the
    # page-space target to image-space only in that case; otherwise a PDF
    # coordinate would point at the wrong place in the extracted image.
    if screenshot and screenshot.get("page_rect"):
        rect = screenshot["page_rect"]
        if isinstance(rect, (list, tuple)) and len(rect) >= 4:
            text_target = _find_text_target(query, page)
            if text_target:
                x0, y0, x1, y1 = [float(value) for value in rect[:4]]
                px, py = text_target["point"][:2]
                if x0 <= px <= x1 and y0 <= py <= y1 and x1 > x0 and y1 > y0:
                    point = [(px - x0) / (x1 - x0), (py - y0) / (y1 - y0)]
                    box = text_target.get("bounding_box") or [px, py, px, py]
                    bounding_box = [
                        (float(box[0]) - x0) / (x1 - x0),
                        (float(box[1]) - y0) / (y1 - y0),
                        (float(box[2]) - x0) / (x1 - x0),
                        (float(box[3]) - y0) / (y1 - y0),
                    ]
                    return {
                        "point": point,
                        "source": "embedded-pdf-text",
                        "confidence": text_target.get("confidence", 0.0),
                        "target_name": text_target.get("target_name", query),
                        "bounding_box": bounding_box,
                        "annotation": text_target.get("annotation"),
                    }

    return {
        "point": None,
        "source": "none",
        "confidence": 0.0,
        "target_name": query,
        "bounding_box": None,
        "annotation": None,
    }


def _action_type(action: str) -> str:
    text = _clean_text(action).lower()
    if "double-click" in text:
        return "Double-click"
    if "right-click" in text:
        return "Right-click"
    for verb, label in (
        ("click", "Click"), ("select", "Select"), ("choose", "Choose"),
        ("enter", "Enter"), ("type", "Type"), ("press", "Press"),
        ("open", "Open"), ("navigate", "Navigate"), ("search", "Search"),
        ("save", "Save"), ("submit", "Submit"), ("upload", "Upload"),
        ("download", "Download"), ("create", "Create"), ("delete", "Delete"),
        ("expand", "Expand"), ("collapse", "Collapse"),
    ):
        if re.search(rf"\b{re.escape(verb)}\b", text):
            return label
    if "login" in text or "log in" in text:
        return "Login"
    return "Follow"


def _make_dialogue(action: str, target_name: str = "") -> Dict[str, str]:
    clean = re.sub(r"^\s*\d+[\.\):\-]\s*", "", _clean_text(action)).strip(" .;,")
    action_type = _action_type(clean)
    target = _clean_text(target_name)
    if target:
        current = f"{action_type} {target}"
    else:
        current = _shorten(clean, 90)

    if action_type in {"Click", "Double-click", "Right-click", "Select", "Choose", "Press"} and target:
        brief = f"Use the highlighted {target} to continue."
    elif action_type in {"Enter", "Type"} and target:
        brief = f"Enter the required value in the highlighted {target}."
    elif action_type == "Open" and target:
        brief = f"Open the highlighted {target}."
    elif action_type == "Search":
        brief = "Use the highlighted search control to find the required item."
    else:
        brief = _shorten(clean, 120)

    return {
        "title": "What to do now",
        "action_type": action_type,
        "current_action": current,
        "brief": brief,
        "intro": "Follow the highlighted control and continue the workflow.",
    }


# ============================================================
# NARRATION
# ============================================================

def _make_action_narration(action: str) -> str:
    action = _clean_text(action)
    if not action:
        return "Continue with the next action."

    action = re.sub(r"^\s*\d+[\.\):\-]\s*", "", action).strip(" .;,")
    # Training narration should sound like a human instructor, not a numbered
    # document being read aloud.
    if not action:
        return "Continue with the next action."
    first = action[:1].upper() + action[1:]
    return first.rstrip(" .;,") + "."


def _make_theory_bullets(
    explanation_lines: List[str],
) -> List[str]:

    bullets = []

    for line in explanation_lines:

        line = _clean_text(
            line
        )

        if len(line) < 12:
            continue

        bullets.append(
            _shorten(
                line,
                180,
            )
        )

        if len(bullets) >= 6:
            break

    return bullets


# ============================================================
# SECTION BUILDING
# ============================================================

def _page_has_action(
    page: Dict[str, Any],
) -> bool:

    actions, _ = _extract_action_lines(
        page
    )

    return bool(actions)


def _build_sections(
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    content_pages = [
        page
        for page in pages
        if isinstance(
            page,
            dict,
        )
        and not page.get(
            "skip_reason"
        )
    ]

    if not content_pages:
        content_pages = [
            page
            for page in pages
            if isinstance(
                page,
                dict,
            )
        ]

    if not content_pages:
        return []

    sections = []
    current = None

    for index, page in enumerate(
        content_pages
    ):

        heading = _clean_text(
            page.get(
                "heading"
            )
        )

        if heading or current is None:

            current = {
                "title": (
                    heading
                    or f"Section {len(sections) + 1}"
                ),
                "pages": [],
            }

            sections.append(
                current
            )

        current["pages"].append(
            page
        )

    # If the PDF has no useful headings,
    # split at activity boundaries so the
    # portal gets multiple meaningful sections.
    if (
        len(sections) <= 1
        and len(content_pages) > 6
    ):

        sections = []

        current = None

        for page in content_pages:

            if current is None:

                current = {
                    "title": (
                        page.get(
                            "heading"
                        )
                        or f"Section {len(sections) + 1}"
                    ),
                    "pages": [],
                }

                sections.append(
                    current
                )

            current["pages"].append(
                page
            )

            if _page_has_action(
                page
            ):
                current = None

    return sections


# ============================================================
# MAIN PLAN GENERATION
# ============================================================

def build_tutorial_plan(
    data: Dict[str, Any],
    narration_language: str = "en-us",
    progress_callback=None,
) -> Dict[str, Any]:

    pages = data.get(
        "pages",
        [],
    )

    screenshots = data.get(
        "screenshots",
        [],
    )

    if not isinstance(
        pages,
        list,
    ):
        pages = []

    if not isinstance(
        screenshots,
        list,
    ):
        screenshots = []

    _progress(
        progress_callback,
        2,
        "Preparing document structure",
    )

    shots_by_page = _build_screenshot_map(
        screenshots
    )

    _progress(
        progress_callback,
        8,
        (
            f"Mapped {len(screenshots)} "
            f"embedded visual regions"
        ),
    )

    sections_raw = _build_sections(
        pages
    )

    sections_out = []
    steps = []

    step_id = 0

    total_sections = max(
        1,
        len(sections_raw),
    )

    # ========================================================
    # Process each section
    # ========================================================

    for section_index, section in enumerate(
        sections_raw
    ):

        section_step_ids = []
        action_bullets = []
        explanation_lines = []

        section_actions = []

        # ----------------------------------------------------
        # Collect all actions and explanation text first.
        # ----------------------------------------------------

        for page in section.get(
            "pages",
            [],
        ):

            actions, explanations = (
                _extract_action_lines(
                    page
                )
            )

            explanation_lines.extend(
                explanations
            )

            for action in actions:
                section_actions.append(
                    (
                        page,
                        action,
                    )
                )

        # ----------------------------------------------------
        # Build one video step per action.
        # ----------------------------------------------------

        for action_index, (
            page,
            action,
        ) in enumerate(
            section_actions
        ):

            page_number = _page_number(
                page,
                1,
            )

            page_shots = shots_by_page.get(
                page_number,
                [],
            )

            # Choose the embedded screenshot that actually contains the
            # target. PDF pages often contain logos, arrows and UI screenshots
            # mixed together, so image sequence is not a reliable selector.
            chosen_shot = None
            grounding = None
            best_score = -1.0

            for candidate_shot in page_shots:
                candidate_grounding = _ground_action(
                    action,
                    page,
                    candidate_shot,
                    allow_vision=False,
                )
                score = float(candidate_grounding.get("confidence", 0.0))
                if candidate_grounding.get("point") is not None and score > best_score:
                    chosen_shot = candidate_shot
                    grounding = candidate_grounding
                    best_score = score

            # If no screenshot yielded a target, use the largest embedded
            # image on the page as the most likely application screenshot.
            if chosen_shot is None and page_shots:
                def _area(shot):
                    path = _screenshot_path(shot)
                    try:
                        from PIL import Image
                        with Image.open(path) as im:
                            return im.width * im.height
                    except Exception:
                        return 0
                chosen_shot = max(page_shots, key=_area)
                # If enabled, spend the expensive vision call only once on
                # the most likely application screenshot, not once per image.
                grounding = _ground_action(
                    action, page, chosen_shot, allow_vision=ENABLE_OLLAMA
                )

            if grounding is None:
                grounding = _ground_action(action, page, None, allow_vision=False)

            screenshot_path = (
                _screenshot_path(chosen_shot)
                if chosen_shot
                else None
            )

            narration = _make_action_narration(
                action
            )

            step_id += 1

            point = grounding.get(
                "point"
            )

            source = grounding.get(
                "source",
                "none",
            )

            confidence = grounding.get(
                "confidence",
                0.0,
            )

            target_name = grounding.get(
                "target_name",
                "",
            )

            bounding_box = grounding.get(
                "bounding_box"
            )

            annotation = grounding.get(
                "annotation"
            )

            step = {
                # Required by video_service
                "id": step_id,

                "section_id": (
                    section_index + 1
                ),

                "page_num": page_number,
                "page_width": float(page.get("width") or 0),
                "page_height": float(page.get("height") or 0),

                "kind": "action",

                "title": _shorten(
                    action,
                    80,
                ),

                "narration": narration,

                "tts_narration": narration,

                "caption": narration,

                "screenshot": screenshot_path,

                "is_full_page": (
                    screenshot_path is None
                ),

                # Cursor information
                "cursor": point,

                "targets": (
                    [point]
                    if point is not None
                    else []
                ),

                "cursor_source": source,

                "cursor_confidence": confidence,

                "cursor_target_name": target_name,

                "cursor_annotation": annotation,

                "cursor_bounding_box": bounding_box,

                # Used by the side dialogue panel and the video renderer.
                "action_type": _action_type(action),
                "dialogue": _make_dialogue(action, target_name),
            }

            steps.append(
                step
            )

            section_step_ids.append(
                step_id
            )

            action_bullets.append(
                _shorten(
                    action,
                    140,
                )
            )

        # ----------------------------------------------------
        # Theory / introduction
        # ----------------------------------------------------

        theory_bullets = _make_theory_bullets(
            explanation_lines
        )

        section_title = _clean_text(
            section.get(
                "title"
            )
        ) or (
            f"Section {section_index + 1}"
        )

        sections_out.append(
            {
                "id": section_index + 1,

                "title": _shorten(
                    section_title,
                    90,
                ),

                "theory_bullets": theory_bullets,

                "action_bullets": action_bullets,

                "step_ids": section_step_ids,

                # These are useful to the portal/UI
                # and do not interfere with rendering.
                "action_count": len(
                    section_step_ids
                ),

                "has_video": bool(
                    section_step_ids
                ),
            }
        )

        progress = int(
            10
            + 80
            * (
                (section_index + 1)
                / float(total_sections)
            )
        )

        _progress(
            progress_callback,
            progress,
            (
                f"Prepared section "
                f"{section_index + 1} "
                f"of {total_sections}"
            ),
        )

    # ========================================================
    # Safety fallback
    # ========================================================

    # Some PDFs have text where our normal action detector
    # doesn't recognize the instruction. In that case, create
    # action steps from meaningful page text rather than
    # returning a plan with zero renderable videos.
    if not steps and pages:

        _progress(
            progress_callback,
            92,
            "No explicit action verbs found; creating page walkthrough",
        )

        fallback_section = {
            "id": len(sections_out) + 1,
            "title": "Document Walkthrough",
            "theory_bullets": [],
            "action_bullets": [],
            "step_ids": [],
            "action_count": 0,
            "has_video": True,
        }

        sections_out.append(
            fallback_section
        )

        # Create at most one fallback action
        # per page, using meaningful text.
        for page_index, page in enumerate(
            pages
        ):

            if not isinstance(
                page,
                dict,
            ):
                continue

            page_text = _clean_text(
                page.get(
                    "text",
                    "",
                )
            )

            if not page_text:
                continue

            page_number = _page_number(
                page,
                page_index + 1,
            )

            page_shots = shots_by_page.get(
                page_number,
                [],
            )

            chosen_shot = (
                page_shots[0]
                if page_shots
                else None
            )

            screenshot_path = (
                _screenshot_path(
                    chosen_shot
                )
                if chosen_shot
                else None
            )

            description = _shorten(
                page_text,
                180,
            )

            narration = (
                "This step walks through "
                + description.rstrip(
                    " .;,"
                )
                + "."
            )

            step_id += 1

            step = {
                "id": step_id,

                "section_id": (
                    fallback_section["id"]
                ),

                "page_num": page_number,
                "page_width": float(page.get("width") or 0),
                "page_height": float(page.get("height") or 0),

                "kind": "action",

                "title": (
                    f"Review page {page_number}"
                ),

                "narration": narration,

                "tts_narration": narration,

                "caption": narration,

                "screenshot": screenshot_path,

                "is_full_page": (
                    screenshot_path is None
                ),

                "cursor": None,

                "targets": [],

                "cursor_source": "none",

                "cursor_confidence": 0.0,

                "cursor_target_name": "",

                "cursor_annotation": None,

                "cursor_bounding_box": None,
                "action_type": "Follow",
                "dialogue": _make_dialogue(description),
            }

            steps.append(
                step
            )

            fallback_section[
                "step_ids"
            ].append(
                step_id
            )

            fallback_section[
                "action_bullets"
            ].append(
                f"Review page {page_number}"
            )

            fallback_section[
                "action_count"
            ] += 1

            if len(steps) >= 100:
                break

    # ========================================================
    # Title
    # ========================================================

    title = None

    for page in pages:

        if not isinstance(
            page,
            dict,
        ):
            continue

        heading = _clean_text(
            page.get(
                "heading"
            )
        )

        if heading:
            title = heading
            break

    if not title:

        title = _clean_text(
            data.get(
                "title"
            )
        )

    if not title:

        title = _clean_text(
            data.get(
                "filename"
            )
        )

        if title.lower().endswith(
            ".pdf"
        ):
            title = title[:-4]

    if not title:
        title = (
            "Software Training Tutorial"
        )

    # ========================================================
    # Overview
    # ========================================================

    overview_parts = []

    for page in pages[:5]:

        if not isinstance(
            page,
            dict,
        ):
            continue

        text = _clean_text(
            page.get(
                "text",
                "",
            )
        )

        if text:
            overview_parts.append(
                text
            )

    overview_source = _shorten(
        " ".join(
            overview_parts
        ),
        500,
    )

    if overview_source:

        description = (
            "This tutorial provides a guided walkthrough "
            "of the uploaded document. "
            + overview_source
        )

    else:

        description = (
            "This tutorial provides a guided walkthrough "
            "of the uploaded software workflow."
        )

    # ========================================================
    # Final plan
    # ========================================================

    plan = {
        "title": title,

        "description": description,

        "overview": description,

        "narration_language": (
            narration_language
            or "en-us"
        ),

        "sections": sections_out,

        # IMPORTANT:
        # video_service.py expects this top-level list.
        "steps": steps,

        "scene_count": len(
            steps
        ),

        "action_count": sum(
            1
            for step in steps
            if step.get("kind") == "action"
        ),

        "total_pages": len(
            pages
        ),

        "total_screenshots": len(
            screenshots
        ),

        "visual_source": (
            "embedded PDF screenshots "
            "with full-page fallback"
        ),

        "ai_mode": (
            "local-deterministic"
            if not ENABLE_OLLAMA
            else "ollama-optional"
        ),
    }

    _progress(
        progress_callback,
        100,
        (
            f"Tutorial plan ready: "
            f"{len(steps)} action steps"
        ),
    )

    print(
        f"[AI] Plan created: "
        f"{len(sections_out)} sections, "
        f"{len(steps)} action steps, "
        f"{len(screenshots)} screenshots"
    )

    return plan


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

generate_tutorial = build_tutorial_plan