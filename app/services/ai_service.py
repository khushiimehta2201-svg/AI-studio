from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests


# ============================================================
# CONFIGURATION
# ============================================================

ENABLE_OLLAMA = os.getenv("ENABLE_OLLAMA", "true").lower() in {
    "1", "true", "yes", "on"
}

# Narration uses the same local Ollama installation as visual grounding.
# A separate switch makes it possible to disable only narration if required.
ENABLE_OLLAMA_NARRATION = os.getenv(
    "ENABLE_OLLAMA_NARRATION",
    "true" if ENABLE_OLLAMA else "false",
).lower() in {"1", "true", "yes", "on"}

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2"
OLLAMA_TIMEOUT = max(5, int(os.getenv("OLLAMA_NARRATION_TIMEOUT", os.getenv("OLLAMA_TIMEOUT", "25"))))

_ollama_host = os.getenv("OLLAMA_HOST", "").strip().rstrip("/")
_ollama_url = os.getenv("OLLAMA_URL", "").strip()
if not _ollama_url:
    if _ollama_host:
        if _ollama_host.endswith("/api/generate"):
            _ollama_url = _ollama_host
        else:
            _ollama_url = f"{_ollama_host}/api/generate"
    else:
        _ollama_url = "http://127.0.0.1:11435/api/generate"
OLLAMA_URL = _ollama_url

# Vision results are cached during one generation process so the same
# screenshot is not repeatedly analyzed for identical actions.
_VISION_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}

# Narration results are deterministic for the same source/action/context.
_NARRATION_CACHE: Dict[Tuple[str, str, str], str] = {}

# Once a local Ollama connection has definitively failed, do not wait for the
# same unavailable server once per action. Deterministic fallback remains live.
_OLLAMA_NARRATION_DISABLED = False

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
    "launch",
    "visit",
    "switch",
    "fill",
)

_ACTION_WORD_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(v) for v in ACTION_VERBS)
    + r")\b",
    flags=re.IGNORECASE,
)

_URL_RE = re.compile(
    r"(?<![\w@])"
    r"(?:https?://|www\.)"
    r"[^\s<>\]\[\"')]+",
    flags=re.IGNORECASE,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _shorten(text: str, limit: int = 220) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    result = text[:limit]
    if " " in result:
        result = result.rsplit(" ", 1)[0]
    return result.rstrip(" ,;:.") + "..."


def _progress(callback, value: int, message: str) -> None:
    if callback is None:
        return
    value = max(0, min(100, int(value)))
    try:
        callback(value, message)
    except TypeError:
        callback(value)


def _extract_urls(text: str) -> List[str]:
    """Return HTTP(S)/www URLs in source order without duplicates."""
    urls: List[str] = []
    for match in _URL_RE.findall(str(text or "")):
        url = match.rstrip(".,;:!?)]}")
        if url and url not in urls:
            urls.append(url)
    return urls


def _remove_urls(text: str) -> str:
    """Remove URLs from spoken material while keeping surrounding words."""
    value = _URL_RE.sub(" ", str(text or ""))
    return _clean_text(value)


def _clean_spoken_narration(text: str) -> str:
    """
    Clean model/source output for TTS.

    Narration must never contain URLs, JSON wrappers, Markdown fences, or
    document-reader filler such as "Narration:".
    """
    value = _remove_urls(text)
    value = value.replace("```json", "").replace("```", "").strip()

    # Handle common model prefixes without deleting legitimate content.
    value = re.sub(
        r"^\s*(?:narration|spoken narration|voiceover)\s*:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"^\s*(?:here(?:'s| is)\s+(?:the\s+)?(?:narration|spoken narration)\s*:\s*)",
        "",
        value,
        flags=re.IGNORECASE,
    )

    # A numbered item is useful in the source document but should not be read.
    value = re.sub(r"^\s*\d+[\.\):\-]\s*", "", value)
    value = re.sub(r"^\s*[-*•]\s*", "", value)
    value = _clean_text(value)

    return value.strip(" \"'")[:900]


# ============================================================
# ACTION DETECTION / PARSING
# ============================================================

def _list_marker(line: str) -> Optional[str]:
    if re.match(r"^\s*\d+[\.\):\-]\s+", line):
        return "numbered"
    if re.match(r"^\s*[-*•]\s+", line):
        return "bullet"
    return None


def _line_after_marker(line: str) -> str:
    return re.sub(
        r"^\s*(?:\d+[\.\):\-]|[-*•])\s*",
        "",
        line,
    ).strip()


def _is_action_line(line: str) -> bool:
    """
    Identify actual instructions, not merely sentences containing action
    verbs as nouns/objects.

    Strong cases:
      - numbered/bulleted instruction containing an action verb
      - line beginning with an action verb
      - imperative variants such as "then click ..."
    """
    text = _clean_text(line)
    if not text:
        return False

    remainder = _line_after_marker(text)
    if not remainder:
        return False

    marker = _list_marker(text)
    if marker:
        if re.match(
            r"^(?:the application|the system|the software|the page|the user|"
            r"the tool|the download|the upload|the search|the file|"
            r"this|that|note that|when|after)\b",
            remainder,
            flags=re.IGNORECASE,
        ):
            return False

        match = _ACTION_WORD_RE.search(remainder)
        if not match:
            return False
        # Require the action verb near the front of the instruction. This
        # rejects descriptive list items while still allowing:
        # "1. From the toolbar, click Settings".
        return len(remainder[:match.start()].split()) <= 8

    if re.match(
        r"^(?:"
        + "|".join(re.escape(v) for v in ACTION_VERBS)
        + r")\b",
        remainder,
        flags=re.IGNORECASE,
    ):
        return True

    if re.match(
        r"^(?:now|then|next|finally|first|after that)\s+(?:"
        + "|".join(re.escape(v) for v in ACTION_VERBS)
        + r")\b",
        remainder,
        flags=re.IGNORECASE,
    ):
        return True

    # Common imperative pattern: "In the menu, click Settings."
    verb_match = _ACTION_WORD_RE.search(remainder)
    if verb_match and verb_match.start() <= 28:
        prefix = remainder[:verb_match.start()].strip(" ,:-")
        prefix_first = (
            prefix.split(maxsplit=1)[0].lower()
            if prefix
            else ""
        )

        # Allow short location/context prefixes such as "In the toolbar",
        # but reject declarative/explanatory clauses such as
        # "The application will open" or "The download starts".
        if prefix_first in {
            "the", "a", "an", "this", "that", "these", "those",
            "it", "they", "application", "system", "software",
            "page", "file", "download", "upload", "search",
            "user", "tool",
        }:
            return False

        if len(prefix.split()) <= 5:
            return True

    return False


def _looks_like_continuation(previous: str, current: str) -> bool:
    """
    Detect PDF line wrapping so a target like:
        "1. Click the Configuration"
        "menu."
    remains one action.

    A new explanatory sentence generally starts with a capital letter and the
    previous line already ends with sentence punctuation.
    """
    previous = _clean_text(previous)
    current = _clean_text(current)
    if not previous or not current:
        return False

    if previous.endswith(("-", "/", ":", ",")):
        return True

    # A capitalized explanatory sentence commonly follows an action without
    # PDF punctuation being preserved. Treat these as context, not wrapped
    # action text.
    first_word = current.split(maxsplit=1)[0].lower().strip(".,;:")
    if first_word in {
        "this", "that", "these", "those", "the", "it", "they",
        "this", "the", "it",
    }:
        return False

    if not re.search(r"[.!?;]$", previous):
        if current[:1].islower():
            return True
        # Very short continuation fragments are common after PDF wrapping,
        # but only when the fragment does not look like a new sentence.
        if len(current.split()) <= 5 and not _is_action_line(current):
            return True

    return False


def _extract_instruction_blocks(
    page: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Turn PDF text lines into semantic instruction blocks.

    Each block contains:
      action   -> the actual learner instruction
      context  -> nearby explanatory source text
      lines    -> source lines retained for debugging/auditability

    Crucially, a page heading is NOT blindly removed. If the heading itself is
    an action (for example "1. Click the File menu"), it remains a real step.
    """
    raw_text = page.get("text", "")
    if not raw_text:
        return []

    raw_lines = str(raw_text).splitlines()
    lines = []
    for raw in raw_lines:
        clean = _clean_text(raw)
        if clean:
            lines.append(clean)

    if not lines:
        return []

    blocks: List[List[str]] = []
    current: List[str] = []

    for line in lines:
        starts_new_explicit_step = _list_marker(line) is not None
        is_action = _is_action_line(line)

        if current:
            if starts_new_explicit_step:
                blocks.append(current)
                current = [line]
            elif is_action and not _looks_like_continuation(current[-1], line):
                blocks.append(current)
                current = [line]
            else:
                current.append(line)
        else:
            current = [line]

    if current:
        blocks.append(current)

    heading = _clean_text(page.get("heading"))
    result: List[Dict[str, Any]] = []

    for block_lines in blocks:
        action_index = None
        for index, line in enumerate(block_lines):
            if _is_action_line(line):
                action_index = index
                break

        if action_index is None:
            result.append(
                {
                    "action": None,
                    "context": _clean_text(" ".join(block_lines)),
                    "lines": block_lines,
                }
            )
            continue

        action_parts = [block_lines[action_index]]
        context_parts = block_lines[:action_index]

        previous_action_line = block_lines[action_index]
        for line in block_lines[action_index + 1:]:
            if _looks_like_continuation(previous_action_line, line):
                action_parts.append(line)
                previous_action_line = line
            else:
                context_parts.append(line)

        action = _clean_text(" ".join(action_parts))
        context = _clean_text(" ".join(context_parts))

        # Do not treat the page heading as spoken context when it is simply the
        # title preceding the actual instruction.
        if heading and context == heading:
            context = ""

        result.append(
            {
                "action": action,
                "context": context,
                "lines": block_lines,
            }
        )

    return result


def _extract_action_lines(
    page: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    Backward-compatible view of the new semantic parser.

    The first list contains actual actions; the second contains explanation
    material. Existing callers that rely on this function continue to work.
    """
    actions: List[str] = []
    explanations: List[str] = []

    for block in _extract_instruction_blocks(page):
        action = block.get("action")
        context = _clean_text(block.get("context"))

        if action:
            actions.append(action)
            if context:
                explanations.append(context)
        elif context:
            explanations.append(context)

    return actions, explanations


# ============================================================
# TARGET QUERY
# ============================================================

def _target_query(action: str) -> str:
    text = _clean_text(action)
    if not text:
        return ""

    text = re.sub(r"^\s*\d+[\.\):\-]\s*", "", text).strip()
    text = _remove_urls(text)

    m = re.search(
        r"\b(?:double-click|right-click|click|select|choose|press)\s+"
        r"(?:on\s+)?(?:the\s+|a\s+|an\s+)?"
        r"(.+?)(?:\s+(?:to|for|so|and|before|then)\b|[.,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        candidate = _clean_text(m.group(1))
        if candidate:
            return _shorten(candidate, 100)

    m = re.search(
        r"\b(?:enter|type)\s+(.+?)\s+in(?:to)?\s+"
        r"(?:the\s+)?(.+?)(?:[.,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return _shorten(_clean_text(m.group(2)), 100)

    m = re.search(
        r"\b(?:open|navigate|visit)\s+(?:to\s+)?(?:the\s+)?"
        r"(.+?)(?:\s+and\s+|[.,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        candidate = _clean_text(m.group(1))
        if candidate:
            return _shorten(candidate, 100)

    if re.search(r"\blog[ -]?in\b", text, re.IGNORECASE):
        return "Login button"

    return _shorten(text, 100)


# ============================================================
# PAGE / SCREENSHOT HELPERS
# ============================================================

def _page_number(page: Dict[str, Any], fallback: int) -> int:
    for key in ("page", "page_num", "page_number"):
        if page.get(key) is not None:
            value = _safe_int(page.get(key), -1)
            if value >= 0:
                return value
    return fallback


def _screenshot_page(screenshot: Dict[str, Any]) -> Optional[int]:
    for key in ("page", "page_num", "page_number"):
        if screenshot.get(key) is not None:
            value = _safe_int(screenshot.get(key), -1)
            if value >= 0:
                return value
    return None


def _screenshot_path(screenshot: Dict[str, Any]) -> Optional[str]:
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
    result: Dict[int, List[Dict[str, Any]]] = {}

    for screenshot in screenshots:
        if not isinstance(screenshot, dict):
            continue
        page = _screenshot_page(screenshot)
        if page is None:
            continue
        result.setdefault(page, []).append(screenshot)

    for page_screenshots in result.values():
        page_screenshots.sort(
            key=lambda item: _safe_int(
                item.get("screenshot_index", item.get("index", 0)),
                0,
            )
        )

    return result


# ============================================================
# PDF TEXT TARGET FALLBACK
# ============================================================

def _normalise_words(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    words = page.get("words", [])
    if not isinstance(words, list):
        return []

    result = []
    for word in words:
        if not isinstance(word, (list, tuple, dict)):
            continue

        if isinstance(word, dict):
            text = _clean_text(word.get("text", word.get("word", "")))
            if not text:
                continue
            try:
                x0 = float(word.get("x0", word.get("x", 0)))
                y0 = float(word.get("y0", word.get("y", 0)))
                x1 = float(word.get("x1", x0))
                y1 = float(word.get("y1", y0))
            except Exception:
                continue
        else:
            if len(word) < 5:
                continue
            try:
                x0, y0, x1, y1 = map(float, word[:4])
            except Exception:
                continue
            text = _clean_text(word[4])
            if not text:
                continue

        result.append(
            {"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1}
        )

    return result


def _find_text_target(
    query: str,
    page: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    query = _clean_text(query).lower()
    if not query:
        return None

    words = _normalise_words(page)
    if not words:
        return None

    query_tokens = [
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", query)
        if len(token) > 1
    ]
    if not query_tokens:
        return None

    best = None
    best_score = 0

    for word in words:
        word_text = word["text"].lower()
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
            x0, y0, x1, y1 = (
                word["x0"],
                word["y0"],
                word["x1"],
                word["y1"],
            )
            best = {
                "point": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
                "source": "pdf-text",
                "confidence": min(0.90, 0.50 + score * 0.04),
                "target_name": word["text"],
                "bounding_box": [x0, y0, x1, y1],
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
        value = screenshot.get(key)
        if isinstance(value, list):
            candidates.extend(
                item for item in value if isinstance(item, dict)
            )

    if not candidates:
        return None

    query_tokens = {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", query.lower())
        if len(token) > 1
    }

    best = None
    best_score = 0

    for candidate in candidates:
        text = _clean_text(
            candidate.get(
                "text",
                candidate.get(
                    "label",
                    candidate.get("matched", candidate.get("word", "")),
                ),
            )
        )
        if not text:
            continue

        candidate_tokens = set(
            re.findall(r"[a-zA-Z0-9_]+", text.lower())
        )
        score = len(query_tokens & candidate_tokens)
        if text.lower() == query.lower():
            score += 10

        box = candidate.get("box")
        point = candidate.get("point")

        if point is None and isinstance(box, dict):
            try:
                point = [
                    float(box["x"]) + float(box["width"]) / 2,
                    float(box["y"]) + float(box["height"]) / 2,
                ]
            except Exception:
                point = None

        if point is None:
            continue

        if score > best_score:
            best = {
                "point": list(point),
                "source": "screenshot-text",
                "confidence": min(0.95, 0.55 + score * 0.08),
                "target_name": text,
                "bounding_box": box,
                "annotation": text,
            }
            best_score = score

    return best


def _ocr_screenshot_target(
    query: str,
    screenshot: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not screenshot:
        return None

    path = _screenshot_path(screenshot)
    if not path or not os.path.exists(path):
        return None

    try:
        import pytesseract
        from PIL import Image

        image = Image.open(path)
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
        )
    except Exception:
        return None

    tokens = [
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9_]+", query.lower())
        if len(token) > 1
    ]
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
        score += sum(
            4.0 if tok == low else 1.5 if tok in low else 0
            for tok in tokens
        )

        try:
            conf = float(data.get("conf", [0])[i])
            x, y, w, h = [
                float(data[key][i])
                for key in ("left", "top", "width", "height")
            ]
        except Exception:
            continue

        if conf < 20 or score <= best_score:
            continue

        iw, ih = image.size
        best = {
            "point": [
                (x + w / 2) / max(1, iw),
                (y + h / 2) / max(1, ih),
            ],
            "source": "screenshot-ocr",
            "confidence": min(0.92, 0.45 + score * 0.035),
            "target_name": text,
            "bounding_box": [
                x / max(1, iw),
                y / max(1, ih),
                (x + w) / max(1, iw),
                (y + h) / max(1, ih),
            ],
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
            "point": None,
            "source": "none",
            "confidence": 0.0,
            "target_name": "",
            "bounding_box": None,
            "annotation": None,
        }

    # 1. Structured screenshot candidates.
    if screenshot:
        candidate = _candidate_target(query, screenshot)
        if candidate:
            return candidate

    # 2. Local OCR.
    ocr_target = _ocr_screenshot_target(query, screenshot)
    if ocr_target:
        return ocr_target

    # 3. Optional local vision model.
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
                        "confidence": float(
                            visual.get("confidence", 0.0)
                        ),
                        "target_name": visual.get(
                            "target_name", query
                        ),
                        "bounding_box": visual.get("bounding_box"),
                        "annotation": visual.get("annotation"),
                    }
        except Exception as exc:
            print(
                f"[VISION] Target grounding skipped for "
                f"'{query}': {exc}"
            )

    # 4. PDF text coordinates work for full-page screenshots.
    if screenshot is None or bool(screenshot.get("is_full_page")):
        text_target = _find_text_target(query, page)
        if text_target:
            return text_target

    # Embedded screenshot: convert page-space text coordinates into
    # image-space only if the matched text actually falls inside that image.
    if screenshot and screenshot.get("page_rect"):
        rect = screenshot["page_rect"]
        if isinstance(rect, (list, tuple)) and len(rect) >= 4:
            text_target = _find_text_target(query, page)
            if text_target:
                x0, y0, x1, y1 = [float(value) for value in rect[:4]]
                px, py = text_target["point"][:2]
                if x0 <= px <= x1 and y0 <= py <= y1 and x1 > x0 and y1 > y0:
                    point = [
                        (px - x0) / (x1 - x0),
                        (py - y0) / (y1 - y0),
                    ]
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
                        "target_name": text_target.get(
                            "target_name", query
                        ),
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


# ============================================================
# DIALOGUE / ACTION METADATA
# ============================================================

def _action_type(action: str) -> str:
    text = _clean_text(action).lower()
    if "double-click" in text:
        return "Double-click"
    if "right-click" in text:
        return "Right-click"

    for verb, label in (
        ("click", "Click"),
        ("select", "Select"),
        ("choose", "Choose"),
        ("enter", "Enter"),
        ("type", "Type"),
        ("press", "Press"),
        ("open", "Open"),
        ("navigate", "Navigate"),
        ("search", "Search"),
        ("save", "Save"),
        ("submit", "Submit"),
        ("upload", "Upload"),
        ("download", "Download"),
        ("create", "Create"),
        ("delete", "Delete"),
        ("expand", "Expand"),
        ("collapse", "Collapse"),
    ):
        if re.search(rf"\b{re.escape(verb)}\b", text):
            return label

    if "login" in text or "log in" in text:
        return "Login"
    return "Follow"


def _make_dialogue(
    action: str,
    target_name: str = "",
) -> Dict[str, str]:
    clean = re.sub(
        r"^\s*\d+[\.\):\-]\s*",
        "",
        _clean_text(action),
    ).strip(" .;,")
    action_type = _action_type(clean)
    target = _clean_text(target_name)

    if target:
        current = f"{action_type} {target}"
    else:
        current = _shorten(clean, 90)

    if (
        action_type
        in {
            "Click",
            "Double-click",
            "Right-click",
            "Select",
            "Choose",
            "Press",
        }
        and target
    ):
        brief = f"Use the highlighted {target} to continue."
    elif action_type in {"Enter", "Type"} and target:
        brief = f"Enter the required value in the highlighted {target}."
    elif action_type == "Open" and target:
        brief = f"Open the highlighted {target}."
    elif action_type == "Search":
        brief = (
            "Use the highlighted search control to find the required item."
        )
    else:
        brief = _shorten(clean, 120)

    return {
        "title": "What to do now",
        "action_type": action_type,
        "current_action": current,
        "brief": brief,
        "intro": (
            "Follow the highlighted control and continue the workflow."
        ),
    }


# ============================================================
# CONTEXTUAL NARRATION
# ============================================================

def _select_narration_context(
    action: str,
    context: str,
    page_text: str,
) -> str:
    """
    Supply only nearby, source-derived context to the narration model.

    The model is instructed not to invent purpose. Context is capped so a
    large page cannot turn into a generic summary.
    """
    action_clean = _clean_spoken_narration(action)
    context_clean = _clean_text(_remove_urls(context))
    page_clean = _clean_text(_remove_urls(page_text))

    # When an adjacent context block exists, use it preferentially. Page-wide
    # text is the fallback, not an automatic append, because feeding the model
    # the whole page can dilute the exact reason behind the current action.
    candidates = [context_clean] if context_clean else [page_clean]

    selected: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", candidate)
        for sentence in sentences:
            sentence = _clean_text(sentence)
            if (
                len(sentence) >= 12
                and sentence.lower() != action_clean.lower()
            ):
                selected.append(sentence)
            if len(" ".join(selected)) >= 1200:
                break

        if selected:
            break

    return _shorten(" ".join(selected), 1600)


def _deterministic_narration(
    action: str,
    context: str,
) -> str:
    """
    Local fallback that remains useful when Ollama is unavailable.

    It never invents purpose. When source context contains a concise
    explanatory sentence, that sentence is appended verbatim (minus URLs).
    """
    spoken_action = _clean_spoken_narration(action)
    if not spoken_action:
        return "Continue with the next step."

    spoken_action = re.sub(
        r"^\s*(?:now|then|next|finally|first)\s+",
        "",
        spoken_action,
        flags=re.IGNORECASE,
    )
    spoken_action = spoken_action[:1].upper() + spoken_action[1:]
    spoken_action = spoken_action.rstrip(" .;,") + "."

    clean_context = _clean_text(_remove_urls(context))
    if not clean_context:
        return spoken_action

    if re.search(
        r"\b(?:to|so that|which|this|that|these|those|allows|opens|"
        r"displays|shows|used for|lets you|enables)\b",
        clean_context,
        flags=re.IGNORECASE,
    ):
        sentence = re.split(
            r"(?<=[.!?])\s+",
            clean_context,
        )[0].strip()
        sentence = sentence.rstrip(" .;,") + "."
        if sentence.lower() not in {
            spoken_action.lower(),
            spoken_action.rstrip(".").lower(),
        }:
            candidate = _clean_spoken_narration(
                f"{spoken_action} {sentence}"
            )
            if 8 <= len(candidate.split()) <= 60:
                return candidate

    return spoken_action


def _parse_narration_response(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""

    text = text.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _clean_spoken_narration(
                parsed.get("narration", "")
            )
        if isinstance(parsed, str):
            return _clean_spoken_narration(parsed)
    except Exception:
        pass

    # Recover a JSON object that may have been wrapped in extra prose.
    match = re.search(
        r'\{\s*"narration"\s*:\s*"(?P<narration>(?:\\.|[^"])*)"\s*\}',
        text,
        flags=re.DOTALL,
    )
    if match:
        try:
            decoded = json.loads(
                '"' + match.group("narration") + '"'
            )
            return _clean_spoken_narration(decoded)
        except Exception:
            pass

    return _clean_spoken_narration(text)


def _ollama_narration(
    action: str,
    context: str,
    page_text: str,
) -> str:
    global _OLLAMA_NARRATION_DISABLED

    if not ENABLE_OLLAMA_NARRATION or _OLLAMA_NARRATION_DISABLED:
        return ""

    action_clean = _clean_spoken_narration(action)
    context_clean = _select_narration_context(
        action,
        context,
        page_text,
    )

    prompt = f"""
You are a professional instructor creating voice narration for enterprise
software training.

SOURCE ACTION:
{action_clean}

NEARBY SOURCE CONTEXT:
{context_clean or "(No additional explanatory context is available.)"}

Write the narration for this step.

Requirements:
- Explain what the learner is doing.
- Include the purpose only when the source context supports it.
- Preserve exact UI labels, product names, commands, and technical terms.
- Do not invent details, results, reasons, or business meaning.
- Do not speak or quote URLs. URLs belong in captions/metadata only.
- Do not mention the PDF, page number, screenshot, source text, metadata, or
  "the learner".
- Do not turn a simple UI action into a generic motivational sentence.
- Make it sound like a human instructor guiding a real task.
- Use one natural sentence, or two closely connected sentences.
- Target approximately 18-45 words, but prefer clarity over the word count.

Return ONLY valid JSON:
{{"narration":"..."}}
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        generated = _parse_narration_response(
            data.get("response", "")
            if isinstance(data, dict)
            else ""
        )

        # Guardrails: reject unusable outputs and fall back cleanly.
        if not generated:
            return ""
        if _extract_urls(generated):
            return ""
        if len(generated.split()) > 70:
            return ""
        if re.search(
            r"\b(?:page\s+\d+|screenshot|metadata|source text)\b",
            generated,
            flags=re.IGNORECASE,
        ):
            return ""
        return generated

    except requests.exceptions.RequestException as exc:
        _OLLAMA_NARRATION_DISABLED = True
        print(
            f"[NARRATION] Local Ollama unavailable at {OLLAMA_URL}: {exc}. "
            "Using deterministic narration fallback for the remainder of this run."
        )
        return ""
    except Exception as exc:
        print(f"[NARRATION] Ollama response rejected: {exc}")
        return ""


def _make_action_narration(
    action: str,
    context: str = "",
    page_text: str = "",
) -> str:
    action_clean = _clean_spoken_narration(action)
    context_clean = _clean_text(context)

    if not action_clean:
        return "Continue with the next step."

    cache_key = (
        action_clean.lower(),
        _clean_text(context_clean).lower(),
        _clean_text(page_text).lower()[:1000],
    )
    cached = _NARRATION_CACHE.get(cache_key)
    if cached:
        return cached

    generated = _ollama_narration(
        action_clean,
        context_clean,
        page_text,
    )

    if not generated:
        generated = _deterministic_narration(
            action_clean,
            context_clean,
        )

    generated = _clean_spoken_narration(generated)
    if not generated:
        generated = _deterministic_narration(
            action_clean,
            context_clean,
        )

    _NARRATION_CACHE[cache_key] = generated
    return generated


# ============================================================
# CAPTION / URL METADATA
# ============================================================

def _relevant_urls(
    action: str,
    context: str,
    page: Dict[str, Any],
) -> List[str]:
    urls: List[str] = []

    for source in (action, context):
        for url in _extract_urls(source):
            if url not in urls:
                urls.append(url)

    page_urls = page.get("urls", [])
    if isinstance(page_urls, list):
        for url in page_urls:
            url = _clean_text(url)
            if not url or url in urls or not _URL_RE.search(url):
                continue
            # Add hyperlink-only page URLs when there is one obvious page
            # destination, or when the action is clearly about navigation.
            navigation_words = (
                "open",
                "navigate",
                "visit",
                "url",
                "link",
                "website",
                "documentation",
                "browser",
            )
            action_lower = _clean_text(action).lower()
            if len(page_urls) == 1 or any(
                word in action_lower for word in navigation_words
            ):
                urls.append(url)

    return urls[:5]


def _caption_with_urls(
    narration: str,
    urls: List[str],
) -> str:
    narration = _clean_spoken_narration(narration)
    if not urls:
        return narration
    return narration + "\n" + "\n".join(
        f"URL: {url}" for url in urls
    )


def _make_theory_bullets(
    explanation_lines: List[str],
) -> List[str]:
    bullets = []
    for line in explanation_lines:
        line = _remove_urls(_clean_text(line))
        if len(line) < 12:
            continue
        bullets.append(_shorten(line, 180))
        if len(bullets) >= 6:
            break
    return bullets


# ============================================================
# SECTION BUILDING
# ============================================================

def _page_has_action(page: Dict[str, Any]) -> bool:
    actions, _ = _extract_action_lines(page)
    return bool(actions)


def _build_sections(
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    content_pages = [
        page
        for page in pages
        if isinstance(page, dict) and not page.get("skip_reason")
    ]

    if not content_pages:
        content_pages = [
            page for page in pages if isinstance(page, dict)
        ]

    if not content_pages:
        return []

    sections = []
    current = None

    for page in content_pages:
        heading = _clean_text(page.get("heading"))
        if heading or current is None:
            current = {
                "title": heading or f"Section {len(sections) + 1}",
                "pages": [],
            }
            sections.append(current)

        current["pages"].append(page)

    # Preserve the existing portal concept of multiple organizational
    # sections. This fallback is deliberately unchanged in behavior.
    if len(sections) <= 1 and len(content_pages) > 6:
        sections = []
        current = None

        for page in content_pages:
            if current is None:
                current = {
                    "title": _clean_text(page.get("heading"))
                    or f"Section {len(sections) + 1}",
                    "pages": [],
                }
                sections.append(current)

            current["pages"].append(page)

            if _page_has_action(page):
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
    pages = data.get("pages", [])
    screenshots = data.get("screenshots", [])

    if not isinstance(pages, list):
        pages = []
    if not isinstance(screenshots, list):
        screenshots = []

    _progress(
        progress_callback,
        2,
        "Preparing document structure",
    )

    shots_by_page = _build_screenshot_map(screenshots)

    _progress(
        progress_callback,
        8,
        f"Mapped {len(screenshots)} embedded visual regions",
    )

    sections_raw = _build_sections(pages)

    sections_out: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    step_id = 0

    total_sections = max(1, len(sections_raw))

    for section_index, section in enumerate(sections_raw):
        section_step_ids: List[int] = []
        action_bullets: List[str] = []
        explanation_lines: List[str] = []

        # Keep semantic context with every action instead of throwing it away
        # before narration is generated.
        section_actions: List[
            Tuple[Dict[str, Any], str, str]
        ] = []

        for page in section.get("pages", []):
            if not isinstance(page, dict):
                continue

            blocks = _extract_instruction_blocks(page)

            for block_index, block in enumerate(blocks):
                action = _clean_text(block.get("action"))
                context_parts: List[str] = []

                own_context = _clean_text(block.get("context"))
                if own_context:
                    context_parts.append(own_context)

                # Preserve nearby explanatory prose for narration. Prefer one
                # adjacent non-action block so a reason/purpose paragraph can
                # enrich the instruction without dumping the whole page into
                # every LLM prompt.
                if block_index > 0:
                    previous = blocks[block_index - 1]
                    if not _clean_text(previous.get("action")):
                        previous_context = _clean_text(
                            previous.get("context")
                        )
                        if (
                            previous_context
                            and previous_context
                            != _clean_text(page.get("heading"))
                        ):
                            context_parts.append(previous_context)

                if block_index + 1 < len(blocks):
                    following = blocks[block_index + 1]
                    if not _clean_text(following.get("action")):
                        following_context = _clean_text(
                            following.get("context")
                        )
                        if following_context:
                            context_parts.append(following_context)

                context = _clean_text(" ".join(dict.fromkeys(context_parts)))

                if action:
                    section_actions.append(
                        (
                            page,
                            action,
                            context,
                        )
                    )
                    if context:
                        explanation_lines.append(context)
                elif own_context:
                    explanation_lines.append(own_context)

        for page, action, context in section_actions:
            page_number = _page_number(page, 1)
            page_shots = shots_by_page.get(page_number, [])

            chosen_shot = None
            grounding = None
            best_score = -1.0

            # First use cheap structured/OCR/PDF targeting to choose the image
            # that actually contains the control.
            for candidate_shot in page_shots:
                candidate_grounding = _ground_action(
                    action,
                    page,
                    candidate_shot,
                    allow_vision=False,
                )
                score = float(
                    candidate_grounding.get("confidence", 0.0)
                )

                if (
                    candidate_grounding.get("point") is not None
                    and score > best_score
                ):
                    chosen_shot = candidate_shot
                    grounding = candidate_grounding
                    best_score = score

            # If no target was found, keep the largest screenshot as the
            # application image and spend vision effort once.
            if chosen_shot is None and page_shots:
                def _area(shot):
                    path = _screenshot_path(shot)
                    try:
                        from PIL import Image
                        with Image.open(path) as image:
                            return image.width * image.height
                    except Exception:
                        return 0

                chosen_shot = max(page_shots, key=_area)
                grounding = _ground_action(
                    action,
                    page,
                    chosen_shot,
                    allow_vision=ENABLE_OLLAMA,
                )

            if grounding is None:
                grounding = _ground_action(
                    action,
                    page,
                    None,
                    allow_vision=False,
                )

            screenshot_path = (
                _screenshot_path(chosen_shot)
                if chosen_shot
                else None
            )

            page_text = _clean_text(page.get("text", ""))
            narration = _make_action_narration(
                action,
                context=context,
                page_text=page_text,
            )

            urls = _relevant_urls(
                action,
                context,
                page,
            )
            caption = _caption_with_urls(
                narration,
                urls,
            )

            step_id += 1

            point = grounding.get("point")
            source = grounding.get("source", "none")
            confidence = grounding.get("confidence", 0.0)
            target_name = grounding.get("target_name", "")
            bounding_box = grounding.get("bounding_box")
            annotation = grounding.get("annotation")

            step = {
                "id": step_id,
                "section_id": section_index + 1,
                "page_num": page_number,
                "page_width": float(page.get("width") or 0),
                "page_height": float(page.get("height") or 0),
                "kind": "action",
                "title": _shorten(action, 80),

                # narration is always URL-free.
                "narration": narration,
                "tts_narration": narration,

                # Caption may carry URL metadata.
                "caption": caption,
                "urls": urls,
                "source_context": _shorten(_remove_urls(context), 900),

                "screenshot": screenshot_path,
                "is_full_page": screenshot_path is None,

                "cursor": point,
                "targets": [point] if point is not None else [],
                "cursor_source": source,
                "cursor_confidence": confidence,
                "cursor_target_name": target_name,
                "cursor_annotation": annotation,
                "cursor_bounding_box": bounding_box,

                "action_type": _action_type(action),
                "dialogue": _make_dialogue(action, target_name),
            }

            steps.append(step)
            section_step_ids.append(step_id)
            action_bullets.append(_shorten(action, 140))

        theory_bullets = _make_theory_bullets(
            explanation_lines
        )

        section_title = _clean_text(section.get("title")) or (
            f"Section {section_index + 1}"
        )

        sections_out.append(
            {
                "id": section_index + 1,
                "title": _shorten(section_title, 90),
                "theory_bullets": theory_bullets,
                "action_bullets": action_bullets,
                "step_ids": section_step_ids,
                "action_count": len(section_step_ids),
                "has_video": bool(section_step_ids),
            }
        )

        _progress(
            progress_callback,
            int(
                10
                + 80
                * (
                    (section_index + 1)
                    / float(total_sections)
                )
            ),
            (
                f"Prepared section "
                f"{section_index + 1} of {total_sections}"
            ),
        )

    # ========================================================
    # Safety fallback
    # ========================================================

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
        sections_out.append(fallback_section)

        for page_index, page in enumerate(pages):
            if not isinstance(page, dict):
                continue

            page_text = _clean_text(page.get("text", ""))
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
            chosen_shot = page_shots[0] if page_shots else None
            screenshot_path = (
                _screenshot_path(chosen_shot)
                if chosen_shot
                else None
            )

            description = _shorten(
                _remove_urls(page_text),
                180,
            )
            narration = _deterministic_narration(
                f"Review the content on page {page_number}",
                description,
            )
            urls = _relevant_urls(
                page_text,
                "",
                page,
            )
            caption = _caption_with_urls(
                narration,
                urls,
            )

            step_id += 1

            step = {
                "id": step_id,
                "section_id": fallback_section["id"],
                "page_num": page_number,
                "page_width": float(page.get("width") or 0),
                "page_height": float(page.get("height") or 0),
                "kind": "action",
                "title": f"Review page {page_number}",
                "narration": narration,
                "tts_narration": narration,
                "caption": caption,
                "urls": urls,
                "source_context": description,
                "screenshot": screenshot_path,
                "is_full_page": screenshot_path is None,
                "cursor": None,
                "targets": [],
                "cursor_source": "none",
                "cursor_confidence": 0.0,
                "cursor_target_name": "",
                "cursor_annotation": None,
                "cursor_bounding_box": None,
                "action_type": "Follow",
                "dialogue": _make_dialogue(
                    f"Review the content on page {page_number}"
                ),
            }

            steps.append(step)
            fallback_section["step_ids"].append(step_id)
            fallback_section["action_bullets"].append(
                f"Review page {page_number}"
            )
            fallback_section["action_count"] += 1

            if len(steps) >= 100:
                break

    # ========================================================
    # Title
    # ========================================================

    title = None

    for page in pages:
        if not isinstance(page, dict):
            continue
        heading = _clean_text(page.get("heading"))
        if heading:
            title = heading
            break

    if not title:
        title = _clean_text(data.get("title"))

    if not title:
        title = _clean_text(data.get("filename"))
        if title.lower().endswith(".pdf"):
            title = title[:-4]

    if not title:
        title = "Software Training Tutorial"

    # ========================================================
    # Overview
    # ========================================================

    overview_parts = []

    for page in pages[:5]:
        if not isinstance(page, dict):
            continue
        text = _remove_urls(_clean_text(page.get("text", "")))
        if text:
            overview_parts.append(text)

    overview_source = _shorten(
        " ".join(overview_parts),
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

    plan = {
        "title": title,
        "description": description,
        "overview": description,
        "narration_language": narration_language or "en-us",
        "sections": sections_out,
        "steps": steps,
        "scene_count": len(steps),
        "action_count": sum(
            1 for step in steps
            if step.get("kind") == "action"
        ),
        "total_pages": len(pages),
        "total_screenshots": len(screenshots),
        "visual_source": (
            "embedded PDF screenshots with full-page fallback"
        ),
        "ai_mode": (
            "local-deterministic"
            if not ENABLE_OLLAMA
            else "ollama-optional"
        ),
        "narration_mode": (
            "ollama-contextual-with-local-fallback"
            if ENABLE_OLLAMA_NARRATION
            else "deterministic-local"
        ),
    }

    _progress(
        progress_callback,
        100,
        f"Tutorial plan ready: {len(steps)} action steps",
    )

    print(
        f"[AI] Plan created: {len(sections_out)} sections, "
        f"{len(steps)} action steps, {len(screenshots)} screenshots"
    )

    return plan


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

generate_tutorial = build_tutorial_plan
