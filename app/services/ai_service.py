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
        if not _ollama_host.startswith(("http://", "https://")):
            _ollama_host = "http://" + _ollama_host
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

MIN_OCR_TARGET_CONFIDENCE = 0.72
MIN_VISION_TARGET_CONFIDENCE = 0.82
TARGET_AMBIGUITY_MARGIN = 0.10

_TARGET_TYPE_RE = re.compile(
    r"\b(button|field|textbox|input|menu|tab|link|checkbox|radio|"
    r"dropdown|combobox|list|icon|toolbar|panel|window|dialog|address bar)\b",
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

        continuation_open = True
        previous_action_line = block_lines[action_index]
        for line in block_lines[action_index + 1:]:
            if continuation_open and _looks_like_continuation(previous_action_line, line):
                action_parts.append(line)
                previous_action_line = line
            else:
                continuation_open = False
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

def _target_type(action: str) -> str:
    match = _TARGET_TYPE_RE.search(_clean_text(action))
    return match.group(1).lower() if match else ""


def _target_query(action: str) -> str:
    original = _clean_text(action)
    text = original
    if not text:
        return ""

    text = re.sub(r"^\s*\d+[\.\):\-]\s*", "", text).strip()
    had_url = bool(_extract_urls(text))
    text = _remove_urls(text)

    patterns = (
        r"\b(?:double-click|right-click|click|select|choose|press)\s+"
        r"(?:on\s+)?(?:the\s+|a\s+|an\s+)?"
        r"(.+?)(?:\s+(?:to|for|so|and|before|then|using)\b|[.,;]|$)",
        r"\b(?:enter|type|fill)\s+.+?\s+in(?:to)?\s+"
        r"(?:the\s+)?(.+?)(?:[.,;]|$)",
        r"\b(?:open|navigate|visit)\s+(?:to\s+)?(?:the\s+)?"
        r"(.+?)(?:\s+and\s+|[.,;]|$)",
    )

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            candidate = _clean_text(match.group(1))
            if candidate:
                candidate = _TARGET_TYPE_RE.sub(" ", candidate)
                candidate = _clean_text(candidate)
                if candidate:
                    return _shorten(candidate, 100)

    if re.search(r"\blog[ -]?in\b", text, re.IGNORECASE):
        return "Login button"

    common_controls = {
        "save": "Save button",
        "submit": "Submit button",
        "delete": "Delete button",
        "create": "Create button",
        "upload": "Upload button",
        "download": "Download button",
    }
    first = re.match(r"^(save|submit|delete|create|upload|download)\b", text, re.IGNORECASE)
    if first:
        return common_controls[first.group(1).lower()]

    # A URL-only navigation step has no reliable UI target in a static
    # screenshot. Do not convert the URL into a broad word like "Open" and
    # then let OCR/vision guess a random control.
    if had_url:
        navigation_only = re.sub(r"\b(?:open|visit|navigate|go|to)\b", " ", text, flags=re.IGNORECASE)
        if not _clean_text(navigation_only).strip(" .;,:-"):
            return ""

    return _shorten(text, 100)


def _interaction_for_action(action: str, action_type: str, has_target: bool) -> str:
    if not has_target:
        return "none"
    text = _clean_text(action).lower()
    if action_type == "Press" and re.search(r"\b(?:enter|return|escape|esc|tab|space|shift|ctrl|control|alt|backspace|delete)\b", text):
        return "none"
    if action_type in {
        "Click", "Double-click", "Right-click", "Select", "Choose",
        "Open", "Save", "Submit", "Create", "Delete", "Expand",
        "Collapse", "Login",
    }:
        return "click" if action_type != "Double-click" and action_type != "Right-click" else {"Double-click":"double_click", "Right-click":"right_click"}[action_type]
    if action_type in {"Enter", "Type", "Search"}:
        return "focus"
    return "none"


def _target_spec(action: str) -> Dict[str, str]:
    return {
        "query": _target_query(action),
        "target_type": _target_type(action),
        "action_type": _action_type(action),
    }


def _action_complexity(action: str) -> str:
    text = _clean_text(action)
    verbs = re.findall(_ACTION_WORD_RE, text)
    word_count = len(text.split())
    if re.search(r"\b(?:drag|drop)\b", text, re.IGNORECASE):
        return "drag_drop"
    if len(verbs) >= 2 or word_count >= 24:
        return "complex"
    if re.search(r"\b(?:enter|type|fill|search)\b", text, re.IGNORECASE):
        return "form_entry"
    if re.search(r"\b(?:open|navigate|visit|launch)\b", text, re.IGNORECASE):
        return "navigation"
    return "simple"


def _narration_word_range(action: str) -> Tuple[int, int]:
    complexity = _action_complexity(action)
    return {
        "simple": (6, 18),
        "navigation": (8, 22),
        "form_entry": (10, 26),
        "complex": (18, 40),
        "drag_drop": (10, 28),
    }[complexity]


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
    """Find a strong page-text target without guessing."""
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

    ranked = []
    for word in words:
        word_text = word["text"].lower()
        exact = query == word_text
        substring = query in word_text or word_text in query
        overlap = sum(
            1 for token in query_tokens
            if token == word_text or token in word_text
        )
        coverage = overlap / float(len(query_tokens))
        if not exact and not substring and overlap == 0:
            continue

        if exact:
            confidence = 0.94
        elif len(query_tokens) == 1 and substring:
            confidence = 0.78
        elif coverage >= 0.75:
            confidence = 0.74
        else:
            confidence = 0.58

        x0, y0, x1, y1 = (
            word["x0"], word["y0"],
            word["x1"], word["y1"],
        )
        ranked.append({
            "point": [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
            "source": "pdf-text",
            "confidence": confidence,
            "target_name": word["text"],
            "bounding_box": [x0, y0, x1, y1],
            "annotation": word["text"],
        })

    if not ranked:
        return None

    ranked.sort(key=lambda item: item["confidence"], reverse=True)
    best = ranked[0]
    second = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
    if best["confidence"] < MIN_OCR_TARGET_CONFIDENCE:
        return None
    if second and best["confidence"] - second < TARGET_AMBIGUITY_MARGIN:
        return None
    return best


# ============================================================
# SCREENSHOT TARGET FALLBACK
# ============================================================

def _candidate_point_from_box(box: Any) -> Optional[List[float]]:
    if isinstance(box, dict):
        try:
            return [
                float(box["x"]) + float(box["width"]) / 2.0,
                float(box["y"]) + float(box["height"]) / 2.0,
            ]
        except Exception:
            return None
    return None


def _token_similarity(query: str, candidate: str) -> float:
    q_tokens = [t for t in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(t) > 1]
    c_tokens = [t for t in re.findall(r"[a-zA-Z0-9_]+", candidate.lower()) if len(t) > 1]
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = sum(1 for token in q_tokens if any(token == ct or token in ct or ct in token for ct in c_tokens))
    coverage = overlap / float(len(q_tokens))
    sequence = 1.0 if query.lower() == candidate.lower() else 0.0
    contains = 1.0 if query.lower() in candidate.lower() or candidate.lower() in query.lower() else 0.0
    return min(1.0, 0.60 * coverage + 0.25 * contains + 0.15 * sequence)


def _build_candidate(
    text: str,
    point: Any,
    box: Any,
    source: str,
    confidence: float,
    query: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if point is None:
        point = _candidate_point_from_box(box)
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        point = [float(point[0]), float(point[1])]
    except Exception:
        return None
    result = {
        "point": point,
        "source": source,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "target_name": _clean_text(text) or query,
        "bounding_box": box,
        "annotation": _clean_text(text) or query,
    }
    if extra:
        result.update(extra)
    return result


def _candidate_target(query: str, screenshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = []
    for key in ("candidates", "targets", "vision_words", "words", "text_regions"):
        value = screenshot.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    if not candidates:
        return None

    iw = ih = 1.0
    path = _screenshot_path(screenshot)
    if path and os.path.exists(path):
        try:
            from PIL import Image
            with Image.open(path) as image:
                iw, ih = float(max(1, image.width)), float(max(1, image.height))
        except Exception:
            pass

    ranked = []
    for candidate in candidates:
        text = _clean_text(candidate.get("text", candidate.get("label", candidate.get("matched", candidate.get("word", "")))))
        if not text:
            continue
        point = candidate.get("point")
        box = candidate.get("box")
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                px, py = float(point[0]), float(point[1])
                point = [px / iw, py / ih] if px > 1.5 or py > 1.5 else [px, py]
            except Exception:
                point = None
        if isinstance(box, dict):
            try:
                bx, by = float(box["x"]), float(box["y"])
                bw, bh = float(box["width"]), float(box["height"])
                box = [bx / iw, by / ih, (bx + bw) / iw, (by + bh) / ih] if max(bx + bw, by + bh) > 1.5 else [bx, by, bx + bw, by + bh]
            except Exception:
                box = None
        sim = _token_similarity(query, text)
        score = sim * 0.88
        if text.lower() == query.lower():
            score = 1.0
        elif query.lower() in text.lower() or text.lower() in query.lower():
            score = max(score, 0.90)
        result = _build_candidate(text, point, box, "screenshot-text", score, query)
        if result and 0.0 <= result["point"][0] <= 1.0 and 0.0 <= result["point"][1] <= 1.0:
            ranked.append(result)

    if not ranked:
        return None
    ranked.sort(key=lambda item: item["confidence"], reverse=True)
    best = ranked[0]
    second = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
    if best["confidence"] < MIN_OCR_TARGET_CONFIDENCE:
        return None
    if second and best["confidence"] - second < TARGET_AMBIGUITY_MARGIN:
        best["accepted"] = False
        best["target_status"] = "ambiguous"
        best["ambiguity_margin"] = best["confidence"] - second
        return best
    best["accepted"] = True
    best["target_status"] = "verified"
    best["ambiguity_margin"] = best["confidence"] - second if second else best["confidence"]
    return best


def _ocr_screenshot_target(query: str, screenshot: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not screenshot:
        return None
    path = _screenshot_path(screenshot)
    if not path or not os.path.exists(path):
        return None

    try:
        import pytesseract
        from PIL import Image
        image = Image.open(path).convert("RGB")
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return None

    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    n = len(data.get("text", []))
    for i in range(n):
        text = _clean_text(data["text"][i])
        if not text:
            continue
        try:
            conf = float(data.get("conf", [0])[i])
            if conf < 20:
                continue
            x = float(data["left"][i]); y = float(data["top"][i])
            w = float(data["width"][i]); h = float(data["height"][i])
            key = (str(data.get("block_num", [0])[i]), str(data.get("par_num", [0])[i]), str(data.get("line_num", [0])[i]))
        except Exception:
            continue
        groups.setdefault(key, []).append({"text": text, "conf": conf, "x": x, "y": y, "w": w, "h": h})

    iw, ih = image.size
    ranked = []
    for tokens in groups.values():
        tokens.sort(key=lambda item: item["x"])
        line_text = _clean_text(" ".join(item["text"] for item in tokens))
        if not line_text:
            continue
        score = _token_similarity(query, line_text)
        if query.lower() == line_text.lower():
            score = 1.0
        if score < 0.45:
            continue
        avg_conf = sum(item["conf"] for item in tokens) / len(tokens)
        score = min(1.0, score * 0.78 + (avg_conf / 100.0) * 0.22)
        x0 = min(item["x"] for item in tokens); y0 = min(item["y"] for item in tokens)
        x1 = max(item["x"] + item["w"] for item in tokens); y1 = max(item["y"] + item["h"] for item in tokens)
        ranked.append(_build_candidate(
            line_text,
            [(x0 + x1) / (2 * iw), (y0 + y1) / (2 * ih)],
            [x0 / iw, y0 / ih, x1 / iw, y1 / ih],
            "screenshot-ocr",
            score,
            query,
            {"ocr_confidence": avg_conf},
        ))

    ranked = [item for item in ranked if item]
    if not ranked:
        return None
    ranked.sort(key=lambda item: item["confidence"], reverse=True)
    best = ranked[0]
    second = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
    if best["confidence"] < MIN_OCR_TARGET_CONFIDENCE:
        return None
    if second and best["confidence"] - second < TARGET_AMBIGUITY_MARGIN:
        best["accepted"] = False
        best["target_status"] = "ambiguous"
        best["ambiguity_margin"] = best["confidence"] - second
    else:
        best["accepted"] = True
        best["target_status"] = "verified"
        best["ambiguity_margin"] = best["confidence"] - second if second else best["confidence"]
    return best


def _vision_candidate_is_sane(visual: Dict[str, Any], query: str) -> bool:
    if not visual.get("found") or not visual.get("click_point"):
        return False
    try:
        confidence = float(visual.get("confidence", 0.0))
    except Exception:
        return False
    if confidence < MIN_VISION_TARGET_CONFIDENCE:
        return False
    point = visual.get("click_point")
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return False
    try:
        px, py = float(point[0]), float(point[1])
    except Exception:
        return False
    if not (0.0 <= px <= 1.0 and 0.0 <= py <= 1.0):
        return False
    target_name = _clean_text(visual.get("target_name", ""))
    if target_name and _token_similarity(query, target_name) < 0.30 and confidence < 0.94:
        return False
    box = visual.get("bounding_box")
    if box is not None and isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            x0, y0, x1, y1 = [float(v) for v in box[:4]]
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                return False
            if not (x0 <= px <= x1 and y0 <= py <= y1):
                return False
        except Exception:
            return False
    return True


def _ground_action(
    action: str,
    page: Dict[str, Any],
    screenshot: Optional[Dict[str, Any]],
    allow_vision: bool = True,
) -> Dict[str, Any]:
    spec = _target_spec(action)
    query = spec["query"]
    base = {
        "point": None,
        "source": "none",
        "confidence": 0.0,
        "target_name": query,
        "bounding_box": None,
        "annotation": None,
        "accepted": False,
        "target_status": "unresolved",
        "target_query": query,
        "target_type": spec["target_type"],
    }
    if not query:
        return base

    if screenshot:
        candidate = _candidate_target(query, screenshot)
        if candidate and candidate.get("accepted"):
            candidate.update({"target_query": query, "target_type": spec["target_type"]})
            return candidate

        ocr_target = _ocr_screenshot_target(query, screenshot)
        if ocr_target and ocr_target.get("accepted"):
            ocr_target.update({"target_query": query, "target_type": spec["target_type"]})
            return ocr_target

        if ENABLE_OLLAMA and allow_vision:
            try:
                from app.services.local_vision import analyze_image
                path = _screenshot_path(screenshot)
                if path:
                    cache_key = (os.path.abspath(path), query.lower(), spec["target_type"])
                    visual = _VISION_CACHE.get(cache_key)
                    if visual is None:
                        visual = analyze_image(
                            path,
                            query,
                            target_type=spec["target_type"],
                            action=action,
                        )
                        _VISION_CACHE[cache_key] = visual
                    if _vision_candidate_is_sane(visual, query):
                        return {
                            **base,
                            "point": visual.get("click_point"),
                            "source": visual.get("source", "vision"),
                            "confidence": float(visual.get("confidence", 0.0)),
                            "target_name": visual.get("target_name", query),
                            "bounding_box": visual.get("bounding_box"),
                            "annotation": visual.get("annotation"),
                            "accepted": True,
                            "target_status": "verified",
                        }
            except Exception as exc:
                print(f"[VISION] Target grounding skipped for '{query}': {exc}")

    if screenshot is None or bool(screenshot.get("is_full_page")):
        text_target = _find_text_target(query, page)
        if text_target:
            text_target.update({
                "accepted": True,
                "target_status": "verified",
                "target_query": query,
                "target_type": spec["target_type"],
            })
            return text_target

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
                        **base,
                        "point": point,
                        "source": "embedded-pdf-text",
                        "confidence": float(text_target.get("confidence", 0.0)),
                        "target_name": text_target.get("target_name", query),
                        "bounding_box": bounding_box,
                        "annotation": text_target.get("annotation"),
                        "accepted": True,
                        "target_status": "verified",
                    }

    return base


# ============================================================
# DIALOGUE / ACTION METADATA
# ============================================================

def _action_type(action: str) -> str:
    text = _clean_text(action).lower()
    # Preserve keyboard semantics before scanning contained words. Otherwise
    # "Press Enter" is incorrectly classified as an Enter/focus action.
    if re.match(r"^press\b", text, flags=re.IGNORECASE):
        return "Press"
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
    narration: str = "",
    context: str = "",
) -> Dict[str, str]:
    clean = re.sub(
        r"^\s*\d+[\.\):\-]\s*",
        "",
        _clean_text(action),
    ).strip(" .;,")
    action_type = _action_type(clean)
    target = _clean_text(target_name)

    current = clean or "Continue with the current step."
    if target and action_type not in {"Follow", "Login"}:
        current = f"{action_type}: {target}"

    brief = _clean_spoken_narration(narration) or clean
    purpose = _clean_text(_remove_urls(context))
    if len(purpose.split()) > 24:
        purpose = _shorten(purpose, 180)

    return {
        "title": "Current step",
        "action_type": action_type,
        "current_action": current,
        "brief": brief,
        "purpose": purpose,
        "intro": "Follow the highlighted step and continue the workflow.",
    }


# ============================================================
# CONTEXTUAL NARRATION
# ============================================================

def _context_sentences(context: str) -> List[str]:
    clean = _clean_text(_remove_urls(context))
    if not clean:
        return []
    result = []
    for sentence in re.split(r"(?<=[.!?])\s+", clean):
        sentence = _clean_text(sentence)
        if len(sentence.split()) >= 5:
            result.append(sentence)
    return result


def _select_narration_context(
    action: str,
    context: str,
    page_text: str,
) -> str:
    action_tokens = {
        token for token in re.findall(r"[a-zA-Z0-9_]+", _remove_urls(action).lower())
        if len(token) > 2
    }
    candidates = _context_sentences(context)
    if not candidates:
        # Page-wide fallback is deliberately conservative: only use sentences
        # that share meaningful terms with the action.
        for sentence in _context_sentences(page_text):
            if sentence.lower() == _clean_spoken_narration(action).lower():
                continue
            tokens = {
                token for token in re.findall(r"[a-zA-Z0-9_]+", sentence.lower())
                if len(token) > 2
            }
            if len(action_tokens & tokens) >= 1 or re.search(
                r"\b(?:this|that|these|those|allows|opens|displays|shows|used for|to configure|so that|because)\b",
                sentence,
                flags=re.IGNORECASE,
            ):
                candidates.append(sentence)
            if len(candidates) >= 2:
                break

    scored = []
    for sentence in candidates:
        tokens = {
            token for token in re.findall(r"[a-zA-Z0-9_]+", sentence.lower())
            if len(token) > 2
        }
        overlap = len(action_tokens & tokens)
        causal = 1 if re.search(
            r"\b(?:this|that|allows|opens|displays|shows|used for|to configure|so that|because|after)\b",
            sentence,
            flags=re.IGNORECASE,
        ) else 0
        scored.append((overlap + causal * 2, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [sentence for score, sentence in scored if score > 0][:2]
    return _shorten(" ".join(selected), 700)


def _deterministic_narration(action: str, context: str) -> str:
    spoken_action = _clean_spoken_narration(action)
    if not spoken_action:
        return "Continue with the next step."

    spoken_action = re.sub(
        r"^\s*(?:now|then|next|finally|first|after that)\s+",
        "",
        spoken_action,
        flags=re.IGNORECASE,
    ).strip(" .;,")
    spoken_action = spoken_action[:1].upper() + spoken_action[1:] + "."

    sentence = (_context_sentences(context) or [""])[0]
    if sentence and re.search(
        r"^(?:this|that|these|those)\s+(?:opens|displays|shows|allows|provides|lets|enables)",
        sentence,
        flags=re.IGNORECASE,
    ):
        combined = _clean_spoken_narration(f"{spoken_action.rstrip('.')} {sentence}")
        if len(combined.split()) <= 45:
            return combined
    return spoken_action


def _parse_narration_response(raw: str) -> str:
    text = str(raw or "").strip().replace("```json", "").replace("```", "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _clean_spoken_narration(parsed.get("narration", ""))
        if isinstance(parsed, str):
            return _clean_spoken_narration(parsed)
    except Exception:
        pass
    match = re.search(r'\{\s*"narration"\s*:\s*"(?P<narration>(?:\\.|[^"\])*)"\s*\}', text, flags=re.DOTALL)
    if match:
        try:
            return _clean_spoken_narration(json.loads('"' + match.group("narration") + '"'))
        except Exception:
            return ""
    return _clean_spoken_narration(text)


def _narration_is_usable(text: str, action: str) -> bool:
    text = _clean_spoken_narration(text)
    if not text:
        return False
    words = text.split()
    low = text.lower()
    if len(words) > 45:
        return False
    if len(words) < 5:
        return False
    if not re.search(r"[.!?]$", text):
        return False
    if re.search(r"(?:page\s+\d+|screenshot|metadata|source text|the learner|here is|narration:)", text, re.IGNORECASE):
        return False
    if re.search(r"https?://|www\.", text, re.IGNORECASE):
        return False
    if text.endswith((":", ";", ",", "-")):
        return False
    if text.count('(') != text.count(')') or text.count('"') % 2:
        return False
    # Avoid a purely generic filler output when the action itself contains
    # meaningful UI language.
    if len(words) <= 11 and low in {
        "continue with the next step.",
        "follow the highlighted step and continue.",
        "continue with the current step.",
    }:
        return False
    return True


def _ollama_narration(action: str, context: str, page_text: str) -> str:
    global _OLLAMA_NARRATION_DISABLED
    if not ENABLE_OLLAMA_NARRATION or _OLLAMA_NARRATION_DISABLED:
        return ""

    action_clean = _clean_spoken_narration(action)
    context_clean = _select_narration_context(action, context, page_text)
    min_words, max_words = _narration_word_range(action)
    complexity = _action_complexity(action)

    prompt = f"""
You are an enterprise software instructor writing one spoken instruction for a training video.

ACTION:
{action_clean}

RELEVANT SOURCE EXPLANATION:
{context_clean or "(None. Do not invent a reason.)"}

STEP TYPE: {complexity}
TARGET LENGTH: {min_words}-{max_words} words

Rules:
- Describe the exact action naturally and clearly.
- Add a purpose only when the source explanation directly supports it.
- Preserve exact UI labels, product names, commands, and technical terms.
- Never invent outcomes, reasons, values, or business meaning.
- Never speak URLs.
- Do not mention pages, screenshots, source documents, metadata, or the learner.
- Do not pad a simple action just to reach the word target.
- Produce a complete, natural sentence; two closely connected sentences are acceptable only for complex steps.
- Output only JSON in this exact shape: {{"narration":"..."}}
""".strip()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.15},
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        generated = _parse_narration_response(data.get("response", "") if isinstance(data, dict) else "")
        return generated if _narration_is_usable(generated, action) else ""
    except requests.exceptions.RequestException as exc:
        _OLLAMA_NARRATION_DISABLED = True
        print(f"[NARRATION] Local Ollama unavailable at {OLLAMA_URL}: {exc}. Using fallback.")
        return ""
    except Exception as exc:
        print(f"[NARRATION] Ollama output rejected: {exc}")
        return ""


def _make_action_narration(action: str, context: str = "", page_text: str = "") -> str:
    action_clean = _clean_spoken_narration(action)
    if not action_clean:
        return "Continue with the next step."

    cache_key = (
        action_clean.lower(),
        _clean_text(context).lower(),
        _clean_text(page_text).lower()[:1200],
    )
    cached = _NARRATION_CACHE.get(cache_key)
    if cached:
        return cached

    complexity = _action_complexity(action)
    source_context = _select_narration_context(action, context, page_text)

    # Use the local model where semantic context can improve the result or
    # where the step is complex. Simple action-only steps remain deterministic
    # to reduce latency and unnecessary model calls.
    generated = ""
    if ENABLE_OLLAMA_NARRATION and (source_context or complexity == "complex"):
        generated = _ollama_narration(action_clean, context, page_text)

    if not generated:
        generated = _deterministic_narration(action_clean, source_context)

    generated = _clean_spoken_narration(generated)
    if not _narration_is_usable(generated, action_clean):
        generated = _clean_spoken_narration(action_clean).rstrip(" .;,") + "."

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


def _caption_with_urls(narration: str, urls: List[str]) -> str:
    # The visible/burned caption is intentionally the same clean spoken text.
    # Resource URLs live in `caption_urls` and are appended only to VTT
    # metadata by the renderer, so they are never spoken or burned into the UI.
    return _clean_spoken_narration(narration)


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
    global _OLLAMA_NARRATION_DISABLED
    _VISION_CACHE.clear()
    _NARRATION_CACHE.clear()
    _OLLAMA_NARRATION_DISABLED = False

    pages = data.get("pages", [])
    screenshots = data.get("screenshots", [])
    if not isinstance(pages, list):
        pages = []
    if not isinstance(screenshots, list):
        screenshots = []

    _progress(progress_callback, 2, "Preparing document structure")
    shots_by_page = _build_screenshot_map(screenshots)
    _progress(progress_callback, 8, f"Mapped {len(screenshots)} embedded visual regions")
    sections_raw = _build_sections(pages)

    sections_out: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    step_id = 0
    total_sections = max(1, len(sections_raw))

    for section_index, section in enumerate(sections_raw):
        section_step_ids: List[int] = []
        action_bullets: List[str] = []
        explanation_lines: List[str] = []
        section_actions: List[Tuple[Dict[str, Any], str, str]] = []

        for page in section.get("pages", []):
            if not isinstance(page, dict):
                continue
            blocks = _extract_instruction_blocks(page)
            for block_index, block in enumerate(blocks):
                action = _clean_text(block.get("action"))
                own_context = _clean_text(block.get("context"))
                context_parts: List[str] = []
                if own_context:
                    context_parts.append(own_context)
                if block_index > 0:
                    previous = blocks[block_index - 1]
                    previous_context = _clean_text(previous.get("context"))
                    if not _clean_text(previous.get("action")) and previous_context:
                        context_parts.append(previous_context)
                if block_index + 1 < len(blocks):
                    following = blocks[block_index + 1]
                    following_context = _clean_text(following.get("context"))
                    if not _clean_text(following.get("action")) and following_context:
                        context_parts.append(following_context)
                context = _clean_text(" ".join(dict.fromkeys(context_parts)))
                if action:
                    section_actions.append((page, action, context))
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

            # Cheap deterministic target matching first. This also chooses the
            # correct embedded screenshot when a page contains multiple images.
            for candidate_shot in page_shots:
                candidate_grounding = _ground_action(
                    action, page, candidate_shot, allow_vision=False
                )
                if candidate_grounding.get("accepted"):
                    score = float(candidate_grounding.get("confidence", 0.0))
                    if score > best_score:
                        chosen_shot = candidate_shot
                        grounding = candidate_grounding
                        best_score = score

            # If no deterministic target exists, use the largest screenshot as
            # the visual scene and allow one vision attempt. Crucially, a weak
            # result does NOT create a cursor target.
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
                    action, page, chosen_shot, allow_vision=ENABLE_OLLAMA
                )

            if grounding is None:
                grounding = _ground_action(action, page, None, allow_vision=False)

            if not grounding.get("accepted"):
                # Still show the most relevant screenshot if one exists, but
                # explicitly disable cursor/click behavior.
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

            screenshot_path = _screenshot_path(chosen_shot) if chosen_shot else None
            page_text = _clean_text(page.get("text", ""))
            narration = _make_action_narration(action, context=context, page_text=page_text)
            urls = _relevant_urls(action, context, page)

            step_id += 1
            point = grounding.get("point") if grounding.get("accepted") else None
            source = grounding.get("source", "none") if grounding.get("accepted") else "none"
            confidence = float(grounding.get("confidence", 0.0)) if point is not None else 0.0
            target_name = _clean_text(grounding.get("target_name", "")) if point is not None else ""
            bounding_box = grounding.get("bounding_box") if point is not None else None
            annotation = grounding.get("annotation") if point is not None else None
            action_type = _action_type(action)
            interaction = _interaction_for_action(action, action_type, point is not None)
            # Keyboard-only actions should never display a mouse target, even
            # when OCR/vision happens to find a matching word such as "Enter".
            visual_target_enabled = bool(point is not None and interaction != "none")
            if not visual_target_enabled:
                point = None
                source = "none"
                confidence = 0.0
                target_name = ""
                bounding_box = None
                annotation = None

            dialogue = _make_dialogue(
                action, target_name, narration=narration, context=_select_narration_context(action, context, page_text)
            )

            step = {
                "id": step_id,
                "section_id": section_index + 1,
                "page_num": page_number,
                "page_width": float(page.get("width") or 0),
                "page_height": float(page.get("height") or 0),
                "kind": "action",
                "title": _shorten(action, 100),
                "narration": narration,
                "tts_narration": narration,
                "caption": _caption_with_urls(narration, urls),
                "caption_text": narration,
                "caption_urls": urls,
                "urls": urls,
                "source_context": _shorten(_remove_urls(_select_narration_context(action, context, page_text)), 900),
                "screenshot": screenshot_path,
                "is_full_page": screenshot_path is None,
                "cursor": point,
                "targets": [point] if point is not None else [],
                "cursor_enabled": visual_target_enabled,
                "cursor_source": source,
                "cursor_confidence": confidence,
                "cursor_target_name": target_name,
                "cursor_annotation": annotation,
                "cursor_bounding_box": bounding_box,
                "target_status": (
                    "verified"
                    if visual_target_enabled
                    else (
                        "not_applicable"
                        if interaction == "none" and action_type == "Press"
                        else grounding.get("target_status", "unresolved")
                    )
                ),
                "target_query": grounding.get("target_query", _target_query(action)),
                "target_type": grounding.get("target_type", _target_type(action)),
                "target_debug": {
                    "status": grounding.get("target_status", "unresolved"),
                    "accepted": bool(grounding.get("accepted")),
                    "source": source,
                    "confidence": confidence,
                    "query": grounding.get("target_query", _target_query(action)),
                    "target_name": target_name,
                    "ambiguity_margin": grounding.get("ambiguity_margin"),
                },
                "action_type": action_type,
                "interaction": interaction,
                "interaction_plan": {
                    "type": interaction,
                    "target_required": interaction != "none",
                },
                "complexity": _action_complexity(action),
                "dialogue": dialogue,
            }

            steps.append(step)
            section_step_ids.append(step_id)
            action_bullets.append(_shorten(action, 160))

        theory_bullets = _make_theory_bullets(explanation_lines)
        section_title = _clean_text(section.get("title")) or f"Section {section_index + 1}"
        sections_out.append({
            "id": section_index + 1,
            "title": _shorten(section_title, 90),
            "theory_bullets": theory_bullets,
            "action_bullets": action_bullets,
            "step_ids": section_step_ids,
            "action_count": len(section_step_ids),
            "has_video": bool(section_step_ids),
        })

        _progress(
            progress_callback,
            int(10 + 80 * ((section_index + 1) / float(total_sections))),
            f"Prepared section {section_index + 1} of {total_sections}",
        )

    if not steps and pages:
        _progress(progress_callback, 92, "No explicit action verbs found; creating page walkthrough")
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
            page_number = _page_number(page, page_index + 1)
            page_shots = shots_by_page.get(page_number, [])
            chosen_shot = page_shots[0] if page_shots else None
            screenshot_path = _screenshot_path(chosen_shot) if chosen_shot else None
            description = _shorten(_remove_urls(page_text), 220)
            narration = _deterministic_narration(f"Review the content on page {page_number}", description)
            urls = _relevant_urls(page_text, "", page)
            step_id += 1
            step = {
                "id": step_id, "section_id": fallback_section["id"],
                "page_num": page_number, "page_width": float(page.get("width") or 0),
                "page_height": float(page.get("height") or 0), "kind": "action",
                "title": f"Review page {page_number}", "narration": narration,
                "tts_narration": narration, "caption": _caption_with_urls(narration, urls), "caption_text": narration,
                "caption_urls": urls, "urls": urls, "source_context": description,
                "screenshot": screenshot_path, "is_full_page": screenshot_path is None,
                "cursor": None, "targets": [], "cursor_enabled": False, "cursor_source": "none",
                "cursor_confidence": 0.0, "cursor_target_name": "", "cursor_annotation": None,
                "cursor_bounding_box": None, "target_status": "not_applicable", "target_query": "",
                "target_type": "", "action_type": "Follow", "interaction": "none",
                "complexity": "simple", "dialogue": _make_dialogue(
                    f"Review the content on page {page_number}", narration=narration, context=description
                ),
            }
            steps.append(step)
            fallback_section["step_ids"].append(step_id)
            fallback_section["action_bullets"].append(f"Review page {page_number}")
            fallback_section["action_count"] += 1
            if len(steps) >= 100:
                break

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

    overview_parts = []
    for page in pages[:5]:
        if not isinstance(page, dict):
            continue
        text = _remove_urls(_clean_text(page.get("text", "")))
        if text:
            overview_parts.append(text)
    overview_source = _shorten(" ".join(overview_parts), 500)
    description = (
        "This tutorial provides a guided walkthrough of the uploaded document. " + overview_source
        if overview_source else
        "This tutorial provides a guided walkthrough of the uploaded software workflow."
    )

    plan = {
        "title": title, "description": description, "overview": description,
        "narration_language": narration_language or "en-us", "sections": sections_out,
        "steps": steps, "scene_count": len(steps),
        "action_count": sum(1 for step in steps if step.get("kind") == "action"),
        "total_pages": len(pages), "total_screenshots": len(screenshots),
        "visual_source": "embedded PDF screenshots with full-page fallback",
        "ai_mode": "local-deterministic" if not ENABLE_OLLAMA else "local-ollama-optional",
        "narration_mode": "ollama-contextual-with-local-fallback" if ENABLE_OLLAMA_NARRATION else "deterministic-local",
    }
    _progress(progress_callback, 100, f"Tutorial plan ready: {len(steps)} action steps")
    print(f"[AI] Plan created: {len(sections_out)} sections, {len(steps)} action steps, {len(screenshots)} screenshots")
    return plan


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

generate_tutorial = build_tutorial_plan
