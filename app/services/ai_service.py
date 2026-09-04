import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import requests

from app.services.local_vision import analyze_image


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2",
)

OLLAMA_TIMEOUT = int(
    os.getenv("OLLAMA_TIMEOUT", "60")
)

VISION_ENABLED = (
    os.getenv(
        "OLLAMA_VISION_ENABLED",
        "true",
    ).lower()
    == "true"
)

VISION_MIN_CONFIDENCE = float(
    os.getenv(
        "VISION_MIN_CONFIDENCE",
        "0.40",
    )
)


# ----------------------------------------------------------------------
# GENERAL HELPERS
# ----------------------------------------------------------------------

def _ollama(
    prompt: str,
    timeout: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout or OLLAMA_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        text = data.get("response", "")

        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL,
        )

        if match:
            return json.loads(
                match.group(0)
            )

    except Exception as exc:
        print(f"[OLLAMA] error: {exc}")

    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)

    text = text.replace(
        "\r",
        " ",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    return text.strip()


def _extract_urls(text: str) -> List[str]:
    urls = re.findall(
        r"https?://[^\s)\]>]+",
        text,
        flags=re.IGNORECASE,
    )

    result = []

    for url in urls:
        url = url.rstrip(
            ".,;:"
        )

        if url not in result:
            result.append(url)

    return result


def _remove_urls(text: str) -> str:
    return re.sub(
        r"https?://[^\s)\]>]+",
        "",
        text,
        flags=re.IGNORECASE,
    )


def _is_metadata(line: str) -> bool:
    line = _clean_text(line)

    if not line:
        return True

    if re.match(
        r"^page\s*\|?\s*\d+$",
        line,
        flags=re.IGNORECASE,
    ):
        return True

    if re.match(
        r"^page\s+\d+$",
        line,
        flags=re.IGNORECASE,
    ):
        return True

    if line.lower() in {
        "itl plm user manual",
        "uploaded pdf",
    }:
        return True

    return False


def _normalise_source_text(
    text: str,
) -> str:
    lines = []

    for raw in text.splitlines():
        line = _clean_text(raw)

        if _is_metadata(line):
            continue

        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        lines.append(line)

    return "\n".join(lines)


# ----------------------------------------------------------------------
# HEADINGS / STRUCTURE
# ----------------------------------------------------------------------

def _looks_like_heading(
    line: str,
) -> bool:
    line = _clean_text(line)

    if not line:
        return False

    # Examples:
    # 2 Classification Search
    # 2.1 Purpose
    # 3.4 Procedure
    if re.match(
        r"^\d+(?:\.\d+)*\s+[A-Za-z]",
        line,
    ):
        return True

    heading_words = {
        "purpose",
        "procedure",
        "execute search",
        "view filtered results",
        "apply property-based filters",
        "compare the selected parts",
        "identify the required part",
        "view in full screen",
        "open the selected part in a new window",
        "open the part in nx",
        "select search type",
        "part creation in teamcenter",
        "update attributes",
        "open in nx command",
        "attachments visibility in active workspace",
        "part creation in nx",
        "conversion of temporary part to 3 series production part",
        "geolus shape search",
    }

    return (
        line.lower().rstrip(":")
        in heading_words
    )


def _heading_level(
    line: str,
) -> int:
    match = re.match(
        r"^(\d+(?:\.\d+)*)\s+",
        line,
    )

    if not match:
        return 0

    return match.group(1).count(".") + 1


# ----------------------------------------------------------------------
# ACTION DETECTION
# ----------------------------------------------------------------------

ACTION_VERBS = (
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
    "double click",
    "right click",
    "drag",
    "press",
    "check",
    "uncheck",
)


def _is_action_line(
    line: str,
) -> bool:
    low = line.lower().strip()

    return any(
        re.match(
            rf"^{re.escape(verb)}\b",
            low,
        )
        for verb in ACTION_VERBS
    )


def _is_bad_segment(
    text: str,
) -> bool:
    text = _clean_text(text)

    if not text:
        return True

    if len(text.split()) <= 1:
        if text.lower() in {
            "type",
            "select",
            "click",
            "open",
            "add",
            "enter",
        }:
            return True

    if re.search(
        r"\bonce\s+ready\s+can\s+be$",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    return False


# ----------------------------------------------------------------------
# COMPOUND ACTIONS
# ----------------------------------------------------------------------

def _split_compound_action(
    text: str,
) -> List[str]:
    text = _clean_text(text)

    if not text:
        return []

    text = re.sub(
        r"\s+as shown below\.?$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Explicit UI chains.
    pieces = re.split(
        r"\s*(?:>\s*|;\s*|\bthen\b|\band then\b)\s*",
        text,
        flags=re.IGNORECASE,
    )

    expanded = []

    action_start = re.compile(
        r"^(?:"
        r"open|click|select|enter|type|browse|launch|"
        r"choose|apply|create|save|run|fill|search|locate|"
        r"expand|collapse|double\s+click|right\s+click|"
        r"drag|press|check|uncheck"
        r")\b",
        flags=re.IGNORECASE,
    )

    for piece in pieces:
        piece = _clean_text(piece)

        if not piece:
            continue

        # Split:
        # "Select the part and click Attachment tab."
        matches = list(
            re.finditer(
                r"\s+\band\b\s+(?="
                r"(?:open|click|select|enter|type|browse|"
                r"launch|choose|apply|create|save|run|fill|"
                r"search|locate|expand|collapse|double\s+click|"
                r"right\s+click|drag|press|check|uncheck)"
                r"\b)",
                piece,
                flags=re.IGNORECASE,
            )
        )

        if matches:
            start = 0

            for match in matches:
                first = _clean_text(
                    piece[
                        start:match.start()
                    ]
                )

                if first:
                    expanded.append(first)

                start = match.end()

            last = _clean_text(
                piece[start:]
            )

            if last:
                expanded.append(last)

        else:
            expanded.append(piece)

    result = []

    for piece in expanded:
        piece = piece.strip(" .")

        if not piece:
            continue

        # Do not accidentally turn ordinary prose into actions.
        if not action_start.match(piece):
            result.append(
                piece + "."
                if not piece.endswith(".")
                else piece
            )
            continue

        if not piece.endswith("."):
            piece += "."

        result.append(piece)

    return result


# ----------------------------------------------------------------------
# HEURISTIC PLANNER
# ----------------------------------------------------------------------

def _heuristic_segments(
    document_text: str,
) -> List[Dict[str, Any]]:
    text = _normalise_source_text(
        document_text
    )

    lines = [
        x
        for x in text.splitlines()
        if x.strip()
    ]

    segments = []

    for line in lines:
        if _looks_like_heading(line):
            segments.append(
                {
                    "kind": "topic",
                    "title": line,
                    "content": line,
                }
            )
            continue

        if _is_action_line(line):
            for piece in _split_compound_action(
                line
            ):
                if not _is_bad_segment(piece):
                    segments.append(
                        {
                            "kind": "action",
                            "title": piece,
                            "content": piece,
                        }
                    )
            continue

        # Ignore isolated UI labels.
        if (
            len(line.split()) <= 2
            and line.lower()
            in {
                "type",
                "select",
                "add",
            }
        ):
            continue

        segments.append(
            {
                "kind": "explanation",
                "title": "",
                "content": line,
            }
        )

    return segments


# ----------------------------------------------------------------------
# AI PLANNER
# ----------------------------------------------------------------------

def _ai_extract_segments(
    document_text: str,
) -> List[Dict[str, Any]]:
    cleaned = _normalise_source_text(
        document_text
    )

    prompt = f"""
You are the document planner for a professional
software training video.

Convert the supplied software manual into an ordered
training plan.

The source document is authoritative.

DO NOT invent information.

DO NOT omit meaningful information.

DO NOT expand the document with your own explanations.

DO NOT add generic training filler.

DO NOT read page numbers or document metadata.

PRESERVE THE ORDER of the source document.

Classify meaningful content as:

topic
------
A section or subsection introducing a subject or procedure.

Examples:
"2 Classification Search"
"3 Geolus Shape Search"
"4 Part Creation in Teamcenter"

explanation
-----------
Important information explaining:
- purpose
- context
- prerequisites
- what a feature does
- why it is used
- important operational information

action
------
A concrete user operation:
- open
- click
- select
- enter
- type
- browse
- launch
- choose
- apply
- create
- save
- run
- fill
- search
- locate
- expand
- collapse
- etc.

VERY IMPORTANT:

Do not merge separate UI interactions.

For example:

"Click Add button and select ITL Design Part from drop down."

MUST become:

action: "Click Add button."
action: "Select ITL Design Part from the drop down."

Likewise:

"Select the part and click Attachment tab."

MUST become:

action: "Select the part."
action: "Click the Attachment tab."

Do not create fragments such as:

"Type"
"Select"
"Create temporary Part and once ready can be"

Use surrounding source text to reconstruct an extraction-fragment
when the continuation is explicitly present elsewhere in the source.

Do not invent missing text.

EXPLANATIONS:

Keep useful explanatory information.

Do NOT compress several important facts into one vague sentence.

If a paragraph or bullet list contains several related facts,
combine them into one or two concise sentences while preserving
the important meaning.

URLS:

Preserve URLs in content.

Never put the URL into spoken narration.

The video renderer will display the URL visually.

Return ONLY JSON:

{{
  "title": "short useful tutorial title",
  "segments": [
    {{
      "kind": "topic|explanation|action",
      "title": "short title",
      "content": "document-grounded content"
    }}
  ]
}}

DOCUMENT:
{cleaned}
"""

    result = _ollama(prompt)

    if not result:
        return []

    raw_segments = result.get(
        "segments"
    )

    if not isinstance(
        raw_segments,
        list,
    ):
        return []

    output = []

    for item in raw_segments:
        if not isinstance(
            item,
            dict,
        ):
            continue

        kind = _clean_text(
            item.get("kind")
        ).lower()

        content = _clean_text(
            item.get("content")
        )

        title = _clean_text(
            item.get("title")
        )

        if kind not in {
            "topic",
            "explanation",
            "action",
        }:
            continue

        if not content:
            continue

        # Protect against AI returning compound actions.
        if kind == "action":
            pieces = _split_compound_action(
                content
            )

            for piece in pieces:
                if not _is_bad_segment(
                    piece
                ):
                    output.append(
                        {
                            "kind": "action",
                            "title": piece,
                            "content": piece,
                        }
                    )
        else:
            output.append(
                {
                    "kind": kind,
                    "title": title or content,
                    "content": content,
                }
            )

    return output


# ----------------------------------------------------------------------
# PAGE ASSOCIATION
# ----------------------------------------------------------------------

def _page_text(
    page: Any,
) -> str:
    if isinstance(page, str):
        return _clean_text(page)

    if not isinstance(page, dict):
        return ""

    for key in (
        "text",
        "content",
        "document_text",
        "page_text",
    ):
        value = page.get(key)

        if value:
            return _clean_text(value)

    return ""


def _page_number(
    page: Any,
    fallback: int,
) -> Optional[int]:
    if isinstance(page, dict):
        for key in (
            "page",
            "page_number",
            "number",
        ):
            value = page.get(key)

            if isinstance(
                value,
                int,
            ):
                return value

            if isinstance(
                value,
                str,
            ):
                try:
                    return int(value)
                except Exception:
                    pass

    return fallback


def _tokens(text: str) -> set:
    text = _remove_urls(
        text.lower()
    )

    return {
        token
        for token in re.findall(
            r"[a-z0-9]+",
            text,
        )
        if len(token) >= 3
    }


def _page_for_segment(
    segment: Dict[str, Any],
    pages: List[Any],
    previous_page: Optional[int] = None,
) -> Optional[int]:
    explicit = segment.get(
        "source_page"
    )

    if isinstance(
        explicit,
        int,
    ):
        return explicit

    content = _clean_text(
        segment.get("content")
    )

    if not content or not pages:
        return previous_page

    source_tokens = _tokens(
        content
    )

    if not source_tokens:
        return previous_page

    best_page = None
    best_score = 0.0

    for index, page in enumerate(
        pages,
        start=1,
    ):
        text = _page_text(page)

        if not text:
            continue

        page_tokens = _tokens(
            text
        )

        if not page_tokens:
            continue

        overlap = len(
            source_tokens
            & page_tokens
        )

        if overlap == 0:
            continue

        score = overlap / max(
            1,
            len(source_tokens),
        )

        # Small preference for the current/previous page.
        page_number = _page_number(
            page,
            index,
        )

        if (
            previous_page is not None
            and page_number == previous_page
        ):
            score += 0.08

        if score > best_score:
            best_score = score
            best_page = page_number

    return (
        best_page
        if best_page is not None
        else previous_page
    )


# ----------------------------------------------------------------------
# SCREENSHOTS
# ----------------------------------------------------------------------

def _normalise_screenshots(
    screenshots: List[Any],
) -> List[Dict[str, Any]]:
    result = []

    for item in screenshots:
        if isinstance(
            item,
            str,
        ):
            result.append(
                {
                    "path": item,
                    "page": None,
                }
            )
            continue

        if not isinstance(
            item,
            dict,
        ):
            continue

        path = (
            item.get("path")
            or item.get("file")
            or item.get("screenshot")
        )

        if not path:
            continue

        page = (
            item.get("page")
            or item.get("source_page")
            or item.get("page_number")
        )

        if isinstance(
            page,
            str,
        ):
            try:
                page = int(page)
            except Exception:
                page = None

        result.append(
            {
                "path": path,
                "page": page,
            }
        )

    return result


def _choose_screenshot(
    screenshots: List[Dict[str, Any]],
    source_page: Optional[int],
    action: str,
) -> Optional[Dict[str, Any]]:
    if not screenshots:
        return None

    # Best case: same page.
    if source_page is not None:
        same_page = [
            item
            for item in screenshots
            if item.get("page")
            == source_page
        ]

        if same_page:
            return same_page[0]

    # Otherwise nearest screenshot page.
    if source_page is not None:
        candidates = [
            item
            for item in screenshots
            if isinstance(
                item.get("page"),
                int,
            )
        ]

        if candidates:
            return min(
                candidates,
                key=lambda item: abs(
                    item["page"]
                    - source_page
                ),
            )

    return screenshots[0]


# ----------------------------------------------------------------------
# VISION
# ----------------------------------------------------------------------

def _target_query(
    action: str,
) -> str:
    action = _remove_urls(
        action
    )

    action = re.sub(
        r"\s+as shown below\.?$",
        "",
        action,
        flags=re.IGNORECASE,
    )

    return _clean_text(
        action
    )


def _vision_target(
    screenshot_path: Optional[str],
    action: str,
) -> Dict[str, Any]:
    empty = {
        "found": False,
        "target_name": "",
        "click_point": None,
        "bounding_box": None,
        "confidence": 0.0,
    }

    if not screenshot_path:
        return empty

    if not VISION_ENABLED:
        return empty

    try:
        result = analyze_image(
            screenshot_path,
            _target_query(action),
        )
    except Exception as exc:
        print(
            f"[VISION] error: {exc}"
        )
        return empty

    if not isinstance(
        result,
        dict,
    ):
        return empty

    found = bool(
        result.get("found")
    )

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

    if not found:
        return {
            "found": False,
            "target_name": _clean_text(
                result.get("target_name")
            ),
            "click_point": None,
            "bounding_box": result.get(
                "bounding_box"
            ),
            "confidence": confidence,
        }

    click_point = result.get(
        "click_point"
    )

    bbox = result.get(
        "bounding_box"
    )

    if not (
        isinstance(
            click_point,
            (list, tuple),
        )
        and len(click_point) == 2
    ):
        return {
            "found": False,
            "target_name": _clean_text(
                result.get("target_name")
            ),
            "click_point": None,
            "bounding_box": bbox,
            "confidence": confidence,
        }

    if confidence < VISION_MIN_CONFIDENCE:
        print(
            f"[VISION] low confidence "
            f"{confidence:.2f}: {action}"
        )

        return {
            "found": False,
            "target_name": _clean_text(
                result.get("target_name")
            ),
            "click_point": None,
            "bounding_box": bbox,
            "confidence": confidence,
        }

    try:
        x = max(
            0.0,
            min(
                1.0,
                float(click_point[0]),
            ),
        )

        y = max(
            0.0,
            min(
                1.0,
                float(click_point[1]),
            ),
        )
    except Exception:
        return {
            "found": False,
            "target_name": "",
            "click_point": None,
            "bounding_box": bbox,
            "confidence": confidence,
        }

    return {
        "found": True,
        "target_name": _clean_text(
            result.get(
                "target_name"
            )
        ),
        "click_point": [x, y],
        "bounding_box": bbox,
        "confidence": confidence,
    }


# ----------------------------------------------------------------------
# NARRATION
# ----------------------------------------------------------------------

def _narration_for_segment(
    text: str,
    kind: str,
) -> str:
    text = _clean_text(
        text
    )

    if not text:
        return ""

    urls = _extract_urls(
        text
    )

    spoken = _remove_urls(
        text
    )

    spoken = re.sub(
        r"\s+as shown below\.?$",
        "",
        spoken,
        flags=re.IGNORECASE,
    )

    spoken = re.sub(
        r"\bpage\s+\d+\b",
        "",
        spoken,
        flags=re.IGNORECASE,
    )

    spoken = _clean_text(
        spoken
    )

    if not spoken and urls:
        if kind == "action":
            return "Open the following URL."

        return ""

    return spoken


def _ai_narration(
    text: str,
    kind: str,
) -> str:
    source = _narration_for_segment(
        text,
        kind,
    )

    if not source:
        return ""

    # Actions should remain direct and source-grounded.
    if kind == "action":
        return source

    # Topics should be short.
    if kind == "topic":
        words = source.split()

        if len(words) <= 12:
            return source

        return _clean_text(
            source
        )

    # Explanations:
    # preserve important information but allow modest compression.
    prompt = f"""
Rewrite this software-training explanation for spoken narration.

SOURCE:
{source}

Rules:

1. Use ONLY information in the source.
2. Do not invent anything.
3. Preserve the important facts.
4. Do not remove useful prerequisites, purpose, constraints,
   or operational information.
5. Do not elaborate.
6. Do not add generic training language.
7. Do not mention captions.
8. Do not mention URLs.
9. Do not mention page numbers.
10. Do not say "as shown below".
11. Do not say "in this video".
12. Do not say "we will".
13. Use one or two concise spoken sentences.
14. Prefer approximately 15-55 words.
15. If the source contains several important related points,
    retain them compactly rather than reducing everything
    to one vague statement.

Return ONLY JSON:

{{
  "narration": "..."
}}
"""

    result = _ollama(
        prompt
    )

    if result:
        narration = _clean_text(
            result.get(
                "narration"
            )
        )

        if narration:
            return narration

    return source


# ----------------------------------------------------------------------
# CAPTIONS
# ----------------------------------------------------------------------

def _caption_for_segment(
    text: str,
    kind: str,
) -> str:
    text = _clean_text(
        text
    )

    if kind == "topic":
        return text

    # URLs are displayed, but never narrated.
    # Keep them in captions.
    urls = _extract_urls(
        text
    )

    caption = _remove_urls(
        text
    ).strip()

    if urls:
        if caption:
            return (
                f"{caption}\n"
                + "\n".join(urls)
            )

        return "\n".join(
            urls
        )

    return caption


# ----------------------------------------------------------------------
# MAIN PLAN BUILDER
# ----------------------------------------------------------------------

def build_tutorial_plan(
    data: Dict[str, Any],
    narration_language: str = "en-us",
    progress_callback=None,
) -> Dict[str, Any]:

    def report(
        progress: int,
        message: str = "Building tutorial plan",
    ):
        """
        Single callback gateway.

        Supports both old and new callers.
        """

        if not progress_callback:
            return

        try:
            progress_callback(
                int(progress),
                message,
            )
        except TypeError:
            # Backward compatibility with callbacks
            # accepting only progress.
            progress_callback(
                int(progress)
            )

    document_text = (
        data.get("document_text")
        or data.get("text")
        or ""
    )

    screenshots = _normalise_screenshots(
        data.get("screenshots") or []
    )

    pages = data.get(
        "pages"
    ) or []

    report(
        5,
        "Analysing document structure",
    )

    ai_segments = _ai_extract_segments(
        document_text
    )

    report(
        35,
        "Building topics, explanations and actions",
    )

    if not ai_segments:
        print(
            "[PLAN] LLM extraction unavailable; "
            "using deterministic document planner"
        )

        ai_segments = _heuristic_segments(
            document_text
        )

    if not ai_segments:
        raise RuntimeError(
            "Unable to create a tutorial plan from the document."
        )

    steps = []

    next_id = 1
    previous_page = None

    total = len(
        ai_segments
    )

    for index, raw_segment in enumerate(
        ai_segments,
        start=1,
    ):
        kind = _clean_text(
            raw_segment.get(
                "kind"
            )
        ).lower()

        content = _clean_text(
            raw_segment.get(
                "content"
            )
        )

        title = _clean_text(
            raw_segment.get(
                "title"
            )
        )

        if kind not in {
            "topic",
            "explanation",
            "action",
        }:
            continue

        if _is_bad_segment(
            content
        ):
            continue

        source_page = _page_for_segment(
            raw_segment,
            pages,
            previous_page,
        )

        if source_page is not None:
            previous_page = source_page

        if kind == "action":
            pieces = _split_compound_action(
                content
            )
        else:
            pieces = [content]

        for piece in pieces:
            piece = _clean_text(
                piece
            )

            if _is_bad_segment(
                piece
            ):
                continue

            # --------------------------------------------------
            # Screenshot
            # --------------------------------------------------

            screenshot = _choose_screenshot(
                screenshots,
                source_page,
                piece,
            )

            screenshot_path = (
                screenshot.get("path")
                if screenshot
                else None
            )

            screenshot_page = (
                screenshot.get("page")
                if screenshot
                else source_page
            )

            # --------------------------------------------------
            # Visual grounding
            # --------------------------------------------------

            visual = {
                "found": False,
                "target_name": "",
                "click_point": None,
                "bounding_box": None,
                "confidence": 0.0,
            }

            if (
                kind == "action"
                and screenshot_path
            ):
                visual = _vision_target(
                    screenshot_path,
                    piece,
                )

            # --------------------------------------------------
            # Narration
            # --------------------------------------------------

            narration = _ai_narration(
                piece,
                kind,
            )

            caption = _caption_for_segment(
                piece,
                kind,
            )

            step = {
                "id": next_id,
                "kind": kind,
                "number": next_id,

                "title": (
                    piece
                    if kind == "action"
                    else (
                        title
                        or piece
                    )
                ),

                "content": piece,

                "narration": narration,
                "tts_narration": narration,

                "caption": caption,

                "screenshot": screenshot_path,

                "screenshot_page": screenshot_page,

                "source_page": source_page,

                "cursor": (
                    visual["click_point"]
                    if visual["found"]
                    else None
                ),

                "actions": (
                    [
                        {
                            "type": "guided_cursor",
                            "target": visual[
                                "click_point"
                            ],
                            "target_name": visual[
                                "target_name"
                            ],
                            "bounding_box": visual[
                                "bounding_box"
                            ],
                            "confidence": visual[
                                "confidence"
                            ],
                        }
                    ]
                    if (
                        kind == "action"
                        and visual["found"]
                    )
                    else []
                ),

                "visual_type": (
                    "embedded_screenshot"
                    if screenshot_path
                    else "none"
                ),

                "visual_target": visual[
                    "target_name"
                ],

                "visual_confidence": visual[
                    "confidence"
                ],
            }

            steps.append(
                step
            )

            next_id += 1

        report(
            35 + int(
                60
                * (
                    index
                    / max(
                        1,
                        total,
                    )
                )
            ),
            (
                f"Processing tutorial content "
                f"{index}/{total}"
            ),
        )

    # --------------------------------------------------------------
    # Title
    # --------------------------------------------------------------

    title = "Software Training Tutorial"

    for step in steps:
        if step.get("kind") == "topic":
            candidate = _clean_text(
                step.get("title")
            )

            if candidate:
                title = candidate
                break

    # Remove numbering from title where possible.
    title = re.sub(
        r"^\d+(?:\.\d+)*\s+",
        "",
        title,
    ).strip()

    if not title:
        title = "Software Training Tutorial"

    plan = {
        "title": title,

        "description": (
            "Software training tutorial generated "
            "from the uploaded document."
        ),

        "source": "uploaded PDF",

        "narration_language": narration_language,

        "steps": steps,

        "screenshot_count": len(
            screenshots
        ),

        "document_text": document_text,
    }

    report(
        100,
        (
            f"Tutorial plan ready: "
            f"{len(steps)} scenes"
        ),
    )

    topic_count = sum(
        1
        for step in steps
        if step.get("kind")
        == "topic"
    )

    explanation_count = sum(
        1
        for step in steps
        if step.get("kind")
        == "explanation"
    )

    action_count = sum(
        1
        for step in steps
        if step.get("kind")
        == "action"
    )

    grounded_count = sum(
        1
        for step in steps
        if (
            step.get("kind")
            == "action"
            and step.get("cursor")
            is not None
        )
    )

    print(
        f"[PLAN] generated {len(steps)} scenes "
        f"({topic_count} topics, "
        f"{explanation_count} explanations, "
        f"{action_count} actions, "
        f"{grounded_count} visually grounded actions)"
    )

    return plan