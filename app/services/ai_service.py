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

# On Windows, tesseract often isn't on PATH even after installing it. Rather
# than requiring a PATH edit, allow pointing straight at the binary via an
# env var -- e.g. set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
_tesseract_cmd = os.getenv("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

try:
    _TESSERACT_VERSION = pytesseract.get_tesseract_version()
    print(f"[OCR] tesseract binary found (v{_TESSERACT_VERSION}) -- OCR-based cursor grounding is ACTIVE.")
except Exception as _tess_err:
    print(
        "=" * 70 + "\n"
        "[OCR] WARNING: the tesseract-ocr binary was NOT found on this "
        "machine.\n"
        "      pytesseract is installed, but it just wraps the real OCR "
        "engine --\n"
        "      without the engine itself, EVERY cursor-grounding call will "
        "silently\n"
        "      fail and no cursor will be drawn anywhere in the generated "
        "video.\n"
        "      Install it:\n"
        "        Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
        "        macOS   : brew install tesseract\n"
        "        Linux   : apt install tesseract-ocr\n"
        f"      Underlying error: {_tess_err}\n" + "=" * 70
    )

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
VISION_ENABLED = os.getenv("OLLAMA_VISION_ENABLED", "true").lower() == "true"
VISION_MIN_CONFIDENCE = float(os.getenv("OLLAMA_VISION_MIN_CONFIDENCE", "0.55"))
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "40"))
OCR_FUZZY_MIN_RATIO = float(os.getenv("OCR_FUZZY_MIN_RATIO", "0.72"))

ACTION_VERBS = (
    "double click", "double-click", "right click", "right-click",
    "navigate to", "go to", "log in", "sign in",
    "open", "click", "select", "enter", "type", "browse", "launch",
    "choose", "apply", "create", "save", "run", "fill", "search",
    "locate", "expand", "collapse", "drag", "drop", "press", "check",
    "uncheck", "set", "add", "remove", "delete", "submit", "upload",
    "download", "attach", "confirm", "cancel",
)
_ACTION_START = re.compile(
    r"^(?:" + "|".join(re.escape(v) for v in sorted(ACTION_VERBS, key=len, reverse=True)) + r")\b",
    re.I,
)
_LEAD_VERB = re.compile(
    r"^(?:" + "|".join(re.escape(v) for v in sorted(ACTION_VERBS, key=len, reverse=True)) + r")\s+"
    r"(?:the\s+|on\s+the\s+|on\s+)?",
    re.I,
)
_GENERIC_NOUNS = re.compile(
    r"\b(button|field|menu|tab|link|icon|option|box|dropdown|drop-down|area|screen|control|checkbox|panel)\b",
    re.I,
)
_LEADING_NUMBER = re.compile(r"^\d+(\.\d+)*[\).:\-]?\s*")

MIN_BOX_ALIGN = 0.0002


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[ \t]+", " ", str(value).replace("\r", " ")).strip()


def _extract_urls(text: str) -> List[str]:
    return [u.rstrip(".,;:") for u in re.findall(r"https?://[^\s)\]>]+", text, flags=re.IGNORECASE)]


def _remove_urls(text: str) -> str:
    return re.sub(r"https?://[^\s)\]>]+", "", text, flags=re.IGNORECASE).strip()


def _clean_spoken_narration(raw: str) -> str:
    if not raw:
        return ""
    text = _remove_urls(raw)
    text = re.sub(r"^(Rewrite|Narration|Instructor|Audio|Rules|Note|Here is):\s*", "", text, flags=re.I)
    text = re.sub(r"^(In this video|As shown below|We will|You should)\s*,?\s*", "", text, flags=re.I)
    text = re.sub(r"page\s+\d+", "", text, flags=re.I)
    text = re.sub(r"[\"\`\*]", "", text)
    return _clean_text(text)


def _is_action_line(line: str) -> bool:
    line = _LEADING_NUMBER.sub("", line.strip()).strip("-•* ")
    return bool(line) and 2 <= len(line.split()) and len(line) <= 220 and bool(_ACTION_START.match(line))


def _target_query(line: str) -> str:
    line = _LEADING_NUMBER.sub("", line.strip()).strip("-•* ")
    q = _LEAD_VERB.sub("", line, count=1)
    q = _GENERIC_NOUNS.sub("", q)
    q = re.split(r"[.,;:]", q)[0]
    return _clean_text(q).strip(" .,:;\"'")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def _ollama(prompt: str, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout or OLLAMA_TIMEOUT)
        response.raise_for_status()
        text = response.json().get("response", "")
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
    except Exception as exc:
        print(f"[OLLAMA] error: {exc}")
    return None


def _generate_screenshot_narration(text_chunk: str) -> str:
    if not text_chunk:
        return "Continue with the next step shown on screen."

    clean_source = _remove_urls(text_chunk)
    if len(clean_source.split()) <= 20:
        return _clean_spoken_narration(clean_source)

    prompt = f"""
You are a professional software instructor. Rewrite the following step into
ONE concise, natural spoken sentence for a tutorial video.

SOURCE:
{clean_source[:1200]}

RULES:
1. Speak directly to the learner.
2. Keep it between 8 and 24 words.
3. Do NOT include URLs or metadata.
4. Output ONLY valid JSON:
{{"narration": "concise spoken sentence"}}
"""
    result = _ollama(prompt, timeout=20)
    if result and isinstance(result, dict) and result.get("narration"):
        narr = _clean_spoken_narration(result["narration"])
        if narr and len(narr.split()) >= 3:
            return narr

    sentences = [s.strip() for s in re.split(r"[.!?]\s+", clean_source) if len(s.strip()) > 5]
    fallback = sentences[0] if sentences else clean_source[:100]
    return _clean_spoken_narration(fallback) + "."


def _summarize_theory(explanation_text: str) -> List[str]:
    explanation_text = _remove_urls(explanation_text or "").strip()
    if not explanation_text:
        return []

    prompt = f"""
You are writing short study notes from a corporate training document, shown
in the sidebar next to a tutorial video. Read the SOURCE and produce 3 to 6
bullet points capturing the specific concepts, context, or background it
explains -- not click-by-click steps.

Rules:
- Use the SOURCE's own key terms, field names, and product names verbatim --
  do not generalize or paraphrase them into vaguer language.
- Each bullet: one specific idea from the SOURCE, under 20 words.
- No bullet may just restate a heading with no added information.
- Never invent information that is not present in the SOURCE.

SOURCE:
{explanation_text[:3000]}

Return ONLY JSON: {{"bullets": ["...", "..."]}}
"""
    result = _ollama(prompt, timeout=25)
    if result and isinstance(result, dict) and isinstance(result.get("bullets"), list):
        bullets = [_clean_text(b) for b in result["bullets"] if _clean_text(b)]
        bullets = [b for b in bullets if len(b) > 6]
        if bullets:
            return bullets[:6]

    # Fallback (no LLM available): pick clean, substantive sentences instead
    # of naively truncating at a fixed character count, which was cutting
    # sentences mid-word/mid-idea and reading worse than the source text.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", explanation_text) if len(s.strip()) > 12]
    bullets, seen = [], set()
    for s in sentences:
        key = s.lower()[:40]
        if key in seen:
            continue
        seen.add(key)
        bullets.append(s if len(s) <= 220 else s[:217].rsplit(" ", 1)[0].rstrip(",;:") + "…")
        if len(bullets) >= 6:
            break
    return bullets


# ---------------------------------------------------------------------------
# Cursor grounding -- priority order: OCR read directly off the
# screenshot's own pixels (primary, general-purpose text identification) >
# PDF word match (secondary, for the rarer case of a real text layer over
# the image) > vision-model grounding (icons with no readable text) > none.
# Each source is confidence-tagged so the frontend/QA tooling can tell a
# precise hit from a best-effort guess instead of treating every cursor the
# same way. There is deliberately no colour/highlight/marker-based
# detection: grounding is entirely based on identifying the actual words
# named in the action text.
# ---------------------------------------------------------------------------


def _find_word_targets(query: str, words: List[Dict[str, Any]]) -> List[Tuple[float, float, str]]:
    """Returns every page-coordinate hit for query (absolute page units, not
    yet normalized), longest n-gram match length first. A single query word
    like 'Save' commonly appears more than once on a page (once in the
    instructional sentence, once as the on-screen label itself) -- returning
    every hit lets the caller pick whichever one actually falls inside the
    specific screenshot being grounded, instead of blindly taking whatever
    occurs first in reading order."""
    if not words or not query:
        return []
    tokens = [re.sub(r"[^\w@#$%&+\-/]", "", t.lower()) for t in query.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return []
    norm = [re.sub(r"[^\w@#$%&+\-/]", "", str(w.get("text", "")).lower()) for w in words]

    for n in range(min(len(tokens), 5), 0, -1):
        target = tokens[:n]
        hits = []
        for i in range(0, max(0, len(words) - n + 1)):
            if norm[i:i + n] != target:
                continue
            chunk = words[i:i + n]
            x0 = min(float(w["x0"]) for w in chunk)
            y0 = min(float(w["y0"]) for w in chunk)
            x1 = max(float(w["x1"]) for w in chunk)
            y1 = max(float(w["y1"]) for w in chunk)
            hits.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, " ".join(str(w["text"]) for w in chunk)))
        if hits:
            return hits

    hits = []
    for token in tokens:
        if len(token) < 3:
            continue
        for w, nw in zip(words, norm):
            if nw == token:
                cx = (float(w["x0"]) + float(w["x1"])) / 2.0
                cy = (float(w["y0"]) + float(w["y1"])) / 2.0
                hits.append((cx, cy, str(w["text"])))
    return hits


def _map_page_point_to_screenshot(nx: float, ny: float, page_w: float, page_h: float, screenshot: Dict[str, Any]):
    rect = screenshot.get("page_rect")
    if not rect:
        return None
    x0, y0, x1, y1 = rect
    x, y = nx * page_w, ny * page_h
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return None
    return (x - x0) / max(1e-6, x1 - x0), (y - y0) / max(1e-6, y1 - y0)


def _fallback_cursor(shot_path: str):
    """Last-resort generic heuristic: largest plausible UI-control-shaped
    contour. Always tagged with low confidence -- this is a guess, not a
    grounded result, and callers/UI should treat it that way."""
    try:
        img = cv2.imread(str(shot_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_area = None, 0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < w * h * 0.008 or cw < 40 or ch < 16:
                continue
            if cw > w * 0.9 and ch > h * 0.4:
                continue
            if area > best_area:
                best_area, best = area, (x + cw / 2.0, y + ch / 2.0)
        if not best:
            return 0.5, 0.5
        return best[0] / w, best[1] / h
    except Exception:
        return None


@functools.lru_cache(maxsize=512)
def _ocr_words(image_path: str) -> Tuple[Tuple[str, float, float, float, float], ...]:
    """OCRs the screenshot directly and returns (text, x0, y0, x1, y1) in
    that image's own pixel coordinates. Cached per path since several
    actions commonly map to the same screenshot and would otherwise repeat
    the same OCR pass. This is ground truth for what's actually drawn on
    screen -- it works whether or not the PDF happens to carry an
    extractable text layer over the image."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return tuple()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        # Tesseract is noticeably more accurate on small UI text once
        # upscaled -- most screenshots embedded in a PDF are well under
        # the resolution OCR engines are tuned for.
        scale = 2 if max(h, w) < 1600 else 1
        if scale > 1:
            gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        data = pytesseract.image_to_data(gray, output_type=Output.DICT, config="--psm 11")
        words = []
        for i in range(len(data.get("text", []))):
            text = (data["text"][i] or "").strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if not text or conf < OCR_MIN_CONFIDENCE:
                continue
            x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append((text, x / scale, y / scale, (x + bw) / scale, (y + bh) / scale))
        return tuple(words)
    except Exception as e:
        print(f"[OCR] failed on {image_path}: {e}")
        return tuple()


def _find_ocr_target(query: str, image_path: Optional[str]):
    """Finds the query phrase directly in the screenshot's own rendered
    pixels via OCR. This is the primary source for any control with a
    visible text label ('Login', 'Change Password', a dropdown's current
    value, etc.) -- it reads what's actually on screen instead of guessing
    from PDF text placement or trusting a vision model's own coordinate
    space."""
    if not image_path or not query or not os.path.exists(str(image_path)):
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    words = _ocr_words(str(image_path))
    if not words:
        return None

    q_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)]
    if not q_tokens:
        return None
    ocr_tokens = [re.sub(r"[^A-Za-z0-9]", "", t[0]).lower() for t in words]

    for n in range(min(len(q_tokens), 4), 0, -1):
        target = q_tokens[:n]
        for i in range(0, max(0, len(words) - n + 1)):
            if ocr_tokens[i:i + n] != target:
                continue
            chunk = words[i:i + n]
            x0 = min(c[1] for c in chunk)
            y0 = min(c[2] for c in chunk)
            x1 = max(c[3] for c in chunk)
            y1 = max(c[4] for c in chunk)
            matched = " ".join(c[0] for c in chunk)
            return (x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h, matched

    # Fuzzy fallback -- OCR occasionally misreads a character ("Passvvord"),
    # so try approximate matching before giving up on this screenshot.
    query_phrase = " ".join(q_tokens)
    best_ratio, best_hit = 0.0, None
    for window in range(1, min(4, len(words)) + 1):
        for i in range(0, len(words) - window + 1):
            chunk = words[i:i + window]
            phrase = " ".join(re.sub(r"[^A-Za-z0-9]", "", c[0]).lower() for c in chunk)
            if not phrase:
                continue
            ratio = difflib.SequenceMatcher(None, phrase, query_phrase).ratio()
            if ratio > best_ratio:
                x0 = min(c[1] for c in chunk)
                y0 = min(c[2] for c in chunk)
                x1 = max(c[3] for c in chunk)
                y1 = max(c[4] for c in chunk)
                best_ratio = ratio
                best_hit = ((x0 + x1) / 2.0 / w, (y0 + y1) / 2.0 / h, " ".join(c[0] for c in chunk))

    if best_hit and best_ratio >= OCR_FUZZY_MIN_RATIO:
        return best_hit
    return None


def _ground_action(
    query: str,
    page: Optional[Dict[str, Any]],
    screenshot: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not screenshot:
        return None

    if query and screenshot.get("path"):
        ocr_hit = _find_ocr_target(query, screenshot["path"])
        if ocr_hit:
            nx, ny, matched = ocr_hit
            return {"point": [nx, ny], "source": "ocr", "confidence": 0.90, "target_name": matched}

    if page and screenshot.get("page_rect") and query:
        page_w, page_h = page.get("width") or 0, page.get("height") or 0
        if page_w and page_h:
            for ax, ay, matched in _find_word_targets(query, page.get("words") or []):
                nx, ny = ax / page_w, ay / page_h
                mapped = _map_page_point_to_screenshot(nx, ny, page_w, page_h, screenshot)
                if mapped:
                    return {"point": list(mapped), "source": "text", "confidence": 0.80, "target_name": matched}

    if VISION_ENABLED and screenshot.get("path"):
        vis = analyze_image(screenshot["path"], query or "the relevant control")
        if vis.get("found") and vis.get("confidence", 0) >= VISION_MIN_CONFIDENCE:
            return {
                "point": vis["click_point"],
                "source": "vision",
                "confidence": vis["confidence"],
                "target_name": vis.get("target_name", ""),
            }

    if screenshot.get("path"):
        # Diagnostic only -- log what the generic heuristic would have
        # guessed, but never surface it as the actual cursor. A scene with
        # no confidently-grounded target plays narration/caption over the
        # plain screenshot with no cursor at all (see video_service, which
        # already skips drawing when targets is empty).
        diag = _fallback_cursor(screenshot["path"])
        print(f"[GROUND] No confident target for '{query}' on {screenshot['path']} "
              f"(heuristic-only guess would have been {diag}, not used)")

    return {"point": None, "source": "none", "confidence": 0.0, "target_name": query}


# ---------------------------------------------------------------------------
# Section grouping
# ---------------------------------------------------------------------------

def _page_has_action(page: Dict[str, Any]) -> bool:
    return any(_is_action_line(l) for l in page.get("text", "").splitlines())


def _fallback_split_by_activity(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current = None
    for p in pages:
        if current is None:
            current = {"title": f"Section {len(sections) + 1}", "pages": []}
            sections.append(current)
        current["pages"].append(p)
        if _page_has_action(p):
            current = None
    return sections


def _build_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    content_pages = [p for p in pages if not p.get("skip_reason")]
    if not content_pages:
        # Pathological case (everything got classified as boilerplate) --
        # degrade gracefully instead of producing an empty tutorial.
        content_pages = list(pages)

    sections: List[Dict[str, Any]] = []
    current = None
    for p in content_pages:
        heading = p.get("heading")
        if heading or current is None:
            current = {"title": heading or f"Section {len(sections) + 1}", "pages": []}
            sections.append(current)
        current["pages"].append(p)

    if len(sections) <= 1 and len(content_pages) > 6:
        return _fallback_split_by_activity(content_pages)

    return sections


# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------

def build_tutorial_plan(
    data: Dict[str, Any],
    narration_language: str = "en-us",
    progress_callback=None,
) -> Dict[str, Any]:
    pages = data.get("pages", [])
    screenshots = data.get("screenshots", [])

    shots_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for s in screenshots:
        shots_by_page.setdefault(s["page"], []).append(s)
    for lst in shots_by_page.values():
        lst.sort(key=lambda s: s.get("screenshot_index", 0))

    sections_raw = _build_sections(pages)
    if progress_callback:
        progress_callback(5, f"Identified {len(sections_raw)} sections")

    steps: List[Dict[str, Any]] = []
    sections_out: List[Dict[str, Any]] = []
    step_id = 0

    for sec_idx, sec in enumerate(sections_raw):
        explanation_chunks: List[str] = []
        step_ids_for_section: List[int] = []

        # Gather this section's screenshots and action lines across ALL of
        # its pages, in reading order. A section (topic) -- not a single
        # page -- is the natural scope for matching N actions to M
        # screenshots, since a topic's steps commonly span more than one
        # page.
        section_shots: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        section_actions: List[Tuple[Dict[str, Any], str]] = []

        for page in sec["pages"]:
            lines = [l.strip() for l in page.get("text", "").splitlines() if l.strip()]
            heading = page.get("heading")
            content_lines = [l for l in lines if l != heading]
            action_lines = [l for l in content_lines if _is_action_line(l)]
            explanation_lines = [l for l in content_lines if not _is_action_line(l)]
            if explanation_lines:
                explanation_chunks.append(" ".join(explanation_lines))

            page_shots = [s for s in shots_by_page.get(page["page"], []) if not s.get("is_full_page")]
            section_shots.extend((page, s) for s in page_shots)
            section_actions.extend((page, l) for l in action_lines)

        if section_actions:
            n_actions, n_shots = len(section_actions), len(section_shots)
            assigned_idx = (
                [min(n_shots - 1, (i * n_shots) // n_actions) for i in range(n_actions)]
                if n_shots > 0 else [None] * n_actions
            )

            for i, (page, line) in enumerate(section_actions):
                query = _target_query(line)
                s_idx = assigned_idx[i]
                grounding, chosen_page, chosen_shot = None, page, None

                if s_idx is not None:
                    chosen_page, chosen_shot = section_shots[s_idx]
                    grounding = _ground_action(query, chosen_page, chosen_shot)

                # Validation / reassignment: the proportional guess is only
                # a starting point. If it didn't confidently find the named
                # element, check whether a DIFFERENT screenshot in this
                # same section does, and reassign to that one instead of
                # keeping a wrong-but-confident-looking placement.
                if (grounding is None or grounding.get("source") == "none") and n_shots > 1:
                    for alt_page, alt_shot in section_shots:
                        if alt_shot is chosen_shot:
                            continue
                        alt_grounding = _ground_action(query, alt_page, alt_shot)
                        if alt_grounding and alt_grounding.get("source") != "none":
                            grounding = alt_grounding
                            chosen_page, chosen_shot = alt_page, alt_shot
                            break

                narration = _generate_screenshot_narration(line)
                step_id += 1
                steps.append({
                    "id": step_id,
                    "section_id": sec_idx + 1,
                    "page_num": chosen_page["page"],
                    "kind": "action",
                    "title": _clean_text(line)[:80],
                    "narration": narration,
                    "tts_narration": narration,
                    "caption": narration,
                    "screenshot": chosen_shot.get("path") if chosen_shot else None,
                    "is_full_page": chosen_shot is None,
                    "cursor": grounding["point"] if grounding else None,
                    "targets": [grounding["point"]] if grounding and grounding.get("point") else [],
                    "cursor_source": grounding["source"] if grounding else "none",
                    "cursor_confidence": grounding["confidence"] if grounding else 0.0,
                    "cursor_target_name": grounding["target_name"] if grounding else "",
                })
                step_ids_for_section.append(step_id)

        sections_out.append({
            "id": sec_idx + 1,
            "title": _clean_text(sec["title"])[:90],
            "theory_bullets": _summarize_theory(" ".join(explanation_chunks)),
            "step_ids": step_ids_for_section,
        })

        if progress_callback:
            progress = int(10 + 80 * ((sec_idx + 1) / max(1, len(sections_raw))))
            progress_callback(progress, f"Prepared section {sec_idx + 1} of {len(sections_raw)}")

    doc_title = next((p.get("heading") for p in pages if p.get("heading")), None) or "Software Training Tutorial"

    plan = {
        "title": doc_title,
        "description": "Enterprise tutorial generated per PDF screenshot with section navigation.",
        "narration_language": narration_language,
        "sections": sections_out,
        "steps": steps,
        "scene_count": len(steps),
    }
    print(f"[PLAN] Created {len(steps)} action scenes across {len(sections_out)} sections "
          f"({sum(1 for s in sections_out if not s['step_ids'])} theory-only).")
    return plan
