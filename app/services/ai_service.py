import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import requests

from app.services.local_vision import analyze_image

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
VISION_ENABLED = os.getenv("OLLAMA_VISION_ENABLED", "true").lower() == "true"
VISION_MIN_CONFIDENCE = float(os.getenv("OLLAMA_VISION_MIN_CONFIDENCE", "0.55"))

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
# Cursor grounding -- priority order: explicit annotation > exact word match
# on the page > vision-model grounding > generic visual heuristic > none.
# Each source is confidence-tagged so the frontend/QA tooling can tell a
# precise hit from a best-effort guess instead of treating every cursor the
# same way.
# ---------------------------------------------------------------------------

def _color_box_target(screenshot: Dict[str, Any], idx_in_shot: int, count_in_shot: int) -> Optional[List[float]]:
    targets = screenshot.get("targets") or []
    if not targets:
        return None
    if len(targets) == count_in_shot and 0 <= idx_in_shot < len(targets):
        return list(targets[idx_in_shot])
    if count_in_shot == 1 and len(targets) >= 1:
        return list(targets[0])
    return None


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


def _ground_action(
    query: str,
    page: Optional[Dict[str, Any]],
    screenshot: Optional[Dict[str, Any]],
    idx_in_shot: int,
    count_in_shot: int,
) -> Optional[Dict[str, Any]]:
    if not screenshot:
        return None

    box_pt = _color_box_target(screenshot, idx_in_shot, count_in_shot)
    if box_pt:
        return {"point": box_pt, "source": "annotation", "confidence": 0.95, "target_name": query}

    if page and screenshot.get("page_rect") and query:
        page_w, page_h = page.get("width") or 0, page.get("height") or 0
        if page_w and page_h:
            for ax, ay, matched in _find_word_targets(query, page.get("words") or []):
                nx, ny = ax / page_w, ay / page_h
                mapped = _map_page_point_to_screenshot(nx, ny, page_w, page_h, screenshot)
                if mapped:
                    return {"point": list(mapped), "source": "text", "confidence": 0.85, "target_name": matched}

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
        pt = _fallback_cursor(screenshot["path"])
        if pt:
            return {"point": list(pt), "source": "heuristic", "confidence": 0.2, "target_name": query}

    return None


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

        for page in sec["pages"]:
            page_no = page["page"]
            lines = [l.strip() for l in page.get("text", "").splitlines() if l.strip()]
            action_lines = [l for l in lines if _is_action_line(l)]
            explanation_lines = [l for l in lines if not _is_action_line(l)]
            if explanation_lines:
                explanation_chunks.append(" ".join(explanation_lines))

            if not action_lines:
                continue

            page_shots = shots_by_page.get(page_no, [])
            real_shots = [s for s in page_shots if not s.get("is_full_page")]

            if real_shots:
                n_actions, n_shots = len(action_lines), len(real_shots)
                # Spread actions across this page's screenshots in reading
                # order instead of pinning everything to the first region.
                assigned = [min(n_shots - 1, (i * n_shots) // n_actions) for i in range(n_actions)]
                counts: Dict[int, int] = {}
                for s_idx in assigned:
                    counts[s_idx] = counts.get(s_idx, 0) + 1
                running: Dict[int, int] = {}

                for i, line in enumerate(action_lines):
                    s_idx = assigned[i]
                    screenshot = real_shots[s_idx]
                    idx_in_shot = running.get(s_idx, 0)
                    running[s_idx] = idx_in_shot + 1

                    query = _target_query(line)
                    grounding = _ground_action(query, page, screenshot, idx_in_shot, counts[s_idx])
                    narration = _generate_screenshot_narration(line)

                    step_id += 1
                    steps.append({
                        "id": step_id,
                        "section_id": sec_idx + 1,
                        "page_num": page_no,
                        "kind": "action",
                        "title": _clean_text(line)[:80],
                        "narration": narration,
                        "tts_narration": narration,
                        "caption": narration,
                        "screenshot": screenshot.get("path"),
                        "is_full_page": False,
                        "cursor": grounding["point"] if grounding else None,
                        "targets": [grounding["point"]] if grounding else [],
                        "cursor_source": grounding["source"] if grounding else "none",
                        "cursor_confidence": grounding["confidence"] if grounding else 0.0,
                        "cursor_target_name": grounding["target_name"] if grounding else "",
                    })
                    step_ids_for_section.append(step_id)
            else:
                for line in action_lines:
                    narration = _generate_screenshot_narration(line)
                    step_id += 1
                    steps.append({
                        "id": step_id,
                        "section_id": sec_idx + 1,
                        "page_num": page_no,
                        "kind": "action",
                        "title": _clean_text(line)[:80],
                        "narration": narration,
                        "tts_narration": narration,
                        "caption": narration,
                        "screenshot": None,
                        "is_full_page": True,
                        "cursor": None,
                        "targets": [],
                        "cursor_source": "none",
                        "cursor_confidence": 0.0,
                        "cursor_target_name": "",
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
