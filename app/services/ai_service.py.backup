from pathlib import Path
import json
import os
import re
import time
import requests
from dotenv import load_dotenv
load_dotenv()
from app.services.local_vision import analyze_image

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "15"))
OLLAMA_NARRATION_ENABLED = os.getenv("OLLAMA_NARRATION_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
VISION_ENABLED = os.getenv("OLLAMA_VISION_ENABLED", "true").strip().lower() not in {"0", "false", "no", "off"}
VISION_TIMEOUT = int(os.getenv("OLLAMA_VISION_TIMEOUT", "20"))
VISION_MIN_CONFIDENCE = float(os.getenv("OLLAMA_VISION_MIN_CONFIDENCE", "0.55"))


def _clean(v):
    return re.sub(r"\s+", " ", str(v or "")).strip()


def _parse_json(raw):
    raw = str(raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except Exception:
        a, b = raw.find("{"), raw.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("Ollama returned invalid JSON")
        value = json.loads(raw[a:b + 1])
    return value if isinstance(value, dict) else {"items": value}


def _ollama(prompt, timeout=None, num_predict=280, temperature=0.0):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": str(prompt),
        "stream": False,
        "format": "json",
        "keep_alive": "0",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": 4096,
        },
    }
    start = time.time()
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            f"{OLLAMA_HOST.rstrip('/')}/api/generate",
            json=payload,
            timeout=timeout or OLLAMA_TIMEOUT,
        )
    print(f"[AI] Ollama {time.time() - start:.2f}s prompt_chars={len(str(prompt))}")
    response.raise_for_status()
    return _parse_json(response.json().get("response", ""))


_NUMBERED = re.compile(r"^(?:step\s*)?(\d+)\s*[\).:\-]\s*(.+?)\s*$", re.I)
_ACTION_START = re.compile(
    r"^(?:click|double[- ]click|right[- ]click|select|choose|open|launch|press|type|enter|set|fill|check|uncheck|go to|navigate to|create|add|remove|delete|save|submit|upload|download|browse|search|filter|attach|confirm|cancel|expand|collapse|drag|drop|login|sign in)\b",
    re.I,
)


def _numbered_steps(pages):
    candidates = []
    for p in pages or []:
        for idx, raw in enumerate(str(p.get("text", "")).splitlines()):
            line = _clean(raw).strip("-•* ")
            m = _NUMBERED.match(line)
            if not m:
                continue
            title = _clean(m.group(2))
            if 3 <= len(title) <= 220 and not title.lower().startswith(("note:", "warning:", "example:")):
                candidates.append({
                    "number": int(m.group(1)),
                    "title": title,
                    "source_page": p.get("page"),
                    "line_index": idx,
                })
    if not candidates:
        return []
    runs, cur, prev = [], [], None
    for x in candidates:
        if prev is None or x["number"] == prev + 1:
            cur.append(x)
        elif x["number"] == 1:
            if cur:
                runs.append(cur)
            cur = [x]
        else:
            if cur:
                runs.append(cur)
            cur = [x]
        prev = x["number"]
    if cur:
        runs.append(cur)
    runs.sort(key=lambda r: (len(r), r[0]["number"] == 1), reverse=True)
    result, seen = [], set()
    for x in runs[0]:
        if x["number"] not in seen:
            seen.add(x["number"])
            result.append(x)
    return result


def _action_heuristic(pages):
    out, seen = [], set()
    for p in pages or []:
        for idx, raw in enumerate(str(p.get("text", "")).splitlines()):
            line = _clean(raw).strip("-•* ")
            if not line or len(line) > 220 or not _ACTION_START.match(line):
                continue
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append({"number": len(out) + 1, "title": line, "source_page": p.get("page"), "line_index": idx})
    return out


def _ai_extract_actions(pages):
    if not (OLLAMA_NARRATION_ENABLED and pages):
        return []
    source = "\n\n".join(
        f"PAGE {p.get('page')}: {_clean(p.get('text',''))[:900]}"
        for p in pages if _clean(p.get("text", ""))
    )[:9000]
    if not source:
        return []
    prompt = f"""
Identify ONLY the explicit user actions that form the software procedure in this training document.
Exclude introductions, descriptions, notes, warnings, logos, examples, and outcomes.
Preserve the document order.
Return ONLY JSON: {{"steps":[{{"number":1,"title":"...","source_page":1}}]}}

{source}
"""
    try:
        parsed = _ollama(prompt, timeout=OLLAMA_TIMEOUT, num_predict=260, temperature=0.0)
        result = []
        for i, item in enumerate(parsed.get("steps", []), 1):
            if not isinstance(item, dict) or not _clean(item.get("title")):
                continue
            try:
                page = int(item.get("source_page")) if item.get("source_page") is not None else None
            except Exception:
                page = None
            result.append({"number": i, "title": _clean(item["title"]), "source_page": page, "line_index": None})
        return result
    except Exception as exc:
        print(f"[AI] action extraction unavailable: {exc}")
        return []


def _page_map(pages):
    return {int(p["page"]): p for p in pages or [] if p.get("page") is not None}


def _target_query(title):
    title = _clean(title)
    m = re.match(
        r"^(?:double[- ]click|right[- ]click|click|select|choose|open|launch|press|type|enter|set|fill|check|uncheck|go to|navigate to|drag|drop)\s+(?:the\s+)?(.+)$",
        title,
        re.I,
    )
    q = _clean(m.group(1)) if m else title
    q = re.sub(r"\b(button|field|menu|tab|link|icon|option|box|dropdown|drop-down|area|screen|control)\b", "", q, flags=re.I)
    return _clean(q).strip(" .,:;\"'")


def _find_text_target(step, page):
    if not page:
        return None
    words = page.get("words") or []
    if not words:
        return None
    query = _target_query(step.get("title", ""))
    tokens = [re.sub(r"[^\w@#$%&+\-/]", "", t.lower()) for t in re.findall(r"[\w@#$%&+\-/]+", query)]
    tokens = [t for t in tokens if t]
    norm = [re.sub(r"[^\w@#$%&+\-/]", "", _clean(w.get("text", "")).lower()) for w in words]
    for n in range(min(len(tokens), 5), 0, -1):
        target = tokens[:n]
        for i in range(0, max(0, len(words) - n + 1)):
            if norm[i:i + n] != target:
                continue
            chunk = words[i:i + n]
            x0 = min(float(w["x0"]) for w in chunk); y0 = min(float(w["y0"]) for w in chunk)
            x1 = max(float(w["x1"]) for w in chunk); y1 = max(float(w["y1"]) for w in chunk)
            return (
                ((x0 + x1) / 2) / float(page.get("width") or 1),
                ((y0 + y1) / 2) / float(page.get("height") or 1),
                query,
            )
    for token in tokens:
        if len(token) < 3:
            continue
        for w, nw in zip(words, norm):
            if nw == token:
                return (
                    ((float(w["x0"]) + float(w["x1"])) / 2) / float(page.get("width") or 1),
                    ((float(w["y0"]) + float(w["y1"])) / 2) / float(page.get("height") or 1),
                    token,
                )
    return None


def _choose_screenshot(step, screenshots, pages):
    page_no = step.get("source_page")
    candidates = [s for s in screenshots if s.get("page") == page_no and s.get("source_type") == "embedded_screenshot"]
    if not candidates:
        return None
    page = _page_map(pages).get(page_no)
    target = _find_text_target(step, page)
    page_h = float(page.get("height") or 1) if page else 1

    def rank(s):
        score = float(s.get("score", 0) or 0)
        if s.get("repeated_visual"):
            score -= 2.0
        box = s.get("bbox") or [0, 0, 0, 0]
        if target and page:
            target_y = target[1] * page_h
            image_cy = (float(box[1]) + float(box[3])) / 2
            score -= abs(image_cy - target_y) / page_h * 3.0
        return score
    return max(candidates, key=rank)


def _page_cursor_to_screenshot(page_target, screenshot, page):
    try:
        nx, ny = float(page_target[0]), float(page_target[1])
        pw, ph = float(page.get("width") or 1), float(page.get("height") or 1)
        bx0, by0, bx1, by1 = map(float, screenshot.get("bbox") or [0, 0, pw, ph])
        x, y = nx * pw, ny * ph
        if not (bx0 <= x <= bx1 and by0 <= y <= by1):
            return None
        return (
            (x - bx0) / max(1e-6, bx1 - bx0),
            (y - by0) / max(1e-6, by1 - by0),
        )
    except Exception:
        return None


def _vision_target(step, screenshot):
    if not (VISION_ENABLED and screenshot):
        return None
    try:
        result = analyze_image(
            screenshot["path"],
            context=(
                f"Action: {step.get('title','')}. Find the exact visible interactive control where the action happens. "
                "Return its bounding box and confidence."
            ),
            timeout=VISION_TIMEOUT,
        )
        pos = result.get("approximate_position") or {}
        conf = float(result.get("confidence", 0) or 0)
        if conf < VISION_MIN_CONFIDENCE or not isinstance(pos, dict):
            return None
        x = float(pos.get("x")); y = float(pos.get("y"))
        w = float(pos.get("width", 0) or 0); h = float(pos.get("height", 0) or 0)
        if w:
            x += w / 2
        if h:
            y += h / 2
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return None
        return {"cursor": (x, y), "confidence": conf, "target": _clean(result.get("target_name", ""))}
    except Exception as exc:
        print(f"[VISION] target failed: {exc}")
        return None


def _fallback_cursor(screenshot, step):
    """Generic visual fallback: pick a large UI-like region, never a document page."""
    if not screenshot:
        return None
    try:
        from PIL import Image, ImageChops, ImageFilter, ImageStat
        import cv2
        import numpy as np

        image = cv2.imread(str(screenshot["path"]))
        if image is None:
            return None
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        action = _clean(step.get("title", "")).lower()
        text_like = any(k in action for k in ("enter", "type", "fill", "input", "password", "username"))
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < w * h * 0.008 or cw < 60 or ch < 18:
                continue
            if cw > w * 0.9 and ch > h * 0.35:
                continue
            ratio = cw / max(1, ch)
            if text_like and not (ratio >= 2.2 and ch <= h * 0.15):
                continue
            if not text_like and not (0.7 <= ratio <= 8.0):
                continue
            candidates.append((area, (x + cw / 2, y + ch / 2)))
        if not candidates:
            return (0.50, 0.50)
        candidates.sort(reverse=True)
        cx, cy = candidates[0][1]
        return (cx / w, cy / h)
    except Exception:
        return (0.50, 0.50)


def _fallback_narration(title):
    return f"{_clean(title)}."


def _ai_narration(actions, pages):
    if not (OLLAMA_NARRATION_ENABLED and actions):
        return {}
    pm = _page_map(pages)
    units = []
    for x in actions:
        p = pm.get(x.get("source_page"))
        txt = _clean(p.get("text", ""))[:700] if p else ""
        units.append(f"STEP {x['number']}: {x['title']}\nSOURCE PAGE: {x.get('source_page')}\nPAGE CONTEXT: {txt}")
    prompt = f"""
Create concise professional English narration for each software training step.
Each narration MUST state the action.
Add one short purpose clause ONLY when the supplied source context clearly gives a purpose or reason.
Do not read or summarize the rest of the document.
Maximum 24 words per narration.
Keep UI labels exactly as written.
Return ONLY JSON: {{"steps":[{{"id":1,"narration":"..."}}]}}

{chr(10).join(units)[:12000]}
"""
    try:
        data = _ollama(prompt, timeout=OLLAMA_TIMEOUT, num_predict=min(500, max(180, len(actions) * 24)), temperature=0.0)
        return {
            int(x["id"]): _clean(x.get("narration", ""))
            for x in data.get("steps", [])
            if isinstance(x, dict) and str(x.get("id", "")).isdigit()
        }
    except Exception as exc:
        print(f"[AI] concise narration unavailable; using deterministic narration: {exc}")
        return {}


def build_tutorial_plan(data, narration_language="en-us", progress_callback=None):
    pages = list((data or {}).get("pages", []) or []) if isinstance(data, dict) else []
    screenshots = list((data or {}).get("screenshots", []) or []) if isinstance(data, dict) else []
    document_text = str((data or {}).get("text", "")) if isinstance(data, dict) else str(data or "")

    def report(p, message):
        if progress_callback:
            progress_callback(p, message)

    report(30, "Identifying procedure steps")
    actions = _numbered_steps(pages) or _action_heuristic(pages) or _ai_extract_actions(pages)
    if not actions:
        raise ValueError("No actionable procedure steps could be identified in the uploaded PDF.")

    report(34, f"Found {len(actions)} procedure steps")
    report(36, "Generating concise English narration")
    narr = _ai_narration(actions, pages)

    report(39, "Finding screenshots and cursor targets")
    final = []
    pm = _page_map(pages)

    for idx, action in enumerate(actions, 1):
        shot = _choose_screenshot(action, screenshots, pages)
        page = pm.get(action.get("source_page"))
        cursor = None
        target = ""
        conf = 0.0

        text_target = _find_text_target(action, page)
        if text_target and shot:
            cursor = _page_cursor_to_screenshot(text_target, shot, page)
            if cursor:
                target = text_target[2]
                conf = 0.70

        if shot and cursor is None:
            vision = _vision_target(action, shot)
            if vision:
                cursor = vision["cursor"]
                target = vision["target"]
                conf = vision["confidence"]

        if shot and cursor is None:
            cursor = _fallback_cursor(shot, action)
            target = target or _target_query(action.get("title", ""))
            conf = 0.25

        narration = narr.get(idx) or _fallback_narration(action["title"])
        final.append({
            "id": idx,
            "kind": "action",
            "number": idx,
            "title": _clean(action["title"]),
            "narration": narration,
            "tts_narration": narration,
            "caption": narration,
            "screenshot": shot.get("path") if isinstance(shot, dict) else None,
            "screenshot_page": shot.get("page") if isinstance(shot, dict) else None,
            "source_page": action.get("source_page"),
            "cursor": cursor,
            "actions": [{"type": "guided_cursor", "target": cursor}],
            "visual_type": shot.get("source_type", "none") if isinstance(shot, dict) else "none",
            "visual_target": target,
            "visual_confidence": conf,
        })
        report(39 + int(9 * idx / max(1, len(actions))), f"Prepared step {idx} of {len(actions)}")

    report(48, "Tutorial plan ready")
    title = _clean(next((x for x in document_text.splitlines() if 4 <= len(_clean(x)) <= 100), "AI Learning Tutorial"))
    return {
        "title": title or "AI Learning Tutorial",
        "description": "Concise English action tutorial using the document's relevant embedded screenshots with generic cursor guidance.",
        "source": "uploaded PDF",
        "narration_language": "en-us",
        "steps": final,
        "screenshot_count": len(screenshots),
        "document_text": document_text,
    }
