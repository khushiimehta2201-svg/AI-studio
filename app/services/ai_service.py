import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import requests

from app.services.local_vision import analyze_image

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
VISION_ENABLED = os.getenv("OLLAMA_VISION_ENABLED", "true").lower() == "true"
VISION_MIN_CONFIDENCE = float(os.getenv("OLLAMA_VISION_MIN_CONFIDENCE", "0.50"))

ACTION_VERBS = (
    "open", "click", "select", "enter", "type", "browse", "launch",
    "choose", "apply", "create", "save", "run", "fill", "search",
    "locate", "expand", "collapse", "double click", "right click", "drag", "press",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _extract_urls(text: str) -> List[str]:
    return [u.rstrip(".,;:") for u in re.findall(r"https?://[^\s)\]>]+", text, flags=re.IGNORECASE)]


def _remove_urls(text: str) -> str:
    return re.sub(r"https?://[^\s)\]>]+", "", text, flags=re.IGNORECASE).strip()


def _clean_spoken_narration(raw_narration: str) -> str:
    if not raw_narration:
        return ""
    text = _remove_urls(raw_narration)
    text = re.sub(r"^(Rewrite|Narration|Instructor|Audio|Rules|Note|Here is):\s*", "", text, flags=re.I)
    text = re.sub(r"^(In this video|As shown below|We will|You should)\s*,?\s*", "", text, flags=re.I)
    text = re.sub(r"page\s+\d+", "", text, flags=re.I)
    text = re.sub(r"[\"\`\*]", "", text)
    return _clean_text(text)


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
        data = response.json()
        text = data.get("response", "")
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


def _generate_screenshot_narration(text_chunk: str, is_action: bool) -> str:
    if not text_chunk:
        return "Review the configuration options displayed on screen."

    clean_source = _remove_urls(text_chunk)
    if len(clean_source.split()) <= 20:
        return _clean_spoken_narration(clean_source)

    prompt = f"""
You are a professional software instructor. Summarize the following screen instruction into ONE concise, natural spoken sentence for a tutorial video.

SOURCE:
{clean_source[:1200]}

RULES:
1. Speak directly to the learner.
2. Keep it between 12 and 30 words.
3. Do NOT include instructions, URLs, or metadata.
4. Output ONLY valid JSON:
{{"narration": "concise spoken sentence"}}
"""
    result = _ollama(prompt, timeout=20)
    if result and isinstance(result, dict) and result.get("narration"):
        narr = _clean_spoken_narration(result["narration"])
        if narr and len(narr.split()) >= 3:
            return narr

    # Fallback: First sentence of source
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", clean_source) if len(s.strip()) > 5]
    fallback = sentences[0] if sentences else clean_source[:100]
    return _clean_spoken_narration(fallback) + "."


def build_tutorial_plan(
    data: Dict[str, Any],
    narration_language: str = "en-us",
    progress_callback=None,
) -> Dict[str, Any]:
    screenshots = data.get("screenshots", [])
    pages = {p["page"]: p.get("text", "") for p in data.get("pages", [])}

    if not screenshots:
        raise RuntimeError("No visual scenes or screenshots found in document.")

    steps = []
    total = len(screenshots)

    if progress_callback:
        progress_callback(5, f"Structuring {total} screenshot scenes")

    for i, shot in enumerate(screenshots):
        page_no = shot.get("page", 1)
        page_text = pages.get(page_no, "")
        lines = [l.strip() for l in page_text.splitlines() if l.strip()]

        # 1. Identify Action vs Explanation
        has_action = any(re.match(rf"^{re.escape(v)}\b", l.lower()) for l in lines for v in ACTION_VERBS)
        kind = "action" if has_action else "explanation"

        # 2. Extract Step Title
        first_line = lines[0] if lines else "Software Procedure"
        title = re.sub(r"^\d+(\.\d+)*\s*", "", first_line).strip()[:65]

        # 3. Target cursor coordinates inside the screenshot (support multiple sequential targets)
        cursor_targets = list(shot.get("targets", []))
        cursor_target = shot.get("primary_target")
        shot_path = shot.get("path")

        # Fallback to vision model ONLY if no red/orange boxes were detected and an image exists
        if not cursor_targets and kind == "action" and VISION_ENABLED and shot_path:
            vis = analyze_image(shot_path, title)
            if vis.get("found") and vis.get("confidence", 0) >= VISION_MIN_CONFIDENCE:
                cursor_target = vis.get("click_point")
                cursor_targets = [cursor_target]

        # 4. Generate clean spoken narration
        narration = _generate_screenshot_narration(page_text, is_action=(kind == "action"))

        # 5. Format on-screen caption (shows URLs if present)
        urls = _extract_urls(page_text)
        caption = narration
        if urls:
            caption += f"\nURL: {urls[0]}"

        # 6. Build final structured step
        step = {
            "id": i + 1,
            "number": i + 1,
            "page_num": page_no,
            "screenshot_index": shot.get("screenshot_index", 1),
            "kind": kind,
            "title": title,
            "content": page_text,
            "narration": narration,
            "tts_narration": narration,
            "caption": caption,
            "screenshot": shot_path,
            "is_full_page": shot.get("is_full_page", False),
            "cursor": cursor_target,
            "targets": cursor_targets,
            "actions": [{"type": "guided_cursor", "target": t} for t in cursor_targets],
        }
        steps.append(step)

        if progress_callback:
            progress = int(10 + 85 * ((i + 1) / float(total)))
            progress_callback(progress, f"Prepared screenshot scene {i + 1} of {total}")

    doc_title = steps[0]["title"] if steps else "Software Training Tutorial"
    plan = {
        "title": doc_title,
        "description": "Enterprise tutorial generated per UI screenshot.",
        "narration_language": narration_language,
        "steps": steps,
        "scene_count": len(steps),
    }

    print(f"[PLAN] Created {len(steps)} scenes from extracted UI screenshots.")
    return plan
