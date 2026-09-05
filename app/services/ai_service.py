import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import requests

from app.services.local_vision import analyze_image

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11435").rstrip("/")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"{OLLAMA_HOST}/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))
VISION_ENABLED = os.getenv("OLLAMA_VISION_ENABLED", "true").lower() == "true"
VISION_MIN_CONFIDENCE = float(os.getenv("OLLAMA_VISION_MIN_CONFIDENCE", "0.50"))

TRANSITIONS = [
    "To begin,",
    "Now,",
    "Next,",
    "Then,",
    "In the following step,",
    "Proceed by selecting,",
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _sanitize_for_speech(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"^\s*(?:(?:step\s*)?\d+(?:[\.\)]\d+)*[\.\)]?|[A-Za-z][\.\)])\s*", "", text, flags=re.I)
    text = re.sub(r"^(Rewrite|Narration|Instructor|Audio|Rules|Note|Here is):\s*", "", text, flags=re.I)
    text = re.sub(r"^(In this video|In this document|As shown below|We will cover|You should)\s*,?\s*", "", text, flags=re.I)
    text = re.sub(r"\bpage\s+\d+\b", "", text, flags=re.I)
    text = re.sub(r"https?://[^\s)\]>]+", "", text, flags=re.I)
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


def _generate_instructor_voice(text_chunk: str, scene_type: str, title: str, step_idx: int = 0) -> str:
    clean = _sanitize_for_speech(text_chunk)
    clean_title = _sanitize_for_speech(title)

    if scene_type == "title":
        return f"Welcome to this training module on {clean_title}. Let us proceed with the system workflow."

    if scene_type == "toc":
        return "Here is the table of contents outlining the primary topics and workflows covered in this guide."

    if scene_type == "table":
        return f"This reference table outlines the required parameters and configuration values for {clean_title}."

    if scene_type == "screenshot":
        transition = TRANSITIONS[step_idx % len(TRANSITIONS)]
        act_clean = re.sub(r"^\s*(?:\d+[\.\)]|\*|-)\s*", "", clean)
        return f"{transition} {act_clean.lstrip()}."

    prompt = f"""
You are an enterprise software training instructor. Explain this topic clearly to the learner.

TOPIC: {clean_title}
SOURCE CONTENT:
{clean[:1400]}

RULES:
1. Explain the purpose, rules, or use cases in natural, engaging spoken English.
2. DO NOT speak section numbers (like 3.5), bullet letters, or page numbers.
3. DO NOT say "In this video" or "As shown below".
4. Keep the narration between 20 and 38 words.
5. Output ONLY valid JSON:
{{"narration": "clear spoken explanation"}}
"""
    result = _ollama(prompt, timeout=22)
    if result and isinstance(result, dict) and result.get("narration"):
        narr = _sanitize_for_speech(result["narration"])
        if narr and len(narr.split()) >= 4:
            return narr

    sentences = [s.strip() for s in re.split(r"[.!?]\s+", clean) if len(s.strip()) > 5]
    fallback = ". ".join(sentences[:2]) if sentences else clean[:110]
    return _sanitize_for_speech(fallback) + "."


def _extract_theory_card_content(text: str) -> Tuple[str, List[str]]:
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.lower().startswith("page")]
    if not lines:
        return "Review the documented system parameters and guidelines.", []

    paragraphs = []
    bullet_items = []
    for l in lines[1:]:
        l_san = _sanitize_for_speech(l)
        if not l_san:
            continue
        if l.startswith(("-", "•", "*", "1.", "2.", "3.", "1)", "2)")):
            bullet_items.append(l_san)
        elif len(l_san) > 20:
            paragraphs.append(l_san)

    overview_para = " ".join(paragraphs[:3])[:220]
    if not overview_para and lines:
        overview_para = _sanitize_for_speech(lines[0])

    key_bullets = bullet_items[:3] if bullet_items else [l for l in lines[1:4] if len(_sanitize_for_speech(l)) > 10]
    return overview_para, key_bullets


def build_tutorial_plan(data: Dict[str, Any], narration_language: str = "en-us", progress_callback=None) -> Dict[str, Any]:
    scenes = data.get("scenes", [])
    if not scenes:
        raise RuntimeError("No tutorial scenes extracted from document.")

    steps = []
    total = len(scenes)

    if progress_callback:
        progress_callback(5, f"Synthesizing {total} structured tutorial scenes")

    for i, sc in enumerate(scenes):
        scene_type = sc.get("type", "theory")
        text = sc.get("text", "")
        title = _sanitize_for_speech(sc.get("title", "System Procedure"))
        step_idx = sc.get("step_index", 0)

        overview_para, key_bullets = _extract_theory_card_content(text)

        screenshots = sc.get("screenshots", [])
        shot_path = screenshots[0]["path"] if screenshots else None
        cursor_target = sc.get("target_point")

        # 🚀 TRUE FALLBACK: Only invoke Qwen-VL if keyword & contour search found nothing
        if scene_type == "screenshot" and cursor_target is None and shot_path and VISION_ENABLED:
            vis = analyze_image(shot_path, title)
            if vis.get("found") and vis.get("confidence", 0) >= VISION_MIN_CONFIDENCE:
                cursor_target = vis.get("click_point")

        narration = _generate_instructor_voice(text, scene_type, title, step_idx=step_idx)

        step = {
            "id": i + 1,
            "number": i + 1,
            "page_num": sc.get("page", i + 1),
            "type": scene_type,
            "kind": "action" if scene_type == "screenshot" else "explanation",
            "title": title,
            "content": text,
            "overview_paragraph": overview_para,
            "key_bullets": key_bullets,
            "toc_items": sc.get("toc_items", []),
            "table_data": sc.get("tables", [None])[0] if sc.get("tables") else None,
            "narration": narration,
            "tts_narration": narration,
            "caption": narration,
            "screenshot": shot_path,
            "cursor": cursor_target,
            "targets": [cursor_target] if cursor_target else [],
        }
        steps.append(step)

        if progress_callback:
            progress = int(10 + 85 * ((i + 1) / float(total)))
            progress_callback(progress, f"Prepared scene {i + 1} of {total}")

    doc_title = steps[0]["title"] if steps else "Enterprise Training Module"
    plan = {
        "title": doc_title,
        "description": "Universal Enterprise Training Tutorial generated from PDF.",
        "narration_language": narration_language,
        "steps": steps,
        "scene_count": len(steps),
    }

    print(f"[PLAN] Successfully created {len(steps)} tutorial scenes.")
    return plan
