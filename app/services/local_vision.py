from pathlib import Path
import base64
import json
import os
from io import BytesIO

import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_URL = os.getenv("OLLAMA_URL", f"{OLLAMA_HOST.rstrip('/')}/api/generate")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:2b-instruct")
VISION_TIMEOUT = int(os.getenv("OLLAMA_VISION_TIMEOUT", "20"))


def _encode_image(path):
    image = Image.open(path).convert("RGB")
    max_side = 1600
    if max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _vision_model_is_available():
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=3)
        response.raise_for_status()
        names = {str(x.get("name", "")) for x in response.json().get("models", []) or []}
        return VISION_MODEL in names
    except Exception:
        return False


def analyze_image(image_path, context="", timeout=None):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if not _vision_model_is_available():
        raise RuntimeError(f"Vision model '{VISION_MODEL}' is not installed in Ollama")

    prompt = f"""
You are a generic UI visual grounding engine for software training videos.
Analyze the complete screenshot and identify the exact interactive UI control where the supplied action must be performed.

ACTION:
{context}

Return ONLY valid JSON:
{{
  "target_name":"",
  "approximate_position":{{"x":0.0,"y":0.0,"width":0.0,"height":0.0}},
  "confidence":0.0
}}

Rules:
- Coordinates are normalized to the COMPLETE screenshot, from 0 to 1.
- x/y are the top-left of the target box.
- Return the actual interactive control, not its descriptive label when the control itself is visible.
- For typing/entering/filling, target the input field.
- For clicking/selecting/opening, target the button, menu, tab, link, checkbox, row, icon, dropdown, or other control itself.
- Use visible instructor arrows/boxes/highlights as evidence, but do not depend on a specific color.
- Never invent a target that is not visible.
- Return confidence between 0 and 1.
"""

    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [_encode_image(image_path)],
        "stream": False,
        "format": "json",
        "keep_alive": "0",
        "options": {
            "temperature": 0,
            "num_predict": 128,
            "num_ctx": 4096,
        },
    }
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(OLLAMA_URL, json=payload, timeout=timeout or VISION_TIMEOUT)
    response.raise_for_status()
    raw = response.json().get("response", "").strip()
    if not raw:
        raise RuntimeError("Ollama vision returned empty output")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise RuntimeError("Ollama vision returned invalid JSON object")
    return obj


def analyze_frames_with_ollama(image_paths, context="", timeout=None, include_step_matching=True, extract_procedure=False):
    results = []
    for index, path in enumerate(image_paths):
        try:
            extra = context
            if include_step_matching:
                extra += "\nIdentify the exact UI target for the requested step."
            if extract_procedure:
                extra += "\nExtract visible actionable procedure steps only."
            result = analyze_image(path, extra, timeout=timeout)
            result["image"] = str(path)
            result["index"] = index
            results.append(result)
        except Exception as exc:
            print(f"[VISION] failed image {index + 1}: {exc}")
            results.append({"image": str(path), "index": index, "error": str(exc)})
    return results
