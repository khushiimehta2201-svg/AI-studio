from pathlib import Path
import os
import json
import subprocess
import wave
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
FPS = 24
MIN_SCENE_DURATION = 3.2
PRE_ACTION_SECONDS = 1.15
NARRATION_LEAD_IN_SECONDS = 1.45
SCENE_PADDING_SECONDS = 0.35
CURSOR_MOVE_SECONDS = 0.90
HIGHLIGHT_SECONDS = 1.20

TEAMCENTER_CHROME_BG = (238, 238, 235)
TEAMCENTER_ACCENT = (150, 90, 10)
TEAMCENTER_LABEL = "Teamcenter"


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def _sample_chrome_color(image: np.ndarray) -> Tuple[int, int, int]:
    h, w = image.shape[:2]
    if h < 10 or w < 10:
        return TEAMCENTER_CHROME_BG
    border = np.concatenate([
        image[0:4, :].reshape(-1, 3),
        image[-4:, :].reshape(-1, 3),
        image[:, 0:4].reshape(-1, 3),
        image[:, -4:].reshape(-1, 3),
    ])
    return tuple(int(c) for c in np.median(border, axis=0))


def _fit_image(image: Optional[np.ndarray], width: int = WIDTH, height: int = HEIGHT):
    if image is None:
        canvas = np.full((height, width, 3), TEAMCENTER_CHROME_BG, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (width, 42), TEAMCENTER_ACCENT, -1)
        cv2.putText(canvas, TEAMCENTER_LABEL, (18, 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas, (0, 0, width, height)

    ih, iw = image.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.full((height, width, 3), TEAMCENTER_CHROME_BG, dtype=np.uint8), (0, 0, width, height)

    canvas = np.full((height, width, 3), _sample_chrome_color(image), dtype=np.uint8)
    scale = min(width / float(iw), height / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    ox, oy = (width - nw) // 2, (height - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, (ox, oy, nw, nh)


def _audio_files(audio: Any) -> List[Optional[str]]:
    if not audio:
        return []
    items = audio
    if isinstance(audio, dict):
        for key in ("files", "audio_files", "steps", "narration"):
            if key in audio:
                items = audio[key]
                break
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        items = [items]
    paths = []
    for item in items:
        p = None
        if isinstance(item, str):
            p = item
        elif isinstance(item, dict):
            for key in ("path", "file", "audio", "wav", "audio_path"):
                if item.get(key):
                    p = item[key]
                    break
        paths.append(str(p) if p and os.path.exists(str(p)) else None)
    return paths


def _duration(path: Optional[str]) -> float:
    if not path or not os.path.exists(path):
        return MIN_SCENE_DURATION
    try:
        with wave.open(str(path), "rb") as wf:
            return max(1.0, wf.getnframes() / float(wf.getframerate()))
    except Exception:
        return MIN_SCENE_DURATION


def _wrap_text(text: str, max_chars: int = 46) -> List[str]:
    words = str(text or "").split()
    lines, current = [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines or [""]


def _draw_dialogue(frame: np.ndarray, dialogue: Dict[str, Any]) -> np.ndarray:
    # Dialogue is intentionally rendered in the trainee portal beside the
    # player. Keeping it out of the encoded video prevents duplicate panels
    # and lets the browser update it exactly with playback time.
    return frame

def _draw_target(frame: np.ndarray, point: Tuple[int, int], box: Optional[List[float]],
                 active: bool = True, label: str = "Click here") -> np.ndarray:
    if not active:
        return frame
    x, y = point
    overlay = frame.copy()
    if box and len(box) >= 4:
        x0, y0, x1, y1 = [int(v) for v in box[:4]]
        pad = 10
        # Soft glow + strong target outline.
        cv2.rectangle(overlay, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), (0, 190, 255), 10, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)
        cv2.rectangle(frame, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), (0, 190, 255), 3, cv2.LINE_AA)
        anchor_x, anchor_y = x0 - pad, max(28, y0 - pad - 8)
    else:
        cv2.circle(overlay, (x, y), 30, (0, 190, 255), 10, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.24, frame, 0.76, 0, frame)
        cv2.circle(frame, (x, y), 25, (0, 190, 255), 3, cv2.LINE_AA)
        anchor_x, anchor_y = x - 48, max(28, y - 42)

    label = (label or "Click here")[:28]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.43, 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    lx = max(8, min(WIDTH - tw - 18, anchor_x))
    ly = max(th + 12, min(HEIGHT - 12, anchor_y))
    cv2.rectangle(frame, (lx - 6, ly - th - 8), (lx + tw + 8, ly + 5), (10, 25, 35), -1)
    cv2.putText(frame, label, (lx, ly - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

def _draw_mouse_pointer(frame: np.ndarray, x: int, y: int, clicking: bool = False,
                        click_progress: float = 0.0) -> np.ndarray:
    if clicking and click_progress > 0:
        ripple_r = int(8 + 32 * click_progress)
        alpha = max(0.0, 1.0 - click_progress)
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), ripple_r, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 6, (0, 120, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.8, frame, 1.0 - alpha * 0.8, 0, frame)

    pts = np.array([[x, y], [x, y + 24], [x + 6, y + 19], [x + 12, y + 29],
                    [x + 16, y + 27], [x + 10, y + 17], [x + 18, y + 17]], dtype=np.int32)
    cv2.fillPoly(frame, [pts + 2], (15, 15, 15), cv2.LINE_AA)
    cv2.fillPoly(frame, [pts], (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def _draw_caption(frame: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return frame
    lines = _wrap_text(text, 78)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, line_h, pad = 0.55, 2, 25, 10
    total_h = len(lines) * line_h + pad * 2
    y1, y2 = HEIGHT - total_h - 12, HEIGHT - 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (35, y1), (WIDTH - 35, y2), (10, 12, 14), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    y = y1 + pad + 19
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        cv2.putText(frame, line, ((WIDTH - tw) // 2, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h
    return frame


def _eased(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _normalise_point(point: Any, page_width: float, page_height: float) -> Optional[Tuple[float, float]]:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        x, y = float(point[0]), float(point[1])
    except Exception:
        return None
    # PDF-text targets are in page points. Vision/screenshot targets are
    # already normalized. Detect the coordinate space from page dimensions.
    if page_width > 0 and page_height > 0 and (x > 1.5 or y > 1.5):
        x /= page_width
        y /= page_height
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def _normalise_box(box: Any, page_width: float, page_height: float) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        vals = [float(v) for v in box[:4]]
    except Exception:
        return None
    if max(vals) > 1.5 and page_width > 0 and page_height > 0:
        vals = [vals[0] / page_width, vals[1] / page_height,
                vals[2] / page_width, vals[3] / page_height]
    return [max(0.0, min(1.0, v)) for v in vals]


def _target_pixels(scene: Dict[str, Any], page_width: float, page_height: float,
                   ox: int, oy: int, nw: int, nh: int):
    point = scene.get("cursor")
    coordinate_source = str(scene.get("cursor_source") or "")
    if coordinate_source == "embedded-pdf-text" and isinstance(point, (list, tuple)) and len(point) >= 2:
        norm = (
            max(0.0, min(1.0, float(point[0]))),
            max(0.0, min(1.0, float(point[1]))),
        )
    else:
        norm = _normalise_point(point, page_width, page_height)
    if norm is None:
        targets = scene.get("targets") or []
        if targets:
            norm = _normalise_point(targets[0], page_width, page_height)
    if norm is None:
        return None, None
    px = int(ox + norm[0] * nw)
    py = int(oy + norm[1] * nh)
    raw_box = scene.get("cursor_bounding_box")
    if coordinate_source == "embedded-pdf-text":
        box = (
            [max(0.0, min(1.0, float(value))) for value in raw_box[:4]]
            if isinstance(raw_box, (list, tuple)) and len(raw_box) >= 4
            else None
        )
    else:
        box = _normalise_box(raw_box, page_width, page_height)
    if box:
        box_px = [ox + box[0] * nw, oy + box[1] * nh,
                  ox + box[2] * nw, oy + box[3] * nh]
    else:
        box_px = None
    return (px, py), box_px


def _fallback_target(scene: Dict[str, Any], ox: int, oy: int, nw: int, nh: int):
    """Keep action scenes visually directed when no exact target was found."""
    action = str(scene.get("title") or scene.get("caption") or "").lower()
    if any(word in action for word in ("enter", "type", "search", "url", "address")):
        return (ox + int(nw * 0.5), oy + int(nh * 0.22)), None
    return (ox + int(nw * 0.5), oy + int(nh * 0.5)), None


def _render_action_frames(writer, scene, audio_path, raw_img, cues, cumulative):
    base, (ox, oy, nw, nh) = _fit_image(raw_img)
    narration_duration = _duration(audio_path) if audio_path else 0.0
    duration = max(MIN_SCENE_DURATION, NARRATION_LEAD_IN_SECONDS + narration_duration + SCENE_PADDING_SECONDS)
    frame_count = max(1, int(round(duration * FPS)))

    page_w = float(scene.get("page_width") or 0)
    page_h = float(scene.get("page_height") or 0)
    target, box = _target_pixels(scene, page_w, page_h, ox, oy, nw, nh)
    if target is None and raw_img is not None:
        target, box = _fallback_target(scene, ox, oy, nw, nh)
    target_label = scene.get("cursor_target_name") or scene.get("cursor_annotation") or "Click here"

    if target:
        # Start from a visible, natural point near the lower-left of the
        # target and glide to the control before the click.
        start = (max(18, target[0] - 170), max(18, target[1] - 105))
    else:
        start = None

    move_frames = min(frame_count - 1, max(1, int(CURSOR_MOVE_SECONDS * FPS))) if target else 0
    click_start = int(PRE_ACTION_SECONDS * FPS)
    click_len = max(1, int(0.36 * FPS))
    highlight_frames = min(frame_count, max(1, int(HIGHLIGHT_SECONDS * FPS))) if target else 0

    for fi in range(frame_count):
        frame = base.copy()
        if target:
            if fi < move_frames and start is not None:
                t = _eased(fi / float(move_frames))
                cx = int(start[0] + (target[0] - start[0]) * t)
                cy = int(start[1] + (target[1] - start[1]) * t)
            else:
                cx, cy = target

            # Keep the target visible for the entire action so the learner
            # can still see where the narration is directing them.
            active = True
            frame = _draw_target(frame, target, box, active=active, label=f"Target: {target_label}")
            clicking = click_start <= fi < click_start + click_len
            prog = (fi - click_start) / float(click_len) if clicking else 0.0
            frame = _draw_mouse_pointer(frame, cx, cy, clicking, max(0.0, min(1.0, prog)))

        # Captions begin when narration begins, after the visual lead-in.
        if fi >= click_start:
            frame = _draw_caption(frame, scene.get("caption", ""))
        writer.write(frame)

    narration_start = cumulative + NARRATION_LEAD_IN_SECONDS
    narration_end = narration_start + narration_duration
    cues.append((narration_start, min(cumulative + duration, narration_end), scene.get("caption", "")))
    return duration, {
        "start": round(cumulative, 3),
        "end": round(cumulative + duration, 3),
        "narration_start": round(narration_start, 3),
        "narration_end": round(narration_end, 3),
        "action_id": scene.get("id"),
        "title": scene.get("title") or "Current action",
        "current_action": scene.get("dialogue", {}).get("current_action") or scene.get("title") or "Follow the highlighted control.",
        "brief": scene.get("dialogue", {}).get("brief") or scene.get("dialogue", {}).get("intro") or "Follow the highlighted control and continue.",
        "target_name": target_label if target else "",
        "has_target": bool(target),
    }

def _concat_audio(audio_paths: List[Optional[str]], output_dir: Path, prefix: str) -> Optional[str]:
    """Concatenate narration with a short silent lead-in before every action.

    The silent lead-in matches the visual cursor movement, so narration starts
    after the learner has seen the target being highlighted and clicked.
    """
    valid = [p for p in audio_paths if p]
    if not valid:
        return None

    out = output_dir / f"{prefix}_narration.wav"
    first_params = None
    try:
        with wave.open(valid[0], "rb") as wf:
            first_params = wf.getparams()
        with wave.open(str(out), "wb") as writer:
            writer.setnchannels(first_params.nchannels)
            writer.setsampwidth(first_params.sampwidth)
            writer.setframerate(first_params.framerate)
            silence_frames = int(NARRATION_LEAD_IN_SECONDS * first_params.framerate)
            silence = b"\x00" * silence_frames * first_params.nchannels * first_params.sampwidth

            for p in audio_paths:
                if not p or not os.path.exists(p):
                    return None
                with wave.open(str(p), "rb") as wf:
                    params = wf.getparams()
                    if (params.nchannels, params.sampwidth, params.framerate) != (
                        first_params.nchannels, first_params.sampwidth, first_params.framerate
                    ):
                        return None
                    writer.writeframes(silence)
                    writer.writeframes(wf.readframes(wf.getnframes()))
        return str(out)
    except Exception as exc:
        print(f"[AUDIO] Concat failed: {exc}")
        return None

def _format_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _write_vtt(cues, path: Path):
    lines = ["WEBVTT", ""]
    for start, end, text in cues:
        if text:
            lines += [f"{_format_ts(start)} --> {_format_ts(end)}", text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def render_tutorial(plan: Dict[str, Any], audio: Any, job_dir: str) -> Dict[str, Any]:
    """Render the entire ordered action timeline into ONE continuous video."""
    job_dir = Path(job_dir)
    steps = [s for s in plan.get("steps", []) if s.get("kind") == "action"]
    audio_paths = _audio_files(audio)
    raw_video = job_dir / "tutorial_raw.mp4"
    final_video = job_dir / "tutorial.mp4"
    vtt_path = job_dir / "tutorial.vtt"

    if not steps:
        return {"video_file": None, "captions_file": None, "duration": 0.0, "action_count": 0}

    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer for tutorial_raw.mp4")

    cues = []
    timeline = []
    cumulative = 0.0
    audio_for_concat: List[Optional[str]] = []

    try:
        for index, scene in enumerate(steps):
            img_path = scene.get("screenshot")
            raw_img = cv2.imread(str(img_path)) if img_path and os.path.exists(str(img_path)) else None
            audio_path = audio_paths[index] if index < len(audio_paths) else None
            audio_for_concat.append(audio_path)
            duration, timeline_item = _render_action_frames(writer, scene, audio_path, raw_img, cues, cumulative)
            timeline.append(timeline_item)
            cumulative += duration
    finally:
        writer.release()

    _write_vtt(cues, vtt_path)
    timeline_path = job_dir / "tutorial_timeline.json"
    timeline_path.write_text(json.dumps(timeline, indent=2, ensure_ascii=False), encoding="utf-8")
    narration = _concat_audio(audio_for_concat, job_dir, "tutorial")

    ffmpeg = _ffmpeg_exe()
    if narration and len(audio_for_concat) == sum(1 for p in audio_for_concat if p):
        cmd = [ffmpeg, "-y", "-i", str(raw_video), "-i", str(narration),
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", str(final_video)]
    else:
        cmd = [ffmpeg, "-y", "-i", str(raw_video), "-map", "0:v:0", "-c:v", "libx264",
               "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", str(final_video)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not final_video.exists():
        raise RuntimeError(f"Final tutorial video encoding failed: {result.stderr[-1500:]}")

    print(f"[RENDER] Completed ONE continuous tutorial: {len(steps)} actions, {cumulative:.1f}s")
    return {
        "video_file": final_video.name,
        "captions_file": vtt_path.name,
        "timeline_file": timeline_path.name,
        "timeline": timeline,
        "duration": round(cumulative, 2),
        "action_count": len(steps),
    }


def render_all_sections(plan: Dict[str, Any], audio: Any, job_dir: str) -> Dict[str, Any]:
    """Backward-compatible entry point. It now renders ONE tutorial, not sections."""
    result = render_tutorial(plan, audio, job_dir)
    plan["video_file"] = result.get("video_file")
    plan["captions_file"] = result.get("captions_file")
    plan["video_duration"] = result.get("duration", 0.0)
    plan["action_count"] = result.get("action_count", 0)
    plan["timeline_file"] = result.get("timeline_file")
    plan["dialogue_timeline"] = result.get("timeline", [])
    # Keep sections as organizational metadata only; no section videos are generated.
    for section in plan.get("sections", []):
        section["video_file"] = None
        section["captions_file"] = None
        section["duration"] = 0.0
    print("[RENDER] Section rendering disabled; generated a single tutorial.mp4")
    return plan
