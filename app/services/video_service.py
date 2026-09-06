import os
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
FPS = 24
MIN_SCENE_DURATION = 3.5
SCENE_PADDING_SECONDS = 0.6


def _audio_files(audio: Any) -> List[Optional[str]]:
    """Extracts WAV audio file paths (order-preserving, matching plan['steps']
    index) from the TTS result structure. A step whose audio failed keeps a
    None placeholder so later steps don't shift out of alignment."""
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

    paths: List[Optional[str]] = []
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
        return 3.5
    try:
        with wave.open(str(path), "rb") as wf:
            return max(1.0, wf.getnframes() / float(wf.getframerate()))
    except Exception:
        return 3.5


def _fit_image(image: Optional[np.ndarray], width: int = WIDTH, height: int = HEIGHT) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Letterboxes screenshot over a dark backdrop; returns bounding offsets
    used to convert normalized screenshot coordinates into frame pixels."""
    canvas = np.full((height, width, 3), (24, 26, 27), dtype=np.uint8)
    if image is None:
        return canvas, (0, 0, width, height)

    ih, iw = image.shape[:2]
    if iw <= 0 or ih <= 0:
        return canvas, (0, 0, width, height)

    scale = min(width / float(iw), height / float(ih))
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    x_offset = (width - nw) // 2
    y_offset = (height - nh) // 2
    canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized
    return canvas, (x_offset, y_offset, nw, nh)


def _draw_mouse_pointer(frame: np.ndarray, x: int, y: int, clicking: bool = False, click_progress: float = 0.0) -> np.ndarray:
    if clicking and click_progress > 0.0:
        ripple_r = int(10 + 35 * click_progress)
        alpha = max(0.0, 1.0 - click_progress)
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), ripple_r, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 6, (0, 120, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.8, frame, 1.0 - (alpha * 0.8), 0, frame)

    arrow_pts = np.array([
        [x, y], [x, y + 24], [x + 6, y + 19], [x + 12, y + 29],
        [x + 16, y + 27], [x + 10, y + 17], [x + 18, y + 17],
    ], dtype=np.int32)

    cv2.fillPoly(frame, [arrow_pts + 2], (15, 15, 15), cv2.LINE_AA)
    cv2.fillPoly(frame, [arrow_pts], (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(frame, [arrow_pts], isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    return frame


def _draw_caption(frame: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return frame

    words = text.split()
    lines, curr = [], ""
    for w in words:
        if len(curr + " " + w) > 75:
            lines.append(curr.strip())
            curr = w
        else:
            curr += " " + w
    if curr.strip():
        lines.append(curr.strip())

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, line_h, pad = 0.62, 2, 28, 12
    total_h = len(lines) * line_h + pad * 2
    y1, y2 = HEIGHT - total_h - 15, HEIGHT - 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (40, y1), (WIDTH - 40, y2), (10, 12, 14), -1)
    frame = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)

    text_y = y1 + pad + 20
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        tx = (WIDTH - tw) // 2
        cv2.putText(frame, line, (tx, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        text_y += line_h
    return frame


def _interpolate_multi_targets(targets: List[Tuple[int, int]], frame_index: int, total_frames: int) -> Tuple[int, int, bool, float]:
    if not targets:
        return 0, 0, False, 0.0

    if len(targets) == 1:
        tx, ty = targets[0]
        start_x, start_y = max(20, tx - 100), max(20, ty - 60)
        move_frames = max(1, int(FPS * 1.0))
        t = min(1.0, frame_index / float(move_frames))
        t_smooth = t * t * (3.0 - 2.0 * t)
        cx = int(start_x + (tx - start_x) * t_smooth)
        cy = int(start_y + (ty - start_y) * t_smooth)
        clicking = (frame_index >= move_frames) and (frame_index <= move_frames + int(FPS * 0.4))
        progress = (frame_index - move_frames) / float(FPS * 0.4) if clicking else 0.0
        return cx, cy, clicking, max(0.0, min(1.0, progress))

    segments = len(targets)
    frames_per_segment = max(1.0, total_frames / float(segments))
    seg_idx = min(segments - 1, int(frame_index / frames_per_segment))
    from_pt = targets[seg_idx - 1] if seg_idx > 0 else (targets[0][0] - 80, targets[0][1] - 50)
    to_pt = targets[seg_idx]

    local_f = frame_index - (seg_idx * frames_per_segment)
    travel_frames = max(1.0, frames_per_segment * 0.65)
    t = min(1.0, local_f / float(travel_frames))
    t_smooth = t * t * (3.0 - 2.0 * t)
    cx = int(from_pt[0] + (to_pt[0] - from_pt[0]) * t_smooth)
    cy = int(from_pt[1] + (to_pt[1] - from_pt[1]) * t_smooth)

    click_frames = max(1.0, frames_per_segment * 0.35)
    clicking = (local_f >= travel_frames)
    progress = (local_f - travel_frames) / float(click_frames) if clicking else 0.0
    return cx, cy, clicking, max(0.0, min(1.0, progress))


def _build_audio_concat(audio_paths: List[str], output_dir: Path, prefix: str) -> Optional[str]:
    if not audio_paths:
        return None
    output_dir = Path(output_dir)
    concat_file = output_dir / f"{prefix}_concat.txt"
    narration_file = output_dir / f"{prefix}_narration.wav"

    with open(concat_file, "w", encoding="utf-8") as f:
        for path in audio_paths:
            safe_path = Path(path).resolve().as_posix().replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c:a", "pcm_s16le", str(narration_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and narration_file.exists():
        return str(narration_file)
    print(f"[AUDIO] Concat failed for {prefix}: {result.stderr}")
    return None


def _format_ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _write_vtt(cues: List[Tuple[float, float, str]], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for start, end, text in cues:
        if not text:
            continue
        lines.append(f"{_format_ts(start)} --> {_format_ts(end)}")
        lines.append(text)
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def render_section(steps: List[Dict[str, Any]], audio_paths: List[Optional[str]], job_dir: str, section_id: int) -> Optional[Dict[str, Any]]:
    """Renders one section's action steps into its own video + VTT caption
    file. Returns None (no video) if the section has no action steps at
    all -- e.g. a purely conceptual/theory section, which is valid and
    expected, not an error."""
    action_steps = [s for s in steps if s.get("kind") == "action"]
    if not action_steps:
        return None

    job_dir = Path(job_dir)
    prefix = f"section_{section_id:02d}"
    raw_video = job_dir / f"{prefix}_raw.mp4"
    final_video = job_dir / f"{prefix}.mp4"
    vtt_path = job_dir / f"{prefix}.vtt"

    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    cues: List[Tuple[float, float, str]] = []
    section_audio: List[Optional[str]] = []
    cumulative = 0.0

    try:
        for scene in action_steps:
            img_path = scene.get("screenshot")
            raw_img = None
            if img_path and os.path.exists(str(img_path)):
                raw_img = cv2.imread(str(img_path))
            base_frame, (ox, oy, nw, nh) = _fit_image(raw_img)

            audio_idx = int(scene.get("id", 0)) - 1
            audio_path = audio_paths[audio_idx] if 0 <= audio_idx < len(audio_paths) else None
            section_audio.append(audio_path)

            raw_dur = _duration(audio_path)
            duration = max(MIN_SCENE_DURATION, raw_dur + SCENE_PADDING_SECONDS)
            frame_count = int(duration * FPS)

            pixel_targets = []
            for t_norm in (scene.get("targets") or []):
                if t_norm and len(t_norm) == 2:
                    pixel_targets.append((int(ox + t_norm[0] * nw), int(oy + t_norm[1] * nh)))
            has_cursor = len(pixel_targets) > 0

            for fi in range(frame_count):
                frame = base_frame.copy()
                if has_cursor:
                    cx, cy, clicking, prog = _interpolate_multi_targets(pixel_targets, fi, frame_count)
                    frame = _draw_mouse_pointer(frame, cx, cy, clicking=clicking, click_progress=prog)
                frame = _draw_caption(frame, scene.get("caption", ""))
                writer.write(frame)

            cues.append((cumulative, cumulative + duration, scene.get("caption", "")))
            cumulative += duration
    finally:
        writer.release()

    _write_vtt(cues, vtt_path)

    narration_wav = _build_audio_concat([p for p in section_audio if p], job_dir, prefix)
    if narration_wav and os.path.exists(narration_wav) and len(section_audio) == sum(1 for p in section_audio if p):
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_video), "-i", str(narration_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", str(final_video),
        ]
    else:
        # Some steps lack audio (TTS failure) -- ship the video without a
        # combined track rather than desyncing narration against the wrong
        # scenes by muxing a shorter/misaligned audio file.
        cmd = [
            "ffmpeg", "-y", "-i", str(raw_video), "-map", "0:v:0",
            "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an",
            str(final_video),
        ]
    subprocess.run(cmd, capture_output=True)

    return {
        "video_file": final_video.name,
        "captions_file": vtt_path.name,
        "duration": round(cumulative, 2),
    }


def render_all_sections(plan: Dict[str, Any], audio: Any, job_dir: str) -> Dict[str, Any]:
    """Renders every section's video, mutating and returning plan in place
    with 'video_file' / 'captions_file' / 'duration' added to each section
    (None video_file means the section is theory-only, by design)."""
    audio_paths = _audio_files(audio)
    steps_by_section: Dict[int, List[Dict[str, Any]]] = {}
    for s in plan.get("steps", []):
        steps_by_section.setdefault(s["section_id"], []).append(s)

    for section in plan.get("sections", []):
        result = render_section(steps_by_section.get(section["id"], []), audio_paths, job_dir, section["id"])
        if result:
            section.update(result)
        else:
            section["video_file"] = None
            section["captions_file"] = None
            section["duration"] = 0.0

    print(f"[RENDER] Completed {sum(1 for s in plan.get('sections', []) if s.get('video_file'))} section videos.")
    return plan
