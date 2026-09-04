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


def _audio_files(audio: Any) -> List[str]:
    """Extracts valid WAV audio file paths from the TTS result structure."""
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
        if p and os.path.exists(str(p)):
            paths.append(str(p))

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
    """Letterboxes screenshot over a dark backdrop and returns bounding offsets."""
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


def _draw_presentation_slide(title: str, content: str) -> np.ndarray:
    """Renders a modern presentation slide for text-only pages."""
    canvas = np.full((HEIGHT, WIDTH, 3), (30, 32, 34), dtype=np.uint8)

    # Accent top border
    cv2.rectangle(canvas, (0, 0), (WIDTH, 8), (0, 165, 255), -1)

    # Section Title
    font = cv2.FONT_HERSHEY_SIMPLEX
    clean_title = (title or "Overview").strip()[:60]
    cv2.putText(canvas, clean_title, (70, 90), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (70, 115), (WIDTH - 70, 115), (70, 75, 80), 1, cv2.LINE_AA)

    # Bullet point content
    lines = [l.strip() for l in content.splitlines() if l.strip() and len(l.strip()) > 3][:6]
    y = 175
    for line in lines:
        cv2.circle(canvas, (85, y - 6), 5, (0, 165, 255), -1, cv2.LINE_AA)
        words = line.split()
        curr_line = ""
        for w in words:
            if len(curr_line + " " + w) > 65:
                cv2.putText(canvas, curr_line.strip(), (105, y), font, 0.72, (220, 225, 230), 2, cv2.LINE_AA)
                y += 34
                curr_line = w
            else:
                curr_line += " " + w
        if curr_line.strip():
            cv2.putText(canvas, curr_line.strip(), (105, y), font, 0.72, (220, 225, 230), 2, cv2.LINE_AA)
            y += 46

    return canvas


def _draw_mouse_pointer(frame: np.ndarray, x: int, y: int, clicking: bool = False, click_progress: float = 0.0) -> np.ndarray:
    """Draws a medium-large OS mouse cursor arrow with drop-shadow and click ripple."""
    if clicking and click_progress > 0.0:
        ripple_r = int(10 + 35 * click_progress)
        alpha = max(0.0, 1.0 - click_progress)
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), ripple_r, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 6, (0, 120, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.8, frame, 1.0 - (alpha * 0.8), 0, frame)

    arrow_pts = np.array([
        [x, y],
        [x, y + 24],
        [x + 6, y + 19],
        [x + 12, y + 29],
        [x + 16, y + 27],
        [x + 10, y + 17],
        [x + 18, y + 17],
    ], dtype=np.int32)

    # 1. Drop shadow
    shadow_pts = arrow_pts + 2
    cv2.fillPoly(frame, [shadow_pts], (15, 15, 15), cv2.LINE_AA)

    # 2. White cursor body
    cv2.fillPoly(frame, [arrow_pts], (255, 255, 255), cv2.LINE_AA)

    # 3. Crisp Black border
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
    scale = 0.62
    thickness = 2
    line_h = 28
    pad = 12

    total_h = len(lines) * line_h + pad * 2
    y1 = HEIGHT - total_h - 15
    y2 = HEIGHT - 15

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


def _interpolate_multi_targets(
    targets: List[Tuple[int, int]],
    frame_index: int,
    total_frames: int,
) -> Tuple[int, int, bool, float]:
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


def _build_audio_concat(audio_paths: List[str], output_dir: Path) -> Optional[str]:
    """Combines individual audio files into a single continuous narration WAV."""
    if not audio_paths:
        print("[AUDIO] No audio files to concatenate.")
        return None

    concat_file = output_dir / "audio_concat.txt"
    narration_file = output_dir / "narration.wav"

    with open(concat_file, "w", encoding="utf-8") as f:
        for path in audio_paths:
            safe_path = Path(path).resolve().as_posix().replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:a", "pcm_s16le",
        str(narration_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and narration_file.exists():
        return str(narration_file)

    print(f"[AUDIO] Concat failed: {result.stderr}")
    return None


def render(plan: Dict[str, Any], audio: Any, job_dir: str, final_path: str) -> str:
    job_dir = Path(job_dir)
    final_path = Path(final_path)
    raw_video = job_dir / "video_raw.mp4"
    audio_paths = _audio_files(audio)
    steps = plan.get("steps", [])

    writer = cv2.VideoWriter(
        str(raw_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )

    try:
        for idx, scene in enumerate(steps):
            img_path = scene.get("screenshot")
            is_full_page = scene.get("is_full_page", False)

            if is_full_page or not img_path or not os.path.exists(str(img_path)):
                base_frame = _draw_presentation_slide(scene.get("title", ""), scene.get("content", ""))
                ox, oy, nw, nh = 0, 0, WIDTH, HEIGHT
            else:
                raw_img = cv2.imread(str(img_path))
                base_frame, (ox, oy, nw, nh) = _fit_image(raw_img)

            raw_dur = _duration(audio_paths[idx]) if idx < len(audio_paths) else 3.5
            duration = max(MIN_SCENE_DURATION, raw_dur + SCENE_PADDING_SECONDS)
            frame_count = int(duration * FPS)

            pixel_targets = []
            targets_norm = scene.get("targets") or ([scene.get("cursor")] if scene.get("cursor") else [])
            for t_norm in targets_norm:
                if t_norm and len(t_norm) == 2:
                    pixel_targets.append((
                        int(ox + t_norm[0] * nw),
                        int(oy + t_norm[1] * nh),
                    ))

            has_cursor = (scene.get("kind") == "action" and len(pixel_targets) > 0 and not is_full_page)

            for fi in range(frame_count):
                frame = base_frame.copy()

                if has_cursor:
                    cx, cy, clicking, click_prog = _interpolate_multi_targets(pixel_targets, fi, frame_count)
                    frame = _draw_mouse_pointer(frame, cx, cy, clicking=clicking, click_progress=click_prog)

                frame = _draw_caption(frame, scene.get("caption", ""))
                writer.write(frame)
    finally:
        writer.release()

    # Always initialize narration safely to prevent UnboundLocalError
    narration_wav: Optional[str] = _build_audio_concat(audio_paths, job_dir)

    # Final Mux with FFmpeg
    if narration_wav and os.path.exists(narration_wav):
        cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_video),
            "-i", str(narration_wav),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            str(final_path),
        ]
    else:
        # Fallback if no audio files were generated
        cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_video),
            "-map", "0:v:0",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-an",
            str(final_path),
        ]

    subprocess.run(cmd, capture_output=True)
    print(f"[RENDER] Completed tutorial video: {final_path}")
    return str(final_path)
