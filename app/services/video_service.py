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
MIN_SCENE_DURATION = 4.2
SCENE_GAP_SECONDS = 0.6

# Corporate Blue & White Palette (BGR)
COLOR_BG = (248, 246, 244)        # Clean off-white background
COLOR_NAVY = (64, 37, 10)         # #0A2540 Deep Navy
COLOR_BLUE = (204, 102, 0)        # #0066CC Royal Blue accent
COLOR_CARD_BG = (255, 255, 255)   # White Card
COLOR_BORDER = (225, 218, 210)    # Soft slate divider
COLOR_TEXT_DARK = (40, 35, 30)    # Charcoal primary text
COLOR_TEXT_MUTED = (110, 100, 90) # Secondary slate gray


def _audio_files(audio: Any) -> List[str]:
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
        p = item.get("path") if isinstance(item, dict) else str(item)
        if p and os.path.exists(str(p)):
            paths.append(str(p))
    return paths


def _duration(path: Optional[str]) -> float:
    if not path or not os.path.exists(path):
        return 4.2
    try:
        with wave.open(str(path), "rb") as wf:
            return max(1.0, wf.getnframes() / float(wf.getframerate()))
    except Exception:
        return 4.2


def _pad_audio_with_silence(input_wav: str, output_wav: str, target_duration: float) -> None:
    try:
        with wave.open(input_wav, "rb") as r:
            params = r.getparams()
            frames = r.readframes(r.getnframes())
            current_dur = r.getnframes() / float(r.getframerate())

        diff_dur = max(0.0, target_duration - current_dur)
        silence_frames = int(diff_dur * params.framerate)
        silence_bytes = b"\x00" * (silence_frames * params.nchannels * params.sampwidth)

        with wave.open(output_wav, "wb") as w:
            w.setparams(params)
            w.writeframes(frames + silence_bytes)
    except Exception as e:
        print(f"[AUDIO] Pad audio notice: {e}")


def _create_silent_wav(output_wav: str, duration: float) -> None:
    """Creates a silent WAV file to keep tracks synchronized if audio is missing."""
    try:
        framerate = 24000
        nchannels = 1
        sampwidth = 2
        nframes = int(duration * framerate)
        silence_bytes = b"\x00" * (nframes * nchannels * sampwidth)
        with wave.open(output_wav, "wb") as w:
            w.setnchannels(nchannels)
            w.setsampwidth(sampwidth)
            w.setframerate(framerate)
            w.writeframes(silence_bytes)
    except Exception as e:
        print(f"[AUDIO] Silent wav notice: {e}")


def _fit_image(image: Optional[np.ndarray], width: int = WIDTH, height: int = HEIGHT) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    canvas = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)
    if image is None or image.shape[0] <= 0 or image.shape[1] <= 0:
        return canvas, (0, 0, width, height)

    ih, iw = image.shape[:2]
    max_w, max_h = width - 100, height - 120
    scale = min(max_w / float(iw), max_h / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    x_offset = (width - nw) // 2
    y_offset = (height - 80 - nh) // 2 + 20

    cv2.rectangle(canvas, (x_offset - 4, y_offset - 4), (x_offset + nw + 4, y_offset + nh + 4), COLOR_BORDER, 1)
    canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized
    return canvas, (x_offset, y_offset, nw, nh)


def _render_title_slide(title: str) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (WIDTH, 12), COLOR_NAVY, -1)
    cv2.rectangle(canvas, (0, HEIGHT - 10), (WIDTH, HEIGHT), COLOR_BLUE, -1)

    cx1, cy1, cx2, cy2 = 90, 110, WIDTH - 90, HEIGHT - 130
    cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), COLOR_BORDER, 1)

    cv2.rectangle(canvas, (cx1 + 45, cy1 + 45), (cx1 + 255, cy1 + 80), COLOR_BLUE, -1)
    cv2.putText(canvas, "TRAINING MODULE", (cx1 + 58, cy1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    clean_title = (title or "Standard Operating Procedure").strip()
    cv2.putText(canvas, clean_title[:45], (cx1 + 45, cy1 + 145), font, 1.15, COLOR_NAVY, 2, cv2.LINE_AA)
    if len(clean_title) > 45:
        cv2.putText(canvas, clean_title[45:90], (cx1 + 45, cy1 + 190), font, 1.15, COLOR_NAVY, 2, cv2.LINE_AA)

    cv2.line(canvas, (cx1 + 45, cy1 + 225), (cx2 - 45, cy1 + 225), COLOR_BORDER, 1)
    cv2.putText(canvas, "Enterprise Knowledge & System Execution Guide", (cx1 + 45, cy1 + 265), font, 0.68, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    return canvas


def _render_toc_slide(title: str, items: List[str]) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (WIDTH, 8), COLOR_BLUE, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Table of Contents & Module Agenda", (70, 75), font, 1.0, COLOR_NAVY, 2, cv2.LINE_AA)
    cv2.line(canvas, (70, 95), (WIDTH - 70, 95), COLOR_BORDER, 1)

    cv2.rectangle(canvas, (70, 120), (WIDTH - 70, HEIGHT - 100), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (70, 120), (WIDTH - 70, HEIGHT - 100), COLOR_BORDER, 1)

    display_items = items[:6] if items else ["Overview & Purpose", "System Prerequisites", "Step-by-Step Procedure", "Validation & Verification"]
    y = 175
    for idx, item in enumerate(display_items, start=1):
        cv2.circle(canvas, (115, y - 6), 16, COLOR_BLUE, -1)
        cv2.putText(canvas, str(idx), (109, y), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, str(item)[:65], (150, y), font, 0.72, COLOR_TEXT_DARK, 2, cv2.LINE_AA)
        y += 62

    return canvas


def _render_rich_theory_slide(title: str, overview: str, bullets: List[str]) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (WIDTH, 8), COLOR_NAVY, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, (title or "Core Concepts & Principles")[:55], (70, 70), font, 0.95, COLOR_NAVY, 2, cv2.LINE_AA)
    cv2.line(canvas, (70, 90), (WIDTH - 70, 90), COLOR_BORDER, 1)

    # Overview Card
    cv2.rectangle(canvas, (70, 110), (WIDTH - 70, 240), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (70, 110), (WIDTH - 70, 240), COLOR_BORDER, 1)

    cv2.putText(canvas, "OVERVIEW & PURPOSE", (95, 140), font, 0.55, COLOR_BLUE, 2, cv2.LINE_AA)

    words = overview.split()
    y_p = 172
    curr_line = ""
    for w in words:
        if len(curr_line + " " + w) > 78:
            cv2.putText(canvas, curr_line.strip(), (95, y_p), font, 0.62, COLOR_TEXT_DARK, 1, cv2.LINE_AA)
            y_p += 28
            curr_line = w
        else:
            curr_line += " " + w
    if curr_line.strip() and y_p <= 230:
        cv2.putText(canvas, curr_line.strip(), (95, y_p), font, 0.62, COLOR_TEXT_DARK, 1, cv2.LINE_AA)

    # Key Objectives Card
    cv2.rectangle(canvas, (70, 255), (WIDTH - 70, HEIGHT - 95), COLOR_CARD_BG, -1)
    cv2.rectangle(canvas, (70, 255), (WIDTH - 70, HEIGHT - 95), COLOR_BORDER, 1)

    cv2.putText(canvas, "KEY GUIDELINES & OBJECTIVES", (95, 285), font, 0.55, COLOR_BLUE, 2, cv2.LINE_AA)

    display_bullets = bullets[:3] if bullets else ["Follow standard operating protocols.", "Verify system permissions and inputs."]
    y_b = 325
    for b in display_bullets:
        cv2.circle(canvas, (105, y_b - 5), 5, COLOR_BLUE, -1)
        cv2.putText(canvas, str(b)[:80], (125, y_b), font, 0.64, COLOR_TEXT_DARK, 2, cv2.LINE_AA)
        y_b += 46

    return canvas


def _render_table_slide(title: str, table_data: Dict[str, Any]) -> np.ndarray:
    canvas = np.full((HEIGHT, WIDTH, 3), COLOR_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (WIDTH, 8), COLOR_BLUE, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, (title or "Reference Data Table")[:55], (70, 75), font, 0.95, COLOR_NAVY, 2, cv2.LINE_AA)
    cv2.line(canvas, (70, 95), (WIDTH - 70, 95), COLOR_BORDER, 1)

    headers = table_data.get("headers", [])[:5]
    rows = table_data.get("rows", [])[:5]

    if not headers:
        return _render_rich_theory_slide(title, "Reference data overview.", ["Refer to system parameters."])

    start_x, start_y = 70, 120
    tbl_w = WIDTH - 140
    col_w = max(10, tbl_w // len(headers))
    row_h = 42

    cv2.rectangle(canvas, (start_x, start_y), (start_x + tbl_w, start_y + row_h), COLOR_NAVY, -1)
    for c_idx, h_text in enumerate(headers):
        tx = start_x + c_idx * col_w + 12
        cv2.putText(canvas, str(h_text)[:18], (tx, start_y + 26), font, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    for r_idx, row in enumerate(rows):
        ry = start_y + (r_idx + 1) * row_h
        bg_color = (255, 255, 255) if r_idx % 2 == 0 else (244, 238, 230)
        cv2.rectangle(canvas, (start_x, ry), (start_x + tbl_w, ry + row_h), bg_color, -1)
        cv2.rectangle(canvas, (start_x, ry), (start_x + tbl_w, ry + row_h), COLOR_BORDER, 1)

        for c_idx, cell in enumerate(row[:len(headers)]):
            cx = start_x + c_idx * col_w + 12
            cv2.putText(canvas, str(cell)[:20], (cx, ry + 26), font, 0.54, COLOR_TEXT_DARK, 1, cv2.LINE_AA)

    return canvas


def _draw_mouse_pointer(frame: np.ndarray, x: int, y: int, clicking: bool = False, click_progress: float = 0.0) -> np.ndarray:
    if clicking and click_progress > 0.0:
        ripple_r = int(10 + 35 * click_progress)
        alpha = max(0.0, 1.0 - click_progress)
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), ripple_r, COLOR_BLUE, 3, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 6, (0, 100, 255), -1, cv2.LINE_AA)
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

    cv2.fillPoly(frame, [arrow_pts + 2], (40, 40, 40), cv2.LINE_AA)
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
    scale = 0.62
    thickness = 2
    line_h = 28
    pad = 12

    total_h = len(lines) * line_h + pad * 2
    y1 = HEIGHT - total_h - 15
    y2 = HEIGHT - 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (40, y1), (WIDTH - 40, y2), COLOR_NAVY, -1)
    frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0)

    text_y = y1 + pad + 20
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        tx = (WIDTH - tw) // 2
        cv2.putText(frame, line, (tx, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        text_y += line_h

    return frame


def render(plan: Dict[str, Any], audio: Any, job_dir: str, final_path: str) -> str:
    job_dir = Path(job_dir)
    final_path = Path(final_path)
    raw_video = job_dir / "video_raw.mp4"
    padded_audio_dir = job_dir / "audio_padded"
    padded_audio_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = _audio_files(audio)
    steps = plan.get("steps", [])
    padded_audio_paths = []

    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

    try:
        for idx, scene in enumerate(steps):
            scene_type = scene.get("type", "theory")
            img_path = scene.get("screenshot")

            if scene_type == "title":
                base_frame = _render_title_slide(scene.get("title", ""))
                ox, oy, nw, nh = 0, 0, WIDTH, HEIGHT
            elif scene_type == "toc":
                base_frame = _render_toc_slide(scene.get("title", ""), scene.get("toc_items", []))
                ox, oy, nw, nh = 0, 0, WIDTH, HEIGHT
            elif scene_type == "table" and scene.get("table_data"):
                base_frame = _render_table_slide(scene.get("title", ""), scene.get("table_data", {}))
                ox, oy, nw, nh = 0, 0, WIDTH, HEIGHT
            elif scene_type == "screenshot" and img_path and os.path.exists(str(img_path)):
                raw_img = cv2.imread(str(img_path))
                base_frame, (ox, oy, nw, nh) = _fit_image(raw_img)
            else:
                base_frame = _render_rich_theory_slide(
                    scene.get("title", ""),
                    scene.get("overview_paragraph", ""),
                    scene.get("key_bullets", []),
                )
                ox, oy, nw, nh = 0, 0, WIDTH, HEIGHT

            raw_dur = _duration(audio_paths[idx]) if idx < len(audio_paths) else 4.2
            scene_duration = max(MIN_SCENE_DURATION, raw_dur + SCENE_GAP_SECONDS)
            frame_count = int(scene_duration * FPS)

            pad_wav = padded_audio_dir / f"pad_{idx:03d}.wav"
            if idx < len(audio_paths):
                _pad_audio_with_silence(audio_paths[idx], str(pad_wav), scene_duration)
            else:
                _create_silent_wav(str(pad_wav), scene_duration)
            padded_audio_paths.append(str(pad_wav))

            target_norm = scene.get("cursor")
            has_cursor = (scene_type == "screenshot" and target_norm is not None and len(target_norm) == 2)

            if has_cursor:
                safe_nx = max(0.06, min(0.92, float(target_norm[0])))
                safe_ny = max(0.06, min(0.92, float(target_norm[1])))

                tx = int(ox + safe_nx * nw)
                ty = int(oy + safe_ny * nh)
                start_x, start_y = max(ox + 20, tx - 120), max(oy + 20, ty - 80)
                move_frames = max(1, int(FPS * 1.0))

            for fi in range(frame_count):
                frame = base_frame.copy()

                if has_cursor:
                    t = min(1.0, fi / float(move_frames))
                    t_smooth = t * t * (3.0 - 2.0 * t)
                    cx = int(start_x + (tx - start_x) * t_smooth)
                    cy = int(start_y + (ty - start_y) * t_smooth)

                    clicking = (fi >= move_frames) and (fi <= move_frames + int(FPS * 0.4))
                    click_prog = (fi - move_frames) / float(FPS * 0.4) if clicking else 0.0
                    frame = _draw_mouse_pointer(frame, cx, cy, clicking=clicking, click_progress=click_prog)

                frame = _draw_caption(frame, scene.get("caption", ""))
                writer.write(frame)
    finally:
        writer.release()

    concat_list = job_dir / "audio_concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for ap in padded_audio_paths:
            safe_path = Path(ap).resolve().as_posix().replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    merged_wav = job_dir / "narration.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:a", "pcm_s16le", str(merged_wav)],
        capture_output=True,
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(raw_video),
        "-i", str(merged_wav),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(final_path),
    ] if os.path.exists(merged_wav) else [
        "ffmpeg", "-y", "-i", str(raw_video), "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", str(final_path)
    ]

    subprocess.run(cmd, capture_output=True)
    print(f"[RENDER] Final Video Generated Successfully: {final_path}")
    return str(final_path)
