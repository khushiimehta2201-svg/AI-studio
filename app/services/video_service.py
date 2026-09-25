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
MIN_SCENE_DURATION = 2.6
MIN_PRE_ROLL = 0.25
MAX_PRE_ROLL = 0.75
TRAILING_HOLD_SECONDS = 0.45
MIN_CURSOR_MOVE = 0.35
MAX_CURSOR_MOVE = 1.10
CLICK_PULSE_SECONDS = 0.28
HIGHLIGHT_SECONDS = 0.80
CAPTION_MAX_LINES = 2

TEAMCENTER_CHROME_BG = (238, 238, 235)
TEAMCENTER_ACCENT = (150, 90, 10)
TEAMCENTER_LABEL = "Teamcenter"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split()).strip()


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


def _fit_image(
    image: Optional[np.ndarray],
    width: int = WIDTH,
    height: int = HEIGHT,
):
    if image is None:
        canvas = np.full(
            (height, width, 3),
            TEAMCENTER_CHROME_BG,
            dtype=np.uint8,
        )
        cv2.rectangle(
            canvas,
            (0, 0),
            (width, 42),
            TEAMCENTER_ACCENT,
            -1,
        )
        cv2.putText(
            canvas,
            TEAMCENTER_LABEL,
            (18, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas, (0, 0, width, height)

    ih, iw = image.shape[:2]
    if iw <= 0 or ih <= 0:
        return (
            np.full(
                (height, width, 3),
                TEAMCENTER_CHROME_BG,
                dtype=np.uint8,
            ),
            (0, 0, width, height),
        )

    canvas = np.full(
        (height, width, 3),
        _sample_chrome_color(image),
        dtype=np.uint8,
    )
    scale = min(
        width / float(iw),
        height / float(ih),
    )
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )
    ox = (width - nw) // 2
    oy = (height - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, (ox, oy, nw, nh)



def _draw_text_scene(title: str, body: str, kind: str = "explanation") -> np.ndarray:
    """Render non-UI content as a clean training slide.

    These scenes intentionally have no source-PDF pixels. That keeps document
    prose, headings and metadata outside the screenshot/targeting system.
    """
    canvas=np.full((HEIGHT,WIDTH,3),(245,246,243),dtype=np.uint8)
    cv2.rectangle(canvas,(0,0),(WIDTH,54),TEAMCENTER_ACCENT,-1)
    cv2.putText(canvas,"Teamcenter AI Studio",(24,37),cv2.FONT_HERSHEY_SIMPLEX,0.72,(255,255,255),2,cv2.LINE_AA)

    if kind=="transition":
        cv2.putText(canvas,"SECTION",(70,132),cv2.FONT_HERSHEY_SIMPLEX,0.70,TEAMCENTER_ACCENT,2,cv2.LINE_AA)
        wrapped=_wrap_text(title or "Tutorial section",46)
        y=280
        for line in wrapped[:2]:
            scale=1.45 if len(line)<34 else 1.10
            (tw,_),_=cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,scale,2)
            cv2.putText(canvas,line,((WIDTH-tw)//2,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(40,45,48),2,cv2.LINE_AA)
            y+=72
        cv2.line(canvas,(250,430),(WIDTH-250,430),TEAMCENTER_ACCENT,3,cv2.LINE_AA)
        return canvas

    cv2.putText(canvas,"CONCEPT",(70,120),cv2.FONT_HERSHEY_SIMPLEX,0.62,TEAMCENTER_ACCENT,2,cv2.LINE_AA)
    title_text=(title or "Concept")[:110]
    cv2.putText(canvas,title_text,(70,175),cv2.FONT_HERSHEY_SIMPLEX,1.0,(40,45,48),2,cv2.LINE_AA)
    cv2.rectangle(canvas,(62,215),(WIDTH-62,570),(232,235,231),-1)
    cv2.rectangle(canvas,(62,215),(WIDTH-62,570),TEAMCENTER_ACCENT,2,cv2.LINE_AA)
    y=270
    for line in _wrap_text(body or "Continue with the tutorial.",88)[:9]:
        cv2.putText(canvas,line,(88,y),cv2.FONT_HERSHEY_SIMPLEX,0.67,(60,65,68),2,cv2.LINE_AA); y+=38
    return canvas


def _audio_files(audio: Any) -> List[Optional[str]]:
    if not audio:
        return []

    items = audio

    if isinstance(audio, dict):
        for key in (
            "files",
            "audio_files",
            "steps",
            "narration",
        ):
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
            for key in (
                "path",
                "file",
                "audio",
                "wav",
                "audio_path",
            ):
                if item.get(key):
                    p = item[key]
                    break

        paths.append(
            str(p)
            if p and os.path.exists(str(p))
            else None
        )

    return paths


def _duration(path: Optional[str]) -> float:
    if not path or not os.path.exists(path):
        return MIN_SCENE_DURATION

    try:
        with wave.open(str(path), "rb") as wf:
            return max(
                1.0,
                wf.getnframes()
                / float(wf.getframerate()),
            )
    except Exception:
        return MIN_SCENE_DURATION


def _scene_duration(audio_path: Optional[str], timing: Optional[Dict[str, Any]] = None) -> Tuple[float, int, float]:
    narration_duration = _duration(audio_path) if audio_path else 0.0
    timing = timing or {}
    pre_roll = float(timing.get("narration_start", MIN_PRE_ROLL))
    hold = float(timing.get("hold_after", TRAILING_HOLD_SECONDS))
    requested = max(MIN_SCENE_DURATION, pre_roll + narration_duration + hold)
    frame_count = max(1, int(round(requested * FPS)))
    return frame_count / float(FPS), frame_count, narration_duration


def _wrap_text(text: str, max_chars: int = 116) -> List[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    lines: List[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines


def _caption_lines(text: str) -> Tuple[List[str], float]:
    """Fit a complete caption into at most two balanced visual lines."""
    text = _clean_text(text)
    if not text:
        return [], 0.0
    words = text.split()
    if len(words) <= 1:
        return [text], 0.60

    best = None
    for split in range(1, len(words)):
        first = " ".join(words[:split])
        second = " ".join(words[split:])
        longest = max(len(first), len(second))
        imbalance = abs(len(first) - len(second))
        score = longest + imbalance * 0.25
        if best is None or score < best[0]:
            best = (score, first, second)

    lines = [best[1], best[2]] if best else [text]
    longest = max(len(line) for line in lines)
    # Complete captions are preferred over truncation. Smaller type is used
    # for unusually long but still bounded narration rather than adding a
    # third line that could cover important UI content.
    scale = _clamp(0.56 - max(0, longest - 70) * 0.0014, 0.40, 0.56)
    return lines, scale


def _draw_dialogue(frame: np.ndarray, dialogue: Dict[str, Any]) -> np.ndarray:
    return frame


def _draw_target(frame: np.ndarray, point: Tuple[int, int], box: Optional[List[float]], active: bool = True, label: str = "") -> np.ndarray:
    if not active:
        return frame
    x, y = point
    if box and len(box) >= 4:
        x0, y0, x1, y1 = [int(v) for v in box[:4]]
        pad = 8
        cv2.rectangle(frame, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), (0, 175, 235), 3, cv2.LINE_AA)
    else:
        cv2.circle(frame, (x, y), 23, (0, 175, 235), 3, cv2.LINE_AA)
    return frame


def _draw_caption(frame: np.ndarray, text: str) -> np.ndarray:
    if not text:
        return frame
    lines, scale = _caption_lines(text)
    if not lines:
        return frame
    line_h = 23 if scale >= 0.48 else 21
    thickness = 2
    pad = 9
    total_h = len(lines) * line_h + pad * 2
    y1 = max(12, HEIGHT - total_h - 12)
    y2 = HEIGHT - 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (34, y1), (WIDTH - 34, y2), (10, 12, 14), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = y1 + pad + (17 if scale >= 0.48 else 16)
    for line in lines[:CAPTION_MAX_LINES]:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        cv2.putText(
            frame, line, ((WIDTH - tw) // 2, y),
            font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )
        y += line_h
    return frame


def _draw_mouse_pointer(
    frame: np.ndarray,
    x: int,
    y: int,
    clicking: bool = False,
    click_progress: float = 0.0,
) -> np.ndarray:
    if clicking and click_progress > 0:
        radius = int(8 + 28 * click_progress)
        alpha = max(0.0, 1.0 - click_progress)
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius, (0, 150, 225), 3, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 5, (0, 110, 205), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.75, frame, 1.0 - alpha * 0.75, 0, frame)
    pts = np.array([[x, y], [x, y + 24], [x + 6, y + 19], [x + 12, y + 29], [x + 16, y + 27], [x + 10, y + 17], [x + 18, y + 17]], dtype=np.int32)
    cv2.fillPoly(frame, [pts + 2], (15, 15, 15), cv2.LINE_AA)
    cv2.fillPoly(frame, [pts], (255, 255, 255), cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (0, 0, 0), 2, cv2.LINE_AA)
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
    if page_width > 0 and page_height > 0 and (x > 1.5 or y > 1.5):
        x /= page_width
        y /= page_height
    return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))


def _normalise_box(box: Any, page_width: float, page_height: float) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        values = [float(v) for v in box[:4]]
    except Exception:
        return None
    if max(values) > 1.5 and page_width > 0 and page_height > 0:
        values = [values[0] / page_width, values[1] / page_height, values[2] / page_width, values[3] / page_height]
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return [max(0.0, min(1.0, v)) for v in values]


def _path_pixels(scene: Dict[str, Any], page_width: float, page_height: float, ox: int, oy: int, nw: int, nh: int) -> List[Tuple[int, int]]:
    points=[]
    for item in scene.get("cursor_path") or []:
        norm=_normalise_point(item,page_width,page_height)
        if norm is not None:
            points.append((int(ox+norm[0]*nw),int(oy+norm[1]*nh)))
    return points


def _target_pixels(scene: Dict[str, Any], page_width: float, page_height: float, ox: int, oy: int, nw: int, nh: int):
    point = scene.get("cursor")
    norm = _normalise_point(point, page_width, page_height)
    if norm is None:
        targets = scene.get("targets") or []
        if targets:
            norm = _normalise_point(targets[0], page_width, page_height)
    if norm is None:
        return None, None

    px = int(ox + norm[0] * nw)
    py = int(oy + norm[1] * nh)
    box = _normalise_box(scene.get("cursor_bounding_box"), page_width, page_height)
    if box:
        box_px = [ox + box[0] * nw, oy + box[1] * nh, ox + box[2] * nw, oy + box[3] * nh]
    else:
        box_px = None
    return (px, py), box_px


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _cursor_move_duration(target: Tuple[int, int], start: Tuple[int, int]) -> float:
    distance = ((target[0] - start[0]) ** 2 + (target[1] - start[1]) ** 2) ** 0.5
    return _clamp(0.32 + distance / 700.0, MIN_CURSOR_MOVE, MAX_CURSOR_MOVE)


def _interaction_kind(scene: Dict[str, Any], has_target: bool) -> str:
    if not has_target:
        return "none"
    return str(scene.get("interaction") or "none")


def _timing_plan(scene: Dict[str, Any], narration_duration: float, target: Optional[Tuple[int, int]], previous_cursor: Optional[Tuple[int, int]], same_screenshot: bool) -> Dict[str, float]:
    if target is None:
        return {"move_duration": 0.0, "narration_start": MIN_PRE_ROLL, "interaction_time": -1.0, "hold_after": TRAILING_HOLD_SECONDS}
    if previous_cursor is not None and same_screenshot:
        start = previous_cursor
    else:
        start = (max(18, target[0] - 150), max(18, target[1] - 95))
    move_duration = _cursor_move_duration(target, start)
    narration_start = _clamp(move_duration * 0.55, MIN_PRE_ROLL, MAX_PRE_ROLL)
    interaction = _interaction_kind(scene, True)
    if interaction in {"click", "double_click", "right_click"}:
        interaction_time = max(move_duration, narration_start + _clamp(narration_duration * 0.30, 0.55, 1.20))
    elif interaction == "focus":
        interaction_time = max(move_duration, narration_start + 0.45)
    else:
        interaction_time = -1.0
    hold_after = _clamp(TRAILING_HOLD_SECONDS + (0.25 if interaction in {"click", "double_click", "right_click"} else 0.0), 0.35, 0.80)
    return {"move_duration": move_duration, "narration_start": narration_start, "interaction_time": interaction_time, "hold_after": hold_after, "start_x": float(start[0]), "start_y": float(start[1])}


def _click_event_progress(frame_index: int, interaction_frame: int, interaction: str) -> Tuple[bool, float]:
    if interaction_frame < 0:
        return False, 0.0
    pulse = max(1, int(round(CLICK_PULSE_SECONDS * FPS)))
    if interaction == "double_click":
        gap = max(1, int(round(0.18 * FPS)))
        starts = (interaction_frame, interaction_frame + pulse + gap)
        for start in starts:
            offset = frame_index - start
            if 0 <= offset < pulse:
                return True, offset / float(pulse)
        return False, 0.0
    offset = frame_index - interaction_frame
    if 0 <= offset < pulse:
        return True, offset / float(pulse)
    return False, 0.0


def _render_action_frames(writer, scene, audio_path, raw_img, cues, cumulative, previous_cursor=None, previous_screenshot=None):
    if raw_img is None:
        kind=str(scene.get("kind") or "action")
        if kind in {"explanation","transition"}:
            base = _draw_text_scene(
                scene.get("title") or "Tutorial step",
                scene.get("visual_body") or scene.get("caption_text") or scene.get("source_context") or "Continue with the tutorial.",
                kind,
            )
        else:
            base = _draw_text_scene(
                scene.get("title") or "Tutorial step",
                scene.get("caption_text") or scene.get("source_context") or scene.get("action") or "Visual evidence is unavailable for this action.",
                "action",
            )
        ox, oy, nw, nh = 0, 0, WIDTH, HEIGHT
    else:
        base, (ox, oy, nw, nh) = _fit_image(raw_img)
    page_w = float(scene.get("page_width") or 0)
    page_h = float(scene.get("page_height") or 0)
    target, box = _target_pixels(scene, page_w, page_h, ox, oy, nw, nh)
    target_valid = bool(scene.get("cursor_enabled")) and scene.get("target_status") == "verified" and target is not None
    if not target_valid:
        target, box = None, None

    narration_duration = _duration(audio_path) if audio_path else 0.0
    same_screenshot = bool(previous_screenshot and scene.get("screenshot") == previous_screenshot)
    interaction = _interaction_kind(scene, target_valid)
    path_pixels=_path_pixels(scene,page_w,page_h,ox,oy,nw,nh) if interaction=="drag" and target_valid else []
    timing = _timing_plan(scene, narration_duration, target, previous_cursor, same_screenshot)
    if interaction=="drag" and len(path_pixels)>=2:
        drag_start,drag_end=path_pixels[0],path_pixels[-1]
        origin=previous_cursor if (previous_cursor is not None and same_screenshot) else (max(18,drag_start[0]-150),max(18,drag_start[1]-95))
        move_duration=_cursor_move_duration(drag_start,origin)
        drag_distance=((drag_end[0]-drag_start[0])**2+(drag_end[1]-drag_start[1])**2)**0.5
        drag_duration=_clamp(0.55+drag_distance/900.0,0.55,1.35)
        narration_start=_clamp(move_duration*0.55,MIN_PRE_ROLL,MAX_PRE_ROLL)
        interaction_time=max(move_duration,narration_start+_clamp(narration_duration*0.25,0.45,0.90))
        timing.update({"move_duration":move_duration,"narration_start":narration_start,"interaction_time":interaction_time,"interaction_end_time":interaction_time+drag_duration,"drag_duration":drag_duration,"start_x":float(origin[0]),"start_y":float(origin[1])})
    else:
        timing["interaction_end_time"] = timing.get("interaction_time",-1.0) if timing.get("interaction_time",-1.0)>=0 else -1.0
        timing.setdefault("drag_duration",0.0)
    duration, frame_count, narration_duration = _scene_duration(audio_path, timing)
    narration_start_frame = int(round(timing["narration_start"] * FPS))
    move_frames = int(round(timing["move_duration"] * FPS)) if target else 0
    interaction_frame = int(round(timing["interaction_time"] * FPS)) if timing["interaction_time"] >= 0 else -1
    interaction_end_frame = int(round(timing.get("interaction_end_time",-1.0) * FPS)) if timing.get("interaction_end_time",-1.0)>=0 else -1
    start = None
    if target:
        start = (int(timing.get("start_x", target[0] - 150)), int(timing.get("start_y", target[1] - 95)))
    for fi in range(frame_count):
        frame = base.copy()
        if target and start is not None:
            if interaction=="drag" and len(path_pixels)>=2:
                drag_start,drag_end=path_pixels[0],path_pixels[-1]
                if fi < max(1,move_frames):
                    t=_eased(fi/float(max(1,move_frames)))
                    cx=int(start[0]+(drag_start[0]-start[0])*t); cy=int(start[1]+(drag_start[1]-start[1])*t)
                elif interaction_frame>=0 and fi < max(interaction_frame+1,interaction_end_frame):
                    t=_eased((fi-interaction_frame)/float(max(1,interaction_end_frame-interaction_frame)))
                    cx=int(drag_start[0]+(drag_end[0]-drag_start[0])*t); cy=int(drag_start[1]+(drag_end[1]-drag_start[1])*t)
                else:
                    cx,cy=drag_end
                if fi>=interaction_frame and fi<=interaction_end_frame:
                    frame=_draw_target(frame,drag_start,None,active=True)
                frame=_draw_target(frame,target,box,active=True)
                frame=_draw_mouse_pointer(frame,cx,cy,False,0.0)
            else:
                if fi < max(1, move_frames):
                    t = _eased(fi / float(max(1, move_frames)))
                    cx = int(start[0] + (target[0] - start[0]) * t)
                    cy = int(start[1] + (target[1] - start[1]) * t)
                else:
                    cx, cy = target
                frame = _draw_target(frame, target, box, active=True)
                clicking, pulse = _click_event_progress(fi, interaction_frame, interaction)
                frame = _draw_mouse_pointer(frame, cx, cy, clicking, _clamp(pulse, 0.0, 1.0))
        if fi >= narration_start_frame:
            frame = _draw_caption(frame, scene.get("caption_text") or scene.get("caption") or "")
        writer.write(frame)

    narration_start = cumulative + timing["narration_start"]
    narration_end = narration_start + narration_duration
    cue_text = scene.get("caption_text") or scene.get("caption") or ""
    urls = scene.get("caption_urls") or scene.get("urls") or []
    vtt_text = cue_text
    if urls:
        vtt_text += "\n" + "\n".join(f"URL: {url}" for url in urls)
    if cue_text:
        cue_end = min(
            cumulative + duration,
            narration_start + max(narration_duration, 1.35),
        )
        cues.append((narration_start, max(narration_start + 0.25, cue_end), vtt_text))

    dialogue = scene.get("dialogue", {})
    timeline_item = {
        "start": round(cumulative, 3),
        "end": round(cumulative + duration, 3),
        "narration_start": round(narration_start, 3),
        "narration_end": round(min(cumulative + duration, narration_end), 3),
        "interaction_time": round(cumulative + timing["interaction_time"], 3) if timing["interaction_time"] >= 0 else None,
        "interaction_end_time": round(cumulative + timing.get("interaction_end_time", timing["interaction_time"]), 3) if timing.get("interaction_end_time", -1.0) >= 0 else None,
        "interaction": interaction,
        "scene_id": scene.get("scene_id"),
        "action_id": scene.get("id"),
        "title": scene.get("title") or "Current action",
        "current_action": dialogue.get("current_action") or scene.get("title") or "Follow the current step.",
        "brief": dialogue.get("brief") or scene.get("narration") or "Follow the current step.",
        "purpose": dialogue.get("purpose") or scene.get("source_context") or "",
        "target_name": scene.get("cursor_target_name") if target else "",
        "has_target": bool(target),
        "target_status": scene.get("target_status", "unresolved"),
        "urls": urls,
        "step_index": scene.get("id"),
    }
    scene["timing"] = {
        **{k: round(float(v), 3) for k, v in timing.items() if isinstance(v, (int, float))},
        "duration": round(duration, 3),
        "narration_duration": round(narration_duration, 3),
    }
    scene["interaction_plan"] = {
        "type": interaction,
        "target_required": interaction != "none",
        "cursor_start": [
            round(float(timing.get("start_x", target[0] if target else 0.0)), 3),
            round(float(timing.get("start_y", target[1] if target else 0.0)), 3),
        ] if target else None,
        "cursor_move_duration": round(float(timing.get("move_duration", 0.0)), 3),
        "narration_start": round(float(timing.get("narration_start", MIN_PRE_ROLL)), 3),
        "interaction_time": round(float(timing.get("interaction_time", -1.0)), 3) if timing.get("interaction_time", -1.0) >= 0 else None,
        "interaction_end_time": round(float(timing.get("interaction_end_time", -1.0)), 3) if timing.get("interaction_end_time", -1.0) >= 0 else None,
        "drag_duration": round(float(timing.get("drag_duration", 0.0)), 3),
        "hold_duration": round(float(timing.get("hold_after", TRAILING_HOLD_SECONDS)), 3),
        "scene_duration": round(float(duration), 3),
    }
    scene["cursor_enabled"] = bool(target)
    return duration, timeline_item, target


def _write_silence(
    writer: wave.Wave_write,
    seconds: float,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> None:
    frames = max(
        0,
        int(round(seconds * sample_rate)),
    )
    if frames <= 0:
        return

    writer.writeframes(
        b"\x00"
        * frames
        * channels
        * sample_width
    )


def _concat_audio(
    audio_paths: List[Optional[str]],
    scene_timings: List[Dict[str, Any]],
    output_dir: Path,
    prefix: str,
) -> Optional[str]:
    """
    Build an audio track whose segment length exactly matches each rendered
    scene.

    Each segment is:
        lead-in silence + narration + trailing padding silence

    Missing TTS files become silence for that scene instead of deleting the
    entire audio track. This keeps later narration synchronized and produces a
    valid video even when one TTS item fails.
    """
    if not scene_timings:
        return None

    scene_durations = [float(item.get("duration", MIN_SCENE_DURATION)) for item in scene_timings]

    valid = [
        p for p in audio_paths
        if p and os.path.exists(p)
    ]

    out = output_dir / f"{prefix}_narration.wav"

    # Default to mono 24 kHz PCM when all TTS files are missing.
    default_rate = 24000
    default_channels = 1
    default_width = 2

    if valid:
        try:
            with wave.open(
                str(valid[0]),
                "rb",
            ) as wf:
                first_params = wf.getparams()

            target_rate = first_params.framerate
            target_channels = first_params.nchannels
            target_width = first_params.sampwidth
        except Exception as exc:
            print(
                f"[AUDIO] Could not read reference WAV: {exc}"
            )
            target_rate = default_rate
            target_channels = default_channels
            target_width = default_width
    else:
        target_rate = default_rate
        target_channels = default_channels
        target_width = default_width

    try:
        with wave.open(
            str(out),
            "wb",
        ) as writer:
            writer.setnchannels(
                target_channels
            )
            writer.setsampwidth(
                target_width
            )
            writer.setframerate(
                target_rate
            )

            for index, scene_duration in enumerate(scene_durations):
                audio_path = (
                    audio_paths[index]
                    if index < len(audio_paths)
                    else None
                )

                if (
                    not audio_path
                    or not os.path.exists(audio_path)
                ):
                    _write_silence(
                        writer,
                        scene_duration,
                        target_rate,
                        target_channels,
                        target_width,
                    )
                    continue

                try:
                    with wave.open(
                        str(audio_path),
                        "rb",
                    ) as wf:
                        params = wf.getparams()

                        if (
                            params.nchannels != target_channels
                            or params.sampwidth != target_width
                            or params.framerate != target_rate
                        ):
                            raise ValueError(
                                "incompatible WAV format"
                            )

                        narration_frames = wf.getnframes()
                        narration_seconds = (
                            narration_frames
                            / float(
                                wf.getframerate()
                            )
                        )
                        audio_bytes = wf.readframes(
                            narration_frames
                        )

                    target_scene_frames = max(
                        0,
                        int(
                            round(
                                scene_duration
                                * target_rate
                            )
                        ),
                    )
                    narration_start = float(scene_timings[index].get("narration_start", MIN_PRE_ROLL))
                    lead_frames = max(0, int(round(narration_start * target_rate)))

                    # Keep the total scene length exact in audio samples.
                    # Video duration is already quantized to FPS, so this
                    # avoids even sub-frame accumulation across long tutorials.
                    trailing_frames = max(
                        0,
                        target_scene_frames
                        - lead_frames
                        - narration_frames,
                    )

                    if lead_frames + narration_frames > target_scene_frames:
                        # This can only happen when narration is longer than
                        # the rendered scene. Never extend the audio beyond the
                        # visual timeline; the scene duration calculation should
                        # normally make this impossible.
                        available_narration = max(
                            0,
                            target_scene_frames - lead_frames,
                        )
                        bytes_per_frame = (
                            target_channels
                            * target_width
                        )
                        audio_bytes = audio_bytes[
                            :available_narration
                            * bytes_per_frame
                        ]
                        narration_frames = available_narration
                        trailing_frames = 0

                    writer.writeframes(
                        b"\x00"
                        * lead_frames
                        * target_channels
                        * target_width
                    )
                    if audio_bytes:
                        writer.writeframes(
                            audio_bytes
                        )
                    if trailing_frames:
                        writer.writeframes(
                            b"\x00"
                            * trailing_frames
                            * target_channels
                            * target_width
                        )

                except Exception as exc:
                    print(
                        f"[AUDIO] Scene {index + 1} audio invalid: "
                        f"{exc}; replacing scene with silence."
                    )
                    # The current writer can safely continue from this point;
                    # a complete scene of silence preserves the canonical time.
                    _write_silence(
                        writer,
                        scene_duration,
                        target_rate,
                        target_channels,
                        target_width,
                    )

        return str(out)

    except Exception as exc:
        print(
            f"[AUDIO] Concat failed: {exc}"
        )
        return None


def _format_ts(seconds: float) -> str:
    seconds = max(
        0.0,
        seconds,
    )
    h = int(seconds // 3600)
    m = int(
        (seconds % 3600) // 60
    )
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _write_vtt(
    cues,
    path: Path,
):
    lines = [
        "WEBVTT",
        "",
    ]

    for start, end, text in cues:
        if text:
            lines += [
                f"{_format_ts(start)} --> {_format_ts(end)}",
                text,
                "",
            ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def render_tutorial(
    plan: Dict[str, Any],
    audio: Any,
    job_dir: str,
) -> Dict[str, Any]:
    """Render the entire ordered action timeline into ONE continuous video."""
    job_dir = Path(job_dir)

    steps = [
        s
        for s in plan.get("steps", [])
        if s.get("kind") in {"action", "explanation", "transition"}
    ]

    audio_paths = _audio_files(audio)

    raw_video = job_dir / "tutorial_raw.mp4"
    final_video = job_dir / "tutorial.mp4"
    vtt_path = job_dir / "tutorial.vtt"

    if not steps:
        return {
            "video_file": None,
            "captions_file": None,
            "duration": 0.0,
            "action_count": 0,
        }

    writer = cv2.VideoWriter(
        str(raw_video),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        FPS,
        (WIDTH, HEIGHT),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not open video writer for tutorial_raw.mp4"
        )

    cues = []
    timeline = []
    cumulative = 0.0
    audio_for_concat: List[
        Optional[str]
    ] = []
    previous_cursor = None
    previous_screenshot = None
    scene_timings: List[Dict[str, Any]] = []
    try:
        for index, scene in enumerate(steps):
            img_path = scene.get("screenshot")
            raw_img = cv2.imread(str(img_path)) if img_path and os.path.exists(str(img_path)) else None
            audio_path = audio_paths[index] if index < len(audio_paths) else None
            audio_for_concat.append(audio_path)

            rendered_duration, timeline_item, end_cursor = _render_action_frames(
                writer, scene, audio_path, raw_img, cues, cumulative,
                previous_cursor=previous_cursor,
                previous_screenshot=previous_screenshot,
            )
            timeline.append(timeline_item)
            scene_timings.append(dict(scene.get("timing") or {}, duration=rendered_duration))
            cumulative += rendered_duration
            previous_cursor = end_cursor
            previous_screenshot = img_path
    finally:
        writer.release()

    _write_vtt(
        cues,
        vtt_path,
    )

    timeline_path = (
        job_dir / "tutorial_timeline.json"
    )
    timeline_path.write_text(
        json.dumps(
            timeline,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    narration = _concat_audio(
        audio_for_concat,
        scene_timings,
        job_dir,
        "tutorial",
    )

    ffmpeg = _ffmpeg_exe()

    if narration:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(raw_video),
            "-i",
            str(narration),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(final_video),
        ]
    else:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(raw_video),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(final_video),
        ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if (
        result.returncode != 0
        or not final_video.exists()
    ):
        raise RuntimeError(
            "Final tutorial video encoding failed: "
            + result.stderr[-1500:]
        )

    print(
        f"[RENDER] Completed ONE continuous tutorial: "
        f"{sum(1 for s in steps if s.get('kind')=='action')} actions, "
        f"{sum(1 for s in steps if s.get('kind')=='explanation')} explanations, "
        f"{sum(1 for s in steps if s.get('kind')=='transition')} section intros, "
        f"{cumulative:.1f}s"
    )

    return {
        "video_file": final_video.name,
        "captions_file": vtt_path.name,
        "timeline_file": timeline_path.name,
        "timeline": timeline,
        "duration": round(
            cumulative,
            2,
        ),
        "action_count": sum(1 for s in steps if s.get("kind") == "action"),
    }


def render_all_sections(
    plan: Dict[str, Any],
    audio: Any,
    job_dir: str,
) -> Dict[str, Any]:
    """Backward-compatible entry point. It now renders ONE tutorial, not sections."""
    result = render_tutorial(
        plan,
        audio,
        job_dir,
    )

    plan["video_file"] = result.get(
        "video_file"
    )
    plan["captions_file"] = result.get(
        "captions_file"
    )
    plan["video_duration"] = result.get(
        "duration",
        0.0,
    )
    plan["action_count"] = result.get(
        "action_count",
        0,
    )
    plan["timeline_file"] = result.get(
        "timeline_file"
    )
    plan["dialogue_timeline"] = result.get(
        "timeline",
        [],
    )

    for section in plan.get(
        "sections",
        [],
    ):
        section["video_file"] = None
        section["captions_file"] = None
        section["duration"] = 0.0

    print(
        "[RENDER] Section rendering disabled; "
        "generated a single tutorial.mp4"
    )
    return plan
