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


# ----------------------------------------------------------------------
# AUDIO
# ----------------------------------------------------------------------

def _audio_files(
    audio: Any,
) -> List[str]:
    """
    Extract audio paths from several possible TTS result structures.
    """

    if not audio:
        return []

    items = audio

    if isinstance(
        audio,
        dict,
    ):
        for key in (
            "files",
            "audio_files",
            "steps",
            "narration",
        ):
            if key in audio:
                items = audio[key]
                break

    if isinstance(
        items,
        dict,
    ):
        items = list(
            items.values()
        )

    if not isinstance(
        items,
        list,
    ):
        items = [items]

    paths = []

    for item in items:
        path = None

        if isinstance(
            item,
            str,
        ):
            path = item

        elif isinstance(
            item,
            dict,
        ):
            for key in (
                "path",
                "file",
                "audio",
                "wav",
                "audio_path",
            ):
                value = item.get(
                    key
                )

                if value:
                    path = value
                    break

        if not path:
            continue

        path = str(
            path
        )

        if os.path.exists(
            path
        ):
            paths.append(
                path
            )

    return paths


def _duration(
    path: str,
) -> float:
    try:
        with wave.open(
            path,
            "rb",
        ) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()

            if rate:
                return max(
                    0.5,
                    frames / float(rate),
                )
    except Exception:
        pass

    return 2.0


# ----------------------------------------------------------------------
# IMAGE
# ----------------------------------------------------------------------

def _fit_image(
    image: np.ndarray,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> np.ndarray:

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    if image is None:
        return canvas

    ih, iw = image.shape[:2]

    if iw <= 0 or ih <= 0:
        return canvas

    scale = min(
        width / float(iw),
        height / float(ih),
    )

    nw = max(
        1,
        int(iw * scale),
    )

    nh = max(
        1,
        int(ih * scale),
    )

    resized = cv2.resize(
        image,
        (
            nw,
            nh,
        ),
        interpolation=cv2.INTER_AREA,
    )

    x = (
        width - nw
    ) // 2

    y = (
        height - nh
    ) // 2

    canvas[
        y:y + nh,
        x:x + nw,
    ] = resized

    return canvas


def _load_frame(
    path: Optional[str],
) -> Optional[np.ndarray]:

    if not path:
        return None

    if not os.path.exists(
        path
    ):
        return None

    image = cv2.imread(
        path
    )

    if image is None:
        return None

    return _fit_image(
        image
    )


# ----------------------------------------------------------------------
# CAPTIONS
# ----------------------------------------------------------------------

def _draw_caption(
    frame: np.ndarray,
    text: str,
    kind: str,
) -> np.ndarray:

    if not text:
        return frame

    lines = []

    for raw in str(
        text
    ).splitlines():

        raw = raw.strip()

        if not raw:
            continue

        words = raw.split()

        current = ""

        for word in words:
            candidate = (
                f"{current} {word}".strip()
            )

            if len(candidate) > 70:
                if current:
                    lines.append(
                        current
                    )

                current = word

            else:
                current = candidate

        if current:
            lines.append(
                current
            )

    if not lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.72
    thickness = 2
    line_height = 32
    padding = 20

    total_height = (
        len(lines)
        * line_height
        + padding * 2
    )

    y1 = (
        HEIGHT
        - total_height
        - 18
    )

    y2 = HEIGHT - 18

    y1 = max(
        HEIGHT // 2,
        y1,
    )

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            20,
            y1,
        ),
        (
            WIDTH - 20,
            y2,
        ),
        (0, 0, 0),
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0,
    )

    text_y = (
        y1
        + padding
        + 24
    )

    for line in lines:
        (
            tw,
            th,
        ), _ = cv2.getTextSize(
            line,
            font,
            scale,
            thickness,
        )

        x = (
            WIDTH - tw
        ) // 2

        cv2.putText(
            frame,
            line,
            (
                x,
                text_y,
            ),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        text_y += line_height

    return frame


# ----------------------------------------------------------------------
# CURSOR
# ----------------------------------------------------------------------

def _valid_point(
    point: Any,
) -> bool:
    return (
        isinstance(
            point,
            (list, tuple),
        )
        and len(point) == 2
    )


def _normalise_point(
    point,
) -> Optional[
    Tuple[float, float]
]:

    if not _valid_point(
        point
    ):
        return None

    try:
        x = float(
            point[0]
        )

        y = float(
            point[1]
        )

    except Exception:
        return None

    if not (
        0.0 <= x <= 1.0
        and 0.0 <= y <= 1.0
    ):
        return None

    return (
        x,
        y,
    )


def _draw_cursor(
    frame: np.ndarray,
    point: Tuple[float, float],
    scale: float = 1.0,
) -> np.ndarray:

    x = int(
        max(
            0.0,
            min(
                1.0,
                point[0],
            ),
        )
        * WIDTH
    )

    y = int(
        max(
            0.0,
            min(
                1.0,
                point[1],
            ),
        )
        * HEIGHT
    )

    radius = max(
        6,
        int(
            10 * scale
        ),
    )

    cv2.circle(
        frame,
        (
            x + 3,
            y + 3,
        ),
        radius + 2,
        (0, 0, 0),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        (
            x,
            y,
        ),
        radius,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        (
            x,
            y,
        ),
        radius,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return frame


def _draw_click(
    frame: np.ndarray,
    point: Tuple[float, float],
    progress: float,
) -> np.ndarray:

    x = int(
        max(
            0.0,
            min(
                1.0,
                point[0],
            ),
        )
        * WIDTH
    )

    y = int(
        max(
            0.0,
            min(
                1.0,
                point[1],
            ),
        )
        * HEIGHT
    )

    progress = max(
        0.0,
        min(
            1.0,
            progress,
        ),
    )

    radius = int(
        12
        + 35 * progress
    )

    thickness = max(
        1,
        int(
            5
            * (1.0 - progress)
        ),
    )

    cv2.circle(
        frame,
        (
            x,
            y,
        ),
        radius,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )

    return frame


def _interpolate(
    start: Tuple[float, float],
    end: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:

    t = max(
        0.0,
        min(
            1.0,
            t,
        ),
    )

    # Smoothstep.
    t = (
        t
        * t
        * (
            3.0
            - 2.0 * t
        )
    )

    return (
        start[0]
        + (
            end[0]
            - start[0]
        )
        * t,

        start[1]
        + (
            end[1]
            - start[1]
        )
        * t,
    )


# ----------------------------------------------------------------------
# SCENE TARGET
# ----------------------------------------------------------------------

def _scene_target(
    scene: Dict[str, Any],
) -> Optional[
    Tuple[float, float]
]:

    target = _normalise_point(
        scene.get(
            "cursor"
        )
    )

    if target:
        return target

    actions = (
        scene.get(
            "actions"
        )
        or []
    )

    for action in actions:

        if not isinstance(
            action,
            dict,
        ):
            continue

        target = _normalise_point(
            action.get(
                "target"
            )
        )

        if target:
            return target

    return None


# ----------------------------------------------------------------------
# AUDIO / SCENE DURATION
# ----------------------------------------------------------------------

# Minimum on-screen time for a scene, even when the TTS clip is very
# short. Gives the viewer time to actually see the screenshot, watch
# the cursor glide + click, and read the caption instead of the frame
# flashing past. Actions get more floor than explanations because they
# also have to fit the ~0.8s cursor-move-and-click animation.
MIN_SCENE_SECONDS = {
    "action": 2.6,
    "explanation": 2.0,
    "topic": 1.6,
}

# Extra breathing room added on top of the raw narration length so
# pacing doesn't feel rushed even when the TTS engine speaks quickly.
SCENE_PADDING_SECONDS = 0.5


def _audio_duration_for_scene(
    scene: Dict[str, Any],
    index: int,
    audio_paths: List[str],
) -> float:

    kind = scene.get("kind", "action")
    floor = MIN_SCENE_SECONDS.get(kind, 2.0)

    # The most reliable mapping is a direct audio index.
    if index < len(
        audio_paths
    ):
        raw = _duration(
            audio_paths[index]
        )
        return max(
            floor,
            raw + SCENE_PADDING_SECONDS,
        )

    narration = (
        scene.get(
            "narration"
        )
        or scene.get(
            "tts_narration"
        )
        or ""
    )

    if narration:
        # Reasonable fallback when TTS output has
        # fewer files than scenes.
        words = len(
            str(narration).split()
        )

        estimate = min(
            8.0,
            words / 2.5,
        )

        return max(
            floor,
            estimate + SCENE_PADDING_SECONDS,
        )

    return floor


# ----------------------------------------------------------------------
# RAW VIDEO
# ----------------------------------------------------------------------

def _write_raw_video(
    plan: Dict[str, Any],
    audio: Any,
    output_path: str,
) -> None:

    steps = (
        plan.get(
            "steps"
        )
        or []
    )

    if not steps:
        raise RuntimeError(
            "Tutorial plan contains no scenes."
        )

    audio_paths = _audio_files(
        audio
    )

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        FPS,
        (
            WIDTH,
            HEIGHT,
        ),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not open raw video writer."
        )

    previous_frame = None

    try:
        for index, scene in enumerate(
            steps
        ):

            screenshot = scene.get(
                "screenshot"
            )

            frame = _load_frame(
                screenshot
            )

            if frame is None:

                if previous_frame is not None:
                    frame = previous_frame.copy()

                else:
                    frame = np.zeros(
                        (
                            HEIGHT,
                            WIDTH,
                            3,
                        ),
                        dtype=np.uint8,
                    )

            previous_frame = frame.copy()

            duration = _audio_duration_for_scene(
                scene,
                index,
                audio_paths,
            )

            frame_count = max(
                1,
                int(
                    duration * FPS
                ),
            )

            kind = scene.get(
                "kind",
                "action",
            )

            target = _scene_target(
                scene
            )

            # Cursor is shown ONLY when an actual
            # grounded target exists.
            show_cursor = (
                kind == "action"
                and target is not None
            )

            for frame_index in range(
                frame_count
            ):

                current = frame.copy()

                if show_cursor:

                    movement_frames = max(
                        1,
                        int(
                            FPS * 0.55
                        ),
                    )

                    movement_fraction = min(
                        1.0,
                        frame_index
                        / float(
                            movement_frames
                        ),
                    )

                    # Start just outside the
                    # target region rather than
                    # appearing at the target.
                    start = (
                        max(
                            0.02,
                            min(
                                0.95,
                                target[0]
                                - 0.18,
                            ),
                        ),
                        max(
                            0.02,
                            min(
                                0.90,
                                target[1]
                                - 0.12,
                            ),
                        ),
                    )

                    cursor_pos = _interpolate(
                        start,
                        target,
                        movement_fraction,
                    )

                    current = _draw_cursor(
                        current,
                        cursor_pos,
                    )

                    # Click pulse near end.
                    remaining = (
                        frame_count
                        - frame_index
                    )

                    click_frames = max(
                        1,
                        int(
                            FPS * 0.25
                        ),
                    )

                    if remaining <= click_frames:

                        pulse_progress = 1.0 - (
                            remaining
                            / float(
                                click_frames
                            )
                        )

                        current = _draw_click(
                            current,
                            target,
                            pulse_progress,
                        )

                current = _draw_caption(
                    current,
                    scene.get(
                        "caption",
                        "",
                    ),
                    kind,
                )

                writer.write(
                    current
                )

    finally:
        writer.release()


# ----------------------------------------------------------------------
# AUDIO CONCAT
# ----------------------------------------------------------------------

def _build_audio_concat(
    audio_paths: List[str],
    output_dir: str,
) -> Optional[str]:

    if not audio_paths:
        print(
            "[AUDIO] No audio files found."
        )
        return None

    concat_file = os.path.join(
        output_dir,
        "audio_concat.txt",
    )

    narration_file = os.path.join(
        output_dir,
        "narration.wav",
    )

    with open(
        concat_file,
        "w",
        encoding="utf-8",
    ) as f:

        for path in audio_paths:

            absolute = os.path.abspath(
                path
            )

            # FFmpeg concat file syntax.
            safe_path = (
                absolute
                .replace(
                    "\\",
                    "/",
                )
                .replace(
                    "'",
                    "'\\''",
                )
            )

            f.write(
                f"file '{safe_path}'\n"
            )

    command = [
        "ffmpeg",
        "-y",

        "-f",
        "concat",

        "-safe",
        "0",

        "-i",
        concat_file,

        "-c:a",
        "pcm_s16le",

        narration_file,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        print(
            "[FFMPEG] audio concat failed"
        )

        print(
            result.stdout
        )

        print(
            result.stderr
        )

        return None

    if not os.path.exists(
        narration_file
    ):
        return None

    return narration_file


# ----------------------------------------------------------------------
# FINAL MP4
# ----------------------------------------------------------------------

def _finalize_with_ffmpeg(
    raw_video: str,
    narration: Optional[str],
    final_path: str,
) -> None:

    if (
        narration
        and os.path.exists(
            narration
        )
    ):

        command = [
            "ffmpeg",
            "-y",

            "-i",
            raw_video,

            "-i",
            narration,

            "-map",
            "0:v:0",

            "-map",
            "1:a:0",

            "-c:v",
            "libx264",

            "-preset",
            "veryfast",

            "-crf",
            "20",

            "-pix_fmt",
            "yuv420p",

            "-c:a",
            "aac",

            "-b:a",
            "128k",

            "-ar",
            "48000",

            "-movflags",
            "+faststart",

            "-shortest",

            final_path,
        ]

    else:

        print(
            "[AUDIO] Final video will contain no audio "
            "because no narration WAV was produced."
        )

        command = [
            "ffmpeg",
            "-y",

            "-i",
            raw_video,

            "-map",
            "0:v:0",

            "-c:v",
            "libx264",

            "-preset",
            "veryfast",

            "-crf",
            "20",

            "-pix_fmt",
            "yuv420p",

            "-an",

            "-movflags",
            "+faststart",

            final_path,
        ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    print(
        "[FFMPEG]"
    )

    if result.stdout:
        print(
            result.stdout
        )

    if result.stderr:
        print(
            result.stderr
        )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg failed to create final MP4."
        )

    if not os.path.exists(
        final_path
    ):
        raise RuntimeError(
            "FFmpeg reported success but "
            "the final MP4 does not exist."
        )


# ----------------------------------------------------------------------
# PUBLIC RENDER FUNCTION
# ----------------------------------------------------------------------

def render(
    plan: Dict[str, Any],
    audio: Any,
    job_dir: str,
    final_path: str,
) -> str:

    job_dir = str(
        job_dir
    )

    os.makedirs(
        job_dir,
        exist_ok=True,
    )

    final_path = str(
        final_path
    )

    raw_video = os.path.join(
        job_dir,
        "video_raw.mp4",
    )

    audio_paths = _audio_files(
        audio
    )

    print(
        f"[RENDER] scenes={len(plan.get('steps') or [])}"
    )

    print(
        f"[RENDER] audio_files={len(audio_paths)}"
    )

    narration = _build_audio_concat(
        audio_paths,
        job_dir,
    )

    _write_raw_video(
        plan,
        audio,
        raw_video,
    )

    _finalize_with_ffmpeg(
        raw_video,
        narration,
        final_path,
    )

    print(
        f"[RENDER] final={final_path}"
    )

    return final_path