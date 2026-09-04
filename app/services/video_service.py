from pathlib import Path
import math
import subprocess
import wave

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1280, 720, 24


def wav_duration(path):
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / max(1, w.getframerate())
    except Exception:
        return 0.0


def create_silent_wav(path, duration):
    rate = 16000
    frames = int(max(0.1, float(duration)) * rate)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * frames)


def fit_image(image):
    image = image.convert("RGB")
    iw, ih = image.size
    scale = min(W / max(1, iw), H / max(1, ih))
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    ox, oy = (W - nw) // 2, (H - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, ox, oy, scale


def cursor_path(a, b, n):
    if not b:
        return []
    if n <= 1:
        return [b]
    sx, sy = a
    tx, ty = b
    dx, dy = tx - sx, ty - sy
    length = max(1e-6, math.hypot(dx, dy))
    nx, ny = -dy / length, dx / length
    bow = min(55, length * 0.12)
    cx, cy = (sx + tx) / 2 + nx * bow, (sy + ty) / 2 + ny * bow
    out = []
    for i in range(n):
        u = i / max(1, n - 1)
        e = u * u * (3 - 2 * u)
        a = (1 - e) * (1 - e)
        b1 = 2 * (1 - e) * e
        c = e * e
        out.append((a * sx + b1 * cx + c * tx, a * sy + b1 * cy + c * ty))
    return out


def draw_caption(img, text):
    words = str(text or "").split()
    lines, cur = [], ""
    for word in words:
        candidate = (cur + " " + word).strip()
        if cur and len(candidate) > 78:
            lines.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    lines = lines[:3]
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except Exception:
        font = ImageFont.load_default()
    widths, heights = [], []
    for line in lines:
        box = d.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    if not widths:
        return
    bw = min(W - 60, max(widths) + 44)
    lh = max(heights) + 8
    bh = lh * len(lines) + 24
    x = (W - bw) // 2
    y = H - bh - 18
    d.rounded_rectangle((x, y, x + bw, y + bh), radius=12, fill=(0, 0, 0), outline=(255, 255, 255), width=1)
    yy = y + 12
    for line, height in zip(lines, heights):
        box = d.textbbox((0, 0), line, font=font)
        d.text(((W - (box[2] - box[0])) // 2, yy), line, fill=(255, 255, 255), font=font)
        yy += lh


def draw_cursor(img, x, y, click=False):
    d = ImageDraw.Draw(img)
    x = int(max(8, min(W - 28, x)))
    y = int(max(8, min(H - 34, y)))
    p = [(x, y), (x, y + 24), (x + 7, y + 18), (x + 14, y + 30), (x + 19, y + 27), (x + 12, y + 16), (x + 22, y + 16)]
    shadow = [(a + 3, b + 3) for a, b in p]
    d.polygon(shadow, fill=(60, 60, 60))
    d.polygon(p, fill=(255, 255, 255), outline=(0, 0, 0))
    if click:
        for r in (15, 23):
            d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 115, 35), width=3)


def draw_target(img, x, y, click=False):
    d = ImageDraw.Draw(img)
    r = 28
    d.ellipse((x - r, y - r, x + r, y + r), outline=(255, 185, 0), width=3)
    d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 185, 0))
    if click:
        d.rectangle((x - r - 4, y - r - 4, x + r + 4, y + r + 4), outline=(255, 110, 40), width=2)


def _shot(step, job):
    raw = step.get("screenshot")
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    p = Path(job) / raw
    return p if p.exists() else None


def _norm_cursor(step):
    c = step.get("cursor")
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        try:
            return float(c[0]), float(c[1])
        except Exception:
            return None
    return None


def _duration(step, audio, i):
    if i < len(audio):
        item = audio[i]
        p = item.get("path")
        if p and Path(p).exists():
            d = wav_duration(p)
            if d > 0:
                return d
        try:
            d = float(item.get("duration", 0))
            if d > 0:
                return d
        except Exception:
            pass
    words = len(str(step.get("narration") or step.get("title") or "").split())
    return max(2.2, min(7.0, words / 2.8 + 0.8))


def render(plan, audio, job, final, progress_callback=None):
    job = Path(job)
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    if not steps:
        raise RuntimeError("Tutorial plan contains no action scenes.")

    silent = job / "video_silent.mp4"
    cmd = [
        FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    prev = (W * 0.15, H * 0.20)
    timeline = []

    try:
        for i, step in enumerate(steps):
            duration = _duration(step, audio, i)
            shot = _shot(step, job)
            if not shot:
                # Do not inject a random PDF page, logo, cover, or document page.
                # The step remains narratable but is represented by a clean action card.
                base = Image.new("RGB", (W, H), (245, 245, 245))
                cursor_target = (W * 0.50, H * 0.48)
                has_screenshot = False
            else:
                src = Image.open(shot).convert("RGB")
                base, ox, oy, scale = fit_image(src)
                c = _norm_cursor(step) or (0.50, 0.50)
                cursor_target = (ox + c[0] * src.width * scale, oy + c[1] * src.height * scale)
                has_screenshot = True

            frames = max(1, int(round(duration * FPS)))
            move_frames = min(frames, max(18, int(frames * 0.32)))
            path = cursor_path(prev, cursor_target, move_frames)

            for fi in range(frames):
                frame = base.copy()
                if not has_screenshot:
                    d = ImageDraw.Draw(frame)
                    try:
                        title_font = ImageFont.truetype("arial.ttf", 34)
                        body_font = ImageFont.truetype("arial.ttf", 22)
                    except Exception:
                        title_font = body_font = ImageFont.load_default()
                    d.text((70, 80), str(step.get("title") or "Action"), fill=(25, 25, 25), font=title_font)
                    d.text((70, 150), "Visual reference not available in this step.", fill=(70, 70, 70), font=body_font)

                target_pt = path[fi] if fi < len(path) else cursor_target
                clicking = move_frames <= fi < min(frames, move_frames + max(7, int(FPS * 0.22)))
                draw_target(frame, target_pt[0], target_pt[1], clicking)
                draw_cursor(frame, target_pt[0] + 4, target_pt[1] + 4, clicking)
                draw_caption(frame, step.get("caption") or step.get("narration") or step.get("title", ""))
                proc.stdin.write(frame.tobytes())

            prev = cursor_target
            timeline.append({"duration": duration})
            print(f"[VIDEO] scene={i + 1}/{len(steps)} screenshot={shot.name if shot else 'none'} cursor=visible target={step.get('visual_target','')}")
            if progress_callback:
                progress_callback(75 + int(15 * ((i + 1) / max(1, len(steps)))), f"Rendering step {i + 1} of {len(steps)}")
    finally:
        proc.stdin.close()

    err = proc.stderr.read()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("FFmpeg video rendering failed:\n" + err.decode(errors="replace"))

    audio_files = []
    for i, item in enumerate(timeline):
        p = Path(audio[i].get("path")) if i < len(audio) and audio[i].get("path") else None
        if not p or not p.exists():
            p = job / "audio" / f"silent_{i:03d}.wav"
            create_silent_wav(p, item["duration"])
        audio_files.append(p)

    concat = job / "concat_audio.txt"
    concat.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in audio_files), encoding="utf-8")
    narration = job / "narration.wav"
    subprocess.run(
        [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c:a", "pcm_s16le", str(narration)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )

    total_v = sum(x["duration"] for x in timeline)
    total_a = wav_duration(narration)
    duration = min(total_v, total_a) if total_a > 0 else total_v

    subprocess.run(
        [FFMPEG, "-y", "-i", str(silent), "-i", str(narration), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", f"{duration:.3f}", "-movflags", "+faststart", str(final)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    return final
