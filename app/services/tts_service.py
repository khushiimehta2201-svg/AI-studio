import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf


SAMPLE_RATE = 24000

_URL_RE = re.compile(
    r"(?<![\w@])(?:https?://|www\.)[^\s<>\]\[\"')]+",
    flags=re.IGNORECASE,
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _remove_urls(text: str) -> str:
    return _clean_text(_URL_RE.sub(" ", str(text or "")))


def _extract_audio(result):
    """
    Extract the audio array from Kokoro 0.9.4 KPipeline.Result.

    Kokoro 0.9.4 returns:
        KPipeline.Result(
            graphemes=...,
            phonemes=...,
            output=...
        )

    The actual audio is:
        result.output.audio
    """
    if result is None:
        return None

    output = getattr(result, "output", None)

    if output is not None:
        audio = getattr(output, "audio", None)
        if audio is not None:
            return audio

    if isinstance(result, tuple):
        if len(result) >= 3:
            return result[-1]

    if isinstance(result, np.ndarray):
        return result

    if hasattr(result, "numpy"):
        try:
            return result.numpy()
        except Exception:
            pass

    return None


def _to_numpy_audio(audio):
    """Convert Kokoro/PyTorch audio into a clean 1-D float32 NumPy array."""
    if audio is None:
        return None

    if hasattr(audio, "detach"):
        try:
            audio = audio.detach().cpu().numpy()
        except Exception:
            pass
    elif hasattr(audio, "numpy"):
        try:
            audio = audio.numpy()
        except Exception:
            pass

    array = np.asarray(audio)

    if array.dtype == object:
        flattened = []
        for item in array:
            try:
                item_array = np.asarray(
                    item,
                    dtype=np.float32,
                ).reshape(-1)
                flattened.append(item_array)
            except Exception:
                continue

        if not flattened:
            return None

        array = np.concatenate(flattened)
    else:
        array = array.astype(
            np.float32,
            copy=False,
        ).reshape(-1)

    return array


_PIPELINE_UNAVAILABLE = object()


def _create_pipeline():
    try:
        from kokoro import KPipeline
        return KPipeline(lang_code="a")
    except Exception as exc:
        raise RuntimeError(f"Could not initialize Kokoro: {exc}") from exc


def _kokoro_generate(text: str, output_path: Path, voice: str = "af_heart", pipeline=None):
    """Generate a URL-free WAV using a supplied/reused Kokoro pipeline."""
    if pipeline is None:
        pipeline = _create_pipeline()

    text = _remove_urls(text)
    if not text:
        text = "Continue with the next step."

    audio_chunks = []
    for result in pipeline(text, voice=voice, speed=1):
        audio = _extract_audio(result)
        if audio is None:
            continue
        audio_array = _to_numpy_audio(audio)
        if audio_array is None or audio_array.size == 0:
            continue
        audio_chunks.append(audio_array)

    if not audio_chunks:
        raise RuntimeError("Kokoro produced no audio.")

    audio = np.concatenate(audio_chunks).astype(np.float32, copy=False)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, SAMPLE_RATE, subtype="PCM_16")
    return str(output_path)


def generate_narration(
    plan: Dict[str, Any],
    job_dir: str,
    progress_callback=None,
    language: str = "en-us",
    voice: str = "af_heart",
) -> Dict[str, Any]:
    """Generate one WAV per tutorial action while reusing one Kokoro pipeline per job."""
    job_dir = Path(job_dir)
    audio_dir = job_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    steps = plan.get("steps", [])
    total = len(steps)
    if total == 0:
        return {"files": [], "audio_files": [], "language": language, "voice": voice, "count": 0}

    files: List[Optional[str]] = []
    pipeline = None
    pipeline_error: Optional[str] = None
    pending_generation = any(
        not (
            (audio_dir / f"step_{index + 1:04d}.wav").exists()
            and (audio_dir / f"step_{index + 1:04d}.wav").stat().st_size > 1000
        )
        for index in range(total)
    )
    if pending_generation:
        try:
            pipeline = _create_pipeline()
        except Exception as exc:
            pipeline_error = str(exc)
            print(f"[TTS] Kokoro initialization failed: {exc}")

    for index, step in enumerate(steps):
        text = (
            step.get("tts_narration")
            or step.get("narration")
            or step.get("caption_text")
            or step.get("caption")
            or step.get("title")
            or "Continue with the next step."
        )
        text = _remove_urls(str(text).strip()) or "Continue with the next step."
        output_path = audio_dir / f"step_{index + 1:04d}.wav"

        try:
            if output_path.exists() and output_path.stat().st_size > 1000:
                files.append(str(output_path))
                print(f"[TTS] Using cached audio for step {index + 1}/{total}")
            elif pipeline is None:
                files.append(None)
                print(f"[TTS] Skipped step {index + 1}/{total}: {pipeline_error or 'Kokoro unavailable'}")
            else:
                generated = _kokoro_generate(text=text, output_path=output_path, voice=voice, pipeline=pipeline)
                files.append(str(generated))
                print(f"[TTS] Generated step {index + 1}/{total}")
        except Exception as exc:
            print(f"[TTS] Failed step {index + 1}: {exc}")
            files.append(None)

        if progress_callback:
            progress_callback(int(((index + 1) / total) * 100), f"Generated narration {index + 1} of {total}")

    result = {
        "files": files,
        "audio_files": files,
        "language": language,
        "voice": voice,
        "count": len(files),
        "successful": sum(1 for item in files if item),
    }

    manifest_path = job_dir / "audio.json"
    try:
        manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        print(f"[TTS] Could not write audio manifest: {exc}")

    print(f"[TTS] Completed: {result['successful']}/{total} audio files generated")
    if result["successful"] == 0:
        raise RuntimeError(
            "Narration could not be generated for any step. "
            + (pipeline_error or "Check the TTS configuration and voice model.")
        )
    return result
