import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf


SAMPLE_RATE = 24000


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

    # Kokoro 0.9.4:
    # result.output.audio
    output = getattr(result, "output", None)

    if output is not None:
        audio = getattr(output, "audio", None)

        if audio is not None:
            return audio

    # Compatibility with older/different Kokoro result shapes.
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
    """
    Convert Kokoro/PyTorch audio into a clean 1-D float32 NumPy array.
    """

    if audio is None:
        return None

    # Torch tensor
    if hasattr(audio, "detach"):
        try:
            audio = audio.detach().cpu().numpy()
        except Exception:
            pass

    # Objects exposing numpy()
    elif hasattr(audio, "numpy"):
        try:
            audio = audio.numpy()
        except Exception:
            pass

    array = np.asarray(audio)

    # Flatten safely.
    # This avoids the "inhomogeneous shape" error from the old code.
    if array.dtype == object:
        flattened = []

        for item in array:
            try:
                item_array = np.asarray(item, dtype=np.float32).reshape(-1)
                flattened.append(item_array)
            except Exception:
                continue

        if not flattened:
            return None

        array = np.concatenate(flattened)

    else:
        array = array.astype(np.float32, copy=False).reshape(-1)

    return array


def _kokoro_generate(
    text: str,
    output_path: Path,
    voice: str = "af_heart",
):
    """
    Generate a WAV file using Kokoro 0.9.4.
    """

    try:
        from kokoro import KPipeline
    except Exception as exc:
        raise RuntimeError(
            f"Could not import Kokoro: {exc}"
        ) from exc

    # English Kokoro pipeline.
    pipeline = KPipeline(lang_code="a")

    audio_chunks = []

    # Kokoro 0.9.4 returns a generator of KPipeline.Result objects.
    results = pipeline(
        text,
        voice=voice,
        speed=1,
    )

    for result in results:
        audio = _extract_audio(result)

        if audio is None:
            continue

        audio_array = _to_numpy_audio(audio)

        if audio_array is None or audio_array.size == 0:
            continue

        audio_chunks.append(audio_array)

    if not audio_chunks:
        raise RuntimeError(
            "Kokoro produced no audio."
        )

    # Join all sentence/segment outputs into one WAV.
    audio = np.concatenate(audio_chunks).astype(
        np.float32,
        copy=False,
    )

    # Protect against NaN/Inf values.
    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Make sure output directory exists.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(output_path),
        audio,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    return str(output_path)


def generate_narration(
    plan: Dict[str, Any],
    job_dir: str,
    progress_callback=None,
    language: str = "en-us",
    voice: str = "af_heart",
) -> Dict[str, Any]:
    """
    Generate one WAV file for every tutorial action.

    Output remains compatible with video_service.py:
        {
            "files": [...],
            "audio_files": [...],
            ...
        }
    """

    job_dir = Path(job_dir)

    audio_dir = job_dir / "audio"

    audio_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    steps = plan.get("steps", [])

    total = len(steps)

    if total == 0:
        return {
            "files": [],
            "audio_files": [],
            "language": language,
            "voice": voice,
            "count": 0,
        }

    files: List[Optional[str]] = []

    for index, step in enumerate(steps):

        text = (
            step.get("tts_narration")
            or step.get("narration")
            or step.get("caption")
            or step.get("title")
            or "Continue with the next step."
        )

        text = str(text).strip()

        if not text:
            text = "Continue with the next step."

        output_path = (
            audio_dir /
            f"step_{index + 1:04d}.wav"
        )

        try:

            # -------------------------------------------------
            # CACHE
            # -------------------------------------------------
            if (
                output_path.exists()
                and output_path.stat().st_size > 1000
            ):

                files.append(
                    str(output_path)
                )

                print(
                    f"[TTS] Using cached audio "
                    f"for step {index + 1}/{total}"
                )

            else:

                generated = _kokoro_generate(
                    text=text,
                    output_path=output_path,
                    voice=voice,
                )

                files.append(
                    str(generated)
                )

                print(
                    f"[TTS] Generated step "
                    f"{index + 1}/{total}"
                )

        except Exception as exc:

            print(
                f"[TTS] Failed step "
                f"{index + 1}: {exc}"
            )

            files.append(None)

        # Correct progress: 0 -> 100
        if progress_callback:

            progress = int(
                ((index + 1) / total) * 100
            )

            progress_callback(
                progress,
                f"Generated narration "
                f"{index + 1} of {total}",
            )

    result = {
        "files": files,
        "audio_files": files,
        "language": language,
        "voice": voice,
        "count": len(files),
    }

    # Save manifest.
    manifest_path = job_dir / "audio.json"

    try:

        manifest_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    except Exception as exc:

        print(
            f"[TTS] Could not write "
            f"audio manifest: {exc}"
        )

    successful = sum(
        1 for item in files
        if item
    )

    print(
        f"[TTS] Completed: "
        f"{successful}/{total} audio files generated"
    )

    return result