import os
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "sapi").strip().lower()


def wav_duration(path: str) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / max(1.0, float(w.getframerate()))
    except Exception:
        return 0.0


def _generate_windows_sapi(text: str, output_path: str) -> float:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean text to single line and double up single-quotes for PowerShell safety
    clean_text = " ".join(str(text or "Next step.").replace("'", "''").split())
    safe_output = str(output_path.resolve()).replace("'", "''")

    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.Volume = 100
$synth.SetOutputToWaveFile('{safe_output}')
$synth.Speak('{clean_text}')
$synth.SetOutputToNull()
$synth.Dispose()
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Windows SAPI error: {result.stderr.strip()}")

    duration = wav_duration(str(output_path))
    if duration <= 0.05:
        raise RuntimeError("SAPI produced empty audio.")
    return duration


def _generate_kokoro(text: str, output_path: str) -> float:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipeline = KPipeline(lang_code="a")
    segments = [audio for _, _, audio in pipeline(text, voice="af_heart")]
    audio = np.concatenate(segments) if len(segments) > 1 else segments[0]
    sf.write(str(output_path), audio, 24000, subtype="PCM_16")
    return wav_duration(output_path)


def generate_narration(
    plan: Dict[str, Any],
    job_dir: str,
    progress_callback=None,
    language="en-us",
    voice="af_heart",
) -> List[Dict[str, Any]]:
    job_dir = Path(job_dir)
    out_dir = job_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = plan.get("steps", [])
    if not steps:
        return []

    result = []
    total = len(steps)

    for i, step in enumerate(steps):
        text = str(step.get("tts_narration") or step.get("narration") or "Next step.").strip()
        wav_path = out_dir / f"step_{i:03d}.wav"

        if progress_callback:
            progress_callback(i, total, f"Synthesizing voice for scene {i + 1} of {total}")

        try:
            if TTS_PROVIDER == "sapi":
                duration = _generate_windows_sapi(text, str(wav_path))
            else:
                duration = _generate_kokoro(text, str(wav_path))
        except Exception as e:
            print(f"[TTS] Error on step {i+1}: {e}. Retrying SAPI fallback...")
            duration = _generate_windows_sapi(text, str(wav_path))

        result.append({
            "index": i,
            "text": text,
            "path": str(wav_path),
            "duration": duration,
        })

    if progress_callback:
        progress_callback(total, total, "Audio narration ready")

    return result
