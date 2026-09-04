from pathlib import Path
import os
import subprocess
import wave

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

TTS_PROVIDER = os.getenv("TTS_PROVIDER", "sapi").strip().lower()

# English-only configuration.
KOKORO_DEFAULT_LANGUAGE = "en-us"
KOKORO_DEFAULT_VOICE = "af_heart"
KOKORO_LANGUAGE_CODE = "a"
KOKORO_ALLOWED_VOICES = {
    "af_alloy", "af_heart", "af_jessica", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_michael", "am_onyx", "am_puck",
}

# Legacy compatibility names retained for older tests/callers.
# The public application always forces English (en-us).
HINGLISH_TTS_REPO = os.getenv("HINGLISH_TTS_REPO", "").strip()
HINGLISH_REF_AUDIO = os.getenv("HINGLISH_REF_AUDIO", "").strip()
HINGLISH_REF_TEXT_PATH = os.getenv("HINGLISH_REF_TEXT", "").strip()
HINGLISH_PYTHON = os.getenv("HINGLISH_PYTHON", "python3").strip()
HINGLISH_TTS_TIMEOUT = int(os.getenv("HINGLISH_TTS_TIMEOUT", "120"))

KOKORO_LANGUAGE_CONFIG = {
    "hinglish": {"label": "Legacy Hinglish", "lang_code": "h", "default_voice": "hf_alpha"},
    "hi": {"label": "Legacy Hindi", "lang_code": "h", "default_voice": "hf_alpha"},
    "en-us": {"label": "English (US)", "lang_code": "a", "default_voice": "af_heart"},
}

KOKORO_VOICES = {
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "a": sorted(KOKORO_ALLOWED_VOICES),
}

_kokoro_pipeline = {}


def wav_duration(path):
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / max(1, w.getframerate())
    except Exception:
        return 0.0


def _estimate_duration(text):
    words = len((text or "").split())
    return max(2.5, min(12.0, words / 2.5))


def _kokoro_config(language):
    key = str(language or KOKORO_DEFAULT_LANGUAGE).strip().lower()
    if key not in KOKORO_LANGUAGE_CONFIG:
        raise RuntimeError(f"Unsupported Kokoro language '{key}'.")
    return key, KOKORO_LANGUAGE_CONFIG[key]


def _get_kokoro_pipeline(language=None):
    global _kokoro_pipeline
    _, config = _kokoro_config(language or KOKORO_DEFAULT_LANGUAGE)
    lang_code = config["lang_code"]
    # Keep a pipeline per language family for backward compatibility; the
    # application uses English only and therefore normally creates one.
    if not isinstance(_kokoro_pipeline, dict):
        _kokoro_pipeline = {}
    if lang_code not in _kokoro_pipeline:
        from kokoro import KPipeline
        _kokoro_pipeline[lang_code] = KPipeline(lang_code=lang_code)
    return _kokoro_pipeline[lang_code], config


def _generate_kokoro_speech(text, output, language=None, voice=None):
    import numpy as np
    import soundfile as sf

    _, config = _kokoro_config(language or KOKORO_DEFAULT_LANGUAGE)
    selected_voice = str(voice or config["default_voice"]).strip()
    allowed = KOKORO_VOICES.get(config["lang_code"], [])
    if selected_voice not in allowed:
        selected_voice = config["default_voice"]

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pipeline, _ = _get_kokoro_pipeline(language or KOKORO_DEFAULT_LANGUAGE)
    segments = []
    for _, _, audio in pipeline(str(text or "").strip(), voice=selected_voice):
        segments.append(audio)

    if not segments:
        raise RuntimeError("Kokoro produced no audio segments.")

    audio = np.concatenate(segments) if len(segments) > 1 else segments[0]
    sf.write(str(output), audio, 24000, subtype="PCM_16")

    duration = wav_duration(output)
    if duration <= 0.05:
        raise RuntimeError("Kokoro produced an empty or invalid WAV file.")
    return duration


def _generate_local_hinglish_subprocess(text, output):
    """Legacy compatibility backend retained for older tests only.

    The current application never selects this backend; public generation is
    English-only and uses Kokoro.
    """
    if not (HINGLISH_TTS_REPO and HINGLISH_REF_AUDIO and HINGLISH_REF_TEXT_PATH):
        raise RuntimeError(
            "HINGLISH_TTS_REPO, HINGLISH_REF_AUDIO and HINGLISH_REF_TEXT "
            "must all be set for the legacy test backend."
        )

    repo = Path(HINGLISH_TTS_REPO).expanduser().resolve()
    ref_audio = Path(HINGLISH_REF_AUDIO).expanduser()
    ref_text_path = Path(HINGLISH_REF_TEXT_PATH).expanduser()
    output = Path(output)

    if not repo.exists() or not repo.is_dir():
        raise RuntimeError(f"HINGLISH_TTS_REPO does not exist: {repo}")
    inference = repo / "inference.py"
    if not inference.exists():
        raise RuntimeError(f"hinglish-tts inference.py not found: {inference}")
    if not ref_audio.exists():
        raise RuntimeError(f"HINGLISH_REF_AUDIO does not exist: {ref_audio}")
    if not ref_text_path.exists():
        raise RuntimeError(f"HINGLISH_REF_TEXT does not exist: {ref_text_path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    ref_text = ref_text_path.read_text(encoding="utf-8").strip()
    result = subprocess.run(
        [HINGLISH_PYTHON, str(inference), str(text or "").strip(), "--ref-audio", str(ref_audio), "--ref-text", ref_text, "--out", str(output)],
        cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=HINGLISH_TTS_TIMEOUT, text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"hinglish-tts synthesis failed: {detail[-1200:]}")
    duration = wav_duration(output)
    if duration <= 0.05:
        raise RuntimeError("hinglish-tts produced an empty or invalid WAV file")
    return duration


def _generate_windows_sapi(text, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    safe_text = str(text).replace("'", "''")
    safe_output = str(output).replace("'", "''")
    ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.Volume = 100
$synth.SetOutputToWaveFile('{safe_output}')
$synth.Speak('{safe_text}')
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
        raise RuntimeError(result.stderr.strip() or "Windows SAPI narration failed")
    duration = wav_duration(output)
    if duration <= 0.05:
        raise RuntimeError("SAPI produced an empty or invalid WAV file.")
    return duration


def _synthesis_text(step):
    return str(step.get("tts_narration") or step.get("narration") or step.get("caption") or step.get("title") or "").strip()


def _generate_one(text, wav, voice=None):
    if TTS_PROVIDER == "sapi":
        return _generate_windows_sapi(text, wav)
    # Kokoro is the default and the intended local English TTS backend.
    return _generate_kokoro_speech(text, wav, language="en-us", voice="af_heart")


def generate_narration(plan, job_dir, progress_callback=None, language="en-us", voice="af_heart"):
    job_dir = Path(job_dir)
    out = job_dir / "audio"
    out.mkdir(parents=True, exist_ok=True)
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    if not steps:
        return []

    result = []
    total = len(steps)
    for i, step in enumerate(steps):
        text = _synthesis_text(step)
        wav = out / f"step_{i:03d}.wav"
        if progress_callback:
            progress_callback(i, total, f"Generating English narration {i + 1} of {total}")
        try:
            duration = _generate_one(text, wav, voice="af_heart")
        except Exception as exc:
            raise RuntimeError(f"English TTS failed for step {i + 1}: {exc}") from exc

        result.append({
            "index": i,
            "text": text,
            "caption": text,
            "path": str(wav),
            "duration": duration,
            "provider": TTS_PROVIDER,
            "language": "en-us",
            "voice": "af_heart",
        })

    if progress_callback:
        progress_callback(total, total, "English narration complete")
    return result
