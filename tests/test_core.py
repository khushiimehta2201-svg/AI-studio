from app.services.ai_service import (
    _extract_numbered_steps_from_pages,
    _fallback_visual_map,
    _target_center,
)


def test_numbered_procedure_ignores_section_heading():
    pages = [{
        "page": 1,
        "text": "1. Item\nAn Item description.\nCreate Item\n1. Select Folder\n2. Click New\n3. Click Add",
    }]
    steps = _extract_numbered_steps_from_pages(pages)
    assert [s["title"] for s in steps] == ["Select Folder", "Click New", "Click Add"]
    assert all(s["source_page"] == 1 for s in steps)


def test_fallback_visuals_stay_on_source_page():
    steps = [
        {"number": 1, "title": "A", "source_page": 1},
        {"number": 2, "title": "B", "source_page": 1},
        {"number": 3, "title": "C", "source_page": 1},
        {"number": 4, "title": "D", "source_page": 1},
        {"number": 5, "title": "E", "source_page": 1},
    ]
    screenshots = [
        {"index": 0, "page": 1, "bbox": [0, 0, 100, 100], "path": "a"},
        {"index": 1, "page": 1, "bbox": [0, 200, 100, 300], "path": "b"},
        {"index": 2, "page": 2, "bbox": [0, 0, 100, 100], "path": "c"},
    ]
    result = _fallback_visual_map(steps, screenshots)
    assert result[1]["screenshot"]["page"] == 1
    assert result[5]["screenshot"]["page"] == 1
    assert result[1]["cursor"] is None
    assert result[5]["cursor"] is None


def test_invalid_target_does_not_become_center():
    assert _target_center({"approximate_position": {}}) is None
    assert _target_center({"approximate_position": {"x": "bad", "y": 0.5}}) is None
    assert _target_center({"approximate_position": {"x": 0.5, "y": 0.5, "width": 0.1, "height": 0.1}}) == (0.55, 0.55)


def test_local_hinglish_subprocess_backend(tmp_path, monkeypatch):
    import sys
    import wave
    from app.services import tts_service

    repo = tmp_path / "hinglish-tts"
    repo.mkdir()
    ref_audio = tmp_path / "ref.wav"
    ref_text = tmp_path / "ref.txt"
    output = tmp_path / "out.wav"

    # Minimal fake CLI: proves the integration passes the expected arguments
    # and that the app accepts a valid PCM WAV result.
    (repo / "inference.py").write_text(
        """
import sys, wave
args = sys.argv[1:]
out = args[args.index('--out') + 1]
with wave.open(out, 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(24000)
    w.writeframes(b'\\x00\\x00' * 24000)
""",
        encoding="utf-8",
    )
    ref_audio.write_bytes(b"placeholder")
    ref_text.write_text("नमस्ते", encoding="utf-8")

    monkeypatch.setattr(tts_service, "HINGLISH_TTS_REPO", str(repo))
    monkeypatch.setattr(tts_service, "HINGLISH_REF_AUDIO", str(ref_audio))
    monkeypatch.setattr(tts_service, "HINGLISH_REF_TEXT_PATH", str(ref_text))
    monkeypatch.setattr(tts_service, "HINGLISH_PYTHON", sys.executable)

    duration = tts_service._generate_local_hinglish_subprocess(
        "Ab New button par click kijiye.",
        output,
    )

    assert output.exists()
    assert 0.9 < duration < 1.1


def test_local_hinglish_failure_is_fatal(monkeypatch, tmp_path):
    from app.services import tts_service

    monkeypatch.setattr(tts_service, "TTS_PROVIDER", "local_hinglish")
    monkeypatch.setattr(
        tts_service,
        "_generate_one",
        lambda text, wav: (_ for _ in ()).throw(RuntimeError("demo failure")),
    )

    try:
        tts_service.generate_narration(
            {"steps": [{"tts_narration": "Ab click kijiye."}]},
            tmp_path,
        )
    except RuntimeError as exc:
        assert "TTS failed for step 1" in str(exc)
    else:
        raise AssertionError("local_hinglish failure must be fatal")


def test_kokoro_language_config_and_voice_validation():
    from app.services import tts_service
    key, config = tts_service._kokoro_config("en-us")
    assert key == "en-us"
    assert config["lang_code"] == "a"
    assert "af_heart" in tts_service.KOKORO_VOICES["a"]
    key, config = tts_service._kokoro_config("hinglish")
    assert config["lang_code"] == "h"
    assert "hf_alpha" in tts_service.KOKORO_VOICES["h"]


def test_kokoro_pcm16_output_validation(monkeypatch, tmp_path):
    import numpy as np
    import wave
    from app.services import tts_service

    class FakePipeline:
        def __call__(self, text, voice):
            assert voice == "hf_alpha"
            yield ("", "", np.zeros(24000, dtype=np.float32))

    monkeypatch.setattr(tts_service, "_get_kokoro_pipeline", lambda language: (FakePipeline(), tts_service.KOKORO_LANGUAGE_CONFIG["hinglish"]))
    out = tmp_path / "kokoro.wav"
    duration = tts_service._generate_kokoro_speech("test", out, language="hinglish", voice="hf_alpha")
    assert 0.9 < duration < 1.1
    with wave.open(str(out), "rb") as w:
        assert w.getsampwidth() == 2
        assert w.getframerate() == 24000
