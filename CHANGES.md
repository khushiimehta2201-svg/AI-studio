# Changes

## Kokoro + UI language switching

- Added Kokoro-82M as an explicit local `TTS_PROVIDER=kokoro` backend.
- Added PCM-16 WAV output and duration validation.
- Added per-language lazy Kokoro pipeline caching.
- Added web UI dropdowns for narration language and matching Kokoro voice.
- Supported UI language choices: Hinglish, Hindi, English US/UK, Spanish, French, Italian, Japanese, Brazilian Portuguese, and Mandarin Chinese.
- Passed the selected narration language into the local Ollama narration prompt so the generated narration and TTS text match the selected language.
- Azure remains the production backend and `auto` never silently selects Kokoro.
- Added Kokoro setup, espeak-ng, language/voice, licensing, and production/demo guidance to README and `.env.example`.

# Changes

## Current implementation

### Document generalization
- Removed document-specific procedure steps and cursor coordinates.
- Procedure extraction now uses page-aware numbered-list parsing first, then local Ollama text extraction, then local Ollama vision for image-only/scanned PDFs.
- Step objects retain `source_page` provenance.
- Embedded PDF screenshots retain page and bounding-box metadata.
- PDF pages are rendered to images and retained as a visual fallback/source for scanned PDFs.

### Visual grounding
- Existing local Ollama vision model remains the visual backend (`qwen2.5vl:7b` by default).
- Vision results are accepted only for known step numbers, valid normalized coordinates, matching source pages, and confidence at or above `OLLAMA_VISION_MIN_CONFIDENCE`.
- Low-confidence or malformed vision results produce no cursor instead of inventing a coordinate.
- If vision cannot ground a step, screenshot fallback stays on the step's source page; no cross-page assignment is made.

### Narration
- Existing local Ollama text model remains the narration backend (`llama3.2` by default).
- Narration is requested as natural Hinglish in Roman script plus a Devanagari TTS version while preserving English UI/technical labels.
- No document-specific narration fallback remains.

### TTS
- Azure Speech REST TTS is available through `TTS_PROVIDER=azure`.
- Windows SAPI remains available through `TTS_PROVIDER=sapi` for compatibility.
- Azure TTS failures are fatal when Azure is selected so a broken audio track cannot silently produce a misleading tutorial.

### Reliability
- Ollama text calls have configurable retry support (`OLLAMA_RETRIES`).
- Core plan parsing, page-aware fallback mapping, and coordinate validation have unit tests.

## Deliberately unchanged
- Local Ollama text and vision model calls were not replaced with cloud LLMs.
- `video_service.py` remains the existing renderer because it already consumes the generalized plan format.
- Production/QOL work such as richer transitions, zooming, highlighting, authentication, persistence, and advanced job management is out of scope for this pass.


## Local Hinglish TTS backend

- Added optional `TTS_PROVIDER=local_hinglish` using an isolated `hinglish-tts` subprocess.
- Added `HINGLISH_TTS_REPO`, `HINGLISH_REF_AUDIO`, `HINGLISH_REF_TEXT`, `HINGLISH_PYTHON`, and timeout configuration.
- Local IndicF5 failures are fatal when explicitly selected, matching Azure's no-silent-degradation behavior.
- Added `python-dotenv` and project-local `.env` loading before service imports so the documented environment-file workflow works.
- Added explicit demo-only/licensing guidance for the IndicF5/hinglish-tts backend.
- Kept the existing narration and video pipeline unchanged upstream/downstream.
