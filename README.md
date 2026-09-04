# Teamcenter AI Studio v2

PDF -> procedure plan -> screenshot-grounded tutorial -> Hinglish narration -> MP4.

This version is designed for **multiple Teamcenter training PDFs**, not one hard-coded
`Create Item` document. It keeps the existing local Ollama model calls and adds
local vision grounding for screenshot/cursor mapping.

## Pipeline

1. **PDF extraction** — PyMuPDF extracts document text and large embedded screenshots.
   If the PDF has no usable embedded screenshots, PDF pages are rendered as visual
   references instead.
2. **Procedure extraction** — numbered procedures are parsed first. If the PDF has
   no usable numbered procedure, the existing local Ollama text model is asked to
   extract ordered actions.
3. **Narration generation** — the existing Ollama text model generates Hinglish
   narration. It returns both a readable Hinglish script and a Devanagari Hindi
   version for TTS while preserving English UI/technical terms.
4. **Visual grounding** — the existing Ollama vision model analyzes every extracted
   screenshot and attempts to map each procedure step to the visible UI target using
   normalized coordinates. If grounding is unavailable, the system still assigns
   screenshots generically but does **not** invent cursor coordinates.
5. **TTS** — Azure Speech is the production backend for Hinglish narration. Windows
   SAPI remains available as a legacy development backend.
6. **Video** — the existing renderer freezes each grounded screenshot for the exact
   narration duration and overlays the cursor when a visual target was found.

## Windows setup

```powershell
cd C:\path\to\teamcenter_ai_studio_v2
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Open `http://127.0.0.1:8001/`.

## Ollama

The application continues to use the local Ollama backends. No cloud LLM replacement
was made.

Text model:

```text
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2
```

Vision model:

```text
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_VISION_MODEL=qwen2.5vl:7b
OLLAMA_VISION_ENABLED=true
```

Optional timeout settings:

```text
OLLAMA_TIMEOUT=120
OLLAMA_VISION_TIMEOUT=180
```

## Azure Speech for Hinglish

Set these environment variables on the machine that performs generation:

```text
TTS_PROVIDER=azure
AZURE_SPEECH_KEY=<speech-resource-key>
AZURE_SPEECH_REGION=<speech-resource-region>
AZURE_SPEECH_VOICE=hi-IN-SwaraNeural
```

The code uses the Azure Speech REST endpoint and requests a PCM WAV directly, so
no extra Azure Python package is required. The voice is configured for Hindi and
receives Devanagari Hindi with English product/UI terms preserved.

For local legacy testing only:

```text
TTS_PROVIDER=sapi
```

Or use:

```text
TTS_PROVIDER=auto
```

which uses Azure when its credentials are present and otherwise falls back to SAPI.

## Important behavior changes

### Removed document-specific assumptions

The plan builder no longer contains:

- the seven hard-coded `Create Item` steps
- hard-coded screenshot indexes for those steps
- hard-coded cursor coordinates
- exact-title narration fallbacks for `Create Item`
- a fixed tutorial title

### Screenshot mapping

The vision model now receives the complete ordered procedure and each screenshot.
It returns step-to-UI matches with normalized coordinates and confidence. The
highest-confidence match for a step wins if multiple screenshots contain the same
UI element.

### Fallback behavior

If local vision is unavailable, the application does not fabricate coordinates. It
uses a generic screenshot distribution and leaves the cursor unset for those steps.
If the PDF has no procedure and Ollama cannot extract one, generation fails clearly
instead of silently generating the wrong document's procedure.

## Testing handoff

The current environment does not need to run the local LLM/vision models for the
code changes themselves. The person with the capable hardware should test:

1. A numbered Teamcenter PDF with screenshots.
2. A PDF whose procedure is not numbered, to exercise Ollama step extraction.
3. A PDF with several screenshots per page.
4. A PDF with no embedded raster screenshots, to exercise page rendering.
5. Hinglish narration with Azure Speech.
6. A step where the vision model cannot find a target, confirming that no fake
   cursor coordinate is produced.

The project intentionally keeps the local Ollama calls unchanged at the transport
and model level; only their prompts and the data passed to them were generalized.

## Current AI pipeline

The application is document-agnostic at the planning layer:

1. Extract PDF text page-by-page.
2. Extract embedded visual references with page provenance.
3. Render every PDF page as a fallback visual and as input for scanned/image-only documents.
4. Extract ordered procedure steps from text; fall back to the existing local Ollama model; then use local Ollama vision when the PDF has no usable text.
5. Ground steps to visible UI targets with the existing local Ollama vision model. Low-confidence matches do not receive a cursor.
6. Generate Roman-script Hinglish narration plus Devanagari TTS narration with the existing local Ollama text model.
7. Synthesize audio with Azure Speech, Kokoro, local IndicF5, or legacy SAPI depending on the explicit TTS provider.
8. The web UI lets the user select the narration language and matching Kokoro voice when Kokoro is selected.
8. Render the existing video pipeline from the resulting plan/audio artifacts.

The project intentionally keeps the existing local Ollama text and vision model calls. No cloud LLM replacement is required for the AI planning pipeline.


## Kokoro demo TTS and UI language switching

The showcase can use **Kokoro-82M** locally with no API key. Kokoro's official package is Apache-2.0 and supports language-specific pipelines including Hindi, American/British English, Spanish, French, Italian, Japanese, Brazilian Portuguese, and Mandarin Chinese. The UI exposes the narration language and a matching Kokoro voice selector.

Set the demo backend in `.env`:

```text
TTS_PROVIDER=kokoro
KOKORO_LANGUAGE=hinglish
KOKORO_VOICE=hf_alpha
```

Then start the app and choose **Narration language** and **Kokoro voice** directly in the web UI. The selected language is passed to the local Ollama narration prompt and to Kokoro. The backend keeps one Kokoro pipeline loaded per language so switching languages does not require reloading the model for every narration step.

### Supported UI languages

| UI option | Kokoro code | Example voices |
|---|---|---|
| Hinglish (Hindi + English) | `h` | `hf_alpha`, `hf_beta` |
| Hindi | `h` | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi` |
| English (US) | `a` | `af_heart`, `af_sarah`, `am_adam` |
| English (UK) | `b` | `bf_emma`, `bm_daniel` |
| Spanish | `e` | `ef_dora`, `em_alex` |
| French | `f` | `ff_siwis` |
| Italian | `i` | `if_sara`, `im_nicola` |
| Japanese | `j` | `jf_alpha`, `jm_kumo` |
| Portuguese (Brazil) | `p` | `pf_dora`, `pm_alex` |
| Mandarin Chinese | `z` | `zf_xiaobei`, `zm_yunxi` |

Kokoro requires the selected voice to match the selected language family. The UI therefore refreshes the voice dropdown whenever the language changes.

### Kokoro installation

Install the Python dependencies from `requirements.txt`, then install **espeak-ng** at the OS level. On Windows, install it from the official espeak-ng release installer and ensure `espeak-ng.exe` is on `PATH`; verify with:

```powershell
espeak-ng --version
```

The Kokoro package documents Python 3.10 through 3.12 for its 0.9.4 release, so Python 3.11 is the recommended Windows environment for this project.

### Production vs showcase

- `TTS_PROVIDER=azure` remains the production/client-facing default.
- `TTS_PROVIDER=kokoro` is an explicit local showcase option.
- Kokoro is **not** added to `auto`, so a production configuration cannot silently switch to it.
- Existing `sapi` and `local_hinglish` backends remain available.

### Language note

Kokoro has dedicated language/voice families, but it is not specifically trained as a Hinglish code-switching model. Hinglish remains available as the Hindi Kokoro pipeline plus the AI narration style, but English UI terms may sound less natural than with a dedicated Hinglish TTS model. Test the actual Teamcenter narration lines before selecting it for a showcase.

## Optional local Hinglish TTS (IndicF5 / hinglish-tts)

The project can optionally use the local `hinglish-tts` wrapper around AI4Bharat IndicF5 for the POC. This backend accepts the same `tts_narration` already produced by the tutorial planner, so no upstream narration changes are required.

**Licensing gate:** treat this backend as **demo-only** until commercial terms are cleared. The `hinglish-tts` project states separate licensing conditions for its wrapper/evaluation assets, while IndicF5 model weights have their own attribution/commercial-use restrictions. Do not represent this backend as an approved production client deliverable until those terms have been confirmed.

### Recommended setup

Keep the TTS dependencies in a separate Python 3.10 or 3.11 virtual environment. This avoids mixing IndicF5's older/pinned ML dependencies with the FastAPI app environment. The app invokes `hinglish-tts/inference.py` as a subprocess.

Example environment variables:

```text
TTS_PROVIDER=local_hinglish
HINGLISH_TTS_REPO=C:\path\to\hinglish-tts
HINGLISH_REF_AUDIO=C:\path\to\hinglish-tts\data\reference_audio\tc_narrator.wav
HINGLISH_REF_TEXT=C:\path\to\hinglish-tts\data\reference_audio\tc_narrator.txt
HINGLISH_PYTHON=C:\path\to\hinglish-tts-venv\Scripts\python.exe
HINGLISH_TTS_TIMEOUT=120
HF_TOKEN=YOUR_HUGGINGFACE_READ_TOKEN
```

Before using the app, verify the external CLI directly with the commands from the implementation guide: accept the IndicF5 Hugging Face model access, set `HF_TOKEN`, verify the duration patch, and synthesize a short `test.wav`. Then run a 2--3 step PDF through the app and listen for correct Hinglish pronunciation, especially when English UI terms occur inside Hindi sentences.

The current integration intentionally uses a subprocess for isolation and simplicity. It reloads IndicF5 for each narration step, so it is not optimized for speed. If the POC proves the voice quality and the reload time becomes a problem, the next optimization is a small persistent local HTTP service that loads the model once and serves multiple synthesis requests.
