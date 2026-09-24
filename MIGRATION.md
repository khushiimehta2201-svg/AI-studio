# Migration / installation — `new_frontend`

## What this bundle is

This is the definitive application build produced from the audited `new_frontend` baseline.
The existing frontend navigation model is retained (Generator / Trainee / Trainer) while the backend and portal contracts are rebuilt around schema version 4.

## Replace the old application

1. Back up the current `new_frontend` working tree and any existing `jobs/` data that must be retained.
2. Replace the project's `app/` directory with the `app/` directory in this bundle.
3. Replace `requirements.txt` and `.env.example`.
4. Do not copy the bundle's test-generated `jobs/` or `studio.sqlite3` state into production.
5. Create an empty `jobs/` directory; the application creates it automatically if absent.
6. Create `.env` from `.env.example` and set real credentials/secrets.

## Current local AI configuration

The tested current-model configuration is:

```text
OLLAMA_HOST=http://127.0.0.1:11435
AI_TEXT_MODEL=llama3.2
AI_VISION_MODEL=qwen2.5vl:7b
AI_NUM_CTX=4096
```

Changing to a newer Ollama text/vision model is configuration-only as long as the model supports the configured JSON/image interface.

## TTS

For local demonstration / local-only deployment:

```text
TTS_PROVIDER=kokoro
KOKORO_LANGUAGE=a
KOKORO_VOICE=af_heart
```

For an approved client external-voice deployment, configure Azure Speech explicitly. Do not put client API keys into source control.

## Security

For any client deployment:

```text
AUTH_REQUIRED=true
AUTH_SECRET=<long-random-secret>
COOKIE_SECURE=true
```

Set real trainer and trainee credentials only for a controlled prototype. For enterprise delivery, replace the bundled environment-backed login with the client's SSO/AD/LDAP integration.

The generic jobs directory is not statically exposed. Trainee media endpoints expose only final tutorial artifacts; trainer preview endpoints expose selected screenshots through authorization-checked routes.

## Validation before use

Run from a clean environment:

```powershell
python -m pip install -r requirements.txt
python -m pip check
python -m compileall -q app tests
node --check app/static/portal.js
python -m unittest discover -s tests -v
```

Then run a hardware-in-the-loop acceptance set containing at least:

- numbered procedure PDF with screenshots
- unnumbered procedural prose
- repeated logo/header/footer
- ToC/index/list-of-tables pages
- scanned or image-only page
- highlighted action
- no-highlight action
- duplicate UI labels
- multiple screenshots for one topic
- a compound instruction
- a drag instruction
- an explanation-only block
- at least one deliberately unresolved target

## Production acceptance

The automated suite proves application integration and deterministic behavior. It cannot prove that a 7B vision model will correctly ground every arbitrary client UI. Final acceptance must therefore measure grounding precision/recall, false cursor rate, generation latency, VRAM/RAM behavior, TTS quality, and recovery behavior on the client's representative corpus.
