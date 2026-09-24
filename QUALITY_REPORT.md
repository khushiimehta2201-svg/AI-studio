# Quality report — definitive build

## Automated checks completed

- Python bytecode compilation: PASS
- JavaScript syntax check (`node --check`): PASS
- End-to-end `unittest` suite: PASS — 30 tests (24 core + 6 release-hardening)
- FastAPI TestClient page smoke: PASS
- Live Uvicorn startup + `/health` + main portal pages: PASS
- Real PDF ingestion -> planner -> render orchestration: PASS (AI/TTS calls mocked)
- Final rendered MP4 + VTT validation: PASS
- Trainer target review -> rerender -> QA -> publish gate: PASS
- Role boundary/auth test: PASS
- Secure media boundary test: PASS
- Scanned page OCR fallback test: PASS
- Repeated logo exclusion test: PASS
- Numbered explanatory prose containing `click` test: PASS
- Duplicate target ambiguity test: PASS
- Highlight-only review path test: PASS
- Vision target verification test: PASS
- Drag grounding and drag rendering tests: PASS
- Model swap configuration test: PASS

## Known environment note

`pip check` in the shared execution environment reports an unrelated environment-level conflict: the preinstalled `moviepy` package requests Pillow < 12 while the shared environment currently has Pillow 12.3.0. This project pins Pillow 11.3.0 in `requirements.txt`, so a clean installation of this project's requirements is the intended dependency environment. The project itself does not import moviepy.

## Hardware-in-the-loop requirement

The automated suite deliberately mocks external/local AI and TTS calls for deterministic CI-style testing. Before client deployment, run representative PDFs against the actual configured Llama/Qwen models, OCR stack and TTS provider on the deployment hardware.

The supplied Ollama logs show real Qwen vision latency, VRAM pressure and model eviction/reload behavior; these cannot be fully validated without the same model/runtime/hardware being available during the test.

## Non-goals still requiring client integration

- Enterprise SSO/AD/LDAP provider integration
- Multi-instance distributed job queue
- Object-store/NAS policy integration and retention policies
- Production observability/metrics backend
- Formal security penetration test
- Model-specific acceptance benchmark across the client's representative PDF corpus
- Human acceptance of voice quality/language pronunciation


## Second integration audit

A second audit of the packaged release found and corrected six additional release-level issues: unresolved target jobs now remain renderable for trainer review instead of aborting; trainer "no target" remains explicitly unresolved; prerequisite URLs are restricted to HTTP(S); full-page UI fallbacks are classified for targeting and printed instruction text is excluded from OCR target selection; short but valid action narration is accepted; and the Tesseract Python binding is included in project requirements. The upload validator now closes its file handle cleanly.
