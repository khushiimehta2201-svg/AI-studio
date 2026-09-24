# Definitive AI Studio design and implementation plan

## Objective

Convert arbitrary software-training PDFs into one coherent, source-grounded tutorial without assuming that the PDF contains numbering, arrows, highlights, or clean screenshots.

## Design principles

1. **Evidence is not a scene.** Screenshots are evidence objects. Scenes are teaching events.
2. **Source provenance is immutable.** Every scene retains source page and, when available, source step number and source order.
3. **Deterministic first, AI second.** PDF geometry, OCR, repeated-image detection and local heuristics do the cheap work first.
4. **No fabricated cursor.** A cursor exists only for a verified target. Ambiguous/highlight-only/vision-only cases enter review.
5. **Model independence.** Planner/targeting/rendering depend on provider-neutral interfaces, not model names.
6. **One timeline.** Video, audio, captions and trainee action state derive from the same scene timeline.
7. **Generation is not publication.** QA and trainer review are required before publication.
8. **Fail closed for confidentiality.** Source/intermediate artifacts are not exposed through a generic static media mount.

## Processing stages

### 1. Ingestion

- Validate PDF extension and size.
- Stream upload directly to disk.
- Extract text, word boxes, links and page geometry.
- Render full-page fallback images.
- Extract embedded visual assets.
- OCR only when embedded page text is insufficient.
- Detect repeated visuals and author annotation boxes.

### 2. Document understanding

- Identify ToC/index/list/metadata pages.
- Identify likely document furniture such as repeated logos.
- Parse numbered, bulleted and unnumbered instructions.
- Preserve source numbering separately from scene numbering.
- Keep wrapped instruction lines together.
- Separate explanatory blocks from actions.
- Split compound imperatives into atomic actions.
- Fall back to the configured text model for semantic extraction when deterministic extraction is insufficient.

### 3. Evidence association

For each action:

- rank nearby visual assets
- use OCR similarity
- use visual role
- use action/page geometry
- use nearby PDF text
- retain multiple candidates when evidence is close

### 4. Target grounding

For target-requiring actions:

- derive likely target queries
- search OCR candidates
- require confidence + ambiguity margin
- reject document instruction text on full-page fallbacks
- use author highlights only as hints
- call the configured vision model only for unresolved candidates
- request multiple candidates from vision
- reject low-confidence/duplicate candidates
- run a second visual verification pass for vision-only results
- persist target evidence and method

Keyboard actions such as `Press Enter` are explicitly non-mouse interactions.

Drag actions retain both source and destination target points.

### 5. Narration

- Build narration from action + relevant supporting context, not the whole page.
- Batch narration requests within a bounded prompt size.
- Keep language configurable.
- Strip URLs from speech.
- Validate output length and punctuation.
- Fall back to source-grounded wording when the model does not return valid structured output.

### 6. TTS

Provider interface supports:

- Kokoro
- SAPI
- Azure Speech

Kokoro pipeline objects are cached per `(language, voice)` to avoid reloading the neural model for every scene.

Explicit production providers fail loudly instead of silently changing voice backends unless `AUTO_LOCAL_FALLBACK=true` is deliberately configured.

### 7. Scene compilation

Each scene contains:

- stable `scene_id`
- source page/order/step
- action type
- target state
- cursor point/path
- screenshot evidence
- narration/caption text
- interaction plan
- URLs as metadata

### 8. Rendering

- All scenes render into one continuous tutorial.
- Cursor motion is eased and starts from the previous cursor when the screenshot remains unchanged.
- Click/double-click/right-click have dedicated pulses.
- Drag has a start-to-end motion interval.
- Scene duration is derived from narration plus explicit pre-roll/hold timing.
- Audio is padded per scene so cumulative A/V timing remains aligned.
- VTT is generated from the same canonical caption timeline.

### 9. QA

Pre-render QA checks:

- scene IDs/order
- provenance
- narration validity
- required targets
- verified cursor points
- drag paths
- review status
- numbering gaps

Post-render QA checks:

- MP4 exists
- video/audio streams exist
- VTT exists
- timeline is monotonic
- plan and timeline scene IDs match exactly
- rendered duration agrees with timeline

### 10. Trainer review

Trainer can:

- accept a proposed target
- click an exact target point
- choose drag start/end points
- explicitly mark a scene unresolved

A required-target scene cannot be published with a missing target.

### 11. Portal

The trainee portal consumes the canonical timeline directly. It does not calculate action state by dividing video duration into equal buckets.

Trainer and trainee roles are separated when authentication is enabled.

## Model replacement strategy

Changing from `llama3.2` / `qwen2.5vl:7b` to newer Ollama models is configuration-only.

Adding a different provider requires a new adapter implementation with the same `TextModel` / `VisionModel` interface. No planner, targeting, renderer or portal rewrite should be necessary.

## Optimization strategy

- one generation worker by default to respect the observed local GPU memory pressure
- OCR cached in-process
- vision payloads resized before submission
- vision results cached per job/action/screenshot
- bounded narration batches
- full-page OCR only for sparse-text pages
- streaming PDF upload
- persistent job state
- rerender after trainer review instead of regenerating AI content

## Release sequence

1. Run unit/integration quality suite.
2. Run application startup/page/API smoke.
3. Configure current Ollama models and local TTS.
4. Run hardware-in-the-loop representative PDFs.
5. Tune OCR/vision confidence thresholds from measured false positives/negatives.
6. Benchmark generation latency and memory on the target deployment GPU.
7. Enable client authentication/SSO integration.
8. Publish only after trainer review and machine QA.
