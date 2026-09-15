# What changed, and how to apply it

These files are drop-in replacements/additions for your existing repo — copy
them into place at the same paths (they overwrite the same relative
locations under `app/`). No new dependencies, no database: everything reuses
your existing FastAPI app, the `jobs/` output folder, and Jinja2 templates.

## Files

**Modified**
- `app/main.py` — added the trainer/trainee routes (see below). Nothing
  about `/`, `/generate`, `/status`, `/plan` changed.
- `app/services/ai_service.py` — one addition: each section now also
  carries `action_bullets` (the same short action-line text already
  extracted for each step's title, just grouped by section). This feeds the
  trainee portal's "actions performed" list — no new AI calls.
- `app/services/video_service.py` — the video renderer now:
  - Matches the letterbox padding color to the real screenshot's own border
    color, instead of a fixed dark gray, so it blends with Teamcenter's
    actual chrome instead of clashing with it.
  - Renders a light Teamcenter-style placeholder (app-colored background +
    blue ribbon reading "Teamcenter") for the rare step that has no
    screenshot at all, instead of a blank near-black frame.
- `app/templates/index.html` — added two nav links under the header:
  "Publish & manage videos" and "View trainee portal".

**New**
- `app/services/portal_store.py` — all the trainer/trainee business logic:
  publishing, prerequisites (with cycle detection so a trainer can't make A
  require B require A), and per-browser watch progress. Plain JSON files
  (`portal_library.json`, `portal_progress.json`) written next to `jobs/` at
  first use — no setup needed, no migration.
- `app/templates/trainer_library.html` — trainer's publish/prerequisite page.
- `app/templates/trainee_dashboard.html` — trainee's video grid.
- `app/templates/trainee_video.html` — trainee's video page: player,
  dialogue box (actions performed), prerequisites panel, lock overlay.

## How it works

A "video" the trainee sees is one **section** of a generated tutorial (your
pipeline already renders one MP4 per section) — addressed as `jid:section_id`.

- **`/trainer/library`** — lists every generated section-video across every
  job. Each has a Publish/Unpublish toggle and an "Edit prerequisites"
  expander (checkboxes for any other video). Only reachable from this page —
  trainees are never shown these controls, so prerequisites can only be set
  from here. (Note: like the rest of this prototype, there's no login, so
  this is enforced by not exposing the controls anywhere else, not by
  authentication — see "Known limits" below.)
- **`/trainee`** — grid of published videos only, with a lock icon,
  checkmark (watched), or play icon per video.
- **`/trainee/videos/{jid}/{section_id}`** — the video page. If any
  prerequisite is unwatched, the player is replaced with a lock message and
  the prerequisites list shows which ones are still pending, each linking
  straight to that video. Once unlocked, the actual video bytes are served
  through `/trainee/videos/{jid}/{section_id}/stream`, which re-checks the
  lock server-side on every request — so it can't be bypassed by guessing or
  bookmarking a URL, only by knowing your existing public `/media/...` path
  (see below).
- Watch progress is tracked via an anonymous cookie set on first visit to
  any `/trainee...` route — no login, no name entry. Marking watched happens
  automatically when the video finishes (with a 90%-watched fallback for
  clips with a trailing blank frame).

## Known limits (kept deliberately, to match "don't make this more complex")

- **No real authentication anywhere in the app** (this was already true
  before my changes — `/media/...` is a public static mount). The trainer
  page is just an unlisted-ish URL; anyone with the link could publish
  videos or edit prerequisites. The trainee gated `/stream` route stops
  casual bypassing via the trainee UI, but someone who already knows a raw
  `/media/{jid}/{file}.mp4` path can still fetch it directly, same as
  before my changes. If this needs to hold up outside a trusted internal
  team, it needs real accounts — happy to add that as a separate, scoped
  change if/when you want it.
- Watch progress is per-browser (cookie), not per-person — clearing cookies
  or switching browsers resets progress. Fine for a prototype; would need
  real accounts to fix properly.
- `portal_library.json` / `portal_progress.json` are plain files with no
  locking — fine for a single local instance, not for concurrent writers.

## Smoke test after copying the files in

1. Start the app as usual (`uvicorn app.main:app ...`).
2. Generate a tutorial from `/` like before.
3. Go to `/trainer/library`, publish one of its sections.
4. Go to `/trainee` — confirm it appears and plays.
5. Publish a second section and set the first as its prerequisite.
6. Confirm the second is locked on `/trainee` until you watch the first.
