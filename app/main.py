from pathlib import Path
import asyncio
import json
import traceback
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")


from app.services.pdf_service import process_pdf
from app.services.ai_service import build_tutorial_plan
from app.services.tts_service import generate_narration
from app.services.video_service import render_all_sections
from app.services import portal_store as portal


JOBS = BASE / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

jobs = {}

pool = ThreadPoolExecutor(max_workers=2)

app = FastAPI(title="Teamcenter AI Studio v2")

app.mount(
    "/static",
    StaticFiles(directory=BASE / "app" / "static"),
    name="static",
)

app.mount(
    "/media",
    StaticFiles(directory=JOBS),
    name="media",
)

templates = Jinja2Templates(
    directory=str(BASE / "app" / "templates")
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "Teamcenter AI Studio v2",
    }


@app.post("/generate")
async def generate(file: UploadFile = File(...)):
    name = file.filename or ""

    if not name.lower().endswith(".pdf"):
        return JSONResponse(
            {"error": "Please upload a PDF."},
            status_code=400,
        )

    jid = uuid4().hex[:10]

    job = JOBS / jid
    job.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = job / "source.pdf"
    pdf.write_bytes(await file.read())

    jobs[jid] = {
        "status": "queued",
        "progress": 2,
        "message": "PDF uploaded",
    }

    asyncio.get_running_loop().run_in_executor(
        pool,
        run_job,
        jid,
        pdf,
    )

    return {
        "job_id": jid,
    }


@app.get("/status/{jid}")
async def status(jid: str):
    if jid not in jobs:
        return JSONResponse(
            {"error": "Job not found"},
            status_code=404,
        )

    return jobs[jid]


@app.get("/plan/{jid}")
async def get_plan(jid: str):
    plan_path = JOBS / jid / "plan.json"
    if not plan_path.exists():
        return JSONResponse(
            {"error": "Plan not ready yet"},
            status_code=404,
        )
    return json.loads(plan_path.read_text(encoding="utf-8"))


# ===================================================================
# Trainer / trainee portal
#
# Deliberately lightweight, matching the rest of this prototype: no user
# accounts, no database. "Trainer" is just whoever has the /trainer/library
# link; publishing and prerequisites are stored in a small JSON file next
# to jobs/ (see app/services/portal_store.py). Trainees are identified only
# by an anonymous per-browser cookie so per-person watch progress (and
# therefore prerequisite locking) can be tracked without a login flow.
# ===================================================================

TRAINEE_COOKIE = "tc_trainee_id"


def _get_or_new_trainee_id(request: Request):
    """Returns (trainee_id, is_new). Caller sets the cookie on the response
    only when is_new, so the id stays stable across visits."""
    tid = request.cookies.get(TRAINEE_COOKIE)
    if tid:
        return tid, False
    return uuid4().hex, True


def _set_trainee_cookie(response: Response, trainee_id: str) -> None:
    response.set_cookie(
        TRAINEE_COOKIE, trainee_id,
        max_age=60 * 60 * 24 * 365, httponly=True, samesite="lax",
    )


@app.get("/trainer/library", response_class=HTMLResponse)
async def trainer_library(request: Request):
    tutorials = portal.list_all_tutorials()
    lib = portal.load_library()
    videos = []
    for tutorial in tutorials:
        vid = portal.video_id(tutorial["jid"], 0)
        videos.append({
            "vid": vid,
            "tutorial_title": tutorial["title"],
            "section_title": "Complete Tutorial",
            "video_url": tutorial.get("video_url"),
            "published": portal.is_published(vid, lib),
            "prerequisites": portal.get_prerequisites(vid, lib),
            "action_count": tutorial.get("action_count", 0),
        })
    return templates.TemplateResponse(
        request=request,
        name="trainer_library.html",
        context={"videos": videos, "all_videos": videos},
    )


@app.post("/trainer/publish/{jid}/{section_id}")
async def trainer_publish(jid: str, section_id: int, request: Request):
    form = await request.form()
    vid = portal.video_id(jid, 0)
    portal.set_published(vid, form.get("published") == "true")
    return RedirectResponse("/trainer/library", status_code=303)


@app.post("/trainer/prerequisites/{jid}/{section_id}")
async def trainer_set_prerequisites(jid: str, section_id: int, request: Request):
    form = await request.form()
    vid = portal.video_id(jid, 0)
    prereq_ids = form.getlist("prerequisite_ids")
    portal.set_prerequisites(vid, prereq_ids)
    return RedirectResponse("/trainer/library", status_code=303)


@app.get("/trainee", response_class=HTMLResponse)
async def trainee_dashboard(request: Request):
    trainee_id, is_new = _get_or_new_trainee_id(request)
    lib = portal.load_library()
    progress = portal.load_progress()
    videos = []
    for tutorial in portal.list_all_tutorials():
        vid = portal.video_id(tutorial["jid"], 0)
        if not portal.is_published(vid, lib):
            continue
        # Prerequisites are advisory only. They are deliberately NOT used to
        # set locked=true and never prevent the trainee from opening a video.
        missing = portal.get_missing_prerequisites(trainee_id, vid, lib, progress)
        videos.append({
            "vid": vid,
            "jid": tutorial["jid"],
            "section_id": 0,
            "tutorial_title": tutorial["title"],
            "section_title": "Complete Tutorial",
            "locked": False,
            "missing_count": len(missing),
            "watched": portal.is_watched(trainee_id, vid, progress),
            "action_count": tutorial.get("action_count", 0),
        })
    response = templates.TemplateResponse(
        request=request,
        name="trainee_dashboard.html",
        context={"videos": videos},
    )
    if is_new:
        _set_trainee_cookie(response, trainee_id)
    return response


@app.get("/trainee/videos/{jid}/{section_id}", response_class=HTMLResponse)
async def trainee_video_detail(jid: str, section_id: int, request: Request):
    trainee_id, is_new = _get_or_new_trainee_id(request)
    vid = portal.video_id(jid, 0)
    lib = portal.load_library()
    progress = portal.load_progress()
    entry = portal.get_video_entry(vid)
    if not entry or not portal.is_published(vid, lib):
        return RedirectResponse("/trainee", status_code=303)

    prereq_entries = []
    for pid in portal.get_prerequisites(vid, lib):
        p_entry = portal.get_video_entry(pid)
        if p_entry:
            prereq_entries.append({
                **p_entry,
                "watched": portal.is_watched(trainee_id, pid, progress),
            })

    response = templates.TemplateResponse(
        request=request,
        name="trainee_video.html",
        context={
            "video": entry,
            "locked": False,
            "prerequisites": prereq_entries,
            "already_watched": portal.is_watched(trainee_id, vid, progress),
        },
    )
    if is_new:
        _set_trainee_cookie(response, trainee_id)
    return response


@app.post("/trainee/videos/{jid}/{section_id}/watch")
async def trainee_mark_watched(jid: str, section_id: int, request: Request):
    trainee_id, _ = _get_or_new_trainee_id(request)
    vid = portal.video_id(jid, 0)
    portal.mark_watched(trainee_id, vid)
    return {"ok": True}


@app.get("/trainee/videos/{jid}/{section_id}/stream")
async def trainee_stream_video(jid: str, section_id: int, request: Request):
    """Serve the single continuous tutorial. Prerequisites are advisory and
    therefore this endpoint intentionally does not enforce a prerequisite lock."""
    vid = portal.video_id(jid, 0)
    entry = portal.get_video_entry(vid)
    if not entry or not portal.is_published(vid):
        return JSONResponse({"error": "not found"}, status_code=404)
    filename = entry["video_url"].rsplit("/", 1)[-1]
    file_path = JOBS / jid / filename
    if not file_path.exists():
        return JSONResponse({"error": "video file missing on server"}, status_code=404)
    return FileResponse(file_path, media_type="video/mp4")

def run_job(jid, pdf):
    try:
        job = Path(pdf).parent

        jobs[jid].update(
            status="processing",
            progress=8,
            message=(
                "Reading the document and extracting "
                "embedded screenshots"
            ),
        )

        print(f"[JOB] {jid} PDF={pdf}")
        print(f"[JOB] {jid} JOB_DIR={job}")

        # ---------------------------------------------------------
        # 1. PDF
        # ---------------------------------------------------------

        data = process_pdf(
            pdf,
            job,
        )

        pages = data.get("pages", [])
        screenshots = data.get("screenshots", [])

        jobs[jid].update(
            progress=28,
            message=(
                f"Extracted {len(pages)} pages and "
                f"{len(screenshots)} embedded visual regions"
            ),
        )

        # ---------------------------------------------------------
        # 2. AI PLAN
        # ---------------------------------------------------------

        def plan_progress(
            progress,
            message="Building tutorial plan",
        ):
            """
            Planning progress is expected to be 0-100.

            build_tutorial_plan is allowed to report either:
                callback(progress)
            or:
                callback(progress, message)

            The default message makes the interface tolerant.
            """

            try:
                p = int(progress)
            except Exception:
                p = 0

            p = max(0, min(100, p))

            mapped = 28 + int(
                22 * (p / 100.0)
            )

            jobs[jid].update(
                progress=mapped,
                message=str(
                    message or "Building tutorial plan"
                ),
            )

        plan = build_tutorial_plan(
            data,
            narration_language="en-us",
            progress_callback=plan_progress,
        )

        (job / "plan.json").write_text(
            json.dumps(
                plan,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        jobs[jid].update(
            progress=50,
            message="Generating narration",
        )

        # ---------------------------------------------------------
        # 3. TTS
        # ---------------------------------------------------------

        def tts_progress(
            current,
            total,
            message="Generating narration",
        ):
            try:
                current = int(current)
                total = int(total)
            except Exception:
                current = 0
                total = 0

            if total > 0:
                fraction = min(
                    1.0,
                    max(
                        0.0,
                        current / float(total),
                    ),
                )
                progress = 50 + int(
                    20 * fraction
                )
            else:
                progress = 50

            jobs[jid].update(
                progress=min(70, progress),
                message=str(
                    message or "Generating narration"
                ),
            )

        audio = generate_narration(
            plan,
            job,
            progress_callback=tts_progress,
            language="en-us",
            voice="af_heart",
        )

        (job / "audio.json").write_text(
            json.dumps(
                audio,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        jobs[jid].update(
            progress=75,
            message=(
                "Rendering screenshots, cursor guidance, "
                "captions and narration"
            ),
        )

        # ---------------------------------------------------------
        # 4. VIDEO -- ONE continuous action-driven tutorial
        # ---------------------------------------------------------

        render_all_sections(plan, audio, job)

        if not plan.get("video_file"):
            raise RuntimeError(
                "No usable tutorial video was produced -- check that the PDF "
                "contains recognizable action steps or meaningful page content."
            )

        plan["video_url"] = f"/media/{jid}/{plan['video_file']}"
        plan["captions_url"] = (
            f"/media/{jid}/{plan['captions_file']}"
            if plan.get("captions_file") else None
        )

        (job / "plan.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        jobs[jid].update(
            status="completed",
            progress=100,
            message="Tutorial ready",
            plan_url=f"/plan/{jid}",
            video_url=plan.get("video_url"),
            title=plan.get("title", "AI Learning Tutorial"),
            section_count=len(plan.get("sections", [])),
            action_count=plan.get("action_count", 0),
        )

        print(
            f"[VIDEO] {jid} generated successfully: "
            f"one continuous tutorial with {plan.get('action_count', 0)} actions in {job}"
        )

    except Exception as exc:
        traceback.print_exc()

        jobs[jid].update(
            status="error",
            progress=0,
            message=str(exc),
            error=str(exc),
        )