from pathlib import Path
import asyncio
import json
import traceback
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")


from app.services.pdf_service import process_pdf
from app.services.ai_service import build_tutorial_plan
from app.services.tts_service import generate_narration
from app.services.video_service import render_all_sections


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
        # 4. VIDEO (one file per section, not one monolithic video)
        # ---------------------------------------------------------

        render_all_sections(plan, audio, job)

        rendered_sections = sum(
            1 for s in plan.get("sections", []) if s.get("video_file")
        )
        if rendered_sections == 0:
            raise RuntimeError(
                "No section produced a usable video -- check that the "
                "PDF contains recognizable action steps with screenshots."
            )

        # Resolve section media into URLs the frontend can hit directly,
        # and persist the final plan (now including video/caption filenames)
        # so /plan/{jid} can be re-fetched independently of job status.
        for section in plan.get("sections", []):
            section["video_url"] = (
                f"/media/{jid}/{section['video_file']}"
                if section.get("video_file") else None
            )
            section["captions_url"] = (
                f"/media/{jid}/{section['captions_file']}"
                if section.get("captions_file") else None
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
            title=plan.get("title", "AI Learning Tutorial"),
            section_count=len(plan.get("sections", [])),
        )

        print(
            f"[VIDEO] {jid} generated successfully: "
            f"{rendered_sections} section video(s) in {job}"
        )

    except Exception as exc:
        traceback.print_exc()

        jobs[jid].update(
            status="error",
            progress=0,
            message=str(exc),
            error=str(exc),
        )