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


# ============================================================
# BASE CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent.parent

load_dotenv(BASE / ".env")


# ============================================================
# EXISTING VIDEO GENERATION SERVICES
# ============================================================

from app.services.pdf_service import process_pdf
from app.services.ai_service import build_tutorial_plan
from app.services.tts_service import generate_narration
from app.services.video_service import render


# ============================================================
# DIRECTORIES
# ============================================================

JOBS = BASE / "jobs"
JOBS.mkdir(
    parents=True,
    exist_ok=True,
)

PORTAL_DATA = BASE / "portal_data.json"


# ============================================================
# JOB STATE
# ============================================================

jobs = {}

pool = ThreadPoolExecutor(
    max_workers=2
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Teamcenter AI Studio v2"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=BASE / "app" / "static"
    ),
    name="static",
)


app.mount(
    "/media",
    StaticFiles(
        directory=JOBS
    ),
    name="media",
)


# ============================================================
# TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory=str(
        BASE / "app" / "templates"
    )
)


# ============================================================
# PORTAL DATA HELPERS
# ============================================================

def load_portal_data():
    """
    Load portal-specific metadata.

    The video-generation pipeline does not depend on this file.

    Structure:

    {
        "published": {
            "job_id": true
        },
        "prerequisites": {
            "job_id": [
                "https://example.com/video1"
            ]
        }
    }
    """

    if not PORTAL_DATA.exists():
        return {
            "published": {},
            "prerequisites": {},
        }

    try:
        data = json.loads(
            PORTAL_DATA.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, dict):
            raise ValueError(
                "Invalid portal data"
            )

        data.setdefault(
            "published",
            {}
        )

        data.setdefault(
            "prerequisites",
            {}
        )

        return data

    except Exception:
        return {
            "published": {},
            "prerequisites": {},
        }


def save_portal_data(data):
    """
    Save trainer/portal metadata.

    This is intentionally separate from the video-generation
    files and pipeline.
    """

    PORTAL_DATA.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# DISCOVER GENERATED VIDEOS
# ============================================================

def get_generated_videos():
    """
    Discover generated tutorials from the jobs directory.

    A tutorial is considered available when:

        jobs/<job_id>/tutorial.mp4
        jobs/<job_id>/plan.json

    both exist.

    This means the trainee/trainer portal can continue to see
    generated tutorials even after restarting FastAPI.
    """

    videos = []

    if not JOBS.exists():
        return videos

    portal_data = load_portal_data()

    published = portal_data.get(
        "published",
        {}
    )

    prerequisites = portal_data.get(
        "prerequisites",
        {}
    )

    for job_dir in JOBS.iterdir():

        if not job_dir.is_dir():
            continue

        jid = job_dir.name

        video_file = (
            job_dir / "tutorial.mp4"
        )

        plan_file = (
            job_dir / "plan.json"
        )

        if not video_file.exists():
            continue

        if not plan_file.exists():
            continue

        try:
            plan = json.loads(
                plan_file.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            plan = {}

        if not isinstance(plan, dict):
            plan = {}

        title = plan.get(
            "title",
            "Teamcenter Tutorial",
        )

        description = plan.get(
            "description",
            "Interactive Teamcenter training tutorial.",
        )

        # By default, existing generated videos are visible.
        # Once a trainer explicitly changes publication status,
        # that value is used.
        is_published = bool(
            published.get(
                jid,
                True
            )
        )

        video = {
            "id": jid,
            "job_id": jid,
            "title": title,
            "description": description,
            "video_url": (
                f"/media/{jid}/tutorial.mp4"
            ),
            "published": is_published,
            "prerequisites": prerequisites.get(
                jid,
                [],
            ),
        }

        videos.append(video)

    # Newest job first
    videos.sort(
        key=lambda item: item["id"],
        reverse=True,
    )

    return videos


# ============================================================
# BRIEF ACTION EXTRACTION
# ============================================================

def extract_brief_actions(plan):
    """
    Extract short action/topic points for the trainee portal.

    IMPORTANT:
    This does NOT expose the complete detailed procedure.

    The trainee UI should show brief points describing what
    happens in the tutorial.
    """

    actions = []

    if not isinstance(plan, dict):
        return actions

    # --------------------------------------------------------
    # Sections
    # --------------------------------------------------------

    sections = plan.get(
        "sections",
        []
    )

    if isinstance(sections, list):

        for section in sections:

            if not isinstance(section, dict):
                continue

            title = (
                section.get("title")
                or section.get("name")
                or section.get("heading")
            )

            if title:
                actions.append(
                    str(title).strip()
                )

            section_actions = (
                section.get("actions")
                or section.get("steps")
                or section.get("action_items")
                or []
            )

            if isinstance(
                section_actions,
                list
            ):

                for action in section_actions:

                    if isinstance(
                        action,
                        dict
                    ):
                        text = (
                            action.get("caption")
                            or action.get("action")
                            or action.get("description")
                            or action.get("text")
                            or action.get("narration")
                        )

                    else:
                        text = str(action)

                    if text:
                        actions.append(
                            str(text).strip()
                        )

    # --------------------------------------------------------
    # Direct actions
    # --------------------------------------------------------

    direct_actions = plan.get(
        "actions",
        []
    )

    if isinstance(
        direct_actions,
        list
    ):

        for action in direct_actions:

            if isinstance(
                action,
                dict
            ):
                text = (
                    action.get("caption")
                    or action.get("action")
                    or action.get("description")
                    or action.get("text")
                    or action.get("narration")
                )

            else:
                text = str(action)

            if text:
                actions.append(
                    str(text).strip()
                )

    # Older generated plans store their tutorial actions in steps.
    if not actions:
        steps = plan.get("steps", [])

        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    text = (
                        step.get("narration")
                        or step.get("caption")
                        or step.get("title")
                        or step.get("content")
                    )
                else:
                    text = str(step)

                if text:
                    actions.append(str(text).strip())

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    cleaned = []

    seen = set()

    for item in actions:

        item = " ".join(
            item.split()
        )

        if len(item) > 150:
            item = item[:147].rsplit(" ", 1)[0] + "..."

        if not item:
            continue

        key = item.lower()

        if key in seen:
            continue

        seen.add(key)

        cleaned.append(item)

    # Keep the trainee UI concise.
    return cleaned[:8]


# ============================================================
# MAIN GENERATOR UI
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


# ============================================================
# TRAINEE PORTAL PAGE
# ============================================================

@app.get(
    "/trainee",
    response_class=HTMLResponse
)
async def trainee_portal(
    request: Request
):

    return templates.TemplateResponse(
        "trainee.html",
        {
            "request": request
        }
    )


# ============================================================
# TRAINEE WATCH PAGE
# ============================================================

@app.get(
    "/trainee/videos/{video_id}",
    response_class=HTMLResponse
)
async def trainee_watch_page(
    request: Request,
    video_id: str
):

    return templates.TemplateResponse(
        "watch.html",
        {
            "request": request,
            "video_id": video_id,
        }
    )


# ============================================================
# TRAINER PORTAL PAGE
# ============================================================

@app.get(
    "/trainer",
    response_class=HTMLResponse
)
async def trainer_portal(
    request: Request
):

    return templates.TemplateResponse(
        "trainer.html",
        {
            "request": request,
        }
    )


# ============================================================
# TRAINEE API
# ============================================================

@app.get(
    "/api/trainee/videos"
)
async def trainee_videos():

    videos = get_generated_videos()

    # Only published tutorials appear to trainees.
    videos = [
        video
        for video in videos
        if video.get(
            "published",
            True
        )
    ]

    return {
        "videos": videos
    }


@app.get(
    "/api/trainee/videos/{video_id}"
)
async def trainee_video_details(
    video_id: str
):

    videos = get_generated_videos()

    video = next(
        (
            item
            for item in videos
            if item["id"] == video_id
        ),
        None,
    )

    if video is None:

        return JSONResponse(
            {
                "error": "Tutorial not found"
            },
            status_code=404,
        )

    if not video.get(
        "published",
        True
    ):

        return JSONResponse(
            {
                "error": "Tutorial is not published"
            },
            status_code=404,
        )

    plan_file = (
        JOBS
        / video_id
        / "plan.json"
    )

    try:

        plan = json.loads(
            plan_file.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        plan = {}

    if not isinstance(
        plan,
        dict
    ):
        plan = {}

    actions = extract_brief_actions(
        plan
    )

    topic_intro = (
        plan.get(
            "description"
        )
        or "This tutorial provides a guided walkthrough of the selected Teamcenter topic."
    )

    return {
        **video,
        "actions": actions,
        "topic_intro": topic_intro,
    }


# ============================================================
# TRAINER API
# ============================================================

@app.get(
    "/api/trainer/videos"
)
async def trainer_videos():

    return {
        "videos": get_generated_videos()
    }


# ============================================================
# PUBLISH / UNPUBLISH TUTORIAL
# ============================================================

@app.post(
    "/api/trainer/publish/{video_id}"
)
async def trainer_publish(
    video_id: str
):

    videos = get_generated_videos()

    video = next(
        (
            item
            for item in videos
            if item["id"] == video_id
        ),
        None,
    )

    if video is None:

        return JSONResponse(
            {
                "error": "Tutorial not found"
            },
            status_code=404,
        )

    data = load_portal_data()

    current = bool(
        data
        .get("published", {})
        .get(
            video_id,
            True
        )
    )

    new_value = not current

    data.setdefault(
        "published",
        {}
    )[video_id] = new_value

    save_portal_data(data)

    return {
        "success": True,
        "published": new_value,
        "video_id": video_id,
    }


# ============================================================
# TRAINER PREREQUISITES
# ============================================================

@app.post(
    "/api/trainer/prerequisites/{video_id}"
)
async def trainer_prerequisites(
    video_id: str,
    request: Request,
):

    videos = get_generated_videos()

    video = next(
        (
            item
            for item in videos
            if item["id"] == video_id
        ),
        None,
    )

    if video is None:

        return JSONResponse(
            {
                "error": "Tutorial not found"
            },
            status_code=404,
        )

    try:

        body = await request.json()

    except Exception:

        body = {}

    urls = body.get(
        "prerequisites",
        body.get(
            "urls",
            []
        )
    )

    if isinstance(
        urls,
        str
    ):
        urls = [urls]

    if not isinstance(
        urls,
        list
    ):
        urls = []

    cleaned = []

    for url in urls:

        url = str(
            url
        ).strip()

        if not url:
            continue

        if url not in cleaned:
            cleaned.append(url)

    data = load_portal_data()

    data.setdefault(
        "prerequisites",
        {}
    )[video_id] = cleaned

    save_portal_data(data)

    return {
        "success": True,
        "video_id": video_id,
        "prerequisites": cleaned,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
async def health():

    return {
        "status": "ok",
        "service": "Teamcenter AI Studio v2",
    }


# ============================================================
# VIDEO GENERATION
# ============================================================

@app.post(
    "/generate"
)
async def generate(
    file: UploadFile = File(...)
):

    name = file.filename or ""

    if not name.lower().endswith(
        ".pdf"
    ):

        return JSONResponse(
            {
                "error": "Please upload a PDF."
            },
            status_code=400,
        )

    jid = uuid4().hex[:10]

    job = JOBS / jid

    job.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = job / "source.pdf"

    pdf.write_bytes(
        await file.read()
    )

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


# ============================================================
# GENERATION STATUS
# ============================================================

@app.get(
    "/status/{jid}"
)
async def status(
    jid: str
):

    if jid not in jobs:

        return JSONResponse(
            {
                "error": "Job not found"
            },
            status_code=404,
        )

    return jobs[jid]


# ============================================================
# EXISTING VIDEO GENERATION PIPELINE
# ============================================================

def run_job(
    jid,
    pdf
):

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

        print(
            f"[JOB] {jid} PDF={pdf}"
        )

        print(
            f"[JOB] {jid} JOB_DIR={job}"
        )

        # -----------------------------------------------------
        # 1. PDF PROCESSING
        # -----------------------------------------------------

        data = process_pdf(
            pdf,
            job,
        )

        pages = data.get(
            "pages",
            []
        )

        screenshots = data.get(
            "screenshots",
            []
        )

        jobs[jid].update(
            progress=28,
            message=(
                f"Extracted {len(pages)} pages and "
                f"{len(screenshots)} embedded visual regions"
            ),
        )

        # -----------------------------------------------------
        # 2. AI PLAN
        # -----------------------------------------------------

        def plan_progress(
            progress,
            message="Building tutorial plan",
        ):

            """
            Planning progress is expected to be 0-100.

            build_tutorial_plan may report:

                callback(progress)

            or:

                callback(progress, message)
            """

            try:

                p = int(
                    progress
                )

            except Exception:

                p = 0

            p = max(
                0,
                min(
                    100,
                    p
                )
            )

            mapped = (
                28
                + int(
                    22
                    * (
                        p
                        / 100.0
                    )
                )
            )

            jobs[jid].update(
                progress=mapped,
                message=str(
                    message
                    or "Building tutorial plan"
                ),
            )

        plan = build_tutorial_plan(
            data,
            narration_language="en-us",
            progress_callback=plan_progress,
        )

        (
            job / "plan.json"
        ).write_text(
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

        # -----------------------------------------------------
        # 3. TTS
        # -----------------------------------------------------

        def tts_progress(
            current,
            total,
            message="Generating narration",
        ):

            try:

                current = int(
                    current
                )

                total = int(
                    total
                )

            except Exception:

                current = 0
                total = 0

            if total > 0:

                fraction = min(
                    1.0,
                    max(
                        0.0,
                        current
                        / float(total),
                    ),
                )

                progress = (
                    50
                    + int(
                        20
                        * fraction
                    )
                )

            else:

                progress = 50

            jobs[jid].update(
                progress=min(
                    70,
                    progress
                ),
                message=str(
                    message
                    or "Generating narration"
                ),
            )

        audio = generate_narration(
            plan,
            job,
            progress_callback=tts_progress,
            language="en-us",
            voice="af_heart",
        )

        (
            job / "audio.json"
        ).write_text(
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

        # -----------------------------------------------------
        # 4. VIDEO RENDERING
        # -----------------------------------------------------

        final = (
            job / "tutorial.mp4"
        )

        render(
            plan,
            audio,
            job,
            final,
        )

        # -----------------------------------------------------
        # VALIDATE VIDEO
        # -----------------------------------------------------

        if (
            not final.exists()
            or final.stat().st_size < 10000
        ):

            raise RuntimeError(
                "Generated video is missing or invalid."
            )

        # -----------------------------------------------------
        # COMPLETED
        # -----------------------------------------------------

        jobs[jid].update(
            status="completed",
            progress=100,
            message="Tutorial ready",
            video_url=(
                f"/media/{jid}/tutorial.mp4"
            ),
            video_path=str(
                final
            ),
            title=plan.get(
                "title",
                "AI Learning Tutorial",
            ),
        )

        print(
            f"[VIDEO] {jid} generated successfully: "
            f"{final} "
            f"({final.stat().st_size / (1024 * 1024):.2f} MB)"
        )

    except Exception as exc:

        traceback.print_exc()

        jobs[jid].update(
            status="error",
            progress=0,
            message=str(
                exc
            ),
            error=str(
                exc
            ),
        )