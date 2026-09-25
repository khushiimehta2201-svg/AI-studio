from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

JOBS = BASE / "jobs"
JOBS.mkdir(parents=True, exist_ok=True)

STORE_PATH = BASE / "studio.sqlite3"

MAX_MB = max(10, int(os.getenv("MAX_PDF_MB", "200")))
WORKERS = max(1, int(os.getenv("GENERATION_WORKERS", "1")))

AUTH_REQUIRED = (
    os.getenv("AUTH_REQUIRED", "true").lower()
    in {"1", "true", "yes", "on"}
)

AUTH_SECRET_VALUE = os.getenv("AUTH_SECRET", "").strip()

if AUTH_REQUIRED and (
    not AUTH_SECRET_VALUE
    or AUTH_SECRET_VALUE == "change-this-secret"
):
    raise RuntimeError(
        "AUTH_REQUIRED=true requires a non-default AUTH_SECRET"
    )

AUTH_SECRET = (
    AUTH_SECRET_VALUE.encode("utf-8")
    if AUTH_SECRET_VALUE
    else b"development-only-secret"
)


def _job_dir(job_id: str) -> Path:
    return JOBS / job_id


from app.services.pdf_service import process_pdf
from app.services.production_planner import build_tutorial_plan
from app.services.portal_store import PortalStore
from app.services.qa_service import (
    validate_plan,
    validate_rendered_artifacts,
)
from app.services.tts_service import generate_narration
from app.services.video_service import render_all_sections


STORE = PortalStore(STORE_PATH)

# Jobs cannot continue running across a process restart because the worker
# is process-local. Mark unfinished work as interrupted rather than leaving
# the UI in a permanent "processing" state.
for _row in STORE.list_jobs():
    if (
        _row.get("status") in {"queued", "processing"}
        and not (_job_dir(_row["id"]) / "tutorial.mp4").exists()
    ):
        STORE.update_job(
            _row["id"],
            status="interrupted",
            message="Generation was interrupted by an application restart.",
        )

POOL = ThreadPoolExecutor(max_workers=WORKERS)
INMEMORY_STATUS = {}

app = FastAPI(
    title="Teamcenter AI Studio Definitive",
    version="4.0",
)

app.mount(
    "/static",
    StaticFiles(directory=BASE / "app" / "static"),
    name="static",
)

templates = Jinja2Templates(
    directory=str(BASE / "app" / "templates")
)


def _session_token(username: str, role: str) -> str:
    payload = f"{username}:{role}".encode()
    sig = hmac.new(
        AUTH_SECRET,
        payload,
        hashlib.sha256,
    ).hexdigest()

    return f"{username}|{role}|{sig}"


def _session(cookie: str | None):
    if not AUTH_REQUIRED:
        return {
            "username": "local",
            "role": "trainer",
        }

    if not cookie:
        return None

    try:
        username, role, sig = cookie.split("|", 2)

        expected = hmac.new(
            AUTH_SECRET,
            f"{username}:{role}".encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(sig, expected):
            return None

        return {
            "username": username,
            "role": role,
        }

    except Exception:
        return None


def _require(
    request: Request,
    role: str | None = None,
):
    session = _session(
        request.cookies.get("ai_session")
    )

    if not session:
        raise PermissionError("Authentication required")

    if role and session.get("role") != role:
        raise PermissionError("Insufficient permissions")

    return session


def _safe_http_url(value: str) -> bool:
    import urllib.parse

    try:
        parsed = urllib.parse.urlparse(
            str(value).strip()
        )

        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.netloc)
        )

    except Exception:
        return False


def _update(job_id: str, **fields):
    INMEMORY_STATUS.setdefault(job_id, {})
    INMEMORY_STATUS[job_id].update(fields)

    mapped = {
        k: v
        for k, v in fields.items()
        if k in {
            "status",
            "progress",
            "message",
            "error",
            "filename",
        }
    }

    if mapped:
        STORE.update_job(
            job_id,
            **mapped,
        )


def discover_videos() -> list[dict]:
    out = []

    for row in STORE.list_jobs():
        jid = row["id"]
        plan = STORE.get_plan(jid) or {}

        if not (
            _job_dir(jid) / "tutorial.mp4"
        ).exists():
            continue

        out.append(
            {
                "id": jid,
                "job_id": jid,
                "title": plan.get(
                    "title",
                    "Teamcenter Tutorial",
                ),
                "description": plan.get(
                    "description",
                    "Guided software training tutorial.",
                ),
                "video_url": (
                    f"/media/trainee/"
                    f"{jid}/tutorial.mp4"
                ),
                "preview_url": (
                    f"/media/trainer/"
                    f"{jid}/tutorial.mp4"
                ),
                "captions_url": (
                    f"/media/trainee/"
                    f"{jid}/tutorial.vtt"
                ),
                "published": (
                    row.get("publication_status")
                    == "published"
                ),
                "publication_status": (
                    row.get(
                        "publication_status",
                        "draft",
                    )
                ),
                "requires_trainer_review": bool(
                    plan.get(
                        "requires_trainer_review"
                    )
                ),
                "qa": plan.get("qa", {}),
                "action_count": plan.get(
                    "action_count",
                    0,
                ),
                "total_pages": plan.get(
                    "total_pages",
                    0,
                ),
                "created_at": row.get(
                    "created_at"
                ),
            }
        )

    return out


@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get(
    "/login",
    response_class=HTMLResponse,
)
async def login_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="login.html",
    )


@app.post("/login")
async def login(request: Request):
    body = await request.json()

    user = str(
        body.get("username", "")
    ).strip()

    password = str(
        body.get("password", "")
    ).strip()

    trainer_user = os.getenv(
        "TRAINER_USERNAME",
        "",
    ).strip()

    trainer_pass = os.getenv(
        "TRAINER_PASSWORD",
        "",
    )

    trainee_user = os.getenv(
        "TRAINEE_USERNAME",
        "",
    ).strip()

    trainee_pass = os.getenv(
        "TRAINEE_PASSWORD",
        "",
    )

    trainer_ok = (
        bool(trainer_user and trainer_pass)
        and user == trainer_user
        and hmac.compare_digest(
            password,
            trainer_pass,
        )
    )

    trainee_ok = (
        bool(trainee_user and trainee_pass)
        and user == trainee_user
        and hmac.compare_digest(
            password,
            trainee_pass,
        )
    )

    if not (trainer_ok or trainee_ok):
        return JSONResponse(
            {"error": "Invalid credentials"},
            status_code=401,
        )

    role = (
        "trainer"
        if trainer_ok
        else "trainee"
    )

    response = JSONResponse(
        {
            "success": True,
            "role": role,
        }
    )

    response.set_cookie(
        "ai_session",
        _session_token(
            user,
            role,
        ),
        httponly=True,
        samesite="lax",
        secure=(
            os.getenv(
                "COOKIE_SECURE",
                "false",
            ).lower()
            in {"1", "true", "yes", "on"}
        ),
        max_age=8 * 3600,
    )

    return response


@app.post("/logout")
async def logout():
    response = JSONResponse(
        {"success": True}
    )

    response.delete_cookie(
        "ai_session"
    )

    return response


@app.get(
    "/trainee",
    response_class=HTMLResponse,
)
async def trainee_page(request: Request):
    try:
        _require(
            request,
            "trainee",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return RedirectResponse(
                "/login"
            )

    return templates.TemplateResponse(
        request=request,
        name="trainee.html",
    )


@app.get(
    "/trainer",
    response_class=HTMLResponse,
)
async def trainer_page(request: Request):
    try:
        _require(
            request,
            "trainer",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return RedirectResponse(
                "/login"
            )

    return templates.TemplateResponse(
        request=request,
        name="trainer.html",
    )


@app.get(
    "/trainee/videos/{video_id}",
    response_class=HTMLResponse,
)
async def watch_page(
    request: Request,
    video_id: str,
):
    try:
        _require(
            request,
            "trainee",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return RedirectResponse(
                "/login"
            )

    return templates.TemplateResponse(
        "watch.html",
        {
            "request": request,
            "video_id": video_id,
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "Teamcenter AI Studio",
        "schema_version": 5,
        "workers": WORKERS,
    }


@app.post("/generate")
async def generate(
    request: Request,
    file: UploadFile = File(...),
):
    try:
        _require(
            request,
            "trainer",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return JSONResponse(
                {
                    "error": (
                        "Authentication required"
                    )
                },
                status_code=401,
            )

    filename = file.filename or ""

    if not filename.lower().endswith(".pdf"):
        return JSONResponse(
            {
                "error": (
                    "Please upload a PDF."
                )
            },
            status_code=400,
        )

    jid = uuid4().hex[:12]

    job = _job_dir(jid)
    job.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf = job / "source.pdf"

    total = 0
    limit = MAX_MB * 1024 * 1024

    try:
        with pdf.open("wb") as fh:
            while True:
                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total += len(chunk)

                if total > limit:
                    fh.close()
                    pdf.unlink(
                        missing_ok=True
                    )
                    job.rmdir()

                    return JSONResponse(
                        {
                            "error": (
                                f"PDF exceeds "
                                f"{MAX_MB} MB."
                            )
                        },
                        status_code=413,
                    )

                fh.write(chunk)

    except Exception:
        import shutil

        shutil.rmtree(
            job,
            ignore_errors=True,
        )
        raise

    try:
        with pdf.open("rb") as fh:
            magic = fh.read(5)

        if (
            pdf.stat().st_size < 5
            or magic != b"%PDF-"
        ):
            import shutil

            shutil.rmtree(
                job,
                ignore_errors=True,
            )

            return JSONResponse(
                {
                    "error": (
                        "Uploaded file is "
                        "not a valid PDF."
                    )
                },
                status_code=400,
            )

    except Exception:
        import shutil

        shutil.rmtree(
            job,
            ignore_errors=True,
        )

        return JSONResponse(
            {
                "error": (
                    "Unable to validate "
                    "uploaded PDF."
                )
            },
            status_code=400,
        )

    STORE.create_job(
        jid,
        filename,
    )

    _update(
        jid,
        status="queued",
        progress=2,
        message="PDF uploaded",
        filename=filename,
    )

    asyncio.get_running_loop().run_in_executor(
        POOL,
        run_job,
        jid,
        pdf,
    )

    return {
        "job_id": jid
    }


@app.get("/status/{jid}")
async def status(
    request: Request,
    jid: str,
):
    try:
        _require(
            request,
            "trainer",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return JSONResponse(
                {
                    "error": (
                        "Authentication required"
                    )
                },
                status_code=401,
            )

    stored = STORE.get_job(jid)

    if not stored:
        return JSONResponse(
            {
                "error": "Job not found"
            },
            status_code=404,
        )

    payload = {
        "job_id": jid,
        "status": stored.get(
            "status"
        ),
        "progress": stored.get(
            "progress",
            0,
        ),
        "message": stored.get(
            "message",
            "",
        ),
        "error": stored.get(
            "error"
        ),
    }

    plan = STORE.get_plan(jid)

    if plan:
        payload["title"] = plan.get(
            "title"
        )

        payload[
            "requires_trainer_review"
        ] = plan.get(
            "requires_trainer_review",
            True,
        )

    if (
        _job_dir(jid) / "tutorial.mp4"
    ).exists():
        payload["video_url"] = (
            f"/media/trainer/"
            f"{jid}/tutorial.mp4"
        )

    return payload


@app.get("/api/trainee/videos")
async def trainee_videos(
    request: Request,
):
    try:
        _require(
            request,
            "trainee",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return JSONResponse(
                {
                    "error": (
                        "Authentication required"
                    )
                },
                status_code=401,
            )

    return {
        "videos": [
            v
            for v in discover_videos()
            if v["published"]
        ]
    }


@app.get("/api/trainee/videos/{jid}")
async def trainee_details(
    request: Request,
    jid: str,
):
    try:
        _require(
            request,
            "trainee",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return JSONResponse(
                {
                    "error": (
                        "Authentication required"
                    )
                },
                status_code=401,
            )

    video = next(
        (
            v
            for v in discover_videos()
            if v["id"] == jid
            and v["published"]
        ),
        None,
    )

    if not video:
        return JSONResponse(
            {
                "error": (
                    "Tutorial not found"
                )
            },
            status_code=404,
        )

    plan = STORE.get_plan(jid) or {}

    actions = []

    for scene in plan.get(
        "steps",
        [],
    ):
        if scene.get("kind") != "action":
            continue

        start = next(
            (
                x.get("start")
                for x in plan.get(
                    "dialogue_timeline",
                    [],
                )
                if x.get("scene_id")
                == scene.get("scene_id")
            ),
            None,
        )

        end = next(
            (
                x.get("end")
                for x in plan.get(
                    "dialogue_timeline",
                    [],
                )
                if x.get("scene_id")
                == scene.get("scene_id")
            ),
            None,
        )

        actions.append(
            {
                "scene_id": scene.get(
                    "scene_id"
                ),
                "title": scene.get(
                    "title"
                ),
                "source_step_number": scene.get(
                    "source_step_number"
                ),
                "start": start,
                "end": end,
            }
        )

    return {
        **video,
        "topic_intro": plan.get(
            "description",
            "Guided software training tutorial.",
        ),
        "actions": actions,
        "timeline": plan.get(
            "dialogue_timeline",
            [],
        ),
        "sections": plan.get(
            "sections",
            [],
        ),
        "prerequisites": STORE.prerequisites(
            jid
        ),
    }


@app.get("/api/trainer/videos")
async def trainer_videos(
    request: Request,
):
    try:
        _require(
            request,
            "trainer",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return JSONResponse(
                {
                    "error": (
                        "Authentication required"
                    )
                },
                status_code=401,
            )

    return {
        "videos": discover_videos()
    }


@app.get(
    "/api/trainer/videos/{jid}/review"
)
async def trainer_review(
    request: Request,
    jid: str,
):
    try:
        _require(
            request,
            "trainer",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return JSONResponse(
                {
                    "error": (
                        "Authentication required"
                    )
                },
                status_code=401,
            )

    plan = STORE.get_plan(jid)

    if not plan:
        return JSONResponse(
            {
                "error": (
                    "Tutorial not found"
                )
            },
            status_code=404,
        )

    return {
        "job_id": jid,
        "title": plan.get("title"),
        "steps": plan.get(
            "steps",
            [],
        ),
        "qa": plan.get(
            "qa",
            {},
        ),
        "reviews": STORE.reviews(
            jid
        ),
    }


@app.post(
    "/api/trainer/scenes/{jid}/{scene_id}/review"
)
async def review_scene(
    request: Request,
    jid: str,
    scene_id: str,
):
    try:
        _require(
            request,
            "trainer",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return JSONResponse(
                {
                    "error": (
                        "Authentication required"
                    )
                },
                status_code=401,
            )

    plan = STORE.get_plan(jid)

    if not plan:
        return JSONResponse(
            {
                "error": (
                    "Tutorial not found"
                )
            },
            status_code=404,
        )

    body = await request.json()

    action = str(
        body.get(
            "decision",
            "",
        )
    )

    scene = next(
        (
            s
            for s in plan.get(
                "steps",
                [],
            )
            if s.get("scene_id")
            == scene_id
        ),
        None,
    )

    if not scene:
        return JSONResponse(
            {
                "error": "Scene not found"
            },
            status_code=404,
        )

    if action == "approve_target":
        point = scene.get(
            "cursor"
        )

        valid_point = (
            isinstance(
                point,
                list,
            )
            and len(point) == 2
            and all(
                isinstance(
                    v,
                    (int, float),
                )
                and 0 <= float(v) <= 1
                for v in point
            )
        )

        if (
            scene.get("interaction")
            != "none"
            and not valid_point
        ):
            return JSONResponse(
                {
                    "error": (
                        "Cannot approve "
                        "a target without "
                        "a valid cursor "
                        "point."
                    )
                },
                status_code=409,
            )

        if (
            scene.get("interaction")
            == "drag"
            and not (
                isinstance(
                    scene.get(
                        "cursor_path"
                    ),
                    list,
                )
                and len(
                    scene.get(
                        "cursor_path"
                    )
                )
                >= 2
            )
        ):
            return JSONResponse(
                {
                    "error": (
                        "A drag target "
                        "requires both "
                        "a start and end "
                        "point."
                    )
                },
                status_code=409,
            )

        scene[
            "target_status"
        ] = "verified"

        scene[
            "target_review_required"
        ] = False

        scene[
            "cursor_enabled"
        ] = (
            scene.get(
                "interaction"
            )
            != "none"
        )

        scene[
            "cursor_source"
        ] = "trainer_approved"

        scene[
            "cursor_confidence"
        ] = 1.0

        STORE.review(
            jid,
            scene_id,
            "approved",
            "",
        )

    elif action == "set_target":
        point = body.get(
            "point"
        )

        points = body.get(
            "points"
        )

        box = body.get(
            "bounding_box"
        )

        if scene.get(
            "interaction"
        ) == "drag":

            if not (
                isinstance(
                    points,
                    list,
                )
                and len(points) >= 2
            ):
                return JSONResponse(
                    {
                        "error": (
                            "drag targets "
                            "must provide "
                            "points="
                            "[[start_x,"
                            "start_y],"
                            "[end_x,"
                            "end_y]]"
                        )
                    },
                    status_code=400,
                )

            if any(
                not (
                    isinstance(
                        p,
                        (list, tuple),
                    )
                    and len(p) == 2
                    and all(
                        0
                        <= float(v)
                        <= 1
                        for v in p
                    )
                )
                for p in points[:2]
            ):
                return JSONResponse(
                    {
                        "error": (
                            "drag points "
                            "must be "
                            "normalized "
                            "0..1"
                        )
                    },
                    status_code=400,
                )

            norm_points = [
                [float(v) for v in p]
                for p in points[:2]
            ]

            scene.update(
                {
                    "cursor": norm_points[-1],
                    "cursor_path": norm_points,
                    "cursor_bounding_box": box,
                    "target_status": "verified",
                    "target_review_required": False,
                    "cursor_enabled": True,
                    "cursor_source": "trainer",
                    "cursor_confidence": 1.0,
                }
            )

            STORE.review(
                jid,
                scene_id,
                "target_set",
                str(norm_points),
            )

        else:
            if not (
                isinstance(
                    point,
                    list,
                )
                and len(point) == 2
            ):
                return JSONResponse(
                    {
                        "error": (
                            "point must be "
                            "[x,y] in "
                            "normalized "
                            "coordinates"
                        )
                    },
                    status_code=400,
                )

            if any(
                float(x) < 0
                or float(x) > 1
                for x in point
            ):
                return JSONResponse(
                    {
                        "error": (
                            "point must be "
                            "normalized "
                            "0..1"
                        )
                    },
                    status_code=400,
                )

            norm_point = [
                float(point[0]),
                float(point[1]),
            ]

            scene.update(
                {
                    "cursor": norm_point,
                    "cursor_path": [
                        norm_point
                    ],
                    "cursor_bounding_box": box,
                    "target_status": "verified",
                    "target_review_required": False,
                    "cursor_enabled": (
                        scene.get(
                            "interaction"
                        )
                        != "none"
                    ),
                    "cursor_source": "trainer",
                    "cursor_confidence": 1.0,
                }
            )

            STORE.review(
                jid,
                scene_id,
                "target_set",
                str(norm_point),
            )

    elif action == "mark_no_target":
        # Record the observation without treating
        # a required visual target as resolved.
        scene.update(
            {
                "cursor": None,
                "cursor_bounding_box": None,
                "cursor_path": [],
                "cursor_enabled": False,
                "target_status": "unresolved",
                "target_review_required": True,
                "cursor_source": "trainer_no_target",
            }
        )

        STORE.review(
            jid,
            scene_id,
            "no_target",
            (
                "Trainer could not identify "
                "a trustworthy target; "
                "scene remains blocked "
                "from publication."
            ),
        )

    else:
        return JSONResponse(
            {
                "error": (
                    "decision must be "
                    "approve_target, "
                    "set_target, or "
                    "mark_no_target"
                )
            },
            status_code=400,
        )

    # Re-render from stored audio and re-run QA
    # after every review change.
    (
        _job_dir(jid) / "plan.json"
    ).write_text(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    audio = []

    audio_path = (
        _job_dir(jid) / "audio.json"
    )

    if audio_path.exists():
        audio = json.loads(
            audio_path.read_text(
                encoding="utf-8"
            )
        )

    pre = validate_plan(plan)

    result = render_all_sections(
        plan,
        audio,
        str(_job_dir(jid)),
    )

    post = validate_rendered_artifacts(
        str(_job_dir(jid)),
        result,
    )

    plan["qa"] = {
        "pre_render": pre,
        "post_render": post,
        "publishable": (
            bool(
                pre.get("publishable")
            )
            and bool(
                post.get("publishable")
            )
            and not any(
                s.get(
                    "target_review_required"
                )
                for s in plan.get(
                    "steps",
                    [],
                )
            )
        ),
    }

    plan[
        "requires_trainer_review"
    ] = not plan[
        "qa"
    ][
        "publishable"
    ]

    STORE.set_plan(
        jid,
        plan,
    )

    STORE.update_job(
        jid,
        publication_status="draft",
    )

    return {
        "success": True,
        "scene": scene,
        "publishable": plan[
            "qa"
        ][
            "publishable"
        ],
    }


@app.post(
    "/api/trainer/publish/{jid}"
)
async def publish(
    request: Request,
    jid: str,
):
    try:
        _require(
            request,
            "trainer",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return JSONResponse(
                {
                    "error": (
                        "Authentication required"
                    )
                },
                status_code=401,
            )

    plan = STORE.get_plan(jid)

    if not plan:
        return JSONResponse(
            {
                "error": (
                    "Tutorial not found"
                )
            },
            status_code=404,
        )

    if (
        request.headers.get(
            "content-type",
            "",
        ).startswith(
            "application/json"
        )
    ):
        body = await request.json()
    else:
        body = {}

    current = (
        STORE.get_job(jid) or {}
    )

    desired = bool(
        body.get(
            "published",
            current.get(
                "publication_status"
            )
            != "published",
        )
    )

    if desired:
        qa = plan.get(
            "qa",
            {}
        )

        unresolved = [
            s.get("scene_id")
            for s in plan.get(
                "steps",
                [],
            )
            if s.get(
                "target_review_required"
            )
        ]

        if (
            not qa.get("publishable")
            or unresolved
        ):
            return JSONResponse(
                {
                    "error": (
                        "Tutorial has not "
                        "passed QA/review."
                    ),
                    "review_required": unresolved,
                    "qa": qa,
                },
                status_code=409,
            )

        STORE.set_publication(
            jid,
            "published",
        )

    else:
        STORE.set_publication(
            jid,
            "draft",
        )

    return {
        "success": True,
        "published": desired,
    }


@app.post(
    "/api/trainer/prerequisites/{jid}"
)
async def prerequisites(
    request: Request,
    jid: str,
):
    try:
        _require(
            request,
            "trainer",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return JSONResponse(
                {
                    "error": (
                        "Authentication required"
                    )
                },
                status_code=401,
            )

    if not STORE.get_job(jid):
        return JSONResponse(
            {
                "error": (
                    "Tutorial not found"
                )
            },
            status_code=404,
        )

    body = await request.json()

    urls = body.get(
        "prerequisites",
        [],
    )

    if isinstance(urls, str):
        urls = [urls]
    elif not isinstance(urls, list):
        urls = []

    cleaned = []
    invalid = []

    for u in urls:
        value = str(u).strip()

        if not value:
            continue

        if _safe_http_url(value):
            cleaned.append(value)
        else:
            invalid.append(value)

    urls = list(
        dict.fromkeys(cleaned)
    )

    if invalid:
        return JSONResponse(
            {
                "error": (
                    "Prerequisite URLs must "
                    "use http:// or https://."
                ),
                "invalid": invalid,
            },
            status_code=400,
        )

    STORE.set_prerequisites(
        jid,
        urls,
    )

    return {
        "success": True,
        "prerequisites": urls,
    }


@app.get(
    "/media/trainer/{jid}/scene/{scene_id}.png"
)
async def scene_preview(
    request: Request,
    jid: str,
    scene_id: str,
):
    try:
        _require(
            request,
            "trainer",
        )
    except PermissionError:
        if AUTH_REQUIRED:
            return JSONResponse(
                {
                    "error": (
                        "Authentication required"
                    )
                },
                status_code=401,
            )

    plan = STORE.get_plan(jid)

    if not plan:
        return JSONResponse(
            {
                "error": (
                    "Tutorial not found"
                )
            },
            status_code=404,
        )

    scene = next(
        (
            s
            for s in plan.get(
                "steps",
                [],
            )
            if s.get("scene_id")
            == scene_id
        ),
        None,
    )

    if not scene:
        return JSONResponse(
            {
                "error": "Scene not found"
            },
            status_code=404,
        )

    raw = Path(
        str(
            scene.get(
                "screenshot"
            )
            or ""
        )
    )

    try:
        raw = raw.resolve()
        job_root = (
            _job_dir(jid).resolve()
        )

        raw.relative_to(
            job_root
        )

    except Exception:
        return JSONResponse(
            {
                "error": (
                    "Scene preview unavailable"
                )
            },
            status_code=404,
        )

    if not raw.exists():
        return JSONResponse(
            {
                "error": (
                    "Scene preview unavailable"
                )
            },
            status_code=404,
        )

    return FileResponse(raw)


@app.get(
    "/media/{audience}/{jid}/{filename}"
)
async def media(
    request: Request,
    audience: str,
    jid: str,
    filename: str,
):
    safe = {
        "tutorial.mp4",
        "tutorial.vtt",
    }

    if filename not in safe:
        return JSONResponse(
            {
                "error": (
                    "Media not available"
                )
            },
            status_code=404,
        )

    row = STORE.get_job(jid)

    if not row:
        return JSONResponse(
            {
                "error": (
                    "Tutorial not found"
                )
            },
            status_code=404,
        )

    if audience == "trainee":

        try:
            _require(
                request,
                "trainee",
            )
        except PermissionError:
            if AUTH_REQUIRED:
                return JSONResponse(
                    {
                        "error": (
                            "Authentication required"
                        )
                    },
                    status_code=401,
                )

        if (
            row.get(
                "publication_status"
            )
            != "published"
        ):
            return JSONResponse(
                {
                    "error": (
                        "Tutorial is not "
                        "published"
                    )
                },
                status_code=404,
            )

    elif audience == "trainer":

        try:
            _require(
                request,
                "trainer",
            )
        except PermissionError:
            if AUTH_REQUIRED:
                return JSONResponse(
                    {
                        "error": (
                            "Authentication required"
                        )
                    },
                    status_code=401,
                )

    else:
        return JSONResponse(
            {
                "error": (
                    "Invalid audience"
                )
            },
            status_code=404,
        )

    path = (
        _job_dir(jid) / filename
    )

    if not path.exists():
        return JSONResponse(
            {
                "error": (
                    "Media not found"
                )
            },
            status_code=404,
        )

    return FileResponse(path)


def run_job(
    jid: str,
    pdf: Path,
):
    job = pdf.parent

    try:
        _update(
            jid,
            status="processing",
            progress=8,
            message=(
                "Reading document structure "
                "and visual evidence"
            ),
        )

        data = process_pdf(
            str(pdf),
            job,
        )

        data["job_dir"] = str(job)
        data["filename"] = pdf.name

        _update(
            jid,
            progress=25,
            message=(
                f"Indexed "
                f"{data.get('page_count', 0)} "
                f"pages and "
                f"{data.get('screenshot_count', 0)} "
                f"visual assets"
            ),
        )

        plan = build_tutorial_plan(
            data,
            narration_language=os.getenv(
                "NARRATION_LANGUAGE",
                "en-us",
            ),
            progress_callback=lambda p, m="": _update(
                jid,
                progress=28 + int(p * 0.22),
                message=m,
            ),
        )

        pre = validate_plan(plan)

        plan["qa"] = {
            "pre_render": pre
        }

        plan[
            "requires_trainer_review"
        ] = not pre.get(
            "publishable",
            False,
        )

        blocking = [
            x
            for x in pre.get(
                "errors",
                [],
            )
            if x.get("code")
            not in {
                "REQUIRED_TARGET_MISSING"
            }
        ]

        if blocking:
            raise RuntimeError(
                "Plan QA failed: "
                + "; ".join(
                    x["message"]
                    for x in blocking[:4]
                )
            )

        STORE.set_plan(
            jid,
            plan,
        )

        _update(
            jid,
            progress=52,
            message=(
                "Generating narration audio"
            ),
        )

        # IMPORTANT:
        # generate_narration() calls its callback as:
        #     callback(current, total, message)
        # so this lambda must accept all three arguments.
        audio = generate_narration(
            plan,
            job,
            progress_callback=lambda p, total, m="": _update(
                jid,
                progress=52 + int(
                    (p / max(1, total))
                    * 20
                ),
                message=m,
            ),
            language=os.getenv(
                "NARRATION_LANGUAGE",
                "en-us",
            ),
            voice=os.getenv(
                "TTS_VOICE",
                "af_heart",
            ),
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

        _update(
            jid,
            progress=75,
            message=(
                "Rendering synchronized "
                "tutorial, captions "
                "and cursor timeline"
            ),
        )

        rendered = render_all_sections(
            plan,
            audio,
            str(job),
        )

        post = validate_rendered_artifacts(
            str(job),
            rendered,
        )

        plan["qa"][
            "post_render"
        ] = post

        plan["qa"][
            "publishable"
        ] = (
            bool(
                pre.get("publishable")
            )
            and bool(
                post.get("publishable")
            )
            and not any(
                s.get(
                    "target_review_required"
                )
                for s in plan.get(
                    "steps",
                    [],
                )
            )
        )

        plan[
            "requires_trainer_review"
        ] = not plan[
            "qa"
        ][
            "publishable"
        ]

        STORE.set_plan(
            jid,
            plan,
        )

        _update(
            jid,
            status="completed",
            progress=100,
            message=(
                "Tutorial ready for "
                "trainer review"
            ),
            error=None,
        )

    except Exception as exc:
        traceback.print_exc()

        _update(
            jid,
            status="error",
            progress=0,
            message=str(exc),
            error=str(exc),
        )