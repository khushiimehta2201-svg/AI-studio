"""Lightweight trainer/trainee portal state.

The portal exposes ONE generated tutorial per PDF. Prerequisites are
recommendations only: they are displayed to trainees but never lock or block
the current tutorial.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE = Path(__file__).resolve().parent.parent.parent
JOBS_DIR = BASE / "jobs"
LIBRARY_PATH = BASE / "portal_library.json"
PROGRESS_PATH = BASE / "portal_progress.json"


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default
    return default


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def video_id(jid: str, section_id: int = 0) -> str:
    """Backward-compatible ID. The portal now uses section_id=0 for the
    single continuous tutorial generated from a PDF."""
    return f"{jid}:0"


def parse_video_id(vid: str):
    jid, _ = vid.split(":", 1)
    return jid, 0


def list_all_tutorials() -> List[Dict[str, Any]]:
    """Return one portal video entry for each completed tutorial job."""
    out: List[Dict[str, Any]] = []
    if not JOBS_DIR.exists():
        return out

    for job_dir in sorted(JOBS_DIR.iterdir()):
        if not job_dir.is_dir():
            continue
        plan_path = job_dir / "plan.json"
        if not plan_path.exists():
            continue
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        video_file = plan.get("video_file")
        if not video_file or not (job_dir / video_file).exists():
            continue

        vid = video_id(job_dir.name, 0)
        steps = [s for s in plan.get("steps", []) if s.get("kind") == "action"]
        sections = plan.get("sections", [])
        intro = plan.get("overview") or plan.get("description") or "Guided software workflow tutorial."
        action_bullets = [s.get("title") for s in steps[:8] if s.get("title")]
        timeline = plan.get("dialogue_timeline") or []
        if not timeline and plan.get("timeline_file"):
            timeline_path = job_dir / str(plan["timeline_file"])
            try:
                timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                timeline = []

        out.append({
            "jid": job_dir.name,
            "title": plan.get("title") or "Untitled tutorial",
            "id": 0,
            "section_title": plan.get("title") or "Complete Tutorial",
            "video_url": f"/media/{job_dir.name}/{video_file}",
            "captions_url": f"/media/{job_dir.name}/{plan['captions_file']}" if plan.get("captions_file") else None,
            "action_bullets": action_bullets,
            "action_count": len(steps),
            "duration": plan.get("video_duration", 0),
            "overview": intro,
            "sections": sections,
            "dialogue_timeline": timeline,
        })
    return out


def get_video_entry(vid: str) -> Optional[Dict[str, Any]]:
    jid, _ = parse_video_id(vid)
    for tutorial in list_all_tutorials():
        if tutorial["jid"] == jid:
            return {**tutorial, "vid": vid}
    return None


def load_library() -> Dict[str, Any]:
    return _load_json(LIBRARY_PATH, {})


def _entry(lib: Dict[str, Any], vid: str) -> Dict[str, Any]:
    return lib.setdefault(vid, {"published": False, "prerequisites": []})


def is_published(vid: str, lib: Optional[Dict[str, Any]] = None) -> bool:
    lib = lib if lib is not None else load_library()
    return bool(lib.get(vid, {}).get("published"))


def set_published(vid: str, published: bool) -> None:
    lib = load_library()
    _entry(lib, vid)["published"] = bool(published)
    _save_json(LIBRARY_PATH, lib)


def get_prerequisites(vid: str, lib: Optional[Dict[str, Any]] = None) -> List[str]:
    lib = lib if lib is not None else load_library()
    return list(lib.get(vid, {}).get("prerequisites", []))


def _would_cycle(vid: str, candidate: str, lib: Dict[str, Any]) -> bool:
    if vid == candidate:
        return True
    stack = [candidate]
    seen = set()
    while stack:
        current = stack.pop()
        if current == vid:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(lib.get(current, {}).get("prerequisites", []))
    return False


def set_prerequisites(vid: str, prerequisite_ids: List[str]) -> List[str]:
    lib = load_library()
    warnings: List[str] = []
    safe: List[str] = []
    for candidate in prerequisite_ids:
        if _would_cycle(vid, candidate, lib):
            warnings.append(f"Skipped '{candidate}' -- would create a circular prerequisite chain.")
            continue
        safe.append(candidate)
    _entry(lib, vid)["prerequisites"] = safe
    _save_json(LIBRARY_PATH, lib)
    return warnings


def load_progress() -> Dict[str, List[str]]:
    return _load_json(PROGRESS_PATH, {})


def is_watched(trainee_id: str, vid: str, progress: Optional[Dict[str, Any]] = None) -> bool:
    progress = progress if progress is not None else load_progress()
    return vid in progress.get(trainee_id, [])


def mark_watched(trainee_id: str, vid: str) -> None:
    progress = load_progress()
    watched = progress.setdefault(trainee_id, [])
    if vid not in watched:
        watched.append(vid)
    _save_json(PROGRESS_PATH, progress)


def get_missing_prerequisites(trainee_id: str, vid: str, lib=None, progress=None) -> List[str]:
    lib = lib if lib is not None else load_library()
    progress = progress if progress is not None else load_progress()
    watched = set(progress.get(trainee_id, []))
    return [p for p in get_prerequisites(vid, lib) if p not in watched]


def is_locked_for(*args, **kwargs) -> bool:
    """Compatibility shim. Prerequisites are recommendations, never locks."""
    return False
