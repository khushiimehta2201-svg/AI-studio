"""Machine QA for document plans, scene contracts and rendered artifacts."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def issue(code:str,severity:str,message:str,scene_id:Any=None)->Dict[str,Any]:
    x={"code":code,"severity":severity,"message":message}
    if scene_id is not None:x["scene_id"]=scene_id
    return x


def _valid_num(v:Any)->bool:
    return bool(v and re.fullmatch(r"\d+(?:\.\d+)*",str(v)))


def validate_plan(plan:Dict[str,Any])->Dict[str,Any]:
    errors=[];warnings=[];steps=plan.get("steps",[])
    if not isinstance(steps,list) or not steps:errors.append(issue("NO_SCENES","error","Tutorial contains no scenes."));steps=[]
    ids=[]; last_source_page=0
    for s in steps:
        sid=s.get("scene_id")
        if sid in ids:errors.append(issue("DUPLICATE_SCENE_ID","error",f"Duplicate scene id {sid}.",sid))
        ids.append(sid)
        kind=s.get("kind")
        if kind not in {"action","explanation","transition"}:errors.append(issue("INVALID_SCENE_KIND","error","Invalid scene kind.",sid))
        if int(s.get("source_page") or 0)<1:errors.append(issue("MISSING_PROVENANCE","error","Scene is missing source page.",sid))
        page=int(s.get("source_page") or 0)
        if page<last_source_page:warnings.append(issue("SOURCE_ORDER_NONMONOTONIC","warning","Source-page order changes across scenes.",sid))
        last_source_page=max(last_source_page,page)
        n=s.get("source_step_number")
        if kind=="action":
            if not clean_narration(s.get("narration")):errors.append(issue("NARRATION_INVALID","error","Action has no usable narration.",sid))
            target_required=s.get("interaction")!="none"
            if target_required and s.get("target_status") not in {"verified","review_required","unresolved","not_applicable"}:
                errors.append(issue("TARGET_STATUS_INVALID","error","Unknown target status.",sid))
            if target_required and s.get("target_status") in {"unresolved","not_applicable"}:
                warnings.append(issue("TARGET_UNRESOLVED","warning","Action requires a visual target but no verified target is available.",sid))
                errors.append(issue("REQUIRED_TARGET_MISSING","error","A target-requiring action cannot be published without a verified cursor target.",sid))
            if s.get("cursor_enabled") and s.get("target_status")!="verified":errors.append(issue("CURSOR_WITHOUT_VERIFICATION","error","Cursor is enabled for an unverified target.",sid))
            if s.get("target_status")=="verified" and s.get("interaction")!="none":
                point=s.get("cursor")
                valid_point=isinstance(point,(list,tuple)) and len(point)==2 and all(isinstance(v,(int,float)) and 0<=float(v)<=1 for v in point)
                if not valid_point:
                    errors.append(issue("VERIFIED_TARGET_MISSING_POINT","error","Verified target has no valid normalized cursor point.",sid))
                if str(s.get("interaction"))=="drag":
                    path=s.get("cursor_path") or []
                    if not(isinstance(path,list) and len(path)>=2):
                        errors.append(issue("VERIFIED_DRAG_PATH_MISSING","error","Verified drag action must contain a normalized start and end target.",sid))
        if n is not None and not _valid_num(n):warnings.append(issue("SOURCE_NUMBER_UNUSUAL","warning",f"Source step number is non-standard: {n}.",sid))
    # Numbering gap warnings within each exact parent prefix.
    groups={}
    for s in steps:
        n=s.get("source_step_number")
        if not _valid_num(n):continue
        parts=[int(x) for x in str(n).split('.')]; groups.setdefault(tuple(parts[:-1]),set()).add(parts[-1])
    for prefix,values in groups.items():
        if len(values)>=2:
            gaps=[str(n) for n in range(min(values),max(values)+1) if n not in values]
            if gaps:warnings.append(issue("MISSING_SOURCE_NUMBER","warning",f"Possible missing step number(s) under {'.'.join(map(str,prefix)) or 'root'}: {gaps}"))
    return {"schema_version":1,"publishable":not errors and not any(s.get("target_review_required") for s in steps),"errors":errors,"warnings":warnings,"review_required":[x for x in steps if x.get("target_review_required")],"summary":{"scenes":len(steps),"actions":sum(s.get("kind")=="action" for s in steps),"unresolved_targets":sum(s.get("target_status")=="unresolved" for s in steps)}}


def clean_narration(text:Any)->str:
    t=re.sub(r"https?://\S+|www\.\S+"," ",str(text or ""),flags=re.I);t=re.sub(r"\s+"," ",t).strip()
    if len(t.split())<2 or len(t.split())>45:return ""
    if not re.search(r"[.!?]$",t):return ""
    return t


def validate_rendered_artifacts(job_dir:str,render_result:Dict[str,Any])->Dict[str,Any]:
    job=Path(job_dir);errors=[];warnings=[]
    video=job/str(render_result.get("video_file") or "");vtt=job/str(render_result.get("captions_file") or "")
    if not video.exists() or video.stat().st_size<1024:errors.append(issue("VIDEO_INVALID","error","Rendered video is missing or invalid."))
    if not vtt.exists():errors.append(issue("CAPTIONS_MISSING","error","VTT captions are missing."))
    timeline=render_result.get("timeline") or render_result.get("dialogue_timeline") or []
    if not timeline:errors.append(issue("TIMELINE_MISSING","error","Canonical timeline is empty."))
    prev=0.0; timeline_ids=[]
    for item in timeline:
        start=float(item.get("start",-1));end=float(item.get("end",-1)); sid=item.get("scene_id")
        timeline_ids.append(sid)
        if start<prev-0.001 or end<=start:errors.append(issue("TIMELINE_INVALID","error","Timeline intervals are not monotonic.",sid))
        prev=max(prev,end)
    plan_path=job/"plan.json"
    if plan_path.exists():
        try:
            plan=json.loads(plan_path.read_text(encoding="utf-8")); expected=[s.get("scene_id") for s in plan.get("steps",[]) if s.get("kind") in {"action","explanation","transition"}]
            if expected!=timeline_ids:errors.append(issue("TIMELINE_PLAN_MISMATCH","error","Rendered timeline scene IDs do not exactly match the plan scene IDs."))
        except Exception as exc:warnings.append(issue("PLAN_QA_READ_FAILED","warning",str(exc)))
    timeline_end=float(timeline[-1].get("end",0.0)) if timeline else 0.0
    ffprobe=shutil.which("ffprobe")
    if ffprobe and video.exists():
        try:
            p=subprocess.run([ffprobe,"-v","error","-show_entries","stream=codec_type,codec_name,width,height,duration","-of","json",str(video)],capture_output=True,text=True,timeout=20)
            streams=json.loads(p.stdout).get("streams",[])
            if not any(s.get("codec_type")=="video" for s in streams):errors.append(issue("NO_VIDEO_STREAM","error","No video stream present."))
            if not any(s.get("codec_type")=="audio" for s in streams):errors.append(issue("NO_AUDIO_STREAM","error","No audio stream present."))
            durations=[float(s.get("duration")) for s in streams if s.get("duration") not in (None,"")]
            if durations and timeline_end>0:
                actual=max(durations)
                if abs(actual-timeline_end)>0.35:errors.append(issue("TIMELINE_DURATION_MISMATCH","error",f"Rendered duration {actual:.3f}s differs from canonical timeline {timeline_end:.3f}s."))
        except Exception as exc:warnings.append(issue("FFPROBE_FAILED","warning",str(exc)))
    return {"schema_version":1,"publishable":not errors,"errors":errors,"warnings":warnings}
