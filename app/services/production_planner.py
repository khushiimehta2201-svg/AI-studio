"""Canonical document-to-scene compiler.

The output of this module is the single source of truth consumed by TTS, video,
QA, trainer review and the trainee portal.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.document_understanding import (
    clean_text, extract_instruction_blocks, is_metadata_page, section_title,
    split_compound_action, line_marker, is_action_line,
)
from app.services.model_adapters import create_clients
from app.services.targeting import action_type, ground_action, interaction_for, clear_caches

SCHEMA_VERSION=4
MAX_PROMPT_CHARS=max(3000,int(os.getenv("AI_BATCH_MAX_CHARS","7500")))
MIN_WORDS={"simple":6,"navigation":8,"form_entry":10,"complex":18,"drag_drop":10}
MAX_WORDS={"simple":18,"navigation":22,"form_entry":26,"complex":40,"drag_drop":28}


def _urls(text:Any)->List[str]:
    if isinstance(text,list):
        return [str(x).strip() for x in text if str(x).strip()]
    out=[]
    for u in re.findall(r"(?<![\w@])(?:https?://|www\.)[^\s<>\]\[\"')]+",str(text or ""),re.I):
        u=u.rstrip(".,;:!?)]}")
        if u and u not in out:out.append(u)
    return out


def _remove_urls(text:str)->str:
    return clean_text(re.sub(r"(?<![\w@])(?:https?://|www\.)[^\s<>\]\[\"')]+"," ",str(text or ""),flags=re.I))


def complexity(action:str)->str:
    t=clean_text(action).lower(); v=len(re.findall(r"\b(?:click|select|choose|open|enter|type|press|create|save|submit|drag|drop|navigate|search|fill|upload|download)\b",t))
    if "drag" in t or "drop" in t:return "drag_drop"
    if v>=2 or len(t.split())>=24:return "complex"
    if any(x in t for x in ("enter ","type ","fill ","search ")):return "form_entry"
    if any(x in t for x in ("open ","navigate ","visit ","launch ")):return "navigation"
    return "simple"


def _context(action:str,context:str,page_text:str)->str:
    a={t.lower() for t in re.findall(r"[A-Za-z0-9_]+",_remove_urls(action)) if len(t)>2}
    candidates=[]
    for s in re.split(r"(?<=[.!?])\s+",_remove_urls(context)):
        s=clean_text(s)
        if len(s.split())>=5:candidates.append(s)
    if not candidates:
        for s in re.split(r"(?<=[.!?])\s+",_remove_urls(page_text)):
            s=clean_text(s)
            if len(s.split())>=5:candidates.append(s)
    scored=[]
    for s in candidates:
        st={t.lower() for t in re.findall(r"[A-Za-z0-9_]+",s) if len(t)>2}
        overlap=len(a&st); causal=2 if re.search(r"\b(?:allows|opens|displays|shows|used for|so that|because|after|before|enables|provides)\b",s,re.I) else 0
        if overlap or causal:scored.append((overlap+causal,s))
    scored.sort(reverse=True,key=lambda x:x[0])
    return clean_text(" ".join(s for _,s in scored[:2]))[:700]


def _fallback_narration(action:str,context:str)->str:
    action=_remove_urls(action).rstrip(" .;,")
    if not action:return "Continue with the next step."
    ctx=_remove_urls(context).strip()
    if ctx and len((action+" "+ctx).split())<=42 and re.match(r"^[A-Z]",ctx):
        return action[:1].upper()+action[1:]+". "+ctx
    return action[:1].upper()+action[1:]+"."


def _batched(items:List[Dict[str,Any]])->List[List[Dict[str,Any]]]:
    out=[]; cur=[]; chars=0
    for x in items:
        piece=json.dumps(x,ensure_ascii=False)
        if cur and chars+len(piece)>MAX_PROMPT_CHARS:
            out.append(cur); cur=[]; chars=0
        cur.append(x); chars+=len(piece)
    if cur:out.append(cur)
    return out


def _semantic_extract(text_client,page:Dict[str,Any])->List[Dict[str,Any]]:
    text=clean_text(page.get("text",""))
    if not text or len(text.split())<8:return []
    prompt=f"""
Extract only actionable software instructions from this document page.
Do not extract title-page metadata, tables of contents, indexes, authors, versions,
explanations, background, results, or screenshot descriptions as actions.
If an instruction contains multiple independent imperative operations, return them
as separate actions. Preserve source wording and source numbering when present.
Return ONLY JSON:
{{"items":[{{"step_number":"4.2.1 or null","action":"...","context":"..."}}]}}
PAGE:
{text[:7000]}
""".strip()
    try:
        data=page.get("_semantic_cache")
        if data is None:data=text_client.generate_json(prompt) or {}
        items=data.get("items",[]) if isinstance(data,dict) else []
        out=[]
        for x in items:
            if not isinstance(x,dict):
                continue
            action=clean_text(x.get("action"))
            if action and (is_action_line(action) or re.match(r"^(?:use|go to|make sure|select|click|choose|open|enter|type|press|save|create|drag|drop|navigate|search|apply|fill|upload|download)\b",action,re.I)):
                out.append(x)
        return out[:25]
    except Exception as exc:
        print(f"[PLANNER] semantic extraction unavailable: {exc}")
        return []


def _narrate(text_client,items:List[Dict[str,Any]],language:str="en-us")->Dict[str,str]:
    result={}
    for batch in _batched(items):
        prompt=(f"Generate concise, source-grounded narration for each software-training item in {language}. "
                "Preserve exact UI labels and technical terms. Add purpose only when supported. "
                "Never invent facts, outcomes, URLs, page references, or metadata. "
                "Return ONLY JSON {\"narrations\":[{\"id\":\"...\",\"narration\":\"...\"}]}\n\n"+
                json.dumps(batch,ensure_ascii=False))
        try:data=text_client.generate_json(prompt) or {}
        except Exception as exc:
            print(f"[PLANNER] narration batch unavailable: {exc}"); data={}
        for x in data.get("narrations",[]) if isinstance(data,dict) else []:
            if isinstance(x,dict) and x.get("id") and x.get("narration"):
                n=_remove_urls(str(x["narration"]).strip()).strip(" `\"")
                if 4<=len(n.split())<=45 and re.search(r"[.!?]$",n):result[str(x["id"])]=n
    return result


def build_tutorial_plan(data:Dict[str,Any],narration_language:str="en-us",progress_callback=None)->Dict[str,Any]:
    pages=[p for p in data.get("pages",[]) if isinstance(p,dict)]
    assets=[a for a in data.get("screenshots",[]) if isinstance(a,dict) and a.get("visual_role") not in {"decorative","logo","metadata"}]
    job_dir=Path(str(data.get("job_dir") or data.get("output_dir") or "."))
    clear_caches()
    text_client,vision_client=create_clients(job_dir if job_dir.exists() else None)

    action_items=[]; explanation_items=[]; sections={}
    for pidx,page in enumerate(pages):
        if is_metadata_page(page):
            page["page_role"]="metadata"; continue
        page["page_role"]="content"
        blocks=extract_instruction_blocks(page)
        semantic=[]
        explicit_action_count=sum(1 for b in blocks if b.get("action"))
        marker_count=sum(1 for line in str(page.get("text") or "").splitlines() if line_marker(line)[0] is not None)
        verb_lines=sum(1 for line in str(page.get("text") or "").splitlines() if re.search(r"\b(?:click|select|choose|open|enter|type|press|save|create|drag|drop|upload|download|navigate|search|apply|fill)\b", line, re.I))
        suspicious_partial=(marker_count>=2 and explicit_action_count<marker_count) or (explicit_action_count>0 and verb_lines>explicit_action_count+1)
        semantic_needed=suspicious_partial or (explicit_action_count==0 and (marker_count>=1 or verb_lines>=1))
        if semantic_needed:
            semantic=_semantic_extract(text_client,page)
        source_blocks=list(blocks)
        if semantic:
            existing={clean_text(b.get("action")).lower() for b in source_blocks if b.get("action")}
            for x in semantic:
                action=clean_text(x.get("action"))
                if action and action.lower() not in existing:
                    source_blocks.append({"action":action,"context":x.get("context",""),"step_number":x.get("step_number"),"bbox":None})
                    existing.add(action.lower())
        if not source_blocks:
            source_blocks=[]
        sec=clean_text(section_title(page)) or f"Section {pidx+1}"
        detail=sections.setdefault(sec,{"title":sec,"scene_ids":[],"actions":[],"explanations":[]})
        for block_order,b in enumerate(source_blocks):
            act=clean_text(b.get("action")); ctx=clean_text(b.get("context"))
            if act:
                atoms=split_compound_action(act)
                for sub_i,atom in enumerate(atoms,1):
                    item={"id":f"a{len(action_items)+1:04d}","page":int(page.get("page") or pidx+1),"source_order":block_order,"source_step_number":b.get("step_number"),"source_substep":sub_i if len(atoms)>1 else None,"action":atom,"context":ctx,"bbox":b.get("bbox"),"page_text":page.get("text",""),"section":sec}
                    action_items.append(item); detail["actions"].append(item["id"])
            elif ctx and len(_remove_urls(ctx).split())>=12:
                item={"id":f"e{len(explanation_items)+1:04d}","page":int(page.get("page") or pidx+1),"source_order":block_order,"context":ctx,"section":sec}
                explanation_items.append(item); detail["explanations"].append(item["id"])

    if progress_callback:progress_callback(25,f"Identified {len(action_items)} actions and {len(explanation_items)} explanations")
    narr_inputs=[{"id":x["id"],"action":x["action"],"context":_context(x["action"],x.get("context",""),x.get("page_text","")),"range":f"{MIN_WORDS[complexity(x['action'])]}-{MAX_WORDS[complexity(x['action'])]}"} for x in action_items]
    narr_map=_narrate(text_client,narr_inputs,narration_language)
    if progress_callback:progress_callback(45,"Generated contextual narration")

    scenes=[]; scene_by_id={}
    for x in action_items:
        page=next((p for p in pages if int(p.get("page") or 0)==x["page"]),{})
        page["action_bbox"]=x.get("bbox")
        grounding=ground_action(x["action"],page,assets,vision_client)
        shot=grounding.get("screenshot")
        narr=narr_map.get(x["id"]) or _fallback_narration(x["action"],_context(x["action"],x.get("context",""),x.get("page_text","")))
        urls=[]
        for src in (x["action"],x.get("context",""),page.get("urls",[])):
            for u in _urls(src):
                if u not in urls:urls.append(u)
        status=grounding.get("status","unresolved")
        interaction=interaction_for(x["action"],status in {"verified","review_required"})
        scene_id=len(scenes)+1
        scene={"id":scene_id,"scene_id":f"scene_{scene_id:05d}","section_id":0,"kind":"action","title":_remove_urls(x["action"])[:120],"source_step_number":x.get("source_step_number"),"source_substep":x.get("source_substep"),"source_page":x["page"],"source_order":x.get("source_order",0),"page_width":float(page.get("width") or 0),"page_height":float(page.get("height") or 0),"source_bbox":x.get("bbox"),"action":x["action"],"action_type":action_type(x["action"]),"target_query":grounding.get("query",""),"target_status":status,"target_review_required":bool(grounding.get("review_required")),"cursor":grounding.get("point") if status in {"verified","review_required"} else None,"cursor_path":grounding.get("points",[]) if status in {"verified","review_required"} else [],"cursor_enabled":status=="verified" and interaction!="none","cursor_source":grounding.get("source","none"),"cursor_confidence":float(grounding.get("confidence",0.0) or 0.0),"cursor_target_name":grounding.get("target_name",""),"cursor_bounding_box":grounding.get("bounding_box"),"target_debug":grounding.get("debug",{}),"interaction":interaction,"complexity":complexity(x["action"]),"narration":narr,"tts_narration":narr,"caption_text":narr,"caption":narr,"caption_urls":urls,"urls":urls,"source_context":_context(x["action"],x.get("context",""),x.get("page_text","")),"screenshot":shot.get("path") if isinstance(shot,dict) else None,"is_full_page":bool(shot and shot.get("is_full_page")),"screenshot_role":shot.get("visual_role") if shot else None,"evidence":{"candidate_screenshots":grounding.get("debug",{}).get("candidate_screenshots",[]),"target_method":grounding.get("source","none")}}
        scenes.append(scene); scene_by_id[scene_id]=scene

    for x in explanation_items:
        page=next((p for p in pages if int(p.get("page") or 0)==x["page"]),{})
        shot=next((a for a in assets if int(a.get("page") or -1)==x["page"] and a.get("is_full_page") and a.get("visual_role")!="decorative"),None)
        scene_id=len(scenes)+1; text=_remove_urls(x["context"])
        words=text.split()
        if len(words)>42:
            text=" ".join(words[:42]).rstrip(" .")+"."
        scenes.append({"id":scene_id,"scene_id":f"scene_{scene_id:05d}","section_id":0,"kind":"explanation","title":text[:120],"source_step_number":None,"source_substep":None,"source_page":x["page"],"source_order":x.get("source_order",0),"page_width":float(page.get("width") or 0),"page_height":float(page.get("height") or 0),"source_bbox":None,"action":"","action_type":"Follow","target_query":"","target_status":"not_applicable","target_review_required":False,"cursor":None,"cursor_enabled":False,"cursor_source":"none","cursor_confidence":0.0,"cursor_target_name":"","cursor_bounding_box":None,"target_debug":{},"interaction":"none","complexity":"simple","narration":text[:700].rstrip(" .")+".","tts_narration":text[:700].rstrip(" .")+".","caption_text":text[:700].rstrip(" .")+".","caption":text[:700].rstrip(" .")+".","caption_urls":_urls(x["context"]),"urls":_urls(x["context"]),"source_context":text,"screenshot":shot.get("path") if shot else None,"is_full_page":bool(shot and shot.get("is_full_page")),"screenshot_role":shot.get("visual_role") if shot else None,"evidence":{}})

    # Preserve source order: sort by source page, then original action insertion order,
    # with explanation blocks kept immediately around their page-derived actions.
    scenes.sort(key=lambda s:(int(s.get("source_page") or 0),int(s.get("source_order") or 0),0 if s.get("kind")=="action" else 1,int(s.get("source_substep") or 0),int(s.get("id") or 0)))
    for i,s in enumerate(scenes,1):s["id"]=i;s["scene_id"]=f"scene_{i:05d}"
    id_map={old.get("scene_id"):new["scene_id"] for old,new in zip(sorted(scenes,key=lambda s:s["id"]),scenes)}

    # Build section index using source section names from pages.
    section_order=list(sections.keys()); sec_idx={name:i+1 for i,name in enumerate(section_order)}
    for s in scenes:
        page=next((p for p in pages if int(p.get("page") or 0)==int(s.get("source_page") or -1)),{})
        name=clean_text(section_title(page)) or "Untitled section"
        s["section_id"]=sec_idx.get(name,1)
    sections_out=[]
    for name,i in sec_idx.items():
        related=[s["id"] for s in scenes if s.get("section_id")==i]
        acts=[s["title"] for s in scenes if s.get("section_id")==i and s.get("kind")=="action"]
        theories=[s["source_context"] for s in scenes if s.get("section_id")==i and s.get("kind")=="explanation"]
        sections_out.append({"id":i,"title":name,"step_ids":related,"action_bullets":acts[:20],"theory_bullets":theories[:8],"action_count":len(acts),"has_video":bool(related)})

    title=next((clean_text(p.get("heading")) for p in pages if clean_text(p.get("heading")) and not is_metadata_page(p)),None) or clean_text(data.get("title")) or Path(str(data.get("filename") or "")).stem or "Software Training Tutorial"
    description=next((s.get("source_context") for s in scenes if s.get("kind")=="explanation" and s.get("source_context")),"Guided software training generated from the document.")
    plan={"schema_version":SCHEMA_VERSION,"planner":"canonical_scene_compiler","title":title[:120],"description":description[:500],"narration_language":narration_language,"sections":sections_out,"steps":scenes,"scene_count":len(scenes),"action_count":sum(s.get("kind")=="action" for s in scenes),"total_pages":len(pages),"asset_count":len(assets),"publication_status":"draft","requires_trainer_review":any(s.get("target_review_required") for s in scenes),"model_config":{"text_model":text_client.spec.model if hasattr(text_client,"spec") else "configured","vision_model":vision_client.spec.model if hasattr(vision_client,"spec") else "configured"}}
    if progress_callback:progress_callback(90,f"Compiled {len(scenes)} canonical scenes")
    return plan
