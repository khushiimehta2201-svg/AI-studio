"""Evidence selection and UI target grounding.

Targeting is intentionally conservative:
1) identify likely evidence images,
2) use OCR/geometry where strong,
3) use author highlights only as hints,
4) escalate to the configured vision model when needed,
5) verify vision-only candidates or route them to trainer review.

A target is never invented merely because an action exists.
"""
from __future__ import annotations

import base64
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from io import BytesIO

from PIL import Image

_URL_RE=re.compile(r"https?://[^\s)\]>]+",re.I)
_VISION_CACHE: Dict[Tuple[str, Tuple[str, ...], str], List[Dict[str, Any]]] = {}

def clear_caches() -> None:
    _VISION_CACHE.clear()


def clean(value: Any)->str:return re.sub(r"\s+"," ",str(value or "")).strip()

def tokens(text:str)->List[str]:return [t.lower() for t in re.findall(r"[a-z0-9_]+",clean(text)) if len(t)>1]

def token_similarity(query:str,candidate:str)->float:
    q=tokens(query); c=tokens(candidate)
    if not q or not c:return 0.0
    overlap=sum(1 for x in q if any(x==y or x in y or y in x for y in c))/len(q)
    contains=1.0 if clean(query).lower() in clean(candidate).lower() or clean(candidate).lower() in clean(query).lower() else 0.0
    exact=1.0 if clean(query).lower()==clean(candidate).lower() else 0.0
    return min(1.0,0.60*overlap+0.25*contains+0.15*exact)


def action_type(action:str)->str:
    low=clean(action).lower()
    if low.startswith("press "):return "Press"
    if "double-click" in low:return "Double-click"
    if "right-click" in low:return "Right-click"
    for v,l in (("click","Click"),("select","Select"),("choose","Choose"),("enter","Enter"),("type","Type"),("open","Open"),("navigate","Navigate"),("visit","Visit"),("search","Search"),("save","Save"),("submit","Submit"),("create","Create"),("delete","Delete"),("expand","Expand"),("collapse","Collapse"),("drag","Drag"),("drop","Drop"),("fill","Fill"),("upload","Upload"),("download","Download")):
        if re.search(rf"\b{re.escape(v)}\b",low):return l
    return "Follow"


def interaction_for(action:str,has_target:bool)->str:
    if not has_target or action_type(action)=="Press":return "none"
    k=action_type(action)
    if k in {"Click","Double-click","Right-click","Select","Choose","Open","Save","Submit","Create","Delete","Expand","Collapse"}:
        return {"Double-click":"double_click","Right-click":"right_click"}.get(k,"click")
    if k in {"Enter","Type","Search","Fill"}:return "focus"
    if k in {"Drag","Drop"}:return "drag"
    return "none"


def target_queries(action:str)->List[str]:
    text=clean(_URL_RE.sub(" ",action)); text=re.sub(r"^\s*\d+(?:\.\d+)*[.)\-:]?\s*","",text)
    if re.match(r"^press\b",text,re.I):return []
    out=[]
    m=re.search(r"\bdrag\s+(?:the\s+)?(.+?)\s+to\s+(?:the\s+)?(.+?)(?:[.,;]|$)",text,re.I)
    if m:
        for q in (m.group(1),m.group(2)):
            q=re.sub(r"\b(button|field|textbox|input|dropdown|drop-down|combobox|menu|tab|link|list|checkbox|radio|option|control)\b"," ",q,flags=re.I);q=clean(q)
            if q:out.append(q)
        return out[:2]
    m=re.search(r"\b(?:select|choose)\s+(?:the\s+)?(.+?)\s+from\s+(?:the\s+)?(.+?)(?:[.,;]|$)",text,re.I)
    if m:
        for q in (m.group(1),m.group(2)):
            q=re.sub(r"\b(button|field|textbox|input|dropdown|drop-down|combobox|menu|tab|link|list|checkbox|radio|option)\b"," ",q,flags=re.I); q=clean(q)
            if q:out.append(q)
        return out[:3]
    m=re.search(r"\b(?:double-click|right-click|click|select|choose)\s+(?:on\s+)?(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+(?:to|for|so|and then|then|before|using|in|on|at|within|inside|from|of|under)\b|[.,;]|$)",text,re.I)
    if m:
        q=re.sub(r"\b(button|field|textbox|input|dropdown|drop-down|combobox|menu|tab|link|list|checkbox|radio|option)\b"," ",m.group(1),flags=re.I); q=clean(q)
        if q:out.append(q)
        return out[:3]
    m=re.search(r"\b(?:enter|type|fill)\s+.+?\s+(?:in|into)\s+(?:the\s+)?(.+?)(?:[.,;]|$)",text,re.I)
    if m:
        q=re.sub(r"\b(field|textbox|input)\b"," ",m.group(1),flags=re.I); q=clean(q)
        if q:out.append(q)
        return out[:3]
    m=re.search(r"\b(?:open|navigate|visit|launch|switch\s+to)\s+(?:to\s+)?(?:the\s+)?(.+?)(?:\s+(?:and|in|on|at|within|inside|from|under)\b|[.,;]|$)",text,re.I)
    if m:
        q=re.sub(r"\b(menu|tab|window|dialog)\b"," ",m.group(1),flags=re.I); q=clean(q)
        if q:out.append(q)
        return out[:3]
    for x in ("save","submit","delete","create","upload","download","apply","search","cancel","close","ok"):
        if re.match(rf"^{x}\b",text,re.I):return [x]
    return [text[:100]] if text else []


def _point_from_box(box:Sequence[float])->List[float]:return [(float(box[0])+float(box[2]))/2,(float(box[1])+float(box[3]))/2]

def _norm_point(point:Any,w:float,h:float)->Optional[List[float]]:
    if not isinstance(point,(list,tuple)) or len(point)<2:return None
    try:x,y=float(point[0]),float(point[1])
    except Exception:return None
    if x>1.5 or y>1.5:
        if w<=0 or h<=0:return None
        x,y=x/w,y/h
    if not(0<=x<=1 and 0<=y<=1):return None
    return [x,y]

def _norm_box(box:Any,w:float,h:float)->Optional[List[float]]:
    if not isinstance(box,(list,tuple)) or len(box)<4:return None
    try:v=[float(x) for x in box[:4]]
    except Exception:return None
    if max(v)>1.5:
        if w<=0 or h<=0:return None
        v=[v[0]/w,v[1]/h,v[2]/w,v[3]/h]
    if not(0<=v[0]<v[2]<=1 and 0<=v[1]<v[3]<=1):return None
    return v

@lru_cache(maxsize=256)
def _ocr_lines(path:str):
    try:
        import pytesseract
        with Image.open(path) as im:
            data=pytesseract.image_to_data(im.convert("RGB"),output_type=pytesseract.Output.DICT,config="--psm 11")
            width,height=im.size
        groups={}
        for i,text in enumerate(data.get("text",[])):
            text=clean(text)
            if not text:continue
            try:
                conf=float(data.get("conf",[0])[i]); x=float(data["left"][i]); y=float(data["top"][i]); w=float(data["width"][i]); h=float(data["height"][i]); key=(int(data.get("block_num",[0])[i]),int(data.get("par_num",[0])[i]),int(data.get("line_num",[0])[i]))
            except Exception:continue
            if conf<15:continue
            groups.setdefault(key,[]).append((text,conf,x,y,w,h))
        out=[]
        for items in groups.values():
            items.sort(key=lambda x:x[2]); x0=min(x[2] for x in items); y0=min(x[3] for x in items); x1=max(x[2]+x[4] for x in items); y1=max(x[3]+x[5] for x in items)
            out.append({"text":clean(" ".join(x[0] for x in items)),"conf":sum(x[1] for x in items)/len(items),"x0":x0,"y0":y0,"x1":x1,"y1":y1,"width":width,"height":height})
        return tuple(out)
    except Exception:return tuple()


def _ocr_candidates(query:str,path:str)->List[Dict[str,Any]]:
    out=[]
    for line in _ocr_lines(path):
        sim=token_similarity(query,line["text"])
        if sim<0.42:continue
        score=min(1.0,sim*0.80+(line["conf"]/100)*0.20)
        out.append({"point":[(line["x0"]+line["x1"])/(2*line["width"]),(line["y0"]+line["y1"])/(2*line["height"])],"bounding_box":[line["x0"]/line["width"],line["y0"]/line["height"],line["x1"]/line["width"],line["y1"]/line["height"]],"target_name":line["text"],"confidence":score,"source":"ocr"})
    return sorted(out,key=lambda x:x["confidence"],reverse=True)


def _near(point:Sequence[float],box:Sequence[float],radius:float=0.13)->bool:
    bx,by=_point_from_box(box); return ((point[0]-bx)**2+(point[1]-by)**2)**0.5<=radius


def _highlight_candidates(shot:Dict[str,Any])->List[Dict[str,Any]]:
    return [{"point":_point_from_box(b),"bounding_box":list(b),"target_name":"","confidence":0.60,"source":"highlight_only"} for b in (shot.get("annotation_boxes") or []) if isinstance(b,(list,tuple)) and len(b)>=4]


def select_screenshots(action:str,page:Dict[str,Any],screenshots:Sequence[Dict[str,Any]])->Tuple[List[Dict[str,Any]],List[Dict[str,Any]]]:
    queries=target_queries(action); page_no=int(page.get("page") or 1); action_bbox=page.get("action_bbox"); scored=[]
    page_window=max(0,int(os.getenv("EVIDENCE_PAGE_WINDOW","2")))
    for shot in screenshots:
        if not isinstance(shot,dict):continue
        role=str(shot.get("visual_role") or "unknown")
        if role in {"decorative","logo","metadata"}:continue
        path=str(shot.get("path") or "")
        if not path or not os.path.exists(path):continue
        sp=int(shot.get("page") or page_no); dist=abs(sp-page_no)
        if dist>page_window:continue
        score=0.44 if dist==0 else 0.10
        if role in {"screenshot_candidate","screenshot","ui_screenshot","full_page_ui"}:score+=0.22
        if shot.get("is_full_page"):score+=0.03
        ocr_score=0.0
        for q in queries:
            cs=_ocr_candidates(q,path)
            if cs:ocr_score=max(ocr_score,cs[0]["confidence"])
        score+=0.30*ocr_score
        nearby=clean(str(shot.get("nearby_text") or ""))
        if nearby and queries:
            nearby_sim=max(token_similarity(q,nearby) for q in queries)
            score+=0.10*nearby_sim
        if action_bbox and shot.get("page_rect"):
            try:
                r=shot["page_rect"]; acx=(float(action_bbox[0])+float(action_bbox[2]))/2; acy=(float(action_bbox[1])+float(action_bbox[3]))/2; scx=(float(r[0])+float(r[2]))/2; scy=(float(r[1])+float(r[3]))/2
                d=(abs(acx-scx)+abs(acy-scy))/max(1,float(page.get("width") or 1),float(page.get("height") or 1)); score+=max(0,0.08-d*0.03)
            except Exception:pass
        scored.append((score,shot,ocr_score))
    scored.sort(key=lambda x:x[0],reverse=True)
    return [x[1] for x in scored[:5]],[{"path":x[1].get("path"),"score":round(x[0],4),"ocr_score":round(x[2],4),"role":x[1].get("visual_role")} for x in scored[:5]]


def _vision_candidates(client,action:str,queries:List[str],path:str)->List[Dict[str,Any]]:
    cache_key=(str(Path(path).resolve()),tuple(queries),clean(action).lower())
    cached=_VISION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # Keep multimodal payloads bounded while preserving readable UI text.
    try:
        from PIL import Image
        with Image.open(path) as src:
            src=src.convert("RGB")
            src.thumbnail((1280,1280),Image.Resampling.LANCZOS)
            buf=BytesIO();src.save(buf,format="PNG",optimize=True);image=base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        image=base64.b64encode(Path(path).read_bytes()).decode("ascii")
    prompt=f"""
Ground the following software-training action to the actual application UI visible in the screenshot.
ACTION: {clean(action)}
POSSIBLE TARGET TEXT: {queries}

Return ONLY JSON:
{{"candidates":[{{"target_name":"visible UI label or concise control description","bounding_box":[x0,y0,x1,y1],"click_point":[x,y],"confidence":0.0,"evidence":"brief visible evidence"}}]}}

Rules: coordinates are normalized 0..1; return at most 3 candidates; candidates must be actual UI controls relevant to the action; do not return document headings, body text, captions, logos, page numbers, tables of contents, or decorative elements. Return an empty candidates array when the target is not visible.
""".strip()
    data=client.generate_json(prompt,image) or {}; out=[]
    for c in data.get("candidates",[]) if isinstance(data,dict) else []:
        if not isinstance(c,dict):continue
        pt=_norm_point(c.get("click_point"),1,1); box=_norm_box(c.get("bounding_box"),1,1)
        if pt is None:continue
        conf=float(c.get("confidence",0.0) or 0.0); name=clean(c.get("target_name")); sim=max([token_similarity(q,name) for q in queries] or [0])
        accept_conf=float(os.getenv("VISION_ACCEPT_CONFIDENCE","0.86"))
        if conf<accept_conf or (sim<0.25 and conf<0.95):continue
        if box and not(box[0]<=pt[0]<=box[2] and box[1]<=pt[1]<=box[3]):continue
        out.append({"point":pt,"bounding_box":box,"target_name":name,"confidence":conf,"source":"vision","evidence":clean(c.get("evidence")),"name_similarity":sim})
    result=sorted(out,key=lambda x:x["confidence"],reverse=True)[:3]
    _VISION_CACHE[cache_key]=result
    return result


def ground_action(action:str,page:Dict[str,Any],screenshots:Sequence[Dict[str,Any]],vision_client=None)->Dict[str,Any]:
    queries=target_queries(action); kind=action_type(action); interaction=interaction_for(action,bool(queries))
    if not queries:
        return {"status":"not_applicable","point":None,"points":[],"bounding_box":None,"bounding_boxes":[],"target_name":"","query":"","confidence":1.0,"source":"action_semantics","review_required":False,"screenshot":None,"debug":{}}
    candidates,debug_candidates=select_screenshots(action,page,screenshots)
    base={"status":"unresolved","point":None,"points":[],"bounding_box":None,"bounding_boxes":[],"target_name":"","query":queries[0],"confidence":0.0,"source":"none","review_required":True,"screenshot":candidates[0] if candidates else None,"debug":{"candidate_screenshots":debug_candidates,"queries":queries}}
    if not candidates:return base

    for shot in candidates:
        path=str(shot.get("path") or "")
        ocr=[]
        for q in queries:
            ocr.extend([{**c,"query":q} for c in _ocr_candidates(q,path)])
        full_page = bool(shot.get("is_full_page"))
        if full_page and page.get("action_bbox") and page.get("width") and page.get("height"):
            try:
                ax0,ay0,ax1,ay1=[float(x) for x in page.get("action_bbox")]
                pw,ph=float(page.get("width")),float(page.get("height"))
                source_box=[ax0/pw,ay0/ph,ax1/pw,ay1/ph]
                ocr=[c for c in ocr if not _near(c["point"],source_box,0.05)]
            except Exception:
                pass
        ocr.sort(key=lambda x:x["confidence"],reverse=True)
        if kind == "Drag" and len(queries) >= 2:
            per_query=[]
            for q in queries[:2]:
                qs=[{**c,"query":q} for c in _ocr_candidates(q,path)]
                qs.sort(key=lambda x:x["confidence"],reverse=True)
                if not qs:
                    per_query=[]; break
                best_q=qs[0]; margin_q=best_q["confidence"]-(qs[1]["confidence"] if len(qs)>1 else 0.0)
                full_ok_q=not shot.get("is_full_page") or page.get("visual_page_role")=="ui_page"
                if not(full_ok_q and best_q["confidence"]>=float(os.getenv("OCR_TARGET_CONFIDENCE","0.90")) and margin_q>=float(os.getenv("TARGET_AMBIGUITY_MARGIN","0.10"))):
                    per_query=[]; break
                per_query.append(best_q)
            if len(per_query)==2:
                base.update({"status":"verified","point":per_query[1]["point"],"points":[per_query[0]["point"],per_query[1]["point"]],"bounding_box":per_query[1]["bounding_box"],"bounding_boxes":[per_query[0]["bounding_box"],per_query[1]["bounding_box"]],"target_name":per_query[1]["target_name"],"query":queries[1],"confidence":min(x["confidence"] for x in per_query),"source":"ocr","review_required":False,"screenshot":shot})
                base["debug"].update({"method":"ocr_drag","drag_queries":queries})
                return base

        if ocr:
            best=ocr[0]; second=ocr[1]["confidence"] if len(ocr)>1 else 0.0; margin=best["confidence"]-second
            # Full-page PDF text is ambiguous between document prose and UI. We only
            # accept OCR on a full-page asset if the page has been explicitly marked UI-like.
            full_ok=not shot.get("is_full_page") or page.get("visual_page_role")=="ui_page"
            if full_ok and best["confidence"]>=float(os.getenv("OCR_TARGET_CONFIDENCE","0.90")) and margin>=float(os.getenv("TARGET_AMBIGUITY_MARGIN","0.10")):
                ab=page.get("action_bbox")
                if shot.get("is_full_page") and ab and page.get("width") and page.get("height"):
                    try:
                        ax0,ay0,ax1,ay1=[float(x) for x in ab]; pw=float(page["width"]); ph=float(page["height"]); bx=[ax0/pw,ay0/ph,ax1/pw,ay1/ph]
                        if _near(best["point"],bx,0.035):
                            continue
                    except Exception:pass
                base.update({"status":"verified","point":best["point"],"points":[best["point"]],"bounding_box":best["bounding_box"],"bounding_boxes":[best["bounding_box"]],"target_name":best["target_name"],"query":best["query"],"confidence":best["confidence"],"source":"ocr","review_required":False,"screenshot":shot})
                base["debug"].update({"method":"ocr","ambiguity_margin":round(margin,3)})
                return base

        # Highlight is useful even without OCR, but it remains review-required.
        anns=_highlight_candidates(shot)
        if len(anns)==1 and kind in {"Click","Double-click","Right-click","Select","Choose","Open","Save","Submit","Create","Delete","Expand","Collapse"}:
            base.update({**anns[0],"status":"review_required","review_required":True,"query":queries[0],"points":[anns[0]["point"]],"bounding_boxes":[anns[0]["bounding_box"]],"screenshot":shot})
            base["debug"]["method"]="highlight_only"
            return base

    if vision_client:
        # Give vision the top two screenshot candidates when their evidence scores are close.
        chosen=candidates[:2] if len(candidates)>1 and len(debug_candidates)>1 and abs(debug_candidates[0]["score"]-debug_candidates[1]["score"])<0.12 else candidates[:1]
        vision_results=[]
        for shot in chosen:
            path=str(shot.get("path") or "")
            try:
                for cand in _vision_candidates(vision_client,action,queries,path):
                    vision_results.append({**cand,"screenshot":shot})
            except Exception as exc:
                base["debug"].setdefault("vision_errors",[]).append(str(exc))
        vision_results.sort(key=lambda x:x["confidence"]+0.05*x.get("name_similarity",0),reverse=True)
        if vision_results:
            best=vision_results[0]; second=vision_results[1] if len(vision_results)>1 else None
            margin=float(best["confidence"])-(float(second["confidence"]) if second else 0.0)
            if second and margin<float(os.getenv("VISION_AMBIGUITY_MARGIN","0.08")):
                base["status"]="unresolved"; base["debug"]["vision_ambiguous"]=[{k:v for k,v in x.items() if k!="screenshot"} for x in vision_results[:3]]
                return base
            review=True
            if os.getenv("VISION_SECOND_PASS","true").lower() in {"1","true","yes","on"}:
                try:
                    try:
                        with Image.open(best["screenshot"]["path"]) as src:
                            src=src.convert("RGB"); src.thumbnail((1280,1280),Image.Resampling.LANCZOS); buf=BytesIO(); src.save(buf,format="PNG",optimize=True); image=base64.b64encode(buf.getvalue()).decode("ascii")
                    except Exception:
                        image=base64.b64encode(Path(best["screenshot"]["path"]).read_bytes()).decode("ascii")
                    verify=f"""Verify this target candidate for a software-training action. ACTION: {clean(action)} TARGET: {clean(best['target_name'] or queries[0])} POINT: {best['point']}. Return ONLY JSON {{\"confirmed\":true|false,\"reason\":\"brief visual evidence\"}}. Confirm only when the point is on the actual UI control; reject document text, headings, captions, logos and decorative regions."""
                    check=vision_client.generate_json(verify,image) or {}
                    base["debug"]["second_pass"]=check
                    review=not bool(check.get("confirmed"))
                except Exception as exc:
                    base["debug"]["second_pass_error"]=str(exc)
            base.update({"status":"review_required" if review else "verified","point":best["point"],"points":[best["point"]],"bounding_box":best["bounding_box"],"bounding_boxes":[best["bounding_box"]],"target_name":best["target_name"] or queries[0],"query":queries[0],"confidence":best["confidence"],"source":"vision_verified" if not review else "vision_only","review_required":review,"screenshot":best["screenshot"]})
            return base
    return base
