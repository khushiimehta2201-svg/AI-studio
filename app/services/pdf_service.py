"""PDF ingestion with provenance-rich page and visual asset records."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import cv2
import fitz
import numpy as np
from PIL import Image

_URL_RE=re.compile(r"(?<![\w@])(?:https?://|www\.)[^\s<>\]\[\"')]+",re.I)


def _clean(text: str)->str:
    return " ".join((text or "").replace("\r","\n").split())


def _urls(text: str)->List[str]:
    out=[]
    for u in _URL_RE.findall(str(text or "")):
        u=u.rstrip(".,;:!?)]}")
        if u and u not in out: out.append(u)
    return out


def _annotation_boxes(path: Path)->List[List[float]]:
    """Extract colored marks as non-authoritative target hints."""
    try:
        img=cv2.imread(str(path))
        if img is None:return []
        h,w=img.shape[:2]
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        masks=[
            cv2.inRange(hsv,np.array([0,80,80]),np.array([12,255,255])),
            cv2.inRange(hsv,np.array([165,80,80]),np.array([180,255,255])),
            cv2.inRange(hsv,np.array([11,80,80]),np.array([40,255,255])),
        ]
        mask=masks[0]|masks[1]|masks[2]
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxes=[]
        area=float(max(1,w*h))
        for c in contours:
            x,y,bw,bh=cv2.boundingRect(c); frac=(bw*bh)/area
            if 0.0008<=frac<=0.45:
                boxes.append([x/w,y/h,(x+bw)/w,(y+bh)/h])
        boxes.sort(key=lambda b:(b[1],b[0]))
        return boxes[:24]
    except Exception:
        return []


def _hash(path: Path,size:int=16)->str:
    try:
        with Image.open(path) as im:
            a=np.asarray(im.convert("L").resize((size,size)),dtype=np.float32)
        m=float(a.mean())
        return "".join("1" if x>=m else "0" for x in a.flatten())
    except Exception:return ""


def _hamming(a:str,b:str)->int:
    if not a or not b or len(a)!=len(b):return 999999
    return sum(x!=y for x,y in zip(a,b))


def _page_ocr(path: Path) -> str:
    try:
        import pytesseract
        with Image.open(path) as im:
            return _clean(pytesseract.image_to_string(im, config="--psm 6") or "")
    except Exception:
        return ""


def _nearby_text(page: Dict[str, Any], rect: Any) -> str:
    words = page.get("words") or []
    if not rect or not words:
        return ""
    try:
        rx0, ry0, rx1, ry1 = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
    except Exception:
        return ""
    selected=[]
    for w in words:
        try:
            x0,y0,x1,y1=float(w["x0"]),float(w["y0"]),float(w["x1"]),float(w["y1"])
            if y1 < ry0 - 2 or y0 > ry1 + max(18.0, (ry1-ry0)*0.9):
                continue
            if x1 < rx0 - 80 or x0 > rx1 + 80:
                continue
            selected.append(w)
        except Exception:
            continue
    selected.sort(key=lambda w:(float(w.get("y0",0)),float(w.get("x0",0))))
    return _clean(" ".join(str(w.get("text", "")) for w in selected))[:500]



def _page_visual_role(text: str) -> str:
    low = (text or "").lower()
    ui_terms = {"file","edit","view","help","save","cancel","create","search","settings","ok","apply","select","item","type","name","description","browser","workspace","toolbar","menu","dialog","tab"}
    words = set(re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", low))
    score = len(words & ui_terms)
    actionish = len(re.findall(r"\b(?:click|select|choose|enter|type|press|open|save|create)\b", low))
    if score >= 3 and actionish >= 1:
        return "ui_page"
    return "document_page"


def _role(area:float,repeated:int,y0:float,page_h:float,ocr:str)->str:
    low=ocr.lower()
    edge=y0<=page_h*0.15 or y0>=page_h*0.88
    ui_words=sum(1 for x in re.findall(r"\b[a-z][a-z0-9_-]{2,}\b",low) if x in {"file","edit","view","help","save","cancel","create","search","settings","ok","apply","select","item","type","name","description","new","close","browser","workspace","toolbar"})
    if repeated>=3 and area<0.22 and ui_words<2:return "decorative"
    if repeated>=2 and edge and area<0.32 and ui_words<2:return "decorative"
    if ui_words>=2 and area>=0.012:return "screenshot_candidate"
    if area>=0.08:return "image_candidate"
    if area<0.01 and edge:return "decorative"
    return "unknown"


def process_pdf(pdf_path:str, output_dir:str|Path)->Dict[str,Any]:
    pdf=Path(pdf_path); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    sd=out/"screenshots"; sd.mkdir(parents=True,exist_ok=True)
    doc=fitz.open(str(pdf)); pages=[]; assets=[]; hashes=defaultdict(list)
    try:
        for pidx,page in enumerate(doc):
            pno=pidx+1; rect=page.rect; text=page.get_text("text") or ""
            words=[]
            for item in page.get_text("words") or []:
                if len(item)>=5 and str(item[4]).strip():
                    words.append({"x0":float(item[0]),"y0":float(item[1]),"x1":float(item[2]),"y1":float(item[3]),"text":str(item[4])})
            full=sd/f"page_{pno:04d}_full.png"
            try:
                pix=page.get_pixmap(dpi=150,alpha=False); pix.save(str(full))
            except Exception: full=None
            urls=_urls(text)
            try:
                links=page.get_links() or []
            except Exception: links=[]
            for link in links:
                uri=_clean(str(link.get("uri") or "")) if isinstance(link,dict) else ""
                if uri.lower().startswith(("http://","https://","www.")) and uri not in urls:urls.append(uri)
            text_source = "pdf_text"
            ocr_text = ""
            if len((text or "").split()) < 8 and full is not None:
                ocr_text = _page_ocr(full)
                if ocr_text:
                    text = ocr_text
                    text_source = "ocr"
                    words = words or [{"x0":0.0,"y0":0.0,"x1":float(rect.width),"y1":float(rect.height),"text":ocr_text}]
            pages.append({"page":pno,"text":text,"text_source":text_source,"ocr_text":ocr_text,"heading":next((x.strip() for x in text.splitlines() if x.strip()),None),"width":float(rect.width),"height":float(rect.height),"words":words,"urls":urls,"full_page_path":str(full) if full else None,"visual_page_role":_page_visual_role(text)})
            for i,info in enumerate(page.get_images(full=True) or [],1):
                xref=int(info[0])
                try:
                    extracted=doc.extract_image(xref); data=extracted["image"]; ext=extracted.get("ext","png")
                    path=sd/f"page_{pno:04d}_image_{i:03d}.{ext}"; path.write_bytes(data)
                    rects=page.get_image_rects(xref) or []
                    if not rects:
                        rects=[None]
                    ph=_hash(path); base_idx=len(assets)
                    hashes.setdefault(ph, []) if ph else None
                    for ridx,r in enumerate(rects,1):
                        frac=(r.get_area()/rect.get_area()) if r else 0.0
                        idx=len(assets)
                        if ph:hashes[ph].append(idx)
                        assets.append({"asset_id":f"asset_{pno}_{i}_{ridx}","page":pno,"screenshot_index":i,"path":str(path.resolve()),"is_full_page":False,"page_rect":[float(r.x0),float(r.y0),float(r.x1),float(r.y1)] if r else None,"area_fraction":float(frac),"visual_role":"unknown","phash":ph,"annotation_boxes":_annotation_boxes(path),"nearby_text":_nearby_text(pages[-1], [float(r.x0),float(r.y0),float(r.x1),float(r.y1)] if r else None)})
                except Exception as exc: print(f"[PDF] image extract failed p={pno} i={i}: {exc}")
        # classify embedded assets after repeated hashes are known
        for a in assets:
            p=pages[a["page"]-1]; rect=a.get("page_rect") or [0,0,p["width"],p["height"]]
            try:
                with Image.open(a["path"]) as im:
                    # OCR preview is deliberately cached, but only for asset classification.
                    import pytesseract
                    txt=pytesseract.image_to_string(im,config="--psm 11") or ""
            except Exception: txt=""
            a["ocr_preview"]=_clean(txt)[:600]
            a["visual_role"]=_role(float(a.get("area_fraction",0)),len(hashes.get(a.get("phash",""),[])),float(rect[1]),float(p["height"]),a["ocr_preview"])
        for p in pages:
            if p.get("full_page_path"):
                assets.append({"asset_id":f"page_{p['page']}_full","page":p["page"],"screenshot_index":0,"path":p["full_page_path"],"is_full_page":True,"page_rect":[0,0,p["width"],p["height"]],"area_fraction":1.0,"visual_role":"full_page","annotation_boxes":_annotation_boxes(Path(p["full_page_path"]))})
    finally: doc.close()
    return {"pages":pages,"screenshots":assets,"page_count":len(pages),"screenshot_count":len(assets),"filename":pdf.name}

extract_pdf_package=process_pdf
