"""Document structure and procedure extraction.

Design goals:
- metadata/ToC/index/logo-like document furniture never becomes a tutorial action;
- source numbering is preserved separately from rendered scene numbering;
- wrapped PDF lines stay attached to the correct instruction;
- unnumbered procedures can be semantically recovered by the local text model;
- explanation blocks remain explanations instead of being forced into actions.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

ACTION_VERBS = (
    "click", "select", "choose", "open", "enter", "type", "press", "add",
    "create", "delete", "save", "submit", "upload", "download", "drag", "drop",
    "navigate", "login", "log in", "search", "filter", "expand", "collapse",
    "double-click", "right-click", "launch", "visit", "switch", "fill", "apply",
    "remove", "check", "uncheck", "clear", "confirm", "cancel", "rename",
)
_ACTION_RE = re.compile(r"\b(?:" + "|".join(re.escape(v) for v in ACTION_VERBS) + r")\b", re.I)
_STEP_RE = re.compile(r"^\s*((?:\d+\.)+\d*|\d+)[\.)\-:]?\s+(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*•▪◦]\s+(.*)$")
_LETTER_STEP_RE = re.compile(r"^\s*[A-Za-z][\.)]\s+(.*)$")
_METADATA_WORDS = re.compile(
    r"\b(?:table of contents|contents|index|list of figures|list of tables|"
    r"author|authors|version|revision|document history|copyright|"
    r"confidential|confidentiality|glossary|references|bibliography|appendix)\b",
    re.I,
)


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_number(line: str) -> Optional[str]:
    m = _STEP_RE.match(line or "")
    return clean_text(m.group(1)).rstrip(".") if m else None


def line_marker(line: str) -> Tuple[Optional[str], str]:
    text = clean_text(line)
    m = _STEP_RE.match(text)
    if m:
        return clean_text(m.group(1)).rstrip("."), clean_text(m.group(2))
    m = _BULLET_RE.match(text)
    if m:
        return "bullet", clean_text(m.group(1))
    m = _LETTER_STEP_RE.match(text)
    if m:
        return text[:2], clean_text(m.group(1))
    return None, text


def is_likely_toc_line(line: str) -> bool:
    text = clean_text(line)
    if not text:
        return False
    if re.search(r"\.{3,}\s*\d+\s*$", text):
        return True
    if re.search(r"\s{2,}\d+\s*$", text) and len(text.split()) >= 3:
        return True
    return False


def is_metadata_page(page: Dict[str, Any]) -> bool:
    text = str(page.get("text") or "")
    lines = [clean_text(x) for x in text.splitlines() if clean_text(x)]
    heading = clean_text(page.get("heading"))
    head_blob = " ".join(lines[:15])
    if heading and re.fullmatch(r"(?:contents|table of contents|index|list of figures|list of tables)", heading, re.I):
        return True
    toc = sum(is_likely_toc_line(x) for x in lines[:60])
    meta_hits = len(_METADATA_WORDS.findall(head_blob))
    action_signal = sum(is_action_line(x) for x in lines[:60])
    # Metadata terms often appear in running headers/footers of otherwise valid
    # procedure pages. Only classify by metadata density when there is no
    # convincing action signal on the page.
    return toc >= 4 or (meta_hits >= 3 and action_signal == 0)


def is_action_line(line: str) -> bool:
    marker, body = line_marker(line)
    body = clean_text(body)
    if not body:
        return False
    low = body.lower()
    if re.match(r"^(the|this|that|these|those|when|after|before|note|overview|description|purpose|result|because|once|while)\b", low):
        return False
    imperative = re.compile(r"^(?:please\s+)?(?:" + "|".join(re.escape(v) for v in ACTION_VERBS) + r")\b", re.I)
    sequencing = re.compile(r"^(?:now|then|next|finally|first|after that)\s+(?:please\s+)?(?:" + "|".join(re.escape(v) for v in ACTION_VERBS) + r")\b", re.I)
    if imperative.match(body) or sequencing.match(body):
        return True
    # A short lead-in before an imperative is useful for natural procedure prose.
    m = _ACTION_RE.search(body)
    if m and m.start() <= 32:
        prefix = body[:m.start()].strip(" ,:-")
        if 0 < len(prefix.split()) <= 6 and not re.match(r"^(the|this|that|the application|the system|the user|users|you|it|once|when|after|before)\b", prefix, re.I):
            return True
    # Numbering alone is not sufficient: "1. The user can click..." is explanatory.
    return False


def looks_like_continuation(previous: str, current: str) -> bool:
    previous, current = clean_text(previous), clean_text(current)
    if not previous or not current:
        return False
    if re.search(r"[,/:\-]$", previous):
        return True
    if re.search(r"[.!?;:]$", previous):
        return False
    first = current.split()[0].lower().strip(".,;:")
    if first in {"the", "this", "that", "these", "those", "it", "they", "when", "after", "note"}:
        return False
    return current[:1].islower() or len(current.split()) <= 7


def _find_line_bbox(page: Dict[str, Any], line: str) -> Optional[List[float]]:
    words = page.get("words") or []
    if not isinstance(words, list):
        return None
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", line) if len(t) > 1][:12]
    if not tokens:
        return None
    parsed = []
    for item in words:
        if isinstance(item, dict) and item.get("text"):
            try:
                parsed.append({"text": clean_text(item["text"]), "x0": float(item["x0"]), "y0": float(item["y0"]), "x1": float(item["x1"]), "y1": float(item["y1"])})
            except Exception:
                pass
    for i, w in enumerate(parsed):
        if w["text"].lower().strip(".,;:()[]") != tokens[0]:
            continue
        row=[w]
        j=i+1
        for tok in tokens[1:]:
            found=None
            while j < min(len(parsed), i+18):
                nxt=parsed[j]; j+=1
                if abs(nxt["y0"]-w["y0"]) <= max(8,(w["y1"]-w["y0"])*1.5) and nxt["text"].lower().strip(".,;:()[]") == tok:
                    found=nxt; break
            if found is None:
                break
            row.append(found)
        if len(row) >= max(1, int(len(tokens)*0.55)):
            return [min(x["x0"] for x in row), min(x["y0"] for x in row), max(x["x1"] for x in row), max(x["y1"] for x in row)]
    return None


def extract_instruction_blocks(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    if is_metadata_page(page):
        return []
    lines=[clean_text(x) for x in str(page.get("text") or "").splitlines() if clean_text(x)]
    if not lines:
        return []
    blocks=[]; current=[]
    for line in lines:
        marker,_=line_marker(line)
        if current and (marker is not None or (is_action_line(line) and not looks_like_continuation(current[-1],line))):
            blocks.append(current); current=[line]
        else:
            current.append(line)
    if current: blocks.append(current)

    result=[]
    for block in blocks:
        idx=next((i for i,l in enumerate(block) if is_action_line(l)),None)
        if idx is None:
            text=clean_text(" ".join(block))
            if text and not is_likely_toc_line(text):
                result.append({"action":None,"context":text,"lines":block,"step_number":None,"bbox":None,"confidence":0.5})
            continue
        action_lines=[block[idx]]; prev=block[idx]
        for line in block[idx+1:]:
            if looks_like_continuation(prev,line):
                action_lines.append(line); prev=line
            else:
                break
        action=clean_text(" ".join(action_lines))
        context=clean_text(" ".join(block[:idx]+block[idx+len(action_lines):]))
        marker,_=line_marker(block[idx])
        result.append({"action":action,"context":context,"lines":block,"step_number":marker if marker not in {None,"bullet"} else None,"bbox":_find_line_bbox(page,action),"confidence":0.92 if marker else 0.78})
    return result


def split_compound_action(action: str) -> List[str]:
    text=clean_text(action)
    if not text: return []
    chunks=[]
    action_starters = r"(?:click|double-click|right-click|select|choose|open|enter|type|press|create|save|submit|upload|download|drag|drop|navigate|launch|search|fill|apply|remove|check|uncheck|clear|confirm|cancel|rename|expand|collapse|switch)"
    for pattern in (
        r"\s+(?:and then|then|after that)\s+",
        r"\s*;\s*",
        rf"\s+and\s+(?={action_starters}\b)",
    ):
        next_chunks=[]
        for chunk in (chunks or [text]):
            parts=re.split(pattern,chunk,flags=re.I)
            next_chunks.extend(clean_text(p) for p in parts if clean_text(p))
        chunks=next_chunks
    if len(chunks)>1 and all(is_action_line(x) for x in chunks):
        return chunks
    return [text]


def section_title(page: Dict[str, Any]) -> str:
    heading=clean_text(page.get("heading"))
    if heading and not is_metadata_page(page) and not is_action_line(heading) and not is_likely_toc_line(heading):
        return heading[:120]
    for line in str(page.get("text") or "").splitlines():
        line=clean_text(line)
        if line and len(line)<=120 and not is_action_line(line) and not is_likely_toc_line(line) and not _METADATA_WORDS.search(line):
            return line[:120]
    return "Untitled section"


def has_procedure_signal(pages: List[Dict[str, Any]]) -> bool:
    for page in pages:
        if is_metadata_page(page): continue
        if extract_instruction_blocks(page):
            if any(b.get("action") for b in extract_instruction_blocks(page)):
                return True
    return False
