import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import pymupdf

MIN_IMAGE_WIDTH = 140
MIN_IMAGE_HEIGHT = 70
MIN_IMAGE_AREA_FRACTION = 0.02
SCREENSHOT_DPI = 150
MERGE_GAP = 20.0

METADATA_PATTERNS = [
    r"document\s+change\s+record",
    r"revision\s+history",
    r"record\s+of\s+revisions",
    r"change\s+log",
    r"document\s+history",
    r"version\s+control",
    r"document\s+control",
    r"distribution\s+list",
    r"confidentiality\s+notice",
    r"all\s+rights\s+reserved",
    r"sign-off\s+sheet",
]

METADATA_TABLE_HEADERS = {
    "version", "rev", "revision", "date", "author", "author(s)",
    "summary of changes", "description of change", "change description",
    "approved by", "reviewed by", "changed by"
}

ACTION_VERBS = (
    "open", "click", "select", "enter", "type", "browse", "launch",
    "choose", "apply", "create", "save", "run", "fill", "search",
    "locate", "expand", "collapse", "double click", "right click", "drag", "press",
    "navigate", "check", "uncheck", "highlight", "log in", "login"
)


def _sanitize_ascii(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\u2022": "-", "\u2023": "-", "\u25e6": "-", "\u2043": "-", "\u2219": "-",
        "\uf0b7": "-", "\uf0a7": "-", "\u25aa": "-", "\u25cf": "-", "\u2013": "-",
        "\u2014": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\t": " ", "\xa0": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return "".join(c if ord(c) < 128 else " " for c in text)


def _clean_title(text: str) -> str:
    if not text:
        return "System Overview"
    t = _sanitize_ascii(text.splitlines()[0])
    t = re.sub(r"^\s*(?:(?:step\s*)?\d+(?:[\.\)]\d+)*[\.\)]?|[A-Za-z][\.\)])\s*", "", t, flags=re.I)
    t = re.sub(r"\bpage\s+\d+\b", "", t, flags=re.I)
    t = re.sub(r"[\.\:\-]+$", "", t).strip()
    return t[:55] if t else "System Overview"


def _is_metadata_text(text: str) -> bool:
    low = text.lower()
    return any(re.search(pat, low) for pat in METADATA_PATTERNS)


def _is_metadata_table(headers: List[str]) -> bool:
    norm = {re.sub(r"[^a-z0-9]", "", h.lower()) for h in headers if h}
    matches = norm & {re.sub(r"[^a-z0-9]", "", x) for x in METADATA_TABLE_HEADERS}
    return len(matches) >= 2


def _is_toc_page(text: str) -> bool:
    low = text.lower()
    if re.search(r"\b(table\s+of\s+contents|contents|agenda|table\s+des\s+matières)\b", low):
        return True
    dot_leader_lines = len(re.findall(r"\.{3,}\s*\d+", text))
    numbered_toc_lines = len(re.findall(r"^\s*\d+[\.\)]\s+[A-Za-z]", text, re.MULTILINE))
    return (dot_leader_lines >= 3 or numbered_toc_lines >= 3) and len(text.split()) < 350


def _extract_main_toc_topics(text: str) -> List[str]:
    topics = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        line_clean = _sanitize_ascii(line)
        m = re.match(r"^\s*(?:\d+[\.\)]|\d+\s*[-–]|\d+)\s*([A-Za-z\s]{3,45})", line_clean)
        if not m:
            m = re.match(r"^([A-Z][A-Za-z\s]{3,40})\s*\.{2,}\s*\d+", line_clean)

        if m:
            title = m.group(1).strip()
            low = title.lower()
            if low not in {"contents", "table of contents", "page", "revision history", "document control"} and len(title) > 3:
                clean_topic = re.sub(r"[\.\d\s]+$", "", title).strip()
                if clean_topic and clean_topic not in topics:
                    topics.append(clean_topic)

    return topics[:6] if topics else ["Module Overview", "Prerequisites & Concept Rules", "Step-by-Step Procedure", "Validation & Verification"]


def _extract_tables_from_page(page: pymupdf.Page) -> List[Dict[str, Any]]:
    valid_tables = []
    try:
        tabs = page.find_tables()
        for tab in tabs:
            df = tab.extract()
            if df and len(df) >= 2:
                headers = [_sanitize_ascii(str(c or "")).strip() for c in df[0]]
                if _is_metadata_table(headers):
                    continue
                rows = [[_sanitize_ascii(str(c or "")).strip() for c in r] for r in df[1:6]]
                if any(headers) and len(headers) >= 2:
                    valid_tables.append({"headers": headers, "rows": rows})
    except Exception:
        pass
    return valid_tables


def _find_keyword_in_page_crop(page: pymupdf.Page, crop_rect: Tuple[float, float, float, float], action_text: str) -> Optional[Tuple[float, float]]:
    """Finds the keyword's exact bounding box within the cropped screenshot region."""
    try:
        # Extract meaningful UI keywords from action string
        words = re.findall(r"[A-Za-z0-9_]{3,}", action_text)
        stopwords = {"click", "select", "enter", "type", "press", "open", "button", "from", "with", "into", "then", "next"}
        keywords = [w for w in words if w.lower() not in stopwords]

        cx0, cy0, cx1, cy1 = crop_rect
        cw, ch = (cx1 - cx0), (cy1 - cy0)
        if cw <= 0 or ch <= 0:
            return None

        for kw in keywords[:3]:
            rects = page.search_for(kw)
            for r in rects:
                # Check if text rect is inside this screenshot's crop area
                if cx0 <= r.x0 <= cx1 and cy0 <= r.y0 <= cy1:
                    norm_x = (r.x0 + r.x1) / 2.0 - cx0
                    norm_y = (r.y0 + r.y1) / 2.0 - cy0
                    return (max(0.08, min(0.92, norm_x / cw)), max(0.08, min(0.92, norm_y / ch)))
    except Exception:
        pass
    return None


def _detect_ui_targets_on_image(img_bgr: np.ndarray) -> List[Tuple[float, float]]:
    """Detects interactive UI controls and callout boxes, ignoring corner/border noise."""
    if img_bgr is None:
        return []
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    m1 = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([165, 90, 90]), np.array([180, 255, 255]))
    m3 = cv2.inRange(hsv, np.array([11, 100, 100]), np.array([26, 255, 255]))
    mask = m1 | m2 | m3

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    total_area = float(w * h)

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        frac = area / total_area
        # Exclude boxes right on the bottom-right edge (scrollbars/grips)
        if (x + bw) >= (w - 8) and (y + bh) >= (h - 8):
            continue
        if 0.0004 <= frac <= 0.45:
            aspect = bw / float(bh)
            is_badge = (frac < 0.010 and 0.7 <= aspect <= 1.4)
            boxes.append((x, y, bw, bh, is_badge))

    boxes.sort(key=lambda b: (b[1], b[0]))
    targets = []
    for (x, y, bw, bh, is_badge) in boxes:
        if is_badge:
            tx = min(w - 20, x + bw + int(bw * 1.5)) / float(w)
            ty = (y + bh / 2.0) / float(h)
        else:
            tx = (x + bw / 2.0) / float(w)
            ty = (y + bh / 2.0) / float(h)
        # Clamp to avoid extreme edges
        targets.append((max(0.08, min(0.92, tx)), max(0.08, min(0.92, ty))))
    return targets


def _extract_page_screenshots(page: pymupdf.Page, page_no: int, output_dir: Path, xref_counts: Dict[int, int]) -> List[Dict[str, Any]]:
    page_rect = page.rect
    image_list = page.get_images(full=True) or []
    valid_rects = []

    for img_info in image_list:
        xref = int(img_info[0])
        pw, ph = int(img_info[2] or 0), int(img_info[3] or 0)
        if pw < MIN_IMAGE_WIDTH or ph < MIN_IMAGE_HEIGHT or xref_counts.get(xref, 1) >= 3:
            continue
        rects = page.get_image_rects(xref) or []
        for r in rects:
            if r.get_area() / page_rect.get_area() >= MIN_IMAGE_AREA_FRACTION:
                valid_rects.append(r)

    clusters = []
    for r in sorted(valid_rects, key=lambda x: (x.y0, x.x0)):
        matched = False
        for c in clusters:
            if any(max(0.0, max(r.x0, o.x0) - min(r.x1, o.x1)) <= MERGE_GAP and
                   max(0.0, max(r.y0, o.y0) - min(r.y1, o.y1)) <= MERGE_GAP for o in c):
                c.append(r)
                matched = True
                break
        if not matched:
            clusters.append([r])

    shots = []
    for idx, c in enumerate(clusters[:2]):
        bbox = pymupdf.Rect(c[0])
        for item in c[1:]:
            bbox |= item
        pad = 4.0
        clip = pymupdf.Rect(
            max(page_rect.x0, bbox.x0 - pad),
            max(page_rect.y0, bbox.y0 - pad),
            min(page_rect.x1, bbox.x1 + pad),
            min(page_rect.y1, bbox.y1 + pad),
        )
        pix = page.get_pixmap(dpi=SCREENSHOT_DPI, clip=clip)
        img_path = output_dir / f"page_{page_no:03d}_shot_{idx+1:02d}.png"
        pix.save(str(img_path))
        shots.append({
            "path": str(img_path.resolve()),
            "crop_rect": (clip.x0, clip.y0, clip.x1, clip.y1),
            "y_pos": float(clip.y0),
        })
    return sorted(shots, key=lambda s: s["y_pos"])


def process_pdf(pdf_path: str, job_dir: str) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    job_dir = Path(job_dir)
    screenshots_dir = job_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    scenes_data = []
    pages_data = []
    all_screenshots = []

    xref_counts = {}
    for page in doc:
        for img in page.get_images(full=True) or []:
            xref = int(img[0])
            xref_counts[xref] = xref_counts.get(xref, 0) + 1

    toc_created = False

    try:
        for page_idx, page in enumerate(doc):
            page_no = page_idx + 1
            raw_text = page.get_text("text") or ""
            clean_text = _sanitize_ascii(raw_text.strip())

            if _is_metadata_text(clean_text):
                print(f"[PDF] Dropped change log / metadata page {page_no}")
                continue

            pages_data.append({"page": page_no, "text": clean_text})
            tables = _extract_tables_from_page(page)

            if not toc_created and page_no <= 4 and _is_toc_page(clean_text):
                main_topics = _extract_main_toc_topics(clean_text)
                scenes_data.append({
                    "page": page_no,
                    "type": "toc",
                    "title": "Table of Contents",
                    "text": clean_text,
                    "toc_items": main_topics,
                    "tables": [],
                    "screenshots": [],
                })
                toc_created = True
                continue

            page_screenshots = _extract_page_screenshots(page, page_no, screenshots_dir, xref_counts)
            for s in page_screenshots:
                all_screenshots.append(s)

            lines = [l.strip() for l in clean_text.splitlines() if l.strip()]
            first_line = lines[0] if lines else "System Overview"

            concept_headers = ["purpose", "objective", "objectives", "typical use cases", "use cases", "prerequisites", "scope", "overview", "introduction", "summary"]
            is_concept_page = any(re.match(rf"^(?:\d+(?:\.\d+)*\s+)?{re.escape(h)}\b", first_line.lower()) for h in concept_headers)

            action_lines = []
            theory_lines = []
            for l in lines:
                is_act = any(re.match(rf"^(?:\d+[\.\)]|\*|-)?\s*{re.escape(v)}\b", l.lower()) for v in ACTION_VERBS)
                if is_act:
                    action_lines.append(l)
                else:
                    theory_lines.append(l)

            if is_concept_page or (not page_screenshots and not tables) or (theory_lines and not action_lines):
                title = _clean_title(first_line)
                scene_type = "title" if (page_no == 1 and len(clean_text.split()) < 25) else "theory"
                scenes_data.append({
                    "page": page_no,
                    "type": scene_type,
                    "title": title,
                    "text": clean_text,
                    "tables": tables,
                    "screenshots": [],
                })

            elif tables and not page_screenshots:
                scenes_data.append({
                    "page": page_no,
                    "type": "table",
                    "title": _clean_title(first_line),
                    "text": clean_text,
                    "tables": tables,
                    "screenshots": [],
                })

            else:
                if len(theory_lines) >= 2 and len(" ".join(theory_lines).split()) > 25:
                    scenes_data.append({
                        "page": page_no,
                        "type": "theory",
                        "title": _clean_title(first_line),
                        "text": "\n".join(theory_lines),
                        "tables": [],
                        "screenshots": [],
                    })

                shot_info = page_screenshots[0] if page_screenshots else None
                shot_path = shot_info["path"] if shot_info else None
                crop_rect = shot_info["crop_rect"] if shot_info else None

                img_bgr = cv2.imread(shot_path) if shot_path else None
                detected_targets = _detect_ui_targets_on_image(img_bgr) if img_bgr is not None else []

                acts_to_emit = action_lines[:4] if action_lines else [clean_text[:200]]
                for a_idx, act in enumerate(acts_to_emit):
                    # Priority 1: Exact keyword search in crop
                    target_pt = _find_keyword_in_page_crop(page, crop_rect, act) if crop_rect else None
                    # Priority 2: UI callout boxes
                    if not target_pt and detected_targets:
                        target_pt = detected_targets[a_idx % len(detected_targets)]

                    scenes_data.append({
                        "page": page_no,
                        "type": "screenshot",
                        "title": _clean_title(act),
                        "text": act,
                        "step_index": a_idx,
                        "action_text": act,
                        "target_point": target_pt,
                        "tables": [],
                        "screenshots": [{"path": shot_path}] if shot_path else [],
                    })

    finally:
        doc.close()

    print(f"[PDF] Extracted {len(scenes_data)} scenes with accurate keyword/UI targeting.")
    return {
        "text": "\n\n".join(s["text"] for s in scenes_data),
        "scenes": scenes_data,
        "pages": pages_data,
        "screenshots": all_screenshots,
    }


extract_pdf_package = process_pdf
