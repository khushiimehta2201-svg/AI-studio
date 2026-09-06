"""
PDF extraction service.

Responsibilities:
  1. Extract per-page text AND per-word bounding boxes (needed for precise
     cursor grounding later -- see ai_service._find_word_target).
  2. Extract embedded screenshot regions per page, each carrying its
     absolute page-coordinate rect (page_rect) so a page word/target can be
     mapped into that specific screenshot's local coordinate space even
     when a page has several screenshot regions.
  3. Detect explicit colour-annotated boxes (red/orange highlight boxes)
     drawn onto a screenshot, used as the highest-confidence cursor source.
  4. Classify pages so title pages, tables of contents, change logs and
     "prepared by / author" pages can be excluded from the tutorial
     entirely, per spec.
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pymupdf

MIN_IMAGE_WIDTH = 140
MIN_IMAGE_HEIGHT = 70
MIN_IMAGE_AREA_FRACTION = 0.008
SCREENSHOT_DPI = 150
MERGE_GAP = 6.0

# ---------------------------------------------------------------------------
# Page classification (skip logic)
# ---------------------------------------------------------------------------

_TOC_HEADING = re.compile(r"\btable\s+of\s+contents\b|\bcontents\b", re.I)
_TOC_LEADER = re.compile(r"\.{3,}\s*\d{1,4}\s*$")
_CHANGELOG_HEADING = re.compile(
    r"\brevision\s+history\b|\bchange\s*log\b|\bdocument\s+history\b|\bversion\s+history\b",
    re.I,
)
_AUTHOR_LINE = re.compile(
    r"\bprepared\s+by\b|\breviewed\s+by\b|\bapproved\s+by\b|\bauthor(?:ed)?\s*[:\-]|\bowner\s*[:\-]",
    re.I,
)
_TITLE_NOISE = re.compile(
    r"\bconfidential\b|\bproprietary\b|\ball\s+rights\s+reserved\b|\bversion\s+\d+(\.\d+)*\b",
    re.I,
)


def _classify_page(text: str, page_no: int, total_pages: int, has_screenshot: bool) -> Optional[str]:
    """Returns a skip reason ('title' | 'toc' | 'changelog' | 'author') or None."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    joined = "\n".join(lines)
    word_count = len(joined.split())

    if _CHANGELOG_HEADING.search(joined):
        return "changelog"

    if _TOC_HEADING.search(joined):
        return "toc"
    leader_hits = sum(1 for l in lines if _TOC_LEADER.search(l))
    if leader_hits >= 3:
        return "toc"

    # Title / cover page: only plausible in the first couple of pages,
    # short, no screenshot, and reads like a cover (author/confidentiality
    # boilerplate, or just a big sparse heading with nothing actionable).
    if page_no <= 2 and not has_screenshot:
        author_hits = sum(1 for l in lines if _AUTHOR_LINE.search(l))
        noise_hits = sum(1 for l in lines if _TITLE_NOISE.search(l))
        if author_hits >= 1 and word_count <= 120:
            return "author"
        if word_count <= 40 and (noise_hits >= 1 or page_no == 1):
            return "title"

    return None


# ---------------------------------------------------------------------------
# Section heading detection (for grouping pages into navigable sections)
# ---------------------------------------------------------------------------

_HEADING_PATTERN = re.compile(
    r"^(?:chapter|module|section|part|procedure|task|lesson)\s*\d*\s*[:\-]?\s*.+$",
    re.I,
)
_NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)?\s+[A-Z][A-Za-z0-9 /&\-]{3,80}$")


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not (4 <= len(line) <= 80):
        return False
    if line.endswith((".", ",", ";", ":")) and not line.endswith(":"):
        return False
    if _HEADING_PATTERN.match(line):
        return True
    if _NUMBERED_HEADING.match(line):
        return True
    # All-caps or Title Case short line with no terminal punctuation is a
    # reasonable generic heading heuristic for corporate docs.
    words = line.split()
    if 2 <= len(words) <= 9:
        cap_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
        if cap_ratio >= 0.7 and not line.lower().startswith(
            ("click", "select", "enter", "open", "choose", "type", "the ", "this ", "note")
        ):
            return True
    return False


def _page_heading(text: str) -> Optional[str]:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line if _looks_like_heading(line) else None
    return None


# ---------------------------------------------------------------------------
# Word extraction
# ---------------------------------------------------------------------------

def _page_words(page: "pymupdf.Page") -> List[Dict[str, Any]]:
    words = []
    try:
        raw = page.get_text("words") or []
    except Exception:
        raw = []
    for item in raw:
        if len(item) < 5:
            continue
        x0, y0, x1, y1, text = item[:5]
        text = str(text or "").strip()
        if not text:
            continue
        words.append({"text": text, "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)})
    return words


# ---------------------------------------------------------------------------
# Colour annotation detection (explicit highlight boxes drawn by the author)
# ---------------------------------------------------------------------------

def _detect_colored_annotations(image_path: str) -> List[Tuple[float, float, float, float]]:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_red1 = cv2.inRange(hsv, np.array([0, 90, 90]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([165, 90, 90]), np.array([180, 255, 255]))
        mask_orange = cv2.inRange(hsv, np.array([11, 100, 100]), np.array([26, 255, 255]))
        mask = mask_red1 | mask_red2 | mask_orange

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        total_area = float(w * h)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            frac = (bw * bh) / total_area
            if not (0.0015 <= frac <= 0.40):
                continue

            aspect = (bw / float(bh)) if bh else 0.0
            is_small = (bw / float(w) < 0.08) and (bh / float(h) < 0.15)
            is_squarish = 0.7 <= aspect <= 1.4
            extent = cv2.contourArea(cnt) / float(bw * bh) if bw and bh else 0.0
            is_circle_fill = 0.55 <= extent <= 0.92
            if is_small and is_squarish and is_circle_fill:
                # Numbered step-circle markers (a common "1, 2, 3" callout
                # style) sit NEXT TO the control they refer to, not on top
                # of it -- using their position as the cursor point is what
                # made the cursor land on the marker instead of the actual
                # button/field. A genuine highlight box drawn around a
                # control is bigger and rarely this close to square, so
                # only that shape is trusted as a direct cursor target.
                continue

            boxes.append((x / w, y / h, (x + bw) / w, (y + bh) / h))

        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes
    except Exception as e:
        print(f"[PDF] annotation detection error: {e}")
        return []


# ---------------------------------------------------------------------------
# Screenshot region extraction
# ---------------------------------------------------------------------------

def _rects_should_merge(a: "pymupdf.Rect", b: "pymupdf.Rect", gap: float = MERGE_GAP) -> bool:
    hgap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    vgap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return (vgap <= gap and max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) > 20) or \
           (hgap <= gap and max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0)) > 20)


def _cluster_image_rects(rects: List["pymupdf.Rect"]) -> List["pymupdf.Rect"]:
    clusters: List[List["pymupdf.Rect"]] = []
    for r in sorted(rects, key=lambda x: (x.y0, x.x0)):
        matched = False
        for cluster in clusters:
            if any(_rects_should_merge(r, other) for other in cluster):
                cluster.append(r)
                matched = True
                break
        if not matched:
            clusters.append([r])

    merged = []
    for cluster in clusters:
        bbox = pymupdf.Rect(cluster[0])
        for item in cluster[1:]:
            bbox |= item
        merged.append(bbox)
    return merged


def _extract_page_screenshots(page: "pymupdf.Page", page_no: int, output_dir: Path) -> List[Dict[str, Any]]:
    page_rect = page.rect
    image_list = page.get_images(full=True) or []
    xref_counts: Dict[int, int] = {}
    for img in image_list:
        xref = int(img[0])
        xref_counts[xref] = xref_counts.get(xref, 0) + 1

    valid_rects = []
    for img_info in image_list:
        xref = int(img_info[0])
        pw, ph = int(img_info[2] or 0), int(img_info[3] or 0)
        if pw < MIN_IMAGE_WIDTH or ph < MIN_IMAGE_HEIGHT or xref_counts.get(xref, 1) >= 3:
            continue
        for r in page.get_image_rects(xref) or []:
            if r.get_area() / page_rect.get_area() >= MIN_IMAGE_AREA_FRACTION:
                valid_rects.append(r)

    clusters = _cluster_image_rects(valid_rects)
    screenshots = []

    for local_idx, crop_rect in enumerate(clusters[:6]):
        pad = 4.0
        clip = pymupdf.Rect(
            max(page_rect.x0, crop_rect.x0 - pad),
            max(page_rect.y0, crop_rect.y0 - pad),
            min(page_rect.x1, crop_rect.x1 + pad),
            min(page_rect.y1, crop_rect.y1 + pad),
        )
        pix = page.get_pixmap(dpi=SCREENSHOT_DPI, clip=clip)
        img_path = output_dir / f"page_{page_no:03d}_shot_{local_idx + 1:02d}.png"
        pix.save(str(img_path))

        annotations = _detect_colored_annotations(str(img_path))
        targets = [[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in annotations]

        screenshots.append({
            "page": page_no,
            "screenshot_index": local_idx + 1,
            "path": str(img_path.resolve()),
            "width": pix.width,
            "height": pix.height,
            # Absolute page-coordinate rect this crop occupies -- required
            # to map a page-level word match into this screenshot's local
            # normalized space (see ai_service._map_page_point_to_screenshot).
            "page_rect": [clip.x0, clip.y0, clip.x1, clip.y1],
            "targets": targets,
            "primary_target": targets[0] if targets else None,
            "is_full_page": False,
        })

    return screenshots


def _extract_page_screenshots_or_placeholder(page, page_no, output_dir):
    shots = _extract_page_screenshots(page, page_no, output_dir)
    if shots:
        return shots
    return [{
        "page": page_no,
        "screenshot_index": 1,
        "path": None,
        "width": 1280,
        "height": 720,
        "page_rect": None,
        "targets": [],
        "primary_target": None,
        "is_full_page": True,
    }]


def process_pdf(pdf_path: str, job_dir: str) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    job_dir = Path(job_dir)
    screenshots_dir = job_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    pages_data = []
    all_screenshots = []
    total_pages = doc.page_count

    try:
        for page_idx, page in enumerate(doc):
            page_no = page_idx + 1
            text = page.get_text("text") or ""
            words = _page_words(page)
            shots = _extract_page_screenshots_or_placeholder(page, page_no, screenshots_dir)
            has_real_shot = any(not s["is_full_page"] for s in shots)

            skip_reason = _classify_page(text, page_no, total_pages, has_real_shot)
            heading = _page_heading(text) if not skip_reason else None

            pages_data.append({
                "page": page_no,
                "text": text.strip(),
                "words": words,
                "width": float(page.rect.width),
                "height": float(page.rect.height),
                "skip_reason": skip_reason,
                "heading": heading,
            })

            if not skip_reason:
                all_screenshots.extend(shots)
    finally:
        doc.close()

    full_text = "\n\n".join(p["text"] for p in pages_data if p["text"] and not p["skip_reason"]).strip()
    return {
        "text": full_text,
        "pages": pages_data,
        "screenshots": all_screenshots,
    }


extract_pdf_package = process_pdf
