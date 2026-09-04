from pathlib import Path
import hashlib
from collections import Counter
import pymupdf

SCREENSHOT_SCALE = 2.0
MIN_IMAGE_WIDTH = 180
MIN_IMAGE_HEIGHT = 90
MIN_IMAGE_AREA_FRACTION = 0.01
MIN_SCREENSHOT_WIDTH_FRACTION = 0.22
MIN_SCREENSHOT_HEIGHT_FRACTION = 0.10
MERGE_GAP = 24.0


def _page_words(page):
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
        words.append({
            "text": text,
            "x0": float(x0), "y0": float(y0),
            "x1": float(x1), "y1": float(y1),
        })
    return words


def _extract_pages(pdf_path):
    doc = pymupdf.open(str(pdf_path))
    pages = []
    try:
        for i, page in enumerate(doc):
            rect = page.rect
            pages.append({
                "page": i + 1,
                "text": page.get_text("text") or "",
                "words": _page_words(page),
                "width": float(rect.width),
                "height": float(rect.height),
            })
    finally:
        doc.close()
    return pages


def _collect_embedded_images(doc):
    """Collect image placements and track repeated image xrefs across the document."""
    placements = []
    xref_pages = {}
    for page_idx, page in enumerate(doc):
        page_no = page_idx + 1
        try:
            image_list = page.get_images(full=True) or []
        except Exception:
            image_list = []
        seen = set()
        for num, info in enumerate(image_list, 1):
            if not info:
                continue
            xref = int(info[0])
            if xref in seen:
                continue
            seen.add(xref)
            pw, ph = int(info[2] or 0), int(info[3] or 0)
            if pw < MIN_IMAGE_WIDTH or ph < MIN_IMAGE_HEIGHT:
                continue
            try:
                rects = page.get_image_rects(xref) or []
            except Exception:
                rects = []
            if rects:
                xref_pages.setdefault(xref, set()).add(page_no)
            for rect in rects:
                if rect.get_area() <= 0:
                    continue
                placements.append({
                    "page": page_no,
                    "xref": xref,
                    "image_number": num,
                    "pixel_width": pw,
                    "pixel_height": ph,
                    "rect": rect,
                })
    return placements, {xref: len(pages) for xref, pages in xref_pages.items()}


def _overlap_ratio(a0, a1, b0, b1):
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-6, min(a1 - a0, b1 - b0))
    return overlap / denom


def _rects_should_merge(a, b, gap=MERGE_GAP):
    hgap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    vgap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    xover = _overlap_ratio(a.x0, a.x1, b.x0, b.x1)
    yover = _overlap_ratio(a.y0, a.y1, b.y0, b.y1)
    if vgap <= gap and xover >= 0.55:
        return True
    if hgap <= gap and yover >= 0.55:
        return True
    return False


def _cluster_images(images):
    clusters = []
    for item in sorted(images, key=lambda x: (x["rect"].y0, x["rect"].x0)):
        target = None
        for cluster in clusters:
            if any(_rects_should_merge(item["rect"], other["rect"]) for other in cluster):
                target = cluster
                break
        if target is None:
            clusters.append([item])
        else:
            target.append(item)

    changed = True
    while changed:
        changed = False
        merged = []
        while clusters:
            current = clusters.pop(0)
            i = 0
            while i < len(clusters):
                other = clusters[i]
                if any(_rects_should_merge(a["rect"], b["rect"]) for a in current for b in other):
                    current.extend(other)
                    clusters.pop(i)
                    changed = True
                else:
                    i += 1
            merged.append(current)
        clusters = merged
    return clusters


def _bbox(cluster):
    out = pymupdf.Rect(cluster[0]["rect"])
    for item in cluster[1:]:
        out |= item["rect"]
    return out


def _render_clip(page, clip, output):
    page_rect = page.rect
    pad = 1.5
    clip = pymupdf.Rect(
        max(page_rect.x0, clip.x0 - pad),
        max(page_rect.y0, clip.y0 - pad),
        min(page_rect.x1, clip.x1 + pad),
        min(page_rect.y1, clip.y1 + pad),
    )
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(SCREENSHOT_SCALE, SCREENSHOT_SCALE),
        clip=clip,
        alpha=False,
        colorspace=pymupdf.csRGB,
    )
    pix.save(str(output))
    return pix.width, pix.height, [float(clip.x0), float(clip.y0), float(clip.x1), float(clip.y1)]


def _visual_score(box, page_rect, repeated_count, parts):
    frac = box.get_area() / max(1.0, page_rect.get_area())
    wf = box.width / max(1.0, page_rect.width)
    hf = box.height / max(1.0, page_rect.height)
    aspect = box.width / max(1.0, box.height)
    score = frac * 3.0 + wf * 1.2 + hf * 0.8 + min(parts, 4) * 0.35
    if repeated_count >= 2:
        score -= 2.5
    if frac < MIN_IMAGE_AREA_FRACTION:
        score -= 4.0
    if wf < MIN_SCREENSHOT_WIDTH_FRACTION and hf < MIN_SCREENSHOT_HEIGHT_FRACTION:
        score -= 2.0
    if aspect > 9.0 or aspect < 0.12:
        score -= 3.0
    return score


def _is_likely_decorative(box, page_rect, repeated_count, parts):
    frac = box.get_area() / max(1.0, page_rect.get_area())
    wf = box.width / max(1.0, page_rect.width)
    hf = box.height / max(1.0, page_rect.height)
    aspect = box.width / max(1.0, box.height)
    if repeated_count >= 2 and frac < 0.50:
        return True
    if frac < MIN_IMAGE_AREA_FRACTION:
        return True
    if wf < MIN_SCREENSHOT_WIDTH_FRACTION and hf < MIN_SCREENSHOT_HEIGHT_FRACTION:
        return True
    if aspect > 9.0 or aspect < 0.12:
        return True
    if parts == 1 and frac < 0.03:
        return True
    return False


def _extract_visuals(pdf_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(pdf_path))
    visuals = []
    try:
        placements, repeated = _collect_embedded_images(doc)
        by_page = {}
        for item in placements:
            by_page.setdefault(item["page"], []).append(item)

        for page_idx, page in enumerate(doc):
            page_no = page_idx + 1
            page_rect = page.rect
            page_images = by_page.get(page_no, [])
            clusters = _cluster_images(page_images)
            candidates = []
            for cluster in clusters:
                box = _bbox(cluster)
                xrefs = {c["xref"] for c in cluster}
                repeat_max = max((repeated.get(x, 1) for x in xrefs), default=1)
                if _is_likely_decorative(box, page_rect, repeat_max, len(cluster)):
                    continue
                score = _visual_score(box, page_rect, repeat_max, len(cluster))
                candidates.append((score, box, cluster, repeat_max))

            candidates.sort(key=lambda x: x[0], reverse=True)
            # Keep at most the strongest 6 visual regions on a page.
            for local_idx, (score, box, cluster, repeat_max) in enumerate(candidates[:6]):
                path = output_dir / f"page_{page_no:04d}_screenshot_{local_idx + 1:02d}.png"
                w, h, bbox = _render_clip(page, box, path)
                visuals.append({
                    "index": len(visuals),
                    "page": page_no,
                    "screenshot_index_on_page": local_idx,
                    "path": str(path.resolve()),
                    "width": w,
                    "height": h,
                    "page_width": float(page_rect.width),
                    "page_height": float(page_rect.height),
                    "bbox": bbox,
                    "parts": len(cluster),
                    "score": round(score, 4),
                    "repeated_visual": repeat_max >= 2,
                    "source_type": "embedded_screenshot",
                })
    finally:
        doc.close()
    return visuals


def process_pdf(pdf_path, job_dir):
    pdf_path = Path(pdf_path)
    job_dir = Path(job_dir)
    screenshot_dir = job_dir / "screenshots"
    pages = _extract_pages(pdf_path)
    screenshots = _extract_visuals(pdf_path, screenshot_dir)
    text = "\n\n".join(p["text"] for p in pages if p.get("text")).strip()
    print(f"[PDF] pages={len(pages)} text_chars={len(text)} embedded_screenshots={len(screenshots)}")
    return {
        "text": text,
        "pages": pages,
        "screenshots": screenshots,
        "page_images": screenshots,
    }


extract_pdf_package = process_pdf
