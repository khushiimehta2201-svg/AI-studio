import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pymupdf

MIN_IMAGE_WIDTH = 180
MIN_IMAGE_HEIGHT = 90
MIN_IMAGE_AREA_FRACTION = 0.02
SCREENSHOT_DPI = 150
MERGE_GAP = 20.0


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
            area = bw * bh
            frac = area / total_area
            if 0.001 <= frac <= 0.50:
                boxes.append((x / w, y / h, (x + bw) / w, (y + bh) / h))

        # Sort top-to-bottom, left-to-right to follow natural reading/interaction order
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes
    except Exception as e:
        print(f"[PDF] annotation detection error: {e}")
        return []


def _rects_should_merge(a: pymupdf.Rect, b: pymupdf.Rect, gap: float = MERGE_GAP) -> bool:
    hgap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    vgap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return (vgap <= gap and max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) > 20) or \
           (hgap <= gap and max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0)) > 20)


def _cluster_image_rects(rects: List[pymupdf.Rect]) -> List[pymupdf.Rect]:
    clusters = []
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


def _extract_page_screenshots(doc: pymupdf.Document, output_dir: Path) -> List[Dict[str, Any]]:
    screenshots = []
    xref_counts = {}
    for page in doc:
        for img in page.get_images(full=True) or []:
            xref = int(img[0])
            xref_counts[xref] = xref_counts.get(xref, 0) + 1

    for page_idx, page in enumerate(doc):
        page_no = page_idx + 1
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

        clusters = _cluster_image_rects(valid_rects)

        if clusters:
            for local_idx, crop_rect in enumerate(clusters[:4]):
                pad = 4.0
                clip = pymupdf.Rect(
                    max(page_rect.x0, crop_rect.x0 - pad),
                    max(page_rect.y0, crop_rect.y0 - pad),
                    min(page_rect.x1, crop_rect.x1 + pad),
                    min(page_rect.y1, crop_rect.y1 + pad),
                )
                pix = page.get_pixmap(dpi=SCREENSHOT_DPI, clip=clip)
                img_path = output_dir / f"page_{page_no:03d}_shot_{local_idx+1:02d}.png"
                pix.save(str(img_path))

                # Collect all targets in order
                annotations = _detect_colored_annotations(str(img_path))
                all_targets = [
                    [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
                    for box in annotations
                ]

                screenshots.append({
                    "page": page_no,
                    "screenshot_index": local_idx + 1,
                    "path": str(img_path.resolve()),
                    "width": pix.width,
                    "height": pix.height,
                    "targets": all_targets,
                    "primary_target": all_targets[0] if all_targets else None,
                    "is_full_page": False,
                })
        else:
            # Text-heavy or conclusion page without images
            screenshots.append({
                "page": page_no,
                "screenshot_index": 1,
                "path": None,
                "width": 1280,
                "height": 720,
                "targets": [],
                "primary_target": None,
                "is_full_page": True,
            })

    return screenshots


def process_pdf(pdf_path: str, job_dir: str) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    job_dir = Path(job_dir)
    screenshots_dir = job_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    pages_data = []

    try:
        screenshots = _extract_page_screenshots(doc, screenshots_dir)
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages_data.append({
                "page": i + 1,
                "text": text.strip(),
            })
    finally:
        doc.close()

    full_text = "\n\n".join(p["text"] for p in pages_data if p["text"]).strip()
    return {
        "text": full_text,
        "pages": pages_data,
        "screenshots": screenshots,
    }


extract_pdf_package = process_pdf
