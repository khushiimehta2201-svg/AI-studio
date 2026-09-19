from pathlib import Path
import re
from typing import Any, Dict, List

import fitz


_URL_RE = re.compile(
    r"(?<![\w@])(?:https?://|www\.)[^\s<>\]\[\"')]+",
    flags=re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    return " ".join((text or "").replace("\r", "\n").split())


def _extract_urls(text: str) -> List[str]:
    urls: List[str] = []
    for match in _URL_RE.findall(str(text or "")):
        url = match.rstrip(".,;:!?)]}")
        if url and url not in urls:
            urls.append(url)
    return urls


def _heading_from_text(text: str) -> str | None:
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    if not lines:
        return None

    # Prefer a short first line as a section/page heading.
    # Do not assume that every short first line is non-action: ai_service.py
    # independently decides whether the heading is also an instruction.
    first = lines[0]
    if len(first) <= 120:
        return first

    return None


def _page_urls(page) -> List[str]:
    """
    Collect both visible URLs and hyperlink-only PDF annotations.

    A URL is metadata, not narration content. The AI planner uses this field
    to expose useful destinations in captions without sending them to TTS.
    """
    urls: List[str] = []

    for url in _extract_urls(page.get_text("text") or ""):
        if url not in urls:
            urls.append(url)

    try:
        links = page.get_links()
    except Exception:
        links = []

    for link in links:
        if not isinstance(link, dict):
            continue

        uri = link.get("uri")
        if not uri:
            continue

        uri = _clean_text(str(uri))
        if not uri:
            continue

        # Only expose web URLs to the caption layer. Internal PDF destinations
        # are navigation metadata rather than useful external URLs.
        if uri.lower().startswith(("http://", "https://", "www.")):
            if uri not in urls:
                urls.append(uri)

    return urls


def process_pdf(pdf_path: str, output_dir: str | Path) -> Dict[str, Any]:
    """
    Extract text, page dimensions, word coordinates and embedded images
    from a PDF.

    Output remains compatible with app.services.ai_service.build_tutorial_plan.
    In addition, every page receives `urls`, containing visible and
    hyperlink-only web URLs for caption/metadata use.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    pages: List[Dict[str, Any]] = []
    screenshots: List[Dict[str, Any]] = []

    doc = fitz.open(str(pdf_path))

    try:
        for page_index, page in enumerate(doc):
            page_num = page_index + 1

            text = page.get_text("text") or ""
            words_raw = page.get_text("words") or ""

            words = []
            for item in words_raw:
                if len(item) >= 5:
                    x0, y0, x1, y1, word = item[:5]
                    if str(word).strip():
                        words.append(
                            {
                                "x0": float(x0),
                                "y0": float(y0),
                                "x1": float(x1),
                                "y1": float(y1),
                                "text": str(word),
                            }
                        )

            rect = page.rect

            page_urls = _page_urls(page)

            pages.append(
                {
                    "page": page_num,
                    "text": text,
                    "heading": _heading_from_text(text),
                    "width": float(rect.width),
                    "height": float(rect.height),
                    "words": words,
                    "urls": page_urls,
                }
            )

            # Extract embedded PDF images.
            image_list = page.get_images(full=True)

            for image_index, image_info in enumerate(image_list):
                xref = image_info[0]

                try:
                    extracted = doc.extract_image(xref)
                    image_bytes = extracted["image"]
                    extension = extracted.get("ext", "png")

                    image_path = (
                        screenshots_dir
                        / f"page_{page_num:03d}_image_{image_index + 1:03d}.{extension}"
                    )

                    image_rects = page.get_image_rects(xref)
                    image_rect = image_rects[0] if image_rects else None

                    image_path.write_bytes(image_bytes)

                    screenshots.append(
                        {
                            "page": page_num,
                            "screenshot_index": image_index + 1,
                            "path": str(image_path),
                            "is_full_page": False,
                            "page_rect": (
                                [
                                    float(image_rect.x0),
                                    float(image_rect.y0),
                                    float(image_rect.x1),
                                    float(image_rect.y1),
                                ]
                                if image_rect
                                else None
                            ),
                        }
                    )
                except Exception as exc:
                    print(
                        f"[PDF] Could not extract image "
                        f"page={page_num}, image={image_index + 1}: {exc}"
                    )

            # Full-page fallback.
            if not image_list:
                try:
                    matrix = fitz.Matrix(1.5, 1.5)
                    pix = page.get_pixmap(
                        matrix=matrix,
                        alpha=False,
                    )

                    full_page_path = (
                        screenshots_dir
                        / f"page_{page_num:03d}_full_page.png"
                    )

                    pix.save(str(full_page_path))

                    screenshots.append(
                        {
                            "page": page_num,
                            "screenshot_index": 1,
                            "path": str(full_page_path),
                            "is_full_page": True,
                            "page_rect": [
                                0,
                                0,
                                float(rect.width),
                                float(rect.height),
                            ],
                        }
                    )
                except Exception as exc:
                    print(
                        f"[PDF] Could not render full page {page_num}: {exc}"
                    )

    finally:
        doc.close()

    result = {
        "pages": pages,
        "screenshots": screenshots,
        "page_count": len(pages),
        "screenshot_count": len(screenshots),
    }

    print(
        f"[PDF] Processed {pdf_path.name}: "
        f"{len(pages)} pages / {len(screenshots)} screenshots"
    )

    return result
