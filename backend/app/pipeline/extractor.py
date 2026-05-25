import sys
import fitz
from collections import Counter
import json
from pathlib import Path
import argparse

def collect_font_sizes(doc : fitz.Document) -> Counter:
    sizes:Counter = Counter ()
    for page in doc:
        blocks=page.get_text("dict" , flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block["type"]!=0: # not a text block
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    size=round(span["size"] , 1)
                    sizes[size]+=len(span["text"].strip())
    return sizes

def body_font_size(sizes : Counter) -> float:
    if not sizes:
        return 12.0
    return sizes.most_common(1)[0][0]

def heading_level(span_size : float , body_size : float) -> int | None:
    ratio = span_size / body_size if body_size else 1
    if ratio >=1.6:
        return 1 #h1
    if ratio>=1.3:
        return 2 #h2
    if ratio >= 1.1:
        return 3 #h3
    return None #body text

def is_bold(flags : int) -> bool:
    return (flags & 16)

def extract_block(block : dict , body_size:float) -> dict | None:

    """
     Returns a dict with:
        type        : "heading" | "paragraph"
        level       : 1/2/3 for headings, None for paragraphs
        text        : full cleaned text of the block
        bbox        : [x0, y0, x1, y1] bounding box on the page
        lines       : list of line strings (preserves line breaks)
    """

    if block["type"]!=0:
        return None
    
    lines_text:list[str]=[]
    heading_level_votes:list[int]=[]

    for line in block["lines"]:
        line_parts:list[str]=[]
        for span in line["spans"]:
            txt=span["text"]
            if not txt.strip():
                continue
            line_parts.append(txt)

            size=round(span["size"] , 1)
            lvl=heading_level(size , body_size)
            bold=is_bold(span["flags"])

            if lvl is not None or (bold and size>body_size):
                heading_level_votes.append(lvl or 3)
        
        if line_parts:
            lines_text.append(" ".join(line_parts).strip())

    full_text=" ".join(lines_text).strip()
    if not full_text :
        return None
    
    total_spans=sum(len(l["spans"]) for l in block["lines"])
    is_heading = len(heading_level_votes) >= max(1 , total_spans * 0.5)
    level=min(heading_level_votes) if(is_heading and heading_level_votes) else None

    return {
        "type" : "heading" if is_heading else "paragraph",
        "level" : level,
        "text" : full_text,
        "bbox" : [round(v,2) for v in block["bbox"]],
        "lines" : lines_text
    }

def extract_page(page : fitz.Page , page_number : int , body_size:int) -> dict:
    """
    Returns:
        page_number : 1-based
        width/height: page dimensions in points
        text        : concatenated plain text of the whole page
        headings    : list of heading blocks only
        paragraphs  : list of paragraph blocks only
        blocks      : all blocks in reading order
    """
    
    raw=page.get_text("dict" , flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks:list[dict]=[]

    for block in raw["blocks"]:
        parsed=extract_block(block , body_size)
        if parsed:
            parsed["block_index"] = len(blocks)
            blocks.append(parsed)

    return {
        "page_number" : page_number , 
        "height" : round(raw["height"],2),
        "width" : round(raw["width"],2), 
        "text" : "\n".join(b["text"] for b in blocks),
        "headings" : [p for p in blocks if p["type"]=="heading"],
        "paragraphs" : [p for p in blocks if p["type"]=="paragraph"],
        "blocks" : blocks
    }

def extract(pdf_path: str | Path, output_path: str | Path | None = None) -> dict:
    """
    Full extraction pipeline.
 
    Args:
        pdf_path   : path to the input PDF
        output_path: where to write raw_text.json (default: processed/raw_text.json)
 
    Returns the complete structured document dict.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
 
    output_path = Path(output_path) if output_path else Path("processed/raw_text.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    doc = fitz.open(str(pdf_path))
 
    # ── Step 1: learn body font size from the whole document ──────────────────
    sizes = collect_font_sizes(doc)
    body_size = body_font_size(sizes)
 
    # ── Step 2: extract metadata ──────────────────────────────────────────────
    meta = doc.metadata or {}
 
    # ── Step 3: extract every page ────────────────────────────────────────────
    pages: list[dict] = []
    for page in doc:
        pages.append(extract_page(page, page.number + 1, body_size))
 
    doc.close()
 
    # ── Step 4: build top-level document outline (TOC from headings) ──────────
    outline: list[dict] = []
    for pg in pages:
        for h in pg["headings"]:
            outline.append({
                "level": h["level"],
                "text": h["text"],
                "page": pg["page_number"],
            })
 
    result = {
        "source": str(pdf_path),
        "metadata": {
            "title":    meta.get("title", ""),
            "author":   meta.get("author", ""),
            "subject":  meta.get("subject", ""),
            "creator":  meta.get("creator", ""),
            "page_count": len(pages),
            "body_font_size_pt": body_size,
        },
        "outline": outline,       # document-level heading tree
        "pages": pages,           # full page-level hierarchy
    }
 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
 
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract structured text from a PDF.")
    parser.add_argument("pdf", help="Path to the input PDF file")
    parser.add_argument("--output", default="processed/raw_text.json",
                        help="Output JSON path (default: processed/raw_text.json)")
    parser.add_argument("--debug", action="store_true",
                        help="Print extraction summary to stdout")
    args = parser.parse_args()
 
    print(f"Extracting: {args.pdf}")
    result = extract(args.pdf, args.output)
    print(f"Saved → {args.output}")
 
 
 
if __name__ == "__main__":
    main()


