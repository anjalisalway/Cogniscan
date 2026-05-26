import json
import uuid
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


MAX_TOKENS        = 400   
OVERLAP_SENTENCES = 2     
EMBED_MODEL       = "all-MiniLM-L6-v2"   


@dataclass
class Chunk:
    chunk_id   : str
    heading    : Optional[str]          # nearest heading above this chunk
    heading_level: Optional[int]        # 1 / 2 / 3 or None
    text       : str
    page_start : int
    page_end   : int
    token_count: int
    embedding  : list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)




def estimate_tokens(text: str) -> int:
    """Rough token count: ~0.75 words per token."""
    return int(len(text.split()) / 0.75)




def group_by_headings(pages: list[dict]) -> list[dict]:
    """
    Returns a list of raw groups:
        {
            heading      : str | None,
            heading_level: int | None,
            page_start   : int,
            page_end     : int,
            texts        : [str, ...]
        }
    """
    groups: list[dict] = []
    current: dict | None = None

    def flush(group: dict | None) -> None:
        if group and group["texts"]:
            groups.append(group)

    def new_group(heading: str | None, level: int | None, page: int) -> dict:
        return {
            "heading"      : heading,
            "heading_level": level,
            "page_start"   : page,
            "page_end"     : page,
            "texts"        : [],
        }

    for pg in pages:
        page_num = pg["page_number"]

        for block in pg["blocks"]:
            if block["type"] == "heading":
                flush(current)
                # Start a fresh group; 
                current = new_group(block["text"], block.get("level"), page_num)

            elif block["type"] == "paragraph":
                if current is None:
                    current = new_group(None, None, page_num)

                current["texts"].append(block["text"])
                current["page_end"] = page_num

    flush(current)
    return groups



def split_sentences(text: str) -> list[str]:
    
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def split_large_chunk(
    heading     : Optional[str],
    heading_level: Optional[int],
    page_start  : int,
    page_end    : int,
    full_text   : str,
) -> list[Chunk]:
    
    if estimate_tokens(full_text) <= MAX_TOKENS:
        return [
            Chunk(
                chunk_id    = str(uuid.uuid4()),
                heading     = heading,
                heading_level = heading_level,
                text        = full_text,
                page_start  = page_start,
                page_end    = page_end,
                token_count = estimate_tokens(full_text),
            )
        ]

    sentences  = split_sentences(full_text)
    chunks     : list[Chunk] = []
    window     : list[str]   = []
    window_tok : int         = 0

    i = 0
    part_index = 0

    while i < len(sentences):
        sent     = sentences[i]
        sent_tok = estimate_tokens(sent)

        if window_tok + sent_tok > MAX_TOKENS and window:
            # Flush current window as a chunk
            chunk_text = " ".join(window)
            heading_label = f"{heading} (part {part_index + 1})" if heading else f"(part {part_index + 1})"
            chunks.append(
                Chunk(
                    chunk_id    = str(uuid.uuid4()),
                    heading     = heading_label,
                    heading_level = heading_level,
                    text        = chunk_text,
                    page_start  = page_start,
                    page_end    = page_end,
                    token_count = estimate_tokens(chunk_text),
                )
            )
            part_index += 1

            overlap    = window[-OVERLAP_SENTENCES:] if len(window) >= OVERLAP_SENTENCES else window[:]
            window     = overlap
            window_tok = sum(estimate_tokens(s) for s in window)
        else:
            window.append(sent)
            window_tok += sent_tok
            i += 1

    if window:
        chunk_text    = " ".join(window)
        heading_label = f"{heading} (part {part_index + 1})" if heading and part_index > 0 else heading
        chunks.append(
            Chunk(
                chunk_id    = str(uuid.uuid4()),
                heading     = heading_label,
                heading_level = heading_level,
                text        = chunk_text,
                page_start  = page_start,
                page_end    = page_end,
                token_count = estimate_tokens(chunk_text),
            )
        )

    return chunks

def build_chunks(pages: list[dict]) -> list[Chunk]:
    groups = group_by_headings(pages)
    chunks : list[Chunk] = []

    for grp in groups:
        full_text = "\n".join(grp["texts"]).strip()
        if not full_text:
            continue

        sub_chunks = split_large_chunk(
            heading      = grp["heading"],
            heading_level = grp["heading_level"],
            page_start   = grp["page_start"],
            page_end     = grp["page_end"],
            full_text    = full_text,
        )
        chunks.extend(sub_chunks)

    return chunks

def embed_chunks(chunks: list[Chunk], model_name: str = EMBED_MODEL) -> list[Chunk]:
    print(f"Loading embedding model: {model_name} …")
    model  = SentenceTransformer(model_name)
    texts  = [c.text for c in chunks]

    print(f"Embedding {len(texts)} chunks …")
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec.tolist()

    return chunks

def find_similar(query: str, chunks: list[Chunk], model_name: str = EMBED_MODEL, top_k: int = 5) -> list[dict]:
    """
    Given a free-text query, return the top-k most similar chunks by cosine similarity.
    Useful for testing the chunk quality from the CLI (--query flag).
    """
    model      = SentenceTransformer(model_name)
    q_vec      = model.encode([query], convert_to_numpy=True)          # (1, dim)
    chunk_vecs = np.array([c.embedding for c in chunks])               # (N, dim)

    sims  = cosine_similarity(q_vec, chunk_vecs)[0]                    # (N,)
    top   = np.argsort(sims)[::-1][:top_k]

    return [
        {
            "rank"       : rank + 1,
            "score"      : round(float(sims[i]), 4),
            "chunk_id"   : chunks[i].chunk_id,
            "heading"    : chunks[i].heading,
            "page_start" : chunks[i].page_start,
            "text_preview": chunks[i].text[:200] + ("…" if len(chunks[i].text) > 200 else ""),
        }
        for rank, i in enumerate(top)
    ]

def chunk(
    input_path : str | Path,
    output_path: str | Path | None = None,
    embed      : bool = True,
    model_name : str  = EMBED_MODEL,
) -> list[Chunk]:
    """
    Full chunking pipeline.

    Args:
        input_path : (raw data)
        output_path
        embed      : true / false
        model_name : SentenceTransformer model 

    Returns the list of Chunk objects.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path) if output_path else Path("processed/chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    
    print(f"Loading: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        doc = json.load(f)

    pages = doc["pages"]
    print(f"  Pages      : {len(pages)}")

    
    chunks = build_chunks(pages)
    print(f"  Chunks     : {len(chunks)}")
    print(f"  Token range: {min(c.token_count for c in chunks)} – {max(c.token_count for c in chunks)}")

    
    if embed:
        chunks = embed_chunks(chunks, model_name)

    
    output = {
        "source"     : doc.get("source", str(input_path)),
        "model"      : model_name if embed else None,
        "chunk_count": len(chunks),
        "chunks"     : [c.to_dict() for c in chunks],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved → {output_path}")
    return chunks

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw_text.json → semantic chunks with embeddings."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="processed/raw_text.json",
        help="Path to raw_text.json (default: processed/raw_text.json)",
    )
    parser.add_argument(
        "--output",
        default="processed/chunks.json",
        help="Output path for chunks.json",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding (faster, for testing)",
    )
    parser.add_argument(
        "--model",
        default=EMBED_MODEL,
        help=f"SentenceTransformer model (default: {EMBED_MODEL})",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="After chunking, run a similarity search with this query and print top-5 results",
    )
    args = parser.parse_args()

    chunks = chunk(
        input_path  = args.input,
        output_path = args.output,
        embed       = not args.no_embed,
        model_name  = args.model,
    )

    if args.query:
        if not chunks[0].embedding:
            print("Cannot run similarity search — embeddings were skipped.")
        else:
            print(f"\nTop-5 chunks for query: '{args.query}'\n")
            results = find_similar(args.query, chunks, model_name=args.model)
            for r in results:
                print(f"  [{r['rank']}] score={r['score']}  heading='{r['heading']}'  p.{r['page_start']}")
                print(f"       {r['text_preview']}")
                print()


if __name__ == "__main__":
    main()