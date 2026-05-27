"""
analyzer.py — Day 4: Cognitive Load Analyzer
=============================================
Scores each chunk across three difficulty dimensions:

  linguistic  — sentence complexity + dependency depth
  semantic    — vocabulary density + rare-word ratio
  structural  — paragraph variance + heading/length signals

Usage
-----
  # Analyze an existing chunks.json (produced by chunker.py)
  python analyzer.py processed/chunks.json

  # Save annotated output to a custom path
  python analyzer.py processed/chunks.json --output processed/analyzed_chunks.json

  # Pretty-print a summary table (no file write)
  python analyzer.py processed/chunks.json --summary

Install
-------
  pip install spacy textstat
  python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import json
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import spacy
import textstat

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPACY_MODEL = "en_core_web_sm"

# Weights within each sub-score (must sum to 1.0 per group)
LINGUISTIC_WEIGHTS = {
    "flesch_kincaid": 0.35,   # grade-level readability
    "avg_sentence_len": 0.30,  # words per sentence (normalized)
    "dep_depth": 0.35,         # average parse-tree depth
}

SEMANTIC_WEIGHTS = {
    "type_token_ratio": 0.40,  # lexical diversity (unique / total words)
    "rare_word_ratio": 0.35,   # words not in spaCy's top-N vocab
    "entity_density": 0.25,    # named entities per sentence
}

STRUCTURAL_WEIGHTS = {
    "paragraph_variance": 0.45,  # std-dev of sentence lengths within chunk
    "heading_depth_penalty": 0.25,  # deeper headings → slightly harder context
    "length_penalty": 0.30,      # very long chunks add cognitive overhead
}

# Normalisation reference points (tuned empirically; adjust as needed)
MAX_FK_GRADE       = 18.0   # Flesch-Kincaid grade 18 → score 1.0
MAX_AVG_SENT_LEN   = 40.0   # 40 words/sentence → score 1.0
MAX_DEP_DEPTH      = 10.0   # parse depth 10 → score 1.0
MAX_ENTITY_DENSITY = 3.0    # 3 entities/sentence → score 1.0
MAX_PARA_VARIANCE  = 20.0   # std-dev of 20 words → score 1.0
MAX_TOKENS_PENALTY = 400.0  # token count at which length penalty maxes out


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DifficultyProfile:
    linguistic : float   # 0.0 – 1.0
    semantic   : float   # 0.0 – 1.0
    structural : float   # 0.0 – 1.0

    # Raw sub-scores (kept for transparency / debugging)
    flesch_kincaid    : float = 0.0
    avg_sentence_len  : float = 0.0
    dep_depth         : float = 0.0
    type_token_ratio  : float = 0.0
    rare_word_ratio   : float = 0.0
    entity_density    : float = 0.0
    paragraph_variance: float = 0.0
    heading_depth_penalty: float = 0.0
    length_penalty    : float = 0.0

    def to_dict(self) -> dict:
        return {
            "linguistic" : round(self.linguistic,  4),
            "semantic"   : round(self.semantic,    4),
            "structural" : round(self.structural,  4),
            "sub_scores" : {
                "flesch_kincaid"      : round(self.flesch_kincaid,       4),
                "avg_sentence_len"    : round(self.avg_sentence_len,     4),
                "dep_depth"           : round(self.dep_depth,            4),
                "type_token_ratio"    : round(self.type_token_ratio,     4),
                "rare_word_ratio"     : round(self.rare_word_ratio,      4),
                "entity_density"      : round(self.entity_density,       4),
                "paragraph_variance"  : round(self.paragraph_variance,   4),
                "heading_depth_penalty": round(self.heading_depth_penalty, 4),
                "length_penalty"      : round(self.length_penalty,       4),
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _normalise(value: float, maximum: float) -> float:
    """Linear normalisation clamped to [0, 1]."""
    if maximum <= 0:
        return 0.0
    return _clamp(value / maximum)


def _tree_depth(token) -> int:
    """Recursively compute depth of the dependency sub-tree rooted at token."""
    children = list(token.children)
    if not children:
        return 0
    return 1 + max(_tree_depth(c) for c in children)


def _sentence_dep_depth(sent) -> float:
    """Average tree depth across all tokens in a sentence."""
    depths = [_tree_depth(tok) for tok in sent]
    return statistics.mean(depths) if depths else 0.0


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------

class CognitiveLoadAnalyzer:
    """
    Wraps a spaCy pipeline and scores chunks for cognitive difficulty.

    Parameters
    ----------
    model : str
        spaCy model name (default: en_core_web_sm).
    """

    def __init__(self, model: str = SPACY_MODEL) -> None:
        print(f"Loading spaCy model: {model} …")
        self.nlp = spacy.load(model)

        # Build a set of common English words from spaCy's vocab
        # (words with a positive probability are "known")
        self._common_vocab: set[str] = {
            lex.text.lower()
            for lex in self.nlp.vocab
            if lex.prob > -15 and lex.is_alpha
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str, heading: Optional[str] = None,
              heading_level: Optional[int] = None,
              token_count: int = 0) -> DifficultyProfile:
        """
        Return a DifficultyProfile for a single text chunk.
        """
        doc = self.nlp(text)
        sents = list(doc.sents)

        # ---- Linguistic -----------------------------------------------
        fk_raw          = textstat.flesch_kincaid_grade(text)
        fk_score        = _normalise(max(fk_raw, 0), MAX_FK_GRADE)

        sent_lengths    = [len(list(s)) for s in sents]  # tokens per sentence
        avg_sent_len    = statistics.mean(sent_lengths) if sent_lengths else 0.0
        avg_sent_score  = _normalise(avg_sent_len, MAX_AVG_SENT_LEN)

        dep_depths      = [_sentence_dep_depth(s) for s in sents]
        avg_dep_depth   = statistics.mean(dep_depths) if dep_depths else 0.0
        dep_score       = _normalise(avg_dep_depth, MAX_DEP_DEPTH)

        linguistic = (
            LINGUISTIC_WEIGHTS["flesch_kincaid"]  * fk_score
            + LINGUISTIC_WEIGHTS["avg_sentence_len"] * avg_sent_score
            + LINGUISTIC_WEIGHTS["dep_depth"]        * dep_score
        )

        # ---- Semantic -------------------------------------------------
        alpha_tokens   = [t.text.lower() for t in doc if t.is_alpha]
        total_words    = len(alpha_tokens)

        # Type-token ratio (lexical diversity)
        if total_words > 0:
            ttr = len(set(alpha_tokens)) / total_words
        else:
            ttr = 0.0
        # High TTR = more unique words = harder; already in [0,1]
        ttr_score = _clamp(ttr)

        # Rare-word ratio (not in common spaCy vocab)
        rare_count  = sum(1 for w in alpha_tokens if w not in self._common_vocab)
        rare_ratio  = rare_count / total_words if total_words > 0 else 0.0
        rare_score  = _clamp(rare_ratio)

        # Named entity density (entities per sentence)
        n_ents      = len(doc.ents)
        n_sents     = len(sents) if sents else 1
        ent_density = n_ents / n_sents
        ent_score   = _normalise(ent_density, MAX_ENTITY_DENSITY)

        semantic = (
            SEMANTIC_WEIGHTS["type_token_ratio"] * ttr_score
            + SEMANTIC_WEIGHTS["rare_word_ratio"]  * rare_score
            + SEMANTIC_WEIGHTS["entity_density"]   * ent_score
        )

        # ---- Structural -----------------------------------------------
        # Paragraph variance: std-dev of sentence lengths
        if len(sent_lengths) > 1:
            para_var   = statistics.stdev(sent_lengths)
        else:
            para_var   = 0.0
        para_score = _normalise(para_var, MAX_PARA_VARIANCE)

        # Heading depth penalty: h3 > h2 > h1 (deeper = more nested context)
        if heading_level is not None:
            hdg_penalty = _normalise(heading_level, 3)
        else:
            hdg_penalty = 0.0

        # Length penalty: very long chunks are harder to process in one read
        len_penalty = _normalise(token_count, MAX_TOKENS_PENALTY)

        structural = (
            STRUCTURAL_WEIGHTS["paragraph_variance"]     * para_score
            + STRUCTURAL_WEIGHTS["heading_depth_penalty"] * hdg_penalty
            + STRUCTURAL_WEIGHTS["length_penalty"]        * len_penalty
        )

        return DifficultyProfile(
            linguistic           = round(_clamp(linguistic),  4),
            semantic             = round(_clamp(semantic),    4),
            structural           = round(_clamp(structural),  4),
            flesch_kincaid       = round(fk_score,            4),
            avg_sentence_len     = round(avg_sent_score,      4),
            dep_depth            = round(dep_score,           4),
            type_token_ratio     = round(ttr_score,           4),
            rare_word_ratio      = round(rare_score,          4),
            entity_density       = round(ent_score,           4),
            paragraph_variance   = round(para_score,          4),
            heading_depth_penalty= round(hdg_penalty,         4),
            length_penalty       = round(len_penalty,         4),
        )

    def score_all(self, chunks: list[dict]) -> list[dict]:
        """
        Annotate a list of chunk dicts (from chunks.json) in-place and return them.
        Each chunk gets a new 'difficulty' key.
        """
        n = len(chunks)
        print(f"Scoring {n} chunks …")

        for idx, chunk in enumerate(chunks, 1):
            profile = self.score(
                text          = chunk.get("text", ""),
                heading       = chunk.get("heading"),
                heading_level = chunk.get("heading_level"),
                token_count   = chunk.get("token_count", 0),
            )
            chunk["difficulty"] = profile.to_dict()

            if idx % 50 == 0 or idx == n:
                print(f"  {idx}/{n} done", end="\r")

        print()  # newline after progress
        return chunks


# ---------------------------------------------------------------------------
# Corpus-level statistics
# ---------------------------------------------------------------------------

def corpus_stats(chunks: list[dict]) -> dict:
    """
    Compute mean / std / min / max difficulty scores across all chunks.
    """
    dims = ("linguistic", "semantic", "structural")
    stats: dict = {}

    for dim in dims:
        values = [c["difficulty"][dim] for c in chunks if "difficulty" in c]
        if not values:
            continue
        stats[dim] = {
            "mean"  : round(statistics.mean(values),   4),
            "stdev" : round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
            "min"   : round(min(values), 4),
            "max"   : round(max(values), 4),
        }

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(chunks: list[dict]) -> None:
    dims = ("linguistic", "semantic", "structural")
    header = f"{'#':>4}  {'Heading':<35}  " + "  ".join(f"{d[:6]:>6}" for d in dims)
    print("\n" + header)
    print("-" * len(header))

    for i, chunk in enumerate(chunks, 1):
        diff    = chunk.get("difficulty", {})
        heading = (chunk.get("heading") or "(no heading)")[:35]
        scores  = "  ".join(f"{diff.get(d, 0.0):>6.3f}" for d in dims)
        print(f"{i:>4}  {heading:<35}  {scores}")

    print()
    stats = corpus_stats(chunks)
    print("Corpus statistics:")
    for dim, s in stats.items():
        print(f"  {dim:<12}  mean={s['mean']:.3f}  stdev={s['stdev']:.3f}"
              f"  min={s['min']:.3f}  max={s['max']:.3f}")
    print()


def analyze(
    input_path : str | Path,
    output_path: str | Path | None = None,
    model      : str               = SPACY_MODEL,
    summary    : bool              = False,
) -> list[dict]:
    """
    Full analysis pipeline.

    Args:
        input_path  : Path to chunks.json (output of chunker.py).
        output_path : Where to write annotated JSON (None → auto).
        model       : spaCy model name.
        summary     : If True, print a summary table to stdout.

    Returns the annotated list of chunk dicts.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.parent / "analyzed_chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        doc = json.load(f)

    chunks = doc.get("chunks", [])
    print(f"  Chunks found: {len(chunks)}")

    analyzer = CognitiveLoadAnalyzer(model=model)
    chunks   = analyzer.score_all(chunks)

    if summary:
        _print_summary(chunks)

    output = {
        **{k: v for k, v in doc.items() if k != "chunks"},
        "difficulty_model": model,
        "corpus_stats"    : corpus_stats(chunks),
        "chunks"          : chunks,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved → {output_path}")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate chunks.json with cognitive difficulty scores."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="processed/chunks.json",
        help="Path to chunks.json (default: processed/chunks.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: <input_dir>/analyzed_chunks.json)",
    )
    parser.add_argument(
        "--model",
        default=SPACY_MODEL,
        help=f"spaCy model (default: {SPACY_MODEL})",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a per-chunk summary table after scoring",
    )
    args = parser.parse_args()

    analyze(
        input_path  = args.input,
        output_path = args.output,
        model       = args.model,
        summary     = args.summary,
    )


if __name__ == "__main__":
    main()