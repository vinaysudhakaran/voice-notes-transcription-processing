#!/usr/bin/env python3
"""
misrep_domain_analytics.py

Domain-focused analytics by combining:
  1) misrepresented terms (errors) JSONL
  2) LLM-classified glossary (term -> category)

Scans:
    PROCESSED-DATA-DND/<language>/combined_misrep_terms.jsonl
    PROCESSED-DATA-DND/<language>/combined_glossary_terms_<language>.jsonl  (default)
or provide --glossary-jsonl to point to a custom file.

Writes:
    - PROCESSED-DATA-DND/<language>/<output_dir>/*.html (shareable, standalone)
    - PROCESSED-DATA-DND/<language>/<output_dir>/index.html (table of contents)
    - Optional exports (e.g., CSVs for bias lexicons)

Usage:
    python misrep_domain_analytics.py
    python misrep_domain_analytics.py
        --language hindi --base-folder PROCESSED-DATA-DND
    python misrep_domain_analytics.py
        --language hindi --glossary-jsonl PROCESSED-DATA-DND/hindi/combined_glossary_terms_hindi.jsonl --exclude-generic
    python misrep_domain_analytics.py
        --language hindi --top-k 200 --top-cats 6 --normalize-digits --bins 10

Dependencies:
    pip install pandas plotly numpy
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
import logging
import math
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Language script ranges (add more as needed)
# -----------------------------------------------------------------------------
SCRIPT_RANGES: Dict[str, str] = {
    "hindi": "\u0900-\u097f",  # Devanagari
    "bengali": "\u0980-\u09ff",  # Bengali
    "gujarati": "\u0a80-\u0aff",  # Gujarati
    "tamil": "\u0b80-\u0bff",  # Tamil
    "telugu": "\u0c00-\u0c7f",  # Telugu
    "kannada": "\u0c80-\u0cff",  # Kannada
    "malayalam": "\u0d00-\u0d7f",  # Malayalam
    "odiya": "\u0b00-\u0b7f",  # Odiya
    "punjabi": "\u0a00-\u0a7f",  # Gurmukhi
}


# -----------------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, str] = {
    "language": "hindi",
    "base_folder": "PROCESSED-DATA-DND",
    "output_dir": "misrep_domain_analytics",
    # Expected filenames under <base>/<language>/
    "misrep_filename": "combined_misrep_terms.jsonl",
    # If glossary path not provided, we try: combined_glossary_terms_<language>.jsonl
}


# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
try:
    from utils.logger import logger
except Exception:
    logger = logging.getLogger("misrep_domain_analytics")
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    if not logger.handlers:
        logger.addHandler(_handler)


# -----------------------------------------------------------------------------
# HTML helpers
# -----------------------------------------------------------------------------
def write_table_html(df: pd.DataFrame, title: str, path: Path) -> None:
    """Write a simple standalone HTML page with a table (no external deps)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:24px;}}
h1,h2{{margin:0 0 12px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px;}}
th,td{{border:1px solid #ddd;padding:6px;vertical-align:top;}}
th{{background:#f5f5f5;text-align:left;}}
tr:nth-child(even){{background:#fafafa;}}
.code{{font-family:Menlo,Consolas,monospace;font-size:12px;color:#333}}
.small{{font-size:12px;color:#666;margin-top:8px}}
</style>
</head>
<body>
<h1>{title}</h1>
{df.to_html(index=False, escape=False)}
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    logger.info(f"Wrote table: {path}")


def write_plot_html(fig: go.Figure, path: Path, title: Optional[str] = None) -> None:
    """Write a Plotly figure to HTML (standalone)."""
    if title:
        fig.update_layout(title=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    logger.info(f"Wrote plot: {path}")


def write_index_html(
    items: List[Tuple[str, str]], out_path: Path, heading: str
) -> None:
    """Create a simple index HTML linking to generated artifacts. items = [(name, rel_path), ...]"""
    lines = [f'<li><a href="{rel}">{name}</a></li>' for name, rel in items]
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{heading}</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;margin:24px;}}</style>
</head>
<body>
<h1>{heading}</h1>
<ul>
{''.join(lines)}
</ul>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Wrote index: {out_path}")


# -----------------------------------------------------------------------------
# Normalization & script utilities
# -----------------------------------------------------------------------------
def script_regex(language: str) -> Optional[re.Pattern]:
    rng = SCRIPT_RANGES.get(language.lower())
    if not rng:
        return None
    return re.compile(rf"[{rng}]+", flags=re.UNICODE)


def strip_non_script_chars(text: str, language: str) -> str:
    """Strip leading/trailing non-script chars (punct/digits/Latin) based on SCRIPT_RANGES."""
    lang_key = language.lower()
    char_range = SCRIPT_RANGES.get(lang_key)
    if not char_range:
        return text.strip()
    # remove non-block chars from both ends
    pattern = rf"^[^{char_range}]+|[^{char_range}]+$"
    return re.sub(pattern, "", text)


def normalize_digits_to_ascii(token: str) -> str:
    out = []
    for ch in token:
        if unicodedata.category(ch) == "Nd":
            out.append(str(unicodedata.digit(ch)))
        else:
            out.append(ch)
    return "".join(out)


def normalize_token(
    token: str, language: str, keep_punct: bool = False, normalize_digits: bool = True
) -> str:
    """Unicode NFC, optional punctuation trimming (language-aware), optional digit normalization."""
    if token is None or (isinstance(token, float) and math.isnan(token)):
        return ""
    t = unicodedata.normalize("NFC", str(token).strip())
    # mimic the glossary_classifier clean (strip non-script at ends)
    t = strip_non_script_chars(t, language)
    if not keep_punct:
        # remove residual punctuation inside
        t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    if normalize_digits:
        t = normalize_digits_to_ascii(t)
    return t.strip()


def is_in_script(token: str, language: str) -> bool:
    rgx = script_regex(language)
    if not token or rgx is None:
        return False
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return False
    in_count = sum(1 for ch in letters if rgx.fullmatch(ch) or re.match(rgx, ch or ""))
    return in_count >= max(1, int(0.6 * len(letters)))


def midpoint(row: pd.Series) -> float:
    st = row.get("start_time", np.nan)
    ed = row.get("end_time", np.nan)
    if pd.notna(st) and pd.notna(ed):
        return (st + ed) / 2.0
    return np.nan


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if (b and not pd.isna(b)) else np.nan


def compute_utterance_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-uid approximate duration span from min(start) to max(end)."""
    tmp = df.dropna(subset=["start_time", "end_time"]).copy()
    if tmp.empty:
        df["utt_start"] = np.nan
        df["utt_end"] = np.nan
        df["utt_dur"] = np.nan
        return df
    grp = tmp.groupby("uid").agg(
        utt_start=("start_time", "min"), utt_end=("end_time", "max")
    )
    grp["utt_dur"] = grp["utt_end"] - grp["utt_start"]
    out = df.merge(grp, left_on="uid", right_index=True, how="left")
    return out


def slugify(text: str) -> str:
    """Safe file name from category labels."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()


# -----------------------------------------------------------------------------
# Loaders & joins
# -----------------------------------------------------------------------------
def load_misrep(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Misrep JSONL not found: {path}")
    df = pd.read_json(path, lines=True, dtype=False)
    # Expected columns
    expected = [
        "_id",
        "uid",
        "model",
        "error_type",
        "ref_word",
        "hyp_word",
        "start_time",
        "end_time",
        "clip_path",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    # Types
    for c in ["_id", "uid", "model", "error_type", "ref_word", "hyp_word", "clip_path"]:
        df[c] = df[c].astype("string")
    for c in ["start_time", "end_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["error_type"])


def load_glossary(
    path: Path, language: str, keep_punct: bool, normalize_digits: bool
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Glossary JSONL not found: {path}")
    g = pd.read_json(path, lines=True, dtype=False)
    # Ensure term/category cols
    if "term" not in g.columns or "category" not in g.columns:
        raise ValueError("Glossary file must have 'term' and 'category' columns")
    g["term_norm"] = (
        g["term"]
        .astype("string")
        .apply(
            lambda s: normalize_token(
                s, language, keep_punct=keep_punct, normalize_digits=normalize_digits
            )
        )
    )
    g = g.drop_duplicates(subset=["term_norm"]).reset_index(drop=True)
    return g[["term_norm", "category"]]


def join_with_glossary(
    df: pd.DataFrame,
    g: pd.DataFrame,
    language: str,
    keep_punct: bool,
    normalize_digits: bool,
) -> pd.DataFrame:
    df = df.copy()
    df["ref_norm"] = (
        df["ref_word"]
        .astype("string")
        .apply(
            lambda s: normalize_token(
                s, language, keep_punct=keep_punct, normalize_digits=normalize_digits
            )
        )
    )
    df["hyp_norm"] = (
        df["hyp_word"]
        .astype("string")
        .apply(
            lambda s: normalize_token(
                s, language, keep_punct=keep_punct, normalize_digits=normalize_digits
            )
        )
    )
    # ref side join
    df = df.merge(
        g.rename(columns={"term_norm": "ref_norm", "category": "ref_cat"}),
        on="ref_norm",
        how="left",
    )
    # Optional hyp side join (will be NaN if term not in glossary)
    df = df.merge(
        g.rename(columns={"term_norm": "hyp_norm", "category": "hyp_cat"}),
        on="hyp_norm",
        how="left",
    )
    return df


# -----------------------------------------------------------------------------
# Analytics (domain-focused)
# -----------------------------------------------------------------------------
DOMAIN_EXCLUDE = {"Generic", "Not Applicable", None, np.nan}


def glossary_coverage(df: pd.DataFrame, out_dir: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    # Unique ref terms
    uniq_terms = df["ref_norm"].dropna().unique().tolist()
    total_unique = len(uniq_terms)
    with_cat = df.dropna(subset=["ref_cat"])["ref_norm"].unique().tolist()
    total_with_cat = len(with_cat)
    with_domain = (
        df[df["ref_cat"].isin(set(df["ref_cat"].unique()) - DOMAIN_EXCLUDE)]["ref_norm"]
        .unique()
        .tolist()
    )
    total_with_domain = len(with_domain)

    table = pd.DataFrame(
        [
            {
                "unique_ref_terms": total_unique,
                "with_any_category": total_with_cat,
                "with_domain_category": total_with_domain,
                "domain_coverage_share": safe_div(total_with_domain, total_unique),
            }
        ]
    )
    write_table_html(
        table, "Glossary coverage (reference terms)", out_dir / "glossary_coverage.html"
    )
    items.append(("Glossary coverage (reference terms)", "glossary_coverage.html"))
    return items


def filter_domain(df: pd.DataFrame, include_generic: bool) -> pd.DataFrame:
    if include_generic:
        return df
    return df[df["ref_cat"].notna() & (~df["ref_cat"].isin(DOMAIN_EXCLUDE))].copy()


def category_composition(df: pd.DataFrame, out_dir: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    # Overall counts by category
    comp = (
        df.groupby("ref_cat")
        .size()
        .reset_index(name="error_events")
        .sort_values("error_events", ascending=False)
    )
    write_table_html(
        comp,
        "Category composition of error events (ref_cat)",
        out_dir / "category_composition.html",
    )
    items.append(("Category composition (ref_cat)", "category_composition.html"))
    if not comp.empty:
        fig = px.bar(
            comp,
            x="ref_cat",
            y="error_events",
            title="Category composition (error events by ref_cat)",
        )
        fig.update_layout(xaxis_tickangle=-45, margin=dict(b=160))
        write_plot_html(fig, out_dir / "category_composition_plot.html")
        items.append(("Bar: Category composition", "category_composition_plot.html"))

    # Split by error_type
    comp2 = df.groupby(["ref_cat", "error_type"]).size().reset_index(name="count")
    write_table_html(
        comp2, "Category × error_type counts", out_dir / "category_by_error_type.html"
    )
    items.append(("Category × error_type counts", "category_by_error_type.html"))
    if not comp2.empty:
        fig2 = px.bar(
            comp2,
            x="ref_cat",
            y="count",
            color="error_type",
            barmode="group",
            title="Category × error_type counts",
        )
        fig2.update_layout(xaxis_tickangle=-45, margin=dict(b=160))
        write_plot_html(fig2, out_dir / "category_by_error_type_plot.html")
        items.append(("Bar: Category × error_type", "category_by_error_type_plot.html"))
    return items


def category_model_heatmap(df: pd.DataFrame, out_dir: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    piv = df.groupby(["ref_cat", "model"]).size().unstack(fill_value=0)
    tbl = piv.reset_index()
    write_table_html(
        tbl, "Error counts by ref_cat × model", out_dir / "cat_model_counts.html"
    )
    items.append(("ref_cat × model (counts)", "cat_model_counts.html"))

    if not piv.empty:
        fig = go.Figure(
            data=go.Heatmap(
                z=piv.values, x=piv.columns, y=piv.index, coloraxis="coloraxis"
            )
        )
        fig.update_layout(
            coloraxis=dict(colorscale="Blues"),
            title="ref_cat × model — error counts",
            xaxis_nticks=36,
            yaxis_nticks=36,
            margin=dict(l=160, b=160),
        )
        write_plot_html(fig, out_dir / "cat_model_heatmap.html")
        items.append(("Heatmap: ref_cat × model", "cat_model_heatmap.html"))
    return items


def formatting_only_share_by_cat(
    df: pd.DataFrame,
    language: str,
    out_dir: Path,
    keep_punct: bool,
    normalize_digits: bool,
) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    subs = df[df["error_type"] == "sub"].copy()
    if subs.empty:
        return items
    # normalized tokens (punct stripped unless keep_punct True)
    subs["ref_norm2"] = (
        subs["ref_word"]
        .astype("string")
        .apply(
            lambda s: normalize_token(
                s, language, keep_punct=False, normalize_digits=normalize_digits
            )
        )
    )
    subs["hyp_norm2"] = (
        subs["hyp_word"]
        .astype("string")
        .apply(
            lambda s: normalize_token(
                s, language, keep_punct=False, normalize_digits=normalize_digits
            )
        )
    )
    subs["fmt_only"] = (subs["ref_norm2"] == subs["hyp_norm2"]) & (
        subs["ref_word"] != subs["hyp_word"]
    )
    g = (
        subs.groupby("ref_cat")
        .agg(
            total_substitutions=("fmt_only", "size"),
            formatting_only_subs=("fmt_only", "sum"),
        )
        .reset_index()
    )
    g["formatting_only_share"] = g.apply(
        lambda r: safe_div(r["formatting_only_subs"], r["total_substitutions"]), axis=1
    )
    write_table_html(
        g.sort_values("formatting_only_share", ascending=False),
        "Formatting-only substitutions by category",
        out_dir / "formatting_only_by_cat.html",
    )
    items.append(
        ("Formatting-only substitutions by category", "formatting_only_by_cat.html")
    )
    if not g.empty:
        fig = px.bar(
            g,
            x="ref_cat",
            y="formatting_only_share",
            title="Formatting-only share by category",
        )
        fig.update_layout(xaxis_tickangle=-45, margin=dict(b=160))
        write_plot_html(fig, out_dir / "formatting_only_by_cat_plot.html")
        items.append(
            (
                "Bar: Formatting-only share by category",
                "formatting_only_by_cat_plot.html",
            )
        )
    return items


def top_problem_terms_by_cat(
    df: pd.DataFrame, out_dir: Path, top_k: int = 50
) -> List[Tuple[str, str]]:
    """For each category, top ref_norm terms by (sub+del) counts."""
    items: List[Tuple[str, str]] = []
    dd = df[df["error_type"].isin(["sub", "del"])].copy()
    if dd.empty:
        return items
    grp = (
        dd.groupby(["ref_cat", "ref_norm"])
        .size()
        .reset_index(name="count")
        .sort_values(["ref_cat", "count"], ascending=[True, False])
    )
    # take top_k per category
    grp["rank"] = grp.groupby("ref_cat")["count"].rank(method="first", ascending=False)
    top = grp[grp["rank"] <= top_k].drop(columns=["rank"])
    write_table_html(
        top,
        f"Top problem terms (reference) — top {top_k} per category",
        out_dir / "top_problem_terms_by_cat.html",
    )
    items.append(
        (f"Top problem terms (ref) — top {top_k}/cat", "top_problem_terms_by_cat.html")
    )
    # simple bar treemap
    if not top.empty:
        fig = px.treemap(
            top,
            path=["ref_cat", "ref_norm"],
            values="count",
            title=f"Top problem terms by category — top {top_k}/cat",
        )
        write_plot_html(fig, out_dir / "top_problem_terms_by_cat_treemap.html")
        items.append(
            (
                "Treemap: Top problem terms by category",
                "top_problem_terms_by_cat_treemap.html",
            )
        )
    return items


def category_bounded_confusions(
    df: pd.DataFrame, out_dir: Path, top_k: int = 50
) -> List[Tuple[str, str]]:
    """For each category, top substitution pairs (ref_norm → hyp_word)."""
    items: List[Tuple[str, str]] = []
    subs = df[df["error_type"] == "sub"].copy()
    if subs.empty:
        return items
    grp = (
        subs.groupby(["ref_cat", "ref_norm", "hyp_word"])
        .size()
        .reset_index(name="count")
        .sort_values(["ref_cat", "count"], ascending=[True, False])
    )
    # top per category
    grp["rank"] = grp.groupby("ref_cat")["count"].rank(method="first", ascending=False)
    top = grp[grp["rank"] <= top_k].drop(columns=["rank"])
    write_table_html(
        top,
        f"Top confusion pairs by category — top {top_k} per category",
        out_dir / "top_confusions_by_cat.html",
    )
    items.append(
        (f"Top confusions by category — top {top_k}/cat", "top_confusions_by_cat.html")
    )
    if not top.empty:
        fig = px.treemap(
            top,
            path=["ref_cat", "ref_norm", "hyp_word"],
            values="count",
            title=f"Top confusions by category — top {top_k}/cat",
        )
        write_plot_html(fig, out_dir / "top_confusions_by_cat_treemap.html")
        items.append(
            (
                "Treemap: Top confusions by category",
                "top_confusions_by_cat_treemap.html",
            )
        )
    return items


def per_category_top_pairs(
    df: pd.DataFrame, out_dir: Path, top_k: int = 100
) -> List[Tuple[str, str]]:
    """Write a separate table per ref_cat with top (ref_norm -> hyp_word) substitution pairs."""
    items: List[Tuple[str, str]] = []
    subs = (
        df[df["error_type"] == "sub"]
        .dropna(subset=["ref_cat", "ref_norm", "hyp_word"])
        .copy()
    )
    if subs.empty:
        return items

    cat_dir = out_dir / "per_category"
    cat_dir.mkdir(parents=True, exist_ok=True)

    cat_links: List[Tuple[str, str]] = []
    for cat in sorted(subs["ref_cat"].unique()):
        cdf = subs[subs["ref_cat"] == cat]
        top_pairs = (
            cdf.groupby(["ref_norm", "hyp_word"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(top_k)
        )
        fname = f"{slugify(cat)}_top_pairs_top{top_k}.html"
        fpath = cat_dir / fname
        write_table_html(top_pairs, f"Top {top_k} substitution pairs — {cat}", fpath)
        items.append((f"Top pairs — {cat}", f"per_category/{fname}"))
        cat_links.append((f"{cat} — top pairs (K={top_k})", fname))

    # local index for these pages
    write_index_html(
        cat_links, cat_dir / "index_pairs.html", "Per-category: Top substitution pairs"
    )
    items.append(
        (
            "Per-category: Top substitution pairs (index)",
            "per_category/index_pairs.html",
        )
    )
    return items


def per_category_confusion_matrices(
    df: pd.DataFrame, out_dir: Path, top_k: int = 100
) -> List[Tuple[str, str]]:
    """For each category, build a ref_norm × hyp_word confusion matrix limited to top-K rows/cols."""
    items: List[Tuple[str, str]] = []
    subs = (
        df[df["error_type"] == "sub"]
        .dropna(subset=["ref_cat", "ref_norm", "hyp_word"])
        .copy()
    )
    if subs.empty:
        return items

    cat_dir = out_dir / "per_category"
    cat_dir.mkdir(parents=True, exist_ok=True)

    cat_links: List[Tuple[str, str]] = []
    for cat in sorted(subs["ref_cat"].unique()):
        cdf = subs[subs["ref_cat"] == cat]
        pair_counts = (
            cdf.groupby(["ref_norm", "hyp_word"]).size().reset_index(name="count")
        )

        # pick top-K rows/cols by marginal totals
        top_ref = (
            pair_counts.groupby("ref_norm")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(top_k)
            .index
        )
        top_hyp = (
            pair_counts.groupby("hyp_word")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(top_k)
            .index
        )

        filt = pair_counts[
            pair_counts["ref_norm"].isin(top_ref)
            & pair_counts["hyp_word"].isin(top_hyp)
        ]
        if filt.empty:
            continue

        pivot = (
            filt.pivot_table(
                index="ref_norm", columns="hyp_word", values="count", fill_value=0
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

        # Table view
        tbl_name = f"{slugify(cat)}_confusion_matrix_top{top_k}.html"
        tbl_path = cat_dir / tbl_name
        write_table_html(
            pivot.reset_index(),
            f"{cat}: Confusion matrix (top {top_k}×{top_k})",
            tbl_path,
        )
        items.append((f"{cat} — confusion matrix (table)", f"per_category/{tbl_name}"))

        # Heatmap view
        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values, x=pivot.columns, y=pivot.index, coloraxis="coloraxis"
            )
        )
        fig.update_layout(
            coloraxis=dict(colorscale="Blues"),
            title=f"{cat}: Confusion matrix (top {top_k}×{top_k})",
            xaxis_nticks=36,
            yaxis_nticks=36,
            margin=dict(l=160, b=160),
        )
        hm_name = f"{slugify(cat)}_confusion_heatmap_top{top_k}.html"
        write_plot_html(fig, cat_dir / hm_name)
        items.append((f"{cat} — confusion heatmap", f"per_category/{hm_name}"))

        cat_links.append((f"{cat} — matrix (table, K={top_k})", tbl_name))
        cat_links.append((f"{cat} — heatmap (K={top_k})", hm_name))

    # local index for these pages
    write_index_html(
        cat_links, cat_dir / "index_confusions.html", "Per-category: Confusion matrices"
    )
    items.append(
        (
            "Per-category: Confusion matrices (index)",
            "per_category/index_confusions.html",
        )
    )
    return items


def position_binned_by_cat(
    df: pd.DataFrame,
    out_dir: Path,
    bins: int = 10,
    exclude_fullspan_dels: bool = True,
    top_cats: int = 6,
) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    df2 = compute_utterance_durations(df.copy())
    df2["mid"] = df2.apply(midpoint, axis=1)
    if exclude_fullspan_dels:
        mask = (
            (df2["error_type"] == "del")
            & (abs(df2["start_time"] - df2["utt_start"]) < 1e-8)
            & (abs(df2["end_time"] - df2["utt_end"]) < 1e-8)
        )
        df2 = df2[~mask]
    df2["pos_norm"] = df2.apply(
        lambda r: safe_div((r["mid"] - r["utt_start"]), r["utt_dur"]), axis=1
    )
    df2 = df2.replace([np.inf, -np.inf], np.nan).dropna(subset=["pos_norm", "ref_cat"])
    if df2.empty:
        return items
    df2["pos_bin"] = (df2["pos_norm"].clip(0, 0.999) * bins).astype(int)

    counts = (
        df2.groupby(["ref_cat", "error_type", "pos_bin"])
        .size()
        .reset_index(name="count")
        .sort_values(["ref_cat", "error_type", "pos_bin"])
    )
    write_table_html(
        counts,
        f"Position-binned error counts by category (bins={bins})",
        out_dir / "position_binned_by_cat.html",
    )
    items.append(
        (
            f"Position-binned error counts by category (bins={bins})",
            "position_binned_by_cat.html",
        )
    )

    # plot only top categories by total volume for readability
    topcats = (
        df.groupby("ref_cat")
        .size()
        .sort_values(ascending=False)
        .head(top_cats)
        .index.tolist()
    )
    plot_df = counts[counts["ref_cat"].isin(topcats)].copy()
    if not plot_df.empty:
        fig = px.bar(
            plot_df,
            x="pos_bin",
            y="count",
            color="error_type",
            barmode="group",
            facet_col="ref_cat",
            facet_col_wrap=3,
            title=f"Position-binned error counts by category (top {top_cats} cats)",
        )
        write_plot_html(fig, out_dir / "position_binned_by_cat_plot.html")
        items.append(
            (
                "Bar: position-binned by category (top cats)",
                "position_binned_by_cat_plot.html",
            )
        )
    return items


def duration_effects_by_cat(
    df: pd.DataFrame, out_dir: Path, top_cats: int = 6
) -> List[Tuple[str, str]]:
    """Compute a truer errors/min per category using utterance-level aggregation.
    For each (uid, ref_cat): use that uid's utt_dur once; sum errors for that uid & category.
    """
    items: List[Tuple[str, str]] = []
    df_utts = compute_utterance_durations(df.copy())
    if df_utts["utt_dur"].dropna().empty:
        empty = pd.DataFrame(
            columns=[
                "ref_cat",
                "duration_bucket",
                "n_utterances",
                "total_errors",
                "total_dur_s",
                "errors_per_min",
            ]
        )
        write_table_html(
            empty,
            "Duration effects by category (no durations found)",
            out_dir / "duration_effects_by_cat.html",
        )
        return [("Duration effects by category", "duration_effects_by_cat.html")]

    # per uid durations
    per_uid = df_utts.groupby("uid").agg(utt_dur=("utt_dur", "max")).reset_index()

    # errors per (uid, category)
    err_uc = df_utts.groupby(["uid", "ref_cat"]).size().reset_index(name="n_errors")
    # join durations
    err_uc = err_uc.merge(per_uid, on="uid", how="left")
    err_uc = err_uc.dropna(subset=["ref_cat", "utt_dur"])

    # bucket by duration
    buckets = pd.cut(
        err_uc["utt_dur"],
        bins=[-np.inf, 1.0, 3.0, 6.0, 12.0, np.inf],
        labels=["<1s", "1–3s", "3–6s", "6–12s", ">12s"],
    )
    buckets.name = "duration_bucket"

    g = (
        err_uc.groupby(["ref_cat", buckets], observed=False)
        .agg(
            n_utterances=("uid", "nunique"),
            total_errors=("n_errors", "sum"),
            total_dur_s=("utt_dur", "sum"),
        )
        .reset_index()
    )

    g["errors_per_min"] = g.apply(
        lambda r: (
            safe_div(r["total_errors"], r["total_dur_s"] / 60.0)
            if pd.notna(r["total_dur_s"]) and r["total_dur_s"] > 0
            else np.nan
        ),
        axis=1,
    )

    write_table_html(
        g,
        "Duration bucket effects by category",
        out_dir / "duration_effects_by_cat.html",
    )
    items.append(
        ("Duration bucket effects by category", "duration_effects_by_cat.html")
    )

    # plot top categories
    topcats = (
        df.groupby("ref_cat")
        .size()
        .sort_values(ascending=False)
        .head(top_cats)
        .index.tolist()
    )
    plot_df = g[g["ref_cat"].isin(topcats)].copy()
    if not plot_df.empty:
        fig = px.bar(
            plot_df,
            x="duration_bucket",
            y="errors_per_min",
            color="ref_cat",
            barmode="group",
            title=f"Errors/min by duration bucket (top {top_cats} cats)",
        )
        write_plot_html(fig, out_dir / "duration_effects_by_cat_plot.html")
        items.append(
            (
                "Bar: errors/min by duration bucket (top cats)",
                "duration_effects_by_cat_plot.html",
            )
        )
    return items


def code_mix_by_cat(
    df: pd.DataFrame, language: str, out_dir: Path
) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    # compute out_of_script on ref_word only (domain focus)
    tmp = df[["ref_cat", "ref_word"]].copy()
    tmp["ref_word"] = tmp["ref_word"].fillna("")
    tmp["out_of_script"] = tmp["ref_word"].apply(
        lambda t: not is_in_script(str(t), language)
    )
    g = (
        tmp.groupby("ref_cat")
        .agg(total=("ref_word", "size"), out_of_script=("out_of_script", "sum"))
        .reset_index()
    )
    g["share_out_of_script"] = g.apply(
        lambda r: safe_div(r["out_of_script"], r["total"]), axis=1
    )
    write_table_html(
        g.sort_values("share_out_of_script", ascending=False),
        f"Code-mix (out-of-script share) by category ({language})",
        out_dir / "code_mix_by_cat.html",
    )
    items.append(("Code-mix by category", "code_mix_by_cat.html"))
    if not g.empty:
        fig = px.bar(
            g,
            x="ref_cat",
            y="share_out_of_script",
            title="Out-of-script share by ref_cat",
        )
        fig.update_layout(xaxis_tickangle=-45, margin=dict(b=160))
        write_plot_html(fig, out_dir / "code_mix_by_cat_plot.html")
        items.append(
            ("Bar: Out-of-script share by ref_cat", "code_mix_by_cat_plot.html")
        )
    return items


def snippet_index_by_cat(
    df: pd.DataFrame, out_dir: Path, top_n_per_cat: int = 40
) -> List[Tuple[str, str]]:
    """Export a review-ready table of examples per category for top problematic pairs/terms."""
    items: List[Tuple[str, str]] = []
    # choose top pairs by category
    subs = df[df["error_type"] == "sub"].copy()
    if subs.empty:
        return items
    top_pairs = (
        subs.groupby(["ref_cat", "ref_norm", "hyp_word"])
        .size()
        .reset_index(name="count")
        .sort_values(["ref_cat", "count"], ascending=[True, False])
    )
    top_pairs["rank"] = top_pairs.groupby("ref_cat")["count"].rank(
        method="first", ascending=False
    )
    tp = top_pairs[top_pairs["rank"] <= top_n_per_cat].drop(columns=["rank"])

    # sample a few rows per (ref_cat, ref_norm, hyp_word)
    samples: List[pd.DataFrame] = []
    for _, r in tp.iterrows():
        ex = subs[
            (subs["ref_cat"] == r["ref_cat"])
            & (subs["ref_norm"] == r["ref_norm"])
            & (subs["hyp_word"] == r["hyp_word"])
        ].head(3)
        ex = ex[
            [
                "ref_cat",
                "ref_norm",
                "hyp_word",
                "uid",
                "model",
                "start_time",
                "end_time",
                "clip_path",
            ]
        ]
        samples.append(ex)
    if samples:
        out = pd.concat(samples, ignore_index=True)
        write_table_html(
            out,
            f"Snippet index (top substitution pairs by category; up to {top_n_per_cat} per cat)",
            out_dir / "snippet_index_by_cat.html",
        )
        items.append(("Snippet index by category", "snippet_index_by_cat.html"))
    return items


# -----------------------------------------------------------------------------
# CLI & Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Domain-focused analytics for misrepresented terms using an LLM-classified glossary."
    )
    p.add_argument(
        "--language", type=str, help="Language folder name (default from config)"
    )
    p.add_argument(
        "--base-folder",
        type=Path,
        help="Base folder containing processed data (default from config)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        help="Subdirectory for analytics under <base>/<lang> (default from config)",
    )
    p.add_argument(
        "--glossary-jsonl",
        type=Path,
        help="Path to classified glossary JSONL (term,category)",
    )
    p.add_argument(
        "--include-generic",
        action="store_true",
        help="Include 'Generic'/'Not Applicable' categories (default: excluded)",
    )
    p.add_argument(
        "--keep-punct",
        action="store_true",
        help="Keep punctuation during normalization comparisons",
    )
    p.add_argument(
        "--normalize-digits",
        action="store_true",
        help="Normalize Unicode digits to ASCII",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K items per category for some tables (default=50)",
    )
    p.add_argument(
        "--top-cats",
        type=int,
        default=6,
        help="How many top categories to plot in faceted charts (default=6)",
    )
    p.add_argument(
        "--per-cat-top-k",
        type=int,
        default=100,
        help="Top-K per category for per-category tables/matrices (default=100)",
    )
    p.add_argument(
        "--bins", type=int, default=10, help="Number of position bins (default=10)"
    )
    p.add_argument(
        "--exclude-fullspan-dels",
        action="store_true",
        help="Exclude deletions with full-span timestamps from position charts",
    )
    return p.parse_args()


def main():
    # Defaults
    language = DEFAULT_CONFIG["language"]
    base_folder = Path(DEFAULT_CONFIG["base_folder"])
    output_subdir = DEFAULT_CONFIG["output_dir"]
    misrep_filename = DEFAULT_CONFIG["misrep_filename"]

    # Overrides
    args = parse_args()
    if args.language:
        language = args.language
    if args.base_folder:
        base_folder = args.base_folder
    if args.output_dir:
        output_subdir = args.output_dir

    logger.info(
        f"Configuration: language={language}, base_folder={base_folder}, output_dir={output_subdir}"
    )

    # Paths
    lang_root = base_folder / language
    analytics_dir = lang_root / output_subdir
    analytics_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    misrep_path = lang_root / misrep_filename
    if args.glossary_jsonl:
        glossary_path = args.glossary_jsonl
    else:
        glossary_path = lang_root / f"combined_glossary_terms_{language}.jsonl"

    # Load
    df_mis = load_misrep(misrep_path)
    g = load_glossary(
        glossary_path,
        language,
        keep_punct=args.keep_punct,
        normalize_digits=args.normalize_digits,
    )

    # Join + normalize
    df_join = join_with_glossary(
        df_mis,
        g,
        language,
        keep_punct=args.keep_punct,
        normalize_digits=args.normalize_digits,
    )

    # Coverage sanity
    index_items: List[Tuple[str, str]] = []
    index_items += glossary_coverage(df_join, analytics_dir)

    # Filter to domain (exclude Generic/NA unless requested)
    df_dom = filter_domain(df_join, include_generic=args.include_generic)
    if df_dom.empty:
        empty = pd.DataFrame(columns=["message"])
        empty.loc[0, "message"] = (
            "No domain-category rows after filtering. Try --include-generic or verify glossary."
        )
        write_table_html(empty, "No domain rows", analytics_dir / "no_domain_rows.html")
        write_index_html(
            [("No domain rows", "no_domain_rows.html")],
            analytics_dir / "index.html",
            f"Domain Analytics — {language}",
        )
        logger.info("No domain-category rows; exiting early.")
        return

    # Analytics
    index_items += category_composition(df_dom, analytics_dir)
    index_items += category_model_heatmap(df_dom, analytics_dir)
    index_items += formatting_only_share_by_cat(
        df_dom,
        language,
        analytics_dir,
        keep_punct=args.keep_punct,
        normalize_digits=args.normalize_digits,
    )
    index_items += top_problem_terms_by_cat(df_dom, analytics_dir, top_k=args.top_k)
    index_items += category_bounded_confusions(df_dom, analytics_dir, top_k=args.top_k)
    index_items += per_category_top_pairs(
        df_dom, analytics_dir, top_k=args.per_cat_top_k
    )
    index_items += per_category_confusion_matrices(
        df_dom, analytics_dir, top_k=args.per_cat_top_k
    )
    index_items += position_binned_by_cat(
        df_dom,
        analytics_dir,
        bins=args.bins,
        exclude_fullspan_dels=args.exclude_fullspan_dels,
        top_cats=args.top_cats,
    )
    index_items += duration_effects_by_cat(
        df_dom, analytics_dir, top_cats=args.top_cats
    )
    index_items += code_mix_by_cat(df_dom, language, analytics_dir)
    index_items += snippet_index_by_cat(
        df_dom, analytics_dir, top_n_per_cat=min(40, args.top_k)
    )

    # Index HTML
    write_index_html(
        index_items,
        analytics_dir / "index.html",
        heading=f"Domain Misrep Analytics — {language}",
    )

    logger.info("All domain analytics complete.")


if __name__ == "__main__":
    main()
