#!/usr/bin/env python3
"""
misrep_analytics.py

Combine misrepresented-terms JSONL files and generate analytics HTML reports.

Scans:
    PROCESSED-DATA-DND/<language>/*/misrep_terms.jsonl

Writes:
    - PROCESSED-DATA-DND/<language>/combined_misrep_terms.jsonl
    - PROCESSED-DATA-DND/<language>/misrep_analytics/*.html  (shareable, standalone)
    - PROCESSED-DATA-DND/<language>/misrep_analytics/index.html (table of contents)

Usage:
    # With defaults defined in DEFAULT_CONFIG:
    python misrep_analytics.py

    # Override defaults via CLI:
    python misrep_analytics.py \
      --language hindi \
      --base-folder PROCESSED-DATA-DND \
      --output-dir misrep_analytics \
      --critical-terms data/hindi_critical_terms.txt \
      --normalize-digits

Notes:
    - The script is language-aware for Indian scripts using SCRIPT_RANGES.
    - All analytics are exported as HTML (tables/plots) for easy sharing.
    - Dependencies: pandas, plotly, numpy
      pip install pandas plotly numpy
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterable
import argparse
import json
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
    # All analytics HTML files will be written under:
    # PROCESSED-DATA-DND/<language>/<output_dir>/
    "output_dir": "misrep_analytics",
    # Combined JSONL filename (under base_folder/<language>)
    "combined_filename": "combined_misrep_terms.jsonl",
}


# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
try:
    # If your project provides utils.logger, use that for consistency.
    from utils.logger import logger
except Exception:
    # Fallback to standard logging.
    logger = logging.getLogger("misrep_analytics")
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    if not logger.handlers:
        logger.addHandler(_handler)


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def get_jsonl_files(base_folder: Path, language: str) -> List[Path]:
    """Return list of misrep_terms.jsonl files under PROCESSED-DATA-DND/<language>/*/"""
    lang_dir = base_folder / language
    if not lang_dir.exists():
        logger.error(f"Language folder not found: {lang_dir}")
        return []
    files = sorted(lang_dir.glob("*/misrep_terms.jsonl"))
    if not files:
        logger.warning(f"No misrep_terms.jsonl files found in {lang_dir}")
    return files


def combine_jsonl(files: List[Path], out_path: Path) -> int:
    """Concatenate JSONL files into a single JSONL; returns number of lines written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as w:
        for fp in files:
            logger.info(f"Combining: {fp}")
            with fp.open("r", encoding="utf-8") as r:
                for line in r:
                    line = line.strip()
                    if not line:
                        continue
                    # quick validation of JSON line
                    try:
                        json.loads(line)
                        w.write(line + "\n")
                        n += 1
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {fp}")
    logger.info(f"Wrote {n} lines to {out_path}")
    return n


def load_combined_df(path: Path) -> pd.DataFrame:
    """Load combined JSONL into DataFrame; ensure columns exist with reasonable dtypes."""
    if not path.exists():
        raise FileNotFoundError(f"Combined JSONL not found: {path}")
    df = pd.read_json(path, lines=True, dtype=False)
    # Ensure expected columns exist
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

    # Coerce types
    df["model"] = df["model"].astype("string")
    df["error_type"] = df["error_type"].astype("string")
    for col in ["ref_word", "hyp_word", "_id", "uid", "clip_path"]:
        df[col] = df[col].astype("string")
    for col in ["start_time", "end_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows missing minimal essentials
    df = df.dropna(subset=["error_type"])
    return df


# -----------------------------------------------------------------------------
# Normalization & script utilities
# -----------------------------------------------------------------------------
def script_regex(language: str) -> Optional[re.Pattern]:
    """Compile a regex that matches characters in the primary script block for the language."""
    rng = SCRIPT_RANGES.get(language.lower())
    if not rng:
        return None
    return re.compile(rf"[{rng}]+", flags=re.UNICODE)


def is_in_script(token: str, language: str) -> bool:
    """Return True if the majority of letters in token are in the target script range."""
    rgx = script_regex(language)
    if not token or rgx is None:
        return False
    letters = [ch for ch in token if ch.isalpha()]
    if not letters:
        return False
    in_count = sum(1 for ch in letters if rgx.fullmatch(ch) or re.match(rgx, ch or ""))
    return in_count >= max(1, int(0.6 * len(letters)))


def strip_punct_keep_script(token: str, language: str) -> str:
    """Remove punctuation/marks while keeping letters/digits in the language script and ASCII word chars."""
    rng = SCRIPT_RANGES.get(language.lower(), "")
    # keep word chars (\w), script letters, and decimal digits
    return re.sub(rf"[^\w{rng}\d]+", "", token, flags=re.UNICODE)


def normalize_digits_to_ascii(token: str) -> str:
    """Map any Unicode decimal digits to ASCII '0'-'9'."""
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
    """Unicode NFC, optional punctuation stripping (language-aware), optional digit normalization to ASCII."""
    if token is None or (isinstance(token, float) and math.isnan(token)):
        return ""
    t = unicodedata.normalize("NFC", str(token))
    if not keep_punct:
        t = strip_punct_keep_script(t, language)
    if normalize_digits:
        t = normalize_digits_to_ascii(t)
    return t.strip()


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------
def compute_utterance_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-uid approximate utterance duration from min(start) to max(end)."""
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


def midpoint(row: pd.Series) -> float:
    """Token midpoint time (s)."""
    st = row.get("start_time", np.nan)
    ed = row.get("end_time", np.nan)
    if pd.notna(st) and pd.notna(ed):
        return (st + ed) / 2.0
    return np.nan


def safe_div(a: float, b: float) -> float:
    """Safe division that returns NaN for invalid cases."""
    return float(a) / float(b) if (b and not pd.isna(b)) else np.nan


# -----------------------------------------------------------------------------
# HTML writers
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
# Analytics functions (each returns list of (title, filename) for index)
# -----------------------------------------------------------------------------
def summarize_errors(df: pd.DataFrame, out_dir: Path) -> List[Tuple[str, str]]:
    """Counts & rates by model × error_type; also per-UID density (errors per minute)."""
    items: List[Tuple[str, str]] = []

    # Counts by model x error_type
    pivot = (
        df.groupby(["model", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("model")
    )
    path_table = out_dir / "summary_model_error_type.html"
    write_table_html(pivot, "Error counts by model × error_type", path_table)
    items.append(("Error counts by model × error_type", path_table.name))

    # Bar plot
    melted = pivot.melt(id_vars=["model"], var_name="error_type", value_name="count")
    if not melted.empty:
        fig = px.bar(
            melted,
            x="model",
            y="count",
            color="error_type",
            barmode="group",
            text_auto=True,
            title="Error counts by model × error_type",
        )
        path_plot = out_dir / "summary_model_error_type_plot.html"
        write_plot_html(fig, path_plot)
        items.append(("Bar plot: model × error_type", path_plot.name))

    # Per-UID density (errors per minute)
    df_utts = compute_utterance_durations(df.copy())
    per_uid = (
        df_utts.groupby("uid")
        .agg(n_errors=("uid", "size"), utt_dur=("utt_dur", "max"))
        .reset_index()
    )
    per_uid["errors_per_min"] = per_uid.apply(
        lambda r: (
            safe_div(r["n_errors"], r["utt_dur"] / 60.0)
            if pd.notna(r["utt_dur"]) and r["utt_dur"] > 0
            else np.nan
        ),
        axis=1,
    )
    per_uid = per_uid.sort_values("errors_per_min", ascending=False)

    path_uid = out_dir / "summary_per_uid_density.html"
    write_table_html(
        per_uid.head(500), "Top utterances by errors per minute (top 500)", path_uid
    )
    items.append(("Per-UID errors-per-minute (top 500)", path_uid.name))

    return items


def top_confusion_pairs(
    df: pd.DataFrame,
    language: str,
    out_dir: Path,
    k: int = 200,
    normalize: bool = False,
    keep_punct: bool = False,
    normalize_digits: bool = True,
) -> List[Tuple[str, str]]:
    """Top substitution pairs ref_word → hyp_word; optionally normalized view."""
    items: List[Tuple[str, str]] = []
    subs = df[df["error_type"] == "sub"].copy()

    if subs.empty:
        return items

    if normalize:
        subs["ref_norm"] = subs["ref_word"].apply(
            lambda s: normalize_token(s, language, keep_punct, normalize_digits)
        )
        subs["hyp_norm"] = subs["hyp_word"].apply(
            lambda s: normalize_token(s, language, keep_punct, normalize_digits)
        )
        grp = (
            subs.groupby(["ref_norm", "hyp_norm"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        grp = grp.rename(columns={"ref_norm": "ref_word", "hyp_norm": "hyp_word"})
        title = f"Top substitution pairs (normalized) — top {k}"
        fname = "top_confusions_normalized.html"
        plot_name = "top_confusions_normalized_plot.html"
    else:
        grp = (
            subs.groupby(["ref_word", "hyp_word"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        title = f"Top substitution pairs (raw) — top {k}"
        fname = "top_confusions_raw.html"
        plot_name = "top_confusions_raw_plot.html"

    grp_top = grp.head(k)
    write_table_html(grp_top, title, out_dir / fname)
    items.append((title, fname))

    if not grp_top.empty:
        # Treemap plot for compact visual overview
        fig = px.treemap(
            grp_top,
            path=[px.Constant("subs"), "ref_word", "hyp_word"],
            values="count",
            title=title,
        )
        write_plot_html(fig, out_dir / plot_name)
        items.append((f"Treemap: {title}", plot_name))
    return items


def problem_terms(
    df: pd.DataFrame, out_dir: Path, k: int = 200
) -> List[Tuple[str, str]]:
    """Identify ref_word hotspots (sub+del) and hyp_word hotspots (sub+ins)."""
    items: List[Tuple[str, str]] = []
    ref_bad = (
        df[df["error_type"].isin(["sub", "del"])]
        .groupby("ref_word")
        .size()
        .reset_index(name="count")
    )
    hyp_bad = (
        df[df["error_type"].isin(["sub", "ins"])]
        .groupby("hyp_word")
        .size()
        .reset_index(name="count")
    )

    ref_top = ref_bad.sort_values("count", ascending=False).head(k)
    hyp_top = hyp_bad.sort_values("count", ascending=False).head(k)

    write_table_html(
        ref_top,
        f"Problem terms (reference) — top {k}",
        out_dir / "problem_terms_ref.html",
    )
    write_table_html(
        hyp_top,
        f"Problem terms (hypothesis) — top {k}",
        out_dir / "problem_terms_hyp.html",
    )
    items.append((f"Problem terms (reference) — top {k}", "problem_terms_ref.html"))
    items.append((f"Problem terms (hypothesis) — top {k}", "problem_terms_hyp.html"))

    # Simple bar plots
    if not ref_top.empty:
        fig1 = px.bar(
            ref_top,
            x="ref_word",
            y="count",
            title=f"Problem terms (reference) — top {k}",
        )
        fig1.update_layout(xaxis_tickangle=-45, margin=dict(b=160))
        write_plot_html(fig1, out_dir / "problem_terms_ref_plot.html")
        items.append(("Bar: problem terms (reference)", "problem_terms_ref_plot.html"))

    if not hyp_top.empty:
        fig2 = px.bar(
            hyp_top,
            x="hyp_word",
            y="count",
            title=f"Problem terms (hypothesis) — top {k}",
        )
        fig2.update_layout(xaxis_tickangle=-45, margin=dict(b=160))
        write_plot_html(fig2, out_dir / "problem_terms_hyp_plot.html")
        items.append(("Bar: problem terms (hypothesis)", "problem_terms_hyp_plot.html"))

    return items


def confusion_heatmap(
    df: pd.DataFrame, out_dir: Path, top_ref: int = 50, top_hyp: int = 50
) -> List[Tuple[str, str]]:
    """Heatmap of top substitution pairs (rows = ref_word, cols = hyp_word)."""
    items: List[Tuple[str, str]] = []
    subs = df[df["error_type"] == "sub"].copy()
    if subs.empty:
        return items

    top_ref_words = subs.groupby("ref_word").size().nlargest(top_ref).index.tolist()
    top_hyp_words = subs.groupby("hyp_word").size().nlargest(top_hyp).index.tolist()
    mat = (
        subs[
            subs["ref_word"].isin(top_ref_words) & subs["hyp_word"].isin(top_hyp_words)
        ]
        .groupby(["ref_word", "hyp_word"])
        .size()
        .unstack(fill_value=0)
    )

    # Save as HTML table for inspection
    write_table_html(
        mat.reset_index(),
        "Confusion matrix (table view)",
        out_dir / "confusion_matrix_table.html",
    )
    items.append(("Confusion matrix (table view)", "confusion_matrix_table.html"))

    # Plotly heatmap
    fig = go.Figure(data=go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, zmin=0))
    fig.update_layout(
        title="Confusion heatmap (top ref × top hyp)",
        xaxis_nticks=36,
        yaxis_nticks=36,
        margin=dict(l=160, b=160),
    )
    write_plot_html(fig, out_dir / "confusion_heatmap.html")
    items.append(("Confusion heatmap (plot)", "confusion_heatmap.html"))
    return items


def position_binned_errors(
    df: pd.DataFrame, out_dir: Path, bins: int = 10
) -> List[Tuple[str, str]]:
    """Bin errors by normalized token midpoint within utterance duration; split by error_type."""
    items: List[Tuple[str, str]] = []
    df_utts = compute_utterance_durations(df.copy())
    df_utts["mid"] = df_utts.apply(midpoint, axis=1)
    # normalized position in [0,1]
    df_utts["pos_norm"] = df_utts.apply(
        lambda r: safe_div((r["mid"] - r["utt_start"]), r["utt_dur"]), axis=1
    )
    df_utts = df_utts.replace([np.inf, -np.inf], np.nan).dropna(subset=["pos_norm"])

    if df_utts.empty:
        return items

    # Bin into deciles by default
    df_utts["pos_bin"] = (df_utts["pos_norm"].clip(0, 0.999) * bins).astype(int)

    pos = (
        df_utts.groupby(["error_type", "pos_bin"])
        .size()
        .reset_index(name="count")
        .sort_values(["error_type", "pos_bin"])
    )
    write_table_html(
        pos,
        f"Position-binned error counts (bins={bins})",
        out_dir / "position_binned_counts.html",
    )
    items.append(
        (f"Position-binned error counts (bins={bins})", "position_binned_counts.html")
    )

    if not pos.empty:
        fig = px.bar(
            pos,
            x="pos_bin",
            y="count",
            color="error_type",
            barmode="group",
            title=f"Position-binned error counts (bins={bins})",
            text_auto=True,
        )
        write_plot_html(fig, out_dir / "position_binned_plot.html")
        items.append(("Bar: position-binned error counts", "position_binned_plot.html"))
    return items


def duration_effects(df: pd.DataFrame, out_dir: Path) -> List[Tuple[str, str]]:
    """Bucket utterances by duration and compute error rates (errors per minute)."""
    items: List[Tuple[str, str]] = []
    df_utts = compute_utterance_durations(df.copy())

    # If no durations, write a minimal table and return
    if df_utts["utt_dur"].dropna().empty:
        empty = pd.DataFrame(
            columns=["duration_bucket", "n_errors", "total_dur_s", "errors_per_min"]
        )
        write_table_html(
            empty,
            "Duration bucket effects (no durations found)",
            out_dir / "duration_effects.html",
        )
        return [("Duration bucket effects", "duration_effects.html")]

    # Name the bucket series to avoid index/column collisions and silence FutureWarning via observed=False
    buckets = pd.cut(
        df_utts["utt_dur"],
        bins=[-np.inf, 1.0, 3.0, 6.0, 12.0, np.inf],
        labels=["<1s", "1–3s", "3–6s", "6–12s", ">12s"],
    )
    buckets.name = "duration_bucket"

    g = (
        df_utts.groupby(buckets, observed=False)
        .agg(
            n_errors=("uid", "size"),
            total_dur_s=(
                "utt_dur",
                "sum",
            ),  # <-- give the agg column a different name up front
        )
        .reset_index()
    )

    g["errors_per_min"] = g.apply(
        lambda r: (
            safe_div(r["n_errors"], r["total_dur_s"] / 60.0)
            if pd.notna(r["total_dur_s"]) and r["total_dur_s"] > 0
            else np.nan
        ),
        axis=1,
    )

    write_table_html(g, "Duration bucket effects", out_dir / "duration_effects.html")
    items.append(("Duration bucket effects", "duration_effects.html"))

    if not g.empty:
        fig = px.bar(
            g,
            x="duration_bucket",
            y="errors_per_min",
            title="Errors per minute by duration bucket",
            text_auto=True,
        )
        write_plot_html(fig, out_dir / "duration_effects_plot.html")
        items.append(
            ("Bar: errors per minute by duration bucket", "duration_effects_plot.html")
        )

    return items


def code_mix_stats(
    df: pd.DataFrame, language: str, out_dir: Path
) -> List[Tuple[str, str]]:
    """Quantify tokens outside the primary script for the language (approx)."""
    items: List[Tuple[str, str]] = []

    # For substitutions, inspect both ref and hyp. For ins, only hyp; for del, only ref.
    def token_rows():
        for _, r in df.iterrows():
            et = str(r["error_type"])
            if et == "sub":
                yield ("ref", r["ref_word"])
                yield ("hyp", r["hyp_word"])
            elif et == "ins":
                yield ("hyp", r["hyp_word"])
            elif et == "del":
                yield ("ref", r["ref_word"])

    rows = list(token_rows())
    if not rows:
        return items
    tmp = pd.DataFrame(rows, columns=["role", "token"])
    tmp["token"] = tmp["token"].fillna("")
    tmp["in_script"] = tmp["token"].apply(lambda t: is_in_script(str(t), language))
    tmp["out_of_script"] = ~tmp["in_script"]

    agg = (
        tmp.groupby("role")
        .agg(total=("token", "size"), out_of_script=("out_of_script", "sum"))
        .reset_index()
    )
    agg["share_out_of_script"] = agg.apply(
        lambda r: safe_div(r["out_of_script"], r["total"]), axis=1
    )

    write_table_html(
        agg, f"Code-mix stats (script={language})", out_dir / "code_mix_stats.html"
    )
    items.append(("Code-mix stats", "code_mix_stats.html"))

    fig = px.bar(
        agg,
        x="role",
        y="share_out_of_script",
        title=f"Share of out-of-script tokens ({language})",
    )
    write_plot_html(fig, out_dir / "code_mix_stats_plot.html")
    items.append(("Bar: share of out-of-script tokens", "code_mix_stats_plot.html"))
    return items


def punctuation_itn_confusions(
    df: pd.DataFrame, language: str, out_dir: Path
) -> List[Tuple[str, str]]:
    """Pairs that differ only by punctuation/formatting after normalization; estimates formatting-only error share."""
    items: List[Tuple[str, str]] = []
    subs = df[df["error_type"] == "sub"].copy()
    if subs.empty:
        return items

    subs["ref_norm"] = subs["ref_word"].apply(
        lambda s: normalize_token(s, language, keep_punct=False)
    )
    subs["hyp_norm"] = subs["hyp_word"].apply(
        lambda s: normalize_token(s, language, keep_punct=False)
    )

    # Formatting-only if normalized tokens equal but raw differ
    fmt_only = subs[
        (subs["ref_norm"] == subs["hyp_norm"]) & (subs["ref_word"] != subs["hyp_word"])
    ]
    share = safe_div(len(fmt_only), len(subs))

    table = pd.DataFrame(
        {
            "total_substitutions": [len(subs)],
            "formatting_only_subs": [len(fmt_only)],
            "formatting_only_share": [share],
        }
    )
    write_table_html(
        table, "Formatting-only substitutions", out_dir / "formatting_only_subs.html"
    )
    items.append(("Formatting-only substitutions", "formatting_only_subs.html"))

    # Show top raw pairs that collapse after normalization
    top_fmt_pairs = (
        fmt_only.groupby(["ref_word", "hyp_word"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(100)
    )
    write_table_html(
        top_fmt_pairs,
        "Top formatting-only pairs (raw)",
        out_dir / "top_formatting_only_pairs.html",
    )
    items.append(("Top formatting-only pairs (raw)", "top_formatting_only_pairs.html"))
    return items


def critical_term_error_rate(
    df: pd.DataFrame,
    language: str,
    out_dir: Path,
    critical_terms: Optional[Iterable[str]] = None,
) -> List[Tuple[str, str]]:
    """Compute error share for a provided critical-term lexicon (case/normalization aware)."""
    items: List[Tuple[str, str]] = []
    if not critical_terms:
        return items
    # Normalize terms set
    crit_norm = {
        normalize_token(t.strip(), language) for t in critical_terms if t and t.strip()
    }
    crit_norm.discard("")

    # Count occurrences where reference is a critical term and got sub/del
    df_norm = df.copy()
    df_norm["ref_norm"] = df_norm["ref_word"].apply(
        lambda s: normalize_token(s, language)
    )
    crit_rows = df_norm[df_norm["ref_norm"].isin(crit_norm)]
    total_occ = len(crit_rows)
    err_rows = crit_rows[crit_rows["error_type"].isin(["sub", "del"])]
    err_occ = len(err_rows)
    share = safe_div(err_occ, total_occ)

    tbl = pd.DataFrame(
        {
            "critical_ref_occurrences": [total_occ],
            "critical_ref_errors_sub+del": [err_occ],
            "critical_term_error_rate": [share],
        }
    )
    write_table_html(
        tbl,
        "Critical Term Error Rate (CTER)",
        out_dir / "critical_term_error_rate.html",
    )
    items.append(("Critical Term Error Rate (CTER)", "critical_term_error_rate.html"))

    # Top problematic critical terms
    top_terms = (
        err_rows.groupby("ref_norm")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .rename(columns={"ref_norm": "critical_term"})
        .head(200)
    )
    write_table_html(
        top_terms, "Top problematic critical terms", out_dir / "critical_terms_top.html"
    )
    items.append(("Top problematic critical terms", "critical_terms_top.html"))
    return items


def engine_disagreement(
    df: pd.DataFrame, out_dir: Path, time_tolerance: float = 0.35
) -> List[Tuple[str, str]]:
    """Estimate consensus vs disagreement across engines by uid and time-window overlap.
    Approach: within each uid, bucket midpoints into small windows; count per model per window.
    """
    items: List[Tuple[str, str]] = []
    if "model" not in df.columns or df["model"].nunique() <= 1:
        return items

    df2 = df.copy()
    df2["mid"] = df2.apply(midpoint, axis=1)
    # time windows of width `time_tolerance`, index by floor(mid / tol)
    df2["win"] = (df2["mid"] / time_tolerance).apply(
        lambda x: int(x) if pd.notna(x) else -1
    )
    grouped = (
        df2.dropna(subset=["win"])
        .groupby(["uid", "win", "model"])
        .size()
        .reset_index(name="n")
    )
    # count distinct models that reported at least one error in the same window
    consensus = (
        grouped.groupby(["uid", "win"])
        .agg(models_involved=("model", "nunique"), total_errors=("n", "sum"))
        .reset_index()
    )

    # summarize
    dist = consensus["models_involved"].value_counts().reset_index()
    dist.columns = ["models_involved", "windows"]
    write_table_html(
        dist,
        "Engine disagreement distribution (by time window)",
        out_dir / "engine_disagreement.html",
    )
    items.append(
        (
            "Engine disagreement distribution (by time window)",
            "engine_disagreement.html",
        )
    )

    if not dist.empty:
        fig = px.bar(
            dist.sort_values("models_involved"),
            x="models_involved",
            y="windows",
            title="Windows by number of models involved (higher → more consensus)",
            text_auto=True,
        )
        write_plot_html(fig, out_dir / "engine_disagreement_plot.html")
        items.append(
            ("Bar: engine disagreement distribution", "engine_disagreement_plot.html")
        )
    return items


def export_snippet_index(
    df: pd.DataFrame, out_dir: Path, top_n_per_type: int = 50
) -> List[Tuple[str, str]]:
    """Export a review-ready table of top errors per type with clip paths and time spans."""
    items: List[Tuple[str, str]] = []
    rows = []
    for et in ["sub", "ins", "del"]:
        dd = df[df["error_type"] == et]
        if dd.empty:
            continue
        if et == "sub":
            key_cols = ["ref_word", "hyp_word"]
        elif et == "ins":
            key_cols = ["hyp_word"]
        else:
            key_cols = ["ref_word"]

        top = (
            dd.groupby(key_cols)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(top_n_per_type)
        )
        top["error_type"] = et
        rows.append(top)

    if not rows:
        return items

    top_all = pd.concat(rows, ignore_index=True)

    # Join back example rows for each key
    examples = []
    for _, r in top_all.iterrows():
        et = r["error_type"]
        if et == "sub":
            ex = df[
                (df["error_type"] == et)
                & (df["ref_word"] == r["ref_word"])
                & (df["hyp_word"] == r["hyp_word"])
            ].head(3)
        elif et == "ins":
            ex = df[(df["error_type"] == et) & (df["hyp_word"] == r["hyp_word"])].head(
                3
            )
        else:
            ex = df[(df["error_type"] == et) & (df["ref_word"] == r["ref_word"])].head(
                3
            )
        ex = ex[
            [
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
        ]
        examples.append(ex)

    examples_df = pd.concat(examples, ignore_index=True) if examples else pd.DataFrame()
    write_table_html(
        examples_df,
        "Snippet index (top-N examples per error type)",
        out_dir / "snippet_index.html",
    )
    items.append(("Snippet index (examples)", "snippet_index.html"))
    return items


# -----------------------------------------------------------------------------
# CLI & Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine misrepresented-terms JSONL files and generate analytics HTML reports."
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
        "--critical-terms",
        type=Path,
        help="Path to newline-separated critical terms (optional)",
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
        "--skip-combine",
        action="store_true",
        help="Skip combining step and just read existing combined JSONL",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=500,
        help="Top-K items to display for some tables (default=200)",
    )
    p.add_argument(
        "--bins", type=int, default=10, help="Number of position bins (default=10)"
    )
    return p.parse_args()


def main():
    # Defaults
    language = DEFAULT_CONFIG["language"]
    base_folder = Path(DEFAULT_CONFIG["base_folder"])
    output_subdir = DEFAULT_CONFIG["output_dir"]
    combined_filename = DEFAULT_CONFIG["combined_filename"]

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
    combined_path = lang_root / combined_filename

    # Combine
    if not args.skip_combine:
        files = get_jsonl_files(base_folder, language)
        if not files:
            logger.error("No files to combine. Exiting.")
            return
        n = combine_jsonl(files, combined_path)
        if n == 0:
            logger.error("Combined file is empty. Exiting.")
            return
    else:
        logger.info("Skipping combine step as requested.")

    # Load combined
    df = load_combined_df(combined_path)

    # Optional: load critical terms
    crit_terms: Optional[List[str]] = None
    if args.critical_terms and args.critical_terms.exists():
        crit_terms = [
            ln.strip()
            for ln in args.critical_terms.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        logger.info(f"Loaded {len(crit_terms)} critical terms")

    # Generate analytics
    index_items: List[Tuple[str, str]] = []

    index_items += summarize_errors(df, analytics_dir)
    index_items += top_confusion_pairs(
        df, language, analytics_dir, k=args.top_k, normalize=False
    )
    index_items += top_confusion_pairs(
        df,
        language,
        analytics_dir,
        k=args.top_k,
        normalize=True,
        keep_punct=args.keep_punct,
        normalize_digits=args.normalize_digits,
    )
    index_items += confusion_heatmap(df, analytics_dir, top_ref=50, top_hyp=50)
    index_items += problem_terms(df, analytics_dir, k=args.top_k)
    index_items += position_binned_errors(df, analytics_dir, bins=args.bins)
    index_items += duration_effects(df, analytics_dir)
    index_items += code_mix_stats(df, language, analytics_dir)
    index_items += punctuation_itn_confusions(df, language, analytics_dir)
    index_items += critical_term_error_rate(
        df, language, analytics_dir, critical_terms=crit_terms
    )
    index_items += engine_disagreement(df, analytics_dir)
    index_items += export_snippet_index(
        df, analytics_dir, top_n_per_type=min(50, args.top_k)
    )

    # Index HTML
    write_index_html(
        index_items,
        analytics_dir / "index.html",
        heading=f"Misrep Analytics — {language}",
    )

    logger.info("All analytics complete.")


if __name__ == "__main__":
    main()
