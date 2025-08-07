#!/usr/bin/env python3
"""
glossary_category_distribution.py

Combine glossary extraction entries across JSONL files, deduplicate terms, compute category distribution,
and generate an interactive bar chart saved as an HTML file.

Usage:
    # With defaults defined in DEFAULT_CONFIG:
    python glossary_category_distribution.py

    # Override defaults via CLI:
    python glossary_category_distribution.py \
      --language hindi \
      --base-folder PROCESSED-DATA-DND \
      --output-dir results \
      --output-html category_distribution.html
"""

import argparse
import json
from pathlib import Path
from typing import List, Set, Tuple, Dict

import pandas as pd
import plotly.express as px
from utils.logger import logger

# Default configuration
DEFAULT_CONFIG: Dict[str, str] = {
    "language": "hindi",
    "base_folder": "PROCESSED-DATA-DND",
    "output_dir": "results",
    "output_html": "category_distribution.html",
}


def get_jsonl_files(base_folder: Path, language: str) -> List[Path]:
    lang_dir = base_folder / language
    if not lang_dir.exists():
        logger.error(f"Language folder not found: {lang_dir}")
        return []
    files = list(lang_dir.glob("*/glossary_terms.jsonl"))
    if not files:
        logger.warning(f"No glossary_terms.jsonl files found in {lang_dir}")
    return files


def load_entries(files: List[Path]) -> Set[Tuple[str, str]]:
    entries: Set[Tuple[str, str]] = set()
    for file in files:
        logger.info(f"Processing file: {file}")
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    term = record.get("term")
                    category = record.get("category")
                    if term and category:
                        entries.add((term, category))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line in {file}: {e}")
    return entries


def compute_distribution(entries: Set[Tuple[str, str]]) -> pd.Series:
    df = pd.DataFrame(entries, columns=["term", "category"])
    return df["category"].value_counts().sort_values(ascending=False)


def generate_plot(distribution: pd.Series, language: str, output_html: Path) -> None:
    df_plot = distribution.reset_index()
    df_plot.columns = ["category", "count"]

    fig = px.bar(
        df_plot,
        x="category",
        y="count",
        text_auto=True,  # automatically use the y-values as labels
        title=f"Glossary Category Distribution ({language})",
        labels={"count": "Number of Unique Terms", "category": "Category"},
    )
    fig.update_traces(textposition="outside")  # put labels above each bar
    fig.update_layout(
        xaxis_tickangle=-45,
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        margin=dict(t=50, b=100),
    )

    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)
    logger.info(f"Plot saved to {output_html}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate glossary category distribution plot."
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language folder name (overrides default)",
    )
    parser.add_argument(
        "--base-folder",
        type=Path,
        help="Base folder containing processed data (overrides default)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for combined JSONL and HTML plot (overrides default)",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        help="Output HTML file for the plot (overrides default)",
    )
    return parser.parse_args()


def main():
    # Load defaults
    language = DEFAULT_CONFIG["language"]
    base_folder = Path(DEFAULT_CONFIG["base_folder"])
    output_dir = Path(DEFAULT_CONFIG["output_dir"])
    output_html = Path(DEFAULT_CONFIG["output_html"])

    # Override with CLI args if provided
    args = parse_args()
    if args.language:
        language = args.language
    if args.base_folder:
        base_folder = args.base_folder
    if args.output_dir:
        output_dir = args.output_dir
    if args.output_html:
        output_html = args.output_html

    logger.info(
        f"Configuration: language={language}, base_folder={base_folder}, output_dir={output_dir}, output_html={output_html}"
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    files = get_jsonl_files(base_folder, language)
    if not files:
        logger.error("No files to process. Exiting.")
        return
    entries = load_entries(files)
    if not entries:
        logger.error("No entries loaded. Exiting.")
        return

    # Write combined JSONL
    combined_file = output_dir / "combined_glossary_terms.jsonl"
    with combined_file.open("w", encoding="utf-8") as cf:
        for term, category in sorted(entries):
            cf.write(
                json.dumps({"term": term, "category": category}, ensure_ascii=False)
                + "\n"
            )
    logger.info(f"Combined JSONL saved to {combined_file}")

    distribution = compute_distribution(entries)
    # Adjust HTML output path to output directory
    output_html = output_dir / output_html.name
    generate_plot(distribution, language, output_html)


if __name__ == "__main__":
    main()
