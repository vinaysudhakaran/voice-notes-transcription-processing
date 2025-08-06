#!/usr/bin/env python3
"""
reconstruct_manifest.py

Rebuilds the JSONL manifest and final JSON manifest for already-downloaded
audio and transcript files, skipping ffprobe metadata capture.

Configure the DEFAULT_CONFIG below, then simply run:
    python reconstruct_manifest.py

This will produce:
- manifest.jsonl (line-delimited JSON of successful entries)
- errors.jsonl   (line-delimited JSON of any missing/errored UIDs)
- manifest.json  (compiled JSON array for downstream use)
"""
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Union

import pandas as pd

# -----------------------------------------------------------------------------
# === DEFAULT CONFIG ===
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, str] = {
    "input_table": "dataset/metadata/hindi/hindi_transcription_audio_1.xlsx",
    "audio_dir": "dataset/audio/hindi/1",
    "transcripts_dir": "dataset/transcripts/hindi/1",
    "manifest_path": "dataset/manifest.json",
}

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_table(path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV or XLSX into a DataFrame."""
    path_obj = Path(path)
    ext = path_obj.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path_obj)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path_obj)
    raise ValueError(f"Unsupported file type: {ext}")


def compute_md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Core reconstruction function
# -----------------------------------------------------------------------------
def reconstruct_manifest(
    input_table: Union[str, Path],
    audio_dir: Union[str, Path],
    transcripts_dir: Union[str, Path],
    manifest_path: Union[str, Path],
) -> None:
    """
    Scan existing audio and transcripts to build manifest.jsonl, errors.jsonl,
    and compile manifest.json.
    """
    audio_dir = Path(audio_dir)
    transcripts_dir = Path(transcripts_dir)
    manifest_path = Path(manifest_path)
    manifest_jsonl = manifest_path.with_suffix(manifest_path.suffix + "l")
    errors_jsonl = manifest_path.parent / "errors.jsonl"

    # Remove old outputs
    for p in (manifest_jsonl, errors_jsonl, manifest_path):
        if p.exists():
            logger.info("Removing old file %s", p)
            p.unlink()

    # Prepare output files
    manifest_fp = manifest_jsonl.open("w", encoding="utf-8")
    errors_fp = errors_jsonl.open("w", encoding="utf-8")

    df = load_table(input_table)
    required = {"Filename", "Language", "Annotated Transcriptions"}
    if not required.issubset(df.columns):
        logger.error(
            "Input is missing required columns: %s", required - set(df.columns)
        )
        return

    count_ok = 0
    count_err = 0

    for _, row in df.iterrows():
        url = str(row["Filename"])
        lang = str(row["Language"])
        transcript = str(row["Annotated Transcriptions"])
        uid = Path(url).stem.split("_")[0]

        audio_path = audio_dir / f"{uid}.ogg"
        txt_path = transcripts_dir / f"{uid}.txt"

        if not audio_path.exists() or not txt_path.exists():
            err = {"uid": uid, "url": url, "error": "Missing audio or transcript file"}
            errors_fp.write(json.dumps(err, ensure_ascii=False) + "\n")
            count_err += 1
            continue

        try:
            size = audio_path.stat().st_size
            md5 = compute_md5(audio_path)
            wc = len(transcript.split())

            entry: Dict[str, Union[str, int, None]] = {
                "uid": uid,
                "url": url,
                "language": lang,
                "audio_path": str(audio_path),
                "transcript_path": str(txt_path),
                "duration_sec": None,
                "file_size_bytes": size,
                "bit_rate": None,
                "md5": md5,
                "transcript_word_count": wc,
            }
            manifest_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count_ok += 1
        except Exception as e:
            logger.error("Failed to process %s: %s", uid, e)
            err = {"uid": uid, "url": url, "error": str(e)}
            errors_fp.write(json.dumps(err, ensure_ascii=False) + "\n")
            count_err += 1
            continue

    manifest_fp.close()
    errors_fp.close()

    logger.info("Reconstructed manifest entries: %d, errors: %d", count_ok, count_err)

    # Compile JSONL to JSON
    try:
        records = []
        with manifest_jsonl.open("r", encoding="utf-8") as mf:
            for line in mf:
                records.append(json.loads(line))
        with manifest_path.open("w", encoding="utf-8") as f:
            import json as _json

            _json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info("Compiled %d records to %s", len(records), manifest_path)
    except Exception as e:
        logger.error("Error compiling final manifest.json: %s", e)


# -----------------------------------------------------------------------------
# Run reconstruction with DEFAULT_CONFIG
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = DEFAULT_CONFIG
    reconstruct_manifest(
        input_table=cfg["input_table"],
        audio_dir=cfg["audio_dir"],
        transcripts_dir=cfg["transcripts_dir"],
        manifest_path=cfg["manifest_path"],
    )
