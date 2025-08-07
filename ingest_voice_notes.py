#!/usr/bin/env python3
"""
ingest_voice_notes.py

Downloads audio files and saves corresponding transcripts,
building a manifest with basic audio metadata.

Modes of invocation (single-file):

1) Direct-config mode (default, or with --direct):
   Edit the DIRECT_CONFIG dict below, then run without args:
       python ingest_voice_notes.py

2) CLI mode (any other flags present, or omit --direct):
       python ingest_voice_notes.py <input> \
           --audio-dir <dir> \
           --transcripts-dir <dir> \
           --manifest <file> \
           [--skip-metadata]

On each run, the script:
- Maintains an append-only JSONL manifest (`manifest.jsonl`) for resumable processing
- Logs any per-record errors to `errors.jsonl`
- Optionally captures audio metadata via ffprobe (toggle with --skip-metadata)
- At the end, compiles the JSONL manifest into a single JSON array if desired
"""

import sys
import argparse
import logging
import hashlib
import json
import re
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
import requests
import subprocess
from requests.adapters import HTTPAdapter, Retry

# -----------------------------------------------------------------------------
# === DIRECT-CONFIG MODE ===
# Fill in these values, then run `python ingest_voice_notes.py`
# -----------------------------------------------------------------------------
LANGUAGE = "hindi"
SUFFIX = "2"
DIRECT_CONFIG = {
    "input_path": f"dataset/metadata/{LANGUAGE}/hindi_transcription_audio_{SUFFIX}.xlsx",
    "audio_dir": f"dataset/audio/{LANGUAGE}/{SUFFIX}",
    "transcripts_dir": f"dataset/transcripts/{LANGUAGE}/{SUFFIX}",
    "manifest_path": f"dataset/manifest/{LANGUAGE}/{SUFFIX}/manifest.json",
    "capture_metadata": False,  # skip ffprobe by default for direct-config
}

# -----------------------------------------------------------------------------
# Logging setup
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
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {ext}")


def ensure_ffprobe():
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        logger.error("ffprobe not found. Install FFmpeg (`ffprobe`).")
        sys.exit(1)


def get_audio_metadata(path: Path) -> Dict[str, Union[int, float]]:
    """Extract duration, size, and bit_rate via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate",
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error on {path}: {result.stderr}")
    meta: Dict[str, Union[int, float]] = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k] = float(v) if k == "duration" else int(v)
    return meta


def download_with_retry(url: str, dest: Path, max_retries: int = 3):
    """Download a file with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=max_retries, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))

    logger.info("Downloading %s → %s", url, dest)
    with session.get(url, stream=True, timeout=10) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)


def compute_md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


# Helpers for Google Drive links


def extract_drive_file_id(url: str) -> str | None:
    """
    Extracts the file-id from common Drive share URLs, e.g.
      https://drive.google.com/file/d/<ID>/view
      https://drive.google.com/open?id=<ID>
    """
    # /d/<ID>/
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    # id=<ID>
    qs = urlparse(url).query
    params = parse_qs(qs)
    if "id" in params:
        return params["id"][0]
    return None


def normalize_drive_url(url: str) -> str:
    """
    Given any URL, if it's a Drive link, convert it to a direct-download URL:
      https://drive.google.com/uc?export=download&id=<ID>
    Otherwise return the URL unchanged.
    """
    file_id = extract_drive_file_id(url)
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url


# -----------------------------------------------------------------------------
# Core ingestion function
# -----------------------------------------------------------------------------
def ingest_voice_notes(
    input_path: Union[str, Path],
    audio_dir: Union[str, Path] = "audio",
    transcripts_dir: Union[str, Path] = "transcripts",
    manifest_path: Union[str, Path] = "manifest.json",
    capture_metadata: bool = True,
) -> List[Dict]:
    """
    Download audio files, save transcripts, and build a resume-capable manifest.

    Args:
        capture_metadata: if False, skip ffprobe and only record file size.
    """

    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    transcripts_dir = Path(transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_jsonl = manifest_path.with_suffix(manifest_path.suffix + "l")
    errors_jsonl = manifest_path.parent / "errors.jsonl"

    # Pre-check ffprobe only if metadata is desired
    if capture_metadata:
        ensure_ffprobe()

    # Load processed UIDs for resume
    processed_uids = set()
    if manifest_jsonl.exists():
        with open(manifest_jsonl, "r", encoding="utf-8") as mf:
            for line in mf:
                try:
                    rec = json.loads(line)
                    processed_uids.add(rec.get("uid"))
                except json.JSONDecodeError:
                    continue

    # Open JSONL files for append
    manifest_fp = open(manifest_jsonl, "a", encoding="utf-8")
    errors_fp = open(errors_jsonl, "a", encoding="utf-8")

    df = load_table(input_path)
    required = {"Filename", "Language", "Annotated Transcriptions"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing cols: {required - set(df.columns)}")

    for _, row in df.iterrows():
        raw_url = str(row["Filename"])

        # normalize Drive links, else pass through
        download_url = normalize_drive_url(raw_url)

        # choose UID: Drive ID if present, else your old stem logic
        drive_id = extract_drive_file_id(raw_url)
        if drive_id:
            uid = drive_id
        else:
            uid = Path(raw_url).stem.split("_")[0]

        lang = str(row["Language"])
        transcript = str(row["Annotated Transcriptions"])

        if uid in processed_uids:
            logger.debug("Skipping already processed %s", uid)
            continue

        # choose audio file extension type:
        # 1) if the normalized URL’s path has a suffix (e.g. .ogg), use it;
        # 2) else if this is a Drive link, default to .mp3;
        # 3) otherwise fall back to .ogg
        parsed = urlparse(download_url)
        suffix = Path(parsed.path).suffix
        if suffix:
            ext = suffix
        elif drive_id:
            ext = ".mp3"
        else:
            ext = ".ogg"

        audio_path = audio_dir / f"{uid}{ext}"
        txt_path = transcripts_dir / f"{uid}.txt"

        try:
            if not audio_path.exists():
                download_with_retry(download_url, audio_path)
            txt_path.write_text(transcript, encoding="utf-8")

            # Conditional metadata capture
            if capture_metadata:
                try:
                    meta = get_audio_metadata(audio_path)
                except Exception as e:
                    logger.warning("ffprobe failed for %s: %s", audio_path, e)
                    meta = {
                        "duration": None,
                        "size": audio_path.stat().st_size,
                        "bit_rate": None,
                    }
            else:
                meta = {
                    "duration": None,
                    "size": audio_path.stat().st_size,
                    "bit_rate": None,
                }

            entry = {
                "uid": uid,
                "url": raw_url,
                "download_url": download_url,
                "language": lang,
                "audio_path": str(audio_path),
                "transcript_path": str(txt_path),
                "duration_sec": meta.get("duration"),
                "file_size_bytes": meta.get("size"),
                "bit_rate": meta.get("bit_rate"),
                "md5": compute_md5(audio_path),
                "transcript_word_count": len(transcript.split()),
            }
            # Append success record
            manifest_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            manifest_fp.flush()
            processed_uids.add(uid)

        except Exception as e:
            err = {"uid": uid, "url": download_url, "error": str(e)}
            errors_fp.write(json.dumps(err, ensure_ascii=False) + "\n")
            errors_fp.flush()
            logger.error("Error processing %s: %s", uid, e)
            continue

    # Close file pointers
    manifest_fp.close()
    errors_fp.close()

    # Compile JSONL → JSON array for final manifest
    try:
        records = []
        with open(manifest_jsonl, "r", encoding="utf-8") as mf:
            for line in mf:
                records.append(json.loads(line))
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info("Compiled %d records → %s", len(records), manifest_path)
    except Exception as e:
        logger.warning("Could not compile JSONL to JSON: %s", e)

    return list(processed_uids)


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
def _cli_mode():
    p = argparse.ArgumentParser(
        description="Ingest voice notes: download audio & save transcripts."
    )
    p.add_argument("input", nargs="?", help="CSV/XLSX file")
    p.add_argument("--audio-dir", default="audio")
    p.add_argument("--transcripts-dir", default="transcripts")
    p.add_argument("--manifest", default="manifest.json")
    p.add_argument(
        "--direct",
        action="store_true",
        help="Force direct-config mode (ignores other flags)",
    )
    p.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip ffprobe metadata capture for speed",
    )
    args = p.parse_args()

    # decide mode
    if args.direct or not args.input:
        cfg = DIRECT_CONFIG.copy()
    else:
        cfg = {
            "input_path": args.input,
            "audio_dir": args.audio_dir,
            "transcripts_dir": args.transcripts_dir,
            "manifest_path": args.manifest,
        }
    # override metadata flag if present
    cfg["capture_metadata"] = not args.skip_metadata

    ingest_voice_notes(**cfg)


if __name__ == "__main__":
    _cli_mode()
