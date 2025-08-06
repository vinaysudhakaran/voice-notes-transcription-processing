#!/usr/bin/env python3
"""
glossary_classifier.py

Extract unique mis-recognized terms (ref and optionally hyp) from the misrepresentation errors JSONL,
normalize them, classify each term into agricultural categories via OpenAI GPT-4,
and output the resulting glossary in CSV and JSONL formats with fault tolerance,
retry logic (via tenacity), and failure logging.

Configure DEFAULT_CONFIG below, then run:
    python glossary_classifier.py

Outputs:
- results/glossary_terms.csv : term,category
- results/glossary_terms.jsonl : line-delimited JSON {"term":..,"category":..}
"""
import os
import re
import json
import unicodedata
import time
import openai
import pandas as pd
from openai import OpenAI
from pathlib import Path
from typing import List, Set, Dict, Any, Optional
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed, RetryCallState

from utils.logger import logger


# map language names to the Unicode range for that script
SCRIPT_RANGES: Dict[str, str] = {
    "hindi": "\u0900-\u097f",  # Devanagari
    "bengali": "\u0980-\u09ff",  # Bengali
    "gujarati": "\u0a80-\u0aff",  # Gujarati
    "tamil": "\u0b80-\u0bff",  # Tamil
    "telugu": "\u0c00-\u0c7f",  # Telugu
    "kannada": "\u0c80-\u0cff",  # Kannada
    "malayalam": "\u0d00-\u0d7f",  # Malayalam
    "oriya": "\u0b00-\u0b7f",  # Oriya
    "punjabi": "\u0a00-\u0a7f",  # Gurmukhi
    # add more here…
}

# -----------------------------------------------------------------------------
# === DEFAULT CONFIG ===
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Input
    "misrep_terms_jsonl": "results/misrep_terms.jsonl",
    "include_hyp": False,  # whether to include hyp_word in term list
    # Normalization: "unicode" or "indicnlp"
    "normalization": "indicnlp",
    # Classification language
    "language": "Hindi",
    # Classification
    "openai_model": "o4-mini",
    "batch_api_mode": True,  # False → Chat API mode; True → Batch API mode
    "batch_chunk_size": 5000,  # chunk size for both batch API and Chat API paths
    # Retry logic
    "max_attempts": 3,
    # Fault tolerance
    "processed_terms_file": "results/classification_processed_terms.txt",
    "interim_glossary_jsonl": "results/glossary_terms_partial.jsonl",
    # Output
    "output_dir": "results",
    "glossary_csv": "results/glossary_terms.csv",
    "glossary_jsonl": "results/glossary_terms.jsonl",
}

# Optional Indic normalization (requires indic-nlp-library)
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

    _indic_normalizer = IndicNormalizerFactory().get_normalizer("hi")
except ImportError:
    _indic_normalizer = None


# -----------------------------------------------------------------------------
# Helper: normalize text
# -----------------------------------------------------------------------------
def normalize(term: str) -> str:
    mode = DEFAULT_CONFIG["normalization"]
    if mode == "indicnlp" and _indic_normalizer:
        return _indic_normalizer.normalize(term.strip())
    return unicodedata.normalize("NFC", term.strip())


def strip_non_script_chars(text: str, language: str) -> str:
    """
    Remove any leading or trailing characters that are not in the Unicode block
    for the given language.  If the language isn’t known, just strip whitespace.
    """
    lang_key = language.lower()
    char_range = SCRIPT_RANGES.get(lang_key)
    if not char_range:
        # unknown language: fall back to trimming whitespace only
        return text.strip()

    # build a regex that strips anything not in that block off both ends
    pattern = rf"^[^{char_range}]+|[^{char_range}]+$"
    return re.sub(pattern, "", text)


# -----------------------------------------------------------------------------
# Tenacity callback to log before each retry
# -----------------------------------------------------------------------------
def _log_before_retry(retry_state: RetryCallState) -> None:
    outcome = retry_state.outcome
    exc: Optional[BaseException] = None
    if outcome is not None:
        exc = outcome.exception()

    fn: Optional[Any] = getattr(retry_state, "fn", None)
    if fn is not None and hasattr(fn, "__name__"):
        fn_name = fn.__name__
    else:
        fn_name = repr(fn)

    logger.error(
        f"[{fn_name}] attempt {retry_state.attempt_number} "
        f"failed with: {exc!r}; retrying..."
    )


# -----------------------------------------------------------------------------
# Load unique terms from misrep_terms.jsonl
# -----------------------------------------------------------------------------
def extract_terms(records: List[Dict[str, Any]], include_hyp: bool) -> List[str]:
    """
    Given a list of misrepresentation records, pull out all ref_word (and hyp_word if requested),
    normalize and de-duplicate, then return sorted list of terms.
    """
    terms: Set[str] = set()
    for obj in records:
        ref = obj.get("ref_word", "").strip()
        if ref:
            terms.add(ref)
        if include_hyp:
            hyp = obj.get("hyp_word", "").strip()
            if hyp:
                terms.add(hyp)

    # Normalize & dedupe
    normalized = {normalize(t) for t in terms}

    # Strip any non‐script chars (punctuation, digits, etc.) from ends
    cleaned = {
        strip_non_script_chars(t, DEFAULT_CONFIG["language"]) for t in normalized
    }

    # drop any empties that might result from stripping
    final = [t for t in cleaned if t]

    # Sort
    sorted_terms = sorted(final)

    # # Get the full records for the top 'x' terms
    # x = 100
    # top_x = sorted_terms[:x]
    # out_path = Path(DEFAULT_CONFIG["output_dir"]) / f"top{x}_term_records.jsonl"
    # out_path.parent.mkdir(parents=True, exist_ok=True)

    # logger.info(f"Writing full records for top {x} terms to {out_path}")
    # with out_path.open("w", encoding="utf-8") as outf:
    #     for term in top_x:
    #         # find the first record where ref_word or hyp_word matches this term
    #         for obj in records:
    #             # normalize the record's words to compare
    #             ref_norm = normalize(obj.get("ref_word", "").strip())
    #             hyp_norm = (
    #                 normalize(obj.get("hyp_word", "").strip()) if include_hyp else None
    #             )
    #             if ref_norm == term or hyp_norm == term:
    #                 outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
    #                 break
    # import sys
    # sys.exit(0)

    # Return sorted list
    return sorted_terms


# -----------------------------------------------------------------------------
# Classification: via OpenAI Batch API in chunked mode
# -----------------------------------------------------------------------------
@retry(
    reraise=True,
    stop=stop_after_attempt(DEFAULT_CONFIG["max_attempts"]),
    wait=wait_fixed(1),
    before_sleep=_log_before_retry,
)
def classify_via_batch_api(terms: List[str]) -> List[Dict[str, str]]:
    """
    Classify terms using OpenAI Batch API.
    Retries the entire batch upload + polling on any Exception.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    client = OpenAI()
    language = DEFAULT_CONFIG["language"]
    system_prompt = (
        f"Carefully classify each {language} agricultural **term** into one of the categories: "
        "Crop, Variety, Agriculture unit, Pest/Disease, Symptom, Soil Nutrient/Fertiliser, Numeral, Generic.\n"
        f"If the input has no {language} letters, choose 'Not Applicable'.\n"
        "**Do NOT** wrap your response in markdown fences (```); send **raw JSON** only.\n"
        "Respond with a **single JSON array** in this form:\n"
        '[{"term": ..., "category": ...}, ...]\n'
        "**No** extra text."
    )

    # Build tasks
    tasks = []
    for idx, term in enumerate(terms):
        tasks.append(
            {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": DEFAULT_CONFIG["openai_model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": term},
                    ],
                },
            }
        )

    # Write & upload batch file
    batch_dir = Path("batch_jobs")
    batch_dir.mkdir(parents=True, exist_ok=True)
    chunk_id = int(time.time() * 1000)
    batch_file_path = batch_dir / f"batch_terms_{chunk_id}.jsonl"
    logger.debug(
        f"[Batch API] Writing batch of {len(tasks)} tasks to {batch_file_path}"
    )
    with batch_file_path.open("w", encoding="utf-8") as bf:
        for task in tasks:
            bf.write(json.dumps(task, ensure_ascii=False) + "\n")

    logger.debug("[Batch API] Uploading batch file to OpenAI")
    with batch_file_path.open("rb") as bf:
        batch_file = client.files.create(file=bf, purpose="batch")

    logger.info(f"[Batch API] Created file={batch_file.id}, launching batch job")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    logger.info(f"[Batch API] Started batch job id={batch_job.id}")

    # Poll until done
    while True:
        batch_job = client.batches.retrieve(batch_job.id)
        status = batch_job.status
        logger.debug(f"[Batch API] batch_job {batch_job.id} status: {status}")
        if status == "completed":
            break
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"Batch job {batch_job.id} ended with status {status}")
        time.sleep(5)

    logger.info(f"[Batch API] Job {batch_job.id} completed; fetching results")
    output_id = batch_job.output_file_id
    if not output_id:
        raise RuntimeError(f"Batch job {batch_job.id} has no output_file_id")

    raw_bytes = client.files.content(output_id).content
    result_text = raw_bytes.decode("utf-8")

    lines = result_text.splitlines()
    results: List[Dict[str, str]] = []

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Parse the wrapper record
        try:
            wrapper = json.loads(line)
        except json.JSONDecodeError as e:
            logger.error(f"[Batch API] Skipping malformed wrapper line {idx}: {e}")
            continue

        # Drill down to the model's content
        content = (
            wrapper.get("response", {})
            .get("body", {})
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        raw = content.strip()
        if not raw:
            logger.error(f"[Batch API] Empty content for wrapper line {idx}; skipping")
            continue

        # Remove markdown fences and trailing commas
        sublines = [ln for ln in raw.splitlines() if not ln.strip().startswith("```")]
        clean = "\n".join(ln.rstrip().rstrip(",") for ln in sublines).strip()

        # Parse either as array or single object
        parsed = []
        if clean.startswith("["):
            try:
                parsed = json.loads(clean)
                if not isinstance(parsed, list):
                    logger.error(
                        f"[Batch API] Expected list but got {type(parsed)}; skipping"
                    )
                    parsed = []
            except json.JSONDecodeError as e:
                logger.error(
                    f"[Batch API] JSON array parse error on wrapper {idx}: {e}"
                )
                parsed = []
        else:
            try:
                parsed = [json.loads(clean)]
            except json.JSONDecodeError as e:
                logger.error(
                    f"[Batch API] JSON object parse error on wrapper {idx}: {e}"
                )
                parsed = []

        # Validate each parsed object
        for obj in parsed:
            if not isinstance(obj, dict):
                logger.error(
                    f"[Batch API] Parsed non‐dict at wrapper {idx}: {obj!r}; skipping"
                )
                continue
            term = obj.get("term")
            cat = obj.get("category")
            if term is None or cat is None:
                logger.error(
                    f"[Batch API] Missing term/category in wrapper {idx} → {obj!r}; skipping"
                )
                continue
            results.append({"term": term, "category": cat})

    if not results:
        raise RuntimeError(
            f"[Batch API] No valid JSON parsed from batch job {batch_job.id}"
        )

    logger.info(
        f"[Batch API] Parsed {len(results)} results from batch job {batch_job.id}"
    )
    return results


# -----------------------------------------------------------------------------
# Classification: via OpenAI Chat API in chunked mode
# -----------------------------------------------------------------------------
@retry(
    reraise=True,
    stop=stop_after_attempt(DEFAULT_CONFIG["max_attempts"]),
    wait=wait_fixed(1),
    before_sleep=_log_before_retry,
)
def classify_via_chat_api(terms: List[str]) -> List[Dict[str, str]]:
    """
    Classify terms in-code by calling the OpenAI Chat endpoint on a single batch.
    Retries up to DEFAULT_CONFIG["max_attempts"] times on any Exception.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    client = OpenAI()
    language = DEFAULT_CONFIG["language"]
    system_prompt = (
        f"Carefully classify each {language} agricultural **term** into one of the categories: "
        "Crop, Variety, Agriculture unit, Pest/Disease, Symptom, Soil Nutrient/Fertiliser, Numeral, Generic.\n"
        f"If the input has no {language} letters, choose 'Not Applicable'."
        "**Do NOT** wrap your response in markdown fences (```); send **raw JSON** only.\n"
        "Respond with a **single JSON array** in this form:\n"
        '[{"term": ..., "category": ...}, ...]\n'
        "**No** extra text."
    )

    user_prompt = json.dumps(terms, ensure_ascii=False)
    logger.info(f"[Chat API] Sending {len(terms)} terms for classification")
    resp = client.chat.completions.create(
        model=DEFAULT_CONFIG["openai_model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # temperature=0.0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    if not raw:
        logger.error("[Chat API] Empty response received; retrying")
        raise RuntimeError("Empty response from Chat API")

    # Remove any markdown fences entirely
    lines = raw.splitlines()
    clean_lines = [ln for ln in lines if not ln.strip().startswith("```")]
    # Strip trailing commas from each line
    clean = "\n".join(ln.rstrip().rstrip(",") for ln in clean_lines).strip()

    results: List[Dict[str, str]] = []

    # Try parsing as a single JSON array
    if clean.startswith("["):
        try:
            results = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(f"[Chat API] JSON array parse error: {e}")

    # Fallback: one JSON object per line
    if not results:
        for line in clean.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                results.append(obj)
            except json.JSONDecodeError as e:
                logger.error(f"[Chat API] Skipping malformed line: {line!r} ({e})")

    if not results:
        logger.error("[Chat API] No valid JSON parsed; retrying")
        raise RuntimeError("No classifications parsed from Chat API")

    logger.info(f"[Chat API] Received {len(results)} classified terms\n")
    return results


def classify_terms_retry(terms: List[str]) -> List[Dict[str, str]]:
    if DEFAULT_CONFIG["batch_api_mode"]:
        return classify_via_batch_api(terms)
    else:
        return classify_via_chat_api(terms)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    # Explicitly fsync-capable
    def safe_write(fh, data: str):
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())

    load_dotenv()
    cfg = DEFAULT_CONFIG

    # Prepare directories
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Read all mis-recognized entries
    all_recs = []
    with open(cfg["misrep_terms_jsonl"], "r", encoding="utf-8") as fin:
        for line in fin:
            all_recs.append(json.loads(line))

    # Extract unique terms
    terms = extract_terms(all_recs, cfg["include_hyp"])
    if not terms:
        logger.info("No terms to classify; exiting.")
        return

    # Load processed checkpoint
    processed_file = Path(cfg["processed_terms_file"])
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    processed_terms = set()
    if processed_file.exists():
        processed_terms = {
            ln.strip() for ln in processed_file.read_text().splitlines() if ln.strip()
        }
    else:
        processed_file.touch()

    # Prepare interim JSONL
    interim_file = Path(cfg["interim_glossary_jsonl"])
    interim_file.parent.mkdir(parents=True, exist_ok=True)
    interim_file.touch(exist_ok=True)

    # Determine work left and chunk size
    to_do = [t for t in terms if t not in processed_terms]
    if not to_do:
        logger.info("All terms already classified; exiting.")
        return

    chunk_size = cfg["batch_chunk_size"]

    # Process in chunks, but flush/fsync after every write
    with interim_file.open("a", encoding="utf-8") as intf, processed_file.open(
        "a", encoding="utf-8"
    ) as pf:

        for i in range(0, len(to_do), chunk_size):
            batch = to_do[i : i + chunk_size]
            try:
                # call either Batch or Chat API on this sub‐batch
                results_chunk = classify_terms_retry(batch)
            except Exception as e:
                logger.error(f"Classification failed on batch {i//chunk_size}: {e!r}")
                continue

            # write each classification result immediately
            for entry in results_chunk:
                safe_write(intf, json.dumps(entry, ensure_ascii=False) + "\n")

            # **only** mark the successfully parsed terms
            successful_terms = {entry["term"] for entry in results_chunk}
            for term in successful_terms:
                safe_write(pf, term + "\n")

    # Collate and write final outputs
    final_results = []
    with interim_file.open("r", encoding="utf-8") as inf:
        for line in inf:
            final_results.append(json.loads(line))

    # CSV
    df = pd.DataFrame(final_results)
    df.to_csv(cfg["glossary_csv"], index=False, encoding="utf-8")

    # JSONL
    with open(cfg["glossary_jsonl"], "w", encoding="utf-8") as fout:
        for entry in final_results:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        f"Wrote {len(final_results)} terms → "
        f"{cfg['glossary_csv']} and {cfg['glossary_jsonl']}"
    )


if __name__ == "__main__":
    main()
