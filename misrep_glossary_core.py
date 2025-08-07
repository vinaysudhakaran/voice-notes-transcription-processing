#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# WARNING: Do NOT run this file directly!
# If you do, you may hit fatal gRPC C-core errors such as:
#   “sync.cc:131 Check failed: err == 0 || err == ETIMEDOUT || err == EAGAIN”
#   or
#   “timer_manager.cc:69 Check failed:
#   check_result.has_value() ERROR: More than one MainLoop is running.”
#
# Instead, always invoke **misrep_glossary_extractor.py**, which:
#  1. Sets `GRPC_EXPERIMENTAL_EVENT_ENGINE=false` before any gRPC imports
#  2. Uses a small bootstrap wrapper to prevent gRPC’s experimental event
#     engine from registering at-fork handlers
# -----------------------------------------------------------------------------

"""
Configure DEFAULT_CONFIG below, then run:
    python misrep_glossary_extractor.py

Outputs:
- results/misrep_terms.csv    : CSV of uid,model,error,ref_word,hyp_word,start,end,clip
- results/misrep_terms.jsonl  : same data as line‑delimited JSON
- results/extraction_processed_uids.txt : UIDs successfully processed
- results/extraction_errored_uids.txt   : UIDs that failed during processing

"""
import os
import json
import wave
import random
import subprocess
import openai
import torch
import hashlib
import csv

from pathlib import Path
from difflib import SequenceMatcher
from typing import cast, List, Dict, Any
from pprint import pformat
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk
from transformers.pipelines import pipeline
from transformers import Wav2Vec2ForCTC, AutoProcessor

from utils.logger import logger


# -----------------------------------------------------------------------------
# Determine device once
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    PIPELINE_DEVICE = 0
    DEVICE_TYPE = "cuda"
    USE_HALF = True
else:
    PIPELINE_DEVICE = -1
    DEVICE_TYPE = "cpu"
    USE_HALF = False


# -----------------------------------------------------------------------------
# === DEFAULT CONFIG ===
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Engines to run: subset of ["whisper", "azure", "google", "mms"]
    "engines": ["google"],
    # Clipping
    "clipping_mode": "lazy",  # "lazy" or "eager"
    "clips_dir": "clips",
    # Sampling & runs
    "sample_size": 0,  # <=0 or run_all=True → all records
    "run_all": True,
    # Audio preprocessing
    "target_sample_rate": 16000,  # Hz
    # Language settings
    "language_code": "hi-IN",  # for Azure & Google
    "mms_target_lang": "hin",  # for Meta MMS
    # Model settings
    "openai_model": "whisper-1",
    "mms_model": "facebook/mms-1b-all",
    # Inputs
    "manifest_path": "dataset/manifest/hindi/2/manifest.json",
    "processed_uids_file": "results/extraction_processed_uids.txt",
    # Outputs
    "output_dir": "results",
    "wav_dir": "wav_files",
    "misrep_terms_csv": "results/misrep_terms.csv",
    "misrep_terms_jsonl": "results/misrep_terms.jsonl",
    "errors_log": "results/extraction_errored_uids.txt",
}


# -----------------------------------------------------------------------------
# Global for HuggingFace MMS pipeline
# -----------------------------------------------------------------------------
_mms_pipeline = None


# -----------------------------------------------------------------------------
# Helper: prepare uniform WAVs
# -----------------------------------------------------------------------------
def prepare_audio(inp: Path, outdir: Path, sr: int) -> Path:
    """
    Convert `inp` → mono WAV @ `sr` Hz, caching under `outdir`.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    wav = outdir / f"{inp.stem}_{sr}.wav"
    if not wav.exists():
        cmd = ["ffmpeg", "-y", "-i", str(inp), "-ar", str(sr), "-ac", "1", str(wav)]
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.debug(f"\nConverted {inp} → {wav} @{sr}Hz")
    return wav


def audio_duration(wav: Path) -> float:
    """Return duration (s) of a WAV file."""
    if not wav.exists():
        logger.error(f"Audio file not found: {wav}")
        raise FileNotFoundError(f"Audio file not found: {wav}")
    try:
        with wave.open(str(wav), "rb") as w:
            return w.getnframes() / w.getframerate()
    except wave.Error as e:
        logger.error(f"Failed to read WAV {wav}: {e}")
        raise RuntimeError(f"Failed to read WAV {wav}: {e}")


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load JSON array manifest."""
    if not path.exists():
        logger.error(f"Manifest file not found: {path}")
        raise FileNotFoundError(f"Manifest file not found: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Unable to read manifest {path}: {e}")
        raise RuntimeError(f"Unable to read manifest {path}: {e}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in manifest {path}: {e}")
        raise ValueError(f"Invalid JSON in manifest {path}: {e}")
    if not isinstance(data, list):
        logger.error(f"Manifest {path} does not contain a JSON array")
        raise ValueError(f"Manifest {path} does not contain a JSON array")
    if len(data) == 0:
        logger.warning(f"Manifest {path} is empty")
    return data


def select_records(
    recs: List[Dict[str, Any]], n: int, run_all: bool
) -> List[Dict[str, Any]]:
    """
    Sample or return all records.
    """
    if not recs:
        logger.error("No records available to select from.")
        return []
    total = len(recs)
    if run_all or n <= 0 or n >= total:
        logger.info(f"Selecting all {total} records")
        return recs
    if n > total:
        logger.warning(
            f"Requested sample_size={n} > total_records={total}, returning all"
        )
        return recs
    sampled = random.sample(recs, n)
    logger.info(f"Selected {len(sampled)} random records out of {total}")
    return sampled


def select_records_by_uids(
    recs: List[Dict[str, Any]], uids: List[str]
) -> List[Dict[str, Any]]:
    """
    Return only those records whose 'uid' is in the provided list of UIDs.
    """
    if not recs:
        logger.error("No records available to select from.")
        return []
    if not uids:
        logger.warning(
            "No UIDs provided to select_records_by_uids; returning empty list"
        )
        return []
    uid_set = set(uids)
    filtered = [rec for rec in recs if rec.get("uid") in uid_set]
    found_uids = {rec.get("uid") for rec in filtered}
    missing = uid_set - found_uids
    if missing:
        logger.warning(f"UIDs not found in manifest: {sorted(missing)}")
    logger.info(f"Selected {len(filtered)} records by UID")
    return filtered


# -----------------------------------------------------------------------------
# Align words to detect sub/del/ins
# -----------------------------------------------------------------------------
def align_words_sm(ref: List[str], hyp: List[str]) -> List[Dict[str, Any]]:
    """
    Returns list of dicts:
      {error:'sub'|'del'|'ins', ref:r, hyp:h, ref_idx, hyp_idx}
    """
    sm = SequenceMatcher(None, ref, hyp)
    errors = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            for k, (r, h) in enumerate(zip(ref[i1:i2], hyp[j1:j2])):
                errors.append(
                    dict(error="sub", ref=r, hyp=h, ref_idx=i1 + k, hyp_idx=j1 + k)
                )
        elif tag == "delete":
            for k, r in enumerate(ref[i1:i2]):
                errors.append(
                    dict(error="del", ref=r, hyp="", ref_idx=i1 + k, hyp_idx=None)
                )
        elif tag == "insert":
            for k, h in enumerate(hyp[j1:j2]):
                errors.append(
                    dict(error="ins", ref="", hyp=h, ref_idx=None, hyp_idx=j1 + k)
                )
    return errors


def align_words_levenshtein(ref: List[str], hyp: List[str]) -> List[Dict[str, Any]]:
    """
    Compute word‑level edit ops between `ref` and `hyp` via true
    minimum‑edit‑distance (Levenshtein) DP, and return a list of dicts:
      { error:  'sub' | 'del' | 'ins',
        ref:    reference word (or "" for ins),
        hyp:    hypothesis word (or "" for del),
        ref_idx: index in ref (None for ins),
        hyp_idx: index in hyp (None for del) }
    """
    n, m = len(ref), len(hyp)
    # Build DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # substitution
                    dp[i - 1][j],  # deletion
                    dp[i][j - 1],  # insertion
                )

    # Backtrack from dp[n][m]
    i, j = n, m
    ops: List[Dict[str, Any]] = []
    while i > 0 or j > 0:
        # match (no error)
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
            continue

        # substitution
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(
                {
                    "error": "sub",
                    "ref": ref[i - 1],
                    "hyp": hyp[j - 1],
                    "ref_idx": i - 1,
                    "hyp_idx": j - 1,
                }
            )
            i -= 1
            j -= 1

        # deletion
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(
                {
                    "error": "del",
                    "ref": ref[i - 1],
                    "hyp": "",
                    "ref_idx": i - 1,
                    "hyp_idx": None,
                }
            )
            i -= 1

        # insertion
        else:
            ops.append(
                {
                    "error": "ins",
                    "ref": "",
                    "hyp": hyp[j - 1],
                    "ref_idx": None,
                    "hyp_idx": j - 1,
                }
            )
            j -= 1

    ops.reverse()
    return ops


# -----------------------------------------------------------------------------
# Transcription + timestamps: Whisper (segment-level)
# Note: Whisper is a display-only model, so the lexical field isn't populated
# in the transcription.
# -----------------------------------------------------------------------------
def transcribe_whisper(wav: Path) -> Dict[str, Any]:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    with wav.open("rb") as f:
        resp = openai.audio.transcriptions.create(
            file=f,
            model=DEFAULT_CONFIG["openai_model"],
            response_format="verbose_json",
        )
    logger.debug(f"Whisper raw response:\n{pformat(resp.__dict__)}")
    txt = resp.text.strip()
    chunks = []
    for seg in getattr(resp, "segments", []):
        chunks.append({"text": seg.text.strip(), "start": seg.start, "end": seg.end})
    if not chunks:
        duration = audio_duration(wav)
        chunks = [{"text": txt, "start": 0.0, "end": duration}]
    return {"text": txt, "chunks": chunks}


# -----------------------------------------------------------------------------
# Transcription + timestamps: Azure (word-level)
# -----------------------------------------------------------------------------
def transcribe_azure(wav: Path) -> Dict[str, Any]:
    """
    Streaming Azure STT
    """
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY/REGION not set")

    cfg = speechsdk.SpeechConfig(subscription=key, region=region)
    cfg.speech_recognition_language = DEFAULT_CONFIG["language_code"]
    cfg.request_word_level_timestamps()
    cfg.output_format = speechsdk.OutputFormat.Detailed

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=cfg, audio_config=speechsdk.audio.AudioConfig(filename=str(wav))
    )

    seen_offsets = set()
    words = []
    text_order = []

    def on_recognized(evt):
        detail = json.loads(evt.result.json)
        for hyp in detail.get("NBest", []):
            for w in hyp.get("Words", []):
                offset = w["Offset"]
                if offset in seen_offsets:
                    continue
                seen_offsets.add(offset)
                start = offset / 1e7
                end = (offset + w["Duration"]) / 1e7
                words.append({"text": w["Word"], "start": start, "end": end})
                text_order.append(w["Word"])

    done = False

    def stop_cb(evt):
        nonlocal done
        done = True

    recognizer.recognized.connect(on_recognized)
    recognizer.session_stopped.connect(stop_cb)
    recognizer.canceled.connect(stop_cb)

    recognizer.start_continuous_recognition()
    while not done:
        pass
    recognizer.stop_continuous_recognition()

    transcript = " ".join(text_order).strip()
    if not words:
        # fallback if nothing recognized
        duration = audio_duration(wav)
        words = [{"text": "", "start": 0.0, "end": duration}]
    return {"text": transcript, "chunks": words}


# -----------------------------------------------------------------------------
# Transcription + timestamps: Google (word-level)
# -----------------------------------------------------------------------------
def transcribe_google(wav: Path) -> Dict[str, Any]:
    from google.cloud import speech

    cred = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred or not Path(cred).exists():
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set or file not found")

    content = wav.read_bytes()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=DEFAULT_CONFIG["target_sample_rate"],
        language_code=DEFAULT_CONFIG["language_code"],
        enable_word_time_offsets=True,
    )
    with speech.SpeechClient() as client:
        response = client.recognize(config=config, audio=audio)
    # logger.debug(f"Google raw response:\n{pformat(response)}")

    words = []
    text_pieces = []
    for result in response.results:
        alt = result.alternatives[0]
        text_pieces.append(alt.transcript)
        for w in alt.words:
            # w.start_time and w.end_time are datetime.timedelta
            start = w.start_time.total_seconds()
            end = w.end_time.total_seconds()
            words.append(
                {
                    "text": w.word,
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )

    if not words:
        duration = audio_duration(wav)
        words = [{"text": "", "start": 0.0, "end": duration}]

    return {
        "text": " ".join(text_pieces).strip(),
        "chunks": words,
    }


# -----------------------------------------------------------------------------
# Transcription + timestamps: Meta MMS (word-level)
# -----------------------------------------------------------------------------
def transcribe_mms(wav: Path) -> Dict[str, Any]:
    global _mms_pipeline

    if _mms_pipeline is None:
        processor = AutoProcessor.from_pretrained(
            DEFAULT_CONFIG["mms_model"], target_lang=DEFAULT_CONFIG["mms_target_lang"]
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            DEFAULT_CONFIG["mms_model"],
            target_lang=DEFAULT_CONFIG["mms_target_lang"],
            ignore_mismatched_sizes=True,
        )
        model.to(DEVICE_TYPE)  # type: ignore
        if USE_HALF:
            model.half()

        _mms_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps="word",
            chunk_length_s=30,
            device=PIPELINE_DEVICE,
        )
        logger.info(f"Initialized MMS on {DEVICE_TYPE} (pipe_dev={PIPELINE_DEVICE})")

    raw = _mms_pipeline(str(wav))
    logger.debug(f"MMS raw output:\n{pformat(raw)}")

    out = cast(Dict[str, Any], raw)
    text = out["text"].strip()
    chunks = [
        {"text": c["text"], "start": c["timestamp"][0], "end": c["timestamp"][1]}
        for c in out["chunks"]
    ]
    if not chunks:
        duration = audio_duration(wav)
        chunks = [{"text": text, "start": 0.0, "end": duration}]
    return {"text": text, "chunks": chunks}


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main():
    # Load environment variables
    load_dotenv()
    cfg = DEFAULT_CONFIG

    # Ensure directories exist
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["wav_dir"]).mkdir(parents=True, exist_ok=True)
    if cfg["clipping_mode"] == "eager":
        Path(cfg["clips_dir"]).mkdir(parents=True, exist_ok=True)

    # Prepare incremental JSONL writer
    jsonl_path = Path(cfg["misrep_terms_jsonl"])
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_f = jsonl_path.open("a", encoding="utf-8")

    # Prepare incremental CSV writer
    csv_path = Path(cfg["misrep_terms_csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_exists = csv_path.exists()
    csv_f = csv_path.open("a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_f)
    if not csv_exists:
        csv_writer.writerow(
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
        )
        csv_f.flush()

    # Prepare checkpoint files
    completed_file = Path(cfg["processed_uids_file"])
    completed_file.parent.mkdir(parents=True, exist_ok=True)
    errored_file = Path(cfg["errors_log"])
    errored_file.parent.mkdir(parents=True, exist_ok=True)

    completed = (
        set(completed_file.read_text().splitlines())
        if completed_file.exists()
        else set()
    )
    errored = (
        set(errored_file.read_text().splitlines()) if errored_file.exists() else set()
    )

    # Load & select records
    manifest = load_manifest(Path(cfg["manifest_path"]))
    recs = select_records(manifest, cfg["sample_size"], cfg["run_all"])

    # -------------------------------------------
    # OPTIONALLY RUN PIPELINE ON SPECIFIC UUIDs
    # -------------------------------------------
    # uuids = [
    #     "344dcd50-6343-4a23-aa6d-a6a096f85140",
    #     "1f6cec77-4471-44ec-aec3-4884fd089749",
    # ]
    # recs = select_records_by_uids(manifest, uuids)

    recs_to_process = [r for r in recs if r["uid"] not in completed]

    if not recs_to_process:
        logger.info(f"No new records to process. Completed: {len(completed)}")
        jsonl_f.close()
        csv_f.close()
        return

    try:
        for rec in recs_to_process:
            uid = rec["uid"]
            try:
                # Prepare audio & ground truth
                orig = Path(rec["audio_path"])
                wav = prepare_audio(
                    orig, Path(cfg["wav_dir"]), cfg["target_sample_rate"]
                )
                gt_words = (
                    Path(rec["transcript_path"])
                    .read_text(encoding="utf-8")
                    .strip()
                    .split()
                )
                logger.debug(f"ground truth words: {gt_words}")

                for engine in cfg["engines"]:
                    fn = globals().get(f"transcribe_{engine}")
                    if not fn:
                        logger.error(f"Unknown engine specified: {engine}")
                        continue

                    try:
                        result = fn(wav)
                    except Exception as e:
                        logger.error(f"{engine} failed on {uid}: {e}")
                        continue

                    hyp_words = result["text"].split()
                    logger.debug(f"{engine} hypothesis words: {hyp_words}")

                    for idx, err in enumerate(
                        align_words_levenshtein(gt_words, hyp_words)
                    ):
                        # Determine timestamps
                        if err["hyp_idx"] is not None and err["hyp_idx"] < len(
                            result["chunks"]
                        ):
                            st = result["chunks"][err["hyp_idx"]]["start"]
                            ed = result["chunks"][err["hyp_idx"]]["end"]
                        else:
                            st, ed = 0.0, audio_duration(wav)

                        # Clip if eager
                        if cfg["clipping_mode"] == "eager":
                            clip_file = (
                                Path(cfg["clips_dir"]) / f"{uid}_{engine}_{idx}.wav"
                            )
                            cmd = [
                                "ffmpeg",
                                "-y",
                                "-i",
                                str(wav),
                                "-ss",
                                str(st),
                                "-to",
                                str(ed),
                                "-ar",
                                str(cfg["target_sample_rate"]),
                                "-ac",
                                "1",
                                str(clip_file),
                            ]
                            try:
                                subprocess.run(
                                    cmd,
                                    check=True,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL,
                                )
                                clip_path_out = str(clip_file)
                                logger.debug(f"Clipped segment to {clip_file}")
                            except Exception as e:
                                logger.error(
                                    f"Eager clipping failed for {uid}/{engine}/{idx}: {e}"
                                )
                                clip_path_out = str(wav)
                        else:
                            clip_path_out = str(wav)

                        # Build entry
                        raw = f"{uid}-{engine}-{idx}"
                        entry = {
                            "_id": hashlib.md5(raw.encode("utf-8")).hexdigest()[:8],
                            "uid": uid,
                            "model": engine,
                            "error_type": err["error"],
                            "ref_word": err["ref"],
                            "hyp_word": err["hyp"],
                            "start_time": round(st, 3),
                            "end_time": round(ed, 3),
                            "clip_path": clip_path_out,
                        }

                        # Persist immediately
                        try:
                            jsonl_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                            jsonl_f.flush()
                            csv_writer.writerow(
                                [
                                    entry["_id"],
                                    entry["uid"],
                                    entry["model"],
                                    entry["error_type"],
                                    entry["ref_word"],
                                    entry["hyp_word"],
                                    entry["start_time"],
                                    entry["end_time"],
                                    entry["clip_path"],
                                ]
                            )
                            csv_f.flush()
                        except Exception as e:
                            logger.error(
                                f"Failed to write entry for {uid}/{engine}/{idx}: {e}"
                            )

                # Mark this UID as completed
                completed.add(uid)
                with completed_file.open("a", encoding="utf-8") as cf:
                    cf.write(uid + "\n")

            except Exception as e:
                logger.error(f"Record {uid} processing failed: {e}")
                errored.add(uid)
                with errored_file.open("a", encoding="utf-8") as ef:
                    ef.write(uid + "\n")
                continue

    except KeyboardInterrupt:
        logger.warning("Interrupted by user; exiting gracefully.")

    finally:
        # Clean up file handles
        jsonl_f.close()
        csv_f.close()
        logger.info("Processing finished. Outputs flushed to disk.")


if __name__ == "__main__":
    main()
