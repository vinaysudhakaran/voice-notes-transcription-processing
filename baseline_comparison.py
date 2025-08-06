#!/usr/bin/env python3
"""
baseline_comparison.py

Compare multiple ASR models on a sample (or full set) of audio files,
computing WER against ground-truth transcripts stored in the manifest.

Configure DEFAULT_CONFIG below, then run:
    python baseline_comparison.py

Outputs:
- comparison_results.csv       : WER per file per model (flat table)
- comparison_results_pivot.csv : pivot of WER (UID × model)
- average_wer_results.csv      : average WER per model
"""
import os
import json
import random
import subprocess
import base64
from pathlib import Path
from typing import List, Dict, Any, Union

from dotenv import load_dotenv
import pandas as pd
from jiwer import wer
import openai
import azure.cognitiveservices.speech as speechsdk
from googleapiclient.discovery import build
import torch
from transformers.pipelines import pipeline

from utils.logger import logger

# -----------------------------------------------------------------------------
# === DEFAULT CONFIG ===
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Paths
    "manifest_path": "dataset/manifest/hindi/1/manifest.json",
    "prepared_dir": "prepared_audio",
    # Sampling & runs
    "sample_size": 1,  # <=0 or run_all=True → all records
    "run_all": False,  # True to ignore sample_size
    # Audio preprocessing
    "target_sample_rate": 16000,  # Hz
    # Language
    "language_code": "hi-IN",
    "mms_target_lang": "hin",
    # Model settings
    "openai_model": "whisper-1",
    "mms_model": "facebook/mms-1b-all",
    # Output files
    "results_csv": "results/comparison_results.csv",
    "average_csv": "results/average_wer_results.csv",
}


# -----------------------------------------------------------------------------
# ASR pipeline placeholders
# -----------------------------------------------------------------------------
asr_mms: Any = None
mms_processor: Any = None


# -----------------------------------------------------------------------------
# Utility: prepare uniform WAV files
# -----------------------------------------------------------------------------
def prepare_audio(
    input_path: Union[str, Path], prepared_dir: Union[str, Path], sample_rate: int
) -> Path:
    """
    Convert any audio file to mono WAV at `sample_rate` Hz, caching in `prepared_dir`.
    Returns Path to the prepared WAV.
    """
    inp = Path(input_path)
    out_dir = Path(prepared_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / f"{inp.stem}_{sample_rate}.wav"

    if not wav_path.exists():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(inp),
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(wav_path),
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            logger.debug(f"Converted {inp} → {wav_path} @{sample_rate}Hz")
        except Exception as e:
            logger.error(f"FFmpeg conversion failed for {inp}: {e}")
            raise
    return wav_path


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def load_manifest(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSON array manifest and return list of records."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_records(
    manifest: List[Dict[str, Any]], sample_size: int, run_all: bool
) -> List[Dict[str, Any]]:
    """Return either a random sample or the full list based on config."""
    if run_all or sample_size <= 0 or sample_size >= len(manifest):
        return manifest
    return random.sample(manifest, sample_size)


# -----------------------------------------------------------------------------
# Transcription functions
# -----------------------------------------------------------------------------
def transcribe_whisper(audio_path: Path) -> str:
    """Use OpenAI Whisper via API (requires OPENAI_API_KEY env var)."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    logger.info(f"\nRunning OpenAI Whisper on {audio_path.name}")
    with audio_path.open("rb") as f:
        resp = openai.audio.transcriptions.create(
            file=f, model=DEFAULT_CONFIG["openai_model"]
        )
    text = resp.text.strip()
    logger.debug(f"Whisper output: {text}\n")
    return text


def transcribe_azure(audio_path: Path) -> str:
    """
    Use Azure Speech-to-Text via SDK.
    Requires AZURE_SPEECH_KEY & AZURE_SPEECH_REGION env vars.
    Returns empty string on NoMatch.
    """
    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise RuntimeError("AZURE_SPEECH_KEY/REGION not set in environment")

    logger.info(f"\nRunning Azure STT on {audio_path.name}")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = DEFAULT_CONFIG["language_code"]

    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )
    result = recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = result.text.strip()
    elif result.reason == speechsdk.ResultReason.NoMatch:
        logger.warning("Azure NoMatch for %s", audio_path.name)
        text = ""
    else:
        raise RuntimeError(f"Azure STT error: {result.reason}")
    logger.debug(f"Azure output: {text}\n")
    return text


def transcribe_google(audio_path: Path) -> str:
    """
    Use Google Cloud Speech-to-Text via API key (developerKey).
    Requires GOOGLE_API_KEY env var.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in environment")

    logger.info(
        f"\nRunning Google STT on {audio_path.name} "
        f"(lang={DEFAULT_CONFIG['language_code']}, "
        f"sr={DEFAULT_CONFIG['target_sample_rate']})"
    )

    # read & base64‑encode the audio
    content = audio_path.read_bytes()
    content_b64 = base64.b64encode(content).decode("utf-8")

    # build the service with API key
    service = build("speech", "v1", developerKey=api_key)

    # form the request
    request_body = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": DEFAULT_CONFIG["target_sample_rate"],
            "languageCode": DEFAULT_CONFIG["language_code"],
        },
        "audio": {"content": content_b64},
    }
    response = service.speech().recognize(body=request_body).execute()

    # collect transcripts
    texts = [
        result["alternatives"][0]["transcript"]
        for result in response.get("results", [])
    ]
    text = " ".join(texts).strip()

    logger.debug(f"Google output: {text}\n")
    return text


def transcribe_mms(audio_path: Path) -> str:
    """
    Use Meta's MMS via Hugging Face pipeline.
    Requires audio sampled at DEFAULT_CONFIG['target_sample_rate'] Hz.
    """
    global asr_mms
    if asr_mms is None:
        device = 0 if torch.cuda.is_available() else -1
        asr_mms = pipeline(
            "automatic-speech-recognition",
            model=DEFAULT_CONFIG["mms_model"],
            model_kwargs={
                "target_lang": DEFAULT_CONFIG["mms_target_lang"],
                "ignore_mismatched_sizes": True,
            },
            device=device,
        )
        logger.info(f"Initialized MMS on {'GPU' if device == 0 else 'CPU'}\n")

    logger.info(
        f"Running MMS on {audio_path.name} (model={DEFAULT_CONFIG['mms_model']})"
    )
    out = asr_mms(str(audio_path))
    text = out["text"].strip()
    logger.info(f"MMS output: {text}\n")
    return text


# -----------------------------------------------------------------------------
# Main comparison logic
# -----------------------------------------------------------------------------
def main():
    load_dotenv()
    cfg = DEFAULT_CONFIG

    # Ensure output dirs exist
    Path(cfg["results_csv"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["average_csv"]).parent.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(cfg["manifest_path"])
    records = select_records(manifest, cfg["sample_size"], cfg["run_all"])

    results = []
    for rec in records:
        uid = rec["uid"]
        orig_audio = Path(rec["audio_path"])
        wav = prepare_audio(orig_audio, cfg["prepared_dir"], cfg["target_sample_rate"])
        gt = Path(rec["transcript_path"]).read_text(encoding="utf-8").strip()
        logger.debug(f"Ground Truth: {gt}\n")

        for name, fn in [
            ("whisper", transcribe_whisper),
            ("azure", transcribe_azure),
            ("google", transcribe_google),
            ("mms", transcribe_mms),
        ]:
            try:
                pred = fn(wav)
                score = wer(gt, pred)
                logger.debug(f"Score: {score:.3f}")
            except Exception as e:
                logger.error(f"Error in {name} for {uid}: {e}\n")
                score = None
            results.append({"uid": uid, "model": name, "wer": score})

    df = pd.DataFrame(results)
    # round WER to 3 decimals
    df["wer"] = df["wer"].apply(lambda x: round(x, 3) if isinstance(x, float) else x)
    df.to_csv(cfg["results_csv"], index=False)
    df.pivot(index="uid", columns="model", values="wer").to_csv(
        cfg["results_csv"].replace(".csv", "_pivot.csv")
    )
    df.groupby("model")["wer"].mean().reset_index().to_csv(
        cfg["average_csv"], index=False
    )

    logger.info(
        f"\nComparison complete.\n"
        f"Results: {cfg['results_csv']}\n"
        f"Pivot:   {cfg['results_csv'].replace('.csv', '_pivot.csv')}\n"
        f"Average: {cfg['average_csv']}\n"
    )


if __name__ == "__main__":
    main()
