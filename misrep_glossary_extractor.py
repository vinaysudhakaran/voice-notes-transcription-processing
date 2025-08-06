#!/usr/bin/env python3
"""
misrep_glossary_extractor.py

Extract all mis‑recognized words (subs, dels, ins) with timestamps
to build a glossary of errors.
"""

# flake8: noqa E402
import os

os.environ["GRPC_EXPERIMENTAL_EVENT_ENGINE"] = "false"

from misrep_glossary_core import main

if __name__ == "__main__":
    main()
