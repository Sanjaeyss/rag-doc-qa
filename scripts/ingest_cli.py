#!/usr/bin/env python
"""Ingest every supported file in a folder without starting the API server.

Usage:
    python scripts/ingest_cli.py ./data/uploads
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import ingest_file  # noqa: E402

SUPPORTED = {".pdf", ".txt", ".md"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder of documents to ingest")
    args = parser.parse_args()

    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No supported files ({', '.join(SUPPORTED)}) found in {args.folder}")
        return

    for path in files:
        n = ingest_file(path)
        print(f"{path.name}: {n} chunks")


if __name__ == "__main__":
    main()
