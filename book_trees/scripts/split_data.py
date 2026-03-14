"""
scripts/split_data.py

Merges one or more .spacy files, shuffles, and splits into train/dev sets.
Usage:
    python scripts/split_data.py data/2026-03-13.spacy
    python scripts/split_data.py data/*.spacy
    python scripts/split_data.py data/batch1.spacy data/batch2.spacy --split 0.85
    python scripts/split_data.py data/*.spacy --output-dir corpus/
"""

import argparse
import random
from pathlib import Path

import spacy
from spacy.tokens import DocBin


def main():
    parser = argparse.ArgumentParser(description="Merge and split .spacy files for training.")
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more .spacy files to merge.")
    parser.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training (default: 0.8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write train.spacy and dev.spacy (default: current dir).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Validate inputs
    for path in args.inputs:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    if not (0.0 < args.split < 1.0):
        raise ValueError(f"--split must be between 0 and 1, got {args.split}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge all .spacy files
    nlp = spacy.blank("en")
    all_docs = []

    for path in args.inputs:
        db = DocBin().from_disk(path)
        docs = list(db.get_docs(nlp.vocab))
        print(f"  Loaded {len(docs):>5} docs from {path}")
        all_docs.extend(docs)

    print(f"\n  Total docs: {len(all_docs)}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_docs)

    split_idx = int(len(all_docs) * args.split)
    train_docs = all_docs[:split_idx]
    dev_docs = all_docs[split_idx:]

    # Write output
    train_path = args.output_dir / "train.spacy"
    dev_path = args.output_dir / "dev.spacy"

    DocBin(docs=train_docs).to_disk(train_path)
    DocBin(docs=dev_docs).to_disk(dev_path)

    print(f"\n  Train: {len(train_docs)} docs → {train_path}")
    print(f"  Dev:   {len(dev_docs)} docs → {dev_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()