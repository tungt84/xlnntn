#!/usr/bin/env python3
"""Simple script to pre-download the NYU-Mll MultiNLI dataset and save splits to disk.

Usage: python predownload_dataset.py
"""
from pathlib import Path
from datasets import load_dataset


def main():
    out_base = Path("DATASETS") / "multi_nli"
    out_base.mkdir(parents=True, exist_ok=True)

    print("Loading dataset nyu-mll/multi_nli...")
    ds = load_dataset("nyu-mll/multi_nli")

    for split_name, split in ds.items():
        out_dir = out_base / split_name
        print(f"Saving split '{split_name}' to {out_dir} ...")
        split.save_to_disk(str(out_dir))

    print("Done. Saved splits under:")
    for p in sorted(out_base.iterdir()):
        if p.is_dir():
            print(" -", p)


if __name__ == "__main__":
    main()
