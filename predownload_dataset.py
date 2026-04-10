#!/usr/bin/env python3
"""Simple script to pre-download the NYU-Mll MultiNLI dataset and save splits to disk.

Usage: python predownload_dataset.py
"""
from pathlib import Path
from datasets import load_dataset


def save_splits(dataset_name: str, out_base: Path):
    try:
        print(f"Loading dataset {dataset_name}...")
        ds = load_dataset(dataset_name)
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return

    out_base.mkdir(parents=True, exist_ok=True)
    for split_name, split in ds.items():
        out_dir = out_base / split_name
        if out_dir.exists():
            print(f"Split {split_name} already saved at {out_dir}, skipping.")
            continue
        print(f"Saving split '{split_name}' to {out_dir} ...")
        split.save_to_disk(str(out_dir))


def main():
    data_root = Path("DATASETS")

    # Download SNLI (official HF dataset id: 'snli')
    snli_out = data_root / "snli"
    save_splits("stanfordnlp/snli", snli_out)

    # Download MultiNLI (use nyu-mll fork if desired)
    multi_out = data_root / "multi_nli"
    # prefer nyu-mll/multi_nli if available, fallback to 'multi_nli'
    try:
        save_splits("nyu-mll/multi_nli", multi_out)
    except Exception:
        save_splits("multi_nli", multi_out)

    print("Done. Saved dataset folders under DATASETS:")
    for p in sorted(data_root.iterdir()):
        if p.is_dir():
            print(" -", p)


if __name__ == "__main__":
    main()
