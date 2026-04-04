#!/usr/bin/env python3
"""Split and join files with checksum support.

Usage examples:
  python splitjoin.py split -i bigfile.bin -s 10M -o parts_dir --prefix part
  python splitjoin.py join -d parts_dir -p part -o restored.bin --checksum bigfile.bin.sha256
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from typing import List


def parse_size(s: str) -> int:
    s = s.strip()
    if s.isdigit():
        return int(s)
    unit = s[-1].upper()
    num = float(s[:-1])
    if unit == 'K':
        return int(num * 1024)
    if unit == 'M':
        return int(num * 1024 ** 2)
    if unit == 'G':
        return int(num * 1024 ** 3)
    raise argparse.ArgumentTypeError(f"Invalid size: {s}")


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_checksum_file(orig_path: str, checksum_path: str | None = None) -> str:
    if checksum_path is None:
        checksum_path = orig_path + '.sha256'
    digest = compute_sha256(orig_path)
    with open(checksum_path, 'w') as cf:
        cf.write(f"{digest}  {os.path.basename(orig_path)}\n")
    return checksum_path


def split_file(input_path: str, part_size: int, out_dir: str, prefix: str = 'part') -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    parts: List[str] = []
    i = 1
    with open(input_path, 'rb') as src:
        while True:
            chunk = src.read(part_size)
            if not chunk:
                break
            if prefix:
                name = f"{prefix}{i:04d}"
            else:
                base = os.path.basename(input_path)
                name = f"{base}.part{i:04d}"
            full = os.path.join(out_dir, name)
            with open(full, 'wb') as pf:
                pf.write(chunk)
            parts.append(full)
            i += 1
    return parts


def _part_sort_key(name: str, prefix: str) -> int:
    b = os.path.basename(name)
    if b.startswith(prefix):
        s = b[len(prefix):]
        try:
            return int(s)
        except Exception:
            pass
    # fallback
    return 0


def join_files(parts_dir: str, prefix: str, out_path: str) -> None:
    # collect parts
    files = [os.path.join(parts_dir, f) for f in os.listdir(parts_dir) if f.startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"No parts with prefix '{prefix}' found in {parts_dir}")
    # try numeric ordering
    # extract numeric suffix after prefix (handles .partNNNN)
    def _extract_num(p: str) -> int:
        b = os.path.basename(p)
        s = b[len(prefix):]
        # strip non-digits
        num = ''.join(ch for ch in s if ch.isdigit())
        try:
            return int(num)
        except Exception:
            return 0

    try:
        files.sort(key=_extract_num)
    except Exception:
        files.sort()

    with open(out_path, 'wb') as out:
        for p in files:
            with open(p, 'rb') as pf:
                for chunk in iter(lambda: pf.read(1024 * 1024), b''):
                    out.write(chunk)


def read_checksum_file(path: str) -> tuple[str, str]:
    # returns (hex, filename)
    with open(path, 'r') as f:
        line = f.readline().strip()
    parts = line.split()
    if not parts:
        raise ValueError('Empty checksum file')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return parts[0], ''


def main(argv=None):
    parser = argparse.ArgumentParser(description='Split and join files with checksum')
    sub = parser.add_subparsers(dest='cmd')

    sp = sub.add_parser('split', help='Split a file into parts')
    sp.add_argument('-i', '--input', required=True, help='Input file to split')
    sp.add_argument('-s', '--size', required=True, help='Part size (e.g. 10M, 512K, or bytes)')
    sp.add_argument('-o', '--out-dir', default=None, help='Output directory for parts (default: same dir as input)')
    sp.add_argument('--prefix', default=None, help='Prefix for part files (default: <inputname>.part)')
    sp.add_argument('--checksum', action='store_true', help='Write checksum file for original')

    jp = sub.add_parser('join', help='Join parts into a file')
    jp.add_argument('-d', '--parts-dir', help='Directory containing parts (default: same dir as --orig or current dir)')
    jp.add_argument('-p', '--prefix', default=None, help='Prefix used for parts (default: <origname>.part)')
    jp.add_argument('-o', '--output', required=True, help='Output file path')
    jp.add_argument('-c', '--checksum-file', help='Checksum file to verify after joining')
    jp.add_argument('--orig', help='Original filename used when creating parts; join files named <orig>.partXXXX in same dir as orig')

    args = parser.parse_args(argv)

    if args.cmd == 'split':
        part_size = parse_size(args.size)
        out_dir = args.out_dir
        if out_dir is None:
            out_dir = os.path.dirname(os.path.abspath(args.input)) or '.'
        print(f"Splitting {args.input} into parts of {part_size} bytes in {out_dir}")
        parts = split_file(args.input, part_size, out_dir, args.prefix)
        print(f"Wrote {len(parts)} parts to {out_dir}")
        if args.checksum:
            ck = write_checksum_file(args.input)
            print(f"Wrote checksum to {ck}")
    elif args.cmd == 'join':
        # determine parts directory and prefix
        if args.orig:
            parts_dir = os.path.dirname(os.path.abspath(args.orig)) or '.'
            prefix = os.path.basename(args.orig) + '.part'
        else:
            parts_dir = args.parts_dir or '.'
            prefix = args.prefix or ''

        print(f"Joining parts from {parts_dir} with prefix '{prefix}' into {args.output}")
        join_files(parts_dir, prefix, args.output)
        print(f"Joined to {args.output}")
        if args.checksum_file:
            expected, name = read_checksum_file(args.checksum_file)
            actual = compute_sha256(args.output)
            if actual.lower() == expected.lower():
                print('Checksum OK')
                return 0
            else:
                print('Checksum MISMATCH', file=sys.stderr)
                print(f'expected {expected}', file=sys.stderr)
                print(f'actual   {actual}', file=sys.stderr)
                return 2
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
