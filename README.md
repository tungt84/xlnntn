# splitjoin

Small Python tool to split a file into parts (part001, part002, ...) and join them back with checksum verification.

Files added:

- [splitjoin.py](splitjoin.py)

Usage examples:


Split a file into 10 MB parts (parts will be created next to the original file and named like `model.safetensors.part0001`):

```bash
python splitjoin.py split -i model.safetensors -s 10M --checksum
```

If you want parts in a different directory (optional):

```bash
python splitjoin.py split -i model.safetensors -s 10M -o /path/to/dir --checksum
```

Join parts by pointing to the original filename (looks for `model.safetensors.partNNNN` in the same dir as `model.safetensors`):

```bash
python splitjoin.py join --orig model.safetensors -o restored.safetensors -c model.safetensors.sha256
```

Or join by specifying a parts directory and prefix (backwards-compatible):

```bash
python splitjoin.py join -d /path/to/parts -p part -o restored.bin
```

Notes:
- Part files are named `<prefix>NNN` (e.g., `part001`).
- Checksum files are simple text files containing `SHA256  filename`.
