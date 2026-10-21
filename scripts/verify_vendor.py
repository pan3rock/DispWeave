#!/usr/bin/env python3
"""Verify bundled source hashes with the Python standard library only."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "third_party"
count = 0
for manifest in sorted(root.glob("*/provenance.json")):
    data = json.loads(manifest.read_text())
    for relative, expected in data["sha256"].items():
        source = manifest.parent / relative
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Hash mismatch: {source}")
        count += 1
    print(f"OK {manifest.parent.name}")
print(f"Verified {count} bundled files")
