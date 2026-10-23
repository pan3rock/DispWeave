#!/usr/bin/env python3
"""Replay saved cutoff cases over the full trapped interval with two LU implementations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'benchmarks'))
from rayleigh_mode_audit import BoundarySystem


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir()
    exe = args.exe.resolve(); cases = json.loads(args.cases.read_text()); records = []
    frozen = {str(Path(c['source'])): digest(Path(c['source'])) for c in cases}
    if any(frozen[c['source']] != c['input_sha256'] for c in cases):
        raise ValueError('Original audited input changed')
    reference = ROOT/'benchmarks/rayleigh_mode_audit.py'
    save(out/'plan.json', dict(cases=cases, source_sha256=digest(args.cases),
        binary_sha256=digest(exe), reference_sha256=digest(reference), inputs=frozen,
        comparison='Native banded partial-pivot LU versus independent dense LAPACK LU; full trapped interval'))
    shutil.copyfile(reference, out/reference.name); shutil.copyfile(__file__, out/Path(__file__).name)
    for i, case in enumerate(cases):
        m = tomllib.loads(Path(case['source']).read_text())['model']
        table = np.c_[np.arange(len(m['vs'])), m['depth'], m['rho'], m['vs'], m['vp']]
        file = out/f'model-{i:03d}.txt'; np.savetxt(file, table, fmt='%.17g'); start = time.monotonic()
        call = subprocess.run([str(exe), str(file), str(case['frequency'])], capture_output=True, text=True)
        if call.returncode not in (0, 5): raise RuntimeError(call.stderr)
        native = json.loads(call.stdout)
        try:
            independent = BoundarySystem(table, case['frequency']).trapped_count()
            passed = native['resolved'] and native['count'] == independent['count'] == case['count']
        except ValueError as error:
            independent = dict(error=str(error)); passed = False
        records.append(dict(name=case['name'], native=native, independent=independent,
                            passed=passed, seconds=time.monotonic()-start))
        save(out/'cases.json', records)
        print(f"{i+1}/{len(cases)} {case['name']}: {passed}", flush=True)
    if any(digest(Path(path)) != value for path, value in frozen.items()):
        raise ValueError('Input changed during audit')
    summary = dict(queries=len(records), passed=sum(r['passed'] for r in records),
                   seconds=sum(r['seconds'] for r in records), inputs_unchanged=True)
    save(out/'summary.json', summary)
    return 0 if summary['passed'] == summary['queries'] else 4


if __name__ == '__main__': sys.exit(main())
