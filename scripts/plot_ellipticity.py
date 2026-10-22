#!/usr/bin/env python3
"""Plot saved modal ellipticity; missing rows remain visible and break curves."""
import argparse
from pathlib import Path
import shlex
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('input', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
text = a.input.read_text()
if 'observable=rayleigh_modal_ellipticity' not in text:
    raise ValueError('Expected explicit modal Rayleigh ellipticity output')
rows = [shlex.split(l) for l in text.splitlines() if l and not l.startswith('#')]
fig, ax = plt.subplots(figsize=(8, 5))
failures = 0
for mode in sorted({int(r[2]) for r in rows}):
    selected = sorted([r for r in rows if int(r[2]) == mode], key=lambda r: float(r[1]))
    f = np.array([float(r[1]) for r in selected])
    valid = np.array([r[6] == 'valid' for r in selected])
    ratio = np.array([float(r[5]) if ok else np.nan for r, ok in zip(selected, valid)])
    if np.any(~np.isfinite(ratio[valid])):
        raise ValueError('Nonfinite valid ellipticity')
    ax.plot(f, ratio, '.-', label=f'Rayleigh mode {mode}')
    if (~valid).any():
        ax.plot(f[~valid], np.full((~valid).sum(), .03+.03*mode), 'x', color='firebrick',
                transform=ax.get_xaxis_transform(), label=f'Mode {mode}: unavailable / near pole')
        failures += int((~valid).sum())
ax.set(xlabel='Frequency (Hz)', ylabel='Surface |horizontal| / |vertical|',
       title=f'Modal Rayleigh ellipticity — {failures} unavailable rows\nNot an ambient-noise H/V prediction')
ax.grid(alpha=.2);ax.legend();fig.tight_layout()
a.output.parent.mkdir(parents=True, exist_ok=True);fig.savefig(a.output, dpi=160)
print(a.output)
