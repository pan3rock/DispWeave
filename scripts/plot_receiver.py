#!/usr/bin/env python3
"""Plot saved receiver function and raw response, retaining failed bins."""
import argparse
from pathlib import Path
import shlex
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('input', type=Path, help='receiver CLI output directory')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--time-max', type=float, help='Optional display limit; saved waveform is unchanged')
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
meta = dict(line.split('=', 1) for line in (a.input / 'metadata.txt').read_text().splitlines() if '=' in line)
if meta.get('backend') != 'elastic_isotropic_PSV_boundary_system':
    raise ValueError('Expected receiver response output')
wave = np.loadtxt(a.input / 'receiver.txt', usecols=(0, 1))
rows = [shlex.split(line) for line in (a.input / 'spectrum.txt').read_text().splitlines() if line and not line.startswith('#')]
f = np.array([float(row[0]) for row in rows])
valid = np.array([row[1] == 'valid' for row in rows])
radial = np.array([complex(float(row[2]), float(row[3])) if ok else complex(np.nan, np.nan) for row, ok in zip(rows, valid)])
vertical = np.array([complex(float(row[4]), float(row[5])) if ok else complex(np.nan, np.nan) for row, ok in zip(rows, valid)])
fig, axes = plt.subplots(2, 1, figsize=(9, 7))
axes[0].plot(wave[:, 0], wave[:, 1], color='#176B87')
axes[0].axvline(float(meta['time_shift']), color='gray', lw=.8, ls='--')
axes[0].axhline(0, color='gray', lw=.6)
axes[0].set(xlabel='Time relative to incident P (s)', ylabel='Radial receiver function',
            title=f"P receiver function — {meta['status']}\np={float(meta['slowness_s_km']):g} s/km; Gaussian a={float(meta['gaussian_width']):g} rad/s; water level={float(meta['water_level']):g}")
if a.time_max is not None:
    if not np.isfinite(a.time_max) or a.time_max <= wave[0, 0]:
        raise ValueError('time-max must be finite and after the first sample')
    axes[0].set_xlim(wave[0, 0], min(a.time_max, wave[-1, 0]))
axes[1].plot(f, np.abs(radial), label='|R|')
axes[1].plot(f, np.abs(vertical), label='|Z up|')
if (~valid).any():
    axes[1].plot(f[~valid], np.full((~valid).sum(), .05), 'x', color='firebrick',
                 transform=axes[1].get_xaxis_transform(), label='Failed frequency bin')
if meta['status'] != 'valid':
    axes[0].text(.5, .5, 'Waveform unavailable: '+meta.get('message', ''),
                 transform=axes[0].transAxes, ha='center', wrap=True, color='firebrick')
axes[1].set(xlabel='Frequency (Hz)', ylabel='Surface displacement / incident amplitude',
            title=f'Raw response spectrum — {(~valid).sum()} failed bins')
axes[1].legend()
for ax in axes:
    ax.grid(alpha=.2)
fig.tight_layout()
a.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output, dpi=160)
print(a.output)
