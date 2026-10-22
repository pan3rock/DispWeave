#!/usr/bin/env python3
"""Analyze analytic NUTS stress tests; report failures without discarding chains."""
import argparse
import json
from pathlib import Path
import numpy as np
from posterior_diagnostics import diagnose

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
report = {}
for name in ('funnel_centered', 'funnel_noncentered', 'mixture_overlap', 'mixture_separated'):
    folder = args.directory / name
    draws = np.stack([np.loadtxt(folder / f'chain_{i}.txt') for i in range(4)])
    stats = np.loadtxt(folder / 'chains.txt')
    entry = dict(diagnostics=[diagnose(draws[:, :, i]) for i in range(draws.shape[-1])],
                 divergences=int(stats[:, 1].sum()), invalid=int(stats[:, 2].sum()),
                 depth_limits=int(stats[:, 3].sum()), evaluations=int(stats[:, 4].sum()),
                 seconds=float(stats[:, 5].sum()))
    if name.startswith('funnel'):
        # In either parameterization v/3 and x*exp(-v/2) are independent N(0,1).
        whitened = draws.copy()
        whitened[:, :, 1:] *= np.exp(-.5*draws[:, :, :1])
        whitened[:, :, 0] /= 3
        entry.update(whitened_mean=whitened.mean(axis=(0, 1)).tolist(),
                     whitened_second_moment=(whitened**2).mean(axis=(0, 1)).tolist(),
                     neck_fraction=float((draws[:, :, 0] < -3).mean()),
                     exact_neck_probability=0.15865525393145707)
    else:
        positive = draws[:, :, 0] > 0
        entry.update(positive_fraction_per_chain=positive.mean(axis=1).tolist(),
                     sign_crossings_per_chain=(positive[:, 1:] != positive[:, :-1]).sum(axis=1).tolist(),
                     exact_positive_probability=.5,
                     exact_second_moment=[2 if name.endswith('overlap') else 37, 1])
    report[name] = entry
    print(name, 'Rhat(first)=', entry['diagnostics'][0]['rhat'], 'divergences=', entry['divergences'])
with (args.directory / 'analysis.json').open('x') as f:
    json.dump(report, f, indent=2, allow_nan=False)
