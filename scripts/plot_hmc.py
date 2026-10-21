#!/usr/bin/env python3
"""Plot fixed-layer marginal Vs draws and retained-chain traces; NumPy/Matplotlib required."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
paths = sorted(args.directory.glob('chain_*_vs.txt'))
if not paths:
    raise SystemExit('No retained HMC chains')
chains = [np.loadtxt(p, ndmin=2) for p in paths]
n = chains[0].shape[1]
fig, axes = plt.subplots(n, 2, figsize=(10, max(3, 2.4*n)), squeeze=False)
for layer in range(n):
    for path, chain in zip(paths, chains):
        label = path.stem.removesuffix('_vs')
        axes[layer, 0].plot(chain[:, layer], lw=.5, alpha=.65, label=label)
        axes[layer, 1].hist(chain[:, layer], bins=45, density=True, histtype='step', label=label)
    axes[layer, 0].set(ylabel=f'Layer {layer} Vs (km/s)', xlabel='Retained draw')
    axes[layer, 1].set(xlabel='Vs (km/s)', ylabel='Marginal density')
axes[0, 0].legend(fontsize=8)
fig.suptitle('HMC posterior conditional on fixed parameterization\nTraces and marginals do not establish global mixing')
fig.tight_layout()
output = args.output or args.directory / 'posterior.png'
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=160)
print(output)
