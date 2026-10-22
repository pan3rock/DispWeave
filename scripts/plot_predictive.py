#!/usr/bin/env python3
"""Plot latent curves separately from replicated observations; refuses failed draws."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
latent = np.loadtxt(a.directory / 'latent.txt', ndmin=2)
replicated = np.loadtxt(a.directory / 'replicated.txt', ndmin=2)
observed = np.loadtxt(a.directory / 'observations.txt', ndmin=2)
status = np.loadtxt(a.directory / 'draws.txt', dtype=str, ndmin=2, skiprows=1)
if not np.isfinite(latent).all() or not np.isfinite(replicated).all() or not (status[:, 3] == 'valid').all():
    raise SystemExit('Predictive failures present: inspect draws.txt; refusing to silently drop samples')
if latent.shape != replicated.shape or latent.shape[1] != len(observed):
    raise ValueError('Predictive dimensions do not match observations')
modes = sorted(set(observed[:, 2].astype(int)))
fig, axes = plt.subplots(1, len(modes), figsize=(6*len(modes), 4.8), squeeze=False)
for ax, mode in zip(axes[0], modes):
    indices = np.flatnonzero(observed[:, 2] == mode)
    indices = indices[np.argsort(observed[indices, 0])]
    frequency = observed[indices, 0]
    latent_q = np.quantile(latent[:, indices], [.025, .5, .975], axis=0)
    repl_q = np.quantile(replicated[:, indices], [.025, .5, .975], axis=0)
    ax.fill_between(frequency, repl_q[0], repl_q[2], color='C0', alpha=.13, label='95% replicated observation interval')
    ax.fill_between(frequency, latent_q[0], latent_q[2], color='C0', alpha=.35, label='95% latent curve interval')
    ax.plot(frequency, latent_q[1], color='C0', label='Latent median')
    ax.errorbar(frequency, observed[indices, 1], yerr=observed[indices, 3], fmt='k.', label='Observed ± marginal sigma')
    ax.set(xlabel='Frequency (Hz)', ylabel='Phase velocity (km/s)', title=f'Rayleigh mode {mode}')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
metadata = (a.directory / 'metadata.txt').read_text()
kind = 'Transdimensional posterior predictive check' if 'posterior=transdimensional' in metadata else 'Posterior predictive check conditional on fixed parameterization'
fig.suptitle(kind + '\nPointwise intervals, not simultaneous bands\nCovariance retained when generating replicas', fontsize=10)
fig.tight_layout()
output = a.output or a.directory / 'predictive.png'
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=160)
summary = dict(selected_draws=len(latent), posterior_predictive_p=float(np.mean(status[:, 5].astype(float) >= status[:, 4].astype(float))),
               observed_chi2_mean=float(np.mean(status[:, 4].astype(float))),
               replicated_chi2_mean=float(np.mean(status[:, 5].astype(float))),
               interpretation='Discrepancy is squared whitened residual. Data were used to fit the posterior; this is not a uniform calibration p-value.')
print(json.dumps(summary, indent=2))
print(output)
