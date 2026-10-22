#!/usr/bin/env python3
"""Plot saved RJ summaries; keeps warning status visible. No inference is performed."""
import argparse
import json
from pathlib import Path
import tomllib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('diagnostics', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--truth-config', type=Path, help='Explicit synthetic truth; never inferred from a reference model')
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
r = json.loads(a.diagnostics.read_text())
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
layer = r['layer_probabilities'];k = np.array([x['k'] for x in layer]);probability = [x['probability'] for x in layer]
axes[0].bar(k, probability, color='0.8', label='Pooled estimate')
for c in range(len(r['chains'])):
    axes[0].plot(k, [x['per_chain'][c] for x in layer], 'o-', ms=4, lw=1, label=f'Chain {c}')
axes[0].set(xlabel='Total layers (including halfspace)', ylabel='Posterior probability', xticks=k, ylim=(0, 1))
axes[0].legend(fontsize=8)
profile = r['depth_profiles'];z = np.array([x['depth'] for x in profile]);q = np.array([x['quantiles'] for x in profile])
axes[1].fill_betweenx(z, q[:, 0], q[:, 2], alpha=.25, label='Pointwise 95% credible interval')
axes[1].plot(q[:, 1], z, label='Median')
if a.truth_config:
    with a.truth_config.open('rb') as f:
        truth = tomllib.load(f)['model']
    tops, vs = np.array(truth['depth']), np.array(truth['vs'])
    axes[1].plot(np.repeat(vs, 2), np.r_[tops[0], np.repeat(tops[1:], 2), z[-1]], 'k--', label='Synthetic truth')
axes[1].set(xlabel='Vs (km/s)', ylabel='Depth (km)', ylim=(z[-1], 0));axes[1].legend(fontsize=8)
bins = r['interface_bins'];centers = [.5*(x['lower']+x['upper']) for x in bins];widths = [x['upper']-x['lower'] for x in bins]
axes[2].barh(centers, [x['probability_any'] for x in bins], height=widths, alpha=.6)
axes[2].set(xlabel='Pr(at least one interface in bin)', ylabel='Depth (km)', xlim=(0, 1), ylim=(z[-1], 0))
for ax in axes:
    ax.grid(alpha=.2)
failures = sum(x['forward_failures']+x['numerical_failures'] for x in r['sampler'])
fig.suptitle(f"RJMCMC summaries — diagnostic status: {r['status']}\n{failures} failed target evaluations; conditional on declared priors, support and physics", color='firebrick' if r['status'] == 'warning' else 'black', fontsize=11)
fig.tight_layout();a.output.parent.mkdir(parents=True, exist_ok=True);fig.savefig(a.output, dpi=160)
print(a.output)
