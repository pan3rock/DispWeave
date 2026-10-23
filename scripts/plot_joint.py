#!/usr/bin/env python3
"""Plot joint CLI predictions, conditional Vs intervals and all chain diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joint_results import load, summarize


def plot(directory, output):
    doc, samples, labels, _, predictions = load(directory)
    summary = summarize(directory)
    flagged = any(row['status'] != 'no_flag' for row in summary['parameters'].values())
    flagged |= summary['numerical_proposals_including_warmup'] != 0
    warning = '\nWARNING: sampling checks failed; pooled summaries are not reliable posterior estimates' if flagged else ''
    output.mkdir()
    n = len(doc['model']['vs'])
    blocks = doc['observations']
    fig, axes = plt.subplots(1, 1+len(blocks), figsize=(4.2*(1+len(blocks)), 5), constrained_layout=True)
    quantiles = np.quantile(samples[:, :, :n].reshape(-1, n), [.025, .5, .975], axis=0)
    depth = np.array(doc['model']['depth']+[doc['model']['depth'][-1]+10.])
    profiles = np.c_[quantiles, quantiles[:, -1]]
    ax = axes[0]
    ax.fill_betweenx(depth, profiles[0], profiles[2], step='post', alpha=.22, color='C0', label='95% conditional interval')
    ax.stairs(quantiles[1], depth, orientation='horizontal', baseline=None, label='Posterior median')
    if 'synthetic' in doc:
        generating = {}
        for b in blocks:
            truth = tuple(b.get('synthetic_vs', doc['synthetic']['vs']))
            generating.setdefault(truth, []).append(b['name'])
        for j, (truth, names) in enumerate(generating.items()):
            label = 'Generating Vs' if len(generating) == 1 else '/'.join(names)+' generating Vs'
            ax.stairs(truth, depth, orientation='horizontal', baseline=None, color='black' if j == 0 else 'C3',
                      ls='--' if j == 0 else ':', label=label)
    vp_policy = 'Vp/Vs' if doc['parameters'].get('vp_vs_ratio', 0.) else 'Vp'
    ax.invert_yaxis();ax.set(xlabel='Vs (km/s)', ylabel='Depth (km)', title=f'Fixed interfaces, {vp_policy} and density')
    ax.legend(fontsize=8)
    for ax, b in zip(axes[1:], blocks):
        name = b['name'];p = predictions[name]
        latent = np.quantile(p['latent'], [.025, .5, .975], axis=0)
        replicate = np.quantile(p['replicate'], [.025, .5, .975], axis=0)
        sigma = np.array(b['sigma']) if 'sigma' in b else np.sqrt(np.diag(b['covariance']))
        if 'frequency_hz' in b:
            x = np.array(b['frequency_hz']);groups = np.array(b['mode']);xlabel = 'Frequency (Hz)'
        else:
            x = np.array(b['sample_indices'])*b['processing']['dt']-b['processing']['pre_arrival']
            groups = np.zeros(len(x), dtype=int);xlabel = 'Time (s)'
        for group in np.unique(groups):
            ids = np.flatnonzero(groups == group);ids = ids[np.argsort(x[ids])]
            color = f'C{group}'
            ax.fill_between(x[ids], replicate[0, ids], replicate[2, ids], color=color, alpha=.12,
                            label='95% predictive' if group == groups.min() else None)
            ax.fill_between(x[ids], latent[0, ids], latent[2, ids], color=color, alpha=.3,
                            label='95% latent' if group == groups.min() else None)
            ax.plot(x[ids], latent[1, ids], color=color, label=f'Median (mode {group})' if 'mode' in b else 'Median')
            ax.errorbar(x[ids], np.array(b['values'])[ids], yerr=sigma[ids], fmt='.', color=color,
                        label='Data ± base sigma' if group == groups.min() else None, ms=5, capsize=2)
        ax.set(xlabel=xlabel, ylabel=b['unit'], title=name);ax.legend(fontsize=8)
    fig.suptitle(f'{directory.name}: conditional posterior and predictive uncertainty{warning}', fontsize=13)
    fig.savefig(output/'fits.png', dpi=160);plt.close(fig)
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2*len(labels)), squeeze=False, constrained_layout=True)
    # All retained samples shown. No thinning or removal of repeated states.
    for j, label in enumerate(labels):
        for c in range(samples.shape[0]):
            axes[j, 0].plot(samples[c, :, j], alpha=.65, lw=.45, label=f'Chain {c}')
        axes[j, 0].set(ylabel=label)
    axes[0, 0].legend(ncol=samples.shape[0], fontsize=8)
    axes[-1, 0].set_xlabel('Retained iteration')
    fig.suptitle(f'{directory.name}: trace plots (warmup excluded){warning}')
    fig.savefig(output/'traces.png', dpi=140);plt.close(fig)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.atleast_2d(np.corrcoef(samples.reshape(-1, len(labels)), rowvar=False))
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap='RdBu_r')
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha='right');ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(corr[i, j]) > .65 else 'black')
    ax.set_title(f'{directory.name}: retained-sample correlation')
    if flagged:
        fig.suptitle('WARNING: sampling checks failed; correlation may reflect trapped chains', fontsize=10)
    fig.colorbar(image, ax=ax)
    fig.savefig(output/'correlations.png', dpi=160);plt.close(fig)


def comparison(suite, cases, output):
    summaries = {name: json.loads((suite/name/'diagnostics.json').read_text()) for name in cases}
    flagged = {name: any(row['status'] != 'no_flag' for row in s['parameters'].values()) or
               s['numerical_proposals_including_warmup'] != 0 for name, s in summaries.items()}
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True, constrained_layout=True)
    for j, ax in enumerate(axes):
        for i, name in enumerate(cases):
            row = summaries[name]['parameters'][f'Vs[{j}]']
            lo, median, hi = row['quantiles']
            color = 'C3' if name.startswith('conflict') else 'C0'
            if flagged[name]:
                ax.text(.02, i, 'Sampling failed; interval withheld', transform=ax.get_yaxis_transform(),
                        color='C3', fontsize=9, va='center')
            else:
                ax.errorbar(median, i, xerr=[[median-lo], [hi-median]], fmt='o', color=color, capsize=3)
        ax.axvline([2.6, 3.6][j], color='black', ls='--', label='Consistent generating Vs')
        if j == 0:
            ax.axvline(2.9, color='C3', ls=':', label='Conflicting RF generating Vs')
        ax.set(xlabel='Vs (km/s)', title=f'Vs[{j}]: median and 95% conditional interval')
        ax.set_yticks(range(len(cases)), [name+' (flagged)' if flagged[name] else name for name in cases])
        ax.legend(fontsize=8);ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis();fig.savefig(output, dpi=170);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', action='store_true')
    args = parser.parse_args()
    if args.suite:
        args.output.mkdir()
        cases = json.loads((args.directory/'manifest.json').read_text())['cases']
        for name in cases:
            plot(args.directory/name, args.output/name)
        comparison(args.directory, cases, args.output/'comparison.png')
    else:
        plot(args.directory, args.output)


if __name__ == '__main__':
    main()
