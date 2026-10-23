#!/usr/bin/env python3
"""Plot all external-example LM attempts, retaining failure classifications.

Optional posterior-diagnostics.json is read from each case directory. Intervals
are conditional on supplied geometry, elastic properties, bounds and errors.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from check_qedisp_examples import forward_rows, check_queries
from joint_results import read_config, save_json


def plot_case(work, name, output):
    config = read_config(work/'forward.toml')
    model = config['model'];z = np.array(model['depth'])
    edges = np.r_[z, z[-1]*1.2 if z[-1] else 1.]
    data = np.loadtxt(work/'observations.txt', ndmin=2)
    runs = np.loadtxt(work/'lm/runs.txt', dtype=str, ndmin=2, skiprows=1)
    root = forward_rows(work/'forward-none.txt')
    check_queries(root, data[:, 0], data[:, 2])
    reference = np.array([float(r[3]) if r[4] == 'valid' else np.nan for r in root])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    axes[0].stairs(model['vs'], edges, orientation='horizontal', baseline=None,
                   color='black', linewidth=2, label='Supplied model')
    summary = dict(runs=[], interpretation='All attempts shown; stalled/failed runs are not converged solutions. Error bars are assumed, not source measurement errors.')
    for row in runs:
        index = int(row[0]);termination = row[-2];reason = row[-1]
        final = np.loadtxt(work/f'lm/run_{index}_model.txt', ndmin=2)
        style = '-' if termination == 'converged' else '--'
        axes[0].stairs(final[:, 3], edges, orientation='horizontal', baseline=None,
                       color=f'C{index}', linestyle=style, alpha=.8, label=f'LM {index}: {termination}')
        rows = forward_rows(work/f'lm/run_{index}_prediction.txt')
        check_queries(rows, data[:, 0], data[:, 2])
        values = np.array([float(r[3]) if r[4] == 'valid' else np.nan for r in rows])
        invalid = int(np.sum(~np.isfinite(values)))
        summary['runs'].append(dict(run=index, termination=termination, reason=reason,
            cost=float(row[2]), projected_gradient=float(row[9]), invalid_queries=invalid,
            rms_km_s=float(np.sqrt(np.mean((values-data[:, 1])**2))) if invalid == 0 else None,
            vs=final[:, 3].tolist()))
        for mode in np.unique(data[:, 2]).astype(int):
            indices = np.flatnonzero(data[:, 2] == mode)
            indices = indices[np.argsort(data[indices, 0])]
            # Invalid predictions remain NaN gaps, with their counts below.
            axes[1].plot(data[indices, 0], values[indices], color=f'C{mode}', linestyle=style, alpha=.4)
    for mode in np.unique(data[:, 2]).astype(int):
        indices = np.flatnonzero(data[:, 2] == mode)
        indices = indices[np.argsort(data[indices, 0])]
        axes[1].errorbar(data[indices, 0], data[indices, 1], yerr=data[indices, 3], fmt='.',
                         color=f'C{mode}', label=f'Data mode {mode} (assumed sigma)')
        axes[1].plot(data[indices, 0], reference[indices], color=f'C{mode}', linestyle=':', linewidth=2)
    posterior = work/'posterior-diagnostics.json'
    if posterior.exists():
        diagnostics = json.loads(posterior.read_text());parameters = diagnostics['parameters']
        if len(parameters) != len(z):
            raise ValueError('Posterior/model parameter dimensions disagree')
        quantiles = np.array([p['quantiles'] for p in parameters])
        axes[0].fill_betweenx(edges, np.r_[quantiles[:, 0], quantiles[-1, 0]],
            np.r_[quantiles[:, 2], quantiles[-1, 2]], step='post', alpha=.2, color='C0', label='NUTS 95% pointwise interval')
        axes[0].stairs(quantiles[:, 1], edges, orientation='horizontal', baseline=None,
                       color='C0', linewidth=2, label='NUTS median')
        summary['posterior'] = dict(max_rhat=max(p['rhat'] for p in parameters),
            min_bulk_ess=min(p['ess_bulk'] for p in parameters), min_tail_ess=min(p['ess_tail'] for p in parameters),
            parameter_statuses=[p['status'] for p in parameters], sampler=diagnostics['sampler'])
    axes[0].invert_yaxis()
    axes[0].set(xlabel='Vs (km/s)', ylabel='Depth (km)', title='Fixed interfaces, Vp and density')
    axes[1].set(xlabel='Frequency (Hz)', ylabel='Phase velocity (km/s)',
                title='Dotted: supplied model; solid/dashed: LM attempts')
    if data[:, 0].max()/data[:, 0].min() > 30:
        axes[1].set_xscale('log')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    counts = ', '.join(f"{r['run']}: {r['invalid_queries']}" for r in summary['runs'])
    fig.suptitle(f'{name} — QEDispInv synthetic example\nInvalid terminal queries by LM run: {counts}', fontsize=11)
    fig.savefig(output/f'{name}.png', dpi=160)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();args.output.mkdir()
    summaries = {}
    for root in args.roots:
        for name in json.loads((root/'forward-audit.json').read_text()):
            if name in summaries:
                raise ValueError(f'Duplicate case: {name}')
            summaries[name] = plot_case(root/name, name, args.output)
    save_json(args.output/'summary.json', summaries)
    print(args.output)


if __name__ == '__main__':
    main()
