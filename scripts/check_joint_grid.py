#!/usr/bin/env python3
"""Independent Vs-space trapezoid / log-noise Gauss-Legendre posterior check.

The grid stores physical chi-squared only. This code reconstructs priors, scale
normalization and marginalization independently of the C++ posterior/sampler.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from joint_results import read_config, save_json


def log_scale_integral(chi, count, noise, nodes, extent=10., with_scale=False):
    if noise['kind'] == 'fixed':
        scale = noise['scale']
        logp = -.5*chi/scale**2-count*np.log(scale)
        return (logp, np.full(len(chi), scale)) if with_scale else logp
    if noise['kind'] != 'lognormal':
        raise ValueError('This independent reference supports lognormal or fixed scales')
    z, w = np.polynomial.legendre.leggauss(nodes)
    mu, sd = noise['mean_log'], noise['sd_log']
    # +/-10 standard deviations contains all but 1.6e-23 of the PRIOR mass.
    # Node-doubling is checked; this is not a generic proof of tail truncation.
    eta = mu+extent*sd*z
    logweight = np.log(extent*sd*w)-np.log(sd*np.sqrt(2*np.pi))-.5*((eta-mu)/sd)**2
    a = -.5*chi[:, None]*np.exp(-2*eta)-count*eta+logweight
    maximum = a.max(axis=1)
    weight = np.exp(a-maximum[:, None])
    total = weight.sum(axis=1)
    logp = maximum+np.log(total)
    return (logp, (weight@np.exp(eta))/total) if with_scale else logp


def reference(grid, names, config, stride=1, nodes=384, extent=10.):
    axes = [np.unique(grid[:, i])[::stride] for i in range(2)]
    n0, n1 = [len(np.unique(grid[:, i])) for i in range(2)]
    if len(grid) != n0*n1:
        raise ValueError('Incomplete grid')
    grid = grid.reshape(n0, n1, -1)[::stride, ::stride].reshape(-1, grid.shape[1])
    if not np.array_equal(grid[:, :2], np.array(np.meshgrid(*axes, indexing='ij')).reshape(2, -1).T):
        raise ValueError('Reordered grid')
    logp = np.zeros(len(grid))
    scale_means = {}
    for b in config['observations']:
        chi = grid[:, 2+names.index(b['name'])]
        loglike, conditional_scale = log_scale_integral(chi, len(b['values']), b.get('noise', dict(kind='fixed', scale=1.)), nodes, extent, True)
        logp += loglike
        if b.get('noise', {}).get('kind', 'fixed') != 'fixed':
            scale_means[f"scale:{b['name']}"] = conditional_scale
    if config['prior']['kind'] != 'uniform':
        raise ValueError('Independent reference currently supports bounded uniform priors')
    if config['prior']['measure'] == 'log_vs':
        logp -= np.log(grid[:, :2]).sum(axis=1)
    elif config['prior']['measure'] != 'physical_vs':
        raise ValueError('Unknown prior measure')
    weights = []
    for axis in axes:
        delta = np.diff(axis)
        weights.append(np.r_[delta[0]/2, (delta[:-1]+delta[1:])/2, delta[-1]/2])
    weight = np.outer(*weights).ravel()*np.exp(logp-logp.max())
    weight /= weight.sum()
    means = weight@grid[:, :2]
    sd = np.sqrt(weight@((grid[:, :2]-means)**2))
    mass = weight.reshape(len(axes[0]), len(axes[1]))
    quantiles = []
    for i, axis in enumerate(axes):
        marginal = mass.sum(axis=1-i)
        quantiles.append(np.interp([.025, .5, .975], np.cumsum(marginal)-.5*marginal, axis).tolist())
    return dict(mean=means.tolist(), sd=sd.tolist(), quantiles=quantiles,
                scale_mean={label: float(weight@mean) for label, mean in scale_means.items()})


def validation_issues(comparison, summary):
    """Mean agreement is insufficient when poor mixing inflates the MCSE."""
    issues = []
    if not comparison['within_5mcse_and_2grid_error']:
        issues.append('Vs means disagree with independent integration')
    for label, row in comparison['scales'].items():
        if not row['within_5mcse_and_2grid_error']:
            issues.append(f'{label}: mean disagrees with independent integration')
    for label, row in summary['parameters'].items():
        if row['status'] != 'no_flag':
            issues.append(f"{label}: sampling diagnostic {row['status']}")
    if summary['numerical_proposals_including_warmup']:
        issues.append('Numerical proposal failures recorded')
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--consistent-grid', type=Path, required=True)
    parser.add_argument('--conflict-grid', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', action='append', help='Compare only named completed cases (repeatable); default all')
    args = parser.parse_args()
    grids = {}
    for name, path in [('consistent', args.consistent_grid), ('conflict', args.conflict_grid)]:
        with path.open() as stream:
            names = stream.readline().split()[4:]
        codes = np.loadtxt(path, dtype=str, usecols=2, skiprows=1)
        if not (codes == 'valid').all():
            raise ValueError('Unresolved physics in grid; do not renormalize across failed nodes')
        grid = np.loadtxt(path, usecols=[0, 1]+list(range(3, 3+len(names))), skiprows=1)
        grids[name] = (grid, names)
    if args.case:
        if any(not (args.suite/f'{name}.toml').is_file() for name in args.case):
            parser.error('Unknown case configuration')
        cases = args.case
    else:
        cases = json.loads((args.suite/'manifest.json').read_text())['cases']
    report = {}
    for name in cases:
        grid, names = grids['conflict' if name.startswith('conflict') else 'consistent']
        config = read_config(args.suite/f'{name}.toml')
        full = reference(grid, names, config)
        coarse = reference(grid, names, config, stride=2)
        fewer_nodes = reference(grid, names, config, nodes=192)
        wider = reference(grid, names, config, nodes=460, extent=12.)
        summary = json.loads((args.suite/name/'diagnostics.json').read_text())
        sample_means = np.array([summary['parameters'][f'Vs[{j}]']['mean'] for j in range(2)])
        mcse = np.array([summary['parameters'][f'Vs[{j}]']['mcse_mean'] for j in range(2)])
        grid_error = np.abs(np.array(full['mean'])-coarse['mean'])
        node_error = np.abs(np.array(full['mean'])-fewer_nodes['mean'])
        tail_error = np.abs(np.array(full['mean'])-wider['mean'])
        error = np.abs(sample_means-full['mean'])
        report[name] = dict(reference=full, grid_refinement_mean_change=grid_error.tolist(),
                            quadrature_refinement_mean_change=node_error.tolist(),
                            quadrature_extent_mean_change=tail_error.tolist(),
                            sample_mean=sample_means.tolist(), mcse_mean=mcse.tolist(),
                            absolute_mean_error=error.tolist(),
                            within_5mcse_and_2grid_error=bool((error <= 5*mcse+2*(grid_error+node_error+tail_error)).all()))
        report[name]['scales'] = {}
        for label, mean in full['scale_mean'].items():
            row = summary['parameters'][label]
            changes = [abs(mean-result['scale_mean'][label]) for result in [coarse, fewer_nodes, wider]]
            scale_error = abs(row['mean']-mean)
            report[name]['scales'][label] = dict(reference_mean=mean, sample_mean=row['mean'], mcse_mean=row['mcse_mean'],
                grid_node_extent_changes=changes, absolute_mean_error=scale_error,
                within_5mcse_and_2grid_error=bool(scale_error <= 5*row['mcse_mean']+2*sum(changes)))
        report[name]['sampling_diagnostics'] = {
            label: {key: row[key] for key in ['rhat', 'ess_bulk', 'ess_tail', 'status']}
            for label, row in summary['parameters'].items()}
        report[name]['validation_issues'] = validation_issues(report[name], summary)
        report[name]['passed'] = not report[name]['validation_issues']
        print(name, 'passed' if report[name]['passed'] else report[name]['validation_issues'], flush=True)
    save_json(args.output, report)
    if any(not case['passed'] for case in report.values()):
        raise SystemExit('Posterior validation failed; inspect the saved integration and sampling diagnostics')


if __name__ == '__main__':
    main()
