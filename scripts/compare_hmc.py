#!/usr/bin/env python3
"""Compare retained physical-Vs moments, with batch Monte Carlo errors (requires NumPy).
Acceptance gates fixed before reference runs: <= 6 combined batch SE, with
0.04 floor in pooled marginal-standard-deviation units; MCSE(mean) < 0.1 SD.
This is a distribution regression, not evidence of global mixing or coverage.
"""
import argparse
from pathlib import Path
import numpy as np


def estimate(chains):
    means = []
    variance = []
    for x in chains:
        n = len(x) // 40
        if n < 10:
            raise ValueError('Need at least 400 draws per chain')
        b = x[:40*n].reshape(40, n, -1).mean(axis=1)
        means.append(x.mean(axis=0))
        variance.append(b.var(axis=0, ddof=1) / 40)
    return np.mean(means, axis=0), np.sqrt(np.sum(variance, axis=0)) / len(chains)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ours', type=Path)
    parser.add_argument('reference', type=Path, nargs='+')
    args = parser.parse_args()
    ours = [np.loadtxt(p, ndmin=2) for p in sorted(args.ours.glob('chain_*_vs.txt'))]
    refs = [np.loadtxt(p, ndmin=2) for p in args.reference]
    if not ours or not refs:
        raise ValueError('Missing chain files')
    all_draws = np.concatenate(ours + refs)
    center, scale = all_draws.mean(axis=0), all_draws.std(axis=0)
    groups = [[(x-center)/scale for x in group] for group in (ours, refs)]
    passed = True
    for name, transform in [('mean', lambda x: x), ('centered_second_moment', lambda x: np.stack([x[:,i]*x[:,j] for i in range(x.shape[1]) for j in range(i, x.shape[1])], axis=1))]:
        a, sa = estimate([transform(x) for x in groups[0]])
        b, sb = estimate([transform(x) for x in groups[1]])
        tolerance = np.maximum(.04, 6*np.hypot(sa, sb))
        ok = np.abs(a-b) <= tolerance
        if name == 'mean':
            ok &= (sa < .1) & (sb < .1)
        passed &= bool(ok.all())
        print(name, 'difference=', a-b, 'combined_mcse=', np.hypot(sa, sb), 'pass=', ok)
    for label, group in [('DispWeave', ours), ('MCMCLib', refs)]:
        mean, se = estimate(group)
        print(label, 'mean=', mean, 'mean_mcse=', se, 'quantiles=', np.quantile(np.concatenate(group), [.025, .5, .975], axis=0))
    if not passed:
        raise SystemExit('Distribution comparison failed')


if __name__ == '__main__':
    main()
