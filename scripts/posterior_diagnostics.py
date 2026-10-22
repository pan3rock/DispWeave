#!/usr/bin/env python3
"""Rank/folded split R-hat, bulk/tail ESS and mean MCSE; only NumPy is required.
Definitions: Vehtari et al. (2021), doi:10.1214/20-BA1221.
Inputs contain retained draws, including repeated states, never warmup.
"""
# The terminal-even ESS correction follows ArviZ 0.21.0 diagnostics._ess.
# ArviZ developers, Apache-2.0 (see repository LICENSE). Adapted here to
# standalone NumPy FFT processing; no ArviZ dependency at runtime.
import argparse
import json
from pathlib import Path
from statistics import NormalDist
import numpy as np


def split(x):
    half = x.shape[1] // 2
    return np.concatenate((x[:, :half], x[:, -half:]))


def rank_normalize(x):
    flat = x.ravel()
    order = np.argsort(flat, kind='stable')
    sorted_x = flat[order]
    begin = np.r_[0, np.flatnonzero(sorted_x[1:] != sorted_x[:-1]) + 1]
    end = np.r_[begin[1:], len(flat)]
    ranks = np.empty(len(flat))
    ranks[order] = np.repeat((begin + end + 1) / 2, end - begin)
    probabilities = (ranks - .375) / (len(flat) + .25)
    normal = NormalDist()
    return np.fromiter((normal.inv_cdf(p) for p in probabilities), float).reshape(x.shape)


def basic_rhat(x):
    m, n = x.shape
    within = x.var(axis=1, ddof=1).mean()
    between = n * x.mean(axis=1).var(ddof=1)
    if within == 0 or np.all(np.ptp(x, axis=1) == 0):
        return np.nan if between == 0 else np.inf
    return np.sqrt(((n - 1) / n * within + between / n) / within)


def effective_size(x):
    """Multichain variance estimate and Geyer initial positive/monotone pairs."""
    m, n = x.shape
    if np.ptp(x) == 0:
        return float(x.size)
    centered = x - x.mean(axis=1, keepdims=True)
    size = 1 << (2*n - 1).bit_length()
    transform = np.fft.rfft(centered, n=size, axis=1)
    acov = np.fft.irfft(transform * transform.conj(), n=size, axis=1)[:, :n] / n
    within = acov[:, 0].mean() * n / (n - 1)
    variance = within * (n - 1) / n + x.mean(axis=1).var(ddof=1)
    if not np.isfinite(variance) or variance <= 0:
        return np.nan
    rho = 1 - (within - acov.mean(axis=0)) / variance
    rho[0] = 1
    # Stan/ArviZ terminal-even-lag correction to Geyer's paired sequence.
    retained = np.zeros(n)
    retained[:2] = rho[:2]
    cursor, last_even, pair_sum = 1, 1., rho[0] + rho[1]
    while cursor < n-3 and pair_sum > 0:
        last_even = rho[cursor+1]
        pair_sum = last_even + rho[cursor+2]
        if pair_sum >= 0:
            retained[cursor+1:cursor+3] = rho[cursor+1:cursor+3]
        cursor += 2
    end = cursor-2
    retained[end+1] = max(last_even, 0.)
    for odd in range(1, end-1, 2):
        previous = retained[odd-1:odd+1].sum()
        if retained[odd+1:odd+3].sum() > previous:
            retained[odd+1:odd+3] = previous/2
    tau = max(-1 + 2*retained[:end+1].sum() + retained[end+1], 1 / np.log10(m*n))
    return m*n / tau



def diagnose(chains):
    x = np.asarray(chains, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 8 or not np.isfinite(x).all():
        raise ValueError('Need at least two finite chains with >=8 equal-length retained draws')
    s = split(x)
    rank = rank_normalize(s)
    folded = rank_normalize(np.abs(s - np.median(s)))
    qlo, qhi = np.quantile(x, [.05, .95])
    bulk = effective_size(rank)
    tail = min(effective_size(split((x <= qlo).astype(float))), effective_size(split((x <= qhi).astype(float))))
    mean_ess = effective_size(s)
    rhat = max(basic_rhat(rank), basic_rhat(folded))
    if np.ptp(x) == 0:
        rhat = bulk = tail = mean_ess = np.nan
    constant = np.any(np.ptp(x, axis=1) == 0)
    numerical = not np.isfinite([rhat, bulk, tail, mean_ess, x.mean(), x.std(ddof=1)]).all()
    status = ('constant_chain' if constant else 'numerical_failure' if numerical else
              'warning' if rhat > 1.01 or min(bulk, tail) < 100*x.shape[0] else 'no_flag')
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), quantiles=np.quantile(x, [.025, .5, .975]).tolist(),
                rhat=float(rhat), ess_bulk=float(bulk), ess_tail=float(tail), ess_mean=float(mean_ess),
                mcse_mean=float(x.std(ddof=1) / np.sqrt(mean_ess)), status=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.directory.glob('chain_*_vs.txt'), key=lambda p: int(p.stem.split('_')[1]))
    if len(paths) < 2:
        raise ValueError('Need >=2 chain files')
    draws = np.stack([np.loadtxt(p, ndmin=2) for p in paths])
    summaries = [diagnose(draws[:, :, k]) for k in range(draws.shape[2])]
    pooled = draws.reshape(-1, draws.shape[2])
    with np.errstate(invalid='ignore', divide='ignore'):
        correlation = np.corrcoef(pooled, rowvar=False)
    sampler = []
    for path in paths:
        trace_path = path.with_name(path.name.replace('_vs.txt', '_trace.txt'))
        if not trace_path.exists():
            sampler.append(dict(chain=path.name, status='trace_unavailable'))
            continue
        columns = trace_path.read_text().splitlines()[0].lstrip('# ').split()
        trace = np.loadtxt(trace_path, dtype=str, ndmin=2, skiprows=1)
        retained = trace[trace[:, columns.index('warmup')] == '0']
        if len(retained) != draws.shape[1]:
            raise ValueError('Trace and retained draw counts differ')
        sampler.append(dict(chain=path.name, status='recorded',
                            divergences=int(np.sum(retained[:, columns.index('divergent')].astype(int))),
                            invalid_evaluations=int(np.sum(retained[:, columns.index('status')] != 'valid')),
                            mean_acceptance_probability=float(retained[:, columns.index('acceptance_probability')].astype(float).mean())))
        if 'max_depth_reached' in columns:
            sampler[-1].update(
                max_depth_reached=int(retained[:, columns.index('max_depth_reached')].astype(int).sum()),
                maximum_tree_depth=int(retained[:, columns.index('depth')].astype(int).max()),
                mean_tree_depth=float(retained[:, columns.index('depth')].astype(int).mean()))
        if 'evaluations' in columns:
            sampler[-1]['gradient_evaluations'] = int(retained[:, columns.index('evaluations')].astype(int).sum())
    output = dict(sampler=sampler, chains=[p.name for p in paths], draws_per_chain=draws.shape[1], parameters=summaries,
                  correlation=np.atleast_2d(correlation).tolist(),
                  interpretation='Conditional on supplied parameterization. No flag is not proof of convergence or global mixing.')
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(clean(output), f, indent=2, allow_nan=False)
    for k, row in enumerate(summaries):
        print(f"Vs[{k}] Rhat={row['rhat']:.5f} bulk_ESS={row['ess_bulk']:.1f} tail_ESS={row['ess_tail']:.1f} MCSE={row['mcse_mean']:.6g} {row['status']}")


if __name__ == '__main__':
    main()
