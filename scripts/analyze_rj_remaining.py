#!/usr/bin/env python3
"""Locate remaining PT/RJ discrepancies from saved trajectories only; no sampling."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from joint_results import read_config
from posterior_diagnostics import diagnose
from rj_diagnostics import clean


def runs(values):
    """Retained-temperature runs; edge runs are censored, swaps can end a run."""
    edges = np.r_[0, np.flatnonzero(np.diff(values)) + 1, len(values)]
    return [dict(k=int(values[a]), start=int(a), stop=int(b), length=int(b-a),
                 censored=bool(a == 0 or b == len(values))) for a, b in zip(edges[:-1], edges[1:])]


def describe(x):
    x = np.asarray(x)
    return dict(count=len(x), mean=float(x.mean()),
                quantiles=np.quantile(x, [.05, .5, .95], axis=0).tolist()) if len(x) else None


def mixture_difference(p0, p1, m0, m1):
    """Exact symmetric decomposition of a mean difference, not a causal attribution."""
    weights = float(np.sum((p0-p1)*(m0+m1)/2))
    conditional = float(np.sum((m0-m1)*(p0+p1)/2))
    difference = float(p0 @ m0-p1 @ m1)
    assert abs(weights+conditional-difference) < 1e-12
    return dict(chain0_minus_other_mean=difference, occupancy_component=weights,
                within_k_component=conditional)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.directory.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def read(path):
        data = path.read_bytes()
        hashes[str(path.relative_to(source))] = hashlib.sha256(data).hexdigest()
        return data.decode()

    def rows(path):
        return [np.fromstring(s, sep=' ') for s in read(path).splitlines()
                if s.strip() and not s.startswith('#')]

    read(source/'rj.toml')
    config = read_config(source/'rj.toml')
    settings = config['rjmcmc']
    warmup, draws = settings['warmup'], settings['draws']
    temperatures = settings['tempering']['temperatures']
    prior = config['rj_prior']
    low, high = prior['min_layers'], prior['max_layers']
    cold = [rows(source/f'rj/chain_{c}_models.txt') for c in range(len(config['rj_starts']))]
    assert all(len(chain) == draws for chain in cold)
    layers = np.array([[int(r[1]) for r in chain] for chain in cold])
    # Match the previously flagged grid point exactly, not a fitted threshold.
    depth = np.linspace(0, prior['depth_limit'], 31)[10]
    depth_vs = np.array([[r[3+int(r[1])+np.searchsorted(r[4:3+int(r[1])], depth, side='right')]
                          for r in chain] for chain in cold])
    blocks = np.array_split(np.arange(draws), 5)
    block_report = []
    conditional = []
    residence = []
    for c, chain in enumerate(cold):
        rs = runs(layers[c])
        assert sum(r['length'] for r in rs) == draws
        residence.append(dict(chain=c, k2_runs=describe([r['length'] for r in rs if r['k'] == 2]),
                              longest_k2=sorted([r for r in rs if r['k'] == 2], key=lambda r: -r['length'])[:10],
                              all_k_changes=len(rs)-1))
        for b, idx in enumerate(blocks):
            block_report.append(dict(chain=c, block=b, start=int(idx[0]), stop=int(idx[-1]+1),
                probabilities={str(k): float(np.mean(layers[c, idx] == k)) for k in range(low, high+1)},
                depth_vs_mean=float(depth_vs[c, idx].mean())))
        for k in range(low, high+1):
            indices = np.flatnonzero(layers[c] == k)
            if not len(indices):
                conditional.append(dict(chain=c, k=k, count=0))
                continue
            x = np.array([chain[i][4:] for i in indices])
            # Largest |velocity jump| is just a descriptive main-interface locator.
            major = np.array([r[4+np.argmax(np.abs(np.diff(r[3+k:])))] for r in (chain[i] for i in indices)]) if k > 1 else np.array([])
            conditional.append(dict(chain=c, k=k, count=len(indices),
                coordinates=[dict(name=f'z{j+1}' if j < k-1 else f'Vs{j-k+2}', **describe(x[:, j])) for j in range(2*k-1)],
                major_interface=describe(major), depth_vs=describe(depth_vs[c, indices]),
                log_likelihood=describe([chain[i][2] for i in indices]),
                block_depth_vs=[describe(depth_vs[c, idx[layers[c, idx] == k]]) for idx in blocks]))
    ks = sorted(set(layers.ravel()))
    p0 = np.array([np.mean(layers[0] == k) for k in ks])
    p1 = np.array([np.mean(layers[1:] == k) for k in ks])
    m0 = np.array([depth_vs[0][layers[0] == k].mean() for k in ks])
    m1 = np.array([depth_vs[1:][layers[1:] == k].mean() for k in ks])
    decomposition = mixture_difference(p0, p1, m0, m1)
    prefix = [dict(draws=int(idx[-1]+1), k2=diagnose((layers[:, :idx[-1]+1] == 2).astype(float)),
                   depth_vs=diagnose(depth_vs[:, :idx[-1]+1])) for idx in blocks]

    def insertion(row):
        k = int(row[1])
        return np.maximum(np.diff(np.r_[0., row[4:3+k], prior['depth_limit']])-2*prior['min_spacing'], 0).sum()

    def log_prior(k):
        weights = prior['layer_weights']
        return (math.log(weights[k-low]/sum(weights)) + math.lgamma(k)
                -(k-1)*math.log(prior['depth_limit']-k*prior['min_spacing'])
                -k*math.log(prior['vs_upper']-prior['vs_lower']))

    transitions = []
    cold_events = []
    likelihood_error = 0.
    for c in range(len(cold)):
        folder = source/f'rj/ladder_{c}'
        states = [rows(folder/f'temperature_{t}_models.txt') for t in range(len(temperatures))]
        initial = [rows(folder/f'temperature_{t}_warmup_models.txt')[-1] for t in range(len(temperatures))]
        for a, b in zip(states[0], cold[c]):
            np.testing.assert_array_equal(a, b)
        # Undo swaps to recover each local move's immediate result.
        pre_swap = [list(s) for s in states]
        swaps = rows(folder/'swaps.txt')
        for row in reversed(swaps):
            i = int(row[0])-warmup
            if i >= 0 and row[9]:
                l, h = map(int, row[2:4])
                pre_swap[l][i], pre_swap[h][i] = pre_swap[h][i], pre_swap[l][i]
        for t, temperature in enumerate(temperatures):
            trace = np.array([s.split() for s in read(folder/f'temperature_{t}_local_trace.txt').splitlines() if s and not s.startswith('#')])
            trace = trace[trace[:, 0] == '0']
            assert len(trace) == draws
            before = [initial[t], *states[t][:-1]]
            for i, tr in enumerate(trace):
                assert int(tr[6]) == before[i][1]
                assert int(tr[7]) == pre_swap[t][i][1]
                assert int(tr[10]) == states[t][i][1]
                if tr[5] == '0':
                    np.testing.assert_array_equal(before[i][1:], pre_swap[t][i][1:])
            if t == 0:
                local = np.array([int(r[1]) for r in pre_swap[0]])
                previous = np.array([int(r[1]) for r in before])
                cold_events.append(dict(chain=c,
                    local_2_to_3=int(np.sum((previous == 2) & (local == 3))),
                    local_3_to_2=int(np.sum((previous == 3) & (local == 2))),
                    swap_2_to_3=int(np.sum((local == 2) & (layers[c] == 3))),
                    swap_3_to_2=int(np.sum((local == 3) & (layers[c] == 2))),
                    local_k_changes=int(np.sum(previous != local)),
                    swap_k_changes=int(np.sum(local != layers[c])),
                    net_k_changes=int(np.sum(previous != layers[c]))))
            for move, ka, kb in [('birth', 2, 3), ('death', 3, 2)]:
                indices = np.flatnonzero((trace[:, 1] == move) & (trace[:, 6] == str(ka)))
                tr = trace[indices]
                evaluated = tr[:, 3] == '1'
                valid = tr[:, 4] == 'valid'
                accepted = tr[:, 5] == '1'
                failed = evaluated & ~valid
                assert np.all(~accepted | valid)
                ratios = tr[valid, 8].astype(float)
                delta_ll = []
                if move == 'birth':
                    for i in indices[valid]:
                        h = math.log(insertion(before[i])*(prior['vs_upper']-prior['vs_lower'])/ka)
                        delta = temperature*(float(trace[i, 8])-(log_prior(kb)-log_prior(ka))-h)
                        delta_ll.append(delta)
                        if trace[i, 5] == '1':
                            likelihood_error = max(likelihood_error, abs(delta-(pre_swap[t][i][2]-before[i][2])))
                transitions.append(dict(chain=c, temperature=temperature, move=move, from_k=ka, to_k=kb,
                    attempts=len(tr), evaluated=int(evaluated.sum()), forward_failures=int(failed.sum()),
                    outside_support=int(np.sum(tr[:, 2] == 'outside_support')),
                    unavailable=int(np.sum(tr[:, 2] == 'unavailable')),
                    valid_evaluations=int(valid.sum()), accepted=int(accepted.sum()),
                    valid_rejected=int(np.sum(valid & ~accepted)),
                    log_accept_ratio=describe(ratios),
                    valid_log_ratio_below_minus5=int(np.sum(ratios < -5)),
                    mean_mh_probability=float(np.exp(np.minimum(ratios, 0)).mean()) if len(ratios) else None,
                    birth_delta_raw_log_likelihood=describe(delta_ll),
                    birth_valid_delta_logL_below_minus5=int(np.sum(np.array(delta_ll) < -5)) if move == 'birth' else None))
                assert len(tr) == int(valid.sum()+failed.sum()+np.sum(~evaluated))
    assert likelihood_error < 1e-8
    # All inputs must remain byte-identical; no forwards or sampler are invoked.
    for name, digest in hashes.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    report = dict(source=str(source), target_evaluations=0, additional_samples=0,
        depth=float(depth), blocks=block_report, conditional=conditional, residence=residence,
        prefix_diagnostics=prefix, depth_mean_decomposition=decomposition,
        transitions=transitions, cold_transition_sources=cold_events,
        maximum_birth_likelihood_reconstruction_error=likelihood_error, input_hashes=hashes,
        interpretation='Exploratory diagnosis of existing unaccepted chains. Conditional summaries retain repeats; no Rhat on time-compressed per-K samples. Residence is at fixed temperature, includes swaps, and edge runs are censored. No causal attribution or convergence certification from prefix/subset checks. Rejected death candidates are not retained; their likelihood loss is not inferred from total MH ratio.')
    (args.output/'diagnosis.json').write_text(json.dumps(clean(report), indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = ['#0072B2', '#CC79A7', '#E69F00', '#666666']
    for c, color in enumerate(colors[:len(cold)]):
        label = f'Cold chain {c}'
        axes[0, 0].plot(np.arange(1, draws+1), np.cumsum(layers[c] == 2)/np.arange(1, draws+1), color=color, label=label)
        axes[0, 1].plot(np.arange(1, 6), [np.mean(layers[c, idx] == 2) for idx in blocks], 'o-', color=color)
        lengths = sorted(r['length'] for r in runs(layers[c]) if r['k'] == 2)
        axes[0, 2].step(lengths, np.arange(1, len(lengths)+1)/len(lengths), where='post', color=color)
        for j, k in enumerate([2, 3, 4]):
            values = np.sort(depth_vs[c][layers[c] == k])
            axes[1, j].step(values, np.arange(1, len(values)+1)/len(values), where='post', color=color)
    axes[0, 0].set(title='Cumulative K=2 occupancy', xlabel='Production draw', ylabel='Fraction')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title='K=2 occupancy in five equal blocks', xlabel='Chronological block', ylabel='Fraction')
    axes[0, 2].set(title='K=2 residence (swaps included)', xlabel='Consecutive retained draws', ylabel='Empirical CDF', xscale='log')
    for j, k in enumerate([2, 3, 4]):
        axes[1, j].set(title=f'Vs at {depth:.3f} km, conditional on K={k}', xlabel='Vs (km/s)', ylabel='Empirical CDF')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Remaining RJ/PT discrepancies | existing samples only | posterior NOT accepted')
    fig.savefig(args.output/'diagnosis.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), decomposition=decomposition, cold_transition_sources=cold_events), indent=2))


if __name__ == '__main__':
    main()
