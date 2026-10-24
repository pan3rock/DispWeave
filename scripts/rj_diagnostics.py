#!/usr/bin/env python3
"""Cross-dimensional diagnostics and depth-based summaries (Python >=3.11, NumPy).
K is categorical: diagnose its indicators, not integer-valued K alone.
Interface occupancy is a finite-bin probability, not a point probability/density.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from posterior_diagnostics import diagnose
from rj_results import load_run, depth_values


def residence(k, low, high):
    transitions = np.zeros((high-low+1, high-low+1), dtype=int)
    for a, b in zip(k[:-1], k[1:]):
        transitions[a-low, b-low] += 1
    breaks = np.r_[0, np.flatnonzero(k[1:] != k[:-1])+1, len(k)]
    lengths = np.diff(breaks)
    runs = {str(value): lengths[k[breaks[:-1]] == value].tolist() for value in range(low, high+1)}
    phase = trips = 0
    for value in k:
        if value == low:
            if phase == 2:
                trips += 1
            phase = 1
        elif value == high and phase == 1:
            phase = 2
    return dict(transition_counts=transitions.tolist(), changes=int(np.sum(k[1:] != k[:-1])),
                endpoint_roundtrips=trips, endpoints=[low, high],
                residence={key: dict(runs=len(v), mean=float(np.mean(v)) if v else None, maximum=max(v) if v else None) for key, v in runs.items()})


def clean(value):
    if isinstance(value, dict):
        return {key: clean(x) for key, x in value.items()}
    if isinstance(value, list):
        return [clean(x) for x in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def analyze(directory, grid_points=51):
    config, paths, chains = load_run(directory)
    n = len(chains[0])
    if len(chains) < 2 or n < 8 or any(len(c) != n for c in chains):
        raise ValueError('Need >=2 equal-length chains with >=8 retained draws')
    if not 2 <= grid_points <= 1001:
        raise ValueError('Grid points must be between 2 and 1001')
    prior = config['rj_prior']; low, high = prior['min_layers'], prior['max_layers']
    layers = np.array([[r['k'] for r in c] for c in chains])
    probabilities = []
    for k in range(low, high+1):
        indicator = (layers == k).astype(float)
        diagnostic = diagnose(indicator)
        # Constant indicators cannot establish mixing or exact zero probability.
        probabilities.append(dict(k=k, probability=float(indicator.mean()),
                                  per_chain=indicator.mean(axis=1).tolist(), diagnostics=diagnostic))
    depth = np.linspace(0, prior['depth_limit'], grid_points)
    values = depth_values(chains, depth)
    profiles = [dict(depth=float(z), **diagnose(values[:, :, i])) for i, z in enumerate(depth)]
    edges = depth
    occupancy = np.zeros((len(chains), n, len(edges)-1), dtype=np.uint8)
    counts = np.zeros(len(edges)-1)
    for c, chain in enumerate(chains):
        for i, row in enumerate(chain):
            hist = np.histogram(row['interfaces'], edges)[0]
            counts += hist
            occupancy[c, i] = hist > 0
    interfaces = [dict(lower=float(edges[i]), upper=float(edges[i+1]),
                       probability_any=float(occupancy[:, :, i].mean()), expected_count=float(counts[i]/(n*len(chains)))) for i in range(len(edges)-1)]
    sampler = []
    for c, path in enumerate(paths):
        trace_file = path.with_name(path.name.replace('_models', '_trace'))
        lines = trace_file.read_text().splitlines();header = lines[0].lstrip('# ').split()
        rows = [dict(zip(header, row.split())) for row in lines[1:] if row.strip() and not row.startswith('#')]
        rows = [r for r in rows if r['warmup'] == '0']
        if len(rows) != n or [int(r['layers_after']) for r in rows] != layers[c].tolist():
            raise ValueError('Model/trace retained states disagree')
        entry = residence(layers[c], low, high)
        entry.update(chain=path.name, forward_failures=sum(r['evaluated'] == '1' and r['evaluation_status'] == 'forward_failure' for r in rows),
                     numerical_failures=sum(r['evaluated'] == '1' and r['evaluation_status'] == 'numerical_failure' for r in rows),
                     target_evaluations=sum(r['evaluated'] == '1' for r in rows),
                     actions={move: dict(proposals=sum(r['move'] == move for r in rows),
                                         accepted=sum(r['move'] == move and r['accepted'] == '1' for r in rows),
                                         unavailable=sum(r['move'] == move and r['proposal_status'] == 'unavailable' for r in rows)) for move in ('birth', 'death', 'interface', 'velocity', 'velocity_refresh')})
        sampler.append(entry)
    warning = (any(p['diagnostics']['status'] != 'no_flag' for p in probabilities) or
               any(p['status'] != 'no_flag' for p in profiles) or
               any(s['forward_failures'] or s['numerical_failures'] or (high > low and s['changes'] == 0) for s in sampler))
    return clean(dict(status='warning' if warning else 'no_flag', chains=[p.name for p in paths], draws_per_chain=n,
                      layer_probabilities=probabilities, depth_profiles=profiles, interface_bins=interfaces, sampler=sampler,
                      interpretation='Conditional on priors, support, Vp/Vs and density. Finite-bin interface occupancy is not a density. Unvisited K is not certified zero mass. No flag is not proof of global mixing; endpoint trips may be absent when endpoint mass is negligible.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid-points', type=int, default=51)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze(args.directory, args.grid_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print('status:', report['status'])
    for p in report['layer_probabilities']:
        print('K=', p['k'], 'probability=', p['probability'], 'per_chain=', p['per_chain'], 'diagnostics=', p['diagnostics']['status'])


if __name__ == '__main__':
    main()
