#!/usr/bin/env python3
"""Freeze an independence proposal from separate pilot chains, not posterior weights.

Never combine the pilot and new production samples. This proposal cannot find
modes absent from the pilot; the random-walk kernel remains enabled.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from joint_results import load, read_config, write_config


def prepare(pilot, output, seed, probability=.25, covariance_factor=2.25):
    if not 0 < probability < 1 or not np.isfinite(covariance_factor) or covariance_factor <= 0:
        raise ValueError('Require 0<probability<1 and positive covariance factor')
    _, _, labels, traces, _ = load(pilot)
    # The exact input copy preserves precision and every declared target option.
    doc = read_config(Path(pilot)/'config.toml')
    if not 0 <= seed < 2**64 or seed == doc['sampler']['seed']:
        raise ValueError('Declare a fresh uint64 seed distinct from the pilot')
    if any(code not in ('valid', 'outside_support') for chain in traces for phase in chain.values()
           for code in phase['status_counts']):
        raise ValueError('Pilot contains numerical failures')
    cfg = deepcopy(doc)
    components, sources = [], []
    for c in range(doc['sampler']['chains']):
        path = Path(pilot)/f'chain_{c}_state.txt'
        x = np.loadtxt(path, ndmin=2)
        covariance = np.atleast_2d(np.cov(x, rowvar=False))*covariance_factor
        if not np.isfinite(covariance).all() or np.trace(covariance) <= 0:
            raise ValueError('Cannot fit a constant/invalid pilot chain')
        # Fixed small floor in declared log coordinates, unrelated to the prior.
        covariance += np.eye(len(labels))*1e-8
        np.linalg.cholesky(covariance)
        components.append(dict(mean=x.mean(axis=0).tolist(), covariance=covariance.tolist(), weight=1.))
        sources.append(dict(file=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    cfg['sampler']['seed'] = seed
    cfg['sampler']['independence'] = dict(probability=probability, components=components)
    cfg['proposal_provenance'] = dict(pilot_seed=doc['sampler']['seed'], labels=labels,
        covariance_factor=covariance_factor, diagonal_floor=1e-8, sources=sources,
        interpretation='Equal proposal weights per pilot chain; not posterior mode probabilities. Pilot excluded from new production samples.')
    write_config(output, cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--probability', type=float, default=.25)
    parser.add_argument('--covariance-factor', type=float, default=2.25)
    args = parser.parse_args()
    prepare(args.pilot, args.output, args.seed, args.probability, args.covariance_factor)


if __name__ == '__main__':
    main()
