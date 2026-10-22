"""Read RJ output without padding layer-indexed parameters across dimensions."""
from pathlib import Path
import tomllib
import re
import numpy as np


def load_run(directory):
    directory = Path(directory)
    with (directory / 'config.toml').open('rb') as f:
        config = tomllib.load(f)
    prior = config['rj_prior']
    paths = sorted((p for p in directory.glob('chain_*_models.txt') if re.fullmatch(r'chain_[0-9]+_models\.txt', p.name)), key=lambda p: int(p.stem.split('_')[1]))
    if not paths:
        raise ValueError('No RJ model files')
    chains = []
    for path in paths:
        chain = []
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            row = np.array([float(x) for x in line.split()])
            if len(row) < 5 or not np.isfinite(row).all():
                raise ValueError(f'Malformed RJ row: {path}')
            k = int(row[1])
            if row[0] != len(chain) or row[1] != k or not prior['min_layers'] <= k <= prior['max_layers'] or len(row) != 2*k+3:
                raise ValueError(f'Invalid index/dimension: {path}')
            z, vs = row[4:3+k], row[3+k:]
            if (np.diff(np.r_[0., z, prior['depth_limit']]) < prior['min_spacing']).any():
                raise ValueError('Invalid interface support')
            if (vs < prior['vs_lower']).any() or (vs > prior['vs_upper']).any():
                raise ValueError('Invalid Vs support')
            chain.append(dict(k=k, interfaces=z, vs=vs, log_likelihood=row[2], log_prior=row[3]))
        if not chain:
            raise ValueError('Empty RJ chain')
        chains.append(chain)
    return config, paths, chains


def depth_values(chains, depth):
    return np.array([[row['vs'][np.searchsorted(row['interfaces'], depth, side='right')] for row in chain] for chain in chains])
