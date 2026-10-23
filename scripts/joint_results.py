"""Read and diagnose joint CLI results. Python >=3.11 and NumPy only.

No failed predictions, chains, or repeated states are silently removed.
"""
import argparse
import json
import re
from pathlib import Path
import tomllib
import numpy as np
from posterior_diagnostics import diagnose


def write_config(path, document):
    """Small TOML writer for the experiment's plain tables/arrays/scalars."""
    lines = []

    def scalar(value):
        return json.dumps(value, allow_nan=False, ensure_ascii=True)

    def table(value, prefix='', header=None):
        if header:
            lines.extend(['', header])
        for key, item in value.items():
            if not re.fullmatch(r'[A-Za-z0-9_-]+', key):
                raise ValueError('Unsupported TOML key')
            if not isinstance(item, dict) and not (isinstance(item, list) and item and isinstance(item[0], dict)):
                lines.append(f'{key} = {scalar(item)}')
        for key, item in value.items():
            full = f'{prefix}.{key}' if prefix else key
            if isinstance(item, dict):
                table(item, full, f'[{full}]')
            elif isinstance(item, list) and item and isinstance(item[0], dict):
                for row in item:
                    table(row, full, f'[[{full}]]')

    table(document)
    text = '\n'.join(lines)+'\n'
    if tomllib.loads(text) != document:
        raise ValueError('TOML round trip failed')
    with Path(path).open('x') as stream:
        stream.write(text)


def read_config(path):
    with Path(path).open('rb') as stream:
        return tomllib.load(stream)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return None if isinstance(value, float) and not np.isfinite(value) else value


def save_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)


def load(directory):
    directory = Path(directory)
    doc = read_config(directory/'observations.toml')
    sampler = doc['sampler']
    n, chains, draws = len(doc['model']['vs']), sampler['chains'], sampler['draws']
    blocks = doc['observations']
    labels = [f'Vs[{j}]' for j in range(n)]
    labels += [f"scale:{b['name']}" for b in blocks if b.get('noise', {}).get('kind', 'fixed') != 'fixed']
    paths = [directory/f'chain_{c}_state.txt' for c in range(chains)]
    if set(paths) != set(directory.glob('chain_*_state.txt')):
        raise ValueError('Unexpected or missing chains')
    states = np.stack([np.loadtxt(p, ndmin=2) for p in paths])
    if states.shape != (chains, draws, len(labels)) or not np.isfinite(states).all():
        raise ValueError('Invalid retained state dimensions or values')
    samples = np.exp(states)
    if not np.isfinite(samples).all():
        raise ValueError('Physical state overflow')
    traces = []
    for c in range(chains):
        trace = np.loadtxt(directory/f'chain_{c}_trace.txt', dtype=str, ndmin=2, skiprows=1)
        warmup = sampler['warmup']
        if trace.shape != (warmup+draws, 5) or not (trace[:warmup, 0] == '1').all() or not (trace[warmup:, 0] == '0').all():
            raise ValueError('Trace/retained counts disagree')
        retained = trace[warmup:]
        if np.ptp(retained[:, 3].astype(float)) != 0:
            raise ValueError('Production proposal adapted')
        kind_file = directory/f'chain_{c}_proposal_kind.txt'
        kinds = np.zeros(warmup+draws, dtype=int)
        if kind_file.exists():
            kinds = np.loadtxt(kind_file, dtype=int, ndmin=1)
            if kinds.shape != (warmup+draws,) or not np.isin(kinds, [0, 1]).all():
                raise ValueError('Invalid proposal-kind trace')
        elif 'independence' in sampler:
            raise ValueError('Missing independence proposal-kind trace')
        if 'independence' not in sampler and kinds.any():
            raise ValueError('Undeclared independence proposal')
        phases = {}
        for name, rows in [('warmup', trace[:warmup]), ('retained', retained)]:
            codes, counts = np.unique(rows[:, 2], return_counts=True)
            phases[name] = dict(status_counts=dict(zip(codes.tolist(), counts.tolist())),
                               accepted=int(rows[:, 1].astype(int).sum()),
                               transitions=len(rows))
            selected = kinds[:warmup] if name == 'warmup' else kinds[warmup:]
            phases[name]['proposal_kinds'] = {
                kind: dict(proposed=int(np.sum(selected == code)),
                           accepted=int(rows[selected == code, 1].astype(int).sum()))
                for code, kind in [(0, 'random_walk'), (1, 'independence')]}
        traces.append(phases)
    status = [line.split()[:3] for line in (directory/'predictive_status.txt').read_text().splitlines() if not line.startswith('#')]
    ppc = sampler.get('predictive_per_chain', 200)
    expected = [(str(c), str((2*j+1)*draws//(2*ppc))) for c in range(chains) for j in range(ppc)]
    if [tuple(row[:2]) for row in status] != expected or any(row[2] != 'valid' for row in status):
        raise ValueError('Invalid/missing posterior predictions; inspect predictive_status.txt')
    predictions = {}
    for b in blocks:
        name = b['name']
        latent = np.loadtxt(directory/f'prediction_{name}.txt', ndmin=2, skiprows=1)
        replicate = np.loadtxt(directory/f'replicate_{name}.txt', ndmin=2, skiprows=1)
        misfit = np.loadtxt(directory/f'predictive_misfit_{name}.txt', ndmin=2, skiprows=1)
        ids = np.array(expected, dtype=int)
        if latent.shape != (chains*ppc, len(b['values'])+2) or replicate.shape != latent.shape or misfit.shape != (chains*ppc, 5):
            raise ValueError('Predictive dimensions disagree')
        for a in [latent, replicate, misfit]:
            if not np.isfinite(a).all() or not np.array_equal(a[:, :2], ids):
                raise ValueError('Invalid/reordered predictive rows')
        predictions[name] = dict(latent=latent[:, 2:], replicate=replicate[:, 2:], misfit=misfit[:, 2:])
    return doc, samples, labels, traces, predictions


def summarize(directory):
    doc, samples, labels, traces, predictions = load(directory)
    parameters = {label: diagnose(samples[:, :, j]) for j, label in enumerate(labels)}
    with np.errstate(invalid='ignore', divide='ignore'):
        correlation = np.atleast_2d(np.corrcoef(samples.reshape(-1, len(labels)), rowvar=False))
    blocks = {}
    for b in doc['observations']:
        prediction = predictions[b['name']]
        latent = prediction['latent']
        misfit = prediction['misfit']
        blocks[b['name']] = dict(
            observable=b['observable'], unit=b['unit'], count=len(b['values']),
            latent_quantiles=np.quantile(latent, [.025, .5, .975], axis=0).tolist(),
            predictive_quantiles=np.quantile(prediction['replicate'], [.025, .5, .975], axis=0).tolist(),
            rms_at_predictive_median=float(np.sqrt(np.mean((np.median(latent, axis=0)-b['values'])**2))),
            observed_chi_squared_quantiles=np.quantile(misfit[:, 0], [.025, .5, .975]).tolist(),
            replicated_chi_squared_quantiles=np.quantile(misfit[:, 1], [.025, .5, .975]).tolist(),
            predictive_pvalue=float(np.mean(misfit[:, 1] >= misfit[:, 0])),
            noise_scale_quantiles=np.quantile(misfit[:, 2], [.025, .5, .975]).tolist())
    invalid = sum(sum(count for code, count in phase['status_counts'].items() if code not in ('valid', 'outside_support'))
                  for chain in traces for phase in chain.values())
    vp_policy = 'fixed Vp/Vs' if doc['parameters'].get('vp_vs_ratio', 0.) else 'fixed Vp'
    return dict(parameters=parameters, labels=labels, correlation=correlation.tolist(), sampler=traces,
                numerical_proposals_including_warmup=invalid, blocks=blocks,
                interpretation=f'Conditional on fixed interfaces, {vp_policy}, density and declared priors. Credible intervals are not resolution. No diagnostic flag does not prove global mixing.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    save_json(args.output, result)
    for label, row in result['parameters'].items():
        print(f"{label}: Rhat={row['rhat']:.5f} bulk={row['ess_bulk']:.0f} tail={row['ess_tail']:.0f} {row['status']}")


if __name__ == '__main__':
    main()
