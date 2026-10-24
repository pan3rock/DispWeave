"""CLI data/state contract checks, separate from distribution validation."""
from pathlib import Path
import math
import subprocess
import sys
import tempfile
exe = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2])

def run(*args, codes=(0,)):
    result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    assert result.returncode in codes, (result.returncode, result.stdout, result.stderr)
    return result

with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    cfg = work / 'input.toml'
    content = (root / 'examples/rjmcmc/synthetic.toml').read_text().replace('warmup = 10000', 'warmup = 60').replace('draws = 50000', 'draws = 150')
    cfg.write_text(content)
    data = work / 'data'
    run('synthetic', '--config', cfg, '--output', data, codes=(0, 4))
    def sample(dest, config=cfg, observations=data / 'observations.txt', codes=(0, 5)):
        return run('rjmcmc', '--config', config, '--data', observations, '--covariance', data / 'covariance.txt', '--output', dest, codes=codes)
    a, b = work / 'a', work / 'b'
    sample(a); sample(b)
    for chain in range(3):
        for suffix in ('models', 'trace'):
            name = f'chain_{chain}_{suffix}.txt'
            assert (a / name).read_bytes() == (b / name).read_bytes(), 'Seed replay'
        rows = [list(map(float, r.split())) for r in (a / f'chain_{chain}_models.txt').read_text().splitlines() if not r.startswith('#')]
        assert len(rows) == 150
        for index, row in enumerate(rows):
            k = int(row[1])
            assert row[0] == index and row[1] == k and 1 <= k <= 4
            assert len(row) == 2*k+3 and all(math.isfinite(x) for x in row)
            interfaces, vs = row[4:4+k-1], row[4+k-1:]
            depths = [0, *interfaces, 5]
            assert all(y-x >= .3 for x, y in zip(depths[:-1], depths[1:]))
            assert all(.8 <= x <= 2.7 for x in vs)
        trace = [r.split() for r in (a / f'chain_{chain}_trace.txt').read_text().splitlines() if not r.startswith('#')]
        assert len(trace) == 210 and [int(r[0]) for r in trace] == [1]*60 + [0]*150
        assert all(r[1] in ('birth', 'death', 'interface', 'velocity') for r in trace)
        assert all(r[4] == 'not_evaluated' for r in trace if r[3] == '0')
        assert [int(r[7]) for r in trace[60:]] == [int(r[1]) for r in rows]
        failures = [r.split() for r in (a / f'chain_{chain}_failed_models.txt').read_text().splitlines() if not r.startswith('#')]
        expected = {int(r[9]): r[4] for r in trace if r[3] == '1' and r[4] != 'valid'}
        assert {int(r[0]): r[1] for r in failures} == expected
    for dest in ('predict-a', 'predict-b'):
        run('rj-predict', '--run', a, '--output', work / dest, '--draws-per-chain', 8, '--seed', 17)
    for name in ('latent.txt', 'replicated.txt', 'draws.txt', 'selected_models.txt'):
        assert (work / 'predict-a' / name).read_bytes() == (work / 'predict-b' / name).read_bytes()
    selected = [r.split() for r in (work / 'predict-a/draws.txt').read_text().splitlines() if not r.startswith('#')]
    assert [(int(r[1]), int(r[2])) for r in selected] == [(c, j*150//8) for c in range(3) for j in range(8)]
    run('rj-predict', '--run', a, '--output', work / 'predict-a', codes=(1,))
    run('rj-predict', '--run', a, '--output', work / 'too-many', '--draws-per-chain', 151, codes=(1,))
    for name in ('observations.txt', 'covariance.txt'):
        assert (a / name).read_bytes() == (data / name).read_bytes()
    assert 'posterior=transdimensional' in (a / 'metadata.txt').read_text()
    sample(a, codes=(1,))
    invalid = work / 'bad.toml'
    invalid.write_text(content.replace('vp_vs_ratio = 1.8', 'vp_vs_ratio = 1.0'))
    sample(work / 'bad', config=invalid, codes=(1,))
    missing = work / 'missing.txt'
    lines = []
    for line in (data / 'observations.txt').read_text().splitlines():
        if line.startswith('#'): lines.append(line)
        else:
            row = line.split(); row[2] = '100'; lines.append(' '.join(row))
    missing.write_text('\n'.join(lines)+'\n')
    sample(work / 'missing', observations=missing, codes=(1,))
    ptconfig = work / 'pt.toml'
    ptconfig.write_text(content + '\n[rjmcmc.tempering]\ntemperatures = [1.0, 2.0, 4.0, 8.0]\nswap_every = 10\n')
    sample(work / 'pt', config=ptconfig)
    sample(work / 'pt-replay', config=ptconfig)
    for file in (work / 'pt').rglob('*.txt'):
        if file.name != 'metadata.txt':
            assert file.read_bytes() == (work / 'pt-replay' / file.relative_to(work / 'pt')).read_bytes()
    for c in range(3):
        assert (work / f'pt/chain_{c}_models.txt').read_bytes() == (work / f'pt/ladder_{c}/temperature_0_models.txt').read_bytes()
        assert not (work / f'pt/chain_{c}_trace.txt').exists(), 'Do not mislabel local traces as cold posterior transitions'
    run('rj-predict', '--run', work / 'pt', '--output', work / 'pt-predict', '--draws-per-chain', 2, codes=(0, 5))
    selected = [r.split() for r in (work / 'pt-predict/draws.txt').read_text().splitlines() if not r.startswith('#')]
    assert [(int(r[1]), int(r[2])) for r in selected] == [(c, j*150//2) for c in range(3) for j in range(2)]
    ptconfig.write_text(ptconfig.read_text()+'save_warmup = true\n')
    sample(work / 'pt-recorded', config=ptconfig)
    for file in (work / 'pt').rglob('*.txt'):
        if file.name != 'metadata.txt':
            assert file.read_bytes() == (work / 'pt-recorded' / file.relative_to(work / 'pt')).read_bytes()
    ptconfig.write_text(ptconfig.read_text()+'cold_start_steps = 10\n')
    sample(work / 'pt-initialized', config=ptconfig)
    for c in range(3):
        for t in range(4):
            file=work / f'pt-initialized/ladder_{c}/temperature_{t}_warmup_models.txt'
            rows=[r.split() for r in file.read_text().splitlines() if not r.startswith('#')]
            assert [int(r[0]) for r in rows]==list(range(-1,60))
        swaps=[r.split() for r in (work / f'pt-initialized/ladder_{c}/swaps.txt').read_text().splitlines() if not r.startswith('#')]
        assert min(int(r[0]) for r in swaps)==19
    refresh_config=work / 'refresh.toml'
    refresh_config.write_text(content.replace('[rjmcmc]', '[rjmcmc]\nvelocity_refresh_probability = 1.0'))
    sample(work / 'refresh', config=refresh_config)
    for c in range(3):
        trace=[r.split() for r in (work / f'refresh/chain_{c}_trace.txt').read_text().splitlines() if not r.startswith('#')]
        assert any(r[1]=='velocity_refresh' for r in trace)
        assert all(r[1]!='velocity' for r in trace)
    assert 'velocity_refresh_probability=1' in (work / 'refresh/metadata.txt').read_text()
    corr=content.replace('[rjmcmc]', '[rjmcmc]\ncorrelated_probability = 0.4')
    for k in range(1,5):
        dim=2*k-1;mean=[5*(j+1)/k for j in range(k-1)]+[1.7]*k
        cov=[[1.0 if i==j else 0.0 for j in range(dim)] for i in range(dim)]
        corr+=f'\n[[rj_correlated_models]]\nlayers = {k}\n[[rj_correlated_models.components]]\nmean = {mean}\ncovariance = {cov}\n'
    corrfile=work / 'corr.toml';corrfile.write_text(corr)
    sample(work / 'corr', config=corrfile)
    assert any(' correlated ' in f.read_text() for f in (work / 'corr').glob('*_trace.txt'))
    badcorr=work / 'corr-bad.toml';badcorr.write_text(corr.replace('layers = 4', 'layers = 3'))
    sample(work / 'corr-bad', config=badcorr, codes=(1,))
    (a / 'observations.txt').write_bytes(missing.read_bytes())
    run('rj-predict', '--run', a, '--output', work / 'failed-predict', '--draws-per-chain', 2, codes=(5,))
    failed = [r.split() for r in (work / 'failed-predict/draws.txt').read_text().splitlines() if not r.startswith('#')]
    assert len(failed) == 6 and all(r[3] == 'forward_failure' for r in failed)
print('RJ workflow replay, ragged states, diagnostics, covariance and failures passed')
