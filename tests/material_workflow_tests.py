"""CLI material/penalty contracts, including posterior predictive replay."""
from pathlib import Path
import copy
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from joint_results import read_config, write_config
from check_qedisp_examples import forward_rows

exe, source = Path(sys.argv[1]).resolve(), Path(sys.argv[2])


def run(*args, codes=(0,)):
    p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    assert p.returncode in codes, (args, p.returncode, p.stdout, p.stderr)
    return p


with tempfile.TemporaryDirectory(prefix='dw-properties-') as tmp:
    work = Path(tmp);template = read_config(source/'examples/inversion/synthetic.toml')
    for kind in ['brocher05', 'gardner', 'nearsurface', 'constant_rho']:
        d = copy.deepcopy(template);d['inversion']['vs2model'] = kind
        config = work/f'{kind}.toml';write_config(config, d);out = work/kind
        run('synthetic', '--config', config, '--output', out)
        fits = np.loadtxt(out/'runs.txt', dtype=str, skiprows=1, ndmin=2)
        assert all(r[1]=='1' and float(r[2]) < 1e-12 for r in fits)
        model = np.loadtxt(out/'run_0_model.txt')
        np.testing.assert_allclose(model[:, 3], d['model']['vs'], atol=2e-7, rtol=0)
        assert not np.allclose(model[:, 5], template['model']['rho']), 'Derived density not saved'
        assert f'vs2model={kind}' in (out/'metadata.txt').read_text()
    d = read_config(source/'examples/inversion/brocher_regularized.toml')
    for kind in ['reference_damping', 'perturbation_smooth', 'vs_smooth']:
        d['inversion']['regularization'] = dict(type=kind, **{'lambda': 10.})
        config = work/f'{kind}.toml';write_config(config, d);out = work/kind
        run('synthetic', '--config', config, '--output', out)
        costs = np.loadtxt(out/'costs.txt');np.testing.assert_allclose(costs[:, 1]+costs[:, 2], costs[:, 3], atol=1e-10)
        assert (costs[:, 1:3] > 0).all(), 'Penalty/data decomposition missing'
        m = np.loadtxt(out/'run_0_model.txt');vs = m[:, 3];ref = m[:, 1]
        expected = 5*np.sum((vs-ref)**2) if kind=='reference_damping' else 5*np.sum(np.diff(vs-(ref if kind=='perturbation_smooth' else 0))**2/np.diff(m[:, 0]))
        np.testing.assert_allclose(costs[0, 2], expected, atol=1e-10)
        physical = dict(model=dict(depth=m[:, 0].tolist(), vs=vs.tolist(), vp=m[:, 4].tolist(), rho=m[:, 5].tolist()), queries=d['queries'])
        f = work/f'{kind}-physical.toml';write_config(f, physical)
        run('forward', '--config', f, '--root-accuracy', 'refined', '--output', work/f'{kind}-physical.txt')
        c = np.array([float(r[3]) for r in forward_rows(work/f'{kind}-physical.txt')])
        obs = np.loadtxt(out/'observations.txt');np.testing.assert_allclose(.5*np.sum(((c-obs[:, 1])/obs[:, 3])**2), costs[0, 1], atol=1e-8)
    # An LM penalty must not be silently interpreted as a normalized sampler prior.
    d['hmc'] = dict(prior='uniform_vs', warmup=0, draws=4, leapfrog_steps=3, step_size=.0001, adapt_step_size=False)
    rejected = work/'penalty-hmc.toml';write_config(rejected, d)
    response = run('hmc', '--config', rejected, '--data', out/'observations.txt', '--output', work/'rejected', codes=(1,))
    assert 'not a sampler prior' in response.stderr
    del d['inversion']['regularization']
    d['inversion']['starts_vs'] = [d['model']['vs']]
    cfg = work/'empirical-hmc.toml';write_config(cfg, d)
    chain = work/'hmc';run('hmc', '--config', cfg, '--data', out/'observations.txt', '--output', chain)
    prediction = work/'predictive';run('predict', '--run', chain, '--output', prediction, '--draws-per-chain', 2)
    assert 'vs2model=brocher05' in (prediction/'metadata.txt').read_text()
    samples = np.exp(np.loadtxt(prediction/'selected_log_vs.txt', ndmin=2))
    latent = np.loadtxt(prediction/'latent.txt', ndmin=2)
    for i, vs in enumerate(samples):
        vp = .9409+2.0947*vs-.8206*vs**2+.2683*vs**3-.0251*vs**4
        rho = 1.6612*vp-.4721*vp**2+.0671*vp**3-.0043*vp**4+.000106*vp**5
        physical = dict(model=dict(depth=d['model']['depth'], vs=vs.tolist(), vp=vp.tolist(), rho=rho.tolist()), queries=d['queries'])
        config = work/f'draw-{i}.toml';write_config(config, physical)
        run('forward', '--config', config, '--output', work/f'draw-{i}.txt')
        np.testing.assert_allclose(latent[i], [float(r[3]) for r in forward_rows(work/f'draw-{i}.txt')], atol=2e-10, rtol=0)
print('PASS empirical materials, penalty decomposition and predictive replay')
