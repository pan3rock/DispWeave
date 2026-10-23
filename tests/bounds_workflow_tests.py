"""Independent bound expectations, solver replay and halfspace contracts."""
from pathlib import Path
import copy
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from joint_results import read_config, write_config

exe, source = Path(sys.argv[1]).resolve(), Path(sys.argv[2])


def run(*args, codes=(0,)):
    p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    assert p.returncode in codes, (args, p.returncode, p.stdout, p.stderr)
    return p


with tempfile.TemporaryDirectory(prefix='dw-bounds-') as tmp:
    work = Path(tmp)
    window = read_config(source/'examples/inversion/synthetic.toml')
    inv = window['inversion']
    del inv['lower_vs'], inv['upper_vs']
    inv.update(vs_min=.7, vs_max=3.4, vs_width=1.)
    legacy = copy.deepcopy(window)
    for key in ['vs_min', 'vs_max', 'vs_width']:del legacy['inversion'][key]
    legacy['inversion'].update(lower_vs=[.7,1.5,2.5], upper_vs=[1.5,2.5,3.4])
    configs = {}
    for name, doc in [('window', window), ('legacy', legacy)]:
        configs[name] = work/f'{name}.toml';write_config(configs[name], doc)
        run('synthetic', '--config', configs[name], '--output', work/name)
        b = np.loadtxt(work/name/'parameter_bounds.txt')
        np.testing.assert_array_equal(b[:, 3:5], [[.7,1.5],[1.5,2.5],[2.5,3.4]])
        np.testing.assert_array_equal(b[:, 5:], [[0,0],[0,0],[0,1]])
        assert 'halfspace_vs=free' in (work/name/'metadata.txt').read_text()
        assert 'halfspace_observation_floor=none' in (work/name/'metadata.txt').read_text()
    for filename in ['runs.txt', 'costs.txt', 'run_0_model.txt', 'run_0_trace.txt']:
        assert (work/'window'/filename).read_bytes() == (work/'legacy'/filename).read_bytes()
    assert abs(np.loadtxt(work/'window/run_0_model.txt')[-1,3]-3.) < 1e-7  # started at 2.8, halfspace optimized
    fixed = copy.deepcopy(window);fixed['inversion']['fixed_vs_layers'] = [2]
    for row in fixed['inversion']['starts_vs']:row[2] = 3.
    config = work/'fixed.toml';write_config(config, fixed)
    run('synthetic', '--config', config, '--output', work/'fixed')
    assert np.loadtxt(work/'fixed/parameter_bounds.txt')[-1,5] == 1
    assert 'halfspace_vs=fixed' in (work/'fixed/metadata.txt').read_text()

    def reject(name, doc, message):
        cfg = work/f'bad-{name}.toml';write_config(cfg, doc)
        p = run('synthetic', '--config', cfg, '--output', work/f'bad-{name}', codes=(1,))
        assert message in p.stderr, p.stderr

    for key in ['vs_min', 'vs_max', 'vs_width']:
        doc = copy.deepcopy(window);del doc['inversion'][key]
        reject(key, doc, 'require all of')
    doc = copy.deepcopy(window);doc['inversion']['lower_vs'] = [.7,1.5,2.5]
    reject('mixed', doc, 'not both')
    for key, value in [('vs_min',0.), ('vs_max',.1), ('vs_width',0.)]:
        doc = copy.deepcopy(window);doc['inversion'][key] = value
        reject('value-'+key, doc, 'Require finite')
    doc = copy.deepcopy(window);doc['inversion']['vs_max'] = 2.
    reject('empty', doc, 'intersection at layer 2')
    doc = copy.deepcopy(window);doc['inversion']['vs_max'] = 2.8
    reject('reference', doc, 'Reference Vs outside global bounds at layer 2')
    doc = copy.deepcopy(window);doc['inversion']['vs_width'] = 2.
    reject('elastic', doc, 'fixed-Vp elastic stability at layer 0')
    doc = copy.deepcopy(window);doc['inversion']['starts_vs'] = [[.6,2.,3.]]
    cfg = work/'outside.toml';write_config(cfg, doc)
    run('synthetic', '--config', cfg, '--output', work/'outside', codes=(4,))
    assert 'invalid_initial_bounds' in (work/'outside/runs.txt').read_text()  # not clipped

    # A noisy observed phase above halfspace Vs is allowed as a residual.
    # Its physical prediction remains below halfspace Vs; no data-derived floor.
    single = dict(model=dict(depth=[0.], vs=[2.], vp=[4.], rho=[2.5]),
                  inversion=dict(vs_min=1., vs_max=3., vs_width=1., starts_vs=[[2.]],
                                 sensitivity='implicit', max_iterations=100))
    cfg = work/'no-floor.toml';write_config(cfg, single)
    data = work/'noisy.txt';data.write_text('1.0 2.7 0 0.2\n')
    run('invert', '--config', cfg, '--data', data, '--output', work/'no-floor')
    b = np.loadtxt(work/'no-floor/parameter_bounds.txt', ndmin=2)
    assert b[0,3] == 1.5 and b[0,4] == 2.5
    prediction = [line.split() for line in (work/'no-floor/run_0_prediction.txt').read_text().splitlines() if not line.startswith('#')][0]
    assert prediction[4] == 'valid' and float(prediction[3]) < 2.5

    # Same physical prior and transformed support must produce identical chains
    # for the two equivalent configuration forms, for both prior measures.
    for algorithm in ['hmc', 'nuts']:
        for prior in ['uniform_vs', 'uniform_log_vs']:
            outputs = []
            for name, template in [('window', window), ('legacy', legacy)]:
                doc = copy.deepcopy(template);doc['inversion']['starts_vs'] = [[1.,2.,3.]]
                doc[algorithm] = dict(prior=prior, warmup=0, draws=4, seed=20261023,
                                      step_size=.0001, adapt_step_size=False)
                doc[algorithm].update(dict(leapfrog_steps=2) if algorithm=='hmc' else dict(max_depth=3))
                label = f'{algorithm}-{prior}-{name}'
                cfg = work/f'{label}.toml';write_config(cfg, doc);out = work/label
                run(algorithm, '--config', cfg, '--data', work/'window/observations.txt', '--output', out, codes=(0,5))
                outputs.append(out)
            for name in ['chain_0_vs.txt', 'chain_0_trace.txt', 'parameter_bounds.txt']:
                assert (outputs[0]/name).read_bytes() == (outputs[1]/name).read_bytes()
            for out in outputs:
                run('predict', '--run', out, '--output', work/(out.name+'-predict'), '--draws-per-chain', 2)
            for name in ['latent.txt', 'replicated.txt', 'parameter_bounds.txt']:
                assert (work/(outputs[0].name+'-predict')/name).read_bytes() == (work/(outputs[1].name+'-predict')/name).read_bytes()
print('PASS bounds intersection, validation, halfspace freedom/no observation floor, LM/HMC/NUTS/predictive replay')
