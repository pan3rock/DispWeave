"""Reference Gaussian CLI: explicit prior, empirical materials, replay and rejection."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from joint_results import read_config, write_config
from check_qedisp_examples import forward_rows
from benchmark_crust_nuts import sample_separate_chains, digest, audit_posterior

exe, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2])


def run(*args, codes=(0,)):
    p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
    assert p.returncode in codes, (args, p.returncode, p.stdout, p.stderr)
    return p


with tempfile.TemporaryDirectory(prefix='dw-reference-prior-') as temp:
    work = Path(temp)
    d = read_config(root/'examples/inversion/brocher_regularized.toml')
    d['inversion']['regularization'] = dict(type='none', **{'lambda':0.})
    d['inversion']['starts_vs'] = [[1.1, 1.9, 2.9], [1., 2., 3.]]
    generating = work/'generate.toml';write_config(generating, d)
    run('synthetic', '--config', generating, '--output', work/'data')
    for algorithm in ['hmc', 'nuts']:
        cfg = copy.deepcopy(d)
        cfg[algorithm] = dict(prior='reference_gaussian_vs', prior_sigma_vs=[.1,.2,.3],
            warmup=100, draws=40, step_size=.01, target_accept=.95, adapt_step_size=True, seed=8126)
        cfg[algorithm].update(dict(leapfrog_steps=4) if algorithm=='hmc' else dict(max_depth=7))
        cfg[algorithm].update(initial_metric='start_gauss_newton',mode_support='trapped_modes')
        if algorithm=='nuts':cfg[algorithm]['record_progress']=True
        path = work/f'{algorithm}.toml';write_config(path, cfg)
        outputs = [work/f'{algorithm}-{i}' for i in range(2)]
        for out in outputs:
            run(algorithm, '--config', path, '--data', work/'data/observations.txt', '--output', out, codes=(0,5))
            saved = np.loadtxt(out/'prior_parameters.txt')
            np.testing.assert_array_equal(saved[:,1], cfg['model']['vs'])
            np.testing.assert_array_equal(saved[:,2], [.1,.2,.3])
            np.testing.assert_array_equal(saved[:,3:5], np.loadtxt(out/'parameter_bounds.txt')[:,3:5])
            assert 'prior_measure=physical_vs' in (out/'metadata.txt').read_text()
            assert 'initial_metric=start_gauss_newton' in (out/'metadata.txt').read_text()
            for c in range(2):np.linalg.cholesky(np.loadtxt(out/f'chain_{c}_initial_mass.txt'))
            assert not np.allclose(np.loadtxt(out/'chain_0_initial_mass.txt'),np.loadtxt(out/'chain_1_initial_mass.txt'))
            if algorithm=='nuts':
                progress=np.loadtxt(out/'chain_0_progress.txt',dtype=str,skiprows=1)
                assert progress.shape==(140,11) and np.array_equal(progress[:,0].astype(int),np.arange(140))
                np.testing.assert_array_equal(progress[100:,8:].astype(float),np.loadtxt(out/'chain_0_z.txt'))
        for chain in range(2):
            for suffix in ['vs','z','trace']:
                name=f'chain_{chain}_{suffix}.txt'
                assert (outputs[0]/name).read_bytes()==(outputs[1]/name).read_bytes()
        separate=copy.deepcopy(cfg);separate['inversion']['starts_vs']=cfg['inversion']['starts_vs'][1:]
        separate[algorithm]['chain_id_offset']=1
        separate[algorithm]['record_failures']=True
        path=work/f'{algorithm}-separate.toml';write_config(path,separate)
        out=work/f'{algorithm}-separate'
        run(algorithm,'--config',path,'--data',work/'data/observations.txt','--output',out,codes=(0,5))
        for suffix in ['vs','z','trace','mass','metric_updates']:
            assert (out/f'chain_0_{suffix}.txt').read_bytes()==(outputs[0]/f'chain_1_{suffix}.txt').read_bytes()
        if algorithm=='nuts':
            # Recover an interrupted first chain while preserving a completed second chain.
            recovery=work/'recovery';recovery.mkdir()
            recovery_cfg=copy.deepcopy(cfg);recovery_cfg['nuts']['record_failures']=True
            write_config(recovery/'nuts.toml',recovery_cfg)
            for name in ['observations.txt','covariance.txt']:
                shutil.copyfile(outputs[0]/name,recovery/name)
            (recovery/'provenance.json').write_text(json.dumps(dict(binary_sha256=digest(exe),
                files={name:digest(recovery/name) for name in ['nuts.toml','observations.txt','covariance.txt']})))
            for c in range(2):
                directory=recovery/f'chain-{c}';directory.mkdir()
                one=copy.deepcopy(recovery_cfg);one['inversion']['starts_vs']=cfg['inversion']['starts_vs'][c:c+1]
                one['nuts']['chain_id_offset']=c;write_config(directory/'nuts.toml',one)
            (recovery/'chain-0/interruption.txt').write_text('partial output retained')
            shutil.copytree(out,recovery/'chain-1/nuts')
            (recovery/'chain-1/sampler-command.json').write_text(json.dumps(dict(returncode=0)))
            before=(recovery/'chain-1/nuts/chain_0_trace.txt').stat().st_mtime_ns
            sample_separate_chains(recovery,exe,2,resume=True)
            assert (recovery/'chain-0-interrupted-1/interruption.txt').read_text()=='partial output retained'
            assert (recovery/'chain-1/nuts/chain_0_trace.txt').stat().st_mtime_ns==before
            for c in range(2):
                for suffix in ['vs','z','trace','mass','metric_updates']:
                    name=f'chain_{c}_{suffix}.txt'
                    assert (recovery/'nuts'/name).read_bytes()==(outputs[0]/name).read_bytes()
            changed=recovery/'observations.txt';changed.write_text(changed.read_text()+'\n')
            try:sample_separate_chains(recovery,exe,2,resume=True)
            except ValueError as e:assert 'Input changed' in str(e)
            else:raise AssertionError('Changed experiment inputs accepted')
            audit_dir=work/'posterior-audit-check';audit_dir.mkdir()
            shutil.copyfile(outputs[0]/'observations.txt',audit_dir/'observations.txt')
            audit_cfg=copy.deepcopy(cfg);audit_cfg['inversion'].setdefault('fixed_vs_layers',[])
            samples=np.stack([np.loadtxt(outputs[0]/f'chain_{c}_vs.txt') for c in range(2)])
            audits=audit_posterior(audit_dir,exe,audit_cfg,samples)
            assert all(a['passed'] for a in audits)
            assert json.loads((audit_dir/'posterior-audit.json').read_text())==audits
        unstable=copy.deepcopy(separate)
        unstable[algorithm].update(warmup=0,draws=4,step_size=1000.,adapt_step_size=False)
        path=work/f'{algorithm}-unstable.toml';write_config(path,unstable)
        failed=work/f'{algorithm}-unstable'
        run(algorithm,'--config',path,'--data',work/'data/observations.txt','--output',failed,codes=(5,))
        bad=np.loadtxt(failed/'chain_0_failed_evaluations.txt',dtype=str,ndmin=2,skiprows=1)
        assert len(bad)>0 and bad.shape[1]==5 and np.all(bad[:,1]!='valid')
        predicted=work/f'{algorithm}-predicted'
        run('predict', '--run', outputs[0], '--output', predicted, '--draws-per-chain', 2)
        # Recompute the first posterior prediction with independent Brocher polynomials.
        vs=np.exp(np.loadtxt(predicted/'selected_log_vs.txt', ndmin=2)[0])
        vp=.9409+2.0947*vs-.8206*vs**2+.2683*vs**3-.0251*vs**4
        rho=1.6612*vp-.4721*vp**2+.0671*vp**3-.0043*vp**4+.000106*vp**5
        physical=dict(model=dict(depth=cfg['model']['depth'],vs=vs.tolist(),vp=vp.tolist(),rho=rho.tolist()),queries=d['queries'])
        probe=work/f'{algorithm}-physical.toml';write_config(probe,physical)
        run('forward','--config',probe,'--output',work/f'{algorithm}-physical.txt')
        np.testing.assert_allclose(np.loadtxt(predicted/'latent.txt')[0],
            [float(r[3]) for r in forward_rows(work/f'{algorithm}-physical.txt')],atol=2e-10,rtol=0)
        for name, sigma in [('missing',None),('dimension',[.1]),('zero',[.1,0.,.1]),('negative',[.1,-.1,.1])]:
            invalid=copy.deepcopy(cfg)
            if sigma is None:del invalid[algorithm]['prior_sigma_vs']
            else:invalid[algorithm]['prior_sigma_vs']=sigma
            path=work/f'{algorithm}-{name}.toml';write_config(path,invalid)
            run(algorithm,'--config',path,'--data',work/'data/observations.txt','--output',work/f'bad-{algorithm}-{name}',codes=(1,))
        for name in ['penalty','uniform']:
            invalid=copy.deepcopy(cfg)
            if name=='penalty':invalid['inversion']['regularization']=dict(type='reference_damping',**{'lambda':100.})
            else:invalid[algorithm]['prior']='uniform_vs'
            path=work/f'{algorithm}-{name}.toml';write_config(path,invalid)
            p=run(algorithm,'--config',path,'--data',work/'data/observations.txt','--output',work/f'bad-{algorithm}-{name}',codes=(1,))
            assert ('not a sampler prior' if name=='penalty' else 'requires reference_gaussian_vs') in p.stderr
        for field,value in [('mode_support','unknown'),('initial_metric','unknown'),('mass',np.eye(3).tolist())]:
            invalid=copy.deepcopy(cfg);invalid[algorithm][field]=value
            path=work/f'{algorithm}-bad-{field}.toml';write_config(path,invalid)
            run(algorithm,'--config',path,'--data',work/'data/observations.txt','--output',work/f'bad-{algorithm}-{field}',codes=(1,))
print('PASS reference Gaussian HMC/NUTS configuration, retained-state replay and empirical prediction')
