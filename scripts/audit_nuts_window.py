#!/usr/bin/env python3
"""Local, fixed-state diagnostic of the second warmup metric update.

All one-transition probes restart at the same checkpoint; these are NOT
posterior chains. Gauss-Newton curvature is a local proxy, not the exact Hessian.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from joint_results import read_config, write_config, save_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectrum(mass, curvature):
    covariance = np.linalg.solve(mass, np.eye(len(mass)))
    root = np.linalg.cholesky((covariance+covariance.T)/2)
    e = np.linalg.eigvalsh(root.T@curvature@root)
    assert e[0] > 0
    return dict(min_frequency_squared=float(e[0]), max_frequency_squared=float(e[-1]),
                condition=float(e[-1]/e[0]), harmonic_step_limit=float(2/np.sqrt(e[-1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir()
    ref = args.reference.resolve(); pilot = args.pilot.resolve()
    exe = out/'dispweave'; shutil.copy2(args.exe.resolve(), exe)
    shutil.copyfile(Path(__file__), out/Path(__file__).name)
    base = read_config(ref/'nuts.toml')
    assert base['inversion']['vs2model'] == 'brocher05'
    assert base['nuts']['prior'] == 'reference_gaussian_vs' and base['nuts']['mode_support'] == 'trapped_modes'
    assert base['nuts']['initial_metric'] == 'start_gauss_newton'
    initial = np.array(base['model']['vs']); n = len(initial); assert n == 45
    lower = np.log(np.maximum(base['inversion']['vs_min'], initial-base['inversion']['vs_width']/2))
    upper = np.log(np.minimum(base['inversion']['vs_max'], initial+base['inversion']['vs_width']/2))
    inputs = out/'inputs'; inputs.mkdir()
    for name in ['nuts.toml', 'observations.txt', 'covariance.txt']:
        shutil.copyfile(ref/name, inputs/name)
        if name != 'nuts.toml':
            assert digest(ref/name) == digest(pilot/'inputs'/name), 'Pilot observations differ from reference target'
    for c in [0, 1]:
        for name, source in [('progress', pilot/f'multinomial-search-{c}/chain_0_progress.txt'),
                             ('initial_mass', ref/f'nuts/chain_{c}_initial_mass.txt'),
                             ('config', pilot/f'multinomial-search-{c}.toml')]:
            shutil.copyfile(source, inputs/f'{name}-{c}.txt')
    seeds = [17, 2026, 99173, 20261024]
    protocol = dict(checkpoint='state after zero-based transition 149, before metric update',
        covariance_windows=[[75,100],[100,150]], metric_shrinkage=25,
        metric_variants=['before', 'after'], step_variants=['before', 'late'],
        steps='before=progress[149].step; late=median(progress[160:170].step)',
        additional_one_step_probes=['first step after search: progress[150]', 'next step after DA restart: progress[151]'],
        seeds=seeds, draws_per_probe=1, max_depth=8, overshoot_probe_max_depth=1,
        workers=2, timeout_per_probe_seconds=10, target_and_prior='unchanged',
        interpretation='Fixed-state local probes; not posterior draws or ESS measurements',
        binary_sha256=digest(exe), script_sha256=digest(Path(__file__)),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        inputs={p.name:digest(p) for p in inputs.iterdir()})
    save_json(out/'protocol.json', protocol)

    def command(cfg, dest):
        return [str(exe), 'nuts', '--config', str(cfg), '--data', str(inputs/'observations.txt'),
                '--covariance', str(inputs/'covariance.txt'), '--output', str(dest)]

    geometry = []; jobs = []
    for c in [0, 1]:
        rows = np.loadtxt(inputs/f'progress-{c}.txt', dtype=str, skiprows=1)
        assert rows.shape[1] == 8+n and len(rows) >= 170
        assert np.array_equal(rows[:,0].astype(int), np.arange(len(rows)))
        assert np.all(rows[:170,1] == '1'), 'Checkpoint must be within warmup'
        pilot_cfg = read_config(inputs/f'config-{c}.txt')
        assert pilot_cfg['nuts']['metric_shrinkage'] == 25 and pilot_cfg['nuts']['kernel'] == 'multinomial'
        assert pilot_cfg['nuts']['warmup'] == 300 and pilot_cfg['nuts']['metric_adaptation'] == 'dense'
        assert pilot_cfg['model'] == base['model']
        inversion = copy.deepcopy(pilot_cfg['inversion'])
        assert inversion.pop('starts_vs') == [base['inversion']['starts_vs'][c]]
        assert inversion == {k:v for k,v in base['inversion'].items() if k != 'starts_vs'}
        for key in ['prior', 'prior_sigma_vs', 'mode_support', 'initial_metric']:
            assert pilot_cfg['nuts'][key] == base['nuts'][key], 'Pilot target or initial metric differs'
        folder = out/f'chain-{c}'; folder.mkdir()
        z = rows[149,8:].astype(float); np.savetxt(folder/'checkpoint-z.txt', z, fmt='%.17g')
        cfg = copy.deepcopy(base)
        cfg['inversion']['starts_vs'] = [np.exp(lower+(upper-lower)/(1+np.exp(-z))).tolist()]
        cfg['nuts'].update(warmup=0, draws=1, max_depth=1, step_size=1e-6, adapt_step_size=False,
                           metric_adaptation='none', kernel='multinomial', find_initial_step_size=False,
                           record_progress=True, chain_id_offset=c)
        write_config(folder/'local-gn.toml', cfg)
        result = subprocess.run(command(folder/'local-gn.toml', folder/'local-gn'), capture_output=True, text=True, timeout=15)
        (folder/'local-gn.log').write_text(result.stdout+result.stderr)
        assert result.returncode in (0,5), result.stderr
        H = np.loadtxt(folder/'local-gn/chain_0_initial_mass.txt')
        previous = np.linalg.solve(np.loadtxt(inputs/f'initial_mass-{c}.txt'), np.eye(n))
        masses = {}; update_stats = []
        for label, (start, end) in zip(['before','after'], protocol['covariance_windows']):
            x = rows[start:end,8:].astype(float); count = end-start
            covariance = (count*np.cov(x.T)+25*previous+.005*np.eye(n))/(count+30)
            covariance = (covariance+covariance.T)/2
            mass = np.linalg.solve(covariance, np.eye(n)); mass = (mass+mass.T)/2
            masses[label] = mass; previous = covariance
            np.savetxt(folder/f'mass-{label}.txt', mass, fmt='%.17g')
            update_stats.append(dict(label=label, window_samples=count,
                centered_rank=int(np.linalg.matrix_rank(x-x.mean(axis=0))),
                relative_to_local_gauss_newton=spectrum(mass,H)))
        steps = dict(before=float(rows[149,6]), late=float(np.median(rows[160:170,6].astype(float))),
                     searched=float(rows[150,6]), restarted=float(rows[151,6]))
        geometry.append(dict(chain=c,steps=steps,updates=update_stats))
        for metric in ['before','after']:
            for step in ['before','late','searched','restarted']:
                if step in ['searched','restarted'] and metric == 'before':
                    continue
                for seed in seeds:
                    probe = copy.deepcopy(cfg); opt = probe['nuts']; opt.pop('initial_metric')
                    opt.update(mass=masses[metric].tolist(), step_size=steps[step], seed=seed,
                               max_depth=1 if step in ['searched','restarted'] else 8)
                    name = f'{metric}-{step}-{seed}'; path = folder/f'{name}.toml'
                    write_config(path, probe); jobs.append((c, metric, step, seed, path, folder/name))
    save_json(out/'geometry.json', geometry)

    def run(job):
        c, metric, step, seed, path, dest = job; start = time.perf_counter()
        with dest.with_suffix('.log').open('w') as log:
            try:
                result = subprocess.run(command(path,dest), stdout=log, stderr=subprocess.STDOUT, timeout=10)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        record = dict(chain=c,metric=metric,step=step,seed=seed,returncode=code,seconds=time.perf_counter()-start)
        if code in (0,5):
            trace = np.loadtxt(dest/'chain_0_trace.txt',dtype=str,ndmin=2,skiprows=1)
            assert trace.shape == (1,10) and trace[0,0] == '0'
            t=trace[0]
            costs={k:int(v) for k,v in (line.split('=') for line in (dest/'chain_0_cost.txt').read_text().splitlines())}
            assert costs['transition_evaluations'] == int(t[7]) and costs['step_search_evaluations'] == 0
            record.update(complete=True,evaluations=int(t[7]),acceptance=float(t[4]),
                          status=t[6],divergent=int(t[2]),depth=int(t[8]),depth_cap=int(t[9]),
                          max_energy_error=float(t[5]),total_target_evaluations=costs['target_evaluations']+costs['preflight_evaluations'])
        else:
            record['complete']=False
        save_json(dest.with_suffix('.json'),record)
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(run,jobs))
    assert len(records)==48
    groups=[]
    for c in [0,1]:
        for metric,step in [(m,s) for m in ['before','after'] for s in ['before','late']] + [('after','searched'),('after','restarted')]:
            r=[x for x in records if (x['chain'],x['metric'],x['step'])==(c,metric,step)]
            group=dict(chain=c,metric=metric,step=step,complete=all(x['complete'] for x in r))
            if group['complete']:
                group.update(mean_evaluations=float(np.mean([x['evaluations'] for x in r])),
                    mean_acceptance=float(np.mean([x['acceptance'] for x in r])),
                    divergences=sum(x['divergent'] for x in r),depth_caps=sum(x['depth_cap'] for x in r),
                    invalid=sum(x['status'] not in ['valid','outside_support'] for x in r),
                    support_rejections=sum(x['status']=='outside_support' for x in r))
            groups.append(group)
    save_json(out/'summary.json',dict(geometry=geometry,probes=records,groups=groups,posterior_validation=False))
    print(groups,flush=True)


if __name__ == '__main__':
    main()
