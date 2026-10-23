#!/usr/bin/env python3
"""Exercise external QEDispInv examples without modifying or vendoring that repo.

This is a fixed-geometry synthetic recovery test, not a reproduction of
QEDispInv's randomized layering, empirical-property updates or regularization.
Missing observational errors require an explicit illustrative sigma per case.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import numpy as np
from joint_results import read_config, write_config, save_json

CASES = {'nearsurface': ('demo/syn-nearsurface', 'model_data.txt'),
         'crustmantle': ('demo/syn-crustmantle', 'model_data.txt'),
         'lvl-l4': ('examples/lvl-l4', 'model_data.txt'),
         'lvl-l4-m1': ('examples/lvl-l4', 'm1.txt')}


def model_table(path):
    a = np.loadtxt(path, ndmin=2)
    if a.shape[1] != 5 or not np.isfinite(a).all():
        raise ValueError('Model columns: index depth_km density_g_cm3 Vs_km_s Vp_km_s')
    # The crust example repeats its display index; physical depth defines order.
    if a[0, 1] != 0 or (np.diff(a[:, 1]) <= 0).any() or (a[:, 2:4] <= 0).any() or (a[:, 4] <= np.sqrt(4/3)*a[:, 3]).any():
        raise ValueError('Invalid layer depths or elastic properties')
    return dict(depth=a[:, 1].tolist(), vs=a[:, 3].tolist(), vp=a[:, 4].tolist(), rho=a[:, 2].tolist())


def data_table(path, sigma):
    a = np.loadtxt(path, ndmin=2)
    if a.shape[1] not in (3, 4) or not np.isfinite(a).all() or (a[:, :2] <= 0).any():
        raise ValueError('Data columns: frequency_hz phase_km_s zero_based_mode [sigma_km_s]')
    if (a[:, 2] != np.floor(a[:, 2])).any() or (a[:, 2] < 0).any() or (a[:, 2] > 1023).any():
        raise ValueError('Mode labels must be integers in [0,1023]')
    if a.shape[1] == 3:
        if sigma is None or not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('Source has no errors; supply an explicit illustrative sigma')
        a = np.c_[a, np.full(len(a), sigma)]
    if (a[:, 3] <= 0).any():
        raise ValueError('Nonpositive observational error')
    return a


def run(command, log):
    begin = time.perf_counter()
    with log.open('x') as stream:
        result = subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT)
    return dict(command=[str(x) for x in command], returncode=result.returncode,
                seconds=time.perf_counter()-begin)


def forward_rows(path):
    rows = [shlex.split(line) for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]
    return rows


def check_queries(rows, frequencies, modes):
    """Reject dropped/reordered queries before comparing values or derivatives."""
    if len(rows) != len(frequencies) or any(
            int(r[0]) != i or float(r[1]) != f or int(r[2]) != m
            for i, (r, f, m) in enumerate(zip(rows, frequencies, modes))):
        raise ValueError('Lost or reordered forward query rows')


def prepare(source, exe, output, sigmas, cases):
    output.mkdir()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(source), 'status', '--short'], text=True)
    provenance = dict(source=str(source), source_commit=commit, source_status=status,
        dispweave_binary=dict(path=str(exe), sha256=hashlib.sha256(exe.read_bytes()).hexdigest()),
        interpretation='External synthetic cases; supplied model interfaces/Vp/density fixed. File association is audited, not assumed to be ground truth. No random layering, empirical property updates or regularization. Sigma is an illustrative assumption, not a measured uncertainty.', cases={})
    for name in cases:
        folder, model_file = CASES[name]
        original = source/folder;work = output/name;work.mkdir()
        sigma = sigmas['lvl-l4' if name == 'lvl-l4-m1' else name]
        model = model_table(original/model_file);data = data_table(original/'data.txt', sigma)
        provenance['cases'][name] = dict(files={}, rows=len(data), layers=len(model['vs']),
            frequency_range=[float(data[:, 0].min()), float(data[:, 0].max())],
            mode_counts={str(m): int(np.sum(data[:, 2] == m)) for m in np.unique(data[:, 2]).astype(int)},
            assumed_sigma_km_s=sigma, model_file=model_file)
        for filename in ['model_data.txt', 'data.txt', 'config.toml', 'mref.txt', 'm1.txt']:
            path = original/filename
            if path.exists():
                shutil.copyfile(path, work/('source-'+filename))
                provenance['cases'][name]['files'][str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        np.savetxt(work/'observations.txt', data, fmt=['%.17g', '%.17g', '%d', '%.17g'],
                   header='frequency_hz phase_km_s mode sigma_km_s; sigma assumed when absent in source')
        write_config(work/'forward.toml', dict(model=model, queries=dict(frequency_hz=data[:, 0].tolist(), mode=data[:, 2].astype(int).tolist())))
        vs, vp = np.array(model['vs']), np.array(model['vp'])
        lower = .65*vs;upper = np.minimum(1.35*vs, .98*np.sqrt(.75)*vp)
        # Broad synthetic recovery bounds centered on the supplied model;
        # explicitly NOT a blind field-data prior or a posterior model choice.
        starts = [vs*.9, vs*1.1, vs*(1+.08*np.sin(np.arange(len(vs))+1)), vs*(1-.08*np.cos(np.arange(len(vs))+1))]
        write_config(work/'inversion.toml', dict(model=model, inversion=dict(
            lower_vs=lower.tolist(), upper_vs=upper.tolist(), starts_vs=[x.tolist() for x in starts],
            sensitivity='implicit', max_iterations=200, gradient_tolerance=1e-5, initial_damping=.001),
            nuts=dict(prior='uniform_vs', metric_adaptation='dense', warmup=1000, draws=2000,
                      max_depth=10, step_size=.02, target_accept=.95, adapt_step_size=True,
                      seed=83261, mass=(np.eye(len(vs))*100).tolist())))
    binary = source/'bin/forward'
    provenance['reference_binary'] = dict(path=str(binary), sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        note='Existing upstream executable; exact build commit not independently certified. Shared numerical ancestry means agreement is not an independent physics proof.')
    save_json(output/'provenance.json', provenance)


def audit(source, exe, output):
    results = {}
    for name in json.loads((output/'provenance.json').read_text())['cases']:
        folder, model_file = CASES[name]
        work = output/name;data = np.loadtxt(work/'observations.txt', ndmin=2)
        commands = {}
        for method in ['none', 'energy', 'implicit']:
            commands[method] = run([exe, 'forward', '--config', work/'forward.toml', '--sensitivity', method,
                                    '--output', work/f'forward-{method}.txt'], work/f'forward-{method}.log')
        commands['upstream'] = run([source/'bin/forward', '--config', source/folder/'config.toml',
            '--model', source/folder/model_file, '--disp', source/folder/'data.txt',
            '--out', work/'qedisp-forward.txt'], work/'qedisp-forward.log')
        rows = forward_rows(work/'forward-none.txt')
        check_queries(rows, data[:, 0], data[:, 2])
        good = np.array([row[4] == 'valid' for row in rows]);c = np.array([float(row[3]) for row in rows])
        upstream = np.loadtxt(work/'qedisp-forward.txt', ndmin=2) if commands['upstream']['returncode'] == 0 else np.empty((0, 3))
        # Upstream omits unresolved rows; require an exact paired query sequence
        # before declaring a whole-case comparison. Never silently use a subset.
        paired = upstream.shape == data[:, :3].shape and np.array_equal(upstream[:, 2], data[:, 2]) and np.allclose(upstream[:, 0], data[:, 0], atol=5.01e-6, rtol=0)
        case = dict(commands=commands, total_rows=len(rows), invalid_rows=np.flatnonzero(~good).tolist(), upstream_rows=len(upstream),
                    paired_upstream=bool(paired), gradients={})
        for method in ['energy', 'implicit']:
            derivative = forward_rows(work/f'forward-{method}.txt')
            check_queries(derivative, data[:, 0], data[:, 2])
            case['gradients'][method] = dict(unavailable_rows=[i for i, row in enumerate(derivative) if row[7] != 'valid'])
        if good.all():
            case['rms_vs_source_data_km_s'] = float(np.sqrt(np.mean((c-data[:, 1])**2)))
            case['max_vs_source_data_km_s'] = float(np.max(np.abs(c-data[:, 1])))
            if paired:
                case['max_vs_qedisp_forward_km_s'] = float(np.max(np.abs(c-upstream[:, 1])))
        results[name] = case;print(name, case, flush=True)
    save_json(output/'forward-audit.json', results)
    return all(not c['invalid_rows'] and c['paired_upstream']
        and c.get('max_vs_qedisp_forward_km_s', np.inf) <= 1e-7
        and all(not g['unavailable_rows'] for g in c['gradients'].values())
        and all(command['returncode'] == 0 for command in c['commands'].values()) for c in results.values())


def invert(exe, output):
    results = {}
    audit_result = json.loads((output/'forward-audit.json').read_text())
    for name in audit_result:
        work = output/name;case = audit_result[name]
        if case['invalid_rows'] or case['gradients']['implicit']['unavailable_rows']:
            results[name] = dict(skipped='Reference forward/implicit derivative failed; no rows removed')
        else:
            results[name] = run([exe, 'invert', '--config', work/'inversion.toml', '--data', work/'observations.txt',
                                 '--output', work/'lm'], work/'lm.log')
        print(name, results[name], flush=True)
    save_json(output/'lm-commands.json', results)
    return all(c.get('returncode') == 0 for c in results.values())


def sensitivities(exe, output):
    report = {}
    for name in json.loads((output/'forward-audit.json').read_text()):
        work = output/name;config = read_config(work/'forward.toml')
        vs = np.array(config['model']['vs']);n = len(vs)
        energy_rows = forward_rows(work/'forward-energy.txt')
        implicit_rows = forward_rows(work/'forward-implicit.txt')
        frequencies = config['queries']['frequency_hz'];modes = config['queries']['mode']
        for rows in [energy_rows, implicit_rows]:
            check_queries(rows, frequencies, modes)
        if any(r[4] != 'valid' or r[7] != 'valid' for r in energy_rows+implicit_rows):
            report[name] = dict(skipped='Unavailable reference derivative');continue
        # Chain rule: finite differences perturb log(Vs), holding Vp/rho fixed.
        energy = np.array([r[12:12+n] for r in energy_rows], float)*vs
        implicit = np.array([r[12:12+n] for r in implicit_rows], float)*vs
        case = dict(max_energy_implicit_difference=float(np.max(np.abs(energy-implicit))), steps=[])
        with tempfile.TemporaryDirectory(prefix='dw-qedisp-fd-') as temp:
            temp = Path(temp)
            for h in [1e-3, 1e-4, 1e-5]:
                derivative = np.full_like(implicit, np.nan);failed = []
                for j in range(n):
                    values = []
                    for sign in [1, -1]:
                        trial = vs.copy();trial[j] *= np.exp(sign*h);config['model']['vs'] = trial.tolist()
                        path = temp/'config.toml'
                        if path.exists():path.unlink()
                        write_config(path, config)
                        p = subprocess.run([str(exe), 'forward', '--config', str(path), '--output', str(temp/'out.txt')], capture_output=True, text=True)
                        if p.returncode:
                            failed.append(dict(parameter=j, sign=sign, returncode=p.returncode));values.append(None)
                        else:
                            rows = forward_rows(temp/'out.txt')
                            check_queries(rows, frequencies, modes)
                            c = np.array([float(r[3]) for r in rows])
                            if any(r[4] != 'valid' for r in rows) or not np.isfinite(c).all():
                                failed.append(dict(parameter=j, sign=sign, reason='invalid_root'));values.append(None)
                            else:
                                values.append(c)
                    if all(v is not None for v in values):derivative[:, j] = (values[0]-values[1])/(2*h)
                error = np.abs(derivative-implicit)
                scaled = error/(1e-7+1e-3*np.maximum(np.abs(derivative), np.abs(implicit)))
                step = dict(h=h, failed=failed, max_absolute_error=float(error.max()) if not failed else None,
                            max_scaled_error=float(scaled.max()) if not failed else None)
                if not failed:
                    i, j = np.unravel_index(np.argmax(error), error.shape)
                    step['worst_absolute'] = dict(row=int(i), parameter=int(j), frequency_hz=frequencies[i], mode=modes[i],
                        finite_difference=float(derivative[i, j]), implicit=float(implicit[i, j]))
                case['steps'].append(step)
        case['passed'] = not case['steps'][-1]['failed'] and case['steps'][-1]['max_scaled_error'] <= 1
        report[name] = case;print(name, case, flush=True)
    save_json(output/'sensitivity-audit.json', report)
    return all(c.get('passed', False) for c in report.values())


def stalled(exe, output):
    """Audit terminal models without relabeling a stalled attempt as a fit."""
    passed = True
    for name in json.loads((output/'forward-audit.json').read_text()):
        work = output/name
        runs = np.loadtxt(work/'lm/runs.txt', dtype=str, ndmin=2, skiprows=1)
        selected = [int(r[0]) for r in runs if r[-2] == 'stalled']
        if not selected:
            continue
        dest = work/'stalled-audit';dest.mkdir()
        manifest = {}
        for i in selected:
            sub = dest/f'run-{i}';sub.mkdir()
            config = read_config(work/'forward.toml')
            config['model']['vs'] = np.loadtxt(work/f'lm/run_{i}_model.txt', ndmin=2)[:, 3].tolist()
            write_config(sub/'forward.toml', config)
            commands = {kind: run([exe, 'forward', '--config', sub/'forward.toml', '--sensitivity', kind,
                '--output', sub/f'forward-{kind}.txt'], sub/f'forward-{kind}.log') for kind in ['none', 'energy', 'implicit']}
            manifest[f'run-{i}'] = commands
        save_json(dest/'forward-audit.json', manifest)
        passed = sensitivities(exe, dest) and passed
        report = json.loads((dest/'sensitivity-audit.json').read_text())
        upstream = Path(json.loads((output/'provenance.json').read_text())['reference_binary']['path'])
        for run_name, check in report.items():
            if check.get('passed') is False and not check['steps'][-1]['failed']:
                probe_branch(exe, upstream, work/'source-config.toml', dest/run_name, check['steps'][-1])
    return passed


def probe_branch(exe, upstream, source_config, work, step):
    """Compare both root enumerators at the worst central-difference query."""
    spec = step['worst_absolute'];h = step['h'];out = work/'branch-probe';out.mkdir()
    config = read_config(work/'forward.toml');model = config['model']
    vs = np.array(model['vs']);n = len(vs)
    modes = list(range(spec['mode']+2))
    frequencies = [spec['frequency_hz']]*len(modes)
    config['queries'] = dict(frequency_hz=frequencies, mode=modes)
    np.savetxt(out/'queries.txt', np.c_[frequencies, np.ones(len(modes)), modes], fmt=['%.17g', '%.17g', '%d'])
    result = dict(perturbation=spec, log_vs_step=h, trials={})
    for sign in [-1, 0, 1]:
        trial = vs.copy();trial[spec['parameter']] *= np.exp(sign*h);model['vs'] = trial.tolist()
        label = str(sign);write_config(out/f'{label}.toml', config)
        np.savetxt(out/f'{label}-model.txt', np.c_[np.arange(n), model['depth'], model['rho'], trial, model['vp']], fmt='%.17g')
        commands = dict(dispweave=run([exe, 'forward', '--config', out/f'{label}.toml', '--output', out/f'{label}-dw.txt'], out/f'{label}-dw.log'),
            qedisp=run([upstream, '--config', source_config, '--model', out/f'{label}-model.txt', '--disp', out/'queries.txt',
                '--out', out/f'{label}-qe.txt'], out/f'{label}-qe.log'))
        rows = forward_rows(out/f'{label}-dw.txt');check_queries(rows, frequencies, modes)
        qedisp = np.loadtxt(out/f'{label}-qe.txt', ndmin=2) if commands['qedisp']['returncode'] == 0 else np.empty((0, 3))
        result['trials'][label] = dict(commands=commands,
            dispweave=[dict(mode=int(r[2]), phase=float(r[3]), status=r[4]) for r in rows],
            qedisp=qedisp.tolist(), expected_queries=len(modes), returned_upstream_queries=len(qedisp))
    save_json(out/'comparison.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--exe', type=Path, default=Path('build/dispweave'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['forward', 'sensitivity', 'lm', 'stalled'], required=True)
    parser.add_argument('--case', nargs='+', choices=list(CASES), help='Cases for a new forward preparation; default all four model/data pairings')
    parser.add_argument('--sigma', type=float, nargs=3, metavar=('NEARSURFACE', 'CRUSTMANTLE', 'LVL4'),
                        help='Explicit assumed standard deviations in km/s; required for forward preparation')
    args = parser.parse_args()
    if args.stage == 'forward':
        if args.sigma is None:
            parser.error('Examples lack errors; explicitly provide three illustrative sigma values')
        prepare(args.source.resolve(), args.exe.resolve(), args.output, dict(zip(['nearsurface','crustmantle','lvl-l4'], args.sigma)), args.case or list(CASES))
        raise SystemExit(0 if audit(args.source.resolve(), args.exe.resolve(), args.output) else 2)
    elif args.stage == 'sensitivity':
        raise SystemExit(0 if sensitivities(args.exe.resolve(), args.output) else 2)
    elif args.stage == 'stalled':
        raise SystemExit(0 if stalled(args.exe.resolve(), args.output) else 2)
    else:
        raise SystemExit(0 if invert(args.exe.resolve(), args.output) else 4)


if __name__ == '__main__':
    main()
