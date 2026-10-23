#!/usr/bin/env python3
"""Controlled crustal LM comparison: free tail, fixed halfspace, merged tail.

Input is a prepared check_qedisp_examples.py crustmantle case directory. No
external repository is modified. All attempts, including failures, are retained.
Python >=3.11 + NumPy; plotting is a separate optional Matplotlib stage.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from check_qedisp_examples import run, forward_rows, check_queries
from joint_results import read_config, write_config, save_json


def variants(document, merge_depth):
    model = document['model'];inv = document['inversion'];n = len(model['vs'])
    if inv.get('fixed_vs_layers') or inv.get('vp_vs_ratio', 0):
        raise ValueError('Control source must have free Vs and fixed Vp')
    indices = [i for i, z in enumerate(model['depth']) if z == merge_depth]
    if len(indices) != 1 or indices[0] == 0:
        raise ValueError('Merge depth must be an existing non-surface interface')
    cut = indices[0]+1
    for key in ['vs', 'vp', 'rho']:
        if any(v != model[key][cut-1] for v in model[key][cut-1:]):
            raise ValueError('Cannot merge a nonuniform reference tail')
    result = {}
    for name, count, fixed in [('A_free', n, []), ('B_fixed_halfspace', n, [n-1]),
                                ('C_merged_tail', cut, [cut-1])]:
        d = dict(model={k: v[:count] for k, v in model.items()}, inversion=copy.deepcopy(inv))
        d['inversion']['fixed_vs_layers'] = fixed
        for key in ['lower_vs', 'upper_vs']:
            d['inversion'][key] = inv[key][:count]
        d['inversion']['starts_vs'] = [v[:count] for v in inv['starts_vs']]
        for start in d['inversion']['starts_vs']:
            for layer in fixed:
                start[layer] = d['model']['vs'][layer]
        result[name] = d
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')]


def forward(exe, config, output, sensitivity='none'):
    command = [str(exe), 'forward', '--config', str(config), '--output', str(output),
               '--root-accuracy', 'refined', '--sensitivity', sensitivity]
    p = subprocess.run(command, capture_output=True, text=True)
    if p.returncode not in [0, 2, 3]:
        raise RuntimeError(f'Forward command failed: {p.stderr}')
    return forward_rows(output)


def audit_terminal(exe, config, data, work, index):
    """Local FD probes are diagnostics, never a root-completeness certificate."""
    dest = work/f'audit-{index}';dest.mkdir()
    table = np.loadtxt(work/f'lm/run_{index}_model.txt', ndmin=2)
    model = copy.deepcopy(config['model']);model['vs'] = table[:, 3].tolist()
    vs = np.array(model['vs']);n = len(vs)
    query = dict(frequency_hz=data[:, 0].tolist(), mode=data[:, 2].astype(int).tolist())
    doc = dict(model=model, queries=query);write_config(dest/'forward.toml', doc)
    base = forward(exe, dest/'forward.toml', dest/'forward.txt', 'implicit')
    check_queries(base, data[:, 0], data[:, 2])
    invalid = [i for i, r in enumerate(base) if r[4] != 'valid']
    bad_gradient = [i for i, r in enumerate(base) if r[7] != 'valid']
    result = dict(invalid_rows=invalid, unavailable_gradients=bad_gradient, steps=[])
    if invalid or bad_gradient:
        save_json(dest/'audit.json', result);return result
    c = np.array([float(r[3]) for r in base])
    jac = np.array([r[12:12+n] for r in base], float)*vs
    free = [i for i in range(n) if i not in config['inversion']['fixed_vs_layers']]
    result['rms_km_s'] = float(np.sqrt(np.mean((c-data[:, 1])**2)))
    # Preserve query order, full predictions, statuses and every FD column.
    with tempfile.TemporaryDirectory(prefix='dw-depth-fd-') as temp:
        temp = Path(temp)
        def evaluate(layer, h):
            values = [];statuses = []
            for sign in [-1, 1]:
                trial = vs.copy();trial[layer] *= np.exp(sign*h)
                doc['model']['vs'] = trial.tolist()
                path = temp/'input.toml'
                if path.exists():path.unlink()
                write_config(path, doc)
                r = forward(exe, path, temp/'output.txt')
                check_queries(r, data[:, 0], data[:, 2])
                values.append(np.array([float(x[3]) if x[4] == 'valid' else np.nan for x in r]))
                statuses.append([x[4] for x in r])
            return values, statuses
        for h in [1e-5, 1e-6]:
            predictions = [];status = []
            for j in free:
                values, flags = evaluate(j, h);predictions.append(values);status.append(flags)
            pred = np.array(predictions)
            fd = ((pred[:, 1]-pred[:, 0])/(2*h)).T
            error = np.abs(fd-jac[:, free])
            scaled = error/(2e-6+1e-3*np.maximum(np.abs(fd), np.abs(jac[:, free])))
            valid = np.isfinite(fd)
            where = np.unravel_index(np.nanargmax(error), error.shape) if valid.any() else None
            step = dict(h=h, unresolved_fd_entries=int((~valid).sum()),
                        max_scaled_error=float(np.nanmax(scaled)) if valid.any() else None)
            if where is not None:
                i, k = where;j = free[k]
                step['worst'] = dict(row=int(i), layer=j, frequency_hz=float(data[i, 0]), mode=int(data[i, 2]),
                    fd=float(fd[i, k]), implicit=float(jac[i, j]), phases=pred[k, :, i].tolist(),
                    absolute_error=float(error[i, k]))
            np.savez_compressed(dest/f'fd-{h}.npz', free_layers=free, steps=h, predictions=pred,
                                statuses=np.array(status), fd=fd, implicit=jac[:, free])
            result['steps'].append(step)
        last = result['steps'][-1]
        result['fd_passed_at_1e_6'] = last['unresolved_fd_entries'] == 0 and last['max_scaled_error'] <= 1
        if not result['fd_passed_at_1e_6'] and 'worst' in last:
            spec = last['worst'];i, j = spec['row'], spec['layer']
            result['worst_smaller_steps'] = []
            for h in [1e-7, 1e-8]:
                values, flags = evaluate(j, h)
                fd = (values[1][i]-values[0][i])/(2*h)
                result['worst_smaller_steps'].append(dict(h=h, fd=float(fd), implicit=float(jac[i, j]),
                    phases=[float(v[i]) for v in values], statuses=[v[i] for v in flags]))
            # Store a reproducible local branch probe, including extra modes.
            branch = dest/'branch';branch.mkdir()
            doc['queries'] = dict(frequency_hz=[spec['frequency_hz']]*(spec['mode']+3), mode=list(range(spec['mode']+3)))
            for sign in [-1, 0, 1]:
                trial = vs.copy();trial[j] *= np.exp(sign*1e-6);doc['model']['vs'] = trial.tolist()
                path = branch/f'{sign}.toml';write_config(path, doc)
                np.savetxt(branch/f'{sign}-model.txt', np.c_[np.arange(n), model['depth'], model['rho'], trial, model['vp']], fmt='%.17g')
                forward(exe, path, branch/f'{sign}.txt')
    save_json(dest/'audit.json', result)
    return result


def execute(case, exe, output, merge_depth):
    document = read_config(case/'inversion.toml')
    configs = variants(document, merge_depth)
    data = np.loadtxt(case/'observations.txt', ndmin=2)
    if data.shape[1] != 4 or (data[:, 3] <= 0).any():
        raise ValueError('Need paired observations and positive explicit sigma')
    output.mkdir()
    provenance = dict(source=str(case), binary=str(exe), binary_sha256=digest(exe), merge_depth_km=merge_depth,
        source_hashes={name: digest(case/name) for name in ['inversion.toml', 'observations.txt']},
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        working_diff_sha256=hashlib.sha256(subprocess.check_output(['git', 'diff'])).hexdigest(),
        interpretation='Controlled fixed-geometry LM. Same data/errors, common free initial values/bounds and iteration budget. Fixed values from the supplied reference, not estimated. No regularization or adaptive layering. FD perturbations may extend just outside inversion bounds to test the physical map.')
    save_json(output/'provenance.json', provenance)
    summary = {}
    for name, config in configs.items():
        work = output/name;work.mkdir();write_config(work/'inversion.toml', config)
        shutil.copyfile(case/'observations.txt', work/'observations.txt')
        record = run([exe, 'invert', '--config', work/'inversion.toml', '--data', work/'observations.txt',
                      '--output', work/'lm'], work/'lm.log')
        if record['returncode'] not in [0, 4]:
            raise RuntimeError(f'{name} invocation failed; see {work}/lm.log')
        summary[name] = dict(command=record, layers=len(config['model']['vs']),
            free_parameters=len(config['model']['vs'])-len(config['inversion']['fixed_vs_layers']), runs=[])
        for row in rows(work/'lm/runs.txt'):
            i = int(row[0]);trace = rows(work/f'lm/run_{i}_trace.txt')
            flags = [dict(iteration=int(t[0]), step=float(t[10]), cost=float(t[1]), trial_cost=float(t[2]))
                     for t in trace if float(t[10]) < 1e-8 and abs(float(t[2])-float(t[1])) > 1e-4]
            print(name, 'run', i, row[-2:], 'auditing', flush=True)
            audit = audit_terminal(exe, config, data, work, i)
            final = np.loadtxt(work/f'lm/run_{i}_model.txt', ndmin=2)
            for j in config['inversion']['fixed_vs_layers']:
                if not np.isclose(final[j, 3], config['model']['vs'][j], rtol=2e-15, atol=0):
                    raise RuntimeError('Fixed parameter moved')
            weights = np.diff(np.r_[final[final[:, 0] < merge_depth, 0], merge_depth])
            delta = final[:len(weights), 3]-final[:len(weights), 1]
            summary[name]['runs'].append(dict(run=i, converged=row[1]=='1', cost=float(row[2]),
                evaluations=int(row[3]), accepted=int(row[4]), rejected=int(row[5]),
                termination=row[-2], reason=row[-1], projected_gradient=float(row[9]),
                tiny_step_cost_jump_flags=flags, audit=audit,
                vs_rms_above_merge_depth=float(np.sqrt(np.dot(weights, delta**2)/merge_depth))))
            print(name, i, 'RMS', audit.get('rms_km_s'), 'FD', audit.get('fd_passed_at_1e_6'), flush=True)
        save_json(work/'summary.json', summary[name])
    if (case/'lm/runs.txt').exists():
        provenance = dict(source_runs=str(case/'lm/runs.txt'),
            baseline_runs_identical=(case/'lm/runs.txt').read_bytes()==(output/'A_free/lm/runs.txt').read_bytes())
        save_json(output/'baseline-replay.json', provenance)
    save_json(output/'summary.json', summary)


def plot(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = json.loads((output/'summary.json').read_text())
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for row, (name, report) in enumerate(summary.items()):
        work = output/name;config = read_config(work/'inversion.toml');model = config['model']
        data = np.loadtxt(work/'observations.txt', ndmin=2)
        z = np.r_[model['depth'], 350.]
        axes[row, 0].stairs(model['vs'], z, orientation='horizontal', baseline=None, color='black', label='Supplied reference')
        for run in report['runs']:
            i = run['run'];final = np.loadtxt(work/f'lm/run_{i}_model.txt', ndmin=2)
            axes[row, 0].stairs(final[:, 3], z, orientation='horizontal', baseline=None,
                color=f'C{i}', linestyle='-' if run['converged'] else '--', label=f"{i}: {run['termination']}")
            prediction = forward_rows(work/f'audit-{i}/forward.txt')
            c = np.array([float(r[3]) if r[4]=='valid' else np.nan for r in prediction])
            for mode in np.unique(data[:, 2]):
                idx = np.flatnonzero(data[:, 2]==mode);idx = idx[np.argsort(data[idx, 0])]
                axes[row, 1].plot(data[idx, 0], (c[idx]-data[idx, 1])/data[idx, 3], '.', color=f'C{i}', alpha=.6)
        axes[row, 0].set(xlabel='Vs (km/s)', ylabel='Depth (km)', ylim=(350, 0),
                         title=f"{name}: {report['free_parameters']} free Vs")
        axes[row, 0].axhline(100, color='gray', linewidth=.7);axes[row, 0].legend(fontsize=8)
        axes[row, 1].axhline(0, color='gray');axes[row, 1].set(xlabel='Frequency (Hz)', ylabel='Residual / assumed sigma',
            title='All modes; colors match runs; invalid predictions remain gaps')
        for ax in axes[row]:ax.grid(alpha=.2)
    fig.suptitle('Crustal depth controls — dashed models are unconverged attempts')
    fig.savefig(output/'comparison.png', dpi=160);fig.savefig(output/'comparison.pdf');plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--exe', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--merge-depth', type=float, default=100.)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output)
    else:
        if args.case is None or args.exe is None:parser.error('--case and --exe are required')
        execute(args.case.resolve(), args.exe.resolve(), args.output, args.merge_depth)
    print(args.output)


if __name__ == '__main__':
    main()
