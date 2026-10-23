#!/usr/bin/env python3
"""Audit all saved crustal NUTS missing queries without changing the posterior."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from rayleigh_mode_audit import BoundarySystem
from benchmark_crust_depth import brocher, digest
from check_qedisp_examples import forward_rows, run
from joint_results import read_config, write_config, save_json


def table(config):
    m = config['model']
    return np.c_[np.arange(len(m['vs'])), m['depth'], m['rho'], m['vs'], m['vp']]


def cases(replay, conservative):
    result = [dict(name='original-formal-row-74', source=str(replay/'failed-trial.toml'),
                   frequency=.1612, modes=[2], group='original_formal')]
    groups = [('formal', conservative/'failed-trial-audit/summary.json'),
              ('warmup', conservative/'chain-0-failed-trial-audit/summary.json')]
    for group, path in groups:
        records = json.loads(path.read_text())
        if group == 'warmup': records = records['all_events']
        for record in records:
            parent = (f"chain-{record['chain']}-draw-{record['formal_draw']}" if group == 'formal'
                      else f"call-{record['target_call']}")
            source = path.parent/parent/'forward.toml'
            queries = {}
            for row in record['checks']['standard']['invalid_rows']:
                if row['status'] != 'missing_or_unresolved': raise ValueError('Unexpected failure type')
                query = queries.setdefault(row['frequency_hz'], dict(modes=[], rows=[]))
                query['modes'].append(row['mode']); query['rows'].append(row['row'])
            for frequency, query in queries.items():
                result.append(dict(name=f"{group}-{parent}-row-{min(query['rows'])}", source=str(source),
                                   frequency=frequency, modes=sorted(set(query['modes'])), group=group))
    return result


def audit_case(spec, exe, output):
    start = time.monotonic(); dest = output/spec['name']; dest.mkdir()
    shutil.copyfile(spec['source'], dest/'source.toml')
    if digest(dest/'source.toml') != spec['input_sha256']:
        raise ValueError('Source trial changed after the audit plan was saved')
    config = read_config(dest/'source.toml'); model = table(config)
    np.savetxt(dest/'model.txt', model, fmt='%.17g')
    query = dict(model=config['model'], queries=dict(frequency_hz=[spec['frequency']]*6, mode=list(range(6))))
    write_config(dest/'forward.toml', query)
    record = dict(**spec, independent_passed=False, backend_missed_root=False)
    try:
        independent = BoundarySystem(model, spec['frequency']).audit(.25*model[:, 3].min(), model[-1, 3]*(1-1e-10))
        save_json(dest/'independent.json', independent)
        call = run([exe, 'forward', '--config', dest/'forward.toml', '--root-accuracy', 'refined',
                    '--output', dest/'backend.txt'], dest/'backend.log')
        if call['returncode'] not in [0, 2]: raise ValueError('Backend invocation failed')
        rows = forward_rows(dest/'backend.txt')
        roots = [r['phase_km_s'] for r in independent['roots']]
        backend = [float(r[3]) for r in rows if r[4] == 'valid']
        missed = any(mode < len(roots) for mode in spec['modes'])
        agreement = len(backend) == len(roots) and np.allclose(backend, roots, atol=2e-8, rtol=0)
        record.update(independent_passed=True, count=independent['count'], roots=roots,
                      backend_roots=backend, backend_agreement=bool(agreement), backend_missed_root=missed,
                      requested_modes_outside_count=all(mode >= len(roots) for mode in spec['modes']),
                      maximum_root_difference=float(np.max(np.abs(np.array(backend)-roots))) if agreement and roots else None,
                      maximum_relative_singular_value=max((r['relative_singular_value'] for r in independent['roots']), default=0))
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        record['error'] = str(error)
    record['seconds'] = time.monotonic()-start
    save_json(dest/'summary.json', record)
    return record


def continuation(source, output):
    """Changing only halfspace properties is a diagnostic, never a prior edit."""
    output.mkdir(); original = table(read_config(source)); frequency = .1612
    def model_at(vs):
        model = original.copy(); vp, rho, _, _ = brocher(np.array([vs]))
        model[-1, 2:] = [rho[0], vs, vp[0]]; return model
    def boundary(vs): return BoundarySystem(model_at(vs), frequency).sign(vs)
    # The marginal c=Vs_H state is a boundary condition, not a retained trapped root.
    grid = np.linspace(original[-1, 3], original[-1, 3]+.6, 121)
    values = np.array([boundary(v) for v in grid]); brackets = []
    for i in np.flatnonzero(values[:-1]*values[1:] < 0): brackets.append([float(grid[i]), float(grid[i+1])])
    if not brackets: raise ValueError('No marginal boundary zero in declared continuation range')
    lo, hi = brackets[0]; sign = boundary(lo)
    while hi-lo > 2e-12:
        mid = (lo+hi)/2
        if boundary(mid) == sign: lo = mid
        else: hi = mid
    critical = (lo+hi)/2
    result = dict(source=str(source), marginal_brackets=brackets, critical_halfspace_vs=critical,
                  critical_bracket=[lo, hi], frequency_hz=frequency, points=[],
                  interpretation='Halfspace-only Brocher continuation; c=Vs_H marginal state is not a trapped mode. No inference settings are changed.')
    save_json(output/'plan.json', dict(**result, selected_vs=[float(original[-1, 3]), critical-.001, critical+.001, critical+.01]))
    for i, vs in enumerate([float(original[-1, 3]), critical-.001, critical+.001, critical+.01]):
        model = model_at(vs); np.savetxt(output/f'model-{i}.txt', model, fmt='%.17g')
        audit = BoundarySystem(model, frequency).audit(.25*model[:, 3].min(), vs*(1-1e-12))
        save_json(output/f'audit-{i}.json', audit)
        result['points'].append(dict(halfspace_vs=vs, count=audit['count'], roots=[r['phase_km_s'] for r in audit['roots']]))
    # Broader lower endpoint, tighter halfspace margin and different contour heights.
    wider = BoundarySystem(original, frequency).audit(.1*original[:, 3].min(), original[-1, 3]*(1-1e-12), (.015, .0075))
    save_json(output/'expanded-interval.json', wider)
    result['expanded_interval_count'] = wider['count']
    result['cutoff_supported'] = (all(p['count'] == 2 for p in result['points'][:2])
                                  and all(p['count'] == 3 for p in result['points'][2:])
                                  and wider['count'] == 2)
    save_json(output/'summary.json', result)
    return result


def plot_continuations(output, controls):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(controls), figsize=(12, 4.6), squeeze=False, constrained_layout=True)
    for i, (ax, control) in enumerate(zip(axes.flat, controls)):
        points = control['points']; critical = control['critical_halfspace_vs']
        x = np.array([p['halfspace_vs']-critical for p in points])
        counts = np.array([p['count'] for p in points]); order = np.argsort(x)
        ax.plot(x[order], counts[order], 'o:', color=f'C{i}', label='Independent counts at tested models')
        ax.axvline(0, color='black', linestyle='--', linewidth=1, label='Marginal boundary zero')
        ax.axhline(3, color='gray', linewidth=.7, label='Mode 2 requires at least 3 roots')
        ax.scatter(x[0], counts[0], marker='x', s=100, color='red', label='Original failed trial')
        ax.set(xlabel='Halfspace Vs minus critical Vs (km/s)', ylabel='Trapped roots in audited interval',
               title=f"{'Formal' if i == 0 else 'Warmup'} failure; critical Vs = {critical:.9f} km/s",
               yticks=[2, 3], ylim=(1.75, 3.3))
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc='lower right')
    label = 'Numerical cutoff evidence' if all(c['cutoff_supported'] for c in controls) else 'INCOMPLETE numerical audit'
    fig.suptitle(label+' at 0.1612 Hz; varying halfspace properties does not change inference settings')
    for ext in ['png', 'pdf']: fig.savefig(output/f'cutoff-continuation.{ext}', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--conservative', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=[1, 2, 4], default=4)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir()
    specs = cases(args.replay.resolve(), args.conservative.resolve()); exe = args.exe.resolve()
    for spec in specs: spec['input_sha256'] = digest(Path(spec['source']))
    save_json(output/'plan.json', dict(cases=specs, binary_sha256=digest(exe),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        interval='[0.25*min(Vs), Vs_halfspace*(1-1e-10)]', contour_heights=[.02,.01],
        phase_increment_limit_radians=float(np.pi/4), integer_error_limit=1e-8,
        refinement='Two consecutive resolutions and two heights must agree; at most 8192 horizontal segments',
        independent_real_root_tolerance=2e-12, singular_value_limit=1e-9, backend_tolerance=2e-8,
        limitations='Finite-domain, floating-point numerical audit. Never convert missing_or_unresolved to zero likelihood automatically.'))
    (output/'scripts').mkdir()
    shutil.copyfile(Path(__file__), output/'scripts'/Path(__file__).name)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'benchmarks/rayleigh_mode_audit.py', output/'scripts/rayleigh_mode_audit.py')
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(lambda spec: audit_case(spec, exe, output), specs):
            records.append(record)
            if len(records)%10 == 0: print(f'Completed {len(records)}/{len(specs)} independent audits', flush=True)
    save_json(output/'cases.json', records)
    all_agree = all(r['independent_passed'] and r.get('backend_agreement', False)
                    and r.get('requested_modes_outside_count', False) for r in records)
    save_json(output/'summary.json', dict(total=len(records), independently_resolved=sum(r['independent_passed'] for r in records),
        all_saved_missing_queries_supported=all_agree, confirmed_backend_missed_roots=sum(r['backend_missed_root'] for r in records),
        errors=[r['name'] for r in records if not r['independent_passed']]))
    selected = [specs[0], next(s for s in reversed(specs) if s['group']=='warmup' and s['frequency']==.1612)]
    controls = [continuation(Path(s['source']), output/f'continuation-{i}') for i, s in enumerate(selected)]
    save_json(output/'continuations.json', controls)
    if not args.no_plot: plot_continuations(output, controls)
    print(f'All saved missing queries supported={all_agree}; cutoff controls={[c["cutoff_supported"] for c in controls]}', flush=True)
    return 0 if all_agree and all(c['cutoff_supported'] for c in controls) else 4


if __name__ == '__main__': sys.exit(main())
