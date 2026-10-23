#!/usr/bin/env python3
"""2x2 fixed/Brocher05 and unregularized/regularized crustal LM control.

Use an existing prepared case (e.g. the C depth control); keep geometry, fixed
layers, Vs starts/bounds, observations/errors and iteration budget identical.
Lambda and type must be supplied explicitly; no automatic selection is done.
"""
import argparse
import copy
from pathlib import Path
import shutil
import numpy as np
from benchmark_crust_depth import audit_terminal, digest, rows, plot
from check_qedisp_examples import run
from joint_results import read_config, write_config, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regularization', choices=['reference_damping', 'perturbation_smooth', 'vs_smooth'], required=True)
    parser.add_argument('--lambda', dest='weight', type=float, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.weight) or args.weight <= 0:parser.error('Explicit positive finite lambda required')
    case=args.case.resolve();exe=args.exe.resolve();output=args.output
    source=read_config(case/'inversion.toml');data=np.loadtxt(case/'observations.txt', ndmin=2)
    if source['inversion'].get('vp_vs_ratio', 0):raise ValueError('Need fixed-property source for this control')
    output.mkdir()
    save_json(output/'provenance.json', dict(source=str(case), binary=str(exe), binary_sha256=digest(exe),
        source_config_sha256=digest(case/'inversion.toml'), source_data_sha256=digest(case/'observations.txt'),
        regularization=args.regularization, weight=args.weight,
        interpretation='2x2 explicit relation/penalty control. Geometry, free initial Vs, fixed Vs and limits/budget unchanged. Weight is illustrative, not calibrated. No adaptive complexity or automatic weight selection.'))
    summary={}
    for kind in ['fixvprho', 'brocher05']:
        for active in [False, True]:
            name=f'{kind}_{"regularized" if active else "none"}';work=output/name;work.mkdir()
            config=copy.deepcopy(source);inv=config['inversion'];inv['vs2model']=kind
            inv.setdefault('fixed_vs_layers', [])
            inv['regularization']={'type':args.regularization if active else 'none', 'lambda':args.weight if active else 0.}
            write_config(work/'inversion.toml', config);shutil.copyfile(case/'observations.txt', work/'observations.txt')
            command=run([exe, 'invert', '--config', work/'inversion.toml', '--data', work/'observations.txt', '--output', work/'lm'], work/'lm.log')
            if command['returncode'] not in [0, 4]:raise RuntimeError(f'Invocation failed: {work}/lm.log')
            costs=np.loadtxt(work/'lm/costs.txt', ndmin=2)
            report=dict(command=command, layers=len(config['model']['vs']),
                        free_parameters=len(config['model']['vs'])-len(inv['fixed_vs_layers']), runs=[])
            for row in rows(work/'lm/runs.txt'):
                i=int(row[0]);print(name, i, row[-2:], 'auditing', flush=True)
                audit=audit_terminal(exe, config, data, work, i)
                report['runs'].append(dict(run=i, converged=row[1]=='1', termination=row[-2], reason=row[-1],
                    cost=float(row[2]), data_cost=float(costs[i, 1]), regularization_cost=float(costs[i, 2]),
                    projected_gradient=float(row[9]), audit=audit))
                print(name, i, 'RMS', audit.get('rms_km_s'), 'FD', audit.get('fd_passed_at_1e_6'), flush=True)
            summary[name]=report;save_json(work/'summary.json', report)
    if (case/'lm/runs.txt').exists():
        save_json(output/'baseline-replay.json', dict(baseline_runs_identical=
            (case/'lm/runs.txt').read_bytes()==(output/'fixvprho_none/lm/runs.txt').read_bytes()))
    save_json(output/'summary.json', summary);plot(output)
    print(output)


if __name__ == '__main__':
    main()
