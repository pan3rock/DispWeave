#!/usr/bin/env python3
"""Retained failing Dunkin root-enumeration case (not an expected-pass CTest).

Exit 0 only if the independently bracketed close pair is retained for every
halfspace perturbation; exit 2 records an unresolved regression. This is a
generated LM terminal model, not an external example/runtime dependency.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_qedisp_examples import model_table, forward_rows, check_queries, run
from joint_results import write_config, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();args.output.mkdir()
    fixture = Path(__file__).resolve().parents[1]/'tests/fixtures/rayleigh/halfspace-grid-model.txt'
    model = model_table(fixture);vs = np.array(model['vs']);n = len(vs)
    query = dict(frequency_hz=[.2727]*5, mode=list(range(5)))
    expected = np.array([4.2504274453839646, 4.250444859366714])
    report = dict(fixture=str(fixture), frequency_hz=.2727, log_vs_halfspace_step=1e-5,
        independently_bracketed_pair_km_s=expected.tolist(), trials={})
    for sign in [-1, 0, 1]:
        trial = vs.copy();trial[-1] *= np.exp(sign*1e-5);model['vs'] = trial.tolist()
        prefix = args.output/str(sign)
        write_config(prefix.with_suffix('.toml'), dict(model=model, queries=query))
        np.savetxt(args.output/f'{sign}-model.txt', np.c_[np.arange(n), model['depth'], model['rho'], trial, model['vp']], fmt='%.17g')
        command = run([args.exe.resolve(), 'forward', '--config', prefix.with_suffix('.toml'),
            '--root-accuracy', 'refined', '--output', prefix.with_suffix('.txt')], prefix.with_suffix('.log'))
        if command['returncode'] not in [0, 2]:
            raise RuntimeError(f'Forward invocation failed; see {prefix.with_suffix(".log")}')
        r = forward_rows(prefix.with_suffix('.txt'));check_queries(r, query['frequency_hz'], query['mode'])
        phases = np.array([float(x[3]) for x in r])
        passed = command['returncode']==0 and all(x[4]=='valid' for x in r) and np.max(np.abs(phases[3:5]-expected)) < 2e-9
        report['trials'][str(sign)] = dict(command=command, phases=phases.tolist(), passed=bool(passed))
    report['passed'] = all(r['passed'] for r in report['trials'].values())
    save_json(args.output/'report.json', report)
    print('PASS' if report['passed'] else 'UNRESOLVED: clustered roots still omitted', args.output)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
