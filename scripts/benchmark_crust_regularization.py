#!/usr/bin/env python3
"""Scan declared reference-damping weights with otherwise identical crustal LM inputs.

Uses only QEDispInv's model_ref and observations. Retains every start, failure,
and terminal derivative audit. No automatic weight selection or model ranking.
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from benchmark_crust_depth import digest, rows
from benchmark_crust_reference import depth_rms, reference_config
from check_qedisp_examples import data_table, model_table, run
from joint_results import read_config, save_json, write_config


WEIGHTS = [0., 10., 30., 100., 300.]
BASELINE = 100.
SETTINGS = dict(vs_min=.5, vs_max=6., width=1.2, amplitude=.2,
                regularization='reference_damping', iterations=200)
SIGMA = .02


def sweep_configs(model, weights):
    weights = list(weights)
    if (not weights or not np.isfinite(weights).all() or min(weights) < 0
            or len(set(weights)) != len(weights) or BASELINE not in weights):
        raise ValueError('Require distinct finite nonnegative lambdas including baseline 100')
    baseline, labels = reference_config(model, weight=BASELINE, **SETTINGS)
    configs = []
    for weight in weights:
        config = copy.deepcopy(baseline)
        config['inversion']['regularization']['lambda'] = float(weight)
        configs.append(config)
    return configs, labels


def fd_passed(audit):
    steps = audit.get('steps', [])
    return (audit.get('invalid_rows') == [] and audit.get('unavailable_gradients') == []
            and [step['h'] for step in steps] == [1e-5, 1e-6]
            and all(step['unresolved_fd_entries'] == 0
                    and step['max_scaled_error'] is not None
                    and np.isfinite(step['max_scaled_error'])
                    and step['max_scaled_error'] <= 1 for step in steps))


def depth_bands(model):
    bottom = float(model['depth'][-1])
    edges = sorted(set([0., *[z for z in [20., 100.] if z < bottom], bottom]))
    return list(zip(edges[:-1], edges[1:]))


def summarize_case(config, report, profiles):
    """Keep failed runs; only validated convergence supports solution comparisons."""
    nstarts = len(config['inversion']['starts_vs'])
    if (len(report) != nstarts or sorted(r['run'] for r in report) != list(range(nstarts))
            or set(profiles) != set(range(nstarts))):
        raise ValueError('Missing or duplicate initial/terminal run')
    model = config['model'];reference = np.array(model['vs']);z = np.array(model['depth'])
    inv = config['inversion']
    lower = np.maximum(inv['vs_min'], reference-inv['vs_width']/2)
    upper = np.minimum(inv['vs_max'], reference+inv['vs_width']/2)
    result = dict(weight=inv['regularization']['lambda'], runs=[])
    for record in report:
        r = copy.deepcopy(record);vs = np.asarray(profiles[r['run']])
        if vs.shape != reference.shape or not np.isfinite(vs).all():
            raise ValueError('Missing or nonfinite terminal model')
        delta = vs-reference
        r['fd_both_steps_passed'] = fd_passed(r['audit'])
        r['validated'] = r['converged'] and r['fd_both_steps_passed']
        r['final_vs_km_s'] = vs.tolist()
        r['max_reference_change_km_s'] = float(np.max(np.abs(delta)))
        r['reference_l2_km_s'] = float(np.linalg.norm(delta))
        r['halfspace_reference_change_km_s'] = float(delta[-1])
        # A reporting tolerance on physical Vs, distinct from the solver's exact-bound count.
        r['near_bound_layers_at_1e_8_km_s'] = np.flatnonzero(
            (vs-lower <= 1e-8) | (upper-vs <= 1e-8)).tolist()
        r['reference_depth_rms'] = [dict(low_km=low, high_km=high,
            rms_km_s=depth_rms(delta, z, low, high)) for low, high in depth_bands(model)]
        result['runs'].append(r)
    result['converged_count'] = sum(r['converged'] for r in result['runs'])
    result['validated_count'] = sum(r['validated'] for r in result['runs'])
    result['fd_passed_count'] = sum(r['fd_both_steps_passed'] for r in result['runs'])
    valid = [np.array(r['final_vs_km_s']) for r in result['runs'] if r['validated']]
    result['within_weight_max_layer_spread_km_s'] = float(np.ptp(valid, axis=0).max()) if len(valid) > 1 else None
    return result


def compare_weights(groups, model):
    baseline = next(g for g in groups if g['weight'] == BASELINE)
    reference_runs = [r for r in baseline['runs'] if r['validated']]
    result = []
    for group in groups:
        pairs = []
        for r, base in itertools.product([r for r in group['runs'] if r['validated']], reference_runs):
            delta = np.array(r['final_vs_km_s'])-base['final_vs_km_s']
            pairs.append(dict(run=r['run'], baseline_run=base['run'],
                max_layer_difference_km_s=float(np.abs(delta).max()),
                halfspace_difference_km_s=float(delta[-1]),
                depth_rms=[dict(low_km=low, high_km=high,
                    rms_km_s=depth_rms(delta, model['depth'], low, high)) for low, high in depth_bands(model)]))
        result.append(dict(weight=group['weight'], available=bool(pairs), pairs=pairs))
    return result


def metric_range(values):
    if not values:
        return '—'
    if any(x is None or not np.isfinite(x) for x in values):
        return '存在无效/未解析值，见完整 JSON'
    low, high = min(values), max(values)
    return f'{low:.6g}' if abs(high-low) <= 1e-6*max(abs(low), 1e-12) else f'{low:.6g}–{high:.6g}'


def write_report(output, summary):
    lines = ['# 地壳参考阻尼扫描', '',
        '同一参考模型、五个初值、观测、误差、边界、物性关系与迭代预算，仅改变 λ。',
        'σ=0.02 km/s 是明确假设；没有自动选权重。结构差异不是后验不确定性或深度分辨率。', '',
        '下表范围包含全部终态，包括未收敛结果；FD 要求两个步长均通过。', '',
        '| λ | 收敛 | FD 通过 | 相速度 RMS km/s | 数据 cost | 正则 cost | 最大参考偏移 km/s |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for g in summary['groups']:
        records = g['runs'];n = len(records)
        fields = [metric_range([r['audit'].get('rms_km_s') for r in records]),
                  metric_range([r['data_cost'] for r in records]),
                  metric_range([r['regularization_cost'] for r in records]),
                  metric_range([r['max_reference_change_km_s'] for r in records])]
        lines.append(f"| {g['weight']:g} | {g['converged_count']}/{n} | {g['fd_passed_count']}/{n} | "+' | '.join(fields)+' |')
    lines += ['', '## 同一权重下的初值稳定性', '',
        '组内差异仅使用通过验收的解；半空间速度和触界数保留全部运行。', '',
        '| λ | 最大逐层解间差 km/s | 半空间 Vs km/s | 触界参数数 |',
        '|---:|---:|---:|---:|']
    for g in summary['groups']:
        spread = g['within_weight_max_layer_spread_km_s']
        spread_text = '有效解不足两个' if spread is None else f'{spread:.6g}'
        lines.append(f"| {g['weight']:g} | {spread_text} | "
                     +metric_range([r['halfspace_vs_km_s'] for r in g['runs']])+' | '
                     +metric_range([r['bound_parameters'] for r in g['runs']])+' |')
    lines += ['', '## 全部运行', '',
        '| λ | 初值 | 终止 | 原因 | 投影梯度 | 正演/梯度失败 | 触界数 | FD |',
        '|---:|---|---|---|---:|---:|---:|---|']
    for g in summary['groups']:
        for r in g['runs']:
            lines.append(f"| {g['weight']:g} | {r['label']} | {r['termination']} | {r['reason']} | "
                         f"{r['projected_gradient']:.4g} | {r['forward_failures']}/{r['gradient_failures']} | "
                         f"{r['bound_parameters']} | {'pass' if r['fd_both_steps_passed'] else 'fail/unavailable'} |")
    lines += ['', '## 相对 λ=100 的结构变化', '',
        '仅比较收敛且通过两个步长 FD 的解，列出全部有效初值配对中的最大差异；无有效配对时保留空项。', '',
        '| λ | 深度 km | 最大深度加权 Vs RMS 差 km/s |', '|---:|---|---:|']
    for comparison in summary['comparisons']:
        if not comparison['available']:
            lines.append(f"| {comparison['weight']:g} | 无有效配对 | — |")
            continue
        for k, (low, high) in enumerate(depth_bands(summary['model'])):
            maximum = max(p['depth_rms'][k]['rms_km_s'] for p in comparison['pairs'])
            lines.append(f"| {comparison['weight']:g} | {low:g}–{high:g} | {maximum:.6g} |")
    lines += ['', '## 分阶拟合', '', '每组范围包含所有终态，缺失预测不会被静默删去。', '',
        '| λ | 模态 | RMS km/s | 最大绝对残差 km/s |', '|---:|---:|---:|---:|']
    for g in summary['groups']:
        for mode in sorted(set(r['mode'] for r in g['fit_by_mode'])):
            records = [r for r in g['fit_by_mode'] if r['mode'] == mode]
            if any(r['rms_km_s'] is None for r in records):
                rms = maximum = '存在未解析预测，见完整 JSON'
            else:
                rms = metric_range([r['rms_km_s'] for r in records])
                maximum = metric_range([r['max_absolute_km_s'] for r in records])
            lines.append(f"| {g['weight']:g} | {mode} | {rms} | {maximum} |")
    lines += ['', '所有输入、实际源码/程序哈希、预声明配置、模型、轨迹、代价、FD 数组和失败诊断均保留。',
              'λ 稳定性仅针对本次固定参考、界面、速度窗口、误差与有限初值范围；不证明数据独立确定了结构。',
              '各 λ 的总目标不同，不以总 cost 进行跨权重排名。', '']
    (output/'report.md').write_text('\n'.join(lines))


def plot(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = summary['groups'];model = summary['model'];reference = np.array(model['vs'])
    bottom = max(250., 1.2*model['depth'][-1]);z = np.r_[model['depth'], bottom]
    data = np.loadtxt(output/groups[0]['directory']/'observations.txt', ndmin=2)
    modes = np.unique(data[:, 2]).astype(int)
    fig, axes = plt.subplots(len(groups), 3, figsize=(15, 3.8*len(groups)), sharex='col',
                             squeeze=False, constrained_layout=True)
    for index, g in enumerate(groups):
        for ax in axes[index, :2]:
            ax.stairs(reference, z, orientation='horizontal', baseline=None, color='black', label='Reference')
        axes[index, 2].axvline(0, color='black', linewidth=.8)
        for r in g['runs']:
            values = np.array(r['final_vs_km_s'])
            style = '-' if r['validated'] else '--'
            for j, ax in enumerate(axes[index]):
                ax.stairs(values if j < 2 else values-reference, z, orientation='horizontal', baseline=None,
                    color=f"C{r['run']}", linestyle=style, alpha=.75,
                    label=f"{r['label']}: {r['termination']}"+(' [FD failed]' if r['converged'] and not r['validated'] else ''))
        for j, ax in enumerate(axes[index]):
            ax.set(xlabel='Vs (km/s)' if j < 2 else 'Vs - reference (km/s)', ylabel='Depth (km)',
                   ylim=(40 if j == 1 else bottom, 0),
                   title=f"Lambda {g['weight']:g}; {g['converged_count']}/{len(g['runs'])} converged")
            ax.grid(alpha=.2)
        axes[index, 0].legend(fontsize=7)
    fig.suptitle('All starts; dashed = unconverged or failed derivative audit')
    for ext in ['png', 'pdf']:fig.savefig(output/f'structure.{ext}', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(len(groups), len(modes), figsize=(16, 3*len(groups)), sharex='col', sharey=True,
                             squeeze=False, constrained_layout=True)
    for index, g in enumerate(groups):
        for r in g['runs']:
            prediction = rows(output/g['directory']/f"audit-{r['run']}/forward.txt")
            c = np.array([float(row[3]) if row[4] == 'valid' else np.nan for row in prediction])
            for j, mode in enumerate(modes):
                chosen = np.flatnonzero(data[:, 2] == mode);chosen = chosen[np.argsort(data[chosen, 0])]
                axes[index, j].plot(data[chosen, 0], (c[chosen]-data[chosen, 1])/data[chosen, 3],
                    '.-' if r['validated'] else '.--', color=f"C{r['run']}", alpha=.7, label=r['label'])
        for j, ax in enumerate(axes[index]):
            ax.axhline(0, color='black', linewidth=.7);ax.axhspan(-1, 1, color='gray', alpha=.12)
            ax.set(xlabel='Frequency (Hz)', ylabel='Residual / assumed sigma', title=f"Lambda {g['weight']:g}, mode {modes[j]}")
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('All modes and all terminal attempts; unresolved predictions remain gaps')
    for ext in ['png', 'pdf']:fig.savefig(output/f'residuals.{ext}', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    axes[0].stairs(reference, z, orientation='horizontal', baseline=None, color='black', label='Reference')
    axes[1].axvline(0, color='black', linewidth=.7)
    for i, g in enumerate(groups):
        first = True
        for r in g['runs']:
            if r['validated']:
                vs = np.array(r['final_vs_km_s'])
                for j in [0, 1]:
                    axes[j].stairs(vs if j == 0 else vs-reference, z, orientation='horizontal', baseline=None,
                        color=f'C{i}', alpha=.65, label=f"Lambda {g['weight']:g}" if first else None)
                first = False
            rms = r['audit'].get('rms_km_s')
            if rms is not None:
                axes[2].plot(g['weight'], rms, 'o' if r['validated'] else 'x', color=f'C{i}')
    for j in [0, 1]:
        axes[j].set(xlabel='Vs (km/s)' if j == 0 else 'Vs - reference (km/s)', ylabel='Depth (km)',
                    ylim=(bottom, 0), title='All validated solutions' if j == 0 else 'Reference dependence')
        axes[j].legend(fontsize=8)
    axes[2].set(xlabel='Lambda', ylabel='Phase velocity RMS (km/s)', xscale='symlog',
                title='All attempts: circle = validated, cross = other')
    axes[2].set_xticks([g['weight'] for g in groups], [f"{g['weight']:g}" for g in groups])
    axes[2].axhline(SIGMA, color='gray', linestyle=':', label='Assumed sigma');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    for ext in ['png', 'pdf']:fig.savefig(output/f'lambda-comparison.{ext}', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lambdas', type=float, nargs='+', default=WEIGHTS)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args();source = args.source.resolve();exe = args.exe.resolve();output = args.output.resolve()
    folder = source/'demo/syn-crustmantle';upstream = read_config(folder/'config.toml')
    reference = folder/upstream['inversion']['model_ref'];model = model_table(reference)
    data_table(folder/'data.txt', SIGMA)  # Validate before creating output.
    configs, labels = sweep_configs(model, args.lambdas)
    output.mkdir();(output/'planned').mkdir();(output/'scripts').mkdir();(output/'inputs').mkdir()
    inputs = {}
    for path, name in [(reference, 'reference.txt'), (folder/'data.txt', 'data.txt'), (folder/'config.toml', 'config.toml')]:
        shutil.copyfile(path, output/'inputs'/name)
        inputs[name] = dict(path=str(path), sha256=digest(path))
    # Preserve imported helpers as well as the entry points for reproducibility.
    scripts = Path(__file__).resolve().parent
    for path in scripts.glob('*.py'):shutil.copyfile(path, output/'scripts'/path.name)
    names = [f'lambda-{weight:.17g}' for weight in args.lambdas]
    for name, config in zip(names, configs):write_config(output/'planned'/f'{name}.toml', config)
    provenance = dict(weights=args.lambdas, baseline=BASELINE, settings=SETTINGS, sigma=SIGMA,
        start_labels=labels, inputs=inputs, binary=str(exe), binary_sha256=digest(exe),
        source_commit=subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        source_status=subprocess.check_output(['git', '-C', str(source), 'status', '--short'], text=True),
        project_commit=subprocess.check_output(['git', '-C', str(scripts.parent), 'rev-parse', 'HEAD'], text=True).strip(),
        scripts={p.name:digest(p) for p in (output/'scripts').glob('*.py')},
        interpretation='Only reference_damping.lambda changes. All attempts and FD failures retained. '
                       'No automatic weight selection; no posterior uncertainty or depth resolution claim.')
    save_json(output/'provenance.json', provenance)
    summary = dict(model=model, groups=[], comparisons=[])
    for name, config in zip(names, configs):
        weight = config['inversion']['regularization']['lambda'];work = output/name
        print(f'lambda={weight:g}: running all five starts and terminal audits', flush=True)
        call = run([sys.executable, output/'scripts/benchmark_crust_reference.py', '--source', source, '--exe', exe,
                    '--output', work, '--vs-min', SETTINGS['vs_min'], '--vs-max', SETTINGS['vs_max'],
                    '--vs-width', SETTINGS['width'], '--start-amplitude', SETTINGS['amplitude'], '--sigma', SIGMA,
                    '--regularization', SETTINGS['regularization'], '--lambda', weight,
                    '--max-iterations', SETTINGS['iterations'], '--no-plot'], output/f'{name}.log')
        if call['returncode'] not in [0, 4]:raise RuntimeError(f'Experiment invocation failed: {output}/{name}.log')
        if read_config(work/'inversion.toml') != config:raise ValueError('Actual config differs from declared plan')
        for original, local in [('reference.txt', 'source-reference.txt'), ('data.txt', 'source-data.txt'), ('config.toml', 'source-config.toml')]:
            if digest(work/local) != inputs[original]['sha256']:raise ValueError('Source changed during sweep')
        if digest(exe) != provenance['binary_sha256']:raise ValueError('Executable changed during sweep')
        saved = json.loads((work/'summary.json').read_text())
        profiles = {r['run']:np.loadtxt(work/f"lm/run_{r['run']}_model.txt", ndmin=2)[:, 3] for r in saved['runs']}
        group = summarize_case(config, saved['runs'], profiles)
        group.update(directory=name, command=call, fit_by_mode=json.loads((work/'fit-by-mode.json').read_text()))
        summary['groups'].append(group);save_json(work/'sweep-summary.json', group)
        if name != names[0]:
            for filename in ['observations.txt', 'lm/parameter_bounds.txt']:
                if (work/filename).read_bytes() != (output/names[0]/filename).read_bytes():
                    raise ValueError(f'Common inputs differ: {filename}')
        print(f"lambda={weight:g}: converged {group['converged_count']}/5; both-step FD {group['fd_passed_count']}/5", flush=True)
    summary['comparisons'] = compare_weights(summary['groups'], model)
    summary['all_runs_validated'] = all(g['validated_count'] == len(g['runs']) for g in summary['groups'])
    save_json(output/'summary.json', summary);write_report(output, summary)
    if not args.no_plot:plot(output, summary)
    print(output, flush=True)
    # A completed scan can contain failed inversions. Preserve this distinction.
    return 0 if summary['all_runs_validated'] else 4


if __name__ == '__main__':
    sys.exit(main())
