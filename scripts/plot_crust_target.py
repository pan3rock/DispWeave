#!/usr/bin/env python3
"""Compare the supplied synthetic target/data with existing crustal results.

The target's original Vp/rho and layer geometry are preserved. This diagnostic
does not rerun inversion or use the target to choose the reference or prior.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from benchmark_crust_depth import brocher, digest
from check_qedisp_examples import model_table, data_table, check_queries, forward_rows, run
from joint_results import read_config, write_config, save_json


def predict(exe, output, name, model, data):
    config = dict(model=model, queries=dict(frequency_hz=data[:, 0].tolist(),
                                           mode=data[:, 2].astype(int).tolist()))
    write_config(output/f'{name}.toml', config)
    result = run([exe, 'forward', '--config', output/f'{name}.toml',
                  '--root-accuracy', 'refined', '--sensitivity', 'none',
                  '--output', output/f'{name}.txt'], output/f'{name}.log')
    save_json(output/f'{name}-command.json', result)
    if result['returncode'] != 0:
        raise ValueError(f'{name}: forward did not complete successfully; outputs retained')
    rows = forward_rows(output/f'{name}.txt')
    check_queries(rows, data[:, 0], data[:, 2])
    if any(row[4] != 'valid' for row in rows):
        raise ValueError(f'{name}: invalid queries are not omitted')
    return np.array([float(row[3]) for row in rows])


def plots(output, target, reference, recovered, data, predictions, quantiles, chains=3, accepted=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    target_color, reference_color, lm_color = '#b65b2d', '#7f8789', '#cc3377'
    bottom = max(350., target['depth'][-1]+20, reference['depth'][-1]+20)

    def stairs(ax, model, label, color, style='-', width=1.8):
        # Each model retains its own layer boundaries; no index-wise alignment.
        ax.stairs(model['vs'], np.r_[model['depth'], bottom], orientation='horizontal',
                  baseline=None, label=label, color=color, linestyle=style, linewidth=width)

    def finish(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(output/f'{name}.{ext}', dpi=170)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.3), constrained_layout=True)
    for ax, depth in zip(axes, [bottom, 45]):
        stairs(ax, target, 'Supplied synthetic target (model_data.txt)', target_color, width=2.1)
        stairs(ax, reference, 'Inversion reference (mref.txt)', reference_color, '--', 1.5)
        stairs(ax, recovered, 'Existing LM result (lambda=100, run 0)', lm_color, '--', 2.0)
        ax.set(xlabel='Vs (km/s)', ylabel='Depth (km)', ylim=(depth, 0),
               title='Full model / native interfaces' if depth == bottom else 'Shallow structure / native interfaces')
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Synthetic recovery: supplied target, reference and existing LM result\n'
                 'Target: 44 layers, original Vp/rho; inversion: 45 fixed interfaces, Brocher05', fontsize=12)
    finish(fig, 'target-recovery')

    fig, axes = plt.subplots(2, 4, figsize=(18, 7.7), constrained_layout=True)
    for col, mode in enumerate(range(4)):
        idx = np.flatnonzero(data[:, 2] == mode)
        idx = idx[np.argsort(data[idx, 0])]
        frequency, observed = data[idx, 0], data[idx, 1]
        axes[0, col].errorbar(frequency, observed, yerr=data[idx, 3], fmt='k.', ms=4,
                              elinewidth=.6, alpha=.75, label='Source data; assumed sigma=0.02')
        for name, label, color, style in [
                ('target', 'Target: original Vp/rho', target_color, '-'),
                ('reference', 'Reference: Brocher05', reference_color, '--'),
                ('lm', 'Existing LM result', lm_color, '--')]:
            values = predictions[name][idx]
            axes[0, col].plot(frequency, values, color=color, linestyle=style, linewidth=1.4, label=label)
            axes[1, col].plot(frequency, (values-observed)/data[idx, 3], color=color, linestyle=style, linewidth=1.2)
        axes[0, col].set(title=f'Mode {mode} / {len(idx)} observations', ylabel='Phase velocity (km/s)')
        axes[1, col].axhline(0, color='black', linewidth=.6)
        axes[1, col].set(ylabel='(Prediction - source data) / assumed sigma')
        for ax in axes[:, col]:
            ax.set_xlabel('Frequency (Hz)'); ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Original synthetic observations and model predictions at exactly the observed queries\n'
                 '125 unmodified rows; source file has no uncertainty column; sigma=0.02 is an inference assumption', fontsize=12)
    finish(fig, 'synthetic-data')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.3), constrained_layout=True)
    z = np.r_[reference['depth'], bottom]
    for ax, depth in zip(axes, [bottom, 45]):
        ax.fill_betweenx(z, np.r_[quantiles[0], quantiles[0, -1]],
                         np.r_[quantiles[2], quantiles[2, -1]], step='post',
                         color='#0077bb', alpha=.10, label=f'{chains} chains: pointwise sample 95%')
        sample = dict(depth=reference['depth'], vs=quantiles[1])
        stairs(ax, sample, 'Posterior median' if accepted else 'Sample median (diagnostic only)', '#0077bb', width=2.0)
        stairs(ax, reference, 'Inversion reference (mref.txt)', reference_color, '--', 1.4)
        stairs(ax, recovered, 'Existing LM result (lambda=100)', lm_color, '--', width=2.0)
        stairs(ax, target, 'Supplied synthetic target', target_color, width=2.1)
        ax.set(xlabel='Vs (km/s)', ylabel='Depth (km)', ylim=(depth, 0),
               title='Full model / native interfaces' if depth == bottom else 'Shallow structure / native interfaces')
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8, loc='lower left')
    label = ('NUTS: four-chain validation passed; supplied target and LM comparison\n'
             'Pointwise intervals conditional on fixed interfaces, Brocher05, prior and assumed noise' if accepted else
             f'NUTS VALIDATION FAILED ({chains}/4 chains): target overlay for diagnosis only\n'
             'Fixed reference interfaces; the displayed intervals are not validated posterior uncertainty')
    fig.suptitle(label, fontsize=12)
    finish(fig, 'target-nuts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Local QEDispInv checkout')
    parser.add_argument('--scan', type=Path, required=True)
    parser.add_argument('--nuts', type=Path, required=True, help='Analyzed NUTS result directory with acceptance.json')
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(); scan = args.scan.resolve(); nuts = args.nuts.resolve()
    output = args.output.resolve(); exe = args.exe.resolve()
    folder = source/'demo/syn-crustmantle'
    provenance = json.loads((scan/'provenance.json').read_text())
    for filename, key in [('data.txt', 'data.txt'), ('mref.txt', 'reference.txt'), ('config.toml', 'config.toml')]:
        if digest(folder/filename) != provenance['inputs'][key]['sha256']:
            raise ValueError(f'Source {filename} differs from the frozen inversion inputs')
    upstream = read_config(folder/'config.toml')
    if upstream['forward']['file_model'] != 'model_data.txt' or upstream['inversion']['model_ref'] != 'mref.txt':
        raise ValueError('Unexpected target/reference assignment in source config')
    output.mkdir()
    sources = {}
    for name in ['model_data.txt', 'mref.txt', 'data.txt', 'config.toml']:
        shutil.copyfile(folder/name, output/name)
        sources[name] = dict(path=str(folder/name), sha256=digest(output/name))
    shutil.copyfile(Path(__file__), output/'plot-script.py')
    target = model_table(output/'model_data.txt')
    raw_reference = model_table(output/'mref.txt')
    sigma = provenance['sigma']; data = data_table(output/'data.txt', sigma)
    np.testing.assert_array_equal(data, np.loadtxt(scan/'lambda-100/observations.txt'))
    np.testing.assert_array_equal(data, np.loadtxt(nuts/'observations.txt'))
    config = read_config(scan/'lambda-100/inversion.toml')
    sample_config = read_config(nuts/'nuts.toml')
    if config['inversion']['vs2model'] != 'brocher05':
        raise ValueError('This comparison expects the recorded Brocher05 inversion')
    for key in ['vs', 'depth']:
        np.testing.assert_array_equal(raw_reference[key], config['model'][key])
        np.testing.assert_array_equal(raw_reference[key], sample_config['model'][key])
    vp, rho, _, _ = brocher(np.array(raw_reference['vs']))
    reference = dict(**raw_reference, )
    reference.update(vp=vp.tolist(), rho=rho.tolist())
    model_file = scan/'lambda-100/lm/run_0_model.txt'
    lm = np.loadtxt(model_file)
    np.testing.assert_array_equal(lm[:, 0], reference['depth'])
    np.testing.assert_array_equal(lm[:, 1], reference['vs'])
    recovered = dict(depth=lm[:, 0].tolist(), vs=lm[:, 3].tolist(),
                     vp=lm[:, 4].tolist(), rho=lm[:, 5].tolist())
    acceptance = json.loads((nuts/'acceptance.json').read_text())
    chains = len(sample_config['inversion']['starts_vs'])
    if acceptance['accepted'] and chains != 4:raise ValueError('Acceptance requires all four chains')
    sample_files = [nuts/f'nuts/chain_{c}_vs.txt' for c in range(chains)]
    samples = np.stack([np.loadtxt(path) for path in sample_files])
    if samples.shape != (chains, sample_config['nuts']['draws'], len(reference['vs'])) or not np.isfinite(samples).all():
        raise ValueError('Unexpected retained sample dimensions or nonfinite values')
    quantiles = np.quantile(samples.reshape(-1, samples.shape[-1]), [.025, .5, .975], axis=0)
    np.savetxt(output/'sample-quantiles.txt', np.c_[reference['depth'], quantiles.T],
               header='depth_km sample_q025_vs sample_q50_vs sample_q975_vs; '+('conditional posterior' if acceptance['accepted'] else 'diagnostic only'), fmt='%.17g')
    plan = dict(sources=sources, saved_lm=dict(path=str(model_file), sha256=digest(model_file)),
                sample_files={str(path): digest(path) for path in sample_files},
                source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
                source_status=subprocess.check_output(['git','-C',str(source),'status','--short'],text=True),
                binary_sha256=digest(exe), sigma_assumed=sigma,
                interpretation='Supplied synthetic target, not mref. Original target Vp/rho and native layer boundaries retained. Existing inversion/samples unchanged. Original noise draw/seed not reconstructed.')
    save_json(output/'provenance.json', plan)
    predictions = {name: predict(exe, output, name, model, data)
                   for name, model in [('target', target), ('reference', reference), ('lm', recovered)]}
    # Replaying the saved LM model must reproduce the saved terminal prediction.
    saved = forward_rows(scan/'lambda-100/audit-0/forward.txt')
    check_queries(saved, data[:, 0], data[:, 2])
    if any(row[4] != 'valid' for row in saved):
        raise ValueError('Saved LM predictions include an invalid row')
    np.testing.assert_allclose(predictions['lm'], [float(row[3]) for row in saved], rtol=0, atol=2e-9)
    diagnostics = json.loads((nuts/'diagnostics.json').read_text())
    np.testing.assert_allclose(quantiles.T, [entry['quantiles'] for entry in diagnostics['parameters']], rtol=0, atol=1e-12)
    target_vp, target_rho, _, _ = brocher(np.array(target['vs']))
    metrics = dict(observations=len(data), target_layers=len(target['vs']), reference_layers=len(reference['vs']),
                   target_halfspace_depth_km=target['depth'][-1], reference_halfspace_depth_km=reference['depth'][-1],
                   original_target_properties_preserved=True, original_data_rows_preserved=True,
                   target_brocher_max_vp_difference_km_s=float(np.max(abs(target_vp-target['vp']))),
                   target_brocher_max_rho_difference_g_cm3=float(np.max(abs(target_rho-target['rho']))),
                   rms_km_s={name: float(np.sqrt(np.mean((values-data[:, 1])**2))) for name, values in predictions.items()},
                   by_mode=[dict(mode=mode, count=int(np.sum(data[:, 2] == mode))) for mode in range(4)],
                   saved_lm_replay_agrees=True, saved_sample_quantiles_agree=True, nuts_accepted=acceptance['accepted'])
    save_json(output/'summary.json', metrics)
    plots(output, target, reference, recovered, data, predictions, quantiles, chains, acceptance['accepted'])
    for name, record in sources.items():
        if digest(Path(record['path'])) != record['sha256']:
            raise ValueError(f'External source changed during comparison: {name}')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
