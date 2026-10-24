#!/usr/bin/env python3
"""Bounded predictive and model-space checks of both unaccepted RJ runs."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark_crust_depth import digest, forward, rows
from benchmark_rj_nuts import invoke
from joint_results import read_config, write_config, save_json
from rj_results import load_run, depth_values


SOURCES = [Path('results/rj-pt-correlated'), Path('results/rj-pt-correlated-replication')]
SEEDS = [20261025, 20261026]
NOISE_SEEDS = [2026102701, 2026102702]
PER_CHAIN = 100


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    for name in ['dispweave', 'observations.txt', 'covariance.txt', 'truth.toml']:
        assert digest(SOURCES[0]/name) == digest(SOURCES[1]/name)
        shutil.copy2(SOURCES[0]/name, out/name)
    shutil.copy2(__file__, out/'run-script.py')
    baseline = read_config(Path('results/rj-nuts-comparison/nuts.toml'))
    lm = dict(model=baseline['model'], inversion=copy.deepcopy(baseline['inversion']))
    lm['inversion'].update(max_iterations=150, gradient_tolerance=1e-5)
    write_config(out/'lm.toml', lm)
    source_hashes = {}
    for source in SOURCES:
        for name in ['rj.toml', 'rj/config.toml', 'rj/observations.txt', 'rj/covariance.txt',
                     *[f'rj/chain_{c}_models.txt' for c in range(4)]]:
            source_hashes[str((source/name).resolve())] = digest(source/name)
    save_json(out/'protocol.json', dict(sources=[str(s.resolve()) for s in SOURCES],
        sampling_seeds=SEEDS, noise_seeds=NOISE_SEEDS, selected_per_chain=PER_CHAIN,
        selection='floor(j * 25000 / 100), j=0..99, each cold chain; repeats retained; no fit-based selection',
        target_evaluations_for_rj_prediction=800, no_new_sampling=True,
        lm='Same data/physics and Vs bounds; fixed K=2, interface 1.5 km (known true geometry, same as earlier NUTS comparison), four pre-existing starts; unregularized LM. Not a transdimensional competitor.',
        grid='61 depths from 0 to 5 km; interfaces: 50 bins of width 0.1 km',
        intervals='pointwise 2.5/50/97.5% quantiles; latent noiseless response versus Gaussian replicated observations using full saved covariance',
        failure_policy='Record every selected state/status. If any predictive draw fails, do not silently discard it or produce a nominal full-ensemble interval.',
        model_statistics='All 100000 retained cold states per run, separately; no pooling across runs or with training states',
        source_hashes=source_hashes,
        hashes={n:digest(out/n) for n in ['dispweave','observations.txt','covariance.txt','truth.toml','lm.toml','run-script.py']}))
    for i, source in enumerate(SOURCES):
        call = invoke(out/'dispweave', out, f'predict-{SEEDS[i]}', ['rj-predict', '--run', source.resolve()/'rj',
            '--output', out/f'predict-{SEEDS[i]}', '--draws-per-chain', PER_CHAIN, '--seed', NOISE_SEEDS[i]])
        if call['returncode'] not in [0, 5]:
            raise RuntimeError('Prediction incomplete; no retries')
    call = invoke(out/'dispweave', out, 'lm', ['invert', '--config', out/'lm.toml',
        '--data', out/'observations.txt', '--covariance', out/'covariance.txt', '--output', out/'lm'])
    if call['returncode'] != 0:
        raise RuntimeError('LM baseline did not succeed; inspect all starts, no tuning retries')
    truth = forward(out/'dispweave', out/'truth.toml', out/'truth-forward.txt')
    assert all(r[4] == 'valid' for r in truth)


def analyze(out):
    protocol = json.loads((out/'protocol.json').read_text())
    for path, sha in protocol['source_hashes'].items():
        assert digest(Path(path)) == sha
    for name, sha in protocol['hashes'].items():
        assert digest(out/name) == sha
    obs = np.loadtxt(out/'observations.txt')
    covariance = np.loadtxt(out/'covariance.txt')
    factor = np.linalg.cholesky(covariance)
    logdet = np.linalg.slogdet(covariance)[1]
    truth = np.array([float(r[3]) for r in rows(out/'truth-forward.txt')])
    meta = dict(line.split('=', 1) for line in (out/'lm/metadata.txt').read_text().splitlines())
    best = int(meta['best_run'])
    assert best >= 0
    lm_rows = rows(out/f'lm/run_{best}_prediction.txt')
    assert all(r[4] == 'valid' for r in lm_rows)
    lm_curve = np.array([float(r[3]) for r in lm_rows])
    lm_model = np.loadtxt(out/f'lm/run_{best}_model.txt')
    lm_runs = rows(out/'lm/runs.txt')
    lm_chi2 = float(np.sum(np.linalg.solve(factor, lm_curve-obs[:, 1])**2))
    assert abs(lm_chi2/2-float(lm_runs[best][2])) < 1e-8
    z = np.linspace(0, 5, 61)
    edges = np.linspace(0, 5, 51)
    result = []
    for source, seed in zip(SOURCES, SEEDS):
        folder = out/f'predict-{seed}'
        status = np.loadtxt(folder/'draws.txt', dtype=str, skiprows=1)
        latent = np.loadtxt(folder/'latent.txt')
        replicated = np.loadtxt(folder/'replicated.txt')
        assert status.shape == (400, 6) and latent.shape == replicated.shape == (400, len(obs))
        cfg, paths, chains = load_run(source/'rj')
        assert len(chains) == 4 and all(len(c) == 25000 for c in chains)
        chosen = [np.array(r, float) for r in rows(folder/'selected_models.txt')]
        for i, tr in enumerate(status):
            chain, draw = i//PER_CHAIN, (i % PER_CHAIN)*25000//PER_CHAIN
            assert list(map(int, tr[:3])) == [i, chain, draw]
            sample = chains[chain][draw]
            np.testing.assert_array_equal(chosen[i][1:], np.r_[sample['k'], sample['log_likelihood'], sample['log_prior'], sample['interfaces'], sample['vs']])
        failures = [dict(row=int(t[0]), chain=int(t[1]), draw=int(t[2]), status=t[3]) for t in status if t[3] != 'valid']
        predictive = dict(selected=len(status), failures=failures, failed_count=len(failures))
        if not failures:
            assert np.isfinite(latent).all() and np.isfinite(replicated).all()
            observed_chi2 = np.sum(np.linalg.solve(factor, (latent-obs[:, 1]).T)**2, axis=0)
            replicated_chi2 = np.sum(np.linalg.solve(factor, (replicated-latent).T)**2, axis=0)
            np.testing.assert_allclose(observed_chi2, status[:, 4].astype(float), atol=1e-9, rtol=1e-10)
            np.testing.assert_allclose(replicated_chi2, status[:, 5].astype(float), atol=1e-9, rtol=1e-10)
            ll = -.5*(observed_chi2+logdet+len(obs)*np.log(2*np.pi))
            error = float(np.max(np.abs(ll-np.array([r[2] for r in chosen]))))
            assert error < 1e-7
            lq = np.quantile(latent, [.025,.5,.975], axis=0)
            rq = np.quantile(replicated, [.025,.5,.975], axis=0)
            predictive.update(latent_quantiles=lq.tolist(), replicated_quantiles=rq.tolist(),
                observed_inside_replicated=int(np.sum((obs[:, 1]>=rq[0]) & (obs[:, 1]<=rq[2]))),
                truth_inside_latent=int(np.sum((truth>=lq[0]) & (truth<=lq[2]))),
                observed_chi2_mean=float(observed_chi2.mean()), replicated_chi2_mean=float(replicated_chi2.mean()),
                posterior_predictive_p=float(np.mean(replicated_chi2>=observed_chi2)),
                latent_width_mean=float(np.mean(lq[2]-lq[0])), replicated_width_mean=float(np.mean(rq[2]-rq[0])),
                max_saved_log_likelihood_error=error,
                per_chain=[dict(chain=c, observed_chi2_mean=float(observed_chi2[c*100:(c+1)*100].mean()),
                                posterior_predictive_p=float(np.mean(replicated_chi2[c*100:(c+1)*100]>=observed_chi2[c*100:(c+1)*100]))) for c in range(4)])
        values = depth_values(chains, z)
        quantiles = np.quantile(values.reshape(-1, len(z)), [.025,.5,.975], axis=0)
        chain_quantiles = np.quantile(values, [.025,.5,.975], axis=1).transpose(1,0,2)
        occupancy = np.zeros((4, len(edges)-1))
        k_counts = np.zeros(4, int)
        major = []
        for c, chain in enumerate(chains):
            for r in chain:
                k_counts[r['k']-1] += 1
                occupancy[c] += np.histogram(r['interfaces'], edges)[0] > 0
                if r['k'] > 1:
                    major.append(r['interfaces'][np.argmax(np.abs(np.diff(r['vs'])))])
        occupancy /= len(chains[0])
        result.append(dict(seed=seed, predictive=predictive, depth=z.tolist(),
            vs_quantiles=quantiles.tolist(), per_chain_vs_quantiles=chain_quantiles.tolist(),
            vs_95_width=(quantiles[2]-quantiles[0]).tolist(),
            # Independent uniform layer velocities imply the same marginal prior at any fixed depth.
            prior_vs_95_quantiles=[.8+.025*1.9, .8+.975*1.9],
            interface_edges=edges.tolist(), interface_occupancy=occupancy.mean(axis=0).tolist(),
            per_chain_interface_occupancy=occupancy.tolist(), k_counts=k_counts.tolist(),
            largest_jump_depth_quantiles=np.quantile(major,[.025,.5,.975]).tolist(),
            model_states=sum(map(len, chains))))
    report = dict(runs=result, lm=dict(best_run=best, fixed_interface_km=1.5,
        best_vs=lm_model[:, 3].tolist(), chi2=lm_chi2, evaluations=sum(int(r[3]) for r in lm_runs),
        starts=len(lm_runs), converged_starts=sum(r[1]=='1' for r in lm_runs),
        forward_failures=sum(int(r[6]) for r in lm_runs)),
        truth_chi2=float(np.sum(np.linalg.solve(factor, truth-obs[:, 1])**2)),
        rj_predictive_target_evaluations=800, truth_forward_evaluations=1,
        interpretation='In-sample descriptive check, not held-out prediction or calibrated coverage. Pointwise intervals, not simultaneous bands. Selected-state success does not erase rejected-proposal forward failures. No posterior convergence certification; runs remain separate.')
    (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    plot(out, obs, covariance, truth, lm_curve, lm_model, result)
    print(json.dumps(dict(lm=report['lm'], predictive=[r['predictive'] for r in result]), indent=2))


def plot(out, obs, covariance, truth, lm_curve, lm_model, runs):
    colors = ['#0072B2', '#7B3294']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    f = obs[:, 0]
    sigma = np.sqrt(np.diag(covariance))
    for j, (run, color) in enumerate(zip(runs, colors)):
        pred = run['predictive']
        ax, residual = axes[0,j], axes[1,j]
        if not pred['failed_count']:
            lq, rq = np.array(pred['latent_quantiles']), np.array(pred['replicated_quantiles'])
            ax.fill_between(f,rq[0],rq[2],color=color,alpha=.10,label='95% replicated data interval')
            ax.fill_between(f,lq[0],lq[2],color=color,alpha=.25,label='95% latent curve interval')
            ax.plot(f,lq[1],color=color,lw=2,label='RJ latent median')
            residual.fill_between(f,(lq[0]-obs[:,1])/sigma,(lq[2]-obs[:,1])/sigma,color=color,alpha=.22)
            residual.plot(f,(lq[1]-obs[:,1])/sigma,color=color,label='RJ latent median')
        else:
            ax.text(.5,.5,f'{pred["failed_count"]}/400 failures; intervals withheld',transform=ax.transAxes,ha='center')
        ax.errorbar(f,obs[:,1],yerr=sigma,fmt='k.',capsize=2,label='Observed +/- 1 sigma',zorder=5)
        ax.plot(f,truth,color='#228833',ls=':',lw=2.3,label='Target (no noise)')
        ax.plot(f,lm_curve,color='#D55E00',ls='--',lw=1.7,label='LM (fixed K=2, z=1.5 km)')
        ax.set(title=f'Seed {run["seed"]} | failures {pred["failed_count"]}/400',xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)')
        ax.legend(fontsize=7)
        residual.plot(f,(truth-obs[:,1])/sigma,color='#228833',ls=':',label='Target')
        residual.plot(f,(lm_curve-obs[:,1])/sigma,color='#D55E00',ls='--',label='LM')
        residual.axhline(0,color='#333333',lw=.8)
        residual.axhspan(-1,1,color='#999999',alpha=.10)
        residual.set(xlabel='Frequency (Hz)',ylabel='(Prediction - observed) / marginal sigma')
        residual.legend(fontsize=8)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle('Same 12 observations | pointwise intervals | in-sample check, NOT convergence acceptance')
    fig.savefig(out/'predictive.png',dpi=160);plt.close(fig)
    fig, axes = plt.subplots(1,4,figsize=(15,5.5),constrained_layout=True)
    for j,(run,color) in enumerate(zip(runs,colors)):
        ax=axes[j];z=np.array(run['depth']);q=np.array(run['vs_quantiles'])
        ax.axvspan(*run['prior_vs_95_quantiles'],color='#777777',alpha=.08,label='Marginal prior 95%')
        ax.fill_betweenx(z,q[0],q[2],color=color,alpha=.22,label='Pointwise 95%')
        ax.plot(q[1],z,color=color,lw=2,label='RJ median')
        for cq in run['per_chain_vs_quantiles']:ax.plot(cq[1],z,color=color,lw=.7,alpha=.45)
        ax.plot([1,1,2.3,2.3],[0,1.5,1.5,5],color='#228833',ls=':',lw=2.3,label='Target')
        v=lm_model[:,3];ax.plot([v[0],v[0],v[1],v[1]],[0,1.5,1.5,5],color='#D55E00',ls='--',label='LM fixed geometry')
        ax.set(xlabel='Vs (km/s)',ylabel='Depth (km)',ylim=(5,0),xlim=(.8,2.7),title=f'Seed {run["seed"]}')
        ax.legend(fontsize=7)
        edges=np.array(run['interface_edges'])
        axes[2].plot(run['interface_occupancy'],(edges[:-1]+edges[1:])/2,color=color,label=str(run['seed']))
        axes[3].bar(np.arange(1,5)+(j-.5)*.3,np.array(run['k_counts'])/run['model_states'],width=.28,color=color,label=str(run['seed']))
    axes[2].axhline(1.5,color='#228833',ls=':')
    axes[2].set(xlabel='P(any interface in 0.1-km bin)',ylabel='Depth (km)',ylim=(5,0),title='Interface location uncertainty')
    axes[3].set(xticks=[1,2,3,4],xlabel='K (includes halfspace)',ylabel='Observed fraction',title='Layer count remains uncertain')
    axes[2].legend(fontsize=8);axes[3].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('100000 retained cold states per run | separate summaries | posterior NOT fully accepted')
    fig.savefig(out/'model-uncertainty.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not args.analyze_only:prepare(out)
    analyze(out)
