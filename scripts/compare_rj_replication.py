#!/usr/bin/env python3
"""Compare both frozen runs without selecting, trimming, or merging their draws."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmark_crust_depth import digest
from joint_results import read_config, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', type=Path)
    parser.add_argument('replication', type=Path)
    args = parser.parse_args()
    old = read_config(args.original/'rj.toml')
    new = read_config(args.replication/'rj.toml')
    seeds = [old['rjmcmc']['seed'], new['rjmcmc']['seed']]
    assert seeds == [20261025, 20261026]
    new['rjmcmc']['seed'] = seeds[0]
    assert old == new
    for name in ['dispweave', 'observations.txt', 'covariance.txt', 'truth.toml']:
        assert digest(args.original/name) == digest(args.replication/name)
    runs = []
    for path, seed in zip([args.original, args.replication], seeds):
        summary = json.loads((path/'summary.json').read_text())
        diagnosis = json.loads((path/'remaining-diagnosis/diagnosis.json').read_text())
        joint = [s['production']['moves']['correlated'] for s in summary['sampler']]
        attempts = sum(x['proposals'] for x in joint)
        accepts = sum(x['accepted'] for x in joint)
        conditional = []
        for k in [2, 3, 4]:
            entries = [x for x in diagnosis['conditional'] if x['k'] == k]
            counts = np.array([x['count'] for x in entries])
            means = np.array([x['depth_vs']['mean'] for x in entries])
            conditional.append(dict(k=k, per_chain_mean=means.tolist(),
                pooled_mean=float(counts @ means/counts.sum()), chain_mean_span=float(np.ptp(means))))
        runs.append(dict(seed=seed, accepted=summary['accepted'],
            visited_K_mixing_passed=summary['visited_K_mixing_passed'],
            layers=summary['layer_probabilities'], depths=summary['depth_profiles'],
            failed_depths=[x for x in summary['depth_profiles'] if x['status'] != 'no_flag'],
            max_depth_rhat=max(x['rhat'] for x in summary['depth_profiles']),
            min_depth_bulk_ess=min(x['ess_bulk'] for x in summary['depth_profiles']),
            min_depth_tail_ess=min(x['ess_tail'] for x in summary['depth_profiles']),
            target_evaluations=summary['total_target_evaluations'],
            formal_evaluations=summary['formal_evaluations'],
            forward_failures=summary['formal_forward_failures'], numerical_failures=summary['formal_numerical_failures'],
            joint_attempts=attempts, joint_accepted=accepts, joint_acceptance=accepts/attempts,
            conditional_depth=diagnosis['depth'], conditional_means=conditional,
            depth_mean_decomposition=diagnosis['depth_mean_decomposition']))
    save_json(args.replication/'replication-comparison.json', dict(runs=runs, config_only_seed_changed=True,
        frozen_binary_and_data_identical=True, merged_draws=False,
        interpretation='Both runs retained separately. One improved seed does not supersede the original failures. Conditional means are descriptive, not convergence certification.'))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for i, (run, color, marker) in enumerate(zip(runs, ['#0072B2', '#CC79A7'], ['o', 's'])):
        label = f'Seed {run["seed"]}'
        for k in [2, 3, 4]:
            item = run['layers'][k-1]
            axes[0].scatter(np.full(4, k+(i-.5)*.13), item['per_chain'], color=color,
                            marker=marker, s=30, alpha=.8, label=label if k == 2 else None)
            axes[0].plot([k+(i-.5)*.13-.04, k+(i-.5)*.13+.04], [item['probability']]*2, color=color, lw=2)
        z = [x['depth'] for x in run['depths']]
        axes[1].plot([x['rhat'] for x in run['depths']], z, color=color, label=label)
        axes[2].plot([min(x['ess_bulk'], x['ess_tail']) for x in run['depths']], z, color=color, label=label)
    axes[0].set(xticks=[2, 3, 4], xlabel='K (K=1 unvisited in both runs)', ylabel='Per-chain fraction', title='Layer occupancy; bars = run means')
    axes[1].axvline(1.01, color='#333333', ls='--')
    axes[1].set(xlabel='Depth Vs R-hat', ylabel='Depth (km)', ylim=(5, 0))
    axes[2].axvline(400, color='#333333', ls='--')
    axes[2].set(xlabel='min(bulk ESS, tail ESS)', ylabel='Depth (km)', ylim=(5, 0))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Frozen proposal, same budget, seed-only replication | both runs NOT fully accepted')
    fig.savefig(args.replication/'replication-comparison.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
