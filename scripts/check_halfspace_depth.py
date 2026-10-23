#!/usr/bin/env python3
"""Repeat fixed-interface LM at explicitly chosen halfspace top depths.

Only the last layer top changes: this changes the deepest finite layer's
thickness, not the number of layers or the infinite halfspace convention.
Keep every start/failure; never tune bounds, regularization or budgets from fits.
"""
import argparse
import copy
from pathlib import Path
import shutil
import sys
import numpy as np
from joint_results import read_config, write_config, save_json
from benchmark_crust_depth import brocher, digest, rows
from check_qedisp_examples import run, forward_rows

ROOT = Path(__file__).resolve().parents[1]


def demo(exe, output):
    source = ROOT/'examples/inversion/window_bounds.toml'
    config = read_config(source)
    queries = read_config(ROOT/'examples/inversion/synthetic.toml')['queries']
    vs = np.array([1.,2.,3.]);vp,rho,_,_ = brocher(vs)
    truth = dict(model=dict(depth=[0.,2.,4.],vs=vs.tolist(),vp=vp.tolist(),rho=rho.tolist()),queries=queries)
    write_config(output/'truth.toml', truth)
    p = run([exe,'forward','--config',output/'truth.toml','--root-accuracy','refined',
             '--output',output/'truth-forward.txt'], output/'truth-forward.log')
    if p['returncode'] != 0:raise RuntimeError('Demo truth forward failed; no observations generated')
    predicted = forward_rows(output/'truth-forward.txt')
    if any(r[4]!='valid' for r in predicted):raise RuntimeError('Demo has missing truth modes')
    rng = np.random.default_rng(20261023);sigma = .005
    c = np.array([float(r[3]) for r in predicted])+rng.normal(0,sigma,len(predicted))
    data = np.column_stack([queries['frequency_hz'],c,queries['mode'],np.full(len(c),sigma)])
    np.savetxt(output/'observations.txt',data,fmt=['%.17g','%.17g','%d','%.17g'],
               header='frequency_hz phase_km_s mode sigma_km_s; seed=20261023')
    return config, output/'observations.txt', source


def profile(table, depth):
    return table[np.searchsorted(table[:,0],depth,side='right')-1,3]


def plot(output, results, config, truth):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    bottom = 1.2*max(config['model']['depth'][-1],max(r['halfspace_top_km'] for r in results))
    if truth:
        m = read_config(output/'truth.toml')['model']
        axes[0].stairs(m['vs'],np.r_[m['depth'],bottom],orientation='horizontal',baseline=None,color='black',label='Truth')
    m = config['model']
    axes[0].stairs(m['vs'],np.r_[m['depth'],bottom],orientation='horizontal',baseline=None,color='gray',linestyle=':',label='Reference (base geometry)')
    for i,r in enumerate(results):
        if r['selected_run'] is None:continue
        table = np.loadtxt(output/r['directory']/'lm'/f"run_{r['selected_run']}_model.txt",ndmin=2)
        axes[0].stairs(table[:,3],np.r_[table[:,0],bottom],orientation='horizontal',baseline=None,
                       color=f'C{i}',linestyle='-' if r['converged'] else '--',
                       label=f"Halfspace {r['halfspace_top_km']:g} km"+('' if r['converged'] else ' (not converged)'))
        if r['rms_km_s'] is not None:
            axes[1].plot(r['halfspace_top_km'],r['rms_km_s'],'o' if r['converged'] else 'x',color=f'C{i}')
    axes[0].invert_yaxis();axes[0].set(xlabel='Vs (km/s)',ylabel='Depth (km)',title='Halfspace Vs is free')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Halfspace top (km)',ylabel='Phase RMS (km/s)',title='Same observations, bounds and penalty')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(output/'comparison.png',dpi=160);fig.savefig(output/'comparison.pdf');plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--demo',action='store_true',help='Separate three-layer truth/reference, seeded noisy data')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--covariance',type=Path)
    parser.add_argument('--depths',type=float,nargs='+',help='Halfspace tops in km; demo default: 4 6 8')
    parser.add_argument('--no-plot',action='store_true')
    args = parser.parse_args()
    if args.demo:
        if args.config or args.data or args.covariance:parser.error('--demo supplies its own inputs')
    elif args.config is None or args.data is None or args.depths is None:
        parser.error('Supply --config, --data and --depths, or --demo')
    depths = args.depths if args.depths is not None else [4.,6.,8.]
    if any(not np.isfinite(z) for z in depths) or len(set(depths))!=len(depths):parser.error('Depths must be finite and distinct')
    exe = args.exe.resolve();output = args.output
    output.mkdir()  # never mix experiments or overwrite earlier evidence
    if args.demo:
        config,data,source = demo(exe,output)
    else:
        source = args.config.resolve();config = read_config(source)
        data = output/'observations.txt';shutil.copyfile(args.data,data)
    covariance = None
    if args.covariance:
        covariance = output/'covariance.txt';shutil.copyfile(args.covariance,covariance)
    write_config(output/'reference.toml',config)
    model = config['model'];inv = config['inversion']
    if len(model['depth'])<2 or any(z<=model['depth'][-2] for z in depths):
        raise ValueError('Each halfspace top must be below the last finite-layer top')
    if len(model['vs'])-1 in inv.get('fixed_vs_layers',[]):
        raise ValueError('This control requires a free halfspace Vs')
    save_json(output/'provenance.json',dict(binary=str(exe),binary_sha256=digest(exe),source=str(source),
        source_config_sha256=digest(source),observations_sha256=digest(data),
        covariance_sha256=digest(covariance) if covariance else None,demo=args.demo,halfspace_tops_km=depths,
        interpretation='Only the last top depth changes; same Vs reference, starts, bounds, penalty settings and data. '
                       'Depth is user-selected, not an inferred resolution limit; no automatic model selection.'))
    observations = np.loadtxt(data,ndmin=2);results = [];tables = []
    for i,z in enumerate(depths):
        name = f'depth_{i}';work = output/name;work.mkdir()
        doc = copy.deepcopy(config);doc['model']['depth'][-1] = z
        write_config(work/'inversion.toml',doc)
        command = [exe,'invert','--config',work/'inversion.toml','--data',data,'--output',work/'lm']
        if covariance:command += ['--covariance',covariance]
        call = run(command,work/'lm.log')
        if call['returncode'] not in [0,4]:raise RuntimeError(f'Invocation failed: {work}/lm.log')
        attempts = rows(work/'lm/runs.txt');costs = np.loadtxt(work/'lm/costs.txt',ndmin=2)
        valid = [r for r in attempts if np.isfinite(float(r[2]))]
        converged = [r for r in valid if r[1]=='1'];pool = converged or valid
        chosen = min(pool,key=lambda r:float(r[2])) if pool else None
        record = dict(directory=name,halfspace_top_km=z,command=call,converged=bool(converged),
                      converged_starts=len(converged),total_starts=len(attempts),
                      runs=[dict(run=int(r[0]),converged=r[1]=='1',termination=r[-2],reason=r[-1],
                                 data_cost=float(costs[j,1]),regularization_cost=float(costs[j,2])) for j,r in enumerate(attempts)],
                      selected_run=int(chosen[0]) if chosen else None,rms_km_s=None)
        if chosen:
            k=int(chosen[0]);table=np.loadtxt(work/f'lm/run_{k}_model.txt',ndmin=2)
            c=forward_rows(work/f'lm/run_{k}_prediction.txt')
            if all(r[4]=='valid' for r in c) and len(c)==len(observations):
                record['rms_km_s']=float(np.sqrt(np.mean((np.array([float(r[3]) for r in c])-observations[:,1])**2)))
            tables.append(table)
        else:tables.append(None)
        results.append(record);print(z,'km',record['converged_starts'],'/',len(attempts),'converged',flush=True)
    # Compare only common finite depth coverage; do not call this a resolution estimate.
    grid=np.linspace(0,min(depths),501,endpoint=False)
    for record,table in zip(results,tables):
        if record['converged'] and results[0]['converged'] and table is not None and tables[0] is not None:
            change=profile(table,grid)-profile(tables[0],grid)
            record['vs_rms_change_from_first_km_s']=float(np.sqrt(np.mean(change**2)))
            record['comparison_depth_km']=[0.,min(depths)]
    save_json(output/'summary.json',dict(cases=results,profile_comparison_requires_convergence=True))
    if not args.no_plot:plot(output,results,config,args.demo)
    return 0 if all(r['converged'] for r in results) else 4


if __name__=='__main__':sys.exit(main())
