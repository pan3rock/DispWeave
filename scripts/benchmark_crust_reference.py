#!/usr/bin/env python3
"""Fixed-interface crustal LM using QEDispInv's reference, not model_data.txt.

All Vs, including the halfspace, are free. Numerical settings are supplied
explicitly; every initial/terminal failure is retained. No weight selection.
"""
import argparse
import copy
import itertools
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from joint_results import read_config, write_config, save_json
from check_qedisp_examples import model_table, data_table, run, check_queries
from benchmark_crust_depth import brocher, forward, audit_terminal, digest, rows


def reference_config(model, vs_min, vs_max, width, amplitude, regularization, weight, iterations):
    if (not np.isfinite([vs_min,vs_max,width,amplitude,weight]).all() or
            vs_min<=0 or vs_max<=vs_min or width<=0 or amplitude<=0 or weight<0 or iterations<1):
        raise ValueError('Require finite positive bounds, width, amplitude and iteration budget; weight must be nonnegative')
    if regularization not in ['reference_damping','perturbation_smooth','vs_smooth']:
        raise ValueError('Explicit supported regularization required')
    vs=np.asarray(model['vs']);z=np.asarray(model['depth'])
    lower=np.maximum(vs_min,vs-width/2);upper=np.minimum(vs_max,vs+width/2)
    if np.any(lower>=upper) or np.any(vs<lower) or np.any(vs>upper):
        raise ValueError('Reference outside feasible intersection')
    labels=['reference','uniform_minus','uniform_plus','depth_sine','depth_cosine']
    # Physical depth, not layer index: thinly spaced reference layers do not
    # introduce high-frequency alternating initial Vs. Same amplitude in km/s.
    offsets=[np.zeros(len(vs)),np.full(len(vs),-amplitude),np.full(len(vs),amplitude),
             amplitude*np.sin(2*np.pi*z/100.),amplitude*np.cos(2*np.pi*z/100.)]
    starts=[vs+a for a in offsets]
    if any(np.any(a<=lower) or np.any(a>=upper) for a in starts):
        raise ValueError('A declared initial model is outside strict bounds; no clipping or replacement')
    config=dict(model=copy.deepcopy(model),inversion=dict(vs_min=vs_min,vs_max=vs_max,vs_width=width,
        vs2model='brocher05',starts_vs=[v.tolist() for v in starts],fixed_vs_layers=[],
        sensitivity='implicit',max_iterations=iterations,gradient_tolerance=1e-5,initial_damping=.001,
        regularization=dict(type=regularization,**{'lambda':weight})))
    return config,labels


def depth_rms(delta, tops, low, high):
    delta=np.asarray(delta);tops=np.asarray(tops)
    if delta.shape!=tops.shape or not 0<=low<high or tops[0]!=0 or np.any(np.diff(tops)<=0):
        raise ValueError('Invalid piecewise-constant depth comparison')
    bottoms=np.r_[tops[1:],high]
    weights=np.maximum(0,np.minimum(bottoms,high)-np.maximum(tops,low))
    return float(np.sqrt(np.sum(weights*delta**2)/(high-low)))


def stability(config, report, output):
    successful=[r['run'] for r in report if r['converged']]
    profiles={i:np.loadtxt(output/f'lm/run_{i}_model.txt',ndmin=2)[:,3] for i in successful}
    z=np.array(config['model']['depth'])
    result=dict(converged_runs=successful,interpretation='Spread across converged starts, not posterior uncertainty or resolution')
    if len(successful)<2:
        result['available']=False;return result
    result['available']=True
    values=np.array(list(profiles.values()));spread=np.ptp(values,axis=0)
    result['layer_spread_km_s']=spread.tolist()
    result['max_layer_spread_km_s']=float(spread.max())
    result['halfspace_vs_km_s']=values[:,-1].tolist()
    result['halfspace_spread_km_s']=float(spread[-1])
    result['depth_bands']=[]
    for low,high in [(0.,20.),(20.,100.),(100.,float(z[-1]))]:
        if high<=low:continue
        comparisons=[dict(runs=[i,j],rms_km_s=depth_rms(profiles[i]-profiles[j],z,low,high))
                     for i,j in itertools.combinations(successful,2)]
        result['depth_bands'].append(dict(low_km=low,high_km=high,
            max_pairwise_rms_km_s=max(c['rms_km_s'] for c in comparisons),pairs=comparisons))
    return result


def plot(output, config, data, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    model=config['model'];inv=config['inversion'];bottom=max(250.,1.2*model['depth'][-1]);z=np.r_[model['depth'],bottom]
    lo=np.maximum(inv['vs_min'],np.array(model['vs'])-inv['vs_width']/2)
    hi=np.minimum(inv['vs_max'],np.array(model['vs'])+inv['vs_width']/2)
    for ax in axes[:2]:
        ax.fill_betweenx(z,np.r_[lo,lo[-1]],np.r_[hi,hi[-1]],step='post',color='gray',alpha=.15,label='Allowed Vs')
        ax.stairs(model['vs'],z,orientation='horizontal',baseline=None,color='black',linewidth=1.4,label='mref reference')
    for r in report:
        i=r['run'];table=np.loadtxt(output/f'lm/run_{i}_model.txt',ndmin=2)
        label=f"{i}: {r['termination']}"
        for ax in axes[:2]:
            ax.stairs(table[:,3],z,orientation='horizontal',baseline=None,color=f'C{i}',
                linestyle='-' if r['converged'] else '--',alpha=.8,label=label)
        p=rows(output/f'audit-{i}/forward.txt')
        c=np.array([float(v[3]) if v[4]=='valid' else np.nan for v in p])
        for mode in np.unique(data[:,2]):
            indices=np.flatnonzero(data[:,2]==mode);indices=indices[np.argsort(data[indices,0])]
            axes[2].plot(data[indices,0],(c[indices]-data[indices,1])/data[indices,3],'.',color=f'C{i}',alpha=.7)
    axes[0].set(xlabel='Vs (km/s)',ylabel='Depth (km)',ylim=(bottom,0),
                title=f"{len(model['vs'])} free Vs; halfspace at {model['depth'][-1]:g} km")
    axes[0].axhline(model['depth'][-1],color='gray',linewidth=.7);axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Vs (km/s)',ylabel='Depth (km)',ylim=(40,0),title='Shallow structure, same runs')
    axes[2].axhline(0,color='black',linewidth=.7)
    axes[2].set(xlabel='Frequency (Hz)',ylabel='Residual / assumed sigma',title='All modes; invalid predictions remain gaps')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(output/'comparison.png',dpi=160);fig.savefig(output/'comparison.pdf');plt.close(fig)
    modes=np.unique(data[:,2]).astype(int)
    fig,axes=plt.subplots(2,(len(modes)+1)//2,figsize=(12,8),squeeze=False,constrained_layout=True)
    for ax,mode in zip(axes.flat,modes):
        index=np.flatnonzero(data[:,2]==mode);index=index[np.argsort(data[index,0])]
        ax.errorbar(data[index,0],data[index,1],yerr=data[index,3],fmt='k.',elinewidth=.7,label='Data, assumed 1 sigma')
        for r in report:
            p=rows(output/f"audit-{r['run']}/forward.txt")
            c=np.array([float(v[3]) if v[4]=='valid' else np.nan for v in p])
            ax.plot(data[index,0],c[index],color=f"C{r['run']}",
                    linestyle='-' if r['converged'] else '--',label=f"Run {r['run']}: {r['termination']}")
        ax.set(xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)',title=f'Mode {mode}')
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(modes):]:ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Observed and predicted dispersion; overlapping curves show agreement across starts')
    fig.savefig(output/'dispersion.png',dpi=160);fig.savefig(output/'dispersion.pdf');plt.close(fig)


def modal_metrics(output, data, report):
    metrics=[]
    for record in report:
        p=rows(output/f"audit-{record['run']}/forward.txt")
        c=np.array([float(r[3]) if r[4]=='valid' else np.nan for r in p])
        for mode in np.unique(data[:,2]).astype(int):
            mask=data[:,2]==mode;residual=c[mask]-data[mask,1]
            # NaN propagates to all metrics if any prediction is unresolved;
            # save_json writes null rather than silently dropping a datum.
            metrics.append(dict(run=record['run'],mode=int(mode),count=int(mask.sum()),
                rms_km_s=float(np.sqrt(np.mean(residual**2))),mean_km_s=float(residual.mean()),
                max_absolute_km_s=float(abs(residual).max())))
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='Local QEDispInv repository')
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    for flag in ['vs-min','vs-max','vs-width','start-amplitude','sigma','lambda']:
        parser.add_argument('--'+flag,type=float,required=True,dest='weight' if flag=='lambda' else flag.replace('-','_'))
    parser.add_argument('--regularization',choices=['reference_damping','perturbation_smooth','vs_smooth'],required=True)
    parser.add_argument('--max-iterations',type=int,default=200)
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args();source=args.source.resolve();exe=args.exe.resolve();output=args.output
    folder=source/'demo/syn-crustmantle';upstream=read_config(folder/'config.toml')
    reference=folder/upstream['inversion']['model_ref']
    model=model_table(reference);data=data_table(folder/'data.txt',args.sigma)
    config,labels=reference_config(model,args.vs_min,args.vs_max,args.vs_width,args.start_amplitude,
                                  args.regularization,args.weight,args.max_iterations)
    output.mkdir()
    shutil.copyfile(Path(__file__),output/'benchmark-script.py')
    sources={}
    for path,name in [(reference,'source-reference.txt'),(folder/'data.txt','source-data.txt'),(folder/'config.toml','source-config.toml')]:
        shutil.copyfile(path,output/name);sources[name]=dict(path=str(path),sha256=digest(path))
    write_config(output/'inversion.toml',config)
    np.savetxt(output/'observations.txt',data,fmt=['%.17g','%.17g','%d','%.17g'],header='frequency_hz phase_km_s mode sigma_km_s; sigma assumed')
    save_json(output/'provenance.json',dict(source=str(source),files=sources,
        source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        source_status=subprocess.check_output(['git','-C',str(source),'status','--short'],text=True),
        binary=str(exe),binary_sha256=digest(exe),script_sha256=digest(Path(__file__)),
        settings=dict(vs_min=args.vs_min,vs_max=args.vs_max,vs_width=args.vs_width,sigma=args.sigma,
                      regularization=args.regularization,weight=args.weight,start_amplitude=args.start_amplitude),
        start_labels=labels,start_period_km=100.,
        interpretation='Uses model_ref and observed data only, never model_data. Fixed reference interfaces; all Vs free. '
                       'Sigma and penalty weight are explicit illustrative assumptions; no weight tuning. '
                       'Not an exact QEDispInv inversion reproduction; no modal reweighting or skipped invalid points.'))
    # Forward-check every initial model, including ones that will fail; never
    # replace initial models based on root availability.
    initial=[]
    for i,vs in enumerate(config['inversion']['starts_vs']):
        work=output/f'initial-{i}';work.mkdir();vp,rho,_,_=brocher(np.array(vs))
        d=dict(model=dict(depth=model['depth'],vs=vs,vp=vp.tolist(),rho=rho.tolist()),
               queries=dict(frequency_hz=data[:,0].tolist(),mode=data[:,2].astype(int).tolist()))
        write_config(work/'forward.toml',d)
        predicted=forward(exe,work/'forward.toml',work/'forward.txt','implicit')
        check_queries(predicted,data[:,0],data[:,2])
        record=dict(run=i,label=labels[i],invalid_rows=[j for j,r in enumerate(predicted) if r[4]!='valid'],
                    unavailable_gradients=[j for j,r in enumerate(predicted) if r[7]!='valid'])
        initial.append(record);print('initial',i,record,flush=True)
    save_json(output/'initial.json',initial)
    call=run([exe,'invert','--config',output/'inversion.toml','--data',output/'observations.txt','--output',output/'lm'],output/'lm.log')
    if call['returncode'] not in [0,4]:raise RuntimeError(f'Invocation failed: {output}/lm.log')
    costs=np.loadtxt(output/'lm/costs.txt',ndmin=2);report=[]
    for row in rows(output/'lm/runs.txt'):
        i=int(row[0]);print('terminal',i,row[-2:],'auditing',flush=True)
        audit=audit_terminal(exe,config,data,output,i)
        table=np.loadtxt(output/f'lm/run_{i}_model.txt',ndmin=2)
        regularization=config['inversion']['regularization'];vs=table[:,3];ref=table[:,1]
        delta=vs-ref if regularization['type']!='vs_smooth' else vs
        penalty=(.5*args.weight*np.sum(delta**2) if regularization['type']=='reference_damping'
                 else .5*args.weight*np.sum(np.diff(delta)**2/np.diff(table[:,0])))
        if np.isfinite(costs[i,1]):
            np.testing.assert_allclose(costs[i,2],penalty,rtol=1e-10,atol=1e-9)
            if 'rms_km_s' in audit:
                predicted=rows(output/f'audit-{i}/forward.txt')
                c=np.array([float(r[3]) for r in predicted])
                np.testing.assert_allclose(costs[i,1],.5*np.sum(((c-data[:,1])/data[:,3])**2),rtol=1e-9,atol=1e-8)
        r=dict(run=i,label=labels[i],converged=row[1]=='1',termination=row[-2],reason=row[-1],
               evaluations=int(row[3]),forward_failures=int(row[6]),gradient_failures=int(row[7]),
               projected_gradient=float(row[9]),bound_parameters=int(row[10]),
               data_cost=float(costs[i,1]),regularization_cost=float(costs[i,2]),total_cost=float(costs[i,3]),
               halfspace_vs_km_s=float(vs[-1]),audit=audit)
        report.append(r);save_json(output/f'run_{i}_summary.json',r)
        print('result',i,'converged',r['converged'],'RMS',audit.get('rms_km_s'),'FD',audit.get('fd_passed_at_1e_6'),flush=True)
    comparison=stability(config,report,output)
    save_json(output/'summary.json',dict(command=call,layers=len(model['vs']),free_vs=len(model['vs']),
        halfspace_top_km=model['depth'][-1],runs=report,stability=comparison))
    save_json(output/'fit-by-mode.json',modal_metrics(output,data,report))
    if not args.no_plot:plot(output,config,data,report)
    # Full experiment success means every declared start converged and its
    # terminal derivative audit passed. Partial success remains explicitly nonzero.
    return 0 if all(r['converged'] and r['audit'].get('fd_passed_at_1e_6',False) for r in report) else 4


if __name__=='__main__':sys.exit(main())
