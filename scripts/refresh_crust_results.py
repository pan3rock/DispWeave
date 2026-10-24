#!/usr/bin/env python3
"""Package accepted crustal inputs and redraw local results from saved outputs only."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import numpy as np
from joint_results import read_config,save_json
from benchmark_nuts_warmup import digest
from benchmark_crust_nuts import plot
from plot_crust_target import plots
from check_qedisp_examples import model_table,forward_rows,check_queries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nuts',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();nuts=a.nuts.resolve();cache=a.comparison.resolve();out=a.output.resolve()
    acceptance=json.loads((nuts/'acceptance.json').read_text());assert acceptance['accepted']
    cfg=read_config(nuts/'nuts.toml');assert cfg['nuts']['target_accept']==.95 and cfg['nuts']['find_initial_step_size']
    assert cfg['nuts'].get('kernel','slice')=='slice'
    provenance=json.loads((nuts/'provenance.json').read_text())
    for name,h in provenance['files'].items():assert digest(nuts/name)==h
    source=json.loads((cache/'provenance.json').read_text())
    for name,r in source['sources'].items():assert digest(cache/name)==r['sha256']
    target=model_table(cache/'model_data.txt')
    reference=read_config(cache/'reference.toml')['model'];lm=read_config(cache/'lm.toml')['model']
    assert read_config(cache/'target.toml')['model']==target
    for key in ['depth','vs']:np.testing.assert_array_equal(reference[key],cfg['model'][key])
    data=np.loadtxt(nuts/'observations.txt');raw=np.loadtxt(cache/'data.txt')
    np.testing.assert_array_equal(data[:,:3],raw);assert np.all(data[:,3]==source['sigma_assumed'])
    predictions={}
    for name in ['target','reference','lm']:
        rows=forward_rows(cache/f'{name}.txt');check_queries(rows,data[:,0],data[:,2]);assert all(r[4]=='valid' for r in rows)
        predictions[name]=np.array([float(r[3]) for r in rows])
    samples=np.stack([np.loadtxt(nuts/f'nuts/chain_{c}_vs.txt') for c in range(4)])
    assert samples.shape==(4,cfg['nuts']['draws'],len(cfg['model']['vs']))
    diagnostics=json.loads((nuts/'diagnostics.json').read_text());assert diagnostics['accepted']
    q=np.quantile(samples.reshape(-1,samples.shape[-1]),[.025,.5,.975],axis=0)
    np.testing.assert_allclose(q.T,[d['quantiles'] for d in diagnostics['parameters']],rtol=0,atol=1e-12)
    predictive=json.loads((nuts/'predictive-summary.json').read_text());assert predictive['valid'] and predictive['failures']==0
    latent=np.loadtxt(nuts/'predictive/latent.txt');replicated=np.loadtxt(nuts/'predictive/replicated.txt')
    assert latent.shape==replicated.shape==(400,len(data))
    out.mkdir();inputs=out/'inputs';inputs.mkdir()
    for name in ['nuts.toml','observations.txt','covariance.txt']:shutil.copyfile(nuts/name,inputs/name)
    shutil.copyfile(Path(__file__),out/'refresh-script.py')
    plots(out,target,reference,lm,data,predictions,q,chains=4,accepted=True)
    plot(out,cfg,samples,diagnostics,data,latent,replicated,predictive)
    np.savetxt(out/'posterior-quantiles.txt',np.c_[cfg['model']['depth'],q.T],header='depth_km q025 median q975',fmt='%.17g')
    root=Path(__file__).resolve().parent.parent
    command=[str(root/'build/dispweave'),'nuts','--config',str(inputs/'nuts.toml'),'--data',str(inputs/'observations.txt'),'--covariance',str(inputs/'covariance.txt'),'--output']
    (out/'run.sh').write_text('#!/bin/sh\nset -eu\nif [ "$#" -ne 1 ]; then echo "Usage: sh run.sh NEW_OUTPUT_DIRECTORY" >&2; exit 2; fi\nexec '+shlex.join(command)+' "$1"\n')
    (out/'RUN.txt').write_text('完整配置为 inputs/nuts.toml，数据与协方差随附。\n从项目根目录执行：\nsh '+str((out/'run.sh').relative_to(root))+' results/crust-nuts-recommended-new\n\n此直接 CLI 命令串行运行四链，耗时不等于四进程验收的约五分钟。\n运行会生成新的采样结果，不覆盖已验收目录；返回 0 后仍需混合诊断。\n完整并行验收命令见 docs/nuts-recommended-use.md。\n')
    tracked=[nuts/'nuts.toml',nuts/'observations.txt',nuts/'covariance.txt',nuts/'acceptance.json',nuts/'diagnostics.json',nuts/'predictive-summary.json',nuts/'predictive/latent.txt',nuts/'predictive/replicated.txt']
    tracked += [nuts/f'nuts/chain_{c}_vs.txt' for c in range(4)]
    tracked += [cache/n for n in ['model_data.txt','mref.txt','data.txt','target.toml','reference.toml','lm.toml','target.txt','reference.txt','lm.txt']]
    save_json(out/'provenance.json',dict(source_run=str(nuts),inputs={str(f):digest(f) for f in tracked},
        packaged_inputs={f.name:digest(f) for f in inputs.iterdir()},script_sha256=digest(Path(__file__)),
        accepted=True,draws=int(samples.shape[0]*samples.shape[1]),no_sampling_or_forward_calls=True))
    print('Prepared configuration, command and seven figures:',out)


if __name__=='__main__':main()
