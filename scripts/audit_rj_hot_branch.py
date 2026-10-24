#!/usr/bin/env python3
"""Bounded independent root scan and saved-warmup inspection; never samples."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from benchmark_crust_depth import digest
from joint_results import save_json


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('results/rj-pt-transdim'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 src=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 exe=out/'rj_forward_audit';shutil.copy2('build/rj_forward_audit',exe);shutil.copy2(__file__,out/'run-script.py')
 selected=[];lines=['# id kind K interfaces Vs'];hashes={}
 def records(path):
  hashes[str(path.relative_to(src))]=digest(path)
  return [line.split() for line in path.read_text().splitlines() if line and not line.startswith('#')]
 def add(kind,row,path,index,failed=False):
  ident=len(selected)
  if failed:k=int(row[2]);state=row[3:];extra=dict(evaluation=int(row[0]))
  else:k=int(row[1]);state=row[4:];extra=dict(log_likelihood=float(row[2]))
  selected.append(dict(id=ident,kind=kind,source=str(path.relative_to(src)),source_row=index,k=k,**extra))
  lines.append(' '.join(map(str,[ident,kind,k,*state])))
 for c,t,kind in [(0,3,'trapped_retained'),(1,3,'control_hot'),(0,0,'control_cold')]:
  path=src/f'rj/ladder_{c}/temperature_{t}_models.txt';rows=records(path);indices=[0,len(rows)//2,len(rows)-1]
  if kind=='trapped_retained':indices=sorted(set(indices+[int(np.argmin([float(r[2]) for r in rows])),int(np.argmax([float(r[2]) for r in rows]))]))
  for i in indices:add(kind,rows[i],path,i)
 d=src/'rj/ladder_0';path=d/'temperature_3_failed_models.txt';failed=records(path);trace=np.array(records(d/'temperature_3_local_trace.txt'))
 phase={int(r[9]):int(r[0]) for r in trace if r[3]=='1'}
 for warm in [1,0]:
  indices=[i for i,r in enumerate(failed) if phase[int(r[0])]==warm]
  for at in [0,len(indices)//2,len(indices)-1]:
   i=indices[at];add('warmup_failed' if warm else 'production_failed',failed[i],path,i,True)
 (out/'models.txt').write_text('\n'.join(lines)+'\n');save_json(out/'selection.json',selected)
 frequencies=np.loadtxt(src/'observations.txt')[:,0];hashes['observations.txt']=digest(src/'observations.txt')
 save_json(out/'protocol.json',dict(source=str(src),selection=selected,frequencies=frequencies.tolist(),scan_segments=[5000,10000],
   scan_domain='0.5*minimum Vs to halfspace Vs*(1-1e-8); trapped real phase velocities only',
   limitations='Independent unscaled P/SV plane waves; finite sign-change scans can miss tangencies/close roots, can be ill-conditioned; no root absence or continuous mode-tracking certificate.',
   source_hashes=hashes,binary_sha256=digest(exe),script_sha256=digest(Path(__file__)),new_sampling_calls=0))
 def run(job):
  segments,i,f=job;file=out/f'scan-{segments}-f{i}.txt'
  args=[str(exe),str(out/'models.txt'),'1.8','2.3',str(f),str(len(selected)),str(segments)]
  with file.open('w') as stdout,(out/f'scan-{segments}-f{i}.log').open('w') as stderr:
   code=subprocess.run(args,stdout=stdout,stderr=stderr,timeout=120).returncode
  assert code==0,(segments,i,code)
  return segments,i,[r.split() for r in file.read_text().splitlines() if r and not r.startswith('#')]
 jobs=[(s,i,float(f)) for s in [5000,10000] for i,f in enumerate(frequencies)]
 with ThreadPoolExecutor(max_workers=2) as pool:scans=list(pool.map(run,jobs))
 grouped={};issues=[];lookup={}
 for s,i,rows in scans:
  assert len(rows)==len(selected)
  for row in rows:
   ident=int(row[0]);meta=selected[ident];assert int(row[1])==meta['k'];roots=np.array(list(map(float,row[6:])))
   assert len(roots)==int(row[4]);valid=row[2]=='valid';error=abs(float(row[3])-roots[0]) if valid and len(roots) else None
   lookup[s,i,ident]=row
   key=f"{s}/{meta['kind']}";g=grouped.setdefault(key,dict(rows=0,backend_valid=0,unresolved_with_root=0,valid_without_root=0,scan_failures=0,maximum_first_root_error=0.))
   g['rows']+=1;g['backend_valid']+=int(valid);g['unresolved_with_root']+=int(not valid and len(roots)>0);g['valid_without_root']+=int(valid and not len(roots));g['scan_failures']+=int(row[5])
   if error is not None:g['maximum_first_root_error']=max(g['maximum_first_root_error'],float(error))
   if (valid and (error is None or error>1e-5)) or (not valid and len(roots)):
    issues.append(dict(segments=s,frequency_index=i,model=ident,backend_status=row[2],phase=float(row[3]),independent_roots=roots.tolist()))
 disagreements=[]
 for i in range(12):
  for ident in range(len(selected)):
   low,high=lookup[5000,i,ident],lookup[10000,i,ident]
   if low[4]!=high[4] or (int(low[4]) and not np.allclose(np.array(low[6:],float),np.array(high[6:],float),rtol=0,atol=1e-6)):
    disagreements.append(dict(frequency_index=i,model=ident,coarse_roots=low[6:],fine_roots=high[6:]))
 swaps=np.loadtxt(d/'swaps.txt');hot=swaps[swaps[:,2]==2];accepted=hot[hot[:,9]==1]
 ids=np.loadtxt(d/'replicas.txt',dtype=int)
 last=int(accepted[-1,0]);assert np.all(ids[last+1:,-1]==ids[last,-1])
 warmup=dict(hot_pair_accepted_iterations_zero_based=accepted[:,0].astype(int).tolist(),last_accepted_swap_iteration_zero_based=last,
  isolated_replica_id=int(ids[last,-1]),formal_hot_pair_accepts=int(hot[hot[:,1]==0,9].sum()),
  warmup_snapshot_iterations=hot[hot[:,1]==1,0].astype(int).tolist(),warmup_logL_T4=hot[hot[:,1]==1,6].tolist(),warmup_logL_T8=hot[hot[:,1]==1,7].tolist(),
  limitation='Warmup full accepted models were not retained; likelihood snapshots, replica identities, K and failed proposed models cannot reconstruct all accepted physical states.')
 for name,h in hashes.items():assert digest(src/name)==h
 report=dict(groups=grouped,disagreements_between_scan_resolutions=disagreements,suspected_root_mismatches=issues,warmup=warmup,
  model_count=len(selected),model_frequency_pairs_per_resolution=len(selected)*12,source_hashes_unchanged=True,
  interpretation='Agreement supports sampled trapped roots only; no global root completeness, physical absence, or multimode continuity claim.')
 save_json(out/'summary.json',report)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 axes[0].plot(warmup['warmup_snapshot_iterations'],warmup['warmup_logL_T4'],label='T=4',color='#0077bb');axes[0].plot(warmup['warmup_snapshot_iterations'],warmup['warmup_logL_T8'],label='T=8',color='#aa4499')
 axes[0].axvline(last,color='#777777',ls='--',label='Last accepted T4/T8 swap');axes[0].set(xlabel='Warmup iteration (zero-based)',ylabel='Raw log likelihood');axes[0].legend(fontsize=8)
 for ident in [next(x['id'] for x in selected if x['kind']=='trapped_retained'),next(x['id'] for x in selected if x['kind']=='control_hot')]:
  values=[lookup[10000,i,ident] for i in range(12)];label=selected[ident]['kind'];color='#aa4499' if label=='trapped_retained' else '#0077bb'
  axes[1].plot(frequencies,[float(r[3]) for r in values],color=color,label=label+' backend')
  axes[1].scatter(frequencies,[float(r[6]) if int(r[4]) else np.nan for r in values],facecolors='none',edgecolors=color,label=label+' independent first root')
 axes[1].set(xlabel='Frequency (Hz)',ylabel='Phase velocity (km/s)');axes[1].legend(fontsize=7)
 fig.suptitle('Bounded hot-chain audit; finite scans are not an absence certificate')
 fig.savefig(out/'audit.png',dpi=160);plt.close(fig)
 print(json.dumps(dict(groups=grouped,mismatches=len(issues),resolution_disagreements=len(disagreements),last_accepted_swap=last),indent=2))
if __name__=='__main__':main()
