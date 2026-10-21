"""Read accepted ledgers, audit posterior support, and check actual posterior draws.

No fit, chain, original input, or frozen scientific module is modified.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys, json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aa_revision.common import ROOT, read_json, write_json, sha256
from aa_revision.revision import config, load_primary, engine_for
from aa_revision.model import Spec, loglike
from aa_revision.jeans import make_engine

OUT = ROOT/'results/final_audit'

def posterior_accuracy(job):
    cid, prior = job
    cfg = config(); case = next(c for c in cfg['cases'] if c['id']==cid)
    data = load_primary(); spec = Spec(case.get('framework','P2'),prior,cfg['domain'],case['tracer'])
    fast = engine_for(data,case)
    coarse = make_engine(data,case['jeans'],case['tracer'])
    finecfg = dict(case['jeans'], nrad=8192, extent=10000000, abel_nodes=256, angular_nodes=384)
    fine = make_engine(data,finecfg,case['tracer'])
    with np.load(ROOT/'results/production/chains'/cid/prior/'posterior_draws.npz') as z:
        samples = z['samples']
    # Fixed seed: independent random retained posterior draws, rather than MLE perturbations.
    indices = np.random.default_rng(20261021).choice(len(samples),24,replace=False)
    points=[]
    for ix in indices:
        theta=samples[ix]; p=spec.unpack(theta)
        gamma=p['gamma0']+p['gamma_z']*data.table.zl.to_numpy()
        lc=loglike(theta,data,spec,engine=coarse)
        lf=loglike(theta,data,spec,engine=fine)
        la=loglike(theta,data,spec,engine=fast)
        sc=coarse.predict(gamma,p['beta'],data.R); sf=fine.predict(gamma,p['beta'],data.R)
        points.append(dict(index=int(ix),interpolation_lnL_error=abs(la-lc),
            quadrature_lnL_error=abs(lf-lc),max_sigma_relative_change=float(np.max(abs(sf/sc-1)))))
    result=dict(case=cid,prior=prior,points=points,
        passed=all(p['interpolation_lnL_error']<.01 and p['quadrature_lnL_error']<.1
                   and p['max_sigma_relative_change']<5e-4 for p in points))
    write_json(OUT/f'accuracy_{cid}_{prior}.json',result)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=load_primary(); zp=float(np.median(data.table.zl)); xp=float(np.median(data.predictor))
    production=ROOT/'results/production'
    mainledger=read_json(production/'sample_buffered_task_status.json')
    deltaledger=read_json(production/'delta_parallel_task_status.json')
    assert len(mainledger)==16 and all(r['passed'] for r in mainledger)
    assert len(deltaledger)==8 and all(r['passed'] for r in deltaledger)
    parameters=[]; boundaries=[]; diagnostics=[]
    products=[('main',r['case'],r['prior'],production/'chains'/r['case']/r['prior']) for r in mainledger]
    products += [('delta',r['scenario'],r['prior'],Path(r['path'])) for r in deltaledger]
    figures=[]
    for kind,cid,prior,folder in products:
        summary=read_json(folder/'posterior_summary.json'); assert summary['production_eligible']
        spec=Spec(**summary['spec'])
        with np.load(folder/'posterior_draws.npz') as z:
            a=z['samples']; names=list(z['names'])
        assert np.isfinite(a).all() and names==spec.names
        gp=a[:,names.index('gamma0')]+zp*a[:,names.index('gamma_z')]
        if 'gamma_s' in names:
            gp=gp+xp*a[:,names.index('gamma_s')]
        allnames=names+['gamma_p']; values=np.column_stack([a,gp])
        q=np.quantile(values,[.16,.5,.84],axis=0)
        for i,name in enumerate(allnames):
            if name in summary['quantiles']:
                np.testing.assert_allclose(q[:,i],[summary['quantiles'][name][k] for k in ['q16','median','q84']],atol=1e-12,rtol=0)
            parameters.append(dict(kind=kind,case=cid,prior=prior,parameter=name,q16=q[0,i],median=q[1,i],q84=q[2,i]))
        for i,name in enumerate(names):
            lo,hi=spec.bounds[i]; margin=.01*(hi-lo)
            boundaries.append(dict(kind=kind,case=cid,prior=prior,parameter=name,
                fraction_in_lower_1pct=float(np.mean(a[:,i]<lo+margin)),
                fraction_in_upper_1pct=float(np.mean(a[:,i]>hi-margin)),
                observed_min=float(a[:,i].min()),observed_max=float(a[:,i].max())))
        diag=summary['diagnostics']
        diagnostics.append(dict(kind=kind,case=cid,prior=prior,steps=diag['total_steps'],burn=diag['burn'],
            max_rhat=float(np.max(diag['rank_split_rhat'])),min_approximate_ESS=float(np.min(diag['approximate_ESS'])),
            max_tau_change=float(np.max(diag['tau_relative_change'])),passed=diag['passed'],
            posterior_sha256=sha256(folder/'posterior_draws.npz')))
        fig,axes=plt.subplots(len(names),1,figsize=(8,1.45*len(names)),squeeze=False)
        files=sorted(folder.glob('ensemble_*.h5' if kind=='main' else 'latent_*.h5'))
        for ci,file in enumerate(files):
            with h5py.File(file,'r') as f:
                arr=f['mcmc/chain'] if kind=='main' else f['theta']
                stride=max(1,len(arr)//1600); t=np.arange(0,len(arr),stride)
                v=arr[::stride,0,:] if kind=='main' else arr[::stride,:]
            for i,name in enumerate(names):
                axes[i,0].plot(t,v[:,i],lw=.35,alpha=.8,label=f'Independent run {ci+1}')
                axes[i,0].axvline(diag['burn'],color='k',ls=':',lw=.7)
                axes[i,0].set_ylabel(name)
        axes[0,0].legend(fontsize=6,ncol=4)
        axes[0,0].set_title(f'{kind}: {cid} / {prior}',fontsize=10)
        axes[-1,0].set_xlabel('Step'); fig.tight_layout()
        target=OUT/f'trace_{kind}_{cid}_{prior}.png';fig.savefig(target,dpi=110);plt.close(fig)
        figures.append(str(target))
        print(f'audited {kind}/{cid}/{prior}',flush=True)
    pd.DataFrame(parameters).to_csv(OUT/'posterior_parameters.csv',index=False)
    pd.DataFrame(boundaries).to_csv(OUT/'posterior_boundary_fractions.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False)
    jobs=[(r['case'],r['prior']) for r in mainledger if r['case'] in ['hernquist','sersic_n4']]
    accuracy=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(posterior_accuracy,j) for j in jobs]):
            r=future.result();accuracy.append(r);print(f"accuracy {r['case']}/{r['prior']}: {r['passed']}",flush=True)
    passed=all(d['passed'] for d in diagnostics) and all(r['passed'] for r in accuracy)
    write_json(OUT/'audit_manifest.json',dict(passed=passed,main_posteriors=16,delta_posteriors=8,
        pivot_redshift=zp,pivot_predictor=xp,accuracy_points=24*len(jobs),
        accepted_delta_paths=deltaledger,accuracy=[dict(case=r['case'],prior=r['prior'],passed=r['passed']) for r in accuracy],
        visual_review='Pending; passing numerical checks is not visual verification',traces=figures,
        note='Boundary fractions are descriptive. No posterior is censored or prior changed.'))
    assert passed,'Inspect numerical audit failures before manuscript delivery'

if __name__=='__main__':
    main()
