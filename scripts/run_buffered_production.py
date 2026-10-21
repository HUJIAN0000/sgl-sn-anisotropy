"""Continue validated checkpoints after the original sampler writers stop."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import emcee
from functools import partial
from dataclasses import asdict
from scipy.optimize import minimize
from aa_revision.common import ROOT,read_json,write_json,code_hash,environment,fingerprint
from aa_revision.revision import config,load_primary,engine_for,parallel_jobs
from aa_revision.model import Spec,logposterior
from aa_revision.sampling import summarize_emcee,load_ensemble_chains
from aa_revision.diagnostics import rank_split_rhat
from buffered_sampling import buffered_append

def work(job):
    cid,prior=job;cfg=config();case=next(c for c in cfg['cases'] if c['id']==cid)
    d=load_primary();engine=engine_for(d,case);spec=Spec(case.get('framework','P2'),prior,cfg['domain'],case['tracer'])
    dest=ROOT/'results/production/chains'/cid/prior;dest.mkdir(parents=True,exist_ok=True)
    c=cfg['mcmc'];posterior=partial(logposterior,data=d,spec=spec,engine=engine)
    metadata=dict(data_digest=d.digest,spec=asdict(spec),code_hash=code_hash(),walkers=c['walkers'],
       ensembles=c['ensembles'],seed=cfg['seed'],burn=c['burn'],engine_config=case,
       diagnostics=c['diagnostics'],environment=environment())
    signature=fingerprint(metadata)
    if (dest/'run.json').exists():
        if read_json(dest/'run.json')['signature']!=signature:raise ValueError('Run fingerprint changed')
    else:write_json(dest/'run.json',dict(signature=signature,metadata=metadata))
    lp='flat' if prior=='gaussian' else prior
    mle=read_json(ROOT/'results/production/mle'/cid/(lp+'.json'))['result']
    opt=minimize(lambda x:-posterior(x) if np.isfinite(posterior(x)) else 1e100,
        mle['theta'],method='Nelder-Mead',bounds=spec.bounds,options={'maxiter':6000,'xatol':1e-7,'fatol':1e-7})
    center=opt.x if np.isfinite(posterior(opt.x)) else np.array(mle['theta'])
    backends=[];initial=[]
    for i in range(c['ensembles']):
        b=emcee.backends.HDFBackend(str(dest/f'ensemble_{i:02d}.h5'))
        if not b.initialized:b.reset(c['walkers'],len(spec.names))
        # A zero-iteration group can occur if interruption preceded its first save.
        if b.iteration==0:
            rng=np.random.default_rng(cfg['seed']+1009*i);positions=[]
            scales=np.array([.04 if n=='gamma0' else .15 if n!='f_int' else .012 for n in spec.names])
            for attempt in range(50000):
                x=center+rng.normal(size=len(spec.names))*scales
                if np.isfinite(posterior(x)):positions.append(x)
                if len(positions)==c['walkers']:break
            if len(positions)!=c['walkers']:raise ValueError('Could not initialise valid ensemble')
            state=emcee.State(np.asarray(positions),random_state=np.random.RandomState(cfg['seed']+1009*i).get_state())
        else:state=None
        backends.append(b);initial.append(state)
    current=max(b.iteration for b in backends)
    schedule=[n for n in c['extension_steps'] if n>=current]
    if not schedule:schedule=[current]
    for steps in schedule:
        for i,b in enumerate(backends):buffered_append(b,posterior,steps,initial_state=initial[i])
        result=summarize_emcee(dest,d,spec,dict(c,steps=steps))
        all_rhat=None
        if result['production_eligible']:
            arr=load_ensemble_chains(dest,c['ensembles'])[:,c['burn']:]
            all_rhat=max(rank_split_rhat(arr[:,:,w,j]) for w in range(arr.shape[2]) for j in range(arr.shape[3]))
            if all_rhat<1.01:
                write_json(dest/'buffered_extension.json',dict(steps=steps,all_walker_rhat=all_rhat,
                     method='Identical emcee moves and RNG; chunked journalled persistence'))
                return dict(case=cid,prior=prior,passed=True,steps=steps,all_walker_rhat=all_rhat)
        print('Needs extension',cid,prior,steps,result['diagnostics']['passed'],all_rhat,flush=True)
    return dict(case=cid,prior=prior,passed=False,steps=steps,all_walker_rhat=all_rhat)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    jobs=[(c['id'],p) for c in config()['cases'] for p in c.get('posterior_priors',[])]
    results=parallel_jobs(work,jobs,a.workers)
    write_json(ROOT/'results/production/sample_buffered_task_status.json',results)
    if not all(r['passed'] for r in results):raise SystemExit('Some runs still need attention')
