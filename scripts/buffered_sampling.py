"""Exact emcee transitions with chunked HDF persistence for further extensions.

This does not change the sampler target, moves, RNG, or stationary distribution.
Only use after the original writer has finished. An atomic rollback journal
protects the previous checkpoint from an interrupted multi-dataset append.
"""
from pathlib import Path
import sys,os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import emcee
from functools import partial
from aa_revision.common import ROOT,write_json,read_json,code_hash
from aa_revision.model import Spec,logposterior
from aa_revision.revision import config,load_primary,engine_for
from aa_revision.sampling import summarize_emcee


def restore_journal(backend,journal):
    if not journal.exists():return
    old=read_json(journal)
    with backend.open('a') as f:
        g=f[backend.name]
        # A fully committed chunk is already consistent; otherwise rollback.
        if int(g.attrs['iteration'])!=old['new_iteration']:
            g['accepted'][:]=old['accepted']
            for k in list(g.attrs):
                if k.startswith('random_state_'):del g.attrs[k]
            if old['random_state'] is not None:
                for i,v in enumerate(old['random_state']):g.attrs[f'random_state_{i}']=v
            g.attrs['iteration']=old['old_iteration']
            f.flush()
    journal.unlink()


def buffered_append(backend,logprob,steps,chunk=1000,initial_state=None):
    journal=Path(str(backend.filename)+'.buffered_journal.json')
    restore_journal(backend,journal)
    state=backend.get_last_sample() if backend.iteration else initial_state
    if state is None:raise ValueError('An initial state is required for an empty chain')
    walkers,ndim=state.coords.shape
    sampler=emcee.EnsembleSampler(walkers,ndim,logprob)
    while backend.iteration<steps:
        count=min(chunk,steps-backend.iteration)
        sampler.reset()
        state=sampler.run_mcmc(state,count,progress=False)
        chain=sampler.get_chain();lnp=sampler.get_log_prob()
        accepted=np.rint(sampler.acceptance_fraction*count)
        old=backend.iteration;new=old+count
        write_json(journal,dict(old_iteration=old,new_iteration=new,
             accepted=backend.accepted,random_state=backend.random_state))
        with backend.open('a') as f:
            g=f[backend.name]
            if g['chain'].shape[0]<new:
                g['chain'].resize(new,axis=0);g['log_prob'].resize(new,axis=0)
            g['chain'][old:new]=chain;g['log_prob'][old:new]=lnp
            g['accepted'][:]+=accepted
            for i,v in enumerate(state.random_state):g.attrs[f'random_state_{i}']=v
            f.flush()
            g.attrs['iteration']=new
            f.flush()
        journal.unlink()
        print('buffered',Path(backend.filename).parent.name,Path(backend.filename).name,new,flush=True)


def extend(case_id,prior,steps):
    cfg=config();case=next(c for c in cfg['cases'] if c['id']==case_id)
    dest=ROOT/'results/production/chains'/case_id/prior
    metadata=read_json(dest/'run.json')['metadata']
    if metadata['code_hash']!=code_hash():raise ValueError('Scientific code changed')
    d=load_primary()
    if metadata['data_digest']!=d.digest:raise ValueError('Data changed')
    spec=Spec('P2',prior,cfg['domain'],case['tracer'])
    if metadata['spec']!={'framework':spec.framework,'beta_prior':spec.beta_prior,'domain':spec.domain,'tracer':spec.tracer}:
        raise ValueError('Model changed')
    posterior=partial(logposterior,data=d,spec=spec,engine=engine_for(d,case))
    for i in range(cfg['mcmc']['ensembles']):
        b=emcee.backends.HDFBackend(str(dest/f'ensemble_{i:02d}.h5'))
        buffered_append(b,posterior,steps)
    result=summarize_emcee(dest,d,spec,dict(cfg['mcmc'],steps=steps))
    write_json(dest/'buffered_extension.json',dict(target_steps=steps,method='identical emcee moves; chunked persistence',
        scientific_code_hash=code_hash(),diagnostics_passed=result['production_eligible']))
    return dict(case=case_id,prior=prior,steps=steps,passed=result['production_eligible'])


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--prior',required=True)
    p.add_argument('--steps',type=int,required=True);a=p.parse_args()
    print(extend(a.case,a.prior,a.steps))
