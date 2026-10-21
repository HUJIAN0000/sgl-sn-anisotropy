"""Revision v3: frozen inputs, per-case inference, recoveries and validation."""
from pathlib import Path
from dataclasses import replace, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .common import ROOT, write_json, read_json, sha256, code_hash, fingerprint, environment
from .model import load_data, Spec, loglike, mean_and_cov, gamma_values
from .jeans import make_engine
from .fast_jeans import build_accelerator
from .fitting import fit_mle
from .sampling import run_emcee
from .experiments import wilson


def config():
    return read_json(ROOT/'configs/revision_v3.json')


def load_primary():
    cfg=config()
    return load_data(ROOT/cfg['primary_catalogue'])


def engine_for(data, case):
    if case['tracer']=='powerlaw':
        return None
    return build_accelerator(data, case['jeans'],case['tracer'],ROOT/'results/engine_cache')


def prepare():
    cfg=config(); d=load_primary()
    files=list((ROOT/'data').glob('**/*'))
    inputs={str(p.relative_to(ROOT)):sha256(p) for p in files if p.is_file()}
    source={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'aa_revision').glob('*.py')}
    write_json(ROOT/'provenance/run_manifest.json',dict(config=cfg,inputs=inputs,
        source=source,source_digest=code_hash(),environment=environment(),N=d.n))
    frozen=ROOT/'results/production/catalogues/primary'
    frozen.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/cfg['primary_catalogue']).iterdir():
        if p.is_file():
            target=frozen/p.name
            if target.exists() and sha256(target)!=sha256(p):
                raise ValueError('Frozen input changed')
            if not target.exists(): shutil.copy2(p,target)
    write_json(frozen/'manifest.json',dict(file_sha256={p.name:sha256(p) for p in frozen.iterdir()
        if p.is_file() and p.name!='manifest.json'},source='fresh_CosmoMatcher_1.4.3_raw_rebuild',N=d.n))
    expected=[]
    for c in cfg['cases']:
        for prior in ['flat','fixed0','fixed018']:
            expected.append(dict(kind='mle',case=c['id'],prior=prior))
        for prior in c.get('posterior_priors',[]):
            expected.append(dict(kind='posterior',case=c['id'],prior=prior))
    write_json(ROOT/'protocol/expected_products.json',expected)
    print('Frozen',d.n,'systems',d.digest,flush=True)


def cached_mle(data,spec,options,path,engine=None,extra=None,seed=61321):
    sig=fingerprint(dict(data=data.digest,spec=asdict(spec),options=options,
                         extra=extra,seed=seed,source=code_hash()))
    path=Path(path)
    if path.exists():
        old=read_json(path)
        if old['signature']!=sig:
            raise ValueError(f'Fit signature mismatch: {path}')
        return old['result']
    result=fit_mle(data,spec,options,seed=seed,engine=engine)
    write_json(path,dict(signature=sig,result=result,extra=extra))
    return result


def validate_case(data,case,results,engine):
    if engine is None:
        return dict(passed=True,method='analytic',interpolation_passed=True)
    coarse=make_engine(data,case['jeans'],case['tracer'])
    finecfg=dict(case['jeans'],nrad=2*case['jeans']['nrad'],
                 extent=10*case['jeans']['extent'],
                 abel_nodes=2*case['jeans']['abel_nodes'],
                 angular_nodes=2*case['jeans']['angular_nodes'])
    fine=make_engine(data,finecfg,case['tracer'])
    points=[]
    rng=np.random.default_rng(71321)
    spec=Spec(case.get('framework','P2'),'flat','convergent_jeans',case['tracer'])
    center=np.array(results['flat']['theta'])
    candidates=[(prior,np.array(r['theta'])) for prior,r in results.items()]
    for i in range(50):
        for attempt in range(1000):
            t=center+rng.normal(size=len(center))*np.array([.06,.18,.16,.008][:len(center)])
            if np.isfinite(loglike(t,data,spec,engine=engine)):
                candidates.append(('flat',t)); break
    for prior,t in candidates:
        sp=replace(spec,beta_prior=prior)
        lc=loglike(t,data,sp,engine=coarse)
        lf=loglike(t,data,sp,engine=fine)
        la=loglike(t,data,sp,engine=engine)
        p=sp.unpack(t);g=gamma_values(data,p)
        sc=coarse.predict(g,p['beta'],data.R); sf=fine.predict(g,p['beta'],data.R)
        points.append(dict(prior=prior,theta=t.tolist(),lnL_coarse=lc,lnL_fine=lf,
             interpolation_lnL_error=abs(la-lc),quadrature_lnL_error=abs(lf-lc),
             max_sigma_relative_change=float(np.max(abs(sf/sc-1)))))
    result=dict(passed=bool(all(p['quadrature_lnL_error']<.1 and
         p['interpolation_lnL_error']<.01 and p['max_sigma_relative_change']<5e-4 for p in points)),
         interpolation_passed=bool(engine.metadata['validation']['passed']),
         maximum_quadrature_lnL_error=max(p['quadrature_lnL_error'] for p in points),
         maximum_interpolation_lnL_error=max(p['interpolation_lnL_error'] for p in points),
         points=points,coarse=case['jeans'],fine=finecfg)
    return result


def fit_case(case_id):
    cfg=config();case=next(c for c in cfg['cases'] if c['id']==case_id)
    data=load_primary(); engine=engine_for(data,case)
    out=ROOT/'results/production/mle'/case_id
    results={}
    for prior in ['flat','fixed0','fixed018']:
        spec=Spec(case.get('framework','P2'),prior,cfg['domain'],case['tracer'])
        results[prior]=cached_mle(data,spec,cfg['optimizer'],out/(prior+'.json'),
                                  engine,case)
        print(case_id,prior,'lnL=',results[prior]['lnL'],flush=True)
    nested=results['flat']['lnL']>=max(results[x]['lnL'] for x in ['fixed0','fixed018'])-.01
    accuracy=validate_case(data,case,results,engine)
    accepted=all(r['optimizer_agreement_checked'] for r in results.values()) and nested and accuracy['passed']
    write_json(out/'validation.json',dict(passed=accepted,nested_likelihood_check=nested,
           quadrature=accuracy,data_digest=data.digest,case=case))
    return dict(case=case_id,passed=accepted)


def sample_case(job):
    case_id,prior=job
    cfg=config();case=next(c for c in cfg['cases'] if c['id']==case_id)
    out=ROOT/'results/production'
    check=read_json(out/'mle'/case_id/'validation.json')
    if not check['passed']:
        raise ValueError(f'MLE/numerical validation not accepted: {case_id}')
    mle=read_json(out/'mle'/case_id/('flat.json' if prior=='gaussian' else prior+'.json'))['result']
    data=load_primary();engine=engine_for(data,case)
    spec=Spec(case.get('framework','P2'),prior,cfg['domain'],case['tracer'])
    dest=out/'chains'/case_id/prior
    settings=cfg['mcmc']
    # Predeclared extension schedule; do not loosen diagnostics after failures.
    for steps in settings['extension_steps']:
        result=run_emcee(data,spec,mle,dest,dict(settings,steps=steps),cfg['seed'],
                          resume=(dest/'run.json').exists(),engine=engine,engine_config=case)
        if result['production_eligible']:
            return dict(case=case_id,prior=prior,passed=True,steps=steps)
    return dict(case=case_id,prior=prior,passed=False,steps=steps)


def parallel_jobs(function,jobs,workers):
    if workers==1:
        return [function(job) for job in jobs]
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future={pool.submit(function,job):job for job in jobs}
        for f in as_completed(future):
            result=f.result();results.append(result);print('completed',result,flush=True)
    return results


def mock_job(job):
    truth_id,repetition=job
    cfg=config();truth=next(t for t in cfg['mocks']['truth_sets'] if t['id']==truth_id)
    cases={c['id']:c for c in cfg['cases']}
    tc=cases[truth['case']];data=load_primary()
    generation_mle=read_json(ROOT/'results/production/mle'/truth['case']/(truth['prior']+'.json'))['result']
    tp=Spec('P2',truth['prior'],cfg['domain'],tc['tracer']).unpack(generation_mle['theta'])
    ts=Spec('P2','flat',cfg['domain'],tc['tracer'])
    theta=np.array([tp[n] for n in ts.names])
    te=engine_for(data,tc)
    mean,_=mean_and_cov(theta,data,ts,engine=te)
    base=replace(data,anchor=mean.copy())
    mean,cov=mean_and_cov(theta,base,ts,engine=te)
    truth_index=next(i for i,t in enumerate(cfg['mocks']['truth_sets']) if t['id']==truth_id)
    seed=np.random.SeedSequence([cfg['seed'],truth_index,repetition,991])
    rng=np.random.default_rng(seed)
    observed=mean+np.linalg.cholesky(cov)@rng.normal(size=data.n)
    data=base.observed(observed)
    path=ROOT/'results/production/mocks'/truth_id/f'{repetition:05d}.json'
    sig=fingerprint(dict(truth=truth,mle=generation_mle['theta'],data=data.digest,
           observed=observed,source=code_hash(),optimizer=cfg['mock_optimizer'],seed=seed.entropy))
    if path.exists():
        old=read_json(path)
        if old['signature']!=sig:raise ValueError('Mock signature changed')
        return dict(truth=truth_id,repetition=repetition,passed=old['passed'])
    fits=[]
    for case_id in truth['fit_cases']:
        fc=cases[case_id];engine=engine_for(data,fc)
        for prior in ['flat','fixed0','fixed018']:
            spec=Spec('P2',prior,cfg['domain'],fc['tracer'])
            fit=fit_mle(data,spec,cfg['mock_optimizer'],seed=61321+repetition,
                        engine=engine)
            fit['case']=case_id;fits.append(fit)
    passed=all(f['optimizer_agreement_checked'] for f in fits)
    write_json(path,dict(signature=sig,truth=truth,true_parameters=tp,
       generation_mle=generation_mle,seed=seed.entropy,repetition=repetition,fits=fits,
       passed=passed,noise='Conditional Gaussian; true-sigma anchor retained identically in generation and fitting'))
    return dict(truth=truth_id,repetition=repetition,passed=passed)


def summarize_mocks():
    cfg=config();rows=[];summary=[];pivot_z=float(np.median(load_primary().table.zl))
    for truth in cfg['mocks']['truth_sets']:
        files=list((ROOT/'results/production/mocks'/truth['id']).glob('*.json'))
        for path in files:
            item=read_json(path);tp=item['true_parameters']
            for f in item['fits']:
                p=Spec(**f['spec']).unpack(f['theta'])
                rows.append(dict(truth=truth['id'],repetition=item['repetition'],case=f['case'],
                    prior=f['spec']['beta_prior'],BIC=f['BIC'],lnL=f['lnL'],passed=item['passed'],
                    beta=p['beta'],beta_bias=p['beta']-tp['beta'],
                    gamma_p_bias=p['gamma0']+pivot_z*p['gamma_z']-tp['gamma0']-pivot_z*tp['gamma_z']))
    df=pd.DataFrame(rows)
    if df.empty:return []
    for (truth,case),group in df.groupby(['truth','case']):
        valid=group[group.passed]
        free=valid[valid.prior=='flat'].set_index('repetition')
        for fixed in ['fixed0','fixed018']:
            null=valid[valid.prior==fixed].set_index('repetition')
            ids=free.index.intersection(null.index)
            diff=null.loc[ids].BIC-free.loc[ids].BIC
            n=len(diff)
            summary.append(dict(truth=truth,case=case,fixed=fixed,N_successful=n,
                N_planned=cfg['mocks']['repetitions'],mean_delta_BIC=float(diff.mean()) if n else None,
                median_delta_BIC=float(diff.median()) if n else None,
                mean_beta_bias=float(free.loc[ids].beta_bias.mean()) if n else None,
                mean_gamma_p_bias=float(free.loc[ids].gamma_p_bias.mean()) if n else None,
                probability_gt6=float((diff>6).mean()) if n else None,
                wilson95=wilson(int((diff>6).sum()),n),
                complete=bool(n==cfg['mocks']['repetitions'])))
    folder=ROOT/'results/production/report';folder.mkdir(parents=True,exist_ok=True)
    df.to_csv(folder/'mock_fits.csv',index=False)
    write_json(folder/'mock_summary.json',summary)
    return summary
