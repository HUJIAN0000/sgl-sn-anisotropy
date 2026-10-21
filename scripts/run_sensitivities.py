"""New-data sample/photometry/distance checks; comparisons are within scenarios only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.integrate import quad
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aa_revision.common import ROOT, write_json, read_json, fingerprint
from aa_revision.revision import config, load_primary, engine_for, cached_mle, parallel_jobs
from aa_revision.model import Spec
from aa_revision.fast_jeans import ChebyshevJeans


def scenarios():
    data = load_primary()
    frac = np.sqrt(np.diag(data.cov_R))/data.R
    result = [dict(id='R_error_le020', indices=np.flatnonzero(frac <= .2).tolist()),
        dict(id='exclude_BELLS_family', indices=np.flatnonzero(~data.table['Survey name'].str.contains('BELLS')).tolist()),
        dict(id='exclude_S4TM', indices=np.flatnonzero(data.table['Survey name'] != 'S4TM').tolist())]
    for om in [.25, .30, .35]:
        result.append(dict(id=f'fiducial_mismatch_Om{om:.2f}', omega=om))
    result.append(dict(id='fiducial_mismatch_heliocentric_Om0.30', omega=.30, helio=True))
    for delta in [-.1, -.05, .05, .1]:
        result.append(dict(id=f'delta_{delta:+.2f}', delta=delta, tracers=['powerlaw']))
    return result


def transfer(data, omega, helio=False):
    table = data.table
    chi = lambda z: quad(lambda x: (omega*(1+x)**3+1-omega)**(-.5), 0, float(z))[0]
    za = table.sn_lens_zHD.to_numpy(float)
    zb = table.sn_source_zHD.to_numpy(float)
    ca = np.array([chi(z) for z in za])
    cb = np.array([chi(z) for z in zb])
    cl = np.array([chi(z) for z in table.zl])
    cs = np.array([chi(z) for z in table.zs])
    ra = table.sn_lens_zHEL.to_numpy(float) if helio else za
    rb = table.sn_source_zHEL.to_numpy(float) if helio else zb
    qmatched = ca/cb*(1+ra)/(1+rb)*(1+table.zs.to_numpy())/(1+table.zl.to_numpy())
    scale = (cl/cs)/qmatched
    R = 1-(1-data.R)*scale
    assert (R > .001).all()
    return replace(data, R=R, cov_R=scale[:, None]*data.cov_R*scale[None, :],
        digest=fingerprint(dict(parent=data.digest, operation='fiducial redshift transfer', omega=omega, helio=helio)))


def worker(job):
    scenario, cid = job
    cfg = config()
    base = load_primary()
    data = base
    case = next(c for c in cfg['cases'] if c['id'] == cid)
    if 'indices' in scenario:
        data = base.subset(scenario['indices'])
    if 'omega' in scenario:
        data = transfer(base, scenario['omega'], scenario.get('helio', False))
    if 'delta' in scenario:
        table = base.table.copy()
        table.delta += scenario['delta']
        data = replace(base, table=table, digest=fingerprint(dict(parent=base.digest, delta_shift=scenario['delta'])))
    engine = engine_for(base, case)
    if engine is not None and 'indices' in scenario:
        engine = ChebyshevJeans(engine.coefficients[scenario['indices']], engine.metadata)
    dest = ROOT / 'results/production/sensitivity' / scenario['id'] / cid
    fits = []
    for prior in ['flat', 'fixed0', 'fixed018']:
        spec = Spec('P2', prior, cfg['domain'], case['tracer'])
        fits.append(cached_mle(data, spec, cfg['optimizer'], dest / f'{prior}.json',
                    engine=engine, extra=dict(scenario=scenario, case=case), seed=cfg['seed']))
    passed = all(f['optimizer_agreement_checked'] for f in fits)
    passed = passed and fits[0]['lnL'] >= max(f['lnL'] for f in fits[1:])-.01
    row = dict(scenario=scenario['id'], case=cid, N=data.n, passed=bool(passed),
        delta_BIC_0=fits[1]['BIC']-fits[0]['BIC'], delta_BIC_018=fits[2]['BIC']-fits[0]['BIC'],
        data_digest=data.digest, details=scenario,
        interpretation='Compare beta hypotheses only within this row; no absolute BIC comparison across samples')
    write_json(dest / 'summary.json', row)
    return row


def nonlinear_diagnostic():
    data = load_primary()
    ll, ss, ls = [np.loadtxt(ROOT / 'data/primary' / f'cov_{r}.txt') for r in ['lens', 'source', 'cross']]
    Cdm = ll+ss-ls-ls.T
    rng = np.random.default_rng(config()['seed'])
    center = data.table.sn_lens_m_b_corr.to_numpy(float)-data.table.sn_source_m_b_corr.to_numpy(float)
    dm = center[:, None]+np.linalg.cholesky(Cdm)@rng.normal(size=(data.n, 50000))
    R = 1-10**(.2*dm)*((1+data.table.zs)/(1+data.table.zl)).to_numpy()[:, None]
    rows = []
    for i in range(data.n):
        good = R[i] > .001
        q = np.quantile(np.sqrt(data.R[i]/R[i, good]), [.16, .5, .84])
        rows.append(dict(lens=data.table['Lens name'].iloc[i], R=data.R[i],
            relative_R_error=np.sqrt(data.cov_R[i, i])/data.R[i],
            fraction_R_le0001=float((~good).mean()), conditional_sigma_ratio_q16=q[0],
            conditional_sigma_ratio_median=q[1], conditional_sigma_ratio_q84=q[2]))
    out = ROOT / 'results/production/sensitivity'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'nonlinear_distance_diagnostic.csv', index=False)
    write_json(out / 'nonlinear_distance_diagnostic.json', dict(draws=50000,
        seed=config()['seed'], interpretation='Diagnostic uncertainty draws about observations, not a physical latent-distance likelihood', rows=rows))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    nonlinear_diagnostic()
    jobs = [(s, c) for s in scenarios() for c in s.get('tracers', ['powerlaw', 'hernquist', 'sersic_n4'])]
    results = parallel_jobs(worker, jobs, a.workers)
    write_json(ROOT / 'results/production/sensitivity_task_status.json', results)
    out = ROOT / 'results/production/report'
    pd.DataFrame([{k: r[k] for k in ['scenario', 'case', 'N', 'delta_BIC_0', 'delta_BIC_018', 'passed']} for r in results]).to_csv(out / 'new_sensitivity_comparisons.csv', index=False)
    if not all(r['passed'] for r in results):
        raise SystemExit('Some sensitivity fits need attention; do not accept their criterion differences')


if __name__ == '__main__':
    main()
