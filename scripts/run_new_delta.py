"""Assumed delta errors; all chains and posterior products are new."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aa_revision.common import ROOT, read_json, write_json
from aa_revision.model import Spec
from aa_revision.revision import config, load_primary, parallel_jobs
from aa_revision.delta import run_latent

SCENARIOS = [dict(name='independent005', sigma=.05, coherent_sigma=0),
    dict(name='independent010', sigma=.10, coherent_sigma=0),
    dict(name='independent005_coherent005', sigma=.05, coherent_sigma=.05),
    dict(name='statistical_1pct_upper_bound', relative_sigma_bound=.01, coherent_sigma=0,
         interpretation='Conservative upper-bound scenario based on Chen2019 reported relative statistical errors <1%; not measured individual errors')]


def worker(item):
    scenario, prior = item
    cfg = config()
    data = load_primary()
    spec = Spec('P2', prior, cfg['domain'], 'powerlaw')
    mle = read_json(ROOT / 'results/production/mle/powerlaw' / f'{prior}.json')['result']
    dest = ROOT / 'results/production/delta' / scenario['name'] / prior
    settings = dict(chains=4, burn=5000, block_size=32, global_updates=3, checkpoint=1000,
                    diagnostics=cfg['mcmc']['diagnostics'])
    error_table = ROOT / 'data/derived/delta_1percent_statistical_upper_bound.csv' if 'relative_sigma_bound' in scenario else None
    if error_table is not None and not error_table.exists():
        raise FileNotFoundError('Run audit_chen2019.py to record the source and construct the labelled upper-bound input first')
    for steps in [30000, 50000, 80000]:
        result = run_latent(data, spec, mle, dest, dict(settings, steps=steps), scenario,
                           cfg['seed']+711, resume=(dest / 'run.json').exists(), error_table=error_table)
        if result['production_eligible']:
            break
    return dict(scenario=scenario['name'], prior=prior, passed=result['production_eligible'], steps=steps)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    result = parallel_jobs(worker, [(s, p) for s in SCENARIOS for p in ['flat', 'fixed0']], a.workers)
    write_json(ROOT / 'results/production/delta_task_status.json', result)
    if not all(r['passed'] for r in result):
        raise SystemExit('Some latent-delta chains require further convergence work')


if __name__ == '__main__':
    main()
