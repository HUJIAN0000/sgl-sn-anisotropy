"""Entrypoint for revision production. Uses the pre-existing cosmo_gpu environment."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aa_revision.common import ROOT,write_json
from aa_revision.revision import config,prepare,fit_case,sample_case,parallel_jobs,mock_job,summarize_mocks

def main():
    p=argparse.ArgumentParser()
    p.add_argument('task',choices=['prepare','fit','sample','mocks','mock-report'])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--case',action='append')
    p.add_argument('--prior',action='append')
    p.add_argument('--repetitions',type=int)
    args=p.parse_args();cfg=config()
    if args.task=='prepare':prepare();return
    cases=[c for c in cfg['cases'] if not args.case or c['id'] in args.case]
    if args.task=='fit':
        results=parallel_jobs(fit_case,[c['id'] for c in cases],args.workers)
    elif args.task=='sample':
        jobs=[(c['id'],prior) for c in cases for prior in c.get('posterior_priors',[])
               if not args.prior or prior in args.prior]
        results=parallel_jobs(sample_case,jobs,args.workers)
    elif args.task=='mocks':
        jobs=[(t['id'],i) for t in cfg['mocks']['truth_sets']
              for i in range(args.repetitions or cfg['mocks']['repetitions'])]
        results=parallel_jobs(mock_job,jobs,args.workers)
        summarize_mocks()
    else:print(summarize_mocks());return
    write_json(ROOT/'results/production'/f'{args.task}_task_status.json',results)
    if not all(r['passed'] for r in results):
        raise SystemExit('Some jobs did not pass scientific diagnostics; see status JSON')

if __name__=='__main__':main()
