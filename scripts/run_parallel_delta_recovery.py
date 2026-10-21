"""Schedule one independent latent chain per process, preserving serial checkpoints.

Only reads a serial product after its four chains have closed at the declared
80k limit, or accepts a qualified finished serial product. Active files are
never copied, changed, or terminated. Extensions use separate output paths.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import time
import shutil
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import traceback
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import h5py
import psutil
from aa_revision.common import ROOT,read_json,write_json,sha256,code_hash
from aa_revision.revision import config,load_primary
from aa_revision.model import Spec
from aa_revision.delta import run_latent,latent_to_delta,slope_errors
from run_new_delta import SCENARIOS
from delta_parallel_runtime import worker,settings


SCHEDULE = [120000,200000,320000]


def checkpoint_state(path, data, spec, scenario, error_table):
    with h5py.File(path,'r') as f:
        state=json.loads(f.attrs['state'])
        n=int(state['iteration'])
        assert all(len(f[name])==n for name in ['theta','delta','b_delta','log_target'])
        np.testing.assert_array_equal(f['theta'][n-1],state['theta'])
        delta,b=latent_to_delta(np.asarray(state['z']),data,slope_errors(data,scenario,error_table),scenario.get('coherent_sigma',0))
        np.testing.assert_array_equal(f['delta'][n-1],delta)
        assert f['b_delta'][n-1]==b
    return n,state


def prefix_hashes(path,n):
    result={}
    with h5py.File(path,'r') as f:
        for name in ['theta','delta','b_delta','log_target']:
            h=hashlib.sha256()
            for start in range(0,n,10000):
                h.update(np.ascontiguousarray(f[name][start:min(n,start+10000)]).tobytes())
            result[name]=h.hexdigest()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    validation=read_json(ROOT/'provenance/parallel_performance_validation.json')
    assert validation['passed'] and validation['all_four_trajectory_states_exact']
    data=load_primary(); base_hash=code_hash()
    root=ROOT/'results/parallel_delta_work'
    logstate=ROOT/'parallel_delta_status.json'
    lock=ROOT/'parallel_delta_recovery.lock'
    with lock.open('x',encoding='utf-8') as f:
        f.write(str(os.getpid()))
    protocol=dict(extension_steps=SCHEDULE,workers=args.workers,thresholds=config()['mcmc']['diagnostics'],
        method='One independent saved chain per process, same RNG/adaptation/target; prepared arithmetic verified exact',
        immutable_originals='All original delta products retained; active serial writers continue without interruption',
        validation='provenance/parallel_performance_validation.json',base_science_code_hash=base_hash)
    write_json(ROOT/'protocol/parallel_delta_extension_20261021.json',protocol)
    groups={}
    for scenario in SCENARIOS:
        for prior in ['flat','fixed0']:
            key=scenario['name']+'/'+prior
            groups[key]=dict(scenario=scenario,prior=prior,status='waiting_for_serial',finished=set(),round=0)
    futures={};previous_cpu={};previous_time=time.perf_counter();last_progress=0
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            while True:
                assert code_hash()==base_hash,'Base science implementation changed'
                for key,group in groups.items():
                    if group['status']!='waiting_for_serial':
                        continue
                    scenario,prior=group['scenario'],group['prior']
                    source=ROOT/'results/production/delta'/scenario['name']/prior
                    summary_file=source/'posterior_summary.json'
                    if not summary_file.exists():
                        continue
                    summary=read_json(summary_file)
                    if summary['production_eligible']:
                        group.update(status='accepted',path=str(source),steps=summary['diagnostics']['total_steps'],source='original_serial')
                        print('Accepted qualified serial product',key,group['steps'],flush=True)
                        continue
                    if summary['diagnostics']['total_steps']!=80000:
                        continue
                    spec=Spec('P2',prior,config()['domain'],'powerlaw')
                    err=ROOT/'data/derived/delta_1percent_statistical_upper_bound.csv' if 'relative_sigma_bound' in scenario else None
                    meta=read_json(source/'run.json')['metadata']
                    assert meta['code_hash']==base_hash and meta['data_digest']==data.digest
                    original=[]
                    try:
                        for i in range(4):
                            src=source/f'latent_{i:02d}.h5'
                            n,_=checkpoint_state(src,data,spec,scenario,err)
                            if n!=80000:
                                raise OSError('Serial group is not at a closed, complete common checkpoint')
                            original.append((src,n))
                    except (OSError,BlockingIOError):
                        continue
                    for i,(src,n) in enumerate(original):
                        dest=root/scenario['name']/prior/f'chain_{i:02d}'
                        dest.mkdir(parents=True,exist_ok=True)
                        target=dest/'latent_00.h5'
                        if target.exists():
                            raise FileExistsError(f'Parallel work exists: {target}; inspect before resuming')
                        before=sha256(src)
                        shutil.copy2(src,target)
                        assert sha256(src)==before==sha256(target)
                        checkpoint_state(target,data,spec,scenario,err)
                        origin=dict(original_path=str(src),source_sha256=before,steps=n,
                            chain_index=i,original_run_signature=read_json(source/'run.json')['signature'],
                            dataset_prefix_sha256=prefix_hashes(src,n))
                        write_json(dest/'parent_checkpoint.json',origin)
                        item=(scenario,prior,i,SCHEDULE[0],str(dest))
                        futures[pool.submit(worker,item)]=(key,i)
                    group.update(status='sampling',steps=SCHEDULE[0],finished=set(),source='parallel_extension')
                    print('Dispatched four independent chains',key,SCHEDULE[0],flush=True)
                if futures:
                    done,_=wait(futures,timeout=2,return_when=FIRST_COMPLETED)
                    for future in done:
                        key,index=futures.pop(future)
                        result=future.result()
                        group=groups[key]
                        child=Path(result['path']).parent
                        origin=read_json(child/'parent_checkpoint.json')
                        assert prefix_hashes(result['path'],origin['steps'])==origin['dataset_prefix_sha256']
                        assert sha256(origin['original_path'])==origin['source_sha256']
                        group['finished'].add(index)
                        print('Chain complete',key,index,result['steps'],f"{result['elapsed_seconds']:.1f}s",flush=True)
                        if len(group['finished'])!=4:
                            continue
                        scenario,prior=group['scenario'],group['prior']
                        combined=ROOT/'results/production/delta_parallel'/scenario['name']/prior
                        combined.mkdir(parents=True,exist_ok=True)
                        for i in range(4):
                            src=root/scenario['name']/prior/f'chain_{i:02d}/latent_00.h5'
                            target=combined/f'latent_{i:02d}.h5'
                            temporary=combined/f'latent_{i:02d}.merge.tmp'
                            shutil.copy2(src,temporary)
                            os.replace(temporary,target)
                        err=ROOT/'data/derived/delta_1percent_statistical_upper_bound.csv' if 'relative_sigma_bound' in scenario else None
                        spec=Spec('P2',prior,config()['domain'],'powerlaw')
                        mle=read_json(ROOT/'results/production/mle/powerlaw'/f'{prior}.json')['result']
                        print('Combined diagnostics',key,group['steps'],flush=True)
                        summary=run_latent(data,spec,mle,combined,settings(group['steps'],chains=4),scenario,
                            config()['seed']+711,resume=True,error_table=err)
                        write_json(combined/'parallel_extension.json',dict(protocol=protocol,
                            steps=group['steps'],passed=summary['production_eligible'],
                            all_original_dataset_prefixes_unchanged=True,
                            all_original_files_unchanged=True))
                        if summary['production_eligible']:
                            group.update(status='accepted',path=str(combined))
                            print('ACCEPTED',key,group['steps'],flush=True)
                        elif group['round']+1<len(SCHEDULE):
                            group['round']+=1
                            group['steps']=SCHEDULE[group['round']]
                            group['finished']=set()
                            for i in range(4):
                                dest=root/scenario['name']/prior/f'chain_{i:02d}'
                                futures[pool.submit(worker,(scenario,prior,i,group['steps'],str(dest)))]=(key,i)
                            print('Needs more retained steps under unchanged thresholds',key,group['steps'],flush=True)
                        else:
                            group.update(status='needs_attention',path=str(combined))
                else:
                    time.sleep(2)
                now=time.perf_counter()
                if now-last_progress>=5:
                    processes=psutil.Process(os.getpid()).children(recursive=True)
                    current={}
                    for p in processes:
                        try:
                            t=p.cpu_times();current[p.pid]=t.user+t.system
                        except psutil.NoSuchProcess:
                            pass
                    delta_cpu=sum(max(0,v-previous_cpu.get(k,v)) for k,v in current.items())
                    cpu_percent=100*delta_cpu/max(now-previous_time,.001)/(os.cpu_count() or 1)
                    public={k:{field:value for field,value in g.items() if field not in ['finished','scenario']} for k,g in groups.items()}
                    status=dict(status='running',supervisor_PID=os.getpid(),workers=args.workers,
                        active_pool_processes=len(current),owned_pool_CPU_percent=cpu_percent,
                        accepted=sum(g['status']=='accepted' for g in groups.values()),total=len(groups),
                        active_chain_tasks=len(futures),groups=public,protocol=protocol)
                    write_json(logstate,status)
                    print('POOL STATUS',status['accepted'],'/',len(groups),'accepted,',len(futures),'tasks,',f'{cpu_percent:.1f}% whole-machine CPU',flush=True)
                    previous_time=now;previous_cpu=current;last_progress=now
                if all(g['status'] in ['accepted','needs_attention'] for g in groups.values()):
                    break
        records=[dict(scenario=g['scenario']['name'],prior=g['prior'],passed=g['status']=='accepted',
                      steps=g.get('steps'),path=g.get('path'),source=g.get('source')) for g in groups.values()]
        write_json(ROOT/'results/production/delta_parallel_task_status.json',records)
        write_json(logstate,dict(status='completed' if all(r['passed'] for r in records) else 'needs_attention',
            supervisor_PID=os.getpid(),workers=args.workers,products=records,protocol=protocol))
        if not all(r['passed'] for r in records):
            raise RuntimeError('Some chains still require convergence work; thresholds were not relaxed')
    except Exception as exc:
        write_json(ROOT/'parallel_delta_error.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
