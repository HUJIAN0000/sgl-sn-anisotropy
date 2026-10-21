"""Exact target/trajectory checks and measured throughput before changing scheduling."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import time
import json
import shutil
from dataclasses import replace
from contextlib import redirect_stdout
from io import StringIO
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import h5py
import psutil
from aa_revision.common import ROOT, read_json, write_json
from aa_revision.revision import load_primary
from aa_revision.model import Spec, logposterior
from aa_revision.delta import run_latent, latent_to_delta
from prepared_delta_likelihood import PreparedDeltaLikelihood
from delta_parallel_runtime import chain_function


def benchmark_worker(item):
    index, calls = item
    data = load_primary()
    spec = Spec('P2', 'flat')
    fast = PreparedDeltaLikelihood(data, spec)
    theta = np.array([2.1, -.7, -.3, .08])
    sigma = np.full(data.n, .05)
    rng = np.random.default_rng(781+index)
    zz = rng.normal(size=(32, data.n))
    start = time.perf_counter()
    accumulator = 0.
    for j in range(calls):
        d, _ = latent_to_delta(zz[j%32], data, sigma, 0.)
        accumulator += fast(theta, data, spec, d)
    return dict(elapsed=time.perf_counter()-start, calls=calls, value=accumulator)


def main():
    out = ROOT/'validation/performance_20261021'/f'attempt_{time.time_ns()}'
    out.mkdir(parents=True, exist_ok=True)
    data = load_primary()
    rng = np.random.default_rng(20261021)
    errors = []
    for prior in ['flat', 'fixed0']:
        spec = Spec('P2', prior)
        fast = PreparedDeltaLikelihood(data, spec)
        mle = np.array(read_json(ROOT/'results/production/mle/powerlaw'/f'{prior}.json')['result']['theta'])
        tests = []
        for _ in range(600):
            t = mle+rng.normal(size=len(mle))*np.array([.06 if n=='gamma0' else .008 if n=='f_int' else .15 for n in spec.names])
            d = data.table.delta.to_numpy()+rng.normal(size=data.n)*.07
            tests.append((t,d))
        tests += [(mle, np.full(data.n, .5)), (mle, np.full(data.n, 3.5))]
        for t,d in tests:
            original = logposterior(t,data,spec,delta=d)
            prepared = fast(t,data,spec,delta=d)
            if original != prepared:
                errors.append(dict(prior=prior, original=original, prepared=prepared, difference=prepared-original))
    assert not errors, errors[:4]
    print('Exact target check: 1204 points, bitwise equal', flush=True)
    small = data.subset(np.arange(2))
    scenario = dict(name='trajectory_test', sigma=.05, coherent_sigma=.05)
    spec = Spec('P2','flat')
    mle = dict(theta=[2.1,-.7,-.3,.08])
    cfg = dict(chains=4,burn=128,block_size=2,global_updates=3,checkpoint=40,
        diagnostics=dict(rhat_max=1.01,ess_min=2000,tau_multiple=100,tau_stability=.05))
    serial, checkpoint = out/'serial_reference', out/'partial_reference'
    if serial.exists() or checkpoint.exists():
        raise FileExistsError('Use fresh verification directories')
    with redirect_stdout(StringIO()):
        run_latent(small,spec,mle,serial,dict(cfg,steps=256),scenario,731)
        run_latent(small,spec,mle,checkpoint,dict(cfg,steps=160),scenario,731)
        for i in range(4):
            child = out/f'independent_{i}'
            child.mkdir()
            shutil.copy2(checkpoint/f'latent_{i:02d}.h5',child/'latent_00.h5')
            function,_ = chain_function(small,spec,prepared=True)
            function(small,spec,mle,child,dict(cfg,chains=1,steps=256),scenario,731+2003*i,resume=True)
            with h5py.File(serial/f'latent_{i:02d}.h5','r') as a,h5py.File(child/'latent_00.h5','r') as b:
                for name in ['theta','delta','b_delta','log_target']:
                    np.testing.assert_array_equal(a[name][:],b[name][:])
                assert json.loads(a.attrs['state']) == json.loads(b.attrs['state'])
    print('Exact continuation check: all 4 chain trajectories, adaptation and RNG match',flush=True)
    # Compare old and prepared evaluator for one fixed-global latent-update workload.
    spec = Spec('P2','flat')
    theta = np.array([2.1,-.7,-.3,.08])
    fast = PreparedDeltaLikelihood(data,spec)
    delta = data.table.delta.to_numpy()
    timings = {}
    for name, function in [('original',logposterior),('prepared',fast)]:
        start=time.perf_counter()
        for _ in range(3000):
            function(theta,data,spec,delta=delta)
        timings[name]=time.perf_counter()-start
    print('Kernel timings',timings,flush=True)
    results = []
    jobs = [(i,4000) for i in range(32)]
    for workers in [2,8,16,32]:
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            batch=list(pool.map(benchmark_worker,jobs))
        elapsed=time.perf_counter()-start
        results.append(dict(workers=workers,seconds=elapsed,calls=128000,calls_per_second=128000/elapsed,
                            sampling_kernel_seconds=sum(r['elapsed'] for r in batch)))
        print('Throughput benchmark',results[-1],flush=True)
    report=dict(passed=True,physical_cores=psutil.cpu_count(logical=False),logical_threads=os.cpu_count(),
        RAM_GB=psutil.virtual_memory().total/2**30,target_points_exact=1204,all_four_trajectory_states_exact=True,
        kernel_timings=timings,throughput_benchmarks=results,
        note='No dependency change, no fastmath; small-matrix likelihood benchmark includes worker startup and latent-coordinate transform')
    write_json(ROOT/'provenance/parallel_performance_validation.json',report)


if __name__=='__main__':
    main()
