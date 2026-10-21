"""Independent-chain scheduling adapter for the unchanged latent sampler transitions."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import inspect
import hashlib
import shutil
import json
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import h5py
import aa_revision.delta as delta_module
from aa_revision.common import ROOT, read_json, write_json, sha256, code_hash
from aa_revision.revision import config, load_primary
from aa_revision.model import Spec
from prepared_delta_likelihood import PreparedDeltaLikelihood


def chain_function(data, spec, prepared=True):
    source = inspect.getsource(delta_module.run_latent)
    marker = '    chains, deltas, bs = [], [], []'
    if source.count(marker) != 1:
        raise ValueError('Sampler source structure changed; review the adapter')
    source = source.split(marker)[0]
    source = source.replace('def run_latent(', 'def run_latent_chain_only(', 1)
    source += "    return dict(chain_file=str(out/'latent_00.h5'), steps=int(cfg['steps']))\n"
    globals_copy = dict(delta_module.__dict__)
    if prepared:
        globals_copy['logposterior'] = PreparedDeltaLikelihood(data, spec)
    exec(compile(source, '<audited-latent-chain-adapter>', 'exec'), globals_copy)
    return globals_copy['run_latent_chain_only'], hashlib.sha256(source.encode()).hexdigest()


def settings(steps, chains=1):
    return dict(chains=chains, burn=5000, block_size=32, global_updates=3, checkpoint=1000,
                diagnostics=config()['mcmc']['diagnostics'], steps=steps)


def worker(item):
    scenario, prior, chain_index, target, work_folder = item
    data = load_primary()
    spec = Spec('P2', prior, config()['domain'], 'powerlaw')
    dest = Path(work_folder)
    error_table = ROOT/'data/derived/delta_1percent_statistical_upper_bound.csv' if 'relative_sigma_bound' in scenario else None
    mle = read_json(ROOT/'results/production/mle/powerlaw'/f'{prior}.json')['result']
    function, adapter_hash = chain_function(data, spec, prepared=True)
    signature = dict(data_digest=data.digest, base_science_code_hash=code_hash(), adapter_hash=adapter_hash,
        evaluator_sha256=sha256(Path(__file__).with_name('prepared_delta_likelihood.py')),
        runtime_sha256=sha256(Path(__file__)), scenario=scenario, prior=prior,
        chain_index=chain_index, seed=config()['seed']+711+2003*chain_index)
    dest.mkdir(parents=True, exist_ok=True)
    meta = dest/'parallel_target.json'
    if meta.exists():
        if read_json(meta) != signature:
            raise ValueError('Parallel continuation target fingerprint changed')
    else:
        write_json(meta, signature)
    start = time.perf_counter()
    result = function(data, spec, mle, dest, settings(target), scenario,
        config()['seed']+711+2003*chain_index, resume=(dest/'run.json').exists() or (dest/'latent_00.h5').exists(), error_table=error_table)
    with h5py.File(result['chain_file'], 'r') as f:
        saved = json.loads(f.attrs['state'])
        assert saved['iteration'] == target == len(f['theta'])
    return dict(scenario=scenario['name'], prior=prior, chain_index=chain_index, steps=target,
                path=result['chain_file'], elapsed_seconds=time.perf_counter()-start, pid=os.getpid())
