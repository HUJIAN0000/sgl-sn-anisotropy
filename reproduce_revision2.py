"""Portable reproduction entrypoint; the default task only verifies the release.

Most calculation stages accept --workers (default 8). matching-variants is
serial and rejects an explicit --workers option. No calculation runs by default.
"""
from pathlib import Path
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
LATENT_SCENARIOS = (
    'independent005', 'independent010', 'independent005_coherent005',
    'statistical_1pct_upper_bound',
)
LATENT_PRIORS = ('flat', 'fixed0')
EXPECTED_CONVERGENCE_EXIT = 'Some latent-delta chains require further convergence work'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ensure_raw_covariance():
    record = read_json(ROOT/'provenance/compressed_input.json')
    target = ROOT/record['target']
    if target.exists():
        if digest(target) != record['sha256']:
            raise ValueError('Raw covariance differs from the published input')
        return
    with zipfile.ZipFile(ROOT/record['archive']) as z:
        if z.namelist() != [record['member']]:
            raise ValueError('Unexpected covariance archive members')
        temporary = target.with_suffix(target.suffix+'.partial')
        try:
            with z.open(record['member']) as f, temporary.open('wb') as out:
                shutil.copyfileobj(f, out)
            if digest(temporary) != record['sha256']:
                raise ValueError('Covariance archive checksum mismatch')
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()


def verify():
    manifest = read_json(ROOT/'MANIFEST_SHA256.json')
    errors = [name for name, sha in manifest.items()
              if not (ROOT/name).is_file() or digest(ROOT/name) != sha]
    if errors:
        raise RuntimeError('Package file mismatch: '+repr(errors))
    from aa_revision.model import load_data
    import numpy as np
    data = load_data(ROOT/'data/primary')
    assert data.n == 107
    assert len(set(data.table.sn_lens_CID.astype(str)) |
               set(data.table.sn_source_CID.astype(str))) == 214
    assert int((np.sqrt(np.diag(data.cov_R))/data.R > .20).sum()) == 12
    cfg = read_json(ROOT/'configs/matching_primary.json')
    for name in cfg['file_paths'].values():
        assert not Path(name).is_absolute() and (ROOT/name).is_file(), name
    print(f'Verified {len(manifest)} files; 107 lenses, 214 distinct CIDs, '
          '12 ratios above 20% uncertainty.')


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout_log: Path
    stderr_log: Path


def run(script, *args, check=True):
    """Capture both streams, keeping failure evidence even after an abrupt exit."""
    logdir = ROOT/'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    prefix = logdir/(Path(script).stem+'_'+stamp)
    stdout_log, stderr_log = Path(str(prefix)+'.stdout.log'), Path(str(prefix)+'.stderr.log')
    print(f'Running {script}. stdout: {stdout_log}; stderr: {stderr_log}', flush=True)
    with stdout_log.open('w', encoding='utf-8') as out, stderr_log.open('w', encoding='utf-8') as err:
        result = subprocess.run([sys.executable, '-u', '-B', 'scripts/'+script, *map(str, args)],
                                cwd=ROOT, stdout=out, stderr=err)
    record = RunResult(result.returncode, stdout_log, stderr_log)
    if check and record.returncode:
        raise RuntimeError(f'{script} exited with code {record.returncode}. '
                           f'See stdout: {stdout_log}; stderr: {stderr_log}')
    return record


def validate_latent_initial(root):
    """Validate all eight CLOSED initial products using metadata and HDF5 tails.

    No time series is read in full and no convergence statistic is recomputed.
    This guard does not change the sampling algorithm or its acceptance criteria.
    """
    import h5py
    import numpy as np
    from scipy.special import ndtr, ndtri

    root = Path(root)
    base = root/'results/production/delta'
    status_path = root/'results/production/delta_task_status.json'

    def require(condition, message):
        if not condition:
            raise RuntimeError('Incomplete or inconsistent initial latent stage: '+message)

    with (root/'data/primary/matched_catalog.csv').open(encoding='utf-8-sig', newline='') as f:
        observed = np.array([float(row['delta']) for row in csv.DictReader(f)])
    n_lenses = len(observed)
    input_names = ['matched_catalog.csv', 'cov_lens.txt', 'cov_source.txt', 'cov_cross.txt']
    data_digest = fingerprint({name: digest(root/'data/primary'/name) for name in input_names})
    source_files = sorted((root/'aa_revision').glob('*.py'))
    require(bool(source_files), 'scientific implementation files are missing')
    code_digest = fingerprint({p.name: digest(p) for p in source_files})
    status = read_json(status_path)
    require(isinstance(status, list) and len(status) == 8, f'{status_path} must contain eight records')
    statuses = {(row['scenario'], row['prior']): row for row in status}
    expected = {(s, p) for s in LATENT_SCENARIOS for p in LATENT_PRIORS}
    require(set(statuses) == expected, 'the eight scenario/prior status records do not match')
    accepted = []
    for scenario, prior in sorted(expected):
        directory = base/scenario/prior
        for filename in ('posterior_summary.json', 'posterior_draws.npz', 'run.json'):
            require((directory/filename).is_file(), f'{directory/filename} is missing')
        summary = read_json(directory/'posterior_summary.json')
        run_info = read_json(directory/'run.json')
        meta = run_info['metadata']
        qualified = summary['production_eligible']
        diag = summary['diagnostics']
        steps = diag['total_steps']
        require(type(qualified) is bool, f'{directory}: production_eligible must be a JSON boolean')
        require(type(diag['passed']) is bool and diag['passed'] == qualified,
                f'{directory}: diagnostic acceptance disagrees with production_eligible')
        require(type(steps) is int and steps in (30000, 50000, 80000),
                f'{directory}: invalid initial-stage total_steps {steps!r}')
        require(qualified or steps == 80000, f'{directory}: an unaccepted group must finish 80000 steps')
        require(run_info['signature'] == fingerprint(meta), f'{directory}: run signature mismatch')
        require(meta['code_hash'] == code_digest and meta['data_digest'] == data_digest,
                f'{directory}: run does not describe the current scientific code and inputs')
        require(summary['data_digest'] == data_digest, f'{directory}: summary data digest mismatch')
        require(meta['spec'] == summary['spec'] and
                summary['spec']['beta_prior'] == prior and
                summary['spec']['framework'] == 'P2' and summary['spec']['tracer'] == 'powerlaw',
                f'{directory}: model metadata mismatch')
        require(summary['scenario']['name'] == scenario, f'{directory}: scenario metadata mismatch')
        burn = meta['burn']
        require(type(burn) is int and burn == 5000 and meta['chains'] == 4 and
                diag['burn'] == burn and diag['independent_ensembles_or_chains'] == 4 and
                diag['retained_steps_per_chain'] == steps-burn,
                f'{directory}: burn, chain count or retained-step metadata mismatch')
        stat = statuses[(scenario, prior)]
        require(type(stat['passed']) is bool and stat['passed'] == qualified and stat['steps'] == steps,
                f'{directory}: task status disagrees with summary')
        sigma = np.asarray(meta['sigma'], dtype=float)
        coherent = .05 if scenario == 'independent005_coherent005' else 0.
        require(sigma.shape == (n_lenses,) and np.isfinite(sigma).all() and (sigma > 0).all() and
                summary['sigma'] == meta['sigma'] and meta['coherent_sigma'] == coherent and
                summary['scenario']['coherent_sigma'] == coherent,
                f'{directory}: slope-error metadata mismatch')
        names = ['gamma0', 'gamma_z']+(['beta'] if prior == 'flat' else [])+['f_int']
        ndim = len(names)
        require(set(summary['quantiles']) == set(names), f'{directory}: posterior parameter names mismatch')
        # Only the tiny NPY header and names array are decoded, not all retained samples.
        with zipfile.ZipFile(directory/'posterior_draws.npz') as archive:
            with archive.open('samples.npy') as sample_file:
                version = np.lib.format.read_magic(sample_file)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(sample_file)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(sample_file)
                else:
                    raise RuntimeError(f'{directory}: unsupported sample NPY format {version}')
            with archive.open('names.npy') as names_file:
                stored_names = np.load(names_file, allow_pickle=False).tolist()
        require(shape == (4*(steps-burn), ndim) and stored_names == names,
                f'{directory}: posterior sample header does not match summary')
        for index in range(4):
            path = directory/f'latent_{index:02d}.h5'
            require(path.is_file(), f'{path} is missing')
            with h5py.File(path, 'r') as f:
                state = json.loads(f.attrs['state'])
                required_state = {'iteration', 'theta', 'z', 'covariance', 'mean', 'm2',
                                  'logscale', 'count', 'accepted', 'rng'}
                require(required_state.issubset(state), f'{path}: saved resume state is incomplete')
                require(type(state['iteration']) is int and state['iteration'] == steps,
                        f'{path}: checkpoint iteration differs from summary')
                for name, width in [('theta', ndim), ('delta', n_lenses), ('b_delta', None), ('log_target', None)]:
                    shape = (steps,) if width is None else (steps, width)
                    require(name in f and f[name].shape == shape,
                            f'{path}: {name} dataset length or width differs from checkpoint')
                theta = np.asarray(state['theta'], dtype=float)
                z = np.asarray(state['z'], dtype=float)
                require(theta.shape == (ndim,) and np.isfinite(theta).all() and
                        z.shape == (n_lenses+int(coherent > 0),) and np.isfinite(z).all(),
                        f'{path}: invalid saved parameter state')
                for key, shape in [('covariance', (ndim, ndim)), ('mean', (ndim,)), ('m2', (ndim, ndim))]:
                    array = np.asarray(state[key], dtype=float)
                    require(array.shape == shape and np.isfinite(array).all(),
                            f'{path}: invalid saved adaptation state {key}')
                require(isinstance(state['rng'], dict) and 'bit_generator' in state['rng'] and
                        np.isfinite(state['logscale']) and type(state['count']) is int and
                        type(state['accepted']) is int and 0 <= state['accepted'] <= state['count'],
                        f'{path}: invalid saved RNG or adaptation counters')
                require(np.array_equal(f['theta'][steps-1], theta), f'{path}: last theta disagrees with state')
                b = coherent*z[-1] if coherent > 0 else 0.
                mu = observed+b
                lo, hi = ndtr((1.001-mu)/sigma), ndtr((2.999-mu)/sigma)
                expected_delta = mu+sigma*ndtri(np.clip(lo+(hi-lo)*ndtr(z[:n_lenses]), 1e-15, 1-1e-15))
                require(np.array_equal(f['delta'][steps-1], expected_delta) and f['b_delta'][steps-1] == b,
                        f'{path}: last slope state disagrees with checkpoint')
                require(np.isfinite(f['log_target'][steps-1]), f'{path}: nonfinite final target')
        accepted.append(qualified)
    return accepted


def require_initial_latent_completion(result, root=None):
    """Permit recovery only for the initial driver's explicit convergence exit."""
    root = ROOT if root is None else Path(root)
    logs = f'stdout: {result.stdout_log}; stderr: {result.stderr_log}'
    stderr = result.stderr_log.read_text(encoding='utf-8', errors='replace').strip()
    expected_exit = result.returncode == 1 and stderr.splitlines() == [EXPECTED_CONVERGENCE_EXIT]
    if result.returncode != 0 and not expected_exit:
        raise RuntimeError(f'Initial latent subprocess failed (exit {result.returncode}); '
                           f'recovery was not started. See {logs}')
    try:
        accepted = validate_latent_initial(root)
        if result.returncode == 0 and not all(accepted):
            raise RuntimeError('The initial process reported success but some groups were not accepted')
        if expected_exit and all(accepted):
            raise RuntimeError('The initial process reported unaccepted groups but all summaries were accepted')
    except Exception as exc:
        raise RuntimeError(f'Initial latent products failed validation; recovery was not started. '
                           f'{exc}. See {logs}') from exc
    return not all(accepted)


def record_accepted_initial_latent(root):
    """Keep the downstream audit ledger when no chain needs an extension."""
    root = Path(root)
    records = []
    for scenario in LATENT_SCENARIOS:
        for prior in LATENT_PRIORS:
            directory = root/'results/production/delta'/scenario/prior
            summary = read_json(directory/'posterior_summary.json')
            records.append(dict(scenario=scenario, prior=prior, passed=True,
                                steps=summary['diagnostics']['total_steps'],
                                path=str(directory), source='original_serial'))
    path = root/'results/production/delta_parallel_task_status.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', default='verify',
                        choices=['verify', 'fit', 'posteriors', 'mocks', 'sensitivities',
                                 'matching-variants', 'latent', 'figures', 'audit'])
    parser.add_argument('--workers', type=int, default=None,
                        help='Independent worker processes (default 8, range 1-32). '
                             'Not accepted for the serial matching-variants task.')
    args = parser.parse_args(argv)
    if args.task == 'matching-variants' and args.workers is not None:
        parser.error('matching-variants runs serially; omit --workers for this task.')
    if args.workers is None:
        args.workers = 8
    if not 1 <= args.workers <= 32:
        parser.error('--workers must be between 1 and 32')
    return args


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    ensure_raw_covariance()
    if args.task == 'verify':
        verify()
    elif args.task == 'fit':
        run('run_revision.py', 'prepare')
        run('run_revision.py', 'fit', '--workers', args.workers)
        run('summarize_fits.py')
    elif args.task == 'posteriors':
        run('run_buffered_production.py', '--workers', args.workers)
    elif args.task == 'mocks':
        run('run_revision.py', 'mocks', '--workers', args.workers)
    elif args.task == 'sensitivities':
        run('run_sensitivities.py', '--workers', args.workers)
    elif args.task == 'matching-variants':
        run('run_matching_variants.py')
    elif args.task == 'latent':
        initial = run('run_new_delta.py', '--workers', min(8, args.workers), check=False)
        needs_extension = require_initial_latent_completion(initial)
        if needs_extension:
            run('run_parallel_delta_recovery.py', '--workers', args.workers)
        else:
            record_accepted_initial_latent(ROOT)
            print('All eight initial latent products meet the unchanged acceptance criteria; no extension needed.')
    elif args.task == 'figures':
        run('plot_bounded_posteriors.py')
    elif args.task == 'audit':
        run('final_result_audit.py')


if __name__ == '__main__':
    main()
