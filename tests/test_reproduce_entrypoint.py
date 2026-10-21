"""Fast orchestration tests using small sparse checkpoints; never run a sampler."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from scipy.special import ndtr, ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproduce_revision2 as entry


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class LatentGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_dir = tempfile.TemporaryDirectory()
        root = Path(cls.template_dir.name)
        primary = root/'data/primary'
        primary.mkdir(parents=True)
        (primary/'matched_catalog.csv').write_text('Lens name,delta\na,2.1\nb,2.2\n', encoding='utf-8')
        for name in ('cov_lens.txt', 'cov_source.txt', 'cov_cross.txt'):
            (primary/name).write_text('1 0\n0 1\n', encoding='utf-8')
        science = root/'aa_revision'
        science.mkdir()
        (science/'dummy.py').write_text('# immutable test scientific source\n', encoding='utf-8')
        data_digest = entry.fingerprint({name: entry.digest(primary/name) for name in
            ('matched_catalog.csv', 'cov_lens.txt', 'cov_source.txt', 'cov_cross.txt')})
        code_hash = entry.fingerprint({'dummy.py': entry.digest(science/'dummy.py')})
        status = []
        for scenario in entry.LATENT_SCENARIOS:
            for prior in entry.LATENT_PRIORS:
                directory = root/'results/production/delta'/scenario/prior
                names = ['gamma0', 'gamma_z']+(['beta'] if prior == 'flat' else [])+['f_int']
                coherent = .05 if scenario == 'independent005_coherent005' else 0.
                meta = dict(data_digest=data_digest, code_hash=code_hash,
                    spec=dict(framework='P2', beta_prior=prior, domain='convergent_jeans', tracer='powerlaw'),
                    sigma=[.05, .05], coherent_sigma=coherent, burn=5000, chains=4)
                summary = dict(production_eligible=True, data_digest=data_digest,
                    spec=meta['spec'], scenario=dict(name=scenario, coherent_sigma=coherent),
                    sigma=meta['sigma'], quantiles={name: {} for name in names},
                    diagnostics=dict(total_steps=30000, burn=5000, passed=True,
                                     independent_ensembles_or_chains=4, retained_steps_per_chain=25000))
                write_json(directory/'run.json', dict(signature=entry.fingerprint(meta), metadata=meta))
                write_json(directory/'posterior_summary.json', summary)
                # Highly compressible fixture samples, not scientific posterior draws.
                np.savez_compressed(directory/'posterior_draws.npz',
                    samples=np.zeros((100000, len(names))), names=np.array(names))
                for index in range(4):
                    cls.make_checkpoint(directory/f'latent_{index:02d}.h5', 30000, len(names), coherent)
                status.append(dict(scenario=scenario, prior=prior, passed=True, steps=30000))
        write_json(root/'results/production/delta_task_status.json', status)

    @staticmethod
    def make_checkpoint(path, steps, ndim, coherent):
        z = np.zeros(2+int(coherent > 0))
        theta = np.arange(ndim, dtype=float)
        mu, sigma = np.array([2.1, 2.2]), np.array([.05, .05])
        lo, hi = ndtr((1.001-mu)/sigma), ndtr((2.999-mu)/sigma)
        delta = mu+sigma*ndtri(np.clip(lo+(hi-lo)*ndtr(z[:2]), 1e-15, 1-1e-15))
        with h5py.File(path, 'w') as f:
            for name, width, tail in [('theta', ndim, theta), ('delta', 2, delta),
                                      ('b_delta', None, 0.), ('log_target', None, -1.)]:
                shape = (steps,) if width is None else (steps, width)
                ds = f.create_dataset(name, shape=shape, maxshape=(None,)+shape[1:], chunks=True, dtype='f8')
                ds[steps-1] = tail
            f.attrs['state'] = json.dumps(dict(iteration=steps, theta=theta.tolist(), z=z.tolist(),
                covariance=np.eye(ndim).tolist(), mean=theta.tolist(), m2=np.zeros((ndim, ndim)).tolist(),
                logscale=0., count=0, accepted=0, rng=np.random.default_rng(21).bit_generator.state))

    @classmethod
    def tearDownClass(cls):
        cls.template_dir.cleanup()

    def setUp(self):
        original_cwd = Path.cwd()
        self.work_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.work_dir.cleanup)
        self.addCleanup(lambda: __import__('os').chdir(original_cwd))
        self.root = Path(self.work_dir.name)/'repository'
        shutil.copytree(Path(self.template_dir.name), self.root)
        self.group = self.root/'results/production/delta/independent005/flat'

    def make_result(self, code=0, stderr=''):
        stdout_path, stderr_path = self.root/'stdout.log', self.root/'stderr.log'
        stdout_path.write_text('fixture child output\n', encoding='utf-8')
        stderr_path.write_text(stderr, encoding='utf-8')
        return entry.RunResult(code, stdout_path, stderr_path)

    def change_summary(self, update):
        path = self.group/'posterior_summary.json'
        value = entry.read_json(path)
        update(value)
        write_json(path, value)

    def make_unaccepted_at_80k(self):
        def update(summary):
            summary['production_eligible'] = False
            summary['diagnostics'].update(passed=False, total_steps=80000, retained_steps_per_chain=75000)
        self.change_summary(update)
        path = self.root/'results/production/delta_task_status.json'
        status = entry.read_json(path)
        for record in status:
            if (record['scenario'], record['prior']) == ('independent005', 'flat'):
                record.update(passed=False, steps=80000)
        write_json(path, status)
        np.savez_compressed(self.group/'posterior_draws.npz', samples=np.zeros((300000, 4)),
                            names=np.array(['gamma0', 'gamma_z', 'beta', 'f_int']))
        for index in range(4):
            self.make_checkpoint(self.group/f'latent_{index:02d}.h5', 80000, 4, 0.)

    def test_all_eight_normal_acceptance(self):
        self.assertEqual(entry.validate_latent_initial(self.root), [True]*8)
        self.assertFalse(entry.require_initial_latent_completion(self.make_result(), self.root))
        entry.record_accepted_initial_latent(self.root)
        ledger = entry.read_json(self.root/'results/production/delta_parallel_task_status.json')
        self.assertEqual(len(ledger), 8)
        self.assertTrue(all(row['passed'] and row['source'] == 'original_serial' for row in ledger))

    def test_expected_unaccepted_80k_allows_recovery(self):
        self.make_unaccepted_at_80k()
        result = self.make_result(1, entry.EXPECTED_CONVERGENCE_EXIT+'\n')
        self.assertTrue(entry.require_initial_latent_completion(result, self.root))

    def test_crash_does_not_start_recovery(self):
        (self.group/'posterior_summary.json').unlink()
        result = self.make_result(1, 'Traceback (most recent call last):\nMemoryError\n')
        with self.assertRaisesRegex(RuntimeError, 'recovery was not started.*stdout:'):
            entry.require_initial_latent_completion(result, self.root)

    def test_expected_exit_with_missing_summary_fails_fast(self):
        (self.group/'posterior_summary.json').unlink()
        result = self.make_result(1, entry.EXPECTED_CONVERGENCE_EXIT+'\n')
        with self.assertRaisesRegex(RuntimeError, 'products failed validation'):
            entry.require_initial_latent_completion(result, self.root)

    def test_unaccepted_group_below_80k_fails(self):
        self.change_summary(lambda s: (s.update(production_eligible=False), s['diagnostics'].update(passed=False)))
        with self.assertRaisesRegex(RuntimeError, 'must finish 80000'):
            entry.validate_latent_initial(self.root)

    def test_invalid_total_steps_fails(self):
        self.change_summary(lambda s: s['diagnostics'].update(total_steps=30001))
        with self.assertRaisesRegex(RuntimeError, 'invalid initial-stage total_steps'):
            entry.validate_latent_initial(self.root)

    def test_nonboolean_acceptance_fails(self):
        self.change_summary(lambda s: s.update(production_eligible='false'))
        with self.assertRaisesRegex(RuntimeError, 'JSON boolean'):
            entry.validate_latent_initial(self.root)

    def test_missing_checkpoint_fails(self):
        (self.group/'latent_03.h5').unlink()
        with self.assertRaisesRegex(RuntimeError, 'latent_03.h5 is missing'):
            entry.validate_latent_initial(self.root)

    def test_checkpoint_iteration_disagreement_fails(self):
        path = self.group/'latent_01.h5'
        with h5py.File(path, 'a') as f:
            state = json.loads(f.attrs['state'])
            state['iteration'] = 29999
            f.attrs['state'] = json.dumps(state)
        with self.assertRaisesRegex(RuntimeError, 'iteration differs'):
            entry.validate_latent_initial(self.root)

    def test_checkpoint_dataset_length_disagreement_fails(self):
        with h5py.File(self.group/'latent_00.h5', 'a') as f:
            f['delta'].resize((29999, 2))
        with self.assertRaisesRegex(RuntimeError, 'dataset length or width'):
            entry.validate_latent_initial(self.root)

    def test_checkpoint_final_state_disagreement_fails(self):
        with h5py.File(self.group/'latent_02.h5', 'a') as f:
            f['theta'][-1, 0] = 999
        with self.assertRaisesRegex(RuntimeError, 'last theta'):
            entry.validate_latent_initial(self.root)

    def test_incomplete_resume_state_fails(self):
        with h5py.File(self.group/'latent_02.h5', 'a') as f:
            state = json.loads(f.attrs['state'])
            del state['rng']
            f.attrs['state'] = json.dumps(state)
        with self.assertRaisesRegex(RuntimeError, 'resume state is incomplete'):
            entry.validate_latent_initial(self.root)

    def test_run_signature_disagreement_fails(self):
        path = self.group/'run.json'
        run = entry.read_json(path)
        run['metadata']['burn'] = 4000
        write_json(path, run)
        with self.assertRaisesRegex(RuntimeError, 'signature mismatch'):
            entry.validate_latent_initial(self.root)

    def test_success_with_unaccepted_groups_fails(self):
        self.make_unaccepted_at_80k()
        with self.assertRaisesRegex(RuntimeError, 'reported success'):
            entry.require_initial_latent_completion(self.make_result(), self.root)

    def test_convergence_exit_with_all_accepted_fails(self):
        result = self.make_result(1, entry.EXPECTED_CONVERGENCE_EXIT+'\n')
        with self.assertRaisesRegex(RuntimeError, 'all summaries were accepted'):
            entry.require_initial_latent_completion(result, self.root)

    def test_latent_crash_never_invokes_recovery(self):
        result = self.make_result(2, 'fixture failure\n')
        with patch.object(entry, 'ROOT', self.root), patch.object(entry, 'ensure_raw_covariance'), \
             patch.object(entry, 'run', return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, 'recovery was not started'):
                entry.main(['latent'])
        self.assertEqual(run.call_count, 1)


class DriverInterfaceTests(unittest.TestCase):
    def test_default_workers_eight(self):
        self.assertEqual(entry.parse_args(['fit']).workers, 8)

    def test_matching_workers_explicitly_rejected(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
            entry.parse_args(['matching-variants', '--workers', '8'])
        self.assertEqual(error.exception.code, 2)
        self.assertIn('runs serially; omit --workers', stderr.getvalue())

    def test_matching_has_no_workers_forwarded(self):
        original_cwd = Path.cwd()
        self.addCleanup(lambda: __import__('os').chdir(original_cwd))
        with patch.object(entry, 'ensure_raw_covariance'), patch.object(entry, 'run') as run:
            entry.main(['matching-variants'])
        run.assert_called_once_with('run_matching_variants.py')

    def test_worker_range_checked(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            entry.parse_args(['fit', '--workers', '33'])

    def test_child_error_is_logged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'scripts').mkdir()
            (root/'scripts/fail.py').write_text(
                'import sys\nprint("before failure", flush=True)\n'
                'print("fixture error", file=sys.stderr)\nraise SystemExit(7)\n', encoding='utf-8')
            with patch.object(entry, 'ROOT', root):
                result = entry.run('fail.py', check=False)
                self.assertEqual(result.returncode, 7)
                self.assertIn('before failure', result.stdout_log.read_text(encoding='utf-8'))
                self.assertIn('fixture error', result.stderr_log.read_text(encoding='utf-8'))
                with self.assertRaisesRegex(RuntimeError, 'exited with code 7.*stdout:'):
                    entry.run('fail.py')


if __name__ == '__main__':
    unittest.main()
