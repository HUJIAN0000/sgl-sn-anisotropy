"""Call the unmodified author plotting module on newly qualified posterior samples."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aa_revision.common import ROOT, read_json, write_json, sha256
from aa_revision.revision import config
sys.path.insert(0, str(ROOT / 'tools'))
import cosmo_tools

LABELS = dict(gamma0=r'\gamma_0', gamma_z=r'\gamma_z', beta=r'\beta',
              gamma_s=r'\gamma_s', f_int=r'f_{\rm int}')


def main():
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    status = read_json(ROOT / 'results/production/sample_buffered_task_status.json')
    assert all(r['passed'] for r in status)
    before = sha256(ROOT / 'tools/cosmo_tools.py')
    out = ROOT / 'results/production/figures'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for case in config()['cases']:
        overlay, legends = [], []
        common = ['gamma0', 'gamma_z']+(['gamma_s'] if case.get('framework') == 'P3' else [])+['f_int']
        for prior in case.get('posterior_priors', []):
            folder = ROOT / 'results/production/chains' / case['id'] / prior
            summary = read_json(folder / 'posterior_summary.json')
            assert summary['production_eligible']
            with np.load(folder / 'posterior_draws.npz') as f:
                samples = f['samples']
                names = list(f['names'])
            labels = [LABELS[n] for n in names]
            stats = cosmo_tools.calculate_stats(samples, labels)
            for i, name in enumerate(names):
                np.testing.assert_allclose(stats[i]['median'], summary['quantiles'][name]['median'], rtol=0, atol=1e-12)
            stem = f"corner_{case['id']}_{prior}"
            cosmo_tools.plot_getdist_advanced(samples, labels, stats_list=stats,
                filename=str(out / f'{stem}.pdf'), width_inch=8 if len(names) >= 4 else 6)
            fig = plt.gcf()
            # Export the same author-generated figure with normal artist-aware bounding.
            fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', pad_inches=.15)
            fig.savefig(out / f'{stem}.png', bbox_inches='tight', pad_inches=.15, dpi=145)
            plt.close(fig)
            overlay.append(samples[:, [names.index(n) for n in common]])
            legends.append(prior)
            records.append(dict(case=case['id'], prior=prior, retained_draws=len(samples),
                samples_sha256=sha256(folder / 'posterior_draws.npz'),
                pdf=f'{stem}.pdf', method='cosmo_tools.plot_getdist_advanced'))
        if overlay:
            stem = f"overlay_{case['id']}"
            cosmo_tools.plot_getdist_comparison(overlay, [LABELS[n] for n in common],
                legend_labels=legends, filename=str(out / f'{stem}.pdf'))
            fig = plt.gcf()
            fig.savefig(out / f'{stem}.pdf', bbox_inches='tight', pad_inches=.15)
            fig.savefig(out / f'{stem}.png', bbox_inches='tight', pad_inches=.15, dpi=145)
            plt.close(fig)
            records.append(dict(case=case['id'], pdf=f'{stem}.pdf',
                                method='cosmo_tools.plot_getdist_comparison', parameters=common))
    assert sha256(ROOT / 'tools/cosmo_tools.py') == before
    write_json(out / 'figure_manifest.json', dict(author_tool_sha256=before,
        no_author_tool_modification=True, visual_QA='Pending visual review of rendered final PDFs', figures=records))


if __name__ == '__main__':
    main()
