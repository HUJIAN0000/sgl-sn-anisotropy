"""Descriptive residual plots with full covariance, not calibrated goodness-of-fit p-values."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aa_revision.common import ROOT, read_json, write_json
from aa_revision.revision import config, load_primary, engine_for
from aa_revision.model import Spec, mean_and_cov, gamma_values


def main():
    cfg = config()
    data = load_primary()
    out = ROOT / 'results/diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    choices = [('powerlaw', 'flat'), ('powerlaw', 'fixed0'), ('hernquist', 'fixed0'), ('sersic_n4', 'fixed0')]
    fig, axes = plt.subplots(4, 4, figsize=(15, 12), constrained_layout=True)
    surveys = sorted(data.table['Survey name'].unique())
    colors = dict(zip(surveys, plt.get_cmap('tab10').colors))
    for row, (cid, prior) in enumerate(choices):
        case = next(c for c in cfg['cases'] if c['id'] == cid)
        fit = read_json(ROOT / 'results/production/mle' / cid / f'{prior}.json')['result']
        spec = Spec('P2', prior, cfg['domain'], case['tracer'])
        theta = np.asarray(fit['theta'])
        mean, cov = mean_and_cov(theta, data, spec, engine=engine_for(data, case))
        y = data.table.sigma_ap.to_numpy()
        residual = y-mean
        standard = residual/np.sqrt(np.diag(cov))
        # Cholesky-whitened components depend on catalogue order; no lens-level interpretation.
        whitened = solve_triangular(np.linalg.cholesky(cov), residual, lower=True)
        gamma = gamma_values(data, spec.unpack(theta))
        for survey in surveys:
            mask = data.table['Survey name'].to_numpy() == survey
            style = dict(s=21, alpha=.8, color=colors[survey], label=survey)
            axes[row, 0].scatter(mean[mask], y[mask], **style)
            axes[row, 1].scatter(data.table.zl.to_numpy()[mask], standard[mask], **style)
            ratio = (data.table.thetaap/data.table.thetaeff).to_numpy()
            axes[row, 2].scatter(ratio[mask], standard[mask], **style)
        limits = [min(y.min(), mean.min())*.92, max(y.max(), mean.max())*1.08]
        axes[row, 0].plot(limits, limits, 'k--', lw=.8)
        axes[row, 0].set(xlim=limits, ylim=limits, xlabel='Predicted dispersion [km/s]', ylabel='Observed dispersion [km/s]')
        for col, label in [(1, 'Lens redshift'), (2, 'Aperture / effective radius')]:
            axes[row, col].axhline(0, color='black', ls='--', lw=.8)
            axes[row, col].set(xlabel=label, ylabel='Residual / marginal model SD')
        axes[row, 3].hist(whitened, bins=15, density=True, alpha=.6, color='#336699')
        x = np.linspace(-4, 4, 150)
        axes[row, 3].plot(x, np.exp(-x*x/2)/np.sqrt(2*np.pi), 'k--', lw=1)
        axes[row, 3].set(xlabel='Cholesky-whitened residual', ylabel='Density')
        label = f'{cid}: {prior}'
        axes[row, 0].set_title(label)
        for i in range(data.n):
            rows.append(dict(case=cid, prior=prior, lens=data.table['Lens name'].iloc[i],
                survey=data.table['Survey name'].iloc[i], observed=y[i], predicted=mean[i],
                residual=residual[i], marginal_SD=np.sqrt(cov[i, i]), standardized_residual=standard[i],
                gamma=gamma[i], zl=data.table.zl.iloc[i], aperture_over_Re=ratio[i]))
        summaries.append(dict(case=cid, prior=prior, N=data.n, chi2=float(whitened@whitened),
            lnL=fit['lnL'], f_int=spec.unpack(theta)['f_int'],
            min_gamma=float(gamma.min()), max_gamma=float(gamma.max()),
            warning='Descriptive fit diagnostics; no chi-square p-value calibration claimed. Whitened components depend on row order.'))
    axes[0, 0].legend(fontsize=7)
    fig.savefig(out / 'residual_comparison.png', dpi=170)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / 'per_lens_residuals.csv', index=False)
    write_json(out / 'fit_diagnostics.json', summaries)
    print('Residual diagnostics saved', flush=True)


if __name__ == '__main__':
    main()
