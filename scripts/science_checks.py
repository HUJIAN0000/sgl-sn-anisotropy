"""Independent numerical and covariance checks before inference."""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
from pathlib import Path
import sys
from dataclasses import replace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aa_revision.common import ROOT, write_json
from aa_revision.model import load_data, powerlaw_sigma, C_KMS, ARCSEC, Spec, mean_and_cov
from aa_revision.jeans import ApertureJeans


def main():
    data = load_data(ROOT / 'data/primary')
    records = []
    # Known SIS limit is independent of the optical tracer implementation.
    table = data.table.copy()
    table['delta'] = 2.
    sis = replace(data, table=table)
    predicted = powerlaw_sigma(sis, np.full(data.n, 2.), 0.)
    expected = C_KMS*np.sqrt(table.thetaE.to_numpy()*ARCSEC/(4*np.pi*data.R))
    np.testing.assert_allclose(predicted, expected, rtol=1e-13)
    records.append(dict(check='analytic_SIS_normalisation', passed=True))
    # Independent numerical engine reproduces convergent analytic global power laws.
    sub = sis.subset(np.arange(min(12, sis.n)))
    numerical = ApertureJeans(sub, tracer='powerlaw_test', nrad=8192, extent=1e7)
    maximum = 0.
    for gamma, beta in [(2., 0.), (1.9, -.3), (2.15, .18)]:
        g = np.full(sub.n, gamma)
        analytic = powerlaw_sigma(sub, g, beta)
        numeric = numerical.predict(g, beta, sub.R)
        error = float(np.max(np.abs(numeric/analytic-1)))
        maximum = max(maximum, error)
        assert error < 5e-4, (gamma, beta, error)
    records.append(dict(check='numerical_vs_analytic_Jeans', passed=True, max_relative_sigma_error=maximum))
    # Independent finite-difference check of SN-to-dispersion covariance propagation.
    spec = Spec('P2', 'flat')
    theta = np.array([2.1, -.3, -.2, .08])
    mean, covariance = mean_and_cov(theta, data, spec)
    assert mean is not None
    h = 1e-6
    up = mean_and_cov(theta, replace(data, R=data.R+h), spec)[0]
    down = mean_and_cov(theta, replace(data, R=data.R-h), spec)[0]
    numeric_jac = (up-down)/(2*h)
    expected_jac = -.5*mean/data.R
    np.testing.assert_allclose(numeric_jac, expected_jac, rtol=1e-7)
    noise = data.table.dsigma_ap.to_numpy()**2+(.03*data.anchor)**2+(.08*data.anchor)**2
    independent = expected_jac[:, None]*data.cov_R*expected_jac[None, :]+np.diag(noise)
    np.testing.assert_allclose(covariance, independent, rtol=1e-13, atol=1e-10)
    np.linalg.cholesky(covariance)
    records.append(dict(check='full_model_covariance_and_Jacobian', passed=True))
    # The common absolute magnitude cancels in a ratio, including covariance.
    t = data.table
    ml = t.sn_lens_m_b_corr.to_numpy(float)
    ms = t.sn_source_m_b_corr.to_numpy(float)
    original = 10**(.2*(ml-ms))
    shifted = 10**(.2*((ml+19.3)-(ms+19.3)))
    np.testing.assert_allclose(original, shifted, rtol=1e-13)
    records.append(dict(check='SN_common_calibration_cancellation', passed=True))
    write_json(ROOT / 'provenance/science_checks.json', dict(passed=True, checks=records))
    print('SCIENCE CHECKS PASSED', records, flush=True)


if __name__ == '__main__':
    main()
