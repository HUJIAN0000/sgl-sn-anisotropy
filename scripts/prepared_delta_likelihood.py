"""Prepared P2 power-law likelihood; preserve original arithmetic order, no fastmath."""
import numpy as np
from scipy.special import gamma as gamma_function
from aa_revision.model import C_KMS, ARCSEC, gaussian_loglike


class PreparedDeltaLikelihood:
    def __init__(self, data, spec):
        if spec.framework != 'P2' or spec.tracer != 'powerlaw':
            raise ValueError('Prepared evaluator is only for existing P2 latent-delta targets')
        self.data, self.spec = data, spec
        self.zl = data.table.zl.to_numpy()
        self.predictor = data.predictor
        self.observed = data.table.sigma_ap.to_numpy()
        thetaE = data.table.thetaE.to_numpy()
        self.amplitude = C_KMS**2/(2*np.sqrt(np.pi))*thetaE*ARCSEC/data.R
        self.aperture = data.table.thetaap.to_numpy()/thetaE
        self.diagonal = data.table.dsigma_ap.to_numpy()**2+(.03*data.anchor)**2
        self.last_global = None

    def __call__(self, theta, data, spec, delta=None):
        if data is not self.data or spec is not self.spec:
            raise ValueError('Prepared target called with another data/spec object')
        prior = spec.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        p = spec.unpack(theta)
        global_key = (p['gamma0'], p['gamma_z'], p['beta'])
        if global_key != self.last_global:
            g = p['gamma0']+p['gamma_z']*self.zl+p['gamma_s']*self.predictor
            if np.any(g <= 1.05) or np.any(g >= 2.95):
                return -np.inf
            self.gamma = g
            self.gamma_numerator = gamma_function(g/2)
            self.gamma_denominator = gamma_function((g-1)/2)
            self.radial_factor = self.aperture**(2-g)
            self.last_global = global_key
        g = self.gamma
        d = self.data.table.delta.to_numpy() if delta is None else np.asarray(delta)
        beta = p['beta']
        xi = g+d-2
        if spec.domain == 'convergent_jeans':
            if np.any(d <= 1) or np.any(d >= 3) or np.any(xi <= 1) or np.any(xi >= 3) or np.any(xi <= 2*beta):
                return -np.inf
        with np.errstate(all='ignore'):
            a = gamma_function((xi-1)/2)/gamma_function(xi/2)
            b = beta*gamma_function((xi+1)/2)/gamma_function((xi+2)/2)
            c = self.gamma_numerator*gamma_function(d/2)/(self.gamma_denominator*gamma_function((d-1)/2))
            F = (3-d)/((xi-2*beta)*(3-xi))*(a-b)*c
            s2 = self.amplitude*F*self.radial_factor
        if not np.isfinite(s2).all() or np.any(s2 <= 0):
            return -np.inf
        mean = np.sqrt(s2)
        jac = -.5*mean/data.R
        covariance = jac[:, None]*data.cov_R*jac[None, :]
        diagonal = self.diagonal+(p['f_int']*data.anchor)**2
        covariance[np.diag_indices(data.n)] += diagonal
        return prior+gaussian_loglike(self.observed-mean, covariance)
