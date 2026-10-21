"""Spherical Jeans power-law model and full correlated Gaussian likelihood.

The SN covariance is propagated at every parameter evaluation. All covariance
terms, including the determinant and N log(2 pi), enter the likelihood.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gamma as gamma_function, ndtr
from .common import fingerprint, sha256, read_json

C_KMS = 299792.458
ARCSEC = np.pi/(180*3600)


@dataclass
class Data:
    table: pd.DataFrame
    R: np.ndarray
    cov_R: np.ndarray
    predictor: np.ndarray
    anchor: np.ndarray
    digest: str

    def subset(self, indices):
        indices = np.asarray(indices, dtype=int)
        return replace(self, table=self.table.iloc[indices].reset_index(drop=True),
            R=self.R[indices], cov_R=self.cov_R[np.ix_(indices, indices)],
            predictor=self.predictor[indices], anchor=self.anchor[indices],
            digest=fingerprint(dict(parent=self.digest, indices=indices.tolist())))

    @property
    def n(self):
        return len(self.table)

    def observed(self, values):
        """Replace y but retain the variance anchor: used by coherent mocks."""
        table = self.table.copy()
        table["sigma_ap"] = np.asarray(values)
        return replace(self, table=table)


def p3_predictor(df):
    """Distances in Mpc, Om=.30, H0=70. This is NOT Chen's sigma_e2^2/Re."""
    def dc(z):
        return C_KMS/70 * quad(lambda x: (.3*(1+x)**3+.7)**(-.5), 0, z)[0]
    cl = np.array([dc(z) for z in df.zl])
    cs = np.array([dc(z) for z in df.zs])
    dl, ds, dls = cl/(1+df.zl), cs/(1+df.zs), (cs-cl)/(1+df.zs)
    return np.asarray(np.log10(166*ds/(dl*dls)*(df.thetaE/df.thetaeff)**2))


def load_data(folder):
    folder = Path(folder)
    names = ["matched_catalog.csv", "cov_lens.txt", "cov_source.txt", "cov_cross.txt"]
    mf = folder / "manifest.json"
    if mf.exists():
        for name, digest in read_json(mf)["file_sha256"].items():
            if sha256(folder/name) != digest:
                raise ValueError(f"Frozen input changed: {name}")
    df = pd.read_csv(folder / names[0])
    df.columns = df.columns.str.strip()
    positive = ["zl", "zs", "thetaE", "thetaeff", "thetaap", "sigma_ap", "dsigma_ap"]
    numeric = positive+["delta", "sn_lens_m_b_corr", "sn_source_m_b_corr"]
    if not np.isfinite(df[numeric].to_numpy()).all() or (df[positive] <= 0).any().any() or (df.zs <= df.zl).any():
        raise ValueError("Invalid redshift/angle/dispersion or nonfinite catalogue values")
    if df["Lens name"].duplicated().any():
        raise ValueError("Duplicate lens identifiers in the matched catalogue")
    n = len(df)
    ll, ss, ls = [np.loadtxt(folder / f"cov_{r}.txt", ndmin=2)
                  for r in ("lens", "source", "cross")]
    if any(c.shape != (n, n) for c in [ll, ss, ls]):
        raise ValueError("Covariance/catalogue dimension mismatch")
    cdiff = ll + ss - ls - ls.T
    cdiff = (cdiff + cdiff.T)/2
    np.linalg.cholesky(cdiff)
    ratio = 1 - 10**(.2*(df.sn_lens_m_b_corr-df.sn_source_m_b_corr))*((1+df.zs)/(1+df.zl))
    ratio = np.asarray(ratio)
    if np.any(ratio <= .001):
        raise ValueError("Nonpositive/near-zero distance ratio in matched data")
    derivative = -.2*np.log(10)*(1-ratio)
    cov_R = derivative[:, None]*cdiff*derivative[None, :]
    return Data(df, ratio, cov_R, p3_predictor(df), df.sigma_ap.to_numpy().copy(),
                fingerprint({name: sha256(folder/name) for name in names}))


@dataclass(frozen=True)
class Spec:
    framework: str = "P2"
    beta_prior: str = "flat"  # flat, gaussian, fixed0, fixed018
    domain: str = "convergent_jeans"  # legacy is only for regression
    tracer: str = "powerlaw"

    def __post_init__(self):
        if self.framework not in ("P2", "P3") or self.beta_prior not in (
                "flat", "gaussian", "fixed0", "fixed018"):
            raise ValueError("Unknown framework/beta prior")
        if self.domain not in ("legacy", "convergent_jeans"):
            raise ValueError("Unknown support convention")
        if self.tracer not in ("powerlaw", "powerlaw_test", "hernquist", "sersic"):
            raise ValueError("Unknown tracer")

    @property
    def fixed_beta(self):
        return {"fixed0": 0.0, "fixed018": .18}.get(self.beta_prior)

    @property
    def names(self):
        names = ["gamma0", "gamma_z"]
        if self.fixed_beta is None:
            names += ["beta"]
        if self.framework == "P3":
            names += ["gamma_s"]
        return names + ["f_int"]

    @property
    def bounds(self):
        b = {"gamma0": (1., 3.), "gamma_z": (-2., 2.), "beta": (-1., 1.),
             "gamma_s": (-1.5, 1.5), "f_int": (0., 1.)}
        return np.array([b[name] for name in self.names])

    def unpack(self, theta):
        p = dict(zip(self.names, theta))
        p["beta"] = p.get("beta", self.fixed_beta)
        p["gamma_s"] = p.get("gamma_s", 0.)
        return p

    def log_prior(self, theta):
        """Base box-prior density; physical support is checked by the model.

        The joint support-normalization constant cancels in a posterior run.
        Evidence comparisons estimate and include it separately.
        """
        theta = np.asarray(theta)
        b = self.bounds
        if theta.shape != (len(b),) or not np.isfinite(theta).all() or np.any(theta <= b[:, 0]) or np.any(theta >= b[:, 1]):
            return -np.inf
        val = -np.log(b[:, 1]-b[:, 0]).sum()
        if self.beta_prior == "gaussian":
            beta = theta[self.names.index("beta")]
            norm = ndtr((1-.18)/.13)-ndtr((-1-.18)/.13)
            # Replace the U(-1,1) density with normalized truncated N(.18,.13^2).
            val += np.log(2) - .5*((beta-.18)/.13)**2 - np.log(.13*np.sqrt(2*np.pi)*norm)
        return float(val)


def gamma_values(data, p):
    return p["gamma0"] + p["gamma_z"]*data.table.zl.to_numpy() + p["gamma_s"]*data.predictor


def powerlaw_sigma(data, gamma, beta, delta=None, domain="convergent_jeans"):
    delta = data.table.delta.to_numpy() if delta is None else np.asarray(delta)
    xi = gamma + delta - 2
    if domain == "convergent_jeans":
        # These conditions ensure the relevant infinite power-law moments
        # converge. They do not establish a nonnegative distribution function.
        if np.any(delta <= 1) or np.any(delta >= 3) or np.any(xi <= 1) or np.any(xi >= 3) or np.any(xi <= 2*beta):
            return None
    with np.errstate(all="ignore"):
        a = gamma_function((xi-1)/2)/gamma_function(xi/2)
        b = beta*gamma_function((xi+1)/2)/gamma_function((xi+2)/2)
        c = gamma_function(gamma/2)*gamma_function(delta/2)/(
            gamma_function((gamma-1)/2)*gamma_function((delta-1)/2))
        F = (3-delta)/((xi-2*beta)*(3-xi))*(a-b)*c
        df = data.table
        s2 = C_KMS**2/(2*np.sqrt(np.pi))*df.thetaE.to_numpy()*ARCSEC/data.R*F*(
              df.thetaap.to_numpy()/df.thetaE.to_numpy())**(2-gamma)
    if not np.isfinite(s2).all() or np.any(s2 <= 0):
        return None
    return np.sqrt(s2)


def mean_and_cov(theta, data, spec, delta=None, engine=None):
    if not np.isfinite(spec.log_prior(theta)):
        return None, None
    p = spec.unpack(theta)
    gamma = gamma_values(data, p)
    if np.any(gamma <= 1.05) or np.any(gamma >= 2.95):
        return None, None
    if spec.tracer == "powerlaw":
        mean = powerlaw_sigma(data, gamma, p["beta"], delta, spec.domain)
    else:
        if engine is None:
            raise ValueError("A numerical Jeans engine is required for curved tracers")
        if spec.tracer == "powerlaw_test" and powerlaw_sigma(data, gamma, p["beta"], delta, spec.domain) is None:
            return None, None
        mean = engine.predict(gamma, p["beta"], data.R)
    if mean is None or not np.isfinite(mean).all():
        return None, None
    jac = -.5*mean/data.R
    cov = jac[:, None]*data.cov_R*jac[None, :]
    diag = data.table.dsigma_ap.to_numpy()**2 + (.03*data.anchor)**2 + (p["f_int"]*data.anchor)**2
    cov[np.diag_indices(data.n)] += diag
    return mean, cov


def gaussian_loglike(residual, cov):
    try:
        factor = cho_factor(cov, lower=True, check_finite=False)
        chi2 = residual @ cho_solve(factor, residual, check_finite=False)
        logdet = 2*np.log(np.diag(factor[0])).sum()
        return float(-.5*(chi2+logdet+len(residual)*np.log(2*np.pi)))
    except (np.linalg.LinAlgError, ValueError):
        return -np.inf


def loglike(theta, data, spec, delta=None, engine=None):
    mean, cov = mean_and_cov(theta, data, spec, delta, engine)
    if mean is None:
        return -np.inf
    return gaussian_loglike(data.table.sigma_ap.to_numpy()-mean, cov)


def logposterior(theta, data, spec, delta=None, engine=None):
    prior = spec.log_prior(theta)
    if not np.isfinite(prior):
        return -np.inf
    return prior + loglike(theta, data, spec, delta, engine)
