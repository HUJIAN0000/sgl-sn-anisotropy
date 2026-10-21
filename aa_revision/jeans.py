"""Numerical spherical Jeans aperture moments for curved light profiles.

Sersic luminosity densities use numerical Abel deprojection of the exact
projected Sersic profile, not a Prugniel-Simien density approximation.
The mass profile is still a spherical power law. Apertures are circular.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import gamma as gamma_function, gammaincinv
from scipy.stats import ncx2
from .model import C_KMS, ARCSEC


class ApertureJeans:
    def __init__(self, data, tracer="hernquist", sersic_n=4., nrad=2048,
                 extent=1e6, psf_sigma_arcsec=0., abel_nodes=96, angular_nodes=128):
        df = data.table
        self.tracer, self.nrad, self.extent = tracer, nrad, extent
        self.thetaE = df.thetaE.to_numpy()*ARCSEC
        re = df.thetaeff.to_numpy()/df.thetaE.to_numpy()
        rap = df.thetaap.to_numpy()/df.thetaE.to_numpy()
        # Logarithmic limits are scaled for every lens. Extent tests are required
        # before using unusually compact, diffuse or steep systems.
        lower = np.minimum(re, rap)/extent
        upper = np.maximum(re, rap)*extent
        fraction = np.linspace(0, 1, nrad)
        self.logx = np.log(lower)[:, None]+fraction[None, :]*np.log(upper/lower)[:, None]
        self.x = np.exp(self.logx)
        x = self.x
        if tracer == "hernquist":
            a = re[:, None]/1.81527
            self.nu = 1/(x*(x+a)**3)
        elif tracer == "sersic":
            n = np.broadcast_to(np.asarray(sersic_n, float), (len(df),))
            if np.any(n <= 1):
                raise ValueError("This implementation requires Sersic n>1")
            self.nu = self._sersic_density(x, re, n, abel_nodes)
        elif tracer == "powerlaw_test":
            self.nu = x**(-df.delta.to_numpy()[:, None])
        else:
            raise ValueError("Unknown tracer")
        q = np.minimum(rap[:, None]/x, 1.)
        # Stable angular cap integrals. f2 = integral sin^2(theta) dcos(theta).
        self.f0 = q*q/(1+np.sqrt(np.maximum(0, 1-q*q)))
        self.f2 = self.f0**2*(1-self.f0/3)
        psf = np.broadcast_to(np.asarray(psf_sigma_arcsec, float), (len(df),))
        if np.any(psf < 0):
            raise ValueError("PSF sigma must be >=0 (sigma, not FWHM)")
        positive = np.flatnonzero(psf > 0)
        if len(positive):
            # Fraction of a circular Gaussian centred at projected radius R
            # that enters the aperture is a noncentral chi-square CDF.
            nodes, weights = leggauss(angular_nodes)
            mu, w = (nodes+1)/2, weights/2
            sin2 = 1-mu*mu
            for i in positive:
                s = psf[i]/df.thetaE.iloc[i]
                probability = ncx2.cdf((rap[i]/s)**2, 2,
                                       (x[i, :, None]/s)**2*sin2[None, :])
                self.f0[i] = probability @ w
                self.f2[i] = probability @ (w*sin2)
        self.denominator = trapezoid(self.nu*x**3*self.f0, self.logx, axis=1)
        if not np.isfinite(self.denominator).all() or np.any(self.denominator <= 0):
            raise ValueError("Invalid luminosity aperture integral")

    @staticmethod
    def _sersic_density(x, re, n, nodes):
        # R = r cosh(u) removes the Abel endpoint singularity:
        # nu(r)=bn/(pi n Re) int exp[-bn(R/Re)^(1/n)](R/Re)^(1/n-1) du.
        # The outer endpoint has Sersic exponent >=50; omitted tail is tiny.
        v, w = leggauss(nodes)
        result = np.zeros_like(x)
        for i in range(len(re)):
            bn = gammaincinv(2*n[i], .5)
            cutoff = re[i]*(50/bn)**n[i]
            umax = np.arccosh(np.maximum(2., cutoff/x[i]))
            u = umax[:, None]*(v[None, :]+1)/2
            rr = x[i, :, None]*np.cosh(u)/re[i]
            integrand = np.exp(-bn*rr**(1/n[i]))*rr**(1/n[i]-1)
            result[i] = bn/(np.pi*n[i]*re[i])*umax/2*(integrand @ w)
        return result

    def predict(self, gamma, beta, R):
        gamma = np.asarray(gamma)
        if not (-1 < beta < 1) or np.any(gamma <= 1) or np.any(gamma >= 3):
            return None
        # Jeans: nu sigma_r^2 / (G M_E/R_E) =
        # A(gamma) x^(-2 beta) int_x^inf nu(t)t^(2 beta+1-gamma) dt.
        integrand = self.nu*np.exp((2*beta+2-gamma[:, None])*self.logx)
        tail = -cumulative_trapezoid(integrand[:, ::-1], self.logx[:, ::-1],
                                     axis=1, initial=0)[:, ::-1]
        pressure = np.exp(-2*beta*self.logx)*tail
        numerator = trapezoid(pressure*self.x**3*(self.f0-beta*self.f2), self.logx, axis=1)
        A = 2/np.sqrt(np.pi)*gamma_function(gamma/2)/gamma_function((gamma-1)/2)
        s2 = C_KMS**2*self.thetaE/(4*np.asarray(R))*A*numerator/self.denominator
        if not np.isfinite(s2).all() or np.any(s2 <= 0):
            return None
        return np.sqrt(s2)


def make_engine(data, cfg, tracer):
    return ApertureJeans(data, tracer=tracer, sersic_n=cfg.get("sersic_n", 4),
        nrad=int(cfg.get("nrad", 2048)), extent=float(cfg.get("extent", 1e6)),
        psf_sigma_arcsec=cfg.get("psf_sigma_arcsec", 0.),
        abel_nodes=int(cfg.get("abel_nodes", 96)),
        angular_nodes=int(cfg.get("angular_nodes", 128)))
