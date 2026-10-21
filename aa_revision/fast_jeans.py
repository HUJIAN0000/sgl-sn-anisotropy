"""Validated Chebyshev acceleration of the direct numerical Jeans integral.

No change to the likelihood or radial quadrature. The approximation is used
only after independent direct-engine sigma/likelihood checks pass. Coefficients
represent log(sigma^2 R) on gamma in [1.05,2.95], beta in [-1,1].
"""
from pathlib import Path
import time
import math
import numpy as np
from numba import njit
from scipy.fft import dctn
from .jeans import make_engine
from .common import fingerprint, sha256, write_json, read_json


@njit(cache=True)
def _evaluate(coefficients, gammas, beta, ratios):
    n, ng, nb = coefficients.shape
    result = np.empty(n)
    for i in range(n):
        vals = np.empty(ng)
        for g in range(ng):
            b1 = 0.; b2 = 0.
            for b in range(nb-1, 0, -1):
                new = 2*beta*b1-b2+coefficients[i,g,b]
                b2=b1; b1=new
            vals[g]=beta*b1-b2+coefficients[i,g,0]
        x=(gammas[i]-2.)/.95
        b1=0.; b2=0.
        for g in range(ng-1,0,-1):
            new=2*x*b1-b2+vals[g]
            b2=b1; b1=new
        value=x*b1-b2+vals[0]
        # Treat the known normalization analytically rather than interpolating
        # its nearby gamma=1 singularity.
        g=gammas[i]
        A=2/np.sqrt(np.pi)*np.exp(math.lgamma(g/2)-math.lgamma((g-1)/2))
        result[i]=np.sqrt(np.exp(value)*A/ratios[i])
    return result


class ChebyshevJeans:
    def __init__(self, coefficients, metadata):
        self.coefficients=np.ascontiguousarray(coefficients)
        self.metadata=metadata

    def predict(self, gamma, beta, R):
        gamma=np.asarray(gamma, dtype=float)
        if not -1<beta<1 or np.any(gamma<=1.05) or np.any(gamma>=2.95):
            return None
        return _evaluate(self.coefficients, gamma, beta, np.asarray(R))


def build_accelerator(data, config, tracer, folder, ng=32, nb=28):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    metadata=dict(data_digest=data.digest, config=config, tracer=tracer, ng=ng, nb=nb,
                  implementation=sha256(Path(__file__)), direct_engine=sha256(Path(__file__).with_name('jeans.py')))
    signature=fingerprint(metadata)
    path=folder/(signature+'.npz'); check=folder/(signature+'.json')
    if path.exists() and check.exists():
        saved=read_json(check)
        if not saved['validation']['passed']:
            raise ValueError('Cached Jeans accelerator failed validation')
        return ChebyshevJeans(np.load(path)['coefficients'], saved)
    start=time.perf_counter()
    direct=make_engine(data,config,tracer)
    gs=2+.95*np.cos(np.pi*(np.arange(ng)+.5)/ng)
    bs=np.cos(np.pi*(np.arange(nb)+.5)/nb)
    values=np.empty((ng,nb,data.n))
    for a,g in enumerate(gs):
        for b,beta in enumerate(bs):
            s=direct.predict(np.full(data.n,g),beta,data.R)
            A=2/np.sqrt(np.pi)*np.exp(math.lgamma(g/2)-math.lgamma((g-1)/2))
            values[a,b]=np.log(s*s*data.R/A)
    coefficients=dctn(values,type=2,axes=(0,1))/(ng*nb)
    coefficients[0]*=.5; coefficients[:,0]*=.5
    coefficients=np.transpose(coefficients,(2,0,1)).copy()
    result=ChebyshevJeans(coefficients,metadata)
    rng=np.random.default_rng(71321)
    errors=[]
    # Independent non-node gamma per object, including the edges.
    for i in range(120):
        gamma=rng.uniform(1.05001,2.94999,data.n)
        beta=rng.uniform(-.9999,.9999)
        exact=direct.predict(gamma,beta,data.R)
        approx=result.predict(gamma,beta,data.R)
        errors.append(float(np.max(np.abs(approx/exact-1))))
    validation=dict(points=120,maximum_relative_sigma_error=max(errors),
                    passed=bool(max(errors)<2e-6),
                    interpretation='Interpolation only; radial quadrature is validated separately')
    saved=dict(signature=signature,metadata=metadata,validation=validation,
               elapsed_seconds=time.perf_counter()-start)
    write_json(check,saved)
    if not validation['passed']:
        raise RuntimeError(f'Accelerator inaccurate: {validation}')
    np.savez_compressed(path,coefficients=coefficients)
    result.metadata=saved
    return result
