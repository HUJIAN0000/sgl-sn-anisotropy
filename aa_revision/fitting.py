"""Likelihood optimization, kept distinct from posterior/MAP optimization."""
from dataclasses import asdict
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from .model import loglike


def fit_mle(data, spec, options, seed=61321, engine=None, delta=None):
    start = time.perf_counter()
    def objective(x):
        ll = loglike(x, data, spec, delta=delta, engine=engine)
        return -ll if np.isfinite(ll) else 1e100
    results = []
    for j in range(int(options.get("starts", 2))):
        de = differential_evolution(objective, spec.bounds, seed=seed+j,
            popsize=int(options.get("popsize", 16)), maxiter=int(options.get("maxiter", 500)),
            tol=1e-8, init="sobol", updating="deferred", polish=False)
        local = minimize(objective, de.x, method="Nelder-Mead", bounds=spec.bounds,
            options={"maxiter": int(options.get("local_maxiter", 8000)),
                     "xatol": 1e-8, "fatol": 1e-8})
        best = local if local.fun < de.fun else de
        if not np.isfinite(best.fun) or best.fun >= 1e90:
            raise RuntimeError("Optimizer found no valid likelihood point")
        results.append(dict(theta=best.x.tolist(), lnL=float(-best.fun),
            de_success=bool(de.success), local_success=bool(local.success),
            de_message=str(de.message), local_message=str(local.message)))
    best = max(results, key=lambda r: r["lnL"])
    k, n = len(spec.names), data.n
    return dict(spec=asdict(spec), names=spec.names, N=n, k=k,
        theta=best["theta"], lnL=best["lnL"], AIC=2*k-2*best["lnL"],
        BIC=k*np.log(n)-2*best["lnL"], starts=results,
        multi_start_lnL_range=float(np.ptp([r["lnL"] for r in results])),
        optimizer_agreement_checked=bool(len(results) >= 2 and
            np.ptp([r["lnL"] for r in results]) < .05 and all(r["local_success"] for r in results)),
        elapsed_seconds=time.perf_counter()-start, data_digest=data.digest,
        optimization="likelihood only; Gaussian beta prior does not enter AIC/BIC")
