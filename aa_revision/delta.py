"""Joint latent-delta posterior: random-walk global steps + elliptical slices.

delta_i follows N(delta_hat_i+b, sigma_i^2), truncated to (1.001,2.999).
b has an optional centred Gaussian prior. Latent standard-normal coordinates
are updated by blocked elliptical slice sampling; no averaged fits substitute
for integrating the likelihood. Adaptation stops at the fixed burn-in boundary.
"""
from pathlib import Path
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from .model import logposterior
from .common import fingerprint, code_hash, environment, write_json, read_json
from .diagnostics import diagnose, quantiles


def slope_errors(data, scenario, table=None):
    if table is None:
        sigma = np.full(data.n, float(scenario["sigma"]))
    else:
        df = pd.read_csv(table)
        if df["Lens name"].duplicated().any():
            raise ValueError("Duplicate names in delta-error table")
        sigma = data.table[["Lens name"]].merge(df[["Lens name", "delta_sigma"]],
                    on="Lens name", how="left", validate="one_to_one").delta_sigma.to_numpy()
    if not np.isfinite(sigma).all() or np.any(sigma < 0):
        raise ValueError("delta_sigma must be finite and nonnegative for every lens")
    return sigma


def latent_to_delta(z, data, sigma, coherent_sigma):
    b = coherent_sigma*z[-1] if coherent_sigma > 0 else 0.
    mu = data.table.delta.to_numpy()+b
    result = mu.copy()
    positive = sigma > 0
    if np.any(positive):
        s = sigma[positive]
        lo = ndtr((1.001-mu[positive])/s)
        hi = ndtr((2.999-mu[positive])/s)
        u = lo+(hi-lo)*ndtr(z[:data.n][positive])
        result[positive] = mu[positive]+s*ndtri(np.clip(u, 1e-15, 1-1e-15))
    return result, b


def elliptical_slice(z, current, log_target, rng, indices=None):
    """One exact Gaussian-prior-invariant slice update; target excludes N(z)."""
    indices = np.arange(len(z)) if indices is None else np.asarray(indices)
    direction = rng.normal(size=len(indices))
    angle = rng.uniform(0, 2*np.pi)
    lower, upper = angle-2*np.pi, angle
    threshold = current+np.log(rng.uniform())
    for _ in range(10000):
        proposal = z.copy()
        proposal[indices] = z[indices]*np.cos(angle)+direction*np.sin(angle)
        value = log_target(proposal)
        if value > threshold:
            return proposal, value
        if angle < 0:
            lower = angle
        else:
            upper = angle
        angle = rng.uniform(lower, upper)
    raise RuntimeError("Elliptical slice failed to close; inspect the target/support")


def run_latent(data, spec, mle, out, cfg, scenario, seed, resume=False, error_table=None):
    import h5py
    if spec.tracer != "powerlaw":
        raise ValueError("delta scenarios refer to the power-law tracer")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sigma = slope_errors(data, scenario, error_table)
    coherent = float(scenario.get("coherent_sigma", 0.))
    if coherent < 0:
        raise ValueError("coherent_sigma must be >=0")
    latent_dim = data.n+int(coherent > 0)
    ndim, burn = len(spec.names), int(cfg["burn"])
    meta = dict(data_digest=data.digest, spec=asdict(spec), code_hash=code_hash(),
        sigma=sigma.tolist(), coherent_sigma=coherent, seed=seed, burn=burn,
        chains=int(cfg["chains"]), block_size=int(cfg["block_size"]),
        global_updates=int(cfg.get("global_updates", 3)), environment=environment())
    signature = fingerprint(meta)
    sidecar = out/"run.json"
    if sidecar.exists():
        if not resume:
            raise FileExistsError(f"Latent chains exist: {out}")
        if read_json(sidecar)["signature"] != signature:
            raise ValueError("Latent resume fingerprint mismatch")
    else:
        write_json(sidecar, dict(signature=signature, metadata=meta))
    for c in range(int(cfg["chains"])):
        path = out/f"latent_{c:02d}.h5"
        if path.exists() and not resume:
            raise FileExistsError(path)
        rng = np.random.default_rng(seed+2003*c)
        with h5py.File(path, "a") as f:
            if "state" in f.attrs:
                st = json.loads(f.attrs["state"])
                rng.bit_generator.state = st["rng"]
                theta, z = np.array(st["theta"]), np.array(st["z"])
                covariance, mean, m2 = [np.array(st[k]) for k in ("covariance", "mean", "m2")]
                logscale, count, accepted = st["logscale"], st["count"], st["accepted"]
                # If a write was interrupted before the checkpoint attr update,
                # discard only the uncommitted rows; preserve the saved RNG state.
                for name in ["theta", "delta", "b_delta", "log_target"]:
                    f[name].resize((st["iteration"],)+f[name].shape[1:])
            else:
                theta = np.array(mle["theta"])
                scales = np.array([.02 if n == "gamma0" else .06 if n != "f_int" else .008
                                   for n in spec.names])
                for _ in range(10000):
                    initial = theta+rng.normal(size=ndim)*scales
                    if np.isfinite(logposterior(initial, data, spec)):
                        theta = initial
                        break
                z = np.zeros(latent_dim)
                covariance = np.diag(scales**2)
                mean, m2 = theta.copy(), np.zeros((ndim, ndim))
                logscale, count, accepted = 0., 0, 0
                for name, width in [("theta", ndim), ("delta", data.n)]:
                    f.create_dataset(name, (0, width), maxshape=(None, width), chunks=True, dtype="f8")
                for name in ("b_delta", "log_target"):
                    f.create_dataset(name, (0,), maxshape=(None,), chunks=True, dtype="f8")
            def target(t, zz):
                d, _ = latent_to_delta(zz, data, sigma, coherent)
                if np.any(d <= 1.001) or np.any(d >= 2.999):
                    return -np.inf
                return logposterior(t, data, spec, delta=d)
            value = target(theta, z)
            if not np.isfinite(value):
                raise RuntimeError("Invalid latent initial/checkpoint state")
            start = len(f["theta"])
            interval = int(cfg.get("checkpoint", 100))
            pending = []
            for iteration in range(start, int(cfg["steps"])):
                for _ in range(int(cfg.get("global_updates", 3))):
                    proposal = theta+np.exp(logscale)*np.linalg.cholesky(covariance) @ rng.normal(size=ndim)
                    proposed = target(proposal, z)
                    if np.log(rng.uniform()) < proposed-value:
                        theta, value = proposal, proposed
                        accepted += 1
                    count += 1
                for a in range(0, latent_dim, int(cfg["block_size"])):
                    indices = np.arange(a, min(a+int(cfg["block_size"]), latent_dim))
                    z, value = elliptical_slice(z, value, lambda zz: target(theta, zz), rng, indices)
                if iteration < burn:
                    t = iteration+1
                    diff = theta-mean
                    mean += diff/t
                    m2 += np.outer(diff, theta-mean)
                    if t >= 100 and t % 50 == 0:
                        covariance = 2.38**2/ndim*(m2/max(1, t-1)+np.diag(scales**2)*1e-3)
                        logscale = np.clip(logscale+min(.1, 1/np.sqrt(t/50))*(accepted/max(1, count)-.234), -3, 3)
                        accepted, count = 0, 0
                d, b = latent_to_delta(z, data, sigma, coherent)
                pending.append((theta.copy(), d.copy(), b, value))
                if len(pending) >= interval or iteration+1 == int(cfg["steps"]):
                    old, new = len(f["theta"]), len(f["theta"])+len(pending)
                    for name, values in [("theta", [v[0] for v in pending]),
                                         ("delta", [v[1] for v in pending]),
                                         ("b_delta", [v[2] for v in pending]),
                                         ("log_target", [v[3] for v in pending])]:
                        f[name].resize((new,)+f[name].shape[1:])
                        f[name][old:new] = values
                    state = dict(theta=theta.tolist(), z=z.tolist(), covariance=covariance.tolist(),
                        mean=mean.tolist(), m2=m2.tolist(), logscale=float(logscale), count=count,
                        accepted=accepted, iteration=iteration+1, rng=rng.bit_generator.state)
                    f.attrs["state"] = json.dumps(state)
                    f.flush()
                    pending.clear()
                    print(f"delta {scenario['name']} {spec.beta_prior}: chain {c+1}, {iteration+1}/{cfg['steps']}", flush=True)
    chains, deltas, bs = [], [], []
    for c in range(int(cfg["chains"])):
        with h5py.File(out/f"latent_{c:02d}.h5", "r") as f:
            chains.append(f["theta"][:]); deltas.append(f["delta"][:]); bs.append(f["b_delta"][:])
    chains, deltas, bs = np.asarray(chains), np.asarray(deltas), np.asarray(bs)
    diagnostic_values = np.concatenate([chains, deltas], axis=2)
    diagnostic_names = spec.names+["delta:"+n for n in data.table["Lens name"]]
    if coherent > 0:
        diagnostic_values = np.concatenate([diagnostic_values, bs[:, :, None]], axis=2)
        diagnostic_names.append("b_delta")
    # Remove truly fixed delta coordinates, which have no mixing diagnostic.
    variable = np.var(diagnostic_values.reshape(-1, diagnostic_values.shape[-1]), axis=0) > 1e-24
    diagnostics = diagnose(diagnostic_values[:, :, variable], burn, cfg["diagnostics"], ensemble=False)
    samples = chains[:, burn:].reshape(-1, ndim)
    result = dict(spec=asdict(spec), scenario=scenario, sigma=sigma.tolist(),
        data_digest=data.digest, quantiles=quantiles(samples, spec.names),
        diagnostics=diagnostics, diagnostic_names=np.array(diagnostic_names)[variable].tolist(),
        b_delta=quantiles(bs[:, burn:].reshape(-1, 1), ["b_delta"])["b_delta"],
        production_eligible=bool(diagnostics["passed"]),
        interpretation="Assumed uncertainty scenario unless an external measured error table is provided")
    write_json(out/"posterior_summary.json", result)
    np.savez_compressed(out/"posterior_draws.npz", samples=samples, names=np.array(spec.names))
    return result
