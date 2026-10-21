"""Independent emcee ensembles with persistent, fingerprinted HDF5 chains."""
from dataclasses import asdict
from pathlib import Path
from functools import partial
from multiprocessing import get_context
import numpy as np
from scipy.optimize import minimize
from .common import read_json, write_json, fingerprint, code_hash, environment
from .model import logposterior
from .diagnostics import diagnose, quantiles


def run_emcee(data, spec, mle, out, cfg, seed, resume=False, processes=1, engine=None,
              engine_config=None):
    import emcee
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ndim, walkers = len(spec.names), int(cfg["walkers"])
    if walkers < 2*ndim or walkers % 2:
        raise ValueError("Use an even walker count >= 2*ndim")
    posterior = partial(logposterior, data=data, spec=spec, engine=engine)
    # A Gaussian beta prior can put the MAP far from the unconstrained MLE.
    opt = minimize(lambda x: -posterior(x) if np.isfinite(posterior(x)) else 1e100,
                   mle["theta"], method="Nelder-Mead", bounds=spec.bounds,
                   options={"maxiter": 6000, "xatol": 1e-7, "fatol": 1e-7})
    center = opt.x if np.isfinite(posterior(opt.x)) else np.array(mle["theta"])
    metadata = dict(data_digest=data.digest, spec=asdict(spec), code_hash=code_hash(),
        walkers=walkers, ensembles=int(cfg["ensembles"]), seed=seed, burn=int(cfg["burn"]),
        engine_config=engine_config, diagnostics=cfg["diagnostics"], environment=environment())
    signature = fingerprint(metadata)
    sidecar = out / "run.json"
    if sidecar.exists():
        if not resume:
            raise FileExistsError(f"Chains exist in {out}; use --resume or a new output")
        if read_json(sidecar)["signature"] != signature:
            raise ValueError("Resume rejected: code, data, priors or sampler settings changed")
    else:
        if resume and list(out.glob("*.h5")):
            raise ValueError("Cannot resume unlabelled HDF5 files")
        write_json(sidecar, dict(signature=signature, metadata=metadata))
    pool = get_context("spawn").Pool(processes) if processes > 1 else None
    try:
        for ensemble in range(int(cfg["ensembles"])):
            rng = np.random.default_rng(seed+1009*ensemble)
            backend = emcee.backends.HDFBackend(str(out/f"ensemble_{ensemble:02d}.h5"))
            exists = backend.initialized and backend.iteration > 0
            if exists and not resume:
                raise FileExistsError("Refusing to overwrite a chain")
            if not exists:
                backend.reset(walkers, ndim)
                positions = []
                scales = np.array([.04 if n == "gamma0" else .15 if n != "f_int" else .012
                                   for n in spec.names])
                for _ in range(50000):
                    x = center + rng.normal(size=ndim)*scales
                    if np.isfinite(posterior(x)):
                        positions.append(x)
                    if len(positions) == walkers:
                        break
                if len(positions) != walkers:
                    raise RuntimeError("Could not initialize valid posterior walkers")
                state = np.asarray(positions)
            else:
                state = backend.get_last_sample()
            sampler = emcee.EnsembleSampler(walkers, ndim, posterior, backend=backend, pool=pool)
            if not exists:
                sampler.random_state = np.random.RandomState(seed+1009*ensemble).get_state()
            remaining = max(0, int(cfg["steps"])-backend.iteration)
            if remaining:
                for offset in range(0, remaining, 1000):
                    count = min(1000, remaining-offset)
                    state = sampler.run_mcmc(state, count, progress=bool(cfg.get("progress", True)))
                    print(f"sample {spec.tracer} {spec.beta_prior} ensemble {ensemble+1}: {backend.iteration}/{cfg['steps']}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return summarize_emcee(out, data, spec, cfg)


def load_ensemble_chains(out, ensembles):
    import emcee
    chains = []
    for i in range(ensembles):
        path = Path(out)/f"ensemble_{i:02d}.h5"
        if not path.exists():
            raise ValueError(f"Missing independent ensemble: {path}")
        chains.append(emcee.backends.HDFBackend(str(path), read_only=True).get_chain())
    if len({a.shape for a in chains}) != 1:
        raise ValueError(f"Ensemble shapes differ: {[a.shape for a in chains]}; resume all to a common length")
    return np.stack(chains)


def summarize_emcee(out, data, spec, cfg):
    import emcee
    chains = load_ensemble_chains(out, int(cfg["ensembles"]))
    burn = int(cfg["burn"])
    if burn >= chains.shape[1]:
        raise ValueError("Burn-in must be shorter than the chain")
    samples = chains[:, burn:].reshape(-1, chains.shape[-1])
    diagnostic = diagnose(chains, burn, cfg["diagnostics"])
    acceptance = []
    for i in range(int(cfg["ensembles"])):
        b = emcee.backends.HDFBackend(str(Path(out)/f"ensemble_{i:02d}.h5"), read_only=True)
        acceptance.append((b.accepted/b.iteration).tolist())
    result = dict(spec=asdict(spec), names=spec.names, data_digest=data.digest,
        quantiles=quantiles(samples, spec.names), diagnostics=diagnostic,
        acceptance_fraction=acceptance, production_eligible=bool(diagnostic["passed"]),
        count_retained_draws=len(samples))
    # gamma at the median lens redshift is more interpretable than gamma0.
    zp = float(np.median(data.table.zl))
    sp = float(np.median(data.predictor))
    pivot = samples[:, 0] + zp*samples[:, 1]
    if "gamma_s" in spec.names:
        pivot += sp*samples[:, spec.names.index("gamma_s")]
    result["pivot"] = dict(z=zp, log10_sigma_tilde=sp,
                           gamma=quantiles(pivot[:, None], ["gamma_pivot"])["gamma_pivot"])
    write_json(Path(out)/"posterior_summary.json", result)
    np.savez_compressed(Path(out)/"posterior_draws.npz", samples=samples,
                        names=np.asarray(spec.names), gamma_pivot=pivot)
    return result
