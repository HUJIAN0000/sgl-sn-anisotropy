"""Gaussian-vs-flat beta prior evidence by posterior reweighting.

For normalized base box priors and common physical support S:
Z_G^S/Z_U^S = E_post,U[pi_G^base/pi_U^base] P_U(S)/P_G(S).
Importance ESS measures overlap, not MCMC mixing.
"""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import qmc
from .common import read_json, write_json
from .sampling import load_ensemble_chains


def log_prior_ratio(beta):
    norm = ndtr((1-.18)/.13)-ndtr((-1-.18)/.13)
    return np.log(2)-np.log(.13*np.sqrt(2*np.pi)*norm)-.5*((beta-.18)/.13)**2


def support_normalization(data, spec, power=16, repeats=8, seed=91731):
    """Normalize both joint priors after conditioning on the same physical domain.

    Hard per-lens gamma/xi cuts couple beta to nuisance parameters, so a
    normalized 1-D beta prior alone is insufficient for a Bayes factor.
    Scrambled Sobol replicates estimate P_U(S), P_G(S) using common variates.
    f_int integrates out; it does not enter the convergence conditions.
    """
    if spec.domain != "convergent_jeans" or spec.tracer != "powerlaw":
        raise ValueError("Evidence support normalization is implemented for convergent power-law Jeans only")
    delta = data.table.delta.to_numpy()
    if np.any(delta <= 1) or np.any(delta >= 3):
        raise ValueError("No convergent power-law prior support for these delta values")
    bounds, names = spec.bounds[:-1], spec.names[:-1]
    values = []
    zl, predictor = data.table.zl.to_numpy(), data.predictor
    for r in range(int(repeats)):
        unit = qmc.Sobol(d=len(names), scramble=True, seed=seed+r).random_base2(int(power))
        points = bounds[:, 0]+unit*(bounds[:, 1]-bounds[:, 0])
        accepted, weighted = 0., 0.
        for start in range(0, len(points), 4096):
            p = points[start:start+4096]
            gamma = p[:, 0, None]+p[:, 1, None]*zl
            if "gamma_s" in names:
                gamma += p[:, names.index("gamma_s"), None]*predictor
            beta = p[:, names.index("beta")]
            xi = gamma+delta-2
            valid = ((gamma > 1.05).all(axis=1) & (gamma < 2.95).all(axis=1) &
                (xi > 1).all(axis=1) & (xi < 3).all(axis=1) &
                (xi > 2*beta[:, None]).all(axis=1))
            accepted += valid.sum()
            weighted += np.exp(log_prior_ratio(beta[valid])).sum()
        values.append([accepted/len(points), weighted/len(points)])
    values = np.asarray(values)
    if np.any(values <= 0):
        raise RuntimeError("Too few prior-support draws; increase evidence normalization samples")
    means = values.mean(axis=0)
    correction = np.log(means[0]/means[1])
    replicate_logs = np.log(values[:, 0]/values[:, 1])
    se = np.std(replicate_logs, ddof=1)/np.sqrt(repeats) if repeats > 1 else np.inf
    return dict(P_support_flat=float(means[0]), P_support_gaussian=float(means[1]),
        log_P_flat_over_P_gaussian=float(correction), log_correction_se=float(se),
        sobol_power=int(power), independent_scrambles=int(repeats), replicates=values.tolist(),
        convention="Both joint priors normalized conditional on the same gamma/xi convergence support")


def _estimate(logweights, block_length, rng, bootstrap=1000):
    # ensemble, retained step, walker. Resample whole time blocks, preserving
    # within-ensemble dependence. Each independent ensemble contributes equally.
    m, n, walkers = logweights.shape
    logmean = logsumexp(logweights)-np.log(logweights.size)
    logess = 2*logsumexp(logweights)-logsumexp(2*logweights)
    block_length = max(1, int(block_length))
    blocks = []
    for i in range(m):
        values = []
        for a in range(0, n-block_length+1, block_length):
            block = logweights[i, a:a+block_length]
            values.append(logsumexp(block)-np.log(block.size))
        blocks.append(np.array(values))
    if any(len(b) < 2 for b in blocks):
        return dict(log_ratio=float(logmean), importance_ESS=float(np.exp(logess)),
                    block_length=block_length, blocks_per_ensemble=[len(b) for b in blocks],
                    bootstrap_log_ratio_se=None, bootstrap_interval_95=None)
    draws = []
    for _ in range(bootstrap):
        means = [logsumexp(rng.choice(b, size=len(b), replace=True))-np.log(len(b)) for b in blocks]
        draws.append(logsumexp(means)-np.log(m))
    return dict(log_ratio=float(logmean), importance_ESS=float(np.exp(logess)),
        block_length=block_length, blocks_per_ensemble=[len(b) for b in blocks],
        bootstrap_log_ratio_se=float(np.std(draws, ddof=1)),
        bootstrap_interval_95=np.quantile(draws, [.025, .975]).tolist())


def compare_priors(flat_dir, gaussian_dir, cfg, out, data, normalization_cfg=None, seed=61731):
    flat_dir, gaussian_dir = Path(flat_dir), Path(gaussian_dir)
    u, g = [read_json(p/"posterior_summary.json") for p in [flat_dir, gaussian_dir]]
    um, gm = [read_json(p/"run.json")["metadata"] for p in [flat_dir, gaussian_dir]]
    checks = [u["data_digest"] == g["data_digest"],
              um["code_hash"] == gm["code_hash"],
              all(u["spec"][k] == g["spec"][k] for k in ["framework", "domain", "tracer"]),
              u["spec"]["beta_prior"] == "flat", g["spec"]["beta_prior"] == "gaussian"]
    if not all(checks):
        raise ValueError("Evidence identity requires identical data/model/nuisance priors")
    if data.digest != u["data_digest"]:
        raise ValueError("Support-normalization dataset differs from posterior dataset")
    from .model import Spec
    norm_cfg = normalization_cfg or {}
    normalization = support_normalization(data, Spec(**u["spec"]),
        power=int(norm_cfg.get("power", 16)), repeats=int(norm_cfg.get("repeats", 8)))
    ensemble_count = int(cfg["ensembles"])
    forward = load_ensemble_chains(flat_dir, ensemble_count)[:, int(cfg["burn"]):, :,
                    u["names"].index("beta")]
    reverse = load_ensemble_chains(gaussian_dir, ensemble_count)[:, int(cfg["burn"]):, :,
                    g["names"].index("beta")]
    def block(summary, n):
        tau = np.array(summary["diagnostics"].get("tau_by_independent_run", []), dtype=float)
        tau = tau[np.isfinite(tau)]
        return max(1, int(np.ceil(2*tau.max()))) if len(tau) else max(1, int(np.sqrt(n)))
    rng = np.random.default_rng(seed)
    fwd = _estimate(log_prior_ratio(forward), block(u, forward.shape[1]), rng)
    rev = _estimate(-log_prior_ratio(reverse), block(g, reverse.shape[1]), rng)
    # rev estimates log(Z_U/Z_G); negate for a consistency comparison.
    discrepancy = abs(fwd["log_ratio"]+rev["log_ratio"])
    errors = [fwd["bootstrap_log_ratio_se"], rev["bootstrap_log_ratio_se"]]
    consistent = (all(e is not None for e in errors) and
                  discrepancy <= 3*np.hypot(*errors))
    sufficient = (u["production_eligible"] and g["production_eligible"] and consistent
        and min(fwd["importance_ESS"], rev["importance_ESS"]) >= 1000
        and min(fwd["blocks_per_ensemble"]+rev["blocks_per_ensemble"]) >= 20
        and normalization["log_correction_se"] < .02)
    result = dict(log_B_G_over_U=fwd["log_ratio"]+normalization["log_P_flat_over_P_gaussian"],
        forward=fwd, reverse=rev, prior_support_normalization=normalization,
        reverse_consistent=bool(consistent), production_eligible=bool(sufficient),
        caveat="Do not interpret an overlap/chain diagnostic failure as an evidence measurement",
        meaning="Gaussian beta prior versus flat beta prior, not fixed beta=0.18 versus free beta")
    write_json(out, result)
    return result
