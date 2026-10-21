"""Convergence checks without treating interacting ensemble walkers as chains."""
import warnings
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _rhat_basic(x):
    # x has independent-chain, draw dimensions.
    m, n = x.shape
    W = np.mean(np.var(x, axis=1, ddof=1))
    B = n*np.var(np.mean(x, axis=1), ddof=1)
    if W == 0:
        return 1.0 if B == 0 else np.inf
    return np.sqrt(((n-1)/n*W+B/n)/W)


def rank_split_rhat(x):
    """Vehtari et al. rank-normalized and folded split R-hat, max of both."""
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 8:
        return np.inf
    half = x.shape[1]//2
    split = np.concatenate([x[:, :half], x[:, -half:]], axis=0)
    def normalized(a):
        rank = rankdata(a.ravel(), method="average").reshape(a.shape)
        return ndtri((rank-.375)/(a.size+.25))
    return float(max(_rhat_basic(normalized(split)),
                     _rhat_basic(normalized(np.abs(split-np.median(split))))))


def diagnose(chains, burn, thresholds, ensemble=True):
    """chains: independent ensemble/chain, step, walker, parameter.

    R-hat compares a fixed walker across independent ensembles; take the worst
    over up to four fixed indices. Walkers within an ensemble are NOT declared
    independent chains. The emcee ensemble ESS is only an autocorrelation-based
    approximation; passing all checks still requires inspecting trace plots.
    """
    import emcee
    a = np.asarray(chains)
    if a.ndim == 3:
        a = a[:, :, None, :]
    m, steps, walkers, ndim = a.shape
    post = a[:, burn:]
    kept = post.shape[1]
    if kept < 8:
        return dict(passed=False, reason="Insufficient post-burn draws", burn=burn)
    rhat = []
    for p in range(ndim):
        rhat.append(max(rank_split_rhat(post[:, :, w, p]) for w in
                        np.unique(np.linspace(0, walkers-1, min(4, walkers), dtype=int))))
    tau = np.full((m, ndim), np.inf)
    stability = np.full((m, ndim), np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(m):
            try:
                full = emcee.autocorr.integrated_time(post[i], tol=0, quiet=True)
                early = emcee.autocorr.integrated_time(post[i, :max(8, int(.8*kept))],
                                                      tol=0, quiet=True)
                if np.all(full > 0) and np.isfinite(full).all():
                    tau[i] = full
                    stability[i] = np.abs(early-full)/full
            except (ValueError, emcee.autocorr.AutocorrError):
                pass
    ess = np.sum(kept*walkers/tau, axis=0)
    long_enough = np.all(kept >= thresholds.get("tau_multiple", 100)*tau)
    stable = np.all(stability < thresholds.get("tau_stability", .05))
    passed = bool(m >= 2 and np.all(np.array(rhat) < thresholds.get("rhat_max", 1.01))
        and np.all(ess >= thresholds.get("ess_min", 2000)) and long_enough and stable)
    return dict(passed=passed, independent_ensembles_or_chains=m, total_steps=steps,
        burn=burn, retained_steps_per_chain=kept, rank_split_rhat=rhat,
        tau_by_independent_run=tau, tau_relative_change=stability,
        approximate_ESS=ess, long_enough=bool(long_enough), stable_tau=bool(stable),
        thresholds=thresholds, walkers_per_ensemble=walkers,
        diagnostic_convention="Rhat: same fixed walker across independent ensembles; max over <=4 walkers")


def quantiles(samples, names):
    q = np.quantile(samples, [.16, .5, .84], axis=0)
    return {name: {"q16": q[0, i], "median": q[1, i], "q84": q[2, i],
                   "minus": q[1, i]-q[0, i], "plus": q[2, i]-q[1, i]}
            for i, name in enumerate(names)}
