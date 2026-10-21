"""One source for tables/plots; never reads old manuscript posterior numbers."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from .common import read_json, write_json


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def report(out, config):
    out = Path(out)
    dest = out/"report"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((out/"mle").glob("**/*.json")):
        item = read_json(path)
        if not isinstance(item, dict) or "result" not in item:
            continue
        r = item["result"]
        if r["spec"]["beta_prior"] == "gaussian":
            continue
        case = str(path.parent.relative_to(out/"mle"))
        if case == "delta_offsets":
            case += "/"+"_".join(path.stem.split("_")[:2])
        rows.append(dict(case=case,
            framework=r["spec"]["framework"], beta_prior=r["spec"]["beta_prior"],
            N=r["N"], k=r["k"], lnL=r["lnL"], AIC=r["AIC"], BIC=r["BIC"],
            multi_start_lnL_range=r["multi_start_lnL_range"],
            **dict(zip(r["names"], r["theta"]))))
    if rows:
        df = pd.DataFrame(rows)
        df["delta_BIC_vs_free"] = np.nan
        for keys, group in df.groupby(["case", "framework"]):
            free = group[group.beta_prior == "flat"]
            if len(free) == 1:
                df.loc[group.index, "delta_BIC_vs_free"] = group.BIC-free.BIC.iloc[0]
        df.to_csv(dest/"likelihood_comparisons.csv", index=False)
        cols = ["case", "framework", "beta_prior", "N", "lnL", "BIC", "delta_BIC_vs_free"]
        (dest/"likelihood_comparisons.tex").write_text(df[cols].to_latex(index=False,
            float_format="%.3f", escape=True), encoding="utf-8")
        primary = df[df.case == "primary"]
        if {"P2", "P3"}.issubset(set(primary.framework)):
            paired = primary.pivot(index="beta_prior", columns="framework", values=["lnL", "BIC"])
            compare = pd.DataFrame({"P3_minus_P2_lnL": paired["lnL"]["P3"]-paired["lnL"]["P2"],
                                   "P3_minus_P2_BIC": paired["BIC"]["P3"]-paired["BIC"]["P2"]})
            compare.to_csv(dest/"P3_vs_P2_by_beta.csv")
    posteriors = []
    for path in sorted((out/"chains").glob("**/posterior_summary.json")):
        r = read_json(path)
        for name, q in r["quantiles"].items():
            posteriors.append(dict(case=str(path.parent.relative_to(out/"chains")),
                framework=r["spec"]["framework"], beta_prior=r["spec"]["beta_prior"],
                parameter=name, **q, production_eligible=r["production_eligible"]))
    if posteriors:
        pd.DataFrame(posteriors).to_csv(dest/"posterior_parameters.csv", index=False)
        _posterior_plots(out, dest, config)
        _trace_plots(out, dest)
    delta_rows = []
    for path in sorted((out/"delta").glob("**/posterior_summary.json")):
        r = read_json(path)
        for name, q in r["quantiles"].items():
            delta_rows.append(dict(scenario=r["scenario"]["name"], beta_prior=r["spec"]["beta_prior"],
                parameter=name, **q, production_eligible=r["production_eligible"]))
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(dest/"delta_uncertainty_posteriors.csv", index=False)
    tracerfile = out/"tracer"/"per_lens_tracer_comparison.csv"
    if tracerfile.exists():
        tr = pd.read_csv(tracerfile)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for name, group in tr.groupby("tracer"):
            ax.hist(group.fractional_difference, bins=20, histtype="step", label=name)
        ax.set(xlabel=r"$\sigma_{\rm curved}/\sigma_{\rm powerlaw}-1$", ylabel="Number of lenses")
        ax.legend(); _save(fig, dest/"tracer_sensitivity")
        fits = read_json(out/"tracer"/"all_fits.json")
        rows = [dict(tracer=r["spec"]["tracer"], beta_prior=r["spec"]["beta_prior"],
            N=r["N"], lnL=r["lnL"], AIC=r["AIC"], BIC=r["BIC"],
            **dict(zip(r["names"], r["theta"]))) for r in fits]
        trmle = pd.DataFrame(rows)
        trmle["delta_BIC_vs_free"] = np.nan
        for name, group in trmle.groupby("tracer"):
            free_bic = group[group.beta_prior == "flat"].BIC.iloc[0]
            trmle.loc[group.index, "delta_BIC_vs_free"] = group.BIC-free_bic
        trmle.to_csv(dest/"tracer_likelihood_comparisons.csv", index=False)
    summaries = []
    for path in sorted(out.glob("**/posterior_summary.json")):
        r = read_json(path)
        summaries.append(dict(path=str(path.relative_to(out)), passed=r["production_eligible"]))
    status = dict(profile=config["profile"], posterior_runs=summaries,
        all_posterior_diagnostics_passed=bool(summaries and all(s["passed"] for s in summaries)),
        smoke_results_for_publication=False if config["profile"] == "smoke" else None,
        notes=["A chain stopping at the requested length does not establish convergence.",
               "Plots/tables derive only from this output directory.",
               "Tracer and delta scenarios require interpretation alongside their assumed inputs."])
    write_json(dest/"readiness.json", status)
    return status


def _trace_plots(out, dest):
    import emcee
    for sidecar in sorted((out/"chains").glob("**/run.json")):
        folder = sidecar.parent
        if not (folder/"posterior_summary.json").exists():
            continue
        summary = read_json(folder/"posterior_summary.json")
        names = summary["names"]
        fig, axes = plt.subplots(len(names), 1, figsize=(8, 2*len(names)), squeeze=False)
        for j, filename in enumerate(sorted(folder.glob("ensemble_*.h5"))):
            chain = emcee.backends.HDFBackend(str(filename), read_only=True).get_chain()
            stride = max(1, len(chain)//1500)
            time = np.arange(len(chain))[::stride]
            for i, name in enumerate(names):
                # One fixed walker per independent ensemble: useful for the
                # same trajectories used in the between-run R-hat check.
                axes[i, 0].plot(time, chain[::stride, 0, i], lw=.5, label=f"ensemble {j+1}, walker 0")
                axes[i, 0].axvline(summary["diagnostics"]["burn"], color="k", ls=":", lw=.7)
                axes[i, 0].set_ylabel(name)
        axes[0, 0].legend(fontsize=7, ncol=2)
        axes[-1, 0].set_xlabel("Step")
        label = "_".join(folder.relative_to(out/"chains").parts)
        _save(fig, dest/f"trace_{label}")


def _posterior_plots(out, dest, cfg):
    labels = {"gamma0": r"$\gamma_0$", "gamma_z": r"$\gamma_z$", "beta": r"$\beta$", "f_int": r"$f_{\rm int}$"}
    for framework in cfg["frameworks"]:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        any_chain, all_passed = False, True
        for prior in cfg["beta_priors"]:
            folder = out/"chains"/"primary"/f"{framework}_{prior}"
            if not (folder/"posterior_draws.npz").exists():
                continue
            any_chain = True
            passed = read_json(folder/"posterior_summary.json")["production_eligible"]
            all_passed &= passed
            with np.load(folder/"posterior_draws.npz") as z:
                samples, names = z["samples"], z["names"].tolist()
            for name, ax in zip(labels, axes.flat):
                if name in names:
                    v = samples[:, names.index(name)]
                    # Deterministic subsample keeps plotting cost bounded.
                    v = v[::max(1, len(v)//10000)]
                    grid = np.linspace(np.quantile(v, .001), np.quantile(v, .999), 300)
                    ax.plot(grid, gaussian_kde(v)(grid), label=prior)
                elif name == "beta":
                    ax.axvline(0 if prior == "fixed0" else .18, ls="--", lw=1, label=prior+" (fixed)")
                ax.set_xlabel(labels[name]); ax.set_ylabel("Marginal density")
        if any_chain:
            axes[0, 0].legend(fontsize=8)
            axes[1, 0].legend(fontsize=8)
            if not all_passed or cfg["profile"] == "smoke":
                fig.suptitle("DIAGNOSTIC ONLY: convergence not established", color="firebrick", fontsize=10)
            _save(fig, dest/f"{framework}_marginals_including_beta")
        else:
            plt.close(fig)
    # Main-text corner candidates: P2 flat and Gaussian only. All chains remain
    # available in HDF5; no repeated corner gallery is forced into the paper.
    for prior in ["flat", "gaussian"]:
        folder = out/"chains"/"primary"/f"P2_{prior}"
        if not (folder/"posterior_draws.npz").exists():
            continue
        with np.load(folder/"posterior_draws.npz") as z:
            s, names = z["samples"], z["names"].tolist()
        s = s[::max(1, len(s)//15000)]
        ndim = len(names)
        fig, axes = plt.subplots(ndim, ndim, figsize=(8, 8))
        for i in range(ndim):
            for j in range(ndim):
                ax = axes[i, j]
                if i < j:
                    ax.axis("off")
                elif i == j:
                    ax.hist(s[:, i], bins=40, density=True, histtype="step")
                else:
                    ax.hist2d(s[:, j], s[:, i], bins=40, cmap="Blues")
                if i == ndim-1: ax.set_xlabel(labels.get(names[j], names[j]))
                if j == 0: ax.set_ylabel(labels.get(names[i], names[i]))
                ax.tick_params(labelsize=6)
        if not read_json(folder/"posterior_summary.json")["production_eligible"] or cfg["profile"] == "smoke":
            fig.suptitle("DIAGNOSTIC ONLY", color="firebrick")
        _save(fig, dest/f"P2_corner_{prior}")
