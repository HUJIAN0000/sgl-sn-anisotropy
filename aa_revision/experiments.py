"""Tracer sensitivity, coherent delta shifts and controlled recovery mocks."""
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from .model import Spec, loglike, mean_and_cov, gamma_values
from .jeans import make_engine
from .fitting import fit_mle
from .common import write_json, read_json, fingerprint, code_hash


def cached_fit(data, spec, options, path, seed, resume=False, engine=None, delta=None,
               extra=None):
    path = Path(path)
    signature = fingerprint(dict(data_digest=data.digest, spec=asdict(spec), options=options,
        code_hash=code_hash(), seed=seed, delta=delta, extra=extra))
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        old = read_json(path)
        if old["signature"] != signature:
            raise ValueError(f"Fit fingerprint changed: {path}")
        return old["result"]
    result = fit_mle(data, spec, options, seed=seed, engine=engine, delta=delta)
    write_json(path, dict(signature=signature, result=result))
    return result


def tracer_sensitivity(data, cfg, optimizer, out, seed, domain, resume=False):
    out = Path(out)
    rows, predictions = [], []
    for j, tracer in enumerate(cfg["profiles"]):
        engine = make_engine(data, cfg, tracer)
        for prior in ["flat", "fixed0", "fixed018"]:
            spec = Spec("P2", prior, domain, tracer)
            row = cached_fit(data, spec, optimizer, out/f"{tracer}_{prior}.json",
                seed+13*j, resume, engine=engine, extra=cfg)
            row.update(case=f"tracer_{tracer}")
            rows.append(row)
        # Compare model predictions at the SAME gamma,beta, not different optima.
        truth = cfg["comparison_parameters"]
        spec = Spec("P2", "flat", domain, tracer)
        theta = [truth[n] for n in spec.names]
        curved, _ = mean_and_cov(theta, data, spec, engine=engine)
        powerlaw, _ = mean_and_cov(theta, data, replace(spec, tracer="powerlaw"))
        for i in range(data.n):
            predictions.append({"Lens name": data.table["Lens name"].iloc[i],
                "tracer": tracer, "sigma_curved": curved[i], "sigma_powerlaw": powerlaw[i],
                "fractional_difference": curved[i]/powerlaw[i]-1})
        # Every real-catalogue numerical fit gets an independent refined-grid
        # likelihood check. This is not merely a one-lens SIS test.
        best = rows[-3]
        fine_cfg = dict(cfg, nrad=2*int(cfg["nrad"]), extent=10*float(cfg["extent"]),
                        abel_nodes=2*int(cfg.get("abel_nodes", 96)),
                        angular_nodes=2*int(cfg.get("angular_nodes", 128)))
        fine_engine = make_engine(data, fine_cfg, tracer)
        coarse, _ = mean_and_cov(best["theta"], data, Spec("P2", "flat", domain, tracer), engine=engine)
        fine, _ = mean_and_cov(best["theta"], data, Spec("P2", "flat", domain, tracer), engine=fine_engine)
        refined_ll = loglike(best["theta"], data, Spec("P2", "flat", domain, tracer), engine=fine_engine)
        check = dict(max_fractional_sigma_change=np.max(np.abs(fine/coarse-1)),
            absolute_lnL_change=abs(refined_ll-best["lnL"]),
            passed=bool(np.max(np.abs(fine/coarse-1)) < 5e-4 and abs(refined_ll-best["lnL"]) < .1),
            interpretation="Grid refinement at the best-fit point; profiles and PSF are assumed scenarios")
        write_json(out/f"{tracer}_quadrature_check.json", check)
    pd.DataFrame(predictions).to_csv(out/"per_lens_tracer_comparison.csv", index=False)
    write_json(out/"all_fits.json", rows)
    return rows


def coherent_offsets(data, offsets, options, out, seed, domain, resume=False):
    rows = []
    for shift in offsets:
        for prior in ["flat", "fixed0", "fixed018"]:
            spec = Spec("P2", prior, domain)
            row = cached_fit(data, spec, options, Path(out)/f"offset_{shift:+.3f}_{prior}.json",
                seed, resume, delta=data.table.delta.to_numpy()+shift)
            row.update(case=f"delta_offset_{shift:+.3f}")
            rows.append(row)
    write_json(Path(out)/"all_fits.json", rows)
    return rows


def wilson(k, n, level=.95):
    if not n:
        return [None, None]
    z = norm.ppf((1+level)/2)
    p, den = k/n, 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0., center-half), min(1., center+half)]


def run_mocks(data, cfg, optimizer, jeans_cfg, out, seed, domain, resume=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(data_digest=data.digest, code_hash=code_hash(), config=cfg,
                optimizer=optimizer, jeans=jeans_cfg, seed=seed, domain=domain)
    sig = fingerprint(meta)
    mf = out/"run.json"
    if mf.exists():
        if not resume or read_json(mf)["signature"] != sig:
            raise ValueError("Mock results exist or fingerprint changed; use a new output")
    else:
        write_json(mf, dict(signature=sig, metadata=meta))
    truth_spec = Spec("P2", "flat", domain)
    completed = []
    for tracer_index, tracer in enumerate(cfg["truth_tracers"]):
        engine = None if tracer == "powerlaw" else make_engine(data, jeans_cfg, tracer)
        for beta_index, beta in enumerate(cfg["truth_betas"]):
            parameters = dict(cfg["truth"], beta=beta)
            theta = np.array([parameters[n] for n in truth_spec.names])
            generating_spec = replace(truth_spec, tracer=tracer)
            mean, _ = mean_and_cov(theta, data, generating_spec, engine=engine)
            if mean is None:
                raise ValueError("Invalid mock truth")
            # Fix the variance anchor to the true predicted dispersion. Keep
            # this same anchor when fitting y_mock, so these simulations have
            # a coherent conditional generative likelihood, C(theta; anchor).
            base = replace(data, anchor=mean.copy())
            mean, covariance = mean_and_cov(theta, base, generating_spec, engine=engine)
            factor = np.linalg.cholesky(covariance)
            for repetition in range(int(cfg["repetitions"])):
                target = out/f"{tracer}_b{beta:+.3f}_{repetition:05d}.json"
                if target.exists() and resume:
                    completed.append(read_json(target))
                    continue
                # Same standard-normal variates for null/alternative and tracer
                # scenarios in a given realization; no scatter only in H1.
                rng = np.random.default_rng(np.random.SeedSequence([seed, repetition]))
                observed = mean+factor @ rng.normal(size=data.n)
                mock = base.observed(observed)
                fits = []
                fitting_tracers = ["powerlaw"] if tracer == "powerlaw" else ["powerlaw", tracer]
                for fitting_tracer in fitting_tracers:
                    fitting_engine = None if fitting_tracer == "powerlaw" else engine
                    for prior in ["flat", "fixed0"]:
                        spec = Spec("P2", prior, domain, fitting_tracer)
                        fit = fit_mle(mock, spec, optimizer,
                            seed=seed+10000*tracer_index+101*beta_index+repetition,
                            engine=fitting_engine)
                        fits.append(fit)
                row = dict(truth_tracer=tracer, truth_beta=beta, repetition=repetition,
                    true_parameters=parameters, fits=fits,
                    noise="same conditional noise prescription in H0/H1; fixed true-sigma variance anchor")
                write_json(target, row)
                completed.append(row)
                print(f"mock {tracer}, beta={beta}, {repetition+1}/{cfg['repetitions']}", flush=True)
    summaries, tidy = [], []
    for row in completed:
        for fit in row["fits"]:
            tidy.append(dict(truth_tracer=row["truth_tracer"], truth_beta=row["truth_beta"],
                repetition=row["repetition"], fitting_tracer=fit["spec"]["tracer"],
                beta_prior=fit["spec"]["beta_prior"], lnL=fit["lnL"], BIC=fit["BIC"],
                optimizer_agreement_checked=fit["optimizer_agreement_checked"],
                **dict(zip(fit["names"], fit["theta"]))))
    df = pd.DataFrame(tidy)
    df.to_csv(out/"all_mock_fits.csv", index=False)
    groupkeys = ["truth_tracer", "truth_beta", "fitting_tracer"]
    for keys, group in df.groupby(groupkeys, sort=False):
        free = group[group.beta_prior == "flat"].set_index("repetition")
        null = group[group.beta_prior == "fixed0"].set_index("repetition")
        difference = null.BIC-free.BIC
        thresholds = {}
        for t in cfg["delta_bic_thresholds"]:
            k, n = int((difference >= t).sum()), len(difference)
            thresholds[str(t)] = dict(count=k, repetitions=n, fraction=k/n,
                                      wilson_95=wilson(k, n))
        summaries.append(dict(zip(groupkeys, keys), repetitions=len(free),
            beta_mean=float(free.beta.mean()), beta_bias=float(free.beta.mean()-keys[1]),
            gamma0_bias=float(free.gamma0.mean()-cfg["truth"]["gamma0"]),
            gamma_z_bias=float(free.gamma_z.mean()-cfg["truth"]["gamma_z"]),
            mean_delta_BIC=float(difference.mean()), detection=thresholds,
            optimizer_agreement_checked=bool(group.optimizer_agreement_checked.all()),
            production_eligible=bool(len(free) >= 200 and group.optimizer_agreement_checked.all()),
            interpretation="Conditional simulation; tracer misspecification compared with truth-matched recovery"))
    write_json(out/"mock_summary.json", summaries)
    return summaries
