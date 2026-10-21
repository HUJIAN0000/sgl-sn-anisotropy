"""Shared CLI orchestration. No module imports the old all-results scripts."""
from dataclasses import asdict
from pathlib import Path
import argparse
import copy
from .common import ROOT, read_json, write_json, sha256, environment
from .matching import prepare
from .model import load_data, Spec
from .experiments import cached_fit, coherent_offsets, tracer_sensitivity, run_mocks
from .sampling import run_emcee
from .delta import run_latent
from .evidence import compare_priors
from .reporting import report


def _catalogue(raw, folder, matching, resume):
    mf = folder/"manifest.json"
    if mf.exists():
        if not resume:
            raise FileExistsError(f"Catalogue exists: {folder}; add --resume")
        saved = read_json(mf)
        for key, value in matching.items():
            if key == "time_limit":
                continue
            actual = saved["Omega_m" if key == "omega" else key]
            if actual != value:
                raise ValueError("Pairing settings changed; use a new output directory")
        for name, digest in saved["input_sha256"].items():
            if sha256(raw/name) != digest:
                raise ValueError("Raw inputs changed; refusing to reuse a frozen catalogue")
    else:
        info = prepare(raw, folder, matching)
        print(f"Frozen {info['maximum_cardinality']} lenses; cost={info['geometric_cost']:.12g}", flush=True)
    return load_data(folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description="aa61321-26 additional calculations")
    parser.add_argument("task", nargs="?", default="all", choices=["all", "prepare", "mle", "sample", "delta", "tracer", "mocks", "evidence", "report"])
    parser.add_argument("--config", type=Path, default=ROOT/"configs"/"smoke.json")
    parser.add_argument("--output", type=Path, default=Path("results/smoke"))
    parser.add_argument("--raw-dir", type=Path, default=ROOT/"data"/"raw")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--processes", type=int, default=1, help="Parallel likelihoods within each emcee ensemble")
    parser.add_argument("--delta-error-table", type=Path, help="CSV: Lens name,delta_sigma; replaces assumed independent errors")
    args = parser.parse_args(argv)
    cfg = read_json(args.config)
    if cfg["profile"] not in ("smoke", "production"):
        raise ValueError("profile must be smoke or production")
    if args.processes < 1:
        raise ValueError("processes must be >=1")
    out, raw = args.output.resolve(), args.raw_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Each invocation's full configuration is recorded separately, without
    # replacing a production configuration by the settings of a later report.
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(out/"invocations"/f"{stamp}_{args.task}.json",
        dict(config=cfg, arguments=vars(args), environment=environment()))
    tasks = ["prepare", "mle", "sample", "delta", "tracer", "mocks", "evidence", "report"] if args.task == "all" else [args.task]
    if tasks == ["report"]:
        print(report(out, cfg)); return
    data = _catalogue(raw, out/"catalogues"/"primary", cfg["matching"], args.resume)
    domain, seed = cfg["domain"], int(cfg["seed"])
    def get_mle(d, framework, prior, case="primary"):
        # Flat and Gaussian beta priors have the same maximum likelihood.
        mle_prior = "flat" if prior == "gaussian" else prior
        spec = Spec(framework, mle_prior, domain)
        path = out/"mle"/case/f"{framework}_{mle_prior}.json"
        result = cached_fit(d, spec, cfg["optimizer"], path, seed,
                            resume=args.resume or path.exists())
        return result
    variants = {}
    for variant in cfg.get("matching_variants", []):
        matching = dict(cfg["matching"], **variant["matching"])
        variants[variant["name"]] = _catalogue(raw, out/"catalogues"/variant["name"], matching, args.resume)
    if "mle" in tasks:
        for framework in cfg["frameworks"]:
            for prior in ["flat", "fixed0", "fixed018"]:
                r = get_mle(data, framework, prior)
                print(f"{framework} {prior}: lnL={r['lnL']:.6f}, BIC={r['BIC']:.6f}", flush=True)
        for name, d in variants.items():
            for prior in ["flat", "fixed0", "fixed018"]:
                get_mle(d, "P2", prior, name)
        coherent_offsets(data, cfg.get("delta_offsets", []), cfg["optimizer"],
            out/"mle"/"delta_offsets", seed, domain, resume=args.resume)
    if "sample" in tasks:
        for case, d in [("primary", data)]+(list(variants.items()) if cfg.get("variant_posteriors", False) else []):
            frameworks = cfg["frameworks"] if case == "primary" else ["P2"]
            for framework in frameworks:
                for prior in cfg["beta_priors"]:
                    spec = Spec(framework, prior, domain)
                    run_emcee(d, spec, get_mle(d, framework, prior, case),
                        out/"chains"/case/f"{framework}_{prior}", cfg["mcmc"], seed,
                        resume=args.resume, processes=args.processes)
    if "delta" in tasks:
        for scenario in cfg.get("delta_scenarios", []):
            for prior in cfg.get("delta_beta_priors", ["flat", "fixed0"]):
                spec = Spec("P2", prior, domain)
                run_latent(data, spec, get_mle(data, "P2", prior),
                    out/"delta"/scenario["name"]/prior, cfg["latent_mcmc"], scenario,
                    seed+711, args.resume, args.delta_error_table)
    if "tracer" in tasks:
        tracer_sensitivity(data, cfg["jeans"], cfg["tracer_optimizer"], out/"tracer",
                           seed, domain, args.resume)
    if "mocks" in tasks:
        run_mocks(data, cfg["mocks"], cfg["mock_optimizer"], cfg["jeans"],
                  out/"mocks", seed+911, domain, args.resume)
    if "evidence" in tasks:
        for framework in cfg["frameworks"]:
            flat = out/"chains"/"primary"/f"{framework}_flat"
            gauss = out/"chains"/"primary"/f"{framework}_gaussian"
            if not (flat/"posterior_summary.json").exists() or not (gauss/"posterior_summary.json").exists():
                raise FileNotFoundError("Run flat AND Gaussian posterior chains before evidence")
            compare_priors(flat, gauss, cfg["mcmc"], out/"evidence"/f"{framework}.json",
                           data=data, normalization_cfg=cfg.get("evidence", {}))
    if "report" in tasks:
        print(report(out, cfg))
