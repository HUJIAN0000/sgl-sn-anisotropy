# -*- coding: utf-8 -*-
"""
run_all_results0.03.py  --  3%-TOLERANCE one-shot reproduction.

Run this INSIDE the 3% sample folder (program + data in the same folder; all file
names are RELATIVE -- no absolute paths). It reproduces every code-derived result
of the paper for the delta d/d = 0.03 sample, AND draws the getdist triangle plots
via cosmo_tools.py (the same figures as test_beta_necessity_final_P2/P3).

Run once -> produces:
   Table 1  (P2 model comparison)           -> table1_p2_modelcomp.{csv,tex}
   Table 2  (P2 posteriors, emcee)          -> table2_p2_posteriors.csv
   Table 3  (P3 model comparison)           -> table3_p3_modelcomp.{csv,tex}
   Table B.1 (tolerance, 3% row only here) -> tableB1_tolerance.{csv,tex}
   Table C.1 (two-sided power analysis)     -> tableC1_power.csv
   Table D.1 (Omega_m robustness)           -> tableD1_omega.{csv,tex}
   (new)     delta_i sensitivity appendix   -> tableX_delta_sensitivity.csv
   getdist triangles via cosmo_tools:
       P2: P2_Free_Beta_FlatPrior_Opt_GetDist.pdf, ..._GaussPrior..., ..._Fixed_Beta_0...,
           ..._Fixed_Beta_0_18..., P2_Beta_Models_Comparison_Opt.pdf
       P3: P3_*_Opt_GetDist.pdf, P3_Beta_Models_Comparison_Opt.pdf
   full console log                         -> ALL_RESULTS.log

REQUIRED LOCAL FILES (produced by match_milp_headless0.03.py), all in THIS folder:
   matched_result_sgls_Om0.25.csv  matched_result_sgls_Om0.30.csv  matched_result_sgls_Om0.35.csv
   cov_sgls_{lens,source,cross}_Om0.25.txt  _Om0.30.txt  _Om0.35.txt
   cosmo_tools.py                 (needed for the getdist triangle plots)

PHYSICS is identical to test_beta_necessity_final_P2_opt.py / _P3_v4.py (same F,
predict_sigma, Jacobian, covariance, likelihood); P3 adds the
gamma_s*log10(Sigma_tilde) term to gamma(z). The MLE model-comparison tables use a
DE+Nelder-Mead global fit; the posteriors use emcee, exactly as in your scripts.

The 3% and 10% tolerances have their own folders/scripts (run_all_results0.03.py,
run_all_results0.10.py). This script is for 3% only, so Table B.1 fills the 3% row
(the other two rows come from those runs).

DEPENDENCIES:  numpy, pandas, scipy, emcee, getdist, matplotlib
   pip install numpy pandas scipy emcee getdist matplotlib

USAGE
-----
  1) FIRST run with QUICK_TEST = True  (toy sizes, ~minutes) to confirm it runs.
  2) Then set QUICK_TEST = False and run for the production numbers (can take
     a few hours, mostly the 400 power-analysis fits + the 8 MCMC chains).
  3) Each PART can be toggled independently below; each writes its outputs as
     soon as it finishes, so a late crash never loses earlier tables.

      python run_all_results.py
"""

import os, sys, time, json, tempfile, csv as _csv
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize as op
from scipy.integrate import quad
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse
from scipy.special import gamma as gamma_func

# ============================================================================
# 0. MASTER CONFIG  -- edit here
# ============================================================================
QUICK_TEST = True          # <<< set False for the real production run

# ---- input mode --------------------------------------------------------------
# RUN_PAIRING = False  (DEFAULT, RECOMMENDED): do NOT re-pair; consume the matched
#   sample produced by your own CosmoMatcher GUI (this is the exact sample the
#   paper is based on). Point EXISTING_BASELINE_* at those files.
# RUN_PAIRING = True : regenerate the sample with the built-in MILP matcher
#   (now fixed: relative cost + eps guard -> reproduces 107, deterministic).
RUN_PAIRING        = False

# 3%-tolerance run: program + data live in the SAME folder, RELATIVE names only.
# Files expected here (produced by match_milp_headless0.03.py):
#   matched_result_sgls_Om0.25.csv / _Om0.30.csv / _Om0.35.csv
#   cov_sgls_{lens,source,cross}_Om{0.25,0.30,0.35}.txt
def _samp(om):  # (csv, cov_lens, cov_source, cov_cross) -- RELATIVE names
    return (f"matched_result_sgls_Om{om}.csv",
            f"cov_sgls_lens_Om{om}.txt",
            f"cov_sgls_source_Om{om}.txt",
            f"cov_sgls_cross_Om{om}.txt")

# Baseline = 3%, Omega_m=0.30.
EXISTING_BASELINE_CSV = _samp("0.30")[0]
EXISTING_BASELINE_COV = _samp("0.30")[1:]

# Table D.1 (Omega_m robustness) -- all at 3%, from THIS folder.
EXISTING_OMEGA_SAMPLES = {
    "0.25": _samp("0.25"),
    "0.30": None,                  # None -> reuse baseline
    "0.35": _samp("0.35"),
}

# Table B.1 (tolerance) -- this folder is 3% only; the other rows come from their own runs.
EXISTING_TOL_SAMPLES = {"3%": None}



# ---- which parts to run ----
RUN_MAIN_P2        = True   # Table 1 (+ MLE truths used by power analysis)
RUN_MAIN_P3        = True   # Table 3   (P3 is a robustness check)
RUN_POSTERIORS_P2  = True   # Table 2 + P2 getdist triangles (emcee)
RUN_POSTERIORS_P3  = True   # P3 getdist triangles (emcee)
RUN_TOLERANCE      = True   # Table B.1
RUN_OMEGA          = True   # Table D.1
RUN_POWER          = True   # Table C.1
RUN_DELTA          = True   # new delta_i-sensitivity appendix table

# ---- input files ----
SGL_CSV      = "130sgls2.CSV"          # fallback "130sgls.CSV" handled below
SN_DAT       = "Pantheon+SH0ES.dat"
PANTHEON_COV = "Pantheon+SH0ES_STAT+SYS.cov"
DELTA_FILE   = "matched_result_sgls.csv"   # ONLY a fallback: used to look up per-lens
                                           # delta_i by name IF the parent CSV has no
                                           # 'delta' column. 130sgls2.CSV already has it,
                                           # so this file is normally not needed.

SN_Z_COL, SN_MAG_COL = "zHD", "m_b_corr"
LENS_NAME_COL = "Lens name"
ZL_COL, ZS_COL = "zl", "zs"
DELTA_NAME_COL, DELTA_VAL_COL = "Lens name", "delta"

# ---- pairing / cosmology grid ----
OMEGA_BASE = 0.30
TOL_BASE   = 0.03
OMEGA_LIST = [0.25, 0.30, 0.35]     # for Table D.1 (all at TOL_BASE)
TOL_LIST   = [0.03]                 # this folder = 3% only (B.1 3% row)
EPS_MILP   = 0.1                    # deterministic tie-breaker (do NOT use 1e-6)

# ---- physics constants (verbatim) ----
C_LIGHT_KMS    = 299792.458
RAD_PER_ARCSEC = np.pi / (180.0 * 3600.0)
SIGMA_SYS_FRAC = 0.03
H0_FID, OM_FID_P3 = 70.0, 0.30          # only for the P3 Sigma_tilde term
SIGMA_TILDE_NORM  = 166.0               # Chen+2019 constant; shifts gamma0 only,
                                        # NOT gamma_s / beta / dBIC (see notes)

# ---- statistics knobs ----
if QUICK_TEST:
    N_WALKERS, N_STEPS, BURN_IN = 24, 1500, 500
    N_POWER, N_STARTS_POWER     = 10, 2
    N_DELTA_REAL                = 5
    DE_MAXITER, DE_POPSIZE      = 150, 12
else:
    N_WALKERS, N_STEPS, BURN_IN = 32, 15000, 5000
    N_POWER, N_STARTS_POWER     = 200, 4
    N_DELTA_REAL                = 50
    DE_MAXITER, DE_POPSIZE      = 800, 25

POWER_ALT_BETA = -0.73   # representative tangential truth (paper value).
                         # set None to use the actual free-flat MLE beta instead.
POWER_SIGMA_DELTA = 0.05 # delta_i scatter injected on the ALT generation side
RNG = np.random.default_rng(20260601)

# ---- MLE optimisation bounds (DE global + Nelder-Mead polish) ----
B_G0, B_GZ, B_BETA, B_GS, B_DINT = (1.6, 2.7), (-1.5, 1.5), (-0.98, 0.98), (-1.0, 1.0), (0.005, 0.30)

# ============================================================================
# small logging helper
# ============================================================================
_LOG = open("ALL_RESULTS.log", "w", encoding="utf-8")
def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg); _LOG.write(msg + "\n"); _LOG.flush()

def banner(t):
    log("\n" + "=" * 78); log(t); log("=" * 78)

# ============================================================================
# 1. COSMOLOGY + COVARIANCE LOADERS
# ============================================================================
def comoving_distance(z, wm, h0=70.0):
    if z <= 0:
        return 0.0
    val, _ = quad(lambda zz: (wm * (1 + zz)**3 + (1 - wm))**-0.5, 0.0, z)
    return (C_LIGHT_KMS / h0) * val          # Mpc (only the ratio matters for pairing)

def load_pantheon_matrix(path):
    with open(path) as f:
        n = int(f.readline().split()[0])
    flat = pd.read_csv(path, header=None, skiprows=1).values.ravel().astype(float)
    assert flat.size == n * n, f"expected {n*n} cov entries, got {flat.size}"
    return flat.reshape(n, n)

def angular_diameter_distance(z, wm=OM_FID_P3, h0=H0_FID):
    return comoving_distance(z, wm, h0) / (1.0 + z)

def add_log10_sigma_tilde(data_sgl):
    """log10(Sigma_tilde) per system for the P3 term. Sigma_tilde =
       NORM * (Ds/(Dl*Dls)) * (thetaE/theta_eff)^2 ; angular-diameter distances
       at the fiducial (Om=0.3,H0=70). A constant NORM shifts gamma0 only."""
    zl, zs = data_sgl['z_l'], data_sgl['z_s']
    Dl  = np.array([angular_diameter_distance(z) for z in zl])
    Ds  = np.array([angular_diameter_distance(z) for z in zs])
    chi_l = np.array([comoving_distance(z, OM_FID_P3, H0_FID) for z in zl])
    chi_s = np.array([comoving_distance(z, OM_FID_P3, H0_FID) for z in zs])
    Dls = (chi_s - chi_l) / (1.0 + zs)                       # flat-universe D_A(l,s)
    ratio = (data_sgl['theta_E_arcsec'] / data_sgl['theta_eff_arcsec'])**2
    Sigma_tilde = SIGMA_TILDE_NORM * (Ds / (Dl * Dls)) * ratio
    data_sgl['log10_sigma_tilde'] = np.log10(Sigma_tilde)
    return data_sgl

# ============================================================================
# 2. MILP PAIRING  (EPS=0.1 deterministic; mirrors CosmoMatcher exactly)
# ============================================================================
def match_one(obj_df, sn_df, wm, tol, eps=EPS_MILP):
    sn_z   = sn_df[SN_Z_COL].values
    sn_idx = sn_df.index.values
    sn_d   = np.array([comoving_distance(z, wm) for z in sn_z])

    zl, zs  = obj_df[ZL_COL].values, obj_df[ZS_COL].values
    obj_idx = obj_df.index.values
    dl = np.array([comoving_distance(z, wm) for z in zl])
    ds = np.array([comoving_distance(z, wm) for z in zs])

    candidates = []
    for i in range(len(obj_df)):
        if dl[i] <= 0 or ds[i] <= 0:
            continue
        mask_l = np.abs(sn_d - dl[i]) / dl[i] <= tol
        mask_s = np.abs(sn_d - ds[i]) / ds[i] <= tol
        idxs_l, idxs_s = sn_idx[mask_l], sn_idx[mask_s]
        # RELATIVE deviation (|dchi|/chi), scale-invariant: keeps the tie-breaker
        # cost <= ~2*tol^2 regardless of distance units, so EPS cannot override the
        # -1 cardinality term (this is what was wrong before: absolute Mpc^2 costs
        # combined with EPS=0.1 flipped most terms positive -> only ~42 pairs).
        cl, cs = (np.abs(sn_d - dl[i]) / dl[i])[mask_l], (np.abs(sn_d - ds[i]) / ds[i])[mask_s]
        if len(idxs_l) == 0 or len(idxs_s) == 0:
            continue
        for a, sn1 in enumerate(idxs_l):
            for b, sn2 in enumerate(idxs_s):
                if sn1 != sn2:
                    candidates.append((obj_idx[i], sn1, sn2, cl[a]**2 + cs[b]**2))
    if not candidates:
        raise RuntimeError("no candidate triplets at this Omega_m / tolerance")

    map_obj = {u: k for k, u in enumerate(obj_idx)}
    uniq_sn = set(g[1] for g in candidates) | set(g[2] for g in candidates)
    map_sn  = {u: k + len(obj_idx) for k, u in enumerate(uniq_sn)}

    rows, cols, vals = [], [], []
    for j, (oid, sn1, sn2, _) in enumerate(candidates):
        rows += [map_obj[oid], map_sn[sn1], map_sn[sn2]]; cols += [j, j, j]; vals += [1, 1, 1]
    A = sparse.coo_matrix((vals, (rows, cols)),
                          shape=(len(map_obj) + len(map_sn), len(candidates)))
    d = np.array([g[3] for g in candidates])
    # Guarantee -1 + eps*d_j < 0 for every candidate (preserve max-cardinality);
    # with the relative cost above, d.max() <= ~2*tol^2 so eps_safe == EPS_MILP.
    eps_safe = min(eps, 0.5 / max(d.max(), 1e-12))
    c_vec = -1.0 + eps_safe * d
    res = milp(c=c_vec, constraints=LinearConstraint(A, -np.inf, 1),
               integrality=np.ones_like(c_vec), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")
    sel = np.where(res.x > 0.5)[0]
    return [(candidates[k][0], candidates[k][1], candidates[k][2]) for k in sel]

def tag_of(wm, tol):
    return f"Om{wm:.2f}_tol{int(round(tol*100)):02d}"

def generate_pairing(obj_df, sn_df, cov, delta_map, wm, tol):
    tag = tag_of(wm, tol)
    pairs = match_one(obj_df, sn_df, wm, tol)
    rows, idx_l, idx_s, missing = [], [], [], 0
    for oid, sn1, sn2 in pairs:
        r = obj_df.loc[oid].to_dict()
        name = r.get(LENS_NAME_COL)
        if name not in delta_map or pd.isna(delta_map.get(name)):
            missing += 1; continue
        r["delta"] = float(delta_map[name])
        r["sn_lens_m_b_corr"]   = sn_df.loc[sn1, SN_MAG_COL]
        r["sn_source_m_b_corr"] = sn_df.loc[sn2, SN_MAG_COL]
        rows.append(r); idx_l.append(sn1); idx_s.append(sn2)
    out = pd.DataFrame(rows)
    out.to_csv(f"matched_result_sgls_{tag}.csv", index=False)
    idx_l, idx_s = np.array(idx_l, int), np.array(idx_s, int)
    np.savetxt(f"cov_sgls_lens_{tag}.txt",   cov[np.ix_(idx_l, idx_l)], fmt="%.8e")
    np.savetxt(f"cov_sgls_source_{tag}.txt", cov[np.ix_(idx_s, idx_s)], fmt="%.8e")
    np.savetxt(f"cov_sgls_cross_{tag}.txt",  cov[np.ix_(idx_l, idx_s)], fmt="%.8e")
    log(f"  [{tag}] matched {len(out)} pairs"
        + (f" (skipped {missing} lacking delta_i)" if missing else ""))
    return tag, len(out)

# ============================================================================
# 3. DATA LOADER for a given matched sample (path-based)
# ============================================================================
def _csv_tokens(csv_path):
    """Distinguishing tokens from a matched-CSV name (e.g. Om0.30 / Om0_30)."""
    base = os.path.splitext(os.path.basename(csv_path))[0]
    for pre in ("matched_result_sgls_", "matched_sgls_milp_dd_pantheon_",
                "matched_sgls_", "matched_"):
        if base.startswith(pre):
            base = base[len(pre):]; break
    toks = {base, base.replace(".", "_"), base.replace("_", ".")}
    return {t for t in toks if t}

def _resolve_cov(given, role, n, search_dir, tokens):
    """Return a cov file for `role` with n columns. Use `given` if it exists;
    else auto-find in search_dir, disambiguating by CSV tokens, never guessing
    silently among several equally-plausible files. Duplicate/copy files
    ('... - 副本', '... - Copy', '...(1)') are ignored."""
    import glob
    if given and os.path.exists(given):
        return given
    def _is_dup(name):
        low = name.lower()
        return ("副本" in name) or ("copy" in low) or any(f"({d})" in name for d in "123456789")
    cands = []
    for f in glob.glob(os.path.join(search_dir or ".", f"*{role}*.txt")):
        if _is_dup(os.path.basename(f)):
            continue
        try:
            with open(f) as fh:
                if len(fh.readline().split()) == n:
                    cands.append(f)
        except Exception:
            pass
    if not cands:
        raise FileNotFoundError(
            f"no '{role}' covariance with {n} columns found near {search_dir or '.'} "
            f"(looked for '*{role}*.txt'). Set the path explicitly.")
    tagged = [c for c in cands if any(t in os.path.basename(c) for t in tokens)]
    pick = tagged if tagged else cands
    if len(pick) > 1:
        raise FileNotFoundError(
            f"ambiguous '{role}' covariance ({n} cols): {sorted(os.path.basename(c) for c in pick)}.\n"
            f"  Several samples share this size; set EXISTING_*_COV explicitly for this sample.")
    chosen = pick[0]
    if chosen != given:
        log(f"  [auto-cov] {role}: using {os.path.basename(chosen)}")
    return chosen

def paths_for_tag(tag):
    """Internal filenames produced by the built-in matcher for a given tag."""
    return (f"matched_result_sgls_{tag}.csv",
            f"cov_sgls_lens_{tag}.txt",
            f"cov_sgls_source_{tag}.txt",
            f"cov_sgls_cross_{tag}.txt")

def load_sample(csv_path, cov_lens, cov_source, cov_cross):
    """Load a matched sample from explicit paths (auto-resolving cov by size/token
    if a given path is missing). No copying."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    n = len(df)
    sdir = os.path.dirname(os.path.abspath(csv_path))
    toks = _csv_tokens(csv_path)
    cov_lens   = _resolve_cov(cov_lens,   "lens",   n, sdir, toks)
    cov_source = _resolve_cov(cov_source, "source", n, sdir, toks)
    cov_cross  = _resolve_cov(cov_cross,  "cross",  n, sdir, toks)
    data_sgl = {
        'z_l': df['zl'].values, 'z_s': df['zs'].values,
        'theta_E': df['thetaE'].values * RAD_PER_ARCSEC,
        'theta_ap': df['thetaap'].values * RAD_PER_ARCSEC,
        'theta_E_arcsec': df['thetaE'].values,
        'theta_eff_arcsec': df['thetaeff'].values if 'thetaeff' in df else np.full(len(df), np.nan),
        'sigma_ap': df['sigma_ap'].values, 'dsigma_ap': df['dsigma_ap'].values,
        'delta': df['delta'].values,
    }
    data_sn = {'mb_l': df['sn_lens_m_b_corr'].values, 'mb_s': df['sn_source_m_b_corr'].values}
    cov = {'lens': np.atleast_2d(np.loadtxt(cov_lens)),
           'source': np.atleast_2d(np.loadtxt(cov_source)),
           'cross': np.atleast_2d(np.loadtxt(cov_cross))}
    for role, M in cov.items():
        if M.shape != (n, n):
            raise ValueError(
                f"\n  Covariance/sample MISMATCH: '{role}' matrix is {M.shape} but the matched "
                f"CSV\n  '{csv_path}' has {n} rows. The CSV and its three covariance files\n"
                f"  MUST come from the SAME matching run (row i of CSV = row/col i of cov).")
    C_diff = cov['lens'] + cov['source'] - cov['cross'] - cov['cross'].T
    if 'thetaeff' in df:
        add_log10_sigma_tilde(data_sgl)
    return data_sgl, data_sn, C_diff, n

# ============================================================================
# 4. PHYSICS  (predict_sigma + likelihood; P2 and P3 unified)
# ============================================================================
def predict_sigma(data_sgl, g0, gz, beta, R_obs, gs=None):
    gamma_val = g0 + gz * data_sgl['z_l']
    if gs is not None:
        gamma_val = gamma_val + gs * data_sgl['log10_sigma_tilde']
    delta = data_sgl['delta']
    xi = gamma_val + delta - 2.0
    try:
        t1 = gamma_func((xi - 1)/2.0) / gamma_func(xi/2.0)
        t2 = beta * gamma_func((xi + 1)/2.0) / gamma_func((xi + 2)/2.0)
        t3 = (gamma_func(gamma_val/2.0) * gamma_func(delta/2.0)) / \
             (gamma_func((gamma_val-1)/2.0) * gamma_func((delta-1)/2.0))
        denom = (xi - 2*beta) * (3 - xi)
        if np.any(denom == 0) or not np.all(np.isfinite(t1)) \
           or not np.all(np.isfinite(t2)) or not np.all(np.isfinite(t3)):
            raise ValueError
        F = ((3 - delta) / denom) * (t1 - t2) * t3
    except ValueError:
        return np.full_like(R_obs, np.inf)
    term_dist = np.where(R_obs > 0.001, 1.0 / np.where(R_obs == 0, np.nan, R_obs), 0.0)
    pre = (C_LIGHT_KMS**2) / (2 * np.sqrt(np.pi))
    sigma_sq = pre * term_dist * data_sgl['theta_E'] * F * \
               (data_sgl['theta_ap'] / data_sgl['theta_E'])**(2 - gamma_val)
    if not np.all(np.isfinite(sigma_sq)) or np.any(sigma_sq <= 0):
        return np.full_like(R_obs, np.inf)
    return np.sqrt(sigma_sq)

def _unpack(theta, framework, fixed_beta):
    if framework == 'P2':
        if fixed_beta is None: g0, gz, beta, f_int = theta; gs = None
        else:                  g0, gz, f_int = theta; beta = fixed_beta; gs = None
    else:  # P3
        if fixed_beta is None: g0, gz, beta, gs, f_int = theta
        else:                  g0, gz, gs, f_int = theta; beta = fixed_beta
    return g0, gz, beta, gs, f_int

def ln_likelihood(theta, data_sgl, data_sn, C_diff, framework='P2', fixed_beta=None):
    g0, gz, beta, gs, f_int = _unpack(theta, framework, fixed_beta)
    if not (-1.0 < beta < 1.0):  return -np.inf
    if not (1.0 < g0 < 3.0):     return -np.inf
    if not (-2.0 < gz < 2.0):    return -np.inf
    if not (0.0 < f_int < 1.0):  return -np.inf
    if gs is not None and not (-1.5 < gs < 1.5): return -np.inf

    gamma_chk = g0 + gz * data_sgl['z_l']
    if gs is not None:
        gamma_chk = gamma_chk + gs * data_sgl['log10_sigma_tilde']
    if np.any(gamma_chk <= 1.05) or np.any(gamma_chk >= 2.95): return -np.inf

    dmu = data_sn['mb_l'] - data_sn['mb_s']
    dist_ratio = 10.0**(0.2 * dmu)
    R_model = 1.0 - dist_ratio * ((1 + data_sgl['z_s']) / (1 + data_sgl['z_l']))
    if np.any(R_model <= 0.001): return -np.inf

    deriv = -0.2 * np.log(10.0) * (1.0 - R_model)
    Cov_R = deriv[:, None] * deriv[None, :] * C_diff

    sigma_pred = predict_sigma(data_sgl, g0, gz, beta, R_model, gs=gs)
    if not np.all(np.isfinite(sigma_pred)): return -np.inf

    res = data_sgl['sigma_ap'] - sigma_pred
    jac = -0.5 * sigma_pred / R_model
    Cov_sig = jac[:, None] * jac[None, :] * Cov_R
    diag = data_sgl['dsigma_ap']**2 + (SIGMA_SYS_FRAC * data_sgl['sigma_ap'])**2 \
           + (f_int * data_sgl['sigma_ap'])**2
    C_total = Cov_sig.copy(); np.fill_diagonal(C_total, C_total.diagonal() + diag)
    try:
        L = scipy.linalg.cho_factor(C_total, lower=True)
        chi2 = np.dot(res, scipy.linalg.cho_solve(L, res))
        logdet = 2 * np.sum(np.log(np.diag(L[0])))
        return -0.5 * (chi2 + logdet)
    except scipy.linalg.LinAlgError:
        return -np.inf

def neg_ln_likelihood(theta, data_sgl, data_sn, C_diff, framework, fixed_beta):
    ll = ln_likelihood(theta, data_sgl, data_sn, C_diff, framework, fixed_beta)
    return -ll if np.isfinite(ll) else 1e10

def ln_prob(theta, data_sgl, data_sn, C_diff, framework, fixed_beta, beta_prior):
    lp = 0.0
    if fixed_beta is None and beta_prior == "gaussian":
        beta = theta[2]
        if not (-1.0 < beta < 1.0): return -np.inf
        lp += -0.5 * ((beta - 0.18) / 0.13)**2
    ll = ln_likelihood(theta, data_sgl, data_sn, C_diff, framework, fixed_beta)
    return -np.inf if not np.isfinite(ll) else lp + ll

# ============================================================================
# 5. MLE FIT  (DE global search + Nelder-Mead polish) -> lnL, x, k
# ============================================================================
def _bounds(framework, fixed_beta):
    if framework == 'P2':
        return [B_G0, B_GZ, B_BETA, B_DINT] if fixed_beta is None else [B_G0, B_GZ, B_DINT]
    return [B_G0, B_GZ, B_BETA, B_GS, B_DINT] if fixed_beta is None else [B_G0, B_GZ, B_GS, B_DINT]

def mle_fit(data_sgl, data_sn, C_diff, framework, fixed_beta, seed=0):
    bounds = _bounds(framework, fixed_beta)
    args = (data_sgl, data_sn, C_diff, framework, fixed_beta)
    de = op.differential_evolution(neg_ln_likelihood, bounds, args=args,
                                   popsize=DE_POPSIZE, maxiter=DE_MAXITER, tol=1e-8,
                                   mutation=(0.5, 1.0), recombination=0.7,
                                   polish=True, seed=seed, init='sobol', updating='deferred')
    pol = op.minimize(neg_ln_likelihood, de.x, args=args, method='Nelder-Mead',
                      options=dict(maxiter=10000, xatol=1e-7, fatol=1e-7))
    x, f = (pol.x, pol.fun) if pol.fun < de.fun else (de.x, de.fun)
    return -f, x, len(bounds)

def ic(lnL, k, N):
    return dict(lnL=lnL, k=k, AIC=2*k - 2*lnL, BIC=k*np.log(N) - 2*lnL)

# ============================================================================
# 6. TABLE EMITTERS  (csv + minimal LaTeX)
# ============================================================================
def emit_table(name, header, rows, caption=""):
    pd.DataFrame(rows, columns=header).to_csv(f"{name}.csv", index=False)
    with open(f"{name}.tex", "w", encoding="utf-8") as f:
        f.write("% " + caption + "\n")
        f.write("\\begin{tabular}{" + "l" + "c"*(len(header)-1) + "}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    log(f"  wrote {name}.csv and {name}.tex")

def fmt(x, n=3): return f"{x:.{n}f}"

# ============================================================================
# 7. POSTERIORS via MCMC  (Table 2 + getdist triangles via cosmo_tools)
# ============================================================================
def run_mcmc(data_sgl, data_sn, C_diff, framework, fixed_beta, beta_prior, x0):
    import emcee
    ndim = len(x0)
    pos = [np.array(x0) + 1e-4 * np.random.randn(ndim) for _ in range(N_WALKERS)]
    for p in pos: p[-1] = abs(p[-1])
    sampler = emcee.EnsembleSampler(
        N_WALKERS, ndim, ln_prob,
        args=(data_sgl, data_sn, C_diff, framework, fixed_beta, beta_prior))
    sampler.run_mcmc(pos, N_STEPS, progress=True)
    chain = sampler.get_chain(discard=BURN_IN, thin=1, flat=True)
    return chain

def med68(samples_col):
    q16, q50, q84 = np.percentile(samples_col, [16, 50, 84])
    return q50, q84 - q50, q50 - q16

# ============================================================================
# 8. POWER ANALYSIS  (truths come from THIS run's MLE -- no stale constants)
# ============================================================================
def build_model_and_cov(data_sgl, data_sn, C_diff, g0, gz, beta, f_int, gs=None):
    dmu = data_sn['mb_l'] - data_sn['mb_s']
    R = 1.0 - 10.0**(0.2*dmu) * ((1 + data_sgl['z_s']) / (1 + data_sgl['z_l']))
    deriv = -0.2*np.log(10.0)*(1.0 - R)
    Cov_R = deriv[:, None]*deriv[None, :]*C_diff
    sig = predict_sigma(data_sgl, g0, gz, beta, R, gs=gs)
    jac = -0.5*sig/R
    Cov_sig = jac[:, None]*jac[None, :]*Cov_R
    diag = data_sgl['dsigma_ap']**2 + (SIGMA_SYS_FRAC*sig)**2 + (f_int*sig)**2
    C_total = Cov_sig.copy(); np.fill_diagonal(C_total, C_total.diagonal() + diag)
    return sig, 0.5*(C_total + C_total.T)

def power_side(label, data_sgl, data_sn, C_diff, truth, inject_delta):
    dbics, betas = [], []
    for i in range(N_POWER):
        gen = dict(data_sgl)
        if inject_delta:
            gen = dict(data_sgl)
            gen['delta'] = data_sgl['delta'] + RNG.normal(0.0, POWER_SIGMA_DELTA, size=data_sgl['delta'].shape)
        sig, C_total = build_model_and_cov(gen, data_sn, C_diff,
                                           truth['g0'], truth['gz'], truth['beta'], truth['f_int'])
        if not np.all(np.isfinite(sig)):
            continue
        synth = sig + RNG.multivariate_normal(np.zeros_like(sig), C_total)
        s = dict(data_sgl); s['sigma_ap'] = synth
        # multi-start MLE for free and fixed-0
        best_free, best_fix0 = -np.inf, -np.inf
        for st in range(N_STARTS_POWER):
            seedn = int(RNG.integers(1 << 31))
            lf, _, _ = mle_fit(s, data_sn, C_diff, 'P2', None, seed=seedn)
            l0, _, _ = mle_fit(s, data_sn, C_diff, 'P2', 0.0, seed=seedn)
            best_free, best_fix0 = max(best_free, lf), max(best_fix0, l0)
        N = len(s['z_l'])
        dbic = (3*np.log(N) - 2*best_fix0) - (4*np.log(N) - 2*best_free)
        dbics.append(dbic)
        # one free fit for beta_hat
        _, xf, _ = mle_fit(s, data_sn, C_diff, 'P2', None, seed=0)
        betas.append(xf[2])
        log(f"    [{label}] {i+1:3d}/{N_POWER}  dBIC={dbic:+7.2f}  beta_hat={xf[2]:+.3f}")
    dbics, betas = np.array(dbics), np.array(betas)
    return dict(n=len(dbics), mean_dbic=float(np.mean(dbics)),
                frac6=float(np.mean(dbics > 6)), frac10=float(np.mean(dbics > 10)),
                mean_beta=float(np.mean(betas)))

# ============================================================================
# 9. MAIN
# ============================================================================
def main():
    t_start = time.time()
    if QUICK_TEST:
        banner("QUICK_TEST = True  (toy sizes -- numbers are NOT publication-grade)\n"
               "Set QUICK_TEST = False for the real run.")

    summary = {}
    base_tag = tag_of(OMEGA_BASE, TOL_BASE)
    samples = {}   # registry: 'base' / ('tol',t) / ('om',om) -> (csv, cov_lens, cov_source, cov_cross)

    if RUN_PAIRING:
        # ---- REGENERATE with the (fixed) built-in matcher ----
        global SGL_CSV
        if not os.path.exists(SGL_CSV) and os.path.exists("130sgls.CSV"):
            SGL_CSV = "130sgls.CSV"
        miss = [f for f in [SGL_CSV, SN_DAT, PANTHEON_COV] if not os.path.exists(f)]
        if miss:
            log("MISSING raw input files for pairing:", miss); sys.exit(1)
        banner("PART 0/1: MILP pairing (built-in matcher; relative cost, eps guarded)")
        obj_df = pd.read_csv(SGL_CSV); obj_df.columns = [c.strip() for c in obj_df.columns]
        sn_df  = pd.read_csv(SN_DAT, sep=r"\s+")
        cov    = load_pantheon_matrix(PANTHEON_COV)
        assert cov.shape[0] == len(sn_df), "SN length must equal covariance dimension!"
        if DELTA_VAL_COL in obj_df.columns and obj_df[DELTA_VAL_COL].notna().all():
            delta_map = dict(zip(obj_df[LENS_NAME_COL], obj_df[DELTA_VAL_COL]))
            log(f"delta_i from {SGL_CSV} ('{DELTA_VAL_COL}', {len(delta_map)} systems)")
        elif os.path.exists(DELTA_FILE):
            dm = pd.read_csv(DELTA_FILE); dm.columns = [c.strip() for c in dm.columns]
            delta_map = dict(zip(dm[DELTA_NAME_COL], dm[DELTA_VAL_COL]))
        else:
            log("NO delta_i available."); sys.exit(1)
        grid = {(OMEGA_BASE, TOL_BASE)} | {(om, TOL_BASE) for om in OMEGA_LIST} \
               | {(OMEGA_BASE, t) for t in TOL_LIST}
        for (om, t) in sorted(grid):
            generate_pairing(obj_df, sn_df, cov, delta_map, om, t)
        samples['base'] = paths_for_tag(base_tag)
        for t in TOL_LIST:   samples[('tol', t)] = paths_for_tag(tag_of(OMEGA_BASE, t))
        for om in OMEGA_LIST: samples[('om', om)] = paths_for_tag(tag_of(om, TOL_BASE))
    else:
        # ---- CONSUME your CosmoMatcher output directly (no copying) ----
        banner("PART 0/1: consuming your CosmoMatcher matched sample (read in place)")
        samples['base'] = (EXISTING_BASELINE_CSV, *EXISTING_BASELINE_COV)
        if not os.path.exists(EXISTING_BASELINE_CSV):
            log(f"MISSING baseline CSV: {EXISTING_BASELINE_CSV}")
            log("Fix EXISTING_BASELINE_CSV, or set RUN_PAIRING=True.")
            sys.exit(1)
        # cov paths are auto-resolved (by column count + filename tag) and size-checked
        # inside load_sample, so a wrong/old cov name here is tolerated if the right
        # file is present in the folder; a genuine mismatch still errors clearly.
        for lbl, spec in EXISTING_TOL_SAMPLES.items():
            samples[('tol', float(str(lbl).strip('%'))/100)] = spec if spec else samples['base']
        for lbl, spec in EXISTING_OMEGA_SAMPLES.items():
            samples[('om', float(lbl))] = spec if spec else samples['base']
        log(f"  baseline <- {EXISTING_BASELINE_CSV}")

    d_sgl, d_sn, C_diff, N = load_sample(*samples['base'])
    log(f"baseline sample: N = {N}")
    if not RUN_PAIRING:
        log(f"  baseline N={N} (expected ~104 at 3% tolerance)")

    mle_store = {}  # remember MLE params for power-analysis truths

    # ------------------------------------------------------------- Table 1 (P2)
    if RUN_MAIN_P2:
        banner("PART 2: Table 1 -- P2 model comparison (MLE)")
        rows, base_bic = [], None
        for nm, fb in [("Free", None), ("Fixed b=0", 0.0), ("Fixed b=0.18", 0.18)]:
            lnL, x, k = mle_fit(d_sgl, d_sn, C_diff, 'P2', fb)
            mle_store[nm] = (x, fb)
            I = ic(lnL, k, N)
            rows.append([nm, k, I['lnL'], I['AIC'], I['BIC']])
            if nm == "Free": base_bic = I['BIC']
            log(f"  {nm:14s} k={k} lnL={lnL:9.2f} AIC={I['AIC']:9.2f} BIC={I['BIC']:9.2f}")
        rows = [r + [r[4] - base_bic] for r in rows]
        emit_table("table1_p2_modelcomp",
                   ["Model", "k", "MaxlnL", "AIC", "BIC", "dBIC"], rows,
                   "P2 model comparison (N=%d)" % N)
        summary['table1'] = rows

    # ------------------------------------------------------------- Table 3 (P3)
    if RUN_MAIN_P3:
        banner("PART 3: Table 3 -- P3 model comparison (robustness check)")
        if np.any(~np.isfinite(d_sgl.get('log10_sigma_tilde', [np.nan]))):
            log("  WARNING: theta_eff missing or Sigma_tilde non-finite -> P3 skipped.")
        else:
            rows, base_bic = [], None
            for nm, fb in [("Free", None), ("Fixed b=0", 0.0), ("Fixed b=0.18", 0.18)]:
                lnL, x, k = mle_fit(d_sgl, d_sn, C_diff, 'P3', fb)
                I = ic(lnL, k, N)
                rows.append([nm, k, I['lnL'], I['AIC'], I['BIC']])
                if nm == "Free": base_bic = I['BIC']; gs_free = x[3]
                log(f"  {nm:14s} k={k} lnL={lnL:9.2f} BIC={I['BIC']:9.2f}")
            rows = [r + [r[4] - base_bic] for r in rows]
            emit_table("table3_p3_modelcomp",
                       ["Model", "k", "MaxlnL", "AIC", "BIC", "dBIC"], rows,
                       "P3 model comparison (robustness; N=%d)" % N)
            log(f"  P3 free gamma_s = {gs_free:+.3f}  (paper: ~ -0.03; dBIC/gs invariant to Sigma_tilde NORM)")
            log("  >>> CONFIRM dBIC reproduces your Table 3 (~5.65, ~33.23). If not, the")
            log("      Sigma_tilde distance convention differs -- P3 is only a robustness check.")
            summary['table3'] = rows

    # ----------------------------------------------------------- Table 2 (post.)
    if RUN_POSTERIORS_P2:
        banner("PART 4: Table 2 -- P2 posteriors (emcee) + getdist triangles")
        try:
            import emcee  # noqa
        except ImportError:
            log("  emcee not installed -> skipping posteriors. pip install emcee")
        else:
            try:
                import cosmo_tools; HAS_CT = True
            except Exception as e:
                HAS_CT = False; log(f"  cosmo_tools not importable ({e}); plots skipped")
            # (fname_stub, legend, fixed_beta, prior, x0)
            scen = [("P2_Free_Beta_FlatPrior",  "P2 Free (Flat)",  None, "flat",     [2.1, -0.5, -0.5, 0.07]),
                    ("P2_Free_Beta_GaussPrior", "P2 Free (Gauss)", None, "gaussian", [2.0, -0.5,  0.18, 0.07]),
                    ("P2_Fixed_Beta_0",         "P2 Fixed 0",      0.0,  "none",     [1.95, -0.9, 0.08]),
                    ("P2_Fixed_Beta_0_18",      "P2 Fixed 0.18",   0.18, "none",     [1.90, -1.0, 0.10])]
            prows, comp_samples = [], []
            common_labels = [r'\gamma_0', r'\gamma_z', r'\delta_{int}']
            for stub, leg, fb, pr, x0 in scen:
                log(f"  MCMC: {leg}")
                ch = run_mcmc(d_sgl, d_sn, C_diff, 'P2', fb, pr, x0)
                if fb is None:
                    g0 = med68(ch[:, 0]); gz = med68(ch[:, 1]); be = med68(ch[:, 2]); fi = med68(ch[:, 3])
                    bstr = f"{be[0]:.3f} (+{be[1]:.3f},-{be[2]:.3f})"
                    labels = [r'\gamma_0', r'\gamma_z', r'\beta_{ani}', r'\delta_{int}']
                    comp_samples.append(np.column_stack([ch[:, 0], ch[:, 1], ch[:, 3]]))
                else:
                    g0 = med68(ch[:, 0]); gz = med68(ch[:, 1]); fi = med68(ch[:, 2])
                    bstr = f"= {fb}"
                    labels = [r'\gamma_0', r'\gamma_z', r'\delta_{int}']
                    comp_samples.append(np.column_stack([ch[:, 0], ch[:, 1], ch[:, 2]]))
                prows.append([leg,
                              f"{g0[0]:.3f} (+{g0[1]:.3f},-{g0[2]:.3f})",
                              f"{gz[0]:.3f} (+{gz[1]:.3f},-{gz[2]:.3f})",
                              bstr,
                              f"{fi[0]:.3f} (+{fi[1]:.3f},-{fi[2]:.3f})"])
                if HAS_CT:
                    try:
                        sl = cosmo_tools.calculate_stats(ch, labels)
                        cosmo_tools.plot_getdist_advanced(ch, labels, stats_list=sl,
                                                          filename=f"{stub}_Opt_GetDist.pdf")
                    except Exception as e:
                        log(f"  getdist plot failed for {stub}: {e}")
            pd.DataFrame(prows, columns=["Model", "gamma0", "gamma_z", "beta_ani", "delta_int"]
                         ).to_csv("table2_p2_posteriors.csv", index=False)
            log("  wrote table2_p2_posteriors.csv")
            if HAS_CT:
                try:
                    cosmo_tools.plot_getdist_comparison(
                        comp_samples, common_labels,
                        ["P2 Free (Flat)", "P2 Free (Gauss)", "P2 Fixed 0", "P2 Fixed 0.18"],
                        filename="P2_Beta_Models_Comparison_Opt.pdf")
                    log("  wrote P2 getdist triangles + P2_Beta_Models_Comparison_Opt.pdf")
                except Exception as e:
                    log(f"  P2 comparison plot failed: {e}")
            summary['table2'] = prows

    # ------------------------------------------------- P3 posteriors + triangles
    if RUN_POSTERIORS_P3:
        banner("PART 4b: P3 posteriors (emcee) + getdist triangles")
        try:
            import emcee  # noqa
        except ImportError:
            log("  emcee not installed -> skipping P3 posteriors.")
        else:
            try:
                import cosmo_tools; HAS_CT = True
            except Exception as e:
                HAS_CT = False; log(f"  cosmo_tools not importable ({e}); plots skipped")
            # P3 free order: g0,gz,beta,gs,fint ; fixed: g0,gz,gs,fint
            scen3 = [("P3_Free_Beta_FlatPrior",  "P3 Free (Flat)",  None, "flat",     [2.0, -0.5, -0.5, -0.03, 0.07]),
                     ("P3_Free_Beta_GaussPrior", "P3 Free (Gauss)", None, "gaussian", [2.0, -0.5,  0.18, -0.03, 0.07]),
                     ("P3_Fixed_Beta_0",         "P3 Fixed 0",      0.0,  "none",     [1.95, -0.9, -0.03, 0.08]),
                     ("P3_Fixed_Beta_0_18",      "P3 Fixed 0.18",   0.18, "none",     [1.90, -1.0, -0.03, 0.10])]
            comp3 = []
            common3 = [r'\gamma_0', r'\gamma_z', r'\gamma_s', r'\delta_{int}']
            for stub, leg, fb, pr, x0 in scen3:
                log(f"  MCMC: {leg}")
                ch = run_mcmc(d_sgl, d_sn, C_diff, 'P3', fb, pr, x0)
                if fb is None:   # g0,gz,beta,gs,fint -> common = g0,gz,gs,fint
                    labels = [r'\gamma_0', r'\gamma_z', r'\beta_{ani}', r'\gamma_s', r'\delta_{int}']
                    comp3.append(ch[:, [0, 1, 3, 4]])
                else:            # g0,gz,gs,fint
                    labels = [r'\gamma_0', r'\gamma_z', r'\gamma_s', r'\delta_{int}']
                    comp3.append(ch[:, [0, 1, 2, 3]])
                if HAS_CT:
                    try:
                        sl = cosmo_tools.calculate_stats(ch, labels)
                        cosmo_tools.plot_getdist_advanced(ch, labels, stats_list=sl,
                                                          filename=f"{stub}_Opt_GetDist.pdf")
                    except Exception as e:
                        log(f"  getdist plot failed for {stub}: {e}")
            if HAS_CT:
                try:
                    cosmo_tools.plot_getdist_comparison(
                        comp3, common3,
                        ["P3 Free (Flat)", "P3 Free (Gauss)", "P3 Fixed 0", "P3 Fixed 0.18"],
                        filename="P3_Beta_Models_Comparison_Opt.pdf")
                    log("  wrote P3 getdist triangles + P3_Beta_Models_Comparison_Opt.pdf")
                except Exception as e:
                    log(f"  P3 comparison plot failed: {e}")

    # ----------------------------------------------------------- Table B.1 (tol)
    if RUN_TOLERANCE:
        banner("PART 5: Table B.1 -- tolerance sensitivity (Om=0.30)")
        rows = []
        for t in TOL_LIST:
            if ('tol', t) not in samples:
                log(f"  tol={int(t*100)}%: no sample registered. Add it to EXISTING_TOL_SAMPLES "
                    f"(or set RUN_PAIRING=True)."); continue
            try:
                ds, sn, Cd, n = load_sample(*samples[('tol', t)])
            except Exception as e:
                log(f"  tol={int(t*100)}%: load failed ({e})"); continue
            lnF, xF, _ = mle_fit(ds, sn, Cd, 'P2', None)
            ln0, _, _  = mle_fit(ds, sn, Cd, 'P2', 0.0)
            dbic = (3*np.log(n) - 2*ln0) - (4*np.log(n) - 2*lnF)
            rows.append([f"{int(t*100)}%", n, lnF, lnF/n, dbic, xF[2], xF[1], xF[3]])
            log(f"  tol={int(t*100)}%  N={n}  lnL/N={lnF/n:.3f}  dBIC={dbic:.2f} "
                f"beta={xF[2]:+.3f} gz={xF[1]:+.3f}")
        emit_table("tableB1_tolerance",
                   ["Tol", "Pairs", "MaxlnL", "lnL_per_N", "dBIC_vs_fix0", "beta", "gamma_z", "dint"],
                   rows, "Sensitivity to MILP matching tolerance")
        summary['tableB1'] = rows

    # ----------------------------------------------------------- Table D.1 (Om)
    if RUN_OMEGA:
        banner("PART 6: Table D.1 -- Omega_m robustness (tol=3%)")
        rows = []
        for om in OMEGA_LIST:
            if ('om', om) not in samples:
                log(f"  Om={om}: no sample registered. Add it to EXISTING_OMEGA_SAMPLES "
                    f"(or set RUN_PAIRING=True)."); continue
            try:
                ds, sn, Cd, n = load_sample(*samples[('om', om)])
            except Exception as e:
                log(f"  Om={om}: load failed ({e})"); continue
            lnF, xF, _ = mle_fit(ds, sn, Cd, 'P2', None)
            ln0, _, _  = mle_fit(ds, sn, Cd, 'P2', 0.0)
            dbic = (3*np.log(n) - 2*ln0) - (4*np.log(n) - 2*lnF)
            rows.append([om, n, dbic, xF[2], xF[1]])
            log(f"  Om={om}  N={n}  dBIC={dbic:.2f}  beta={xF[2]:+.3f}  gz={xF[1]:+.3f}")
        emit_table("tableD1_omega",
                   ["Omega_m", "Pairs", "dBIC_vs_fix0", "beta", "gamma_z"], rows,
                   "Robustness to fiducial pairing cosmology")
        log("  >>> Om=0.30 row should match this folder B.1 baseline (dBIC ~ 1.9).")
        summary['tableD1'] = rows

    # ------------------------------------------------------------- Table C.1 (power)
    if RUN_POWER:
        banner("PART 7: Table C.1 -- two-sided power analysis")
        if 'Free' not in mle_store or 'Fixed b=0' not in mle_store:
            log("  (need RUN_MAIN_P2=True so truths come from the MLE) -- running quick MLE now")
            xF, _ = mle_fit(d_sgl, d_sn, C_diff, 'P2', None)[1], None
            x0v, _ = mle_fit(d_sgl, d_sn, C_diff, 'P2', 0.0)[1], None
        else:
            xF = mle_store['Free'][0]; x0v = mle_store['Fixed b=0'][0]
        truth_null = dict(g0=x0v[0], gz=x0v[1], beta=0.0,        f_int=x0v[2])
        alt_beta   = POWER_ALT_BETA if POWER_ALT_BETA is not None else xF[2]
        truth_alt  = dict(g0=xF[0],  gz=xF[1],  beta=alt_beta,   f_int=xF[3])
        log(f"  NULL truth: g0={truth_null['g0']:.3f} gz={truth_null['gz']:.3f} beta=0")
        log(f"  ALT  truth: g0={truth_alt['g0']:.3f} gz={truth_alt['gz']:.3f} beta={alt_beta:.3f}")
        nul = power_side("NULL", d_sgl, d_sn, C_diff, truth_null, inject_delta=False)
        alt = power_side("ALT",  d_sgl, d_sn, C_diff, truth_alt,  inject_delta=True)
        rows = [
            ["Realisations", nul['n'], alt['n']],
            ["mean_dBIC", round(nul['mean_dbic'], 2), round(alt['mean_dbic'], 2)],
            ["P(dBIC>6)", f"{nul['frac6']*100:.1f}%", f"{alt['frac6']*100:.1f}%"],
            ["P(dBIC>10)", f"{nul['frac10']*100:.1f}%", f"{alt['frac10']*100:.1f}%"],
            ["mean_beta_hat", round(nul['mean_beta'], 3), round(alt['mean_beta'], 3)],
        ]
        pd.DataFrame(rows, columns=["Quantity", "Null(b=0)", f"Alt(b={alt_beta:.2f})"]
                     ).to_csv("tableC1_power.csv", index=False)
        log("  wrote tableC1_power.csv")
        summary['tableC1'] = rows

    # ------------------------------------------------- delta_i sensitivity (new)
    if RUN_DELTA:
        banner("PART 8: delta_i sensitivity (new appendix table)")
        d0 = d_sgl['delta'].copy()
        def fit_three(ds):
            lnF, xF, _ = mle_fit(ds, d_sn, C_diff, 'P2', None)
            ln0, _, _  = mle_fit(ds, d_sn, C_diff, 'P2', 0.0)
            ln18, _, _ = mle_fit(ds, d_sn, C_diff, 'P2', 0.18)
            bicF = 4*np.log(N) - 2*lnF
            return dict(dbic0=(3*np.log(N)-2*ln0)-bicF,
                        dbic18=(3*np.log(N)-2*ln18)-bicF,   # fix0.18 has k=3 (beta fixed)
                        g0=xF[0], gz=xF[1], beta=xF[2])
        rows = []
        base = fit_three(d_sgl)
        rows.append(["baseline", round(base['dbic0'],2), round(base['dbic18'],1),
                     round(base['g0'],3), round(base['gz'],3), round(base['beta'],3)])
        log(f"  baseline dBIC0={base['dbic0']:+.2f} g0={base['g0']:.3f} beta={base['beta']:+.3f}")
        for off in (+0.05, -0.05, +0.10, -0.10):
            ds = dict(d_sgl); ds['delta'] = d0 + off
            r = fit_three(ds)
            rows.append([f"offset {off:+.2f}", round(r['dbic0'],2), round(r['dbic18'],1),
                         round(r['g0'],3), round(r['gz'],3), round(r['beta'],3)])
            log(f"  offset {off:+.2f}: dBIC0={r['dbic0']:+.2f} g0={r['g0']:.3f} beta={r['beta']:+.3f}")
        for sig in (0.05, 0.10):
            acc = {k: [] for k in ('dbic0','dbic18','g0','gz','beta')}
            for i in range(N_DELTA_REAL):
                ds = dict(d_sgl); ds['delta'] = d0 + RNG.normal(0, sig, size=d0.size)
                r = fit_three(ds)
                for k in acc: acc[k].append(r[k])
                log(f"   [sig={sig:.2f}] {i+1}/{N_DELTA_REAL} dBIC0={r['dbic0']:+.2f} beta={r['beta']:+.3f}")
            rows.append([f"gauss sig={sig:.2f}",
                         round(np.mean(acc['dbic0']),2), round(np.mean(acc['dbic18']),1),
                         round(np.mean(acc['g0']),3), round(np.mean(acc['gz']),3),
                         round(np.mean(acc['beta']),3)])
        pd.DataFrame(rows, columns=["scenario","dBIC_vs_fix0","dBIC_vs_fix018","gamma0","gamma_z","beta_ani"]
                     ).to_csv("tableX_delta_sensitivity.csv", index=False)
        log("  wrote tableX_delta_sensitivity.csv")
        log("  INTERPRETATION: if dBIC and especially gamma0 stay stable across rows,")
        log("  the free-beta need and the slope bias are NOT delta_i artefacts.")
        summary['tableX'] = rows

    # ------------------------------------------------------------------- wrap up
    banner("DONE")
    with open("ALL_RESULTS_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: [[*(map(_jsonable, row))] for row in v] for k, v in summary.items()
                   if isinstance(v, list)}, f, indent=2, default=str)
    log(f"Total wall time: {(time.time()-t_start)/60:.1f} min")
    log("Outputs: table1..tableX .csv/.tex, P2_*/P3_*_Opt_GetDist.pdf + *_Comparison_Opt.pdf, ALL_RESULTS.log/json")
    if QUICK_TEST:
        log("\nREMINDER: this was QUICK_TEST=True. Set it to False for real numbers.")

def _jsonable(v):
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.integer,)):  return int(v)
    return v

if __name__ == "__main__":
    main()
