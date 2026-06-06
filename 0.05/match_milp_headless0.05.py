# -*- coding: utf-8 -*-
"""
match_milp_headless.py  (v2, adapted to your real files)
========================================================
Headless (no-GUI) version of the CosmoMatcher MILP pairing, to:
  (1) regenerate the matched sample at Omega_m = 0.25 / 0.30 / 0.35 for the
      pairing-cosmology robustness test (Appendix D / tab:omega), and
  (2) reproduce matched_result_sgls.csv + cov_sgls_*.txt without the GUI.

Objective / constraints / solver are IDENTICAL to CosmoMatcher_v1_1.py
(max-cardinality assignment, c = -1 + 0.1*dist^2, scipy.optimize.milp / HiGHS;
sub-covariances via np.ix_). Only the GUI is removed and the I/O is wired to
your actual file formats.

YOUR FILE FORMATS (already accounted for below):
  - 130sgls.CSV  : comma-separated; columns
        Lens name, zl, zs, thetaE, thetaeff, thetaap, sigma_ap, dsigma_ap, Survey name
        (NOTE: there is NO 'delta' column here -- see DELTA handling below)
  - Pantheon_SH0ES.dat : WHITESPACE-separated; redshift col 'zHD', mag col 'm_b_corr';
        1701 rows, in the SAME order as the covariance matrix.
  - Pantheon+SH0ES_STAT+SYS.cov : one value per line, first line = 1701 (the size),
        then 1701*1701 entries (row-major).

DELTA (delta_i) HANDLING:
  delta_i is per-lens and does NOT depend on Omega_m. The matched sample you
  already ran (matched_result_sgls.csv, 107 rows) contains both 'Lens name' and
  'delta', so we merge delta from there by lens name. If you prefer, point
  DELTA_FILE at any 2-column table (lens name, delta).
"""

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse

# ============================================================================
# CONFIG  -- file names/columns now match your actual files
# ============================================================================
SGL_CSV      = "130sgls2.CSV"
SN_DAT       = "Pantheon+SH0ES.dat"
PANTHEON_COV = "Pantheon+SH0ES_STAT+SYS.cov"

SN_Z_COL     = "zHD"
SN_MAG_COL   = "m_b_corr"
LENS_NAME_COL = "Lens name"
ZL_COL, ZS_COL = "zl", "zs"

# where to get per-lens delta_i (merged by lens name); your existing matched file works
DELTA_FILE     = "matched_result_sgls.csv"
DELTA_NAME_COL = "Lens name"   # lens-name column inside DELTA_FILE
DELTA_VAL_COL  = "delta"       # delta column inside DELTA_FILE

TOL          = 0.05            # 5% comoving-distance tolerance (baseline)
OMEGA_LIST   = [0.25, 0.30, 0.35]

# Tie-breaker weight on the squared geometric mismatch. The original pipeline used
# 1e-6, which is below the solver tolerance, so the maximum-cardinality matching
# was NON-UNIQUE (the solver returned an arbitrary tight-or-loose vertex; verified
# to differ system-by-system from another valid 107-pair solution). EPS=0.1 makes
# the objective lexicographic -- maximise the number of pairs first, then UNIQUELY
# minimise the total geometric mismatch -- which is deterministic and reproducible
# across machines/solver versions, while still preserving cardinality (any EPS<=1
# keeps every pair's -1+EPS*d < 0, since d<=~0.2 within the 5% tolerance).
EPS          = 0.1


# ============================================================================
# Cosmology + covariance loader
# ============================================================================
def comoving_distance(z, wm):
    if z <= 0:
        return 0.0
    val, _ = quad(lambda zz: (wm * (1 + zz)**3 + (1 - wm))**-0.5, 0.0, z)
    return val


def load_pantheon_matrix(path):
    """First line is the size N; the next N*N lines are the row-major entries."""
    with open(path) as f:
        n = int(f.readline().split()[0])
    flat = pd.read_csv(path, header=None, skiprows=1).values.ravel().astype(float)
    assert flat.size == n * n, f"expected {n*n} entries, got {flat.size}"
    return flat.reshape(n, n)


def resolve_delta(obj_df):
    """Return a dict {lens_name: delta}. Prefer a 'delta' column in the SGL CSV;
    otherwise merge from DELTA_FILE by lens name."""
    if DELTA_VAL_COL in obj_df.columns:
        return dict(zip(obj_df[LENS_NAME_COL], obj_df[DELTA_VAL_COL]))
    try:
        d = pd.read_csv(DELTA_FILE)
    except Exception as e:
        raise RuntimeError(
            f"No 'delta' column in {SGL_CSV} and could not read DELTA_FILE "
            f"'{DELTA_FILE}': {e}. Provide per-lens delta_i keyed by lens name.")
    return dict(zip(d[DELTA_NAME_COL], d[DELTA_VAL_COL]))


# ============================================================================
# Core MILP pairing for one Omega_m  (mirrors CosmoMatcher exactly)
# ============================================================================
def match_one(obj_df, sn_df, wm, tol, eps=EPS):
    sn_z   = sn_df[SN_Z_COL].values
    sn_idx = sn_df.index.values                 # row index == covariance index
    sn_d   = np.array([comoving_distance(z, wm) for z in sn_z])

    zl = obj_df[ZL_COL].values
    zs = obj_df[ZS_COL].values
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
        cl = np.abs(sn_d - dl[i])[mask_l]
        cs = np.abs(sn_d - ds[i])[mask_s]
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
        rows += [map_obj[oid], map_sn[sn1], map_sn[sn2]]
        cols += [j, j, j]
        vals += [1, 1, 1]
    A = sparse.coo_matrix((vals, (rows, cols)),
                          shape=(len(map_obj) + len(map_sn), len(candidates)))

    c_vec = -1.0 + eps * np.array([g[3] for g in candidates])   # max-cardinality, then min total distance
    res = milp(c=c_vec, constraints=LinearConstraint(A, -np.inf, 1),
               integrality=np.ones_like(c_vec), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(f"MILP failed: {res.message}")

    sel = np.where(res.x > 0.5)[0]
    return [(candidates[k][0], candidates[k][1], candidates[k][2]) for k in sel]


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    obj_df = pd.read_csv(SGL_CSV)
    sn_df  = pd.read_csv(SN_DAT, sep=r"\s+")
    print(f"SGL rows={len(obj_df)}  SN rows={len(sn_df)}")

    cov = load_pantheon_matrix(PANTHEON_COV)
    print(f"covariance: {cov.shape}")
    assert cov.shape[0] == len(sn_df), \
        "SN catalogue length must equal covariance dimension (row order matters!)"

    delta_map = resolve_delta(obj_df)

    for wm in OMEGA_LIST:
        tag = f"Om{wm:.2f}"
        pairs = match_one(obj_df, sn_df, wm, TOL)
        print(f"[Omega_m={wm}] matched {len(pairs)} pairs")

        rows, idx_l, idx_s, missing = [], [], [], 0
        for oid, sn1, sn2 in pairs:
            r = obj_df.loc[oid].to_dict()                 # carry ALL SGL columns
            name = r.get(LENS_NAME_COL)
            if name not in delta_map or pd.isna(delta_map.get(name)):
                missing += 1
                continue                                  # skip lenses without delta_i
            r["delta"] = float(delta_map[name])
            r["sn_lens_m_b_corr"]   = sn_df.loc[sn1, SN_MAG_COL]
            r["sn_source_m_b_corr"] = sn_df.loc[sn2, SN_MAG_COL]
            rows.append(r); idx_l.append(sn1); idx_s.append(sn2)
        if missing:
            print(f"  (skipped {missing} matched lenses lacking a delta_i value)")

        out = pd.DataFrame(rows)
        out.to_csv(f"matched_result_sgls_{tag}.csv", index=False)

        idx_l = np.array(idx_l, int); idx_s = np.array(idx_s, int)
        np.savetxt(f"cov_sgls_lens_{tag}.txt",   cov[np.ix_(idx_l, idx_l)], fmt="%.8e")
        np.savetxt(f"cov_sgls_source_{tag}.txt", cov[np.ix_(idx_s, idx_s)], fmt="%.8e")
        np.savetxt(f"cov_sgls_cross_{tag}.txt",  cov[np.ix_(idx_l, idx_s)], fmt="%.8e")
        print(f"  wrote matched_result_sgls_{tag}.csv ({len(out)} rows) "
              f"and cov_sgls_*_{tag}.txt")

    print("\nDone. For each Omega_m: point the analysis script's load_data() at "
          "matched_result_sgls_Om*.csv and cov_sgls_*_Om*.txt (rename to the "
          "default names, or add a path argument to load_data), rerun the P2 "
          "models, and record DeltaBIC, beta_ani, gamma_z for tab:omega.")
