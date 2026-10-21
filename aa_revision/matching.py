"""Strict cardinality-then-cost MILP; one use of each physical supernova."""
from pathlib import Path
import io
import time
import zipfile
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import quad
from scipy.optimize import Bounds, LinearConstraint, milp
from .common import sha256, write_json, environment


def chi(z, omega):
    return quad(lambda t: (omega * (1 + t)**3 + 1 - omega)**(-0.5),
                0, float(z), epsabs=1e-11)[0]


def read_covariance(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".cov")]
            if len(names) != 1:
                raise ValueError("Covariance ZIP must contain exactly one .cov file")
            raw = z.read(names[0])
        stream = io.BytesIO(raw)
    else:
        stream = path.open("rb")
    with stream:
        n = int(stream.readline().split()[0])
        values = np.loadtxt(stream).ravel()
    if values.size != n*n:
        raise ValueError("Incorrect covariance dimension/header")
    cov = values.reshape(n, n)
    asymmetry = float(np.max(np.abs(cov - cov.T)))
    if not np.isfinite(cov).all() or asymmetry > 1e-7:
        raise ValueError("Invalid/asymmetric SN covariance")
    # Decimal serialization has tiny asymmetries. Use one explicitly symmetric
    # matrix for both extraction and the likelihood; record the change.
    return (cov + cov.T) / 2, asymmetry


def solve_matching(lenses, sn, omega=0.30, tolerance=0.05,
                   cost_metric="absolute", unique_by="CID", time_limit=600):
    if cost_metric not in ("absolute", "relative") or unique_by not in ("CID", "row"):
        raise ValueError("Unknown cost metric or uniqueness rule")
    if not (0 < omega < 1 and 0 < tolerance < 1):
        raise ValueError("Require 0<Omega_m,tolerance<1")
    if not lenses.index.equals(pd.RangeIndex(len(lenses))) or not sn.index.equals(pd.RangeIndex(len(sn))):
        raise ValueError("Keep unfiltered input row indices aligned with covariance")
    if sn.CID.isna().any():
        raise ValueError("Missing physical SN identifiers")
    usn = np.array([chi(z, omega) for z in sn.zHD])
    ul = np.array([chi(z, omega) for z in lenses.zl])
    us = np.array([chi(z, omega) for z in lenses.zs])
    labels = sn.CID.astype(str).to_numpy() if unique_by == "CID" else sn.index.to_numpy()
    resources = {key: len(lenses) + i for i, key in enumerate(dict.fromkeys(labels))}
    triplets, cost, rows = [], [], []
    for i in range(len(lenses)):
        if not (0 < ul[i] < us[i]):
            raise ValueError(f"Invalid lens/source redshifts at parent row {i}")
        left = np.flatnonzero(np.abs(usn - ul[i])/ul[i] <= tolerance)
        right = np.flatnonzero(np.abs(usn - us[i])/us[i] <= tolerance)
        for a in left:
            for b in right:
                if labels[a] == labels[b]:
                    continue
                dl, ds = usn[a] - ul[i], usn[b] - us[i]
                triplets.append((i, int(a), int(b)))
                rows.extend([i, resources[labels[a]], resources[labels[b]]])
                cost.append(dl*dl + ds*ds if cost_metric == "absolute" else
                            (dl/ul[i])**2 + (ds/us[i])**2)
    nc, nr = len(triplets), len(lenses) + len(resources)
    if not nc:
        raise RuntimeError("No eligible triplets")
    A = sparse.coo_matrix((np.ones(3*nc), (rows, np.repeat(np.arange(nc), 3))),
                          shape=(nr, nc)).tocsc()
    options = {"mip_rel_gap": 0.0, "time_limit": float(time_limit)}
    start = time.perf_counter()
    first = milp(-np.ones(nc), integrality=np.ones(nc), bounds=Bounds(0, 1),
                 constraints=LinearConstraint(A, -np.inf, 1), options=options)
    if not first.success:
        raise RuntimeError(f"Stage 1 not certified: {first.message}")
    k = int(np.rint(first.x.sum()))
    A2 = sparse.vstack([A, sparse.csc_matrix(np.ones((1, nc)))], format="csc")
    con = LinearConstraint(A2, np.r_[np.full(nr, -np.inf), k], np.r_[np.ones(nr), k])
    cost = np.array(cost)
    scale = max(float(cost.max()), 1e-12)
    second = milp(cost/scale, integrality=np.ones(nc), bounds=Bounds(0, 1),
                  constraints=con, options=options)
    if not second.success:
        raise RuntimeError(f"Stage 2 not certified: {second.message}")
    selected = np.flatnonzero(second.x > 0.5)
    if len(selected) != k or np.max(A @ np.rint(second.x)) > 1.000001:
        raise RuntimeError("Integer assignment failed validation")
    chosen = [triplets[j] for j in selected]
    diagnostics = dict(Omega_m=omega, tolerance=tolerance, cost_metric=cost_metric,
        unique_by=unique_by, candidate_triplets=nc, maximum_cardinality=k,
        geometric_cost=float(cost[selected].sum()), distance_units="dimensionless chi",
        stage1_gap=float(first.mip_gap), stage2_gap=float(second.mip_gap),
        stage1_message=first.message, stage2_message=second.message,
        elapsed_seconds=time.perf_counter()-start,
        optimum_unique="not established; exported triplets define the frozen sample")
    return chosen, diagnostics


def prepare(raw, out, matching):
    raw, out = Path(raw), Path(out)
    # Never overwrite a frozen catalogue after posterior sampling has begun.
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Frozen catalogue exists: {out}; use another output directory")
    lenses = pd.read_csv(raw / "130sgls2.CSV")
    lenses.columns = lenses.columns.str.strip()
    required = ["Lens name", "zl", "zs", "thetaE", "thetaeff", "thetaap",
                "sigma_ap", "dsigma_ap", "delta"]
    if lenses[required].isna().any().any() or lenses["Lens name"].duplicated().any():
        raise ValueError("Missing input values or duplicate lens names")
    sn = pd.read_csv(raw / "Pantheon+SH0ES.dat", sep=r"\s+", dtype={"CID": str})
    covfile = raw / "Pantheon+SH0ES_STAT+SYS.zip"
    if not covfile.exists():
        covfile = raw / "Pantheon+SH0ES_STAT+SYS.cov"
    cov, asymmetry = read_covariance(covfile)
    if cov.shape != (len(sn), len(sn)):
        raise ValueError("SN table/covariance row mismatch")
    chosen, info = solve_matching(lenses, sn, **matching)
    out.mkdir(parents=True, exist_ok=True)
    cat, provenance = [], []
    for i, a, b in chosen:
        row = lenses.iloc[i].to_dict()
        row.update(sn_lens_m_b_corr=float(sn.iloc[a].m_b_corr),
                   sn_source_m_b_corr=float(sn.iloc[b].m_b_corr))
        cat.append(row)
        provenance.append({"Lens name": row["Lens name"], "parent_row": i,
            "sn_lens_index": a, "sn_source_index": b,
            "sn_lens_CID": sn.iloc[a].CID, "sn_source_CID": sn.iloc[b].CID,
            "sn_lens_zHD": sn.iloc[a].zHD, "sn_source_zHD": sn.iloc[b].zHD})
    df, pr = pd.DataFrame(cat), pd.DataFrame(provenance)
    ids = pr[["sn_lens_CID", "sn_source_CID"]].to_numpy().ravel()
    if matching.get("unique_by", "CID") == "CID" and len(set(ids)) != len(ids):
        raise RuntimeError("Physical SN reuse detected")
    df.to_csv(out / "matched_catalog.csv", index=False)
    pr.to_csv(out / "sn_index_provenance.csv", index=False)
    il, js = pr.sn_lens_index.to_numpy(), pr.sn_source_index.to_numpy()
    for role, a, b in [("lens", il, il), ("source", js, js), ("cross", il, js)]:
        np.savetxt(out / f"cov_{role}.txt", cov[np.ix_(a, b)], fmt="%.17e")
    files = ["matched_catalog.csv", "sn_index_provenance.csv", "cov_lens.txt",
             "cov_source.txt", "cov_cross.txt"]
    info.update(input_sha256={p.name: sha256(p) for p in
                 [raw/"130sgls2.CSV", raw/"Pantheon+SH0ES.dat", covfile]},
        file_sha256={name: sha256(out/name) for name in files}, environment=environment(),
        raw_cov_max_asymmetry=asymmetry, covariance_policy="(C+C.T)/2 before extraction",
        survey_counts=df["Survey name"].value_counts().to_dict())
    write_json(out / "manifest.json", info)
    return info
