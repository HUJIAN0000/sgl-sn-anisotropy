# sgl-sn-anisotropy

Reproduction package for the paper

> **Can stellar velocity anisotropy be neglected in strong-lensing cosmology?**
> J. Hu et al. (submitted to *Astronomy & Astrophysics*)

This repository contains the matched galaxy-scale strong-lensing (SGL) + Type Ia
supernova (SN) catalogues, their covariance matrices, the pairing code, and the
analysis scripts that reproduce **every table and figure** in the paper.

The study tests, in a cosmology-independent way, whether the stellar
velocity-anisotropy parameter `β_ani` can be neglected in SGL cosmology. Using a
spherical-Jeans model on 107 SGL–SN pairs we find that a free `β_ani` is
statistically preferred (ΔBIC ≈ 5.6), that fixing it biases the density-slope
parameter `γ₀`, and that the recovered `β_ani` is best read as an **effective**
misfit-diagnostic parameter rather than a physical measurement of orbital
anisotropy.

The pairing itself uses the mixed-integer linear program (MILP) implemented in the
**CosmoMatcher** tool (https://github.com/HUJIAN0000/CosmoMatcher). The samples
here were produced with a headless command-line wrapper of that routine
(`match_milp_headless*.py`) in which the tie-breaker weight is set to `ε = 0.1`
(instead of the released default `1e-6`) to make the maximum-cardinality matching
deterministic and reproducible. See Appendix F of the paper.

---

## Repository layout

The analysis is run **once per matching tolerance** `Δd/d ∈ {0.03, 0.05, 0.10}`,
each in its own folder. Each folder is self-contained: drop into it and run.

```
sgl-sn-anisotropy/
├── README.md
│
├── 0.05/                          # baseline tolerance Δd/d = 5%  (paper main text)
│   ├── Pantheon+SH0ES_STAT+SYS.cov.zip   # raw SN covariance (UNZIP before re-pairing)
│   ├── Pantheon+SH0ES.dat                # raw Pantheon+SH0ES catalogue
│   ├── 130sgls2.CSV                      # parent SGL catalogue (130 systems, with delta_i)
│   ├── match_milp_headless0.05.py        # MILP pairing (headless CosmoMatcher, ε=0.1)
│   ├── matched_result_sgls_Om0.25.csv    # matched samples at Ω_m = 0.25 / 0.30 / 0.35
│   ├── matched_result_sgls_Om0.30.csv
│   ├── matched_result_sgls_Om0.35.csv
│   ├── cov_sgls_{lens,source,cross}_Om0.25.txt
│   ├── cov_sgls_{lens,source,cross}_Om0.30.txt
│   ├── cov_sgls_{lens,source,cross}_Om0.35.txt
│   ├── cosmo_tools.py                    # getdist plotting helper (needed for figures)
│   └── run_all_results0.05.py            # MAIN driver -> all tables + triangle plots
│
├── 0.03/                          # tolerance Δd/d = 3%
│   ├── match_milp_headless0.03.py
│   ├── matched_result_sgls_Om0.25.csv / _Om0.30.csv / _Om0.35.csv
│   ├── cov_sgls_{lens,source,cross}_Om0.25.txt / _Om0.30.txt / _Om0.35.txt
│   ├── cosmo_tools.py
│   └── run_all_results0.03.py
│
└── 0.1/                           # tolerance Δd/d = 10%
    ├── match_milp_headless0.1.py
    ├── matched_result_sgls_Om0.25.csv / _Om0.30.csv / _Om0.35.csv
    ├── cov_sgls_{lens,source,cross}_Om0.25.txt / _Om0.30.txt / _Om0.35.txt
    ├── cosmo_tools.py
    └── run_all_results0.10.py
```

> The three **raw** input files (`Pantheon+SH0ES_STAT+SYS.cov`,
> `Pantheon+SH0ES.dat`, `130sgls2.CSV`) are stored **only in `0.05/`** to avoid
> duplicating the large covariance file. The matched samples for every tolerance
> are already provided, so the analysis runs without them. They are needed only if
> you want to **re-run the pairing** (see below).
>
> `cosmo_tools.py` is required for the triangle plots. If it is missing the driver
> still produces all tables and simply skips the figures.

---

## Requirements

Python 3.9+ and:

```bash
pip install numpy pandas scipy emcee getdist matplotlib
```

(`emcee` is needed for the posteriors/Table 2; `getdist` + `matplotlib` for the
triangle plots. The model-comparison tables need only numpy/pandas/scipy.)

---

## Quick start — reproduce the paper (uses the provided matched samples)

```bash
cd 0.05
python run_all_results0.05.py
```

This consumes the matched samples already in the folder (no re-pairing) and writes:

| Output | Content |
|---|---|
| `table1_p2_modelcomp.{csv,tex}`  | Table 1 — P2 model comparison (free vs fixed β) |
| `table2_p2_posteriors.csv`       | Table 2 — P2 posteriors (emcee) |
| `table3_p3_modelcomp.{csv,tex}`  | Table 3 — P3 robustness check |
| `tableB1_tolerance.{csv,tex}`    | Table B.1 row for this tolerance |
| `tableC1_power.csv`              | Table C.1 — two-sided power analysis |
| `tableD1_omega.{csv,tex}`        | Table D.1 — Ω_m robustness (0.25/0.30/0.35) |
| `tableX_delta_sensitivity.csv`   | δ_i-sensitivity appendix table |
| `P2_*_Opt_GetDist.pdf`, `P2_Beta_Models_Comparison_Opt.pdf` | P2 triangle plots |
| `P3_*_Opt_GetDist.pdf`, `P3_Beta_Models_Comparison_Opt.pdf` | P3 triangle plots |
| `ALL_RESULTS.log`, `ALL_RESULTS_summary.json` | full log + machine-readable summary |

Then repeat in `0.03/` and `0.1/` (these produce the 3% and 10% rows of Table B.1):

```bash
cd ../0.03 && python run_all_results0.03.py
cd ../0.1  && python run_all_results0.10.py
```

**Speed knob.** Each driver starts with `QUICK_TEST = True` (short MCMC chains and a
small number of power realisations — good for a fast sanity check). For the
publication-grade Table 2 error bars, Table C.1 percentages, and final triangle
plots, set `QUICK_TEST = False` near the top of the driver and rerun. The
model-comparison tables (1, 3, B.1, D.1) are MLE-based and are identical either way.

---

## Optional — re-run the pairing from scratch

The matched samples are provided, but you can regenerate them.

**5% folder (has the raw files):**

```bash
cd 0.05
unzip Pantheon+SH0ES_STAT+SYS.cov.zip      # produces Pantheon+SH0ES_STAT+SYS.cov
python match_milp_headless0.05.py          # writes matched_result_sgls_Om*.csv + cov_*.txt
python run_all_results0.05.py
```

**3% / 10% folders:** their `match_milp_headless0.03.py` / `match_milp_headless0.1.py`
read the same three raw files, which are not duplicated here. To re-pair, copy them
in first:

```bash
cd 0.03
cp ../0.05/130sgls2.CSV ../0.05/Pantheon+SH0ES.dat .
cp ../0.05/Pantheon+SH0ES_STAT+SYS.cov .          # after unzipping it in 0.05/
python match_milp_headless0.03.py
python run_all_results0.03.py
```

---

## Verification (self-check)

Headline numbers a correct run should reproduce:

| Tolerance | N pairs | Free max ln L | ΔBIC (free vs β=0) |
|---|---|---|---|
| 3%  | 104 | −398.54 | **1.92** |
| 5%  | 107 | −416.41 | **5.64** |
| 10% | 117 | −459.48 | **7.58** |

At 5%: Table 3 gives ΔBIC = 5.65 / 33.23 (vs β=0 / β=0.18) with γ_s ≈ −0.03; the
Ω_m-robustness row for Ω_m = 0.30 equals the 5% baseline (ΔBIC ≈ 5.64).

---

## Citation

If you use this code or the matched catalogues, please cite the paper (J. Hu et al.,
*A&A*, submitted) and the CosmoMatcher tool
(https://github.com/HUJIAN0000/CosmoMatcher).

## Contact

Jian Hu — dg1626002@smail.nju.edu.cn
