# Strong-lens anisotropy and tracer-profile sensitivity

Code, observational inputs and reference results for manuscript v4 of the aa61321-26 second-review revision:

**Anisotropy and tracer-profile assumptions in aperture-based strong-lens dynamics: Conditional inference from supernova distance ratios and unresolved stellar kinematics**

Jian Hu, Yi Liu, Jian-Ping Hu and Zhongmu Li

Repository: https://github.com/HUJIAN0000/sgl-sn-anisotropy

This revision replaces the initial-submission analysis in the current repository tree. Earlier commits and releases document that earlier analysis; their catalogues, numerical summaries and drivers should not be mixed with this revision. The present study tests how a common-anisotropy comparison depends on tracer profiles, photometric assumptions and catalogue treatment. It does not estimate the Hubble constant or cosmic curvature.

## Quick start: inspect and verify the published inputs

Use Python 3.11 with the packages listed in `requirements-tested.txt`. The pinned list records the calculation environment; it does not install packages automatically.

```text
python -B reproduce_revision2.py verify
```

This command safely expands the compressed Pantheon+ covariance when needed, checks file hashes and loads the frozen matched catalogue. It verifies 107 lenses, 214 distinct supernova CIDs and 12 systems with fractional distance-ratio uncertainty above 20%. It does not run an optimiser or MCMC.

## Repository contents

| Path | Contents |
| --- | --- |
| `data/raw/` | Original 130-lens table with individual luminosity slopes; full Pantheon+SH0ES table; compressed STAT+SYS covariance |
| `data/primary/` | Frozen 107-lens assignment, original row-index map, lens/source/cross covariance blocks |
| `data/matching_variants/`, `data/derived/` | Alternative assignments and supporting catalogue products |
| `aa_revision/` | Scientific model, likelihood, quadrature and inference implementation |
| `tools/` | Original CosmoMatcher 1.4.3 and author plotting utility |
| `scripts/`, `configs/` | Production drivers, plotting adapter and calculation settings |
| `reference_results/` | Completed likelihood comparisons, 16 main posterior summaries, eight latent-slope summaries, mock outcomes, sensitivities and diagnostics |
| `figures/` | All 16 main individual posterior triangles and four overlay figures |
| `manuscript/clean/` | Clean revised TeX source, required tables/figures and compiled manuscript |
| `provenance/` | Input hashes, calculation validation records and the release-sample index |
| `docs/` | Data dictionary, result map, attribution and reproduction notes |

Reference results remain separate from `results/`, which is populated by restored samples or new calculations. Historical provenance may contain the original machine's absolute paths; those records are not execution inputs. Executable matching paths are portable and relative to the repository root.

## Use the completed posterior samples without rerunning MCMC

The five `posterior_*.zip` files accompanying this revision are intended as GitHub Release assets, separate from regular Git-tracked files. Download all five assets to one directory, then run:

```text
python -B restore_posterior_assets.py --assets /path/to/downloaded/assets
```

The helper checks every archive member against `provenance/posterior_release_index.json`, restores the accepted NPZ samples and summaries into `results/production/`, and restores the main sampling status record. It refuses to replace a different existing file. On Windows, quote a path containing spaces.

These assets contain all retained NPZ draws for 16 main and eight latent-slope posterior products, with matching SHA-256 hashes. They do not contain the full HDF5 sampler backends or every latent-coordinate time series. Global-parameter trace plots, numerical-accuracy records and diagnostic summaries are provided under `reference_results/validation/`. Restoring the NPZ files is sufficient to use the retained global-parameter distributions and, with the plotting environment, regenerate the main triangles:

```text
python -B reproduce_revision2.py figures
```

The figures already included in the repository are the manuscript outputs. Plot regeneration is optional; it is not another scientific sampling task.

## Recompute the analysis

For an independent calculation, use a fresh repository copy, keep `reference_results/` intact, and run these stages in order:

```text
python -B reproduce_revision2.py fit --workers 8
python -B reproduce_revision2.py posteriors --workers 8
python -B reproduce_revision2.py sensitivities --workers 8
python -B reproduce_revision2.py matching-variants
python -B reproduce_revision2.py mocks --workers 8
python -B reproduce_revision2.py latent --workers 16
python -B reproduce_revision2.py figures
python -B reproduce_revision2.py audit
```

These commands perform full calculations and may take substantial time. Acceptance depends on the stated convergence criteria, rather than a requested number of steps. The final audit reads outputs of the complete calculation, including sampler backends; restoring NPZ assets alone is not a substitute for those backends. Parallel workers operate on independent jobs or saved chains, with low-level thread counts restricted to avoid nested oversubscription. `matching-variants` runs serially and rejects an explicit `--workers` option. Parallel calculation stages default to eight workers. Subprocess stdout and stderr are saved separately under `logs/`, with paths printed before launch. The latent stage validates all eight completed initial products and their four closed checkpoints before extending any chains. Only the initial driver's explicit convergence exit with complete 80000-step unaccepted products permits recovery; crashes, missing products and inconsistent checkpoints stop the entrypoint. The sampler, seed and convergence criteria are unchanged.

The default calculation starts from the frozen assignment. To repeat raw-input matching, make a separate fresh repository copy and complete these steps in that same copy:

1. Run `python -B reproduce_revision2.py verify` while the published `data/primary/` is still present. This expands and checks the raw covariance.
2. Move that copy's `data/primary/` to a backup location outside the working tree.
3. In the same verified copy, run `python -B scripts/rebuild_inputs.py`. The driver refuses to overwrite an existing assignment.
4. Continue the calculation stages above. If the assignment is deliberately changed, analyse it separately from the published reference results.

The optional initial-submission comparison reads `reference_history/initial_submission_matched_catalog.csv` only if separately supplied. This historical file is not required to generate the new assignment.

The upload package was checked for input integrity, accepted-sample identity, portable matching paths, archive integrity and Python syntax. The completed scientific calculations are the reference run; the portable package was not used to repeat the full production calculation during packaging.

## Reference results and interpretation

The baseline assignment uses a relative comoving-distance matching tolerance of 0.05 in a fiducial flat background with Omega_m = 0.30. That background specifies catalogue matching, not a fitted cosmological parameter.

| Tracer | BIC(fixed beta=0) minus BIC(free beta) | BIC(fixed beta=0.18) minus BIC(free beta) |
| --- | ---: | ---: |
| Individual-slope power law | 6.7380 | 41.5057 |
| Hernquist | -3.8872 | -2.2601 |
| Sersic n=4 | -4.6717 | -4.3015 |

There are 16 accepted main posterior products and eight accepted latent-slope posterior products. Recovery tests use 800 symmetric-noise realisations and 4800 fits. The lower-distance-uncertainty subset contains 95 lenses and has power-law isotropy BIC difference 2.5084. See `docs/RESULTS_MAP.md` for machine-readable sources.

The comparison is conditional on spherical dynamics, power-law mass profiles and a common constant anisotropy. Curved tracer families, Sersic indices and the PSF examples are assumed scenarios, not new fits to the original images. The delta-error widths are explicit sensitivity assumptions; population scatter is not substituted for individual measurement error. Convergence does not establish physical model adequacy, and these comparisons do not identify a unique cause of the observational residuals.

## Sources and licence

The lens compilation and photometric procedure are from Chen, Li, Shu & Cao (2019), MNRAS 488, 3745, https://doi.org/10.1093/mnras/stz1902. Individual delta values were provided by Yun Chen. The original survey papers are cited in the manuscript. Supernova inputs are from the official Pantheon+ release: https://github.com/PantheonPlusSH0ES/DataRelease.

The repository's existing MIT software licence is preserved verbatim in `LICENSE`. Original third-party headers and notices remain in place. This software licence does not relicense observational datasets, journal class files or other third-party material. See `docs/DATA_SOURCES.md` for attribution and file-level provenance.


## Entrypoint checks

```text
python -B -m unittest discover -s tests -p test_reproduce_entrypoint.py -v
```

These 21 lightweight tests exercise normal acceptance, expected convergence extension, crashes, invalid or missing outputs and argument handling using small test fixtures. They do not launch scientific inference.
