# Map from manuscript results to files

| Result | Reference file |
| --- | --- |
| Main maximum-likelihood and BIC comparisons | `reference_results/report/new_mle_comparisons.csv` |
| Main and latent posterior parameters | `reference_results/posterior_parameters.csv` and `reference_results/posteriors/` |
| Convergence diagnostics, 24 accepted products | `reference_results/diagnostics.csv` |
| Boundary sensitivity | `reference_results/posterior_boundary_fractions.csv` |
| Symmetric-noise and tracer-mismatch experiments | `reference_results/report/mock_summary.json`, `mock_fits.csv` |
| Distance/selection/photometry sensitivity | `reference_results/report/new_sensitivity_comparisons.csv` |
| Matching variants | `reference_results/report/new_matching_variant_comparisons.csv` |
| Numerical accuracy and trace plots | `reference_results/validation/` |
| TeX tables used by manuscript | `manuscript/clean/tables/` |
| Author-style posterior triangles and overlays | `figures/` |
| Exact accepted NPZ hashes and Release archives | `provenance/posterior_release_index.json` |

Main posterior products use four independent ensembles. Latent-slope products use four independent chains and 110–112 varying coordinates, depending on fixed/free beta and the common-offset scenario. Approximate ESS and rank-normalised split Rhat support sampling diagnostics; they do not test physical model adequacy.
