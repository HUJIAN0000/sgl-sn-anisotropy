# Data dictionary

## Lens input: `data/raw/sgls130.CSV`

| Column | Meaning |
| --- | --- |
| Lens name | Lens identifier |
| zl, zs | Lens and source redshifts |
| thetaE | Einstein radius, arcseconds |
| thetaeff | Projected effective radius, arcseconds |
| thetaap | Spectroscopic aperture radius, arcseconds |
| sigma_ap, dsigma_ap | Original aperture velocity dispersion and quoted statistical error, km/s |
| delta | Three-dimensional luminosity-density slope in nu(r) proportional to r^(-delta) |
| Survey name | Parent survey attribution |

The aperture dispersion is modelled at its listed aperture. The compilation's transformation to an effective-radius/2 aperture is not applied a second time. The individual delta estimates describe the inner photometric fit; treating them as global power laws is an explicit model assumption.

## Frozen assignment

`matched_catalog.csv` retains lens input fields and the paired supernova fields prefixed `sn_lens_` and `sn_source_`. CID values are identifiers: load them as strings to preserve leading zeros. `sn_*_original_index` and the index map refer to original data rows. Pandas/NumPy row indices are zero-based.

`original_row_index_map.csv` records `matrix_position`, `matched_row`, role, original catalogue/covariance row, identifier and redshift. The covariance blocks are ordered exactly as the matched catalogue. `cov_lens.txt` and `cov_source.txt` are magnitude covariance blocks; `cov_cross.txt` is the lens-counterpart/source-counterpart cross block. The original combined magnitude covariance can contain rounding-level asymmetry; the likelihood uses its symmetrised form.

`R_flat` is the flat-distance-sum-rule ratio D_ls/D_s constructed from the paired supernova magnitudes and target lens/source redshifts. The ratio and its propagated covariance are distinct from the raw magnitude covariance. Refer to the manuscript equations and model implementation for the propagation and treatment of scatter.

## Posterior files

NPZ `names` give the parameter-column order and `samples` are the retained global-parameter draws. Summary JSON records priors, quantiles and convergence checks. Main cases are `powerlaw`, `hernquist`, `sersic_n4`, and `p3_powerlaw`; beta treatments are `flat`, `gaussian`, `fixed0`, and `fixed018`. A fixed beta is not a sampled column. The latent-slope NPZ products contain the retained global draws; they do not replace full latent-coordinate chain backends.
