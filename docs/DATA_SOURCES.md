# Input sources and attribution

- `data/raw/sgls130.CSV`: 130 lenses, with original aperture dispersions, angular radii and individual delta values supplied by Yun Chen. Compilation and photometric procedure: Chen, Li, Shu & Cao (2019), MNRAS 488, 3745, doi:10.1093/mnras/stz1902. Individual delta uncertainties and fit records were not supplied. Do not substitute the published population scatter for a per-lens measurement error.
- `data/raw/Pantheon+SH0ES.dat` and the compressed STAT+SYS covariance: official Pantheon+SH0ES data release, https://github.com/PantheonPlusSH0ES/DataRelease. The full magnitude table and covariance are retained, including the original row ordering. The compressed covariance expands to the original input bytes; the target hash is recorded in `provenance/compressed_input.json`.
- `data/primary/`: generated with the supplied CosmoMatcher 1.4.3 implementation and the packaged settings. No CID is used twice across the 214 assigned supernova counterparts. Original row and covariance indices are retained.
- `tools/`: original matching and plotting utilities, with their existing source headers unchanged. The bounded-density adapter is separate from the original plotting module.
- `aa.cls`, `aa.bst`: journal-provided LaTeX files; retain their embedded notices.

The root MIT licence is preserved from the existing project and applies to project software subject to existing third-party notices. It does not grant new ownership or redistribution terms over externally supplied observational data or journal material. The data provider did not supply a separate licence statement; this package records attribution rather than inventing one. Refer to original sources for their terms.
