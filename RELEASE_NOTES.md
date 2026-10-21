# aa61321-26 revised analysis: manuscript v4

This revision replaces the initial-submission analysis with conditional tests of anisotropy and tracer-profile assumptions. It includes the original 130-lens input, full Pantheon+ magnitude data and covariance, a frozen 107-lens/214-CID assignment, curved-tracer Jeans models, likelihood comparisons, 16 main and eight latent-slope posterior products, 800 symmetric-noise recovery realisations (4800 fits), matching/distance/selection/photometric controls, all manuscript triangles, and the revised clean source.

Upload `sgl-sn-anisotropy_v4_repository_contents.zip` as the core snapshot and all five `posterior_*.zip` files as supplementary posterior assets. The core code/data are also available in the repository tree. The SHA256SUMS file identifies the exact assets. Extract the core snapshot, then use `restore_posterior_assets.py` to verify and restore samples without repeating inference.

Full HDF5 sampler backends are not bundled. Retained global draws, posterior summaries and diagnostic records are included. Earlier releases reproduce the initial-submission analysis; do not mix their assignments or summaries with this revision. No DOI is assigned by this package.

The v4 manuscript foregrounds the quantitative tracer and recovery results while retaining the model scope. The public entrypoint now reports latent-stage failures immediately, validates closed checkpoints before extension, and identifies matching variants as a serial stage. Raw-matching instructions use a single verified working copy. The scientific calculations and all 24 accepted posterior arrays are unchanged.
