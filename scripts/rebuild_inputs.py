"""Audit raw tables; invoke the author's matcher; verify covariance row lineage."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
import importlib.util
import json
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aa_revision.common import ROOT, sha256, write_json, read_json, environment


def matcher_module():
    path = ROOT / 'tools/CosmoMatcherv1.4.3.py'
    spec = importlib.util.spec_from_file_location('author_cosmomatcher', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    m = matcher_module()
    raw = ROOT / 'data/raw'
    lens = pd.read_csv(raw / 'sgls130.CSV')
    sn = pd.read_csv(raw / 'Pantheon+SH0ES.dat', sep=r'\s+', dtype={'CID': str})
    numeric = ['zl', 'zs', 'thetaE', 'thetaeff', 'thetaap', 'sigma_ap', 'dsigma_ap', 'delta']
    assert len(lens) == 130 and lens['Lens name'].is_unique
    assert np.isfinite(lens[numeric].to_numpy()).all()
    assert (lens[numeric[:-1]].to_numpy() > 0).all() and (lens.zs > lens.zl).all()
    assert lens.delta.between(1, 3, inclusive='neither').all()
    assert np.isfinite(sn[['zHD', 'm_b_corr']].to_numpy()).all() and (sn.zHD > 0).all()
    covariance = m.load_matrix(raw / 'Pantheon+SH0ES_STAT+SYS.cov', expected_n=len(sn))
    symmetric, validation = m.normalize_covariance(covariance, 'raw audit total SN covariance', True)
    np.linalg.cholesky(symmetric)
    audit = dict(lens_rows=len(lens), lens_names_unique=True,
        survey_counts=lens['Survey name'].value_counts().to_dict(),
        delta_range=[float(lens.delta.min()), float(lens.delta.max())],
        supernova_rows=len(sn), distinct_CID=int(sn.CID.nunique()),
        duplicate_CID_observation_rows=int(sn.CID.duplicated(keep=False).sum()),
        covariance_shape=list(covariance.shape), covariance_validation=validation,
        covariance_positive_definite=True,
        diagonal_vs_table_max_abs=float(np.max(np.abs(np.diag(symmetric) - sn.m_b_corr_err_DIAG.to_numpy()**2))),
        input_sha256={p.name: sha256(p) for p in sorted(raw.iterdir()) if p.is_file()},
        tools_sha256={p.name: sha256(p) for p in sorted((ROOT / 'tools').glob('*.py'))},
        environment=environment(), parent_161_minus_31='Author supplied provenance; excluded 31 objects not present for independent verification',
        redshift_convention='zHD retained as in earlier pipeline; zHEL and zCMB preserved for explicit sensitivity checks',
        error_convention='STAT+SYS used once; no additional SN diagonal variance added')
    write_json(ROOT / 'provenance/raw_audit.json', audit)
    print('RAW AUDIT', json.dumps(audit, ensure_ascii=False), flush=True)
    cfg = read_json(ROOT / 'configs/matching_primary.json')
    summary_path = ROOT / 'results/matching/primary/rebuild_result.json'
    if summary_path.exists():
        raise FileExistsError('Matching already completed; do not silently replace a frozen assignment')
    result = m.run_config(cfg, log_func=lambda message: print(message, flush=True))
    matched, summary = result.matched, result.summary
    assert summary['solver']['optimal'] and summary['uniqueness'] == 'object'
    n = len(matched)
    li = matched.sn_lens_original_index.to_numpy(int)
    si = matched.sn_source_original_index.to_numpy(int)
    assert len(set(np.r_[li, si])) == 2*n
    assert len(set(sn.CID.iloc[np.r_[li, si]])) == 2*n
    keys = dict(lens='ll', source='ss', cross='ls')
    # Confirm API block names against their actual original row indices.
    direct = dict(ll=symmetric[np.ix_(li, li)], ss=symmetric[np.ix_(si, si)],
                  ls=symmetric[np.ix_(li, si)])
    for block, name in keys.items():
        assert block in result.matrices, f'Matcher matrix keys: {list(result.matrices)}'
        np.testing.assert_allclose(result.matrices[block], direct[name], atol=1e-14, rtol=1e-12)
    for role, idx in [('sn_lens', li), ('sn_source', si)]:
        np.testing.assert_array_equal(matched[f'{role}_CID'].astype(str), sn.CID.iloc[idx].to_numpy())
        np.testing.assert_allclose(matched[f'{role}_m_b_corr'].to_numpy(float), sn.m_b_corr.iloc[idx], rtol=0, atol=0)
    dest = ROOT / 'data/primary'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'manifest.json').exists():
        raise FileExistsError('Primary catalogue is already frozen')
    matched.to_csv(dest / 'matched_catalog.csv', index=False)
    result.index_map.to_csv(dest / 'original_row_index_map.csv', index=False)
    for target, key in keys.items():
        np.savetxt(dest / f'cov_{target}.txt', direct[key], fmt='%.17e')
    # Target-transfer approximation follows the original manuscript; retain exact matched-z ratio too.
    q = 10**(.2*(matched.sn_lens_m_b_corr.to_numpy(float)-matched.sn_source_m_b_corr.to_numpy(float)))
    Rtarget = 1-q*(1+matched.zs.to_numpy(float))/(1+matched.zl.to_numpy(float))
    assert np.all((Rtarget > .001) & (Rtarget < 1))
    matched['R_target_approximation'] = Rtarget
    Cdm = direct['ll']+direct['ss']-direct['ls']-direct['ls'].T
    jac = -.2*np.log(10)*(1-Rtarget)
    CR = jac[:, None]*Cdm*jac[None, :]
    diag = matched[['Lens name', 'zl', 'zs', 'sn_lens_CID', 'sn_source_CID',
                    'sn_lens_original_index', 'sn_source_original_index']].copy()
    diag['R_target_approximation'] = Rtarget
    diag['R_actual_SN_redshifts'] = matched.R_flat.to_numpy(float)
    diag['relative_R_error'] = np.sqrt(np.diag(CR))/Rtarget
    diag.to_csv(dest / 'distance_input_diagnostics.csv', index=False)
    selected = set(matched['Lens name'])
    excluded = lens[~lens['Lens name'].isin(selected)].copy()
    excluded['exclusion_reason'] = 'Not selected by maximum-cardinality / minimum-cost assignment; not necessarily without candidates'
    excluded.to_csv(dest / 'unselected_lenses.csv', index=False)
    old = ROOT / 'reference_history/initial_submission_matched_catalog.csv'
    comparison = {}
    if old.exists():
        prior = pd.read_csv(old)
        shared = set(prior['Lens name']) & selected
        common = prior.merge(matched, on='Lens name', suffixes=('_old', '_new'))
        # Historical catalogue has magnitudes but may not store physical IDs.
        same = np.isclose(common.sn_lens_m_b_corr_old, common.sn_lens_m_b_corr_new.astype(float), rtol=0, atol=1e-10) & np.isclose(common.sn_source_m_b_corr_old, common.sn_source_m_b_corr_new.astype(float), rtol=0, atol=1e-10)
        comparison = dict(historical_N=len(prior), new_N=n, shared_lenses=len(shared),
                          identical_magnitude_pairs=int(same.sum()),
                          note='Magnitude-pair agreement only; not a substitute for original-ID verification')
    manifest = dict(file_sha256={p.name: sha256(p) for p in dest.iterdir() if p.is_file()},
        source='Fresh raw-input pairing with author CosmoMatcher 1.4.3',
        N=n, raw_input_sha256=audit['input_sha256'], matcher_sha256=sha256(ROOT / 'tools/CosmoMatcherv1.4.3.py'),
        configuration_sha256=sha256(ROOT / 'configs/matching_primary.json'),
        relative_R_error_gt020=int((diag.relative_R_error > .2).sum()),
        baseline_R_convention='Target redshifts; explicit finite-matching approximation',
        matched_SN_ratio_convention='Actual counterpart zHD; exported by supplied matcher',
        old_comparison=comparison)
    write_json(dest / 'manifest.json', manifest)
    write_json(summary_path, dict(summary=summary, primary_manifest=manifest))
    print('FROZEN NEW CATALOGUE', n, comparison, flush=True)


if __name__ == '__main__':
    main()
