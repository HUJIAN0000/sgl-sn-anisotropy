"""Tolerance and deterministic tie-order checks using the unmodified supplied matcher."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
import sys
from copy import deepcopy
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aa_revision.common import ROOT, read_json, write_json, sha256
from aa_revision.revision import config, engine_for, cached_mle, validate_case
from aa_revision.model import Spec, load_data
from rebuild_inputs import matcher_module


def main():
    tool = matcher_module()
    cfg = config()
    basecfg = read_json(ROOT / 'configs/matching_primary.json')
    primary = pd.read_csv(ROOT / 'data/primary/matched_catalog.csv')
    raw = pd.read_csv(ROOT / 'data/raw/Pantheon+SH0ES.dat', sep=r'\s+', dtype=str)
    cov = tool.load_matrix(ROOT / 'data/raw/Pantheon+SH0ES_STAT+SYS.cov', len(raw))
    cov = (cov+cov.T)/2
    definitions = [dict(id='tolerance003', tolerance=.03),
                   dict(id='tolerance007', tolerance=.07),
                   dict(id='canonical_reverse_SN_order', tolerance=.05, reverse_SN=True)]
    write_json(ROOT / 'protocol/matching_variants.json', dict(scenarios=definitions,
        interpretation='Specified assignment and tolerance sensitivity, not probabilities over all matching optima',
        criterion_comparison='Within each variant only; never compare absolute BIC across different memberships'))
    rows = []
    for definition in definitions:
        vid = definition['id']
        target = ROOT / 'data/matching_variants' / vid
        manifest = target / 'manifest.json'
        if not manifest.exists():
            matchcfg = deepcopy(basecfg)
            matchcfg['params']['tol'] = definition['tolerance']
            matchcfg['params']['output_dir'] = str(ROOT / 'results/matching' / vid)
            if definition.get('reverse_SN'):
                derived = raw.copy()
                derived['original_covariance_index'] = np.arange(len(raw))
                derived = derived.iloc[::-1].reset_index(drop=True)
                path = ROOT / 'data/derived/reversed_SN_full.csv'
                path.parent.mkdir(parents=True, exist_ok=True)
                derived.to_csv(path, index=False)
                matchcfg['file_paths']['tracer'] = str(path)
                matchcfg['col_cfg']['cov_index_col'] = 'original_covariance_index'
            write_json(ROOT / 'configs' / f'matching_{vid}.json', matchcfg)
            result = tool.run_config(matchcfg, log_func=lambda message: print(vid, message, flush=True))
            assert result.summary['solver']['optimal']
            matched = result.matched
            li = matched.sn_lens_original_index.to_numpy(int)
            si = matched.sn_source_original_index.to_numpy(int)
            assert len(set(raw.CID.iloc[np.r_[li, si]])) == 2*len(matched)
            target.mkdir(parents=True, exist_ok=True)
            matched.to_csv(target / 'matched_catalog.csv', index=False)
            result.index_map.to_csv(target / 'original_row_index_map.csv', index=False)
            for name, ix, iy in [('lens', li, li), ('source', si, si), ('cross', li, si)]:
                block = cov[np.ix_(ix, iy)]
                np.testing.assert_allclose(result.matrices[name], block, atol=1e-14, rtol=1e-12)
                np.savetxt(target / f'cov_{name}.txt', block, fmt='%.17e')
            for role, indices in [('sn_lens', li), ('sn_source', si)]:
                np.testing.assert_array_equal(matched[f'{role}_CID'].astype(str), raw.CID.iloc[indices].to_numpy())
                np.testing.assert_allclose(matched[f'{role}_m_b_corr'].to_numpy(float), raw.m_b_corr.iloc[indices].to_numpy(float), rtol=0, atol=0)
            write_json(target / 'matcher_summary.json', result.summary)
            write_json(manifest, dict(N=len(matched), definition=definition,
                file_sha256={p.name: sha256(p) for p in target.iterdir() if p.is_file()},
                source='Direct supplied matcher, new raw inputs, explicit original covariance row indices'))
        data = load_data(target)
        matched = data.table
        base_triplets = set(zip(primary['Lens name'], primary.sn_lens_original_index, primary.sn_source_original_index))
        triplets = set(zip(matched['Lens name'], matched.sn_lens_original_index, matched.sn_source_original_index))
        for cid in ['powerlaw', 'hernquist', 'sersic_n4']:
            case = next(c for c in cfg['cases'] if c['id'] == cid)
            engine = engine_for(data, case)
            output = ROOT / 'results/production/matching_variants' / vid / cid
            fits = {}
            for prior in ['flat', 'fixed0', 'fixed018']:
                spec = Spec('P2', prior, cfg['domain'], case['tracer'])
                fits[prior] = cached_mle(data, spec, cfg['optimizer'], output / f'{prior}.json',
                    engine=engine, extra=dict(variant=definition, case=case), seed=cfg['seed'])
            numerical = validate_case(data, case, fits, engine)
            passed = bool(all(f['optimizer_agreement_checked'] for f in fits.values()) and numerical['passed'] and
                          fits['flat']['lnL'] >= max(fits['fixed0']['lnL'], fits['fixed018']['lnL'])-.01)
            row = dict(variant=vid, case=cid, N=data.n, shared_triplets_with_primary=len(base_triplets & triplets),
                delta_BIC_0=fits['fixed0']['BIC']-fits['flat']['BIC'],
                delta_BIC_018=fits['fixed018']['BIC']-fits['flat']['BIC'], passed=passed)
            write_json(output / 'validation.json', dict(summary=row, numerical=numerical))
            rows.append(row)
            print('VARIANT FIT', row, flush=True)
            pd.DataFrame(rows).to_csv(ROOT / 'results/production/report/new_matching_variant_comparisons.csv', index=False)
    write_json(ROOT / 'results/production/matching_variants_status.json', rows)
    if not all(r['passed'] for r in rows):
        raise SystemExit('Some matching variant fits need attention')


if __name__ == '__main__':
    main()
