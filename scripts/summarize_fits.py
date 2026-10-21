"""Tables from new qualified MLEs, with no hard-coded historical results."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aa_revision.common import ROOT, read_json, write_json
from aa_revision.revision import config, load_primary
from aa_revision.model import Spec


def main():
    cfg = config()
    data = load_primary()
    zp = float(np.median(data.table.zl))
    rows = []
    for case in cfg['cases']:
        path = ROOT / 'results/production/mle' / case['id']
        validation = read_json(path / 'validation.json')
        assert validation['passed'], case['id']
        free = read_json(path / 'flat.json')['result']
        for prior in ['flat', 'fixed0', 'fixed018']:
            fit = read_json(path / f'{prior}.json')['result']
            spec = Spec(case.get('framework', 'P2'), prior, cfg['domain'], case['tracer'])
            p = spec.unpack(fit['theta'])
            gp = p['gamma0']+zp*p['gamma_z']+p['gamma_s']*float(np.median(data.predictor))
            rows.append(dict(case=case['id'], framework=spec.framework, prior=prior, N=data.n,
                lnL=fit['lnL'], BIC=fit['BIC'], delta_BIC=fit['BIC']-free['BIC'],
                gamma_p=gp, gamma0=p['gamma0'], gamma_z=p['gamma_z'], beta=p['beta'],
                f_int=p['f_int'], gamma_s=p['gamma_s'], optimizer_passed=fit['optimizer_agreement_checked']))
    out = ROOT / 'results/production/report'
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'new_mle_comparisons.csv', index=False)
    df.to_latex(out / 'new_mle_comparisons.tex', index=False, float_format='%.4f')
    write_json(out / 'mle_report_manifest.json', dict(N=data.n, pivot_redshift=zp,
        data_digest=data.digest, qualified=True, rows=len(rows), no_historical_results_imported=True))
    print(df[['case', 'prior', 'lnL', 'delta_BIC', 'beta']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
