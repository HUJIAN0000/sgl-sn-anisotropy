"""Keep the author's plotting functions/style, supply known hard prior ranges.

The original file stays byte-identical. This adapter only constructs GetDist
MCSamples with the actual box support before any KDE is evaluated. It prevents
smoothing from displaying beta<-1 or other out-of-support tails. Raw posterior
draws, quantiles, smoothing scales, colours, and contours are otherwise unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import plot_new_posteriors as driver
from getdist import MCSamples
from aa_revision.common import ROOT, read_json, write_json

BOUNDS={r'\gamma_0':(1.,3.),r'\gamma_z':(-2.,2.),r'\beta':(-1.,1.),
        r'\gamma_s':(-1.5,1.5),r'f_{\rm int}':(0.,1.)}
def bounded_samples(samples,labels,label=None,smooth_1d=.5,smooth_2d=.7):
    names=driver.cosmo_tools._default_names(len(labels))
    ranges={name:BOUNDS[lab] for name,lab in zip(names,labels)}
    return MCSamples(samples=samples,names=names,labels=list(labels),label=label,
        ranges=ranges,settings={'smooth_scale_1D':smooth_1d,'smooth_scale_2D':smooth_2d})

if __name__=='__main__':
    driver.cosmo_tools._make_mcsamples=bounded_samples
    driver.main()
    p=ROOT/'results/production/figures/figure_manifest.json'
    manifest=read_json(p)
    manifest['density_boundary_adapter']='scripts/plot_bounded_posteriors.py'
    manifest['hard_support']={k:list(v) for k,v in BOUNDS.items()}
    manifest['note']='Original author plotting file unchanged; GetDist constructor supplied actual prior ranges to prevent KDE leakage. Joint dynamical support still follows samples.'
    write_json(p,manifest)
