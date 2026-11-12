#!/usr/bin/env python3
"""Audit the old hardness claim and a proposed support-shift channel; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ceac_walkthrough.evidence import verified_bundle
from ceac_walkthrough.hardness import summarize_current_sweep, support_shift_design, assessment_markdown, draw_support_design

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'audits/hardness_review_20261111')
    parser.add_argument('--pdf',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    bundle,manifest=verified_bundle(ROOT/'notebooks')
    review=dict(schema=1,training_launched=False,
        evidence_manifest_sha256=hashlib.sha256((ROOT/'notebooks/evidence/manifest.json').read_bytes()).hexdigest(),
        source_studies=manifest['source_studies'],late_gap_tolerance=.01,
        current_sweep=[r for mode in (0,1) for r in summarize_current_sweep(bundle,mode)],
        proposed_support_shift=support_shift_design())
    (args.output/'assessment.json').write_text(json.dumps(review,indent=2,allow_nan=False)+'\n')
    path=args.output/'assessment.md';path.write_text(assessment_markdown(bundle))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=draw_support_design(plt)
    for ext in ('png','pdf'):fig.savefig(args.output/f'proposed_support_shift.{ext}',dpi=150,bbox_inches='tight')
    plt.close(fig)
    if args.pdf:
        subprocess.run(['pandoc',str(path),'-o',str(args.output/'assessment.pdf'),
            '--from=markdown+tex_math_dollars','--pdf-engine=pdflatex','-V','geometry:margin=0.65in','-V','fontsize=10pt'],check=True)
    print(path)
if __name__=='__main__':main()
