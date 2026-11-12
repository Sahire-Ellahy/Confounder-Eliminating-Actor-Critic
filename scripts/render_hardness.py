#!/usr/bin/env python3
"""Render failure mechanisms, saved critic updates and notebook; never trains."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nbformat as nbf
from ceac_hardness.report import export,load_export,load_study
from ceac_walkthrough.mechanisms import export_reconstruction
from ceac_walkthrough.failure_report import render_curated


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--burden',type=Path,required=True)
    parser.add_argument('--prior',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--stability',type=Path,help='Completed actor-stability study to verify/export; otherwise reuse its existing complete compact export')
    parser.add_argument('--workers',type=int,default=4,help='Workers for saved-update audit, never training')
    args=parser.parse_args()
    evidence=ROOT/'notebooks/evidence'
    stability=None
    from ceac_stability.report import render as render_stability,load_export as load_stability,notebook_narrative
    if args.stability:
        render_stability(args.stability,evidence/'actor_stability')
    if (evidence/'actor_stability/manifest.json').exists():
        candidate=load_stability(evidence/'actor_stability')
        if candidate.get('confirmation') and candidate.get('secondary'):
            stability=candidate
        elif args.stability:
            raise ValueError('Finish both registered confirmation studies before adding their comparison to the primary notebook')
    for path in (args.burden,args.prior):
        bundle=load_study(path)
        methods={r['method'] for r in bundle['runs']}
        if 'ceac' not in methods or methods & {'ceac_exact','ceac_actor'}:
            raise ValueError('Primary notebook requires replay-driven CEAC; historical references are separate')
    export(args.burden,evidence/'hardness_burden')
    export(args.prior,evidence/'hardness_prior')
    burden=load_export(evidence/'hardness_burden');prior=load_export(evidence/'hardness_prior')
    audit=export_reconstruction(args.burden,args.prior,evidence/'mechanisms',args.workers)
    narrative=render_curated(burden,prior,audit,evidence/'mechanisms',stability=stability)
    cells=[]
    def md(identity,text):
        cell=nbf.v4.new_markdown_cell(text);cell.id=identity;cells.append(cell)
    def code(identity,text):
        compile(text,'<notebook>','exec')
        cell=nbf.v4.new_code_cell(text);cell.id=identity;cells.append(cell)
    md('hardness-opening',narrative['opening'])
    code('hardness-setup', '''from pathlib import Path
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'ceac_hardness').is_dir())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from ceac_hardness.report import load_export, draw_junctions
from ceac_walkthrough.mechanisms import load_mechanisms
from ceac_walkthrough.failure_report import draw_failure_fans, draw_critic_mechanism
burden = load_export(ROOT / 'notebooks/evidence/hardness_burden')
prior = load_export(ROOT / 'notebooks/evidence/hardness_prior')
audit = load_mechanisms(ROOT / 'notebooks/evidence/mechanisms')
print(f"Verified {len(burden['runs'])} burden cells, {len(prior['runs'])} prior cells, and {len(audit['cells'])} saved-update audits. No training launched.")''')
    md('hardness-burden-setup',narrative['burden_setup'])
    code('hardness-junction-design','fig = draw_junctions()\nplt.show()')
    code('hardness-burden-fans','fig = draw_failure_fans(burden)\nplt.show()')
    md('hardness-burden-findings',narrative['burden_findings']+'\n\n[All six algorithms, every dose and seed](evidence/hardness_burden/assessment.md).')
    md('hardness-prior-setup',narrative['prior_setup'])
    code('hardness-prior-fans','fig = draw_failure_fans(prior)\nplt.show()')
    md('hardness-prior-findings',narrative['prior_findings']+'\n\n[All endpoints and wrong-task posterior diagnostics](evidence/hardness_prior/assessment.md).')
    md('hardness-critic-setup',narrative['critic_setup'])
    code('hardness-critic-mechanism','fig = draw_critic_mechanism(audit)\nplt.show()')
    md('hardness-critic-findings',narrative['critic_findings'])
    md('hardness-ceac-limits',narrative['limits'])
    if stability:
        md('stability-assessment',notebook_narrative(stability))
        code('stability-learning', '''from ceac_stability.report import load_export as load_stability, combined_confirmation, draw_confirmation_learning
stability = load_stability(ROOT / 'notebooks/evidence/actor_stability')
fig = draw_confirmation_learning(combined_confirmation(stability))
plt.show()''')
    md('hardness-details', '''## Evidence and reproduction

The [curated report](evidence/mechanisms/report.md) and [PDF](evidence/mechanisms/report.pdf) follow this same narrative. [Supporting mechanism checks](evidence/mechanisms/objective_diagnostics.png) show the population IDAC objective, measured POMD saturation and the correlated prior's cue evidence.

The [implementation guide](../docs/hardness_implementation.md), [comparator source ledger](../docs/comparator_provenance.md) and [equation-level CEAC audit](../audits/ceac_pseudocode_fidelity_20261111.md) specify the algorithms. The [historical diagnostic notebook](Undertow_FalseBeacon_Diagnostics.ipynb) retains the earlier grid studies; its exact-deployment reference is not primary CEAC.

The report is rebuilt by `scripts/render_hardness.py`. The notebook reads only compact, hash-verified exports; it never needs raw trajectories or launches training. `scripts/export_critic_updates.py` can rebuild the update audit from the immutable saved episodes and design histories.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.11'},'hardness_report_schema':1,
        'mechanism_report_schema':1,'actor_stability_schema':2 if stability else 0})
    nbf.write(nb,ROOT/'notebooks/Undertow_FalseBeacon.ipynb')
    if args.execute:
        subprocess.run([sys.executable,str(ROOT/'scripts/verify_notebook.py'),'--write'],cwd=ROOT,check=True)
    print('Updated mechanism report, PDF and primary notebook from complete verified evidence.')

if __name__=='__main__':main()
