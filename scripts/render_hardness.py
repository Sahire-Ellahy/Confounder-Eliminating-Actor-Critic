#!/usr/bin/env python3
"""Render failure mechanisms, saved critic updates and primary notebook organized strictly by environment; never trains."""
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

    md('hardness-opening', '''# Confounder-Eliminating Actor-Critic: Undertow & False Beacon Suite (Organized by Environment)

This primary notebook presents all controlled junction interventions, two-critic update reconstructions, held-out Reward & Evidence Critic TD/Bellman diagnostics, and multi-step path-dependent benchmarks **organized strictly by environment family**:

1. **Part I — Undertow Family (`undertow_junction` $\\to$ `undertow_pathdep` $H=3$)**: Dead-end action breadth, exact CEAC two-critic logit updates, held-out critic convergence, and multi-step thermocline path dependence breaking Optimal-Action IDS.
2. **Part II — False Beacon Family (`false_beacon_junction` $\\to$ `false_beacon_pathdep` $H=3$)**: Chamber-breadth entropy traps, Bayesian cue-correlation reversal, held-out critic convergence, and multi-step icebreaker staging breaking Optimal-Action IDS.
3. **Part III — Scope, Actor Optimizer Stability & Reproducibility**.''')

    code('hardness-setup', '''from pathlib import Path
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'ceac_hardness').is_dir())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from IPython import get_ipython
_ip = get_ipython()
if _ip is not None: _ip.run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from ceac_hardness.report import load_export, draw_junctions
from ceac_walkthrough.mechanisms import load_mechanisms
from ceac_walkthrough.failure_report import (
    draw_critic_mechanism,
    draw_false_beacon_prior_reversal,
    draw_single_junction_burden,
)
from ceac_critic_diagnostics import draw_suite_critic_traces, load_suite_critic_audit
from ceac_critic_diagnostics.reconstruct_earlier import BURDEN_METHODS, PRIOR_METHODS
from ceac_pathdep_grids.study import (
    AUDIT_METHODS as PATHDEP_AUDIT_METHODS,
    GRID_PATHDEP_SHORT,
    draw_single_grid_pathdep_scaling,
    draw_single_grid_pathdep_traces,
    load_grid_pathdep_export,
)
burden = load_export(ROOT / 'notebooks/evidence/hardness_burden')
prior = load_export(ROOT / 'notebooks/evidence/hardness_prior')
audit = load_mechanisms(ROOT / 'notebooks/evidence/mechanisms')
eb_dir = ROOT / 'notebooks/evidence/critic_bellman_earlier'
b_aud = load_suite_critic_audit(eb_dir, 'junction_burden_critic_audit.json.gz')
p_aud = load_suite_critic_audit(eb_dir, 'junction_prior_critic_audit.json.gz')
gp_dir = ROOT / 'notebooks/evidence/pathdep_undertow_falsebeacon'
gp_exp = load_grid_pathdep_export(gp_dir)
gp_crit = load_suite_critic_audit(gp_dir, 'critic_bellman_audit.json.gz')
print(f"Verified {len(burden['runs'])} burden cells, {len(prior['runs'])} prior cells, {len(audit['cells'])} update audits, and {len(gp_exp['rows'])} path-dependent cells. No training launched.")''')

    code('hardness-junction-design','fig = draw_junctions()\nplt.show()')

    md('part1-undertow-md', '''---
## Part I — Undertow Family (`undertow_junction` & `undertow_pathdep`)

### 1.1 Controlled Undertow Nuisance-Breadth Junction ($B \\in \\{0, 4, 16, 64\\}$), CEAC Critic Updates & Held-Out Bellman Error

''' + narrative['burden_findings'] + '\n\n' + narrative['critic_findings'])

    code('part1-undertow-junction', '''fig = draw_single_junction_burden(burden, 'undertow_junction')
plt.show()
fig = draw_critic_mechanism(audit)
plt.show()
fig = draw_suite_critic_traces(
    b_aud['cells'],
    [('undertow_junction', 'Undertow Junction (B = 64)')],
    target_dose=64.0,
    suptitle='Controlled Undertow Junction (B = 64, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error',
    method_order=BURDEN_METHODS,
)
plt.show()''')

    md('part1-undertow-pathdep-md', '''### 1.2 Multi-Step Path-Dependent Undertow DeepSea (`undertow_pathdep`, $H = 3$) — Breaking Optimal-Action IDS

In the two-step junction above, well-specified Bayesian IDS (`ids_product`) succeeds because a single task bit fixes the optimal action without state-contingent path dependence. Extending Undertow to a three-step **Thermocline Lock** ($k \\in \\{0, 1, 2, 3\\}$, `32,768` episodes; [full tables](evidence/pathdep_undertow_falsebeacon/assessment.md)) — where diving through the thermocline ($-0.015$, $0$ immediate KL) unlocks $R = 8$ deep vents amidst $6k$ shallow echo plumes and $(3^k - 1)$ off-path eddies — breaks **both** Russo–Van Roy policy-tree IDS (`ids_action_tree`, $I(\\Pi^\\star; Z)$) and quotiented on-path IDS (`ids_action_onpath`, $I(C_{\\text{on-path}}^\\star; Z)$), while periwinkle `ceac_ranked` remains flat near minimal regret with converging held-out Evidence Critic RMSE.''')

    code('part1-undertow-pathdep-plots', '''fig = draw_single_grid_pathdep_scaling(gp_exp['rows'], 'undertow_pathdep')
plt.show()
fig = draw_single_grid_pathdep_traces(gp_exp['rows'], 'undertow_pathdep', target_dose=3)
plt.show()
fig = draw_suite_critic_traces(
    gp_crit['cells'],
    [('undertow_pathdep', GRID_PATHDEP_SHORT['undertow_pathdep'])],
    target_dose=3,
    suptitle='Path-Dependent Undertow DeepSea (k = 3, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error',
    method_order=PATHDEP_AUDIT_METHODS,
    ranked_ceac_present=True,
)
plt.show()''')

    md('part2-falsebeacon-md', '''---
## Part II — False Beacon Family (`false_beacon_junction` & `false_beacon_pathdep`)

### 2.1 Controlled False Beacon Chamber-Breadth ($B \\in \\{0, 4, 16, 64\\}$) & Cue-Reversal ($d \\in [0, 1]$) Junctions

''' + narrative['prior_setup'] + '\n\n' + narrative['prior_findings'])

    code('part2-falsebeacon-junction', '''fig = draw_single_junction_burden(burden, 'false_beacon_junction')
plt.show()
fig = draw_false_beacon_prior_reversal(prior)
plt.show()
fig = draw_suite_critic_traces(
    b_aud['cells'],
    [('false_beacon_junction', 'False Beacon Junction (B = 64)')],
    target_dose=64.0,
    suptitle='Controlled False Beacon Junction (B = 64, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error',
    method_order=BURDEN_METHODS,
)
plt.show()
fig = draw_suite_critic_traces(
    p_aud['cells'],
    [('false_beacon_junction', 'False Beacon Cue-Reversal (d = 1.0)')],
    target_dose=1.0,
    suptitle='Controlled False Beacon Cue-Reversal (d = 1.0, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error',
    method_order=PRIOR_METHODS,
)
plt.show()''')

    md('part2-falsebeacon-pathdep-md', '''### 2.2 Multi-Step Path-Dependent False Beacon Shore (`false_beacon_pathdep`, $H = 3$) — Breaking Optimal-Action IDS

Extending False Beacon to a three-step **Icebreaker Staging** problem ($k \\in \\{0, 1, 2, 3\\}$, `32,768` episodes) — where staging onto `outer_soundings` ($-0.015$, $0$ immediate KL) unlocks $R = 8$ outer causeways amidst $6k$ shoreline false-beacon glints and $(3^k - 1)$ reed slush corridors — breaks `ids_product`, Russo–Van Roy `ids_action_tree` ($I(\\Pi^\\star; Z)$), **and** quotiented `ids_action_onpath` ($I(C_{\\text{on-path}}^\\star; Z)$), while periwinkle `ceac_ranked` stays flat near `~40` regret with `0.0000` Held-Out Evidence Critic RMSE.''')

    code('part2-falsebeacon-pathdep-plots', '''fig = draw_single_grid_pathdep_scaling(gp_exp['rows'], 'false_beacon_pathdep')
plt.show()
fig = draw_single_grid_pathdep_traces(gp_exp['rows'], 'false_beacon_pathdep', target_dose=3)
plt.show()
fig = draw_suite_critic_traces(
    gp_crit['cells'],
    [('false_beacon_pathdep', GRID_PATHDEP_SHORT['false_beacon_pathdep'])],
    target_dose=3,
    suptitle='Path-Dependent False Beacon Shore (k = 3, Task 1) · Overlaid Held-Out Reward & Evidence Critic Error',
    method_order=PATHDEP_AUDIT_METHODS,
    ranked_ceac_present=True,
)
plt.show()''')

    md('hardness-ceac-limits', '---\n## Part III — Scope, Optimization Limits & Actor Stability\n\n' + narrative['limits'])
    if stability:
        md('stability-assessment',notebook_narrative(stability))
        code('stability-learning', '''from ceac_stability.report import load_export as load_stability, combined_confirmation, draw_confirmation_learning
stability = load_stability(ROOT / 'notebooks/evidence/actor_stability')
fig = draw_confirmation_learning(combined_confirmation(stability))
plt.show()''')
    md('hardness-details', '''## Evidence and reproduction

The [curated report](evidence/mechanisms/report.md) and [PDF](evidence/mechanisms/report.pdf) follow this same evidence base. Companion notebooks: [Undertow & False Beacon Complete Diagnostics](Undertow_FalseBeacon_Diagnostics.ipynb) (including the full $8\\times 8$ DeepSea and Frozen Lake shore grids) and [Pathological Search Suite](Pathological_Search_Suite.ipynb) (sequential locks, signalling pathways, and K562 genetic interaction screens).''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.11'},'hardness_report_schema':1,
        'mechanism_report_schema':1,'actor_stability_schema':2 if stability else 0})
    nb_path=ROOT/'notebooks/Undertow_FalseBeacon.ipynb'
    nbf.write(nb,nb_path)
    if args.execute:
        import base64,hashlib,importlib.metadata,io,json,os,platform
        from contextlib import redirect_stdout
        os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib')
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        notebook=nbf.read(nb_path,as_version=4)
        nbf.validate(notebook)
        exec_ns={'__name__':'__main__'}
        exec_count=0
        for cell in notebook.cells:
            if cell.cell_type!='code':continue
            exec_count+=1;cell.execution_count=exec_count;cell.outputs=[]
            pending=[]
            orig_show=plt.show
            def _cap(*_a,**_kw):
                for num in list(plt.get_fignums()):
                    fig=plt.figure(num);buf=io.BytesIO()
                    fig.savefig(buf,format='png',dpi=160,bbox_inches='tight')
                    pending.append(base64.b64encode(buf.getvalue()).decode('ascii'))
                    plt.close(fig)
            plt.show=_cap
            sbuf=io.StringIO()
            try:
                with redirect_stdout(sbuf):
                    exec(compile(cell.source,f'<{cell.id}>','exec'),exec_ns)
            finally:
                plt.show=orig_show;plt.close('all')
            if sbuf.getvalue():
                cell.outputs.append(nbf.v4.new_output('stream',name='stdout',text=sbuf.getvalue()))
            for b64 in pending:
                cell.outputs.append(nbf.v4.new_output('display_data',data={'image/png':b64,'text/plain':'<Figure>'},metadata={}))
        nbf.validate(notebook)
        for cell in notebook.cells:cell.metadata.pop('execution',None)
        nbf.write(notebook,nb_path)
        jcell=next(c for c in notebook.cells if c.id=='hardness-junction-design')
        jpng=next(o['data']['image/png'] for o in jcell.outputs if 'image/png' in o.get('data',{}))
        jasset=ROOT/'notebooks/assets/junctions.png'
        jasset.write_bytes(base64.b64decode(jpng))
        code_cells=[c for c in notebook.cells if c.cell_type=='code']
        images=sum('image/png' in o.get('data',{}) for c in code_cells for o in c.outputs)
        names=['hardness_burden','hardness_prior','mechanisms','critic_bellman_earlier','pathdep_undertow_falsebeacon']
        audit_rec=dict(
            schema=1,status='complete',code_cells=len(code_cells),rendered_figures=images,
            notebook_sha256=hashlib.sha256(nb_path.read_bytes()).hexdigest(),
            evidence_manifest_sha256=hashlib.sha256((ROOT/'notebooks/evidence/manifest.json').read_bytes()).hexdigest(),
            python=platform.python_version(),
            packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','matplotlib','nbclient','ipykernel')},
            training_launched=False,
            explanatory_map_hashes={'assets/junctions.png':hashlib.sha256(jasset.read_bytes()).hexdigest()},
            hardness_evidence_manifests={n:hashlib.sha256((ROOT/'notebooks/evidence'/n/'manifest.json').read_bytes()).hexdigest() for n in names},
        )
        if stability:
            audit_rec['actor_stability_manifest_sha256']=hashlib.sha256((ROOT/'notebooks/evidence/actor_stability/manifest.json').read_bytes()).hexdigest()
        (ROOT/'notebooks/execution.json').write_text(json.dumps(audit_rec,indent=2,sort_keys=True)+'\n')
        print(f'Executed Undertow_FalseBeacon.ipynb in-process: {len(code_cells)} code cells, {images} rendered figures.')
    print('Updated mechanism report, PDF and environment-organized primary notebook from verified evidence.')

if __name__=='__main__':main()
