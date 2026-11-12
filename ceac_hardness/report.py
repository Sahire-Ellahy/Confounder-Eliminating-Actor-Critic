"""Verified small-multiple reports and portable evidence; no training."""
import argparse,csv,gzip,json,math,hashlib
from collections import defaultdict
from dataclasses import asdict
from .environment import JunctionConfig
from pathlib import Path
import numpy as np
from .run import ROOT,atomic_json,digest

LABELS={'idac':'IDAC · occupancy ratio','ucb_q':'UCB Q-learning','pomd':'Optimistic POMD',
    'ids_correlated':'IDS · frozen correlated prior','ids_product':'IDS · product-support prior',
    'ceac':'CEAC · reward + evidence critics','ceac_exact':'Exact-model design reference','ceac_actor':'Archived modified-loop trial','oracle':'Oracle Graves–Lai'}
KINDS={'undertow_junction':'Undertow junction','false_beacon_junction':'False Beacon junction'}
COLORS=['#1765ab','#45976a','#cc7817','#ad2358','#5c2185']


def load_study(folder):
    folder=Path(folder);status=json.loads((folder/'status.json').read_text())
    if status.get('status')!='complete':raise ValueError(f'Incomplete study: {folder}')
    registration=json.loads((folder/'registration.json').read_text())
    for name in ('registration','results'):
        if digest(folder/(name+'.json'))!=status[name+'_sha256']:raise ValueError('Study index hash mismatch')
    index=json.loads((folder/'results.json').read_text());runs=[];seen=set()
    if len(index)!=registration['jobs'] or len(index)!=status['completed']:raise ValueError('Missing study cells')
    for entry in index:
        cell=ROOT/entry['folder'];state=json.loads((cell/'status.json').read_text())
        if state['status']!='complete':raise ValueError('Incomplete cell')
        for name,checksum in state['artifacts'].items():
            if digest(cell/name)!=checksum:raise ValueError(f'Artifact mismatch: {cell/name}')
        manifest=json.loads((cell/'manifest.json').read_text());summary=json.loads((cell/'summary.json').read_text())
        expected_sources=registration.get('sources_by_method',{}).get(summary['settings']['method'],registration['sources'])
        if manifest['sources']!=expected_sources or manifest['identity']!=summary['identity']:
            raise ValueError('Mixed numerical sources or settings')
        settings=summary['settings'];env=settings['environment'];protocol=env['protocol']
        dose=env['burden'] if protocol=='burden' else settings['dose']
        key=(env['kind'],dose,settings['task_evaluator_only'],settings['method'],settings['seed'])
        if key in seen:raise ValueError('Duplicate cell')
        seen.add(key)
        if summary['late_gap']!=entry['late_gap'] or summary['cumulative_regret']!=entry['cumulative_regret']:
            raise ValueError('Index and cell disagree')
        with gzip.open(cell/'trace.jsonl.gz','rt') as f:trace=[json.loads(line) for line in f]
        if trace[-1]['episode']!=settings['episodes'] or trace[-1]['interactions']!=settings['physical_budget']:
            raise ValueError('Incomplete trace budget')
        runs.append(dict(protocol=protocol,kind=env['kind'],dose=dose,task=settings['task_evaluator_only'],
            method=settings['method'],seed=settings['seed'],summary=summary,trace=trace))
    cfg=registration['configuration'];protocol=registration['protocol'];spec=cfg[protocol]
    excluded={r['method'] for r in registration.get('excluded_from_sweep',[])}
    kinds=spec['kinds'] if protocol=='burden' else ['false_beacon_junction']
    expected={(k,d,t,m,seed) for k in kinds for d in spec['doses'] for t in spec['tasks']
        for m in spec['methods'] if m not in excluded for seed in cfg['evaluation_seeds']}
    if seen!=expected:raise ValueError('Registered condition grid is incomplete or substituted')
    for run in runs:
        settings=run['summary']['settings']
        want=json.loads(json.dumps(asdict(JunctionConfig(kind=run['kind'],protocol=protocol,
            burden=int(run['dose']) if protocol=='burden' else 0))))
        if settings['environment']!=want or settings['episodes']!=registration['episodes']:
            raise ValueError('Changed intervention or budget')
    return dict(registration=registration,runs=runs,source_folder=str(folder))


def endpoints(bundle):
    groups=defaultdict(list)
    for run in bundle['runs']:groups[(run['protocol'],run['kind'],run['task'],run['method'],run['dose'])].append(run)
    rows=[]
    for (protocol,kind,task,method,dose),runs in sorted(groups.items()):
        runs=sorted(runs,key=lambda r:r['seed'])
        values=np.array([r['summary']['late_gap'] for r in runs]);regret=np.array([r['summary']['cumulative_regret'] for r in runs])
        row=dict(protocol=protocol,kind=kind,task=task,method=method,dose=dose,seeds=len(runs),
            late_gap_mean=float(values.mean()),late_gap_sd=float(values.std(ddof=1)) if len(values)>1 else 0.,
            regret_mean=float(regret.mean()),regret_sd=float(regret.std(ddof=1)) if len(values)>1 else 0.,
            nuisance_episode_fraction=float(np.mean([r['summary']['events']['nuisance']/r['summary']['settings']['episodes'] for r in runs])),
            probe_episode_fraction=float(np.mean([r['summary']['events']['probe']/r['summary']['settings']['episodes'] for r in runs])),
            pomd_zero_fraction=float(np.mean([r['summary'].get('pomd_zero_fraction',0.) for r in runs])),
            ceac_actor_updates=sum(r['summary']['actor_updates'] for r in runs),
            ceac_priced_actor_updates=sum(r['summary']['priced_actor_updates'] for r in runs),
            seed_ids=[r['seed'] for r in runs],seed_late_gaps=values.tolist())
        if method.startswith('ids_'):
            row['wrong_task_posterior']=float(np.mean([r['summary']['task1_posterior'] if task==0 else 1-r['summary']['task1_posterior'] for r in runs]))
            row['true_joint_prior_supported']=all(r['summary']['prior_contains_true_joint_model'] for r in runs)
        rows.append(row)
    return rows


def draw_learning(bundle,kind,task=1,metric='window_gap'):
    import matplotlib.pyplot as plt
    runs=[r for r in bundle['runs'] if r['kind']==kind and r['task']==task]
    order=['idac','ucb_q','pomd','ids_correlated','ids_product','ceac','ceac_exact','ceac_actor','oracle']
    methods=[m for m in order if any(r['method']==m for r in runs)];doses=sorted({r['dose'] for r in runs})
    cols=3;rows=math.ceil(len(methods)/cols)
    fig,axes=plt.subplots(rows,cols,figsize=(12.6,3.15*rows),squeeze=False,sharex=True,sharey=True)
    stats=endpoints(bundle)
    for ax,method in zip(axes.flat,methods):
        for dose,color in zip(doses,COLORS):
            cells=sorted([r for r in runs if r['method']==method and r['dose']==dose],key=lambda r:r['seed'])
            xs=[np.array([t['interactions'] for t in r['trace']])/1000 for r in cells]
            if any(not np.array_equal(x,xs[0]) for x in xs):raise ValueError('Unequal trace grids')
            values=np.array([[t[metric] for t in r['trace']] for r in cells]);mean=values.mean(axis=0);sd=values.std(axis=0,ddof=1)
            ax.fill_between(xs[0],np.maximum(0,mean-sd),mean+sd,color=color,alpha=.12,linewidth=0)
            ax.plot(xs[0],mean,color=color,lw=1.6,label=f'{"B" if runs[0]["protocol"]=="burden" else "d"} = {dose:g}',zorder=3)
        pair=[next(r for r in stats if r['kind']==kind and r['task']==task and r['method']==method and r['dose']==d)['late_gap_mean'] for d in (doses[0],doses[-1])]
        if metric=='window_gap':ax.text(.98,.96,f'Final: {pair[0]:.3f} → {pair[1]:.3f}',ha='right',va='top',transform=ax.transAxes,fontsize=8.5,color='#333333')
        ax.set_title(LABELS[method],fontsize=10.5,pad=9);ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(0,xs[0][-1]);ax.tick_params(labelsize=9)
        if metric=='window_gap':ax.set_ylim(-.025,.98 if runs[0]['protocol']=='burden' else .53)
    for ax in list(axes.flat)[len(methods):]:ax.set_visible(False)
    fig.supxlabel('Physical training interactions (thousands)',fontsize=11)
    fig.supylabel('Expected policy gap' if metric=='window_gap' else 'Cumulative expected regret',fontsize=11)
    title=(KINDS[kind]+' · irrelevant physical opportunities' if runs[0]['protocol']=='burden' else 'Frozen Bayesian assumptions · environmental cue shift')
    fig.suptitle(title+f' · task {task}',fontsize=14,y=1.02)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=len(doses),frameon=False,bbox_to_anchor=(.5,.99),fontsize=10)
    fig.tight_layout(rect=(.025,.035,1,.92))
    return fig



METHOD_COLORS={'idac':'#C25300','ucb_q':'#0072B2','pomd':'#B34697','ids_correlated':'#C93441',
    'ids_product':'#707070','ceac':'#007D60','oracle':'#3E3163','ceac_exact':'#6C9E87','ceac_actor':'#BB7777'}
METHOD_MARKERS={'idac':'s','ucb_q':'^','pomd':'D','ids_correlated':'s','ids_product':'o','ceac':'*','oracle':'x'}

def draw_response(bundle):
    """Pathology on x, same algorithm across doses; seed is the replicate."""
    import matplotlib.pyplot as plt
    rows=endpoints(bundle);protocol=bundle['registration']['protocol']
    panels=[('undertow_junction',1),('false_beacon_junction',1)] if protocol=='burden' else [('false_beacon_junction',1),('false_beacon_junction',0)]
    order=['idac','ucb_q','pomd','ids_correlated','ids_product','ceac','oracle']
    fig,axes=plt.subplots(1,2,figsize=(11.8,3.8),sharey=True)
    for ax,(kind,task) in zip(axes,panels):
        for method in order:
            group=sorted([r for r in rows if r['kind']==kind and r['task']==task and r['method']==method],key=lambda r:r['dose'])
            if not group:continue
            x=np.array([r['dose'] for r in group]);y=np.array([r['late_gap_mean'] for r in group]);sd=np.array([r['late_gap_sd'] for r in group])
            color=METHOD_COLORS[method]
            ax.fill_between(x,np.maximum(0,y-sd),y+sd,color=color,alpha=.10,linewidth=0)
            ax.plot(x,y,label=LABELS[method],color=color,lw=2 if method=='ceac' else 1.5,
                marker=METHOD_MARKERS[method],markersize=9 if method=='ceac' else 6,
                markevery=({'ceac':[0,len(x)-1],'ids_product':[1],'oracle':[2]}.get(method,1) if protocol=='burden' else 1),
                markerfacecolor='white' if method=='ids_product' else color,
                linestyle=('--' if method=='ids_product' else ':' if method=='oracle' else '-'),
                zorder={'ceac':5,'ids_product':6,'oracle':7}.get(method,3))
        ax.set_title(KINDS[kind] if protocol=='burden' else f'Task {task} · '+('risky route optimal' if task==1 else 'safe route optimal'))
        ax.set_xlabel('Irrelevant actions B' if kind=='undertow_junction' else ('Beacon breadth B' if protocol=='burden' else 'Cue reversal d'))
        ax.set_xticks([0,4,16,64] if protocol=='burden' else [0,.25,.5,.75,1])
        ax.set_ylim(-.025,.98 if protocol=='burden' else .53)
        ax.grid(alpha=.18);ax.spines[['right','top']].set_visible(False)
    if protocol=='burden':
        fig.suptitle('CEAC, product-support IDS and oracle overlap near zero; distinct markers identify each.',fontsize=9,y=1.03)
    axes[0].set_ylabel('Final expected policy loss ↓')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,-.13),ncol=3,frameon=False,fontsize=9)
    fig.tight_layout();return fig


def draw_junctions():
    """Show physical state/action changes; every depicted destination is step 1."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig,axes=plt.subplots(1,2,figsize=(12.6,3.4))
    for ax,kind in zip(axes,('undertow_junction','false_beacon_junction')):
        ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(KINDS[kind],fontsize=13,pad=9)
        ax.add_patch(Circle((.12,.55),.045,color='#173e59'));ax.text(.12,.46,'start',ha='center',fontsize=10)
        ax.annotate('',xy=(.72,.77),xytext=(.16,.57),arrowprops=dict(arrowstyle='->',color='#27825c',lw=2))
        ax.add_patch(Circle((.77,.78),.045,color='#27825c'))
        ax.text(.46,.77,'3 useful actions',ha='center',fontsize=10,color='#17623f',rotation=14)
        ax.text(.77,.91,'shared waypoint',ha='center',fontsize=10)
        xs=[.46,.60,.74,.88]
        if kind=='false_beacon_junction':
            ax.plot([.16,.32],[.53,.33],color='#b75b26',lw=2)
            ax.text(.25,.42,'one action',fontsize=9,rotation=-45,ha='center')
        for x in xs:
            origin=(.16,.52) if kind=='undertow_junction' else (.32,.33)
            ax.annotate('',xy=(x,.24),xytext=origin,arrowprops=dict(arrowstyle='->',color='#b75b26',lw=1.3,alpha=.8))
            ax.add_patch(Circle((x,.20),.034,color='#e3a179'))
        ax.text(.68,.06,'B dead-end actions' if kind=='undertow_junction' else 'B reachable cells · uniform draw',ha='center',fontsize=10)
        ax.text(.64,.39,'zero reward · zero task information',ha='center',fontsize=9,color='#8a411c')
    fig.text(.5,-.01,'Useful payoffs and diagnostic signal stay fixed. Every waypoint has one forced finish step; each episode costs two physical steps.',ha='center',fontsize=10)
    fig.tight_layout();return fig

def summary_text(bundle,rows):
    protocol=bundle['runs'][0]['protocol'];cfg=bundle['registration']['configuration']
    text=['# '+('Physical exploration burden' if protocol=='burden' else 'Environmental violation of a fixed Bayesian prior'),'',
        f"Verified {len(bundle['runs'])} completed cells; {len(cfg['evaluation_seeds'])} held-out seeds per cell; "+
        f"{bundle['registration']['episodes']:,} two-step episodes. No training is launched by report rendering.",'',
        'Each panel follows one algorithm as dose changes. Lines are seed means; shading is one sample standard deviation across seeds, not a confidence interval. Coincident lines indicate similar measured behavior. Lower policy gap is better. Endpoint gap averages the final 256 episodes.','',
        'These are new short junction variants. They do not establish the same result on the historical DeepSea/Frozen Lake grids. Task 1 is the burden study’s primary scope; both tasks are retained in the prior-shift study. The two burden variants share the same zero-burden kernel.','']
    text+=['## Dose response','', '![Each algorithm as environmental pathology increases](dose_response.png)', '', '## Within-algorithm learning traces','']
    for kind in sorted({r['kind'] for r in rows}):
        text+=[f"![{KINDS[kind]}: each panel follows one algorithm as dose changes]({kind}_task1_window_gap.png)", '']
    if protocol=='prior_shift':text+=['[Task 0 traces](false_beacon_junction_task0_window_gap.png) are retained as the second task, not pooled into the primary figure.','']
    primary=[r for r in rows if r['task']==1]
    text+=['## Measured changes from the weakest to strongest dose (task 1)','',
        '| Variant | Algorithm | Final gap: weakest → strongest | Cumulative regret: weakest → strongest |',
        '|---|---|---:|---:|']
    for kind,method in sorted({(r['kind'],r['method']) for r in primary}):
        group=sorted([r for r in primary if r['kind']==kind and r['method']==method],key=lambda r:r['dose']);lo,hi=group[0],group[-1]
        text.append(f"| {KINDS[kind]} | {LABELS[method]} | {lo['late_gap_mean']:.4f} → {hi['late_gap_mean']:.4f} | {lo['regret_mean']:.1f} → {hi['regret_mean']:.1f} |")
    text+=['','Persistent endpoint loss and larger accumulated learning cost are different findings. A control that eventually recovers must not be described as permanently failing.','', '## Complete dose table','']
    text+=['| Variant | Task | Algorithm | Dose | Final policy gap, mean ± SD | Cumulative regret | Nuisance episodes |','|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {KINDS[r['kind']]} | {r['task']} | {LABELS[r['method']]} | {r['dose']:g} | {r['late_gap_mean']:.4f} ± {r['late_gap_sd']:.4f} | {r['regret_mean']:.1f} | {100*r['nuisance_episode_fraction']:.1f}% |")
    text+=['','## What the evidence permits','',
        'The primitive controls receive trajectories and public action/state masks. IDS and CEAC receive declared candidate likelihoods and exact model-based policy responses. The oracle additionally receives the realized task/model. These are different information assumptions, not a matched model-free architecture comparison.','',
        'Primary CEAC uses the manuscript’s reward critic, price-conditioned evidence critic, and policy-labeled replay master. Statistical debt and design debt precede the actor; there is no exact-deployment branch. Exact model responses serve global certificates and target calibration. Oracle Graves–Lai additionally receives the true model. Historical exact-deployment results are excluded from this primary comparison.','']
    ceac=[r for r in rows if r['method']=='ceac']
    text.append(f"CEAC made {sum(r['ceac_actor_updates'] for r in ceac)} actor updates in total, "
        f"{sum(r['ceac_priced_actor_updates'] for r in ceac)} with nonzero evidence prices. This does not isolate a learned-critic or replay-buffer causal contribution.")
    if protocol=='prior_shift':
        text+=['','The correlated prior is frozen at (task 0, q=0.1) and (task 1, q=0.9). Both task labels and all observable outcomes remain possible, but the true joint law leaves prior support for d>0. Product-support IDS and CEAC use the same ten-model Cartesian class throughout. A robust product-support IDS result therefore limits any claim that this is an intrinsic failure of information-directed sampling.','',
            '| Task | IDS prior | Shift | True joint model supported | Final wrong-task probability |','|---:|---|---:|---|---:|']
        for r in rows:
            if r['method'].startswith('ids_'):text.append(f"| {r['task']} | {LABELS[r['method']]} | {r['dose']:g} | {r['true_joint_prior_supported']} | {r['wrong_task_posterior']:.4f} |")
    failures=[r for r in bundle['runs'] if r['method']=='ceac' and r['dose']==0 and r['summary']['late_gap']>cfg['easy_gap_tolerance']]
    if failures:
        text+=['','**Held-out easy-case exception:** '+ '; '.join(f"task {r['task']}, seed {r['seed']}: gap {r['summary']['late_gap']:.4f}" for r in failures)+'. Independent calibration did not guarantee success on every held-out seed. These results remain in the means and bands; they prevent a universal CEAC robustness claim.']
    text+=['','Full individual-seed endpoints are in `endpoints.json`/`endpoints.csv`; all registered conditions are retained. Three seeds support a descriptive initial study, not a population-level performance guarantee.','']
    return '\n'.join(text)


def export(folder,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bundle=load_study(folder);rows=endpoints(bundle);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'endpoints.json',rows)
    per_seed=[dict(protocol=r['protocol'],kind=r['kind'],task=r['task'],method=r['method'],dose=r['dose'],seed=r['seed'],
        late_gap=r['summary']['late_gap'],cumulative_regret=r['summary']['cumulative_regret'],
        episodes=r['summary']['settings']['episodes'],identity=r['summary']['identity']) for r in bundle['runs']]
    with (output/'seed_endpoints.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_seed[0]));writer.writeheader();writer.writerows(per_seed)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (output/'endpoints.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    (output/'assessment.md').write_text(summary_text(bundle,rows))
    for kind,task in sorted({(r['kind'],r['task']) for r in bundle['runs']}):
        for metric in ('window_gap','cumulative_regret'):
            fig=draw_learning(bundle,kind,task,metric);stem=f'{kind}_task{task}_{metric}'
            for suffix in ('png','pdf'):fig.savefig(output/f'{stem}.{suffix}',dpi=180,bbox_inches='tight')
            plt.close(fig)
    fig=draw_response(bundle)
    for suffix in ('png','pdf'):fig.savefig(output/f'dose_response.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    with gzip.open(output/'evidence.json.gz','wt') as f:json.dump(bundle,f,allow_nan=False,separators=(',',':'))
    atomic_json(output/'manifest.json',dict(schema=1,files={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'},
        registration=bundle['registration'],raw_source_bundle_sha256=digest(Path(folder)/'source_bundle.tar.gz')))
    print(f'Exported {len(bundle["runs"])} verified runs to {output}')


def load_export(output):
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
    for name,checksum in manifest['files'].items():
        if Path(name).name!=name:raise ValueError('Artifact path leaves export directory')
        if digest(output/name)!=checksum:raise ValueError(f'Portable evidence mismatch: {name}')
    with gzip.open(output/'evidence.json.gz','rt') as f:return json.load(f)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();export(args.study,args.output)

if __name__=='__main__':main()
