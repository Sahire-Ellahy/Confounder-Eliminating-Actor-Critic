"""Verified calibration/confirmation assessment; does not select or train."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,gzip,json
import numpy as np
from .experiment import load_completed,selection,PROTOCOL
from .run import ROOT,atomic_json,digest

LABELS={'original_reference':'Original\nconstant rates','scheduled_score_004':'Scheduled score\n0.04','scheduled_score_008':'Scheduled score\n0.08',
 'projected_score_004':'Projected score\n0.04','projected_score_008':'Projected score\n0.08','guarded_natural_001':'Guarded natural\n0.01','guarded_natural_003':'Guarded natural\n0.03'}


def portable(study):
    registration,rows=load_completed(study);runs=[]
    for row in rows:
        folder=ROOT/row['folder'];summary=json.loads((folder/'summary.json').read_text())
        with gzip.open(folder/'trace.jsonl.gz','rt') as f:
            trace=[]
            for line in f:
                t=json.loads(line);o=t['optimizer'];prob=np.asarray(t['actor_probabilities'])
                record={k:t[k] for k in ('episode','interactions','window_gap','cumulative_regret','actor_updates',
                    'final_actor_augmented_gap','final_actor_constrained_augmented_gap','final_actor_reward_gap',
                    'price_sum','best_priced_action_visits','actor_decisions')}
                record.update(minimum_probability=o['minimum_probability'],floor_mass=o['floor_mass'],
                    state_entropy=o['state_entropy'],fisher_condition=o['fisher_positive_eigenvalue_condition'],
                    optimizer_totals=o['totals'],
                    maximum_state_nuisance_probability=float(prob[:,3:].sum(axis=1).max()) if prob.shape[1]>3 else 0.)
                trace.append(record)
        opt=summary['ceac_diagnostics']['geometry_diagnostics'];prob=np.asarray(summary['actor_probabilities'])
        runs.append(dict(**row,trace=trace,optimizer_totals=opt['totals'],minimum_probability=opt['minimum_probability'],
            best_priced_action_visits=summary['best_priced_action_visits'],actor_decisions=summary['actor_decisions'],
            final_price_sum=float(np.sum(summary['ceac_diagnostics']['design']['beta'])),
            final_replay_counts=summary['ceac_diagnostics']['design']['arrays']['counts'],
            final_floor_mass=opt['floor_mass'],final_actor_probabilities=summary['actor_probabilities'],
            final_nuisance_probability=float(prob[:,3:].sum(axis=1).max()) if prob.shape[1]>3 else 0.,
            irreducible_floor_gap=summary['augmented_optimum']-summary['constrained_augmented_optimum']))
    return dict(registration=registration,runs=runs)


def draw_calibration(bundle):
    import matplotlib.pyplot as plt
    cfg=bundle['registration']['protocol'];ids=[c['id'] for c in cfg['candidates']]
    fig,axes=plt.subplots(1,2,figsize=(13.5,4.6))
    choices=[('final_actor_augmented_gap',.005,'Final current-actor priced-objective gap'),('late_gap',.025,'Late behavior external-reward gap')]
    for ax,(metric,threshold,title) in zip(axes,choices):
        for x,identity in enumerate(ids):
            for task,color in [(0,'#b6512d'),(1,'#2878a9')]:
                rows=sorted([r for r in bundle['runs'] if r['candidate']==identity and r['task']==task],key=lambda r:r['seed'])
                values=[r[metric] for r in rows]
                positions=x+np.linspace(-.15,.15,len(rows))+(task-.5)*.03
                ax.scatter(positions,values,color=color,s=28,label=f'Task {task}' if x==0 else None,alpha=.8,zorder=4)
            worst=max(r[metric] for r in bundle['runs'] if r['candidate']==identity)
            ax.plot([x-.22,x+.22],[worst,worst],color='#222222',lw=1.5)
        ax.axhline(threshold,color='#555555',ls='--',lw=1.,label=f'Gate: {threshold:g}')
        ax.set_yscale('log');ax.set_xticks(range(len(ids)),[LABELS[i] for i in ids],fontsize=8,rotation=25)
        ax.set_title(title,fontsize=11);ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=9)
    fig.suptitle('All eight calibration cells per setting; black marks are worst-cell values',fontsize=12)
    fig.tight_layout();return fig


def draw_calibration_learning(bundle):
    import matplotlib.pyplot as plt
    ids=[c['id'] for c in bundle['registration']['protocol']['candidates']]
    colors=['#555555','#e7ab45','#b27314','#87aee0','#326db1','#8ca774','#2c7c64']
    fig,axes=plt.subplots(1,2,figsize=(12.6,4.2),sharey=True)
    for ax,task in zip(axes,(0,1)):
        for name,color in zip(ids,colors):
            runs=sorted([r for r in bundle['runs'] if r['task']==task and r['candidate']==name],key=lambda r:r['seed'])
            x=np.array([t['interactions'] for t in runs[0]['trace']])/1000
            y=np.array([[t['cumulative_regret'] for t in r['trace']] for r in runs]);mean=y.mean(axis=0);sd=y.std(axis=0,ddof=1)
            ax.plot(x,mean,color=color,label=LABELS[name].replace('\n',' '),lw=1.6)
            ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.1,lw=0)
        ax.set(title=f'Task {task}',xlabel='Physical interactions (thousands)');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Cumulative external regret')
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',bbox_to_anchor=(.5,-.1),ncol=4,frameon=False,fontsize=8)
    fig.tight_layout();return fig


FAMILIES=[('prior_shift','false_beacon_junction','Cue reversal'),
          ('burden','undertow_junction','Undertow'),('burden','false_beacon_junction','False Beacon')]
COLORS={'original_reference':'#b04c5d','guarded_natural_003':'#18765a'}


def combined_confirmation(payload):
    bundles=[payload.get(name) for name in ('confirmation','secondary')]
    return dict(runs=[r for b in bundles if b for r in b['runs']])


def paired_rows(bundle):
    def key(r):return tuple(r[k] for k in ('protocol','kind','burden','dose','task','seed'))
    reference={key(r):r for r in bundle['runs'] if r['candidate']=='original_reference'}
    if len(reference)!=sum(r['candidate']=='original_reference' for r in bundle['runs']):
        raise ValueError('Duplicate reference condition')
    records=[];seen=set()
    for row in bundle['runs']:
        if row['candidate']=='original_reference':continue
        k=key(row);identity=(row['candidate'],k)
        if identity in seen or k not in reference:raise ValueError('Unpaired or duplicate optimizer condition')
        seen.add(identity);ref=reference[k]
        record={name:row[name] for name in ('protocol','kind','burden','dose','task','seed','candidate')}
        for metric in ('cumulative_regret','late_gap','final_actor_augmented_gap','final_actor_constrained_augmented_gap'):
            record['reference_'+metric]=ref[metric];record['variant_'+metric]=row[metric]
            record['difference_'+metric]=row[metric]-ref[metric]
        records.append(record)
    for method in {r['candidate'] for r in bundle['runs']} - {'original_reference'}:
        if {key(r) for r in bundle['runs'] if r['candidate']==method}!=set(reference):
            raise ValueError('Confirmation condition grid differs across optimizers')
    return records


def family_summary(bundle):
    """Average doses within each seed before computing between-seed variation."""
    pairs=paired_rows(bundle);groups=defaultdict(list)
    for r in pairs:groups[(r['protocol'],r['kind'],r['task'],r['candidate'])].append(r)
    out=[]
    for (protocol,kind,task,candidate),rows in sorted(groups.items()):
        seeds=sorted({r['seed'] for r in rows});record=dict(protocol=protocol,kind=kind,task=task,candidate=candidate,seeds=len(seeds))
        for metric in ('cumulative_regret','late_gap','final_actor_augmented_gap','final_actor_constrained_augmented_gap'):
            diffs=[np.mean([r['difference_'+metric] for r in rows if r['seed']==seed]) for seed in seeds]
            record['difference_'+metric]=float(np.mean(diffs))
            record['sd_difference_'+metric]=float(np.std(diffs,ddof=1))
            for method in ('reference','variant'):
                record[method+'_'+metric]=float(np.mean([r[method+'_'+metric] for r in rows]))
        record['regret_reduction_percent']=100*(1-record['variant_cumulative_regret']/record['reference_cumulative_regret'])
        out.append(record)
    return out


def draw_confirmation(bundle,metric='late_gap'):
    import matplotlib.pyplot as plt
    methods=sorted({r['candidate'] for r in bundle['runs']},key=lambda x:(x!='original_reference',x))
    fig,axes=plt.subplots(2,3,figsize=(12.7,7.0),squeeze=False)
    labels={'late_gap':'Late external-reward gap','cumulative_regret':'Cumulative external regret',
            'final_actor_augmented_gap':'Final actor priced-objective gap'}
    for col,(protocol,kind,title) in enumerate(FAMILIES):
        for task in (0,1):
            ax=axes[task,col]
            for method in methods:
                rows=[r for r in bundle['runs'] if (r['protocol'],r['kind'],r['task'],r['candidate'])==(protocol,kind,task,method)]
                doses=sorted({r['burden'] if protocol=='burden' else r['dose'] for r in rows})
                means=[];sd=[]
                for d in doses:
                    vals=[r[metric] for r in rows if (r['burden'] if protocol=='burden' else r['dose'])==d]
                    means.append(np.mean(vals));sd.append(np.std(vals,ddof=1))
                means=np.array(means);sd=np.array(sd);color=COLORS.get(method,'#237a63')
                # Equal-spaced registered doses keep B=0 and B=4 legible.
                x=np.arange(len(doses))
                ax.plot(x,means,marker='s' if method=='original_reference' else 'o',ls='--' if method=='original_reference' else '-',color=color,label=LABELS[method].replace('\n',' '),lw=1.8)
                ax.fill_between(x,np.maximum(0,means-sd),means+sd,color=color,alpha=.15,lw=0)
                ax.set_xticks(x,[f'{d:g}' for d in doses])
            ax.set(title=f'{title} · task {task}',xlabel='Cue reversal d' if protocol=='prior_shift' else 'Irrelevant opportunities B')
            if col==0:ax.set_ylabel(labels[metric])
            ax.set_ylim(bottom=0);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False)
    fig.suptitle(labels[metric]+' · five fresh seeds, mean ± sample SD',fontsize=12)
    fig.tight_layout(rect=(0,.05,1,.96));return fig


def draw_confirmation_learning(bundle,metric='cumulative_regret'):
    """Registered easy cue baseline and maximum Undertow burden, both tasks."""
    import matplotlib.pyplot as plt
    methods=sorted({r['candidate'] for r in bundle['runs']},key=lambda x:(x!='original_reference',x))
    fig,axes=plt.subplots(2,2,figsize=(11.6,7.0))
    conditions=[('prior_shift','false_beacon_junction',0,'Easy cue case (d=0)'),
                ('burden','undertow_junction',64,'Hard Undertow (B=64)')]
    for i,(protocol,kind,dose,title) in enumerate(conditions):
        for task in (0,1):
            ax=axes[i,task]
            for method in methods:
                rows=sorted([r for r in bundle['runs'] if (r['protocol'],r['kind'],r['task'],r['candidate'])==(protocol,kind,task,method)
                             and (r['dose'] if protocol=='prior_shift' else r['burden'])==dose],key=lambda r:r['seed'])
                if not rows:continue
                x=np.array([t['interactions'] for t in rows[0]['trace']])/1000
                color=COLORS.get(method,'#237a63')
                y=np.array([[t[metric] for t in r['trace']] for r in rows]);mean=y.mean(axis=0);sd=y.std(axis=0,ddof=1)
                ax.plot(x,mean,color=color,lw=1.8,ls='--' if method=='original_reference' else '-',label=LABELS[method].replace('\n',' '))
                ax.fill_between(x,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.15,lw=0)
            ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.set_ylim(bottom=0)
            ax.set_title(f'{title} · task {task}')
            ax.set_xlabel('Physical interactions (thousands)')
            if task==0:ax.set_ylabel('Cumulative external regret' if metric=='cumulative_regret' else 'Current actor priced-objective gap')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Full CEAC, original versus guarded natural update · five fresh seeds, mean ± sample SD',fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.96));return fig


def fresh_findings(payload):
    rows=combined_confirmation(payload)['runs']
    def group(method,task):
        return [r for r in rows if r['protocol']=='prior_shift' and r['candidate']==method and r['task']==task]
    ref0=group('original_reference',0);nat0=group('guarded_natural_003',0)
    ref1=group('original_reference',1);nat1=group('guarded_natural_003',1)
    if not all((ref0,nat0,ref1,nat1)):return ''
    floor_share=np.mean([r['irreducible_floor_gap'] for r in nat1])/np.mean([r['final_actor_augmented_gap'] for r in nat1])
    threshold=payload['calibration']['registration']['protocol']['calibration']['gate']['maximum_final_actor_augmented_gap']
    failed_reference=sum(r['final_actor_augmented_gap']>threshold for r in ref0)
    failed_natural=sum(r['final_actor_augmented_gap']>threshold for r in nat0)
    seeds=sorted({r['seed'] for r in ref0})
    improved=sum(np.mean([r['cumulative_regret'] for r in nat0 if r['seed']==seed]) < np.mean([r['cumulative_regret'] for r in ref0 if r['seed']==seed]) for seed in seeds)
    cue=(f"**Fresh-seed cue result:** the worst task-0 actor priced-objective gap changes from "
        f"{max(r['final_actor_augmented_gap'] for r in ref0):.4f} to {max(r['final_actor_augmented_gap'] for r in nat0):.4f}. "
        f"Using the calibration gap threshold ({threshold:g}) descriptively, {failed_reference}/{len(ref0)} original cue cells and {failed_natural}/{len(nat0)} natural cells exceed it. "
        f"Lower cue regret occurs on {improved}/{len(seeds)} seed-averaged task-0 sequences: the mean benefit is driven by avoiding a costly failure, not consistently faster learning. "
        f"The largest task-0 late reward gap changes from {max(r['late_gap'] for r in ref0):.4f} to {max(r['late_gap'] for r in nat0):.4f}. "
        f"Task 1 shows the cost of stabilization: mean late reward gap changes from {np.mean([r['late_gap'] for r in ref1]):.5f} "
        f"to {np.mean([r['late_gap'] for r in nat1]):.5f}; {100*floor_share:.1f}% of the natural update's mean final priced-objective "
        "gap is the cost of its imposed floor. This supports fewer large actor failures in the measured cue conditions, "
        "not universal improvement. Five seeds and their full paired dispersion are retained below.")
    hard={method:[r for r in rows if r['protocol']=='burden' and r['kind']=='undertow_junction' and r['burden']==64 and r['task']==0 and r['candidate']==method]
          for method in ('original_reference','guarded_natural_003')}
    if not all(hard.values()):return cue
    ref,nat=hard['original_reference'],hard['guarded_natural_003']
    rmean=np.mean([r['cumulative_regret'] for r in ref]);nmean=np.mean([r['cumulative_regret'] for r in nat])
    undertow=(f"**Largest registered Undertow burden (B=64), task 0:** mean regret changes from {rmean:,.0f} to {nmean:,.0f} "
        f"({100*(1-nmean/rmean):.1f}% lower). The worst actor priced-objective gap changes from "
        f"{max(r['final_actor_augmented_gap'] for r in ref):.4f} to {max(r['final_actor_augmented_gap'] for r in nat):.4f}. "
        "The original update can still fail at the longer budget; the natural update avoids those large optimization residuals in these five seeds.")
    return undertow+'\n\n'+cue


def notebook_narrative(payload):
    bundle=combined_confirmation(payload);groups=family_summary(bundle)
    lines=['## Actor stability: learning cost and final optimization error',
        'All settings keep the reward critic, evidence critic and evidence-priced replay. The frozen 56-cell calibration selected the **original constant-rate update** because it minimized worst-cell final priced-objective error. The guarded natural update (initial step 0.03) also passed both gates and reduced mean calibration regret by about 49%.',
        'Before inspecting confirmation outcomes, a separate secondary comparison was registered for every non-reference calibration passer. Only the guarded natural 0.03 setting qualified. It is an explicit optimizer extension; it does not replace the primary CEAC definition or its existing comparator results.',
        'Fresh-seed learning costs below average all registered doses within each family/task. Differences are **natural minus original**, with sample SD across the five seed-paired, dose-averaged differences. Negative is better; these are descriptive differences, not significance claims.',
        '| Family | Task | Original regret | Natural regret | Paired difference ± SD |',
        '|---|---:|---:|---:|---:|']
    for r in groups:
        family=next(title for protocol,kind,title in FAMILIES if (protocol,kind)==(r['protocol'],r['kind']))
        lines.append(f"| {family} | {r['task']} | {r['reference_cumulative_regret']:,.0f} | {r['variant_cumulative_regret']:,.0f} | {r['difference_cumulative_regret']:,.0f} ± {r['sd_difference_cumulative_regret']:,.0f} |")
    lines+=['The figure shows the registered easy cue case (d=0) and maximum Undertow burden (B=64), both tasks and all five seeds. Panels use their own scales and registered episode budgets; all intermediate doses remain in the supporting plots. Full dose responses, terminal reward gaps, floor-constrained optimization gaps and raw per-seed differences are retained in the [stability assessment](evidence/actor_stability/assessment.md). A remaining task-0 reward loss near 0.02 can reflect priced probing (probe reward 0.48 versus safe reward 0.50), even when actor optimization is accurate. Temporary zero prices during replay recertification can produce objective-gap spikes while the actor is frozen; the evaluated objective has changed. These new seeds compare complete CEAC optimizer settings; the external algorithms in the earlier figures used a different seed set.']
    return '\n\n'.join(lines[:3]+[fresh_findings(payload),lines[3]])+'\n\n'+'\n'.join(lines[4:-1])+'\n\n'+lines[-1]


def render(study,output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    study=Path(study);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    calibration=portable(study/'calibration');cfg=calibration['registration']['protocol']
    choice=selection(cfg,calibration['runs'])
    if choice!=json.loads((study/'calibration/selection.json').read_text()):raise ValueError('Selection changed')
    bundles={}
    for key,folder in [('confirmation','confirmation'),('secondary','confirmation_secondary')]:
        path=study/folder/'status.json'
        bundles[key]=portable(study/folder) if path.exists() and json.loads(path.read_text())['status']=='complete' else None
    for bundle in bundles.values():
        if bundle and (bundle['registration']['sources']!=calibration['registration']['sources'] or
                       bundle['registration']['protocol_sha256']!=calibration['registration']['protocol_sha256']):
            raise ValueError('Confirmation numerical sources/protocol differ from calibration')
    secondary=bundles['secondary']
    if secondary:
        expected=sorted(r['candidate'] for r in choice['candidates'] if r['passed'] and r['candidate']!='original_reference')
        if secondary['registration']['candidates']!=expected:raise ValueError('Secondary passers changed')
        for name,expected_hash in secondary['registration']['selection_provenance'].items():
            if digest(study/'calibration'/f'{name}.json')!=expected_hash:raise ValueError('Secondary provenance changed')
    payload=dict(schema=2,calibration=calibration,selection=choice,**bundles)
    combined=combined_confirmation(payload)
    if secondary and not bundles['confirmation']:raise ValueError('Secondary comparison requires complete reference')
    paired=paired_rows(combined);family=family_summary(combined)
    payload['paired_family_summary']=family
    # Deterministic compressed evidence makes regeneration hashes stable.
    with gzip.GzipFile(filename=str(output/'evidence.json.gz'),mode='wb',mtime=0) as f:
        f.write(json.dumps(payload,allow_nan=False,separators=(',',':')).encode())
    rows=choice['candidates'];selected=choice['selected']
    text=['# CEAC actor stability: calibration and independent confirmation','',
        f'**{len(calibration["runs"])} calibration cells; {len(combined["runs"])} completed fresh-seed confirmation cells.** Every setting retains both critics and evidence-priced policy-labeled replay.','',
        '**Primary decision: '+str(selected)+'.** The frozen ranking minimizes worst-cell final actor priced-objective gap before considering cumulative external regret. No candidate, threshold or seed was changed after calibration.','']
    passing={r['candidate']:r for r in rows if r['passed']}
    if 'guarded_natural_003' in passing and 'original_reference' in passing:
        improvement=100*(1-passing['guarded_natural_003']['mean_regret']/passing['original_reference']['mean_regret'])
        text += [f'The guarded natural 0.03 update also passed both calibration gates, with **{improvement:.1f}% lower mean calibration regret**. It had a larger final priced-objective gap, so it was not promoted by the primary rule. Endpoint accuracy and learning cost are distinct criteria.','']
    if secondary:
        text += ['## Fresh-seed secondary comparison','',
            'After calibration, and **before inspecting confirmation outcomes**, we registered a secondary rule: confirm every non-reference calibration passer on the already specified seeds, tasks, doses and budgets. Only guarded natural 0.03 qualified. The primary selection remains unchanged.','',
            fresh_findings(payload),'',
            f'Registered UTC: `{secondary["registration"]["registered_utc"]}`. There are 130 original-reference and 130 guarded-natural cells, using seeds 3001–3005. Cue studies use 32,768 episodes; burden studies use 131,072. Every episode takes two physical interactions.','',
            '![Learning cost in the easy cue case and maximum Undertow burden, both tasks](confirmation_learning.png)','',
            'The easy cue baseline d=0 and maximum Undertow burden B=64 are registered endpoints. Both tasks and all seeds are shown; intermediate doses follow below. Each panel uses its own scale and the original family-specific budget. Lines are seed means and bands are one sample SD. Cumulative regret is the sum of expected episode reward gaps. Current-actor optimization error is evaluated against the currently priced objective; it is not external regret. Brief spikes coincide with replay recertification: prices temporarily become zero and the actor stays frozen, so the evaluated objective changes.','',
            '![Actor optimization in the same baseline and stress conditions](confirmation_optimization.png)','',
            '| Family | Task | Original regret | Natural regret | Paired difference ± SD | Mean late gap: original → natural |',
            '|---|---:|---:|---:|---:|---:|']
        for r in family:
            label=next(title for protocol,kind,title in FAMILIES if (protocol,kind)==(r['protocol'],r['kind']))
            a=f"{r['reference_cumulative_regret']:,.1f}";b=f"{r['variant_cumulative_regret']:,.1f}"
            if r['reference_cumulative_regret']<r['variant_cumulative_regret']:a='**'+a+'**'
            elif r['variant_cumulative_regret']<r['reference_cumulative_regret']:b='**'+b+'**'
            text.append(f"| {label} | {r['task']} | {a} | {b} | {r['difference_cumulative_regret']:,.1f} ± {r['sd_difference_cumulative_regret']:,.1f} | {r['reference_late_gap']:.5f} → {r['variant_late_gap']:.5f} |")
        text+=['','Each row averages all registered doses within a family/task. The paired SD is computed over five seed-level averages, retaining pairing across doses; doses are not treated as independent seeds. Negative differences favor the natural update. Bold means lower average regret, not statistical significance. Different families have different episode budgets.','',
            '![Learning cost over every dose](confirmation_cost.png)','',
            '![Late reward gap over every dose](confirmation_gap.png)','',
            '![Actor optimization error over every dose](confirmation_objective.png)','',
            'All dose points are shown at equal horizontal spacing for legibility, with their actual values labeled. Error bands are one sample SD across five seeds, not confidence intervals. No external-method ranking is updated using these optimizer-only runs.','',
            '## Mechanism checks and remaining costs','',
            '| Family | Task | Optimizer | Maximum priced gap | Maximum gap above floor-constrained optimum | Mean forced-floor gap | Mean final nuisance mass |',
            '|---|---:|---|---:|---:|---:|---:|']
        for protocol,kind,title in FAMILIES:
            for task in (0,1):
                for method in ('original_reference','guarded_natural_003'):
                    group=[r for r in combined['runs'] if (r['protocol'],r['kind'],r['task'],r['candidate'])==(protocol,kind,task,method)]
                    text.append(f"| {title} | {task} | {LABELS[method].replace(chr(10),' ')} | {max(r['final_actor_augmented_gap'] for r in group):.6f} | {max(r['final_actor_constrained_augmented_gap'] for r in group):.6f} | {np.mean([r['irreducible_floor_gap'] for r in group]):.6f} | {np.mean([r['final_nuisance_probability'] for r in group]):.6f} |")
        text+=['','The total probability floor bounds forced nuisance mass independently of the action count, but it still imposes a finite-budget cost. The constrained gap subtracts only that feasible-policy cost. Persistent probe preference can remain even after actor optimization converges, because conservative replay prices may favor a probe with reward 0.48 over the safe route with reward 0.50.','',
            'The saved traces retain minimum propensity, entropy, Fisher conditioning, TD/gradient second moments, accepted-update KL, trust contractions, best-priced-action visits, evidence-price magnitude and floor mass. Final replay counts and policy probabilities are included in the portable evidence. These jointly changed optimizer settings do not isolate a causal contribution from preconditioning, critic scaling or the floor individually.','']
    elif bundles['confirmation']:
        text+=['The primary selection was the original reference, so primary confirmation contains one unique algorithm. A registered secondary extension has not yet completed.','',
            '![Primary confirmation, every dose](confirmation_gap.png)','']
    else:
        text+=['Confirmation is still in progress. This assessment makes no fresh-seed performance claim.','']
    text+=['## Calibration and the frozen decision','',
        '![Every calibration cell against the two gates](calibration_gate.png)','',
        '| Optimizer setting | Worst final priced gap | Worst late reward gap | Mean accumulated regret | Passed both gates |',
        '|---|---:|---:|---:|---:|---|']
    for r in rows:
        label=LABELS[r['candidate']].replace('\n',' ')
        if r['candidate']==selected:label='**'+label+'**'
        text.append(f"| {label} | {r['worst_augmented_gap']:.6f} | {r['worst_late_reward_gap']:.6f} | {r['mean_regret']:,.1f} | {r['passed']} |")
    text+=['','Every calibration cell must have final actor priced-objective gap ≤ 0.005, late behavior reward gap ≤ 0.025, and nonzero-price actor updates. There are four seeds per task, two tasks, and 32,768 episodes per cell. The original reference is eligible under the frozen rule. These thresholds govern calibration, not a new pass/fail claim for all confirmation doses.','',
        '![Learning cost during calibration](calibration_learning.png)','',
        'All settings are retained, including failed candidates. Slower score updates reduce some learning costs but miss the late reward-gap gate. The projected score updates closely track the ordinary scheduled paths; in these runs their floors rarely bind. The endpoint-first rule prefers the sharper final solution of the original update to the natural update’s learning-cost tradeoff.','',
        '## Implementation and reproducibility','',
        'The original setting reproduces frozen actor updates, raw trajectory text and replay design histories exactly in integration testing. Scheduled critics use both TD errors frozen before either critic update. Projected variants use actual KL projection, an annealed total probability floor and a KL bound checked after projection. The natural variant uses the fresh sampled TD error and on-policy propensity, never evaluator Q values.','',
        'Natural/projection variants are explicit optimizer extensions to complete CEAC. They are not the manuscript’s literal score-function update. The floor can add polynomial exploration cost; this experiment does not prove convergence or Graves–Lai optimal regret. Critic/replay prices and policy-label contracts remain intact. No actor-only ablation was introduced.','',
        '[Protocol and equations](../../../docs/actor_stability_protocol.md) · [Original failure diagnosis](../../../audits/ceac_actor_failure_20261112/assessment.md) · [Every endpoint](endpoints.csv) · [Paired differences](paired_differences.csv) · [Figures PDF](figures.pdf).','',
        'Reproduce from the repository root:', '',
        '```bash',
        'python -m ceac_stability.experiment pipeline --output runs/ceac_actor_stability_20261112 --workers 6',
        'python scripts/run_actor_stability_secondary.py --study runs/ceac_actor_stability_20261112 --workers 6',
        'python -m ceac_stability.report --study runs/ceac_actor_stability_20261112 --output notebooks/evidence/actor_stability',
        '```','',
        'Completed cells and registrations are verified before reuse. Incomplete existing cells fail closed. Raw episodes, source bundles and original studies remain unchanged; compact evidence supports offline notebook rendering.','']
    (output/'assessment.md').write_text('\n'.join(text))
    seed_rows=[]
    for phase,bundle in [('calibration',calibration),('confirmation_primary',bundles['confirmation']),('confirmation_secondary',secondary)]:
        if not bundle:continue
        for r in bundle['runs']:
            row=dict(phase=phase,**{k:v for k,v in r.items() if not isinstance(v,(list,dict))})
            row.update({'optimizer_'+key:value for key,value in r['optimizer_totals'].items()});seed_rows.append(row)
    for name,records in [('endpoints',seed_rows),('paired_differences',paired),('paired_family_summary',family)]:
        with (output/f'{name}.csv').open('w') as f:
            if records:
                writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    figures=[('calibration_gate',draw_calibration(calibration)),('calibration_learning',draw_calibration_learning(calibration))]
    if combined['runs']:
        figures += [('confirmation_gap',draw_confirmation(combined)),
                    ('confirmation_cost',draw_confirmation(combined,'cumulative_regret')),
                    ('confirmation_objective',draw_confirmation(combined,'final_actor_augmented_gap')),
                    ('confirmation_learning',draw_confirmation_learning(combined)),
                    ('confirmation_optimization',draw_confirmation_learning(combined,'final_actor_augmented_gap'))]
    with PdfPages(output/'figures.pdf') as pdf:
        for name,fig in figures:
            for ext in ('png','pdf'):fig.savefig(output/f'{name}.{ext}',dpi=180,bbox_inches='tight')
            pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    atomic_json(output/'manifest.json',dict(schema=2,files={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'},
        protocol_sha256=digest(PROTOCOL),analysis_source_sha256=digest(__file__),
        calibration_status_sha256=digest(study/'calibration/status.json'),selection_sha256=digest(study/'calibration/selection.json'),
        confirmation_status_sha256=digest(study/'confirmation/status.json') if bundles['confirmation'] else None,
        secondary_status_sha256=digest(study/'confirmation_secondary/status.json') if secondary else None))
    print(f'Rendered {len(calibration["runs"])} calibration cells and {len(combined["runs"])} confirmation cells.')


def load_export(folder):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if Path(name).name!=name or digest(folder/name)!=expected:raise ValueError('Stability export hash mismatch')
    with gzip.open(folder/'evidence.json.gz','rt') as f:return json.load(f)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();render(args.study,args.output)

if __name__=='__main__':main()
