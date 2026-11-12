"""Mechanism-first figures from verified exports; no training."""
import numpy as np
from ceac_hardness.report import endpoints, LABELS, KINDS
from ceac_walkthrough.mechanisms import occupancy_ratio, population_nuisance

DOSE_COLORS = ['#7da9c7', '#4c8bb5', '#235e8c', '#092f58', '#43245a']


def seed_curve(cells, getter):
    xs = [np.array([t['interactions'] for t in c['trace']])/1000 for c in cells]
    if not xs or any(not np.array_equal(x, xs[0]) for x in xs):
        raise ValueError('Unequal or empty trace grids')
    values = np.array([[getter(t) for t in c['trace']] for c in cells])
    return xs[0], values.mean(axis=0), values.std(axis=0, ddof=1) if len(cells)>1 else np.zeros_like(values[0])


def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.16)
    ax.tick_params(labelsize=9)


def draw_failure_fans(bundle):
    """STEERING-style same-algorithm time traces; shade is across seeds."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    protocol = bundle['registration']['protocol']
    if protocol == 'burden':
        methods = ['idac', 'ucb_q', 'pomd', 'ceac']
        panels = [('undertow_junction', 1), ('false_beacon_junction', 1)]
        names = ['IDAC', 'UCB Q-learning', 'Optimistic POMD', 'CEAC']
    else:
        methods = ['ids_correlated', 'ids_product', 'ceac']
        panels = [('false_beacon_junction', 1), ('false_beacon_junction', 0)]
        names = ['IDS: correlated prior', 'IDS: product-support prior', 'CEAC']
    fig, axes = plt.subplots(2, len(methods), figsize=(12.8, 6.5), sharex=True, sharey=True, squeeze=False)
    doses = sorted({r['dose'] for r in bundle['runs']})
    colors = DOSE_COLORS[:len(doses)]
    largest_upper = 0.
    for row, (kind, task) in enumerate(panels):
        for col, (method, name) in enumerate(zip(methods, names)):
            ax = axes[row, col]
            for dose, color in zip(doses, colors):
                cells = [r for r in bundle['runs'] if (r['kind'],r['task'],r['method'],r['dose']) == (kind,task,method,dose)]
                x, y, sd = seed_curve(cells, lambda t:t['cumulative_regret']/1000)
                largest_upper = max(largest_upper, float(np.max(y+sd)))
                ax.fill_between(x, np.maximum(0,y-sd), y+sd, color=color, alpha=.15, lw=0)
                ax.plot(x, y, color=color, lw=1.8, label=f'{"B" if protocol=="burden" else "d"} = {dose:g}')
            ax.set_title(name, fontsize=11, fontweight='bold', pad=8)
            ax.set_xlim(0,x[-1])
            ax.xaxis.set_major_locator(MaxNLocator(4)); style_axis(ax)
            if col == 0:
                label = KINDS[kind] if protocol == 'burden' else f'Task {task}: {"risky" if task else "safe"} optimal'
                ax.set_ylabel(label+'\nRegret (thousands)', fontsize=10)
            if method == 'ceac' and protocol == 'burden':
                selected = [r['summary']['cumulative_regret'] for r in bundle['runs'] if (r['kind'],r['task'],r['method'],r['dose']) == (kind,task,method,max(doses))]
                ax.text(.05,.85,f'B=64 mean regret: {np.mean(selected):.0f}\nSmall but nonzero on shared scale', transform=ax.transAxes, fontsize=8.5)
            if method == 'ids_product' and protocol == 'prior_shift':
                ax.text(.05,.85,'Full support survives this shift',transform=ax.transAxes,fontsize=9)
            if method == 'ceac' and task == 0:
                ax.text(.05,.85,'Actor-learning failures retained',transform=ax.transAxes,fontsize=9)
    axes[0,0].set_ylim(0, max(1e-3, largest_upper*1.06))
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.01),ncol=len(doses),frameon=False)
    fig.supxlabel('Physical training interactions (thousands)',fontsize=11)
    fig.tight_layout(rect=(0,.03,1,.94))
    return fig


def draw_objective_diagnostics(burden, prior):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12.5,3.8))
    p = np.linspace(0,1,301)
    for b, color in zip((4,16,64), DOSE_COLORS[1:4]):
        axes[0].plot(p,occupancy_ratio(p,b),color=color,label=f'B={b}')
        opt = population_nuisance(b)
        axes[0].scatter([opt],[occupancy_ratio(opt,b)],color=color,s=30,zorder=4)
    axes[0].set(xlabel='Probability of irrelevant excursion',ylabel='Population cost / (entropy + 0.1)',title='IDAC: coverage can improve the wrong objective')
    axes[0].legend(frameon=False,fontsize=8)
    rows = endpoints(burden)
    for kind, marker in [('undertow_junction','o'),('false_beacon_junction','s')]:
        r = sorted([r for r in rows if r['kind']==kind and r['method']=='pomd'],key=lambda r:r['dose'])
        axes[1].plot([x['dose'] for x in r],[x['pomd_zero_fraction'] for x in r],marker=marker,label=KINDS[kind])
    axes[1].set(xlabel='Irrelevant opportunities B',ylabel='Fraction of all-zero optimistic Q iterates',title='POMD: clipped optimism can stay flat',ylim=(-.03,1.05))
    axes[1].legend(frameon=False,fontsize=8)
    d = np.linspace(0,1,201); q = .9-.8*d
    axes[2].plot(d,(2*q-1)*np.log(9),color='#9b2948',label='Correlated-prior cue evidence')
    axes[2].axhline(0,color='#444444',lw=.8);axes[2].axvline(.5,color='#aaaaaa',ls=':')
    axes[2].set(xlabel='Cue reversal d',ylabel='Expected log odds added per cue (task 1)',title='IDS: more cues can strengthen the wrong belief')
    for ax in axes:style_axis(ax);ax.title.set_fontsize(10)
    fig.tight_layout();return fig


def draw_critic_mechanism(data):
    """Representative hard environment, all seeds; exact recorded increments."""
    import matplotlib.pyplot as plt
    cells = sorted([c for c in data['cells'] if c['kind']=='undertow_junction' and c['dose']==64],key=lambda c:c['seed'])
    if len(cells)!=3:raise ValueError('Expected every registered B=64 Undertow CEAC seed')
    fig, axes = plt.subplots(1,3,figsize=(12.8,3.7))
    x = np.arange(4); k = cells[0]['action_task_kl']+[0.]
    axes[0].bar(x,k,color=['#aaaaaa','#2477a8','#744b9b','#b7b7b7'])
    axes[0].set_xticks(x,['Safe','Risky','Probe','Undertow'])
    axes[0].set(ylabel='Expected task log-likelihood ratio (nats)',title='1. What distinguishes the task?')
    for i,val in enumerate(k):axes[0].text(i,val+.04,f'{val:.2f}',ha='center',fontsize=9)
    axes[0].set_ylim(0,2.1)
    for key,label,color,ls in [('reward_contrast','Reward TD contribution','#2477a8','-'),('evidence_contrast','Evidence TD contribution','#9b3c79','-'),('logit_contrast','Their sum','#333333','--')]:
        t,y,sd = seed_curve(cells,lambda r:r[key]['risky_vs_nuisance'])
        axes[1].plot(t,y,label=label,color=color,ls=ls,lw=1.8)
        axes[1].fill_between(t,y-sd,y+sd,color=color,alpha=.12,lw=0)
    axes[1].set(title='2. How did the actor actually move?',xlabel='Physical interactions (thousands)',ylabel='Cumulative risky − mean-dead-end logit update')
    axes[1].legend(frameon=False,fontsize=8)
    for key,label,color in [('risky','Risky route','#2477a8'),('probe','Diagnostic probe','#744b9b'),('nuisance','All 64 dead ends','#ae672e')]:
        t,y,sd=seed_curve(cells,lambda r:r['actor_probabilities'][key])
        axes[2].plot(t,y,label=label,color=color,lw=1.8)
        axes[2].fill_between(t,np.maximum(0,y-sd),np.minimum(1,y+sd),color=color,alpha=.12,lw=0)
    timing=[2*c['first_actor_episode']/1000 for c in cells]
    axes[2].axvspan(min(timing),max(timing)+.1,color='#dddddd',alpha=.8)
    axes[2].set(title='3. What did the learned actor choose?',xlabel='Physical interactions (thousands)',ylabel='Current actor probability',ylim=(-.03,1.05),xlim=(0,12))
    axes[2].text(.97,.6,'First 12k interactions\nReference/design before actor',ha='right',transform=axes[2].transAxes,fontsize=8)
    axes[2].legend(frameon=False,fontsize=8,loc='center right',bbox_to_anchor=(1,.32))
    for ax in axes:style_axis(ax);ax.title.set_fontsize(10)
    fig.tight_layout();return fig


def narrative(burden, prior, audit):
    """One narrative shared by the report, PDF and executable notebook."""
    rows=endpoints(burden); prior_rows=endpoints(prior)
    def row(kind,method,dose):
        return next(r for r in rows if (r['kind'],r['method'],r['dose'])==(kind,method,dose))
    opening = '''# When exploration follows the wrong signal

**Does an algorithm solve the easy case, then deteriorate when irrelevant opportunities or misleading cues increase?** Each main panel follows **one algorithm over time**. Increasing darkness means increasing pathology; upward-fanning cumulative-regret curves expose the additional learning cost. Bands are one sample standard deviation across three held-out seeds, truncated at zero for regret; they are not confidence intervals. All panels within a figure use the same linear scale.

This adopts the visual logic of the variable-sparsity panel in [STEERING](https://arxiv.org/abs/2301.12038) (Figure 2(c) in arXiv v2; the earlier discussion referred to Figure 1(c)). Here the intervention is an explicit environmental change, rather than a sparsity label.

The evidence is 144 physical-burden cells and 90 prior-shift cells. These are **two-step junction variants**, not the historical full DeepSea/Frozen Lake grids. Every dose is a separate stationary run; there is no within-run change point. The original rho sweep did not establish this easy-to-hard transition.'''
    burden_setup = '''## 1. More exploration opportunities, no more task information

**Undertow** adds B zero-reward dead-end actions. **False Beacon** offers one action reaching a uniformly chosen cell in a zero-reward chamber of breadth B. Safe, risky and diagnostic actions keep exactly the same rewards and information. Every episode still takes two physical interactions. B = 0, 4, 16, 64; the two variants share the same B=0 kernel.

Independent calibration checked both tasks on seeds 1001/1002, requiring final gap ≤ 0.025. The held-out burden study uses task 1 (risky route optimal), seeds 0/1/2 and 262,144 interactions for every method. Domain growth changes both occupancy entropy and the published optimism bonuses; this is the intended intervention.'''
    table=['| Variant | Mechanism | Final policy loss, B=0 → 64 | Regret at B=64 | Irrelevant episodes at B=64 |',
           '|---|---|---:|---:|---:|']
    for kind in ('undertow_junction','false_beacon_junction'):
        for method,name in [('idac','IDAC: occupancy'),('ucb_q','UCB: uncertainty'),('pomd','POMD: optimism'),('ceac','CEAC: evidence')]:
            lo,hi=row(kind,method,0),row(kind,method,64)
            table.append(f"| {KINDS[kind].replace(' junction','')} | {name} | {lo['late_gap_mean']:.5f} → {hi['late_gap_mean']:.5f} | {hi['regret_mean']:,.0f} | {100*hi['nuisance_episode_fraction']:.1f}% |")
    burden_findings = '''### Read the failures by mechanism

**Coverage / IDAC:** broad occupancy rewards irrelevant states. For nuisance probability p, with the remaining mass on the optimal useful action, the exact population objective is C(p)/(H(p)+0.1), where C(p)=1.01−0.45(1−p) and H(p)=log(2)+[h(p)+p log B]/2. Its preferred nuisance mass grows with breadth. This is a population-objective diagnosis; the implemented 16-episode occupancy estimator need not reach that optimum. False Beacon IDAC is worse at B=16 than B=64 in mean cumulative regret: breadth exposes a failure relative to B=0, but does not strictly order every finite-budget result.

**Greedy UCB Q-learning:** extra state/action uncertainty incurs a large finite-budget learning cost. A steep early curve is different from permanent failure: some hard seeds recover near the end.

**Optimistic softmax POMD:** at Undertow B=64 all optimistic cost-Q values remain clipped to zero for the entire run. Equal optimistic costs leave the policy uniform across 67 actions. The resulting exact policy gap is 0.9−(0.5+0.9+0.48)/67 = **0.87194**, matching the measured endpoint. This explains the nearly linear hard-case regret.

**CEAC:** the nuisance observations have zero likelihood-ratio evidence for every task pair. Increasing their number does not create evidence credit. The learned policy avoids this large exploration cost, with small nonzero regret on the shared scale.

'''+ '\n'.join(table)+'''

Final loss averages the last 256 episodes; regret counts the entire learning period. **Product-support IDS and the offline Graves–Lai oracle also succeed:** their mean final loss is zero throughout this burden sweep; mean cumulative regret is at most 0.694 for product-support IDS and zero for the oracle. Their complete learning panels are retained in the supporting report. CEAC is not the only successful model-informed method.'''
    prior_setup = '''## 2. Bayesian information can point toward the wrong task

A mandatory cue precedes a safe/risky/probe decision. Freeze the correlated prior at two equally weighted joint models: **(task 0, q=0.1)** and **(task 1, q=0.9)**. Both task labels and every possible observation retain positive probability.

For true task 1, increase reversal d by setting q(d)=0.9−0.8d; task 0 uses q(d)=0.1+0.8d. Every nonzero registered d excludes the true *joint law* from this prior. Under task 1 the cue alone adds expected log odds (2q−1)log(9) for task 1; this sign reverses after d=0.5. More data can then strengthen the wrong belief.

Product-support IDS and CEAC use the same ten-model Cartesian class (two tasks × five cue probabilities). For every cue law it includes an opposite-task alternative with **the same q**, which the cue alone cannot distinguish. All three methods receive 65,536 physical interactions per run. Both tasks stay visible below.'''
    prior_findings = '''At full reversal, correlated IDS has mean final loss **0.4 on both tasks**; product-support IDS remains near zero. This demonstrates failure of the restrictive task–cue prior, not an intrinsic failure of Bayesian IDS. Leaving support at d>0 is not itself a monotone guarantee of failure: the conspicuous behavioral break appears at stronger reversal.

On task 1, CEAC's final loss remains approximately **0.0048** across doses. **Task 0 is an exception:** seed 0 has large loss at d=0, 0.75 and 1; other seeds retain roughly 0.023 loss. These runs remain in the means and bands. Increasing dose is a controlled misspecification intervention, not a universal hardness ranking for every algorithm and task.'''
    updates=sum(c['actor_updates'] for c in audit['cells'])
    max_error=max(c['max_update_sum_error'] for c in audit['cells'])
    hard=[c for c in audit['cells'] if c['kind']=='undertow_junction' and c['dose']==64]
    reward=np.mean([c['trace'][-1]['reward_contrast']['risky_vs_nuisance'] for c in hard])
    evidence=np.mean([c['trace'][-1]['evidence_contrast']['risky_vs_nuisance'] for c in hard])
    beacon=[c for c in audit['cells'] if c['protocol']=='burden' and c['kind']=='false_beacon_junction' and c['dose']==64]
    beacon_evidence=np.mean([c['trace'][-1]['evidence_contrast']['risky_vs_nuisance'] for c in beacon])
    timing=[c['first_actor_episode'] for c in hard]
    timing_label = f'episode {timing[0]:,}' if len(set(timing))==1 else f'episodes {min(timing):,}–{max(timing):,}'
    critic_setup = '''## 3. Where the evidence critic enters the actual CEAC update

CEAC here is the manuscript's **reward critic + price-conditioned evidence critic + evidence-priced, policy-labeled replay**, in one episode loop. Its actor uses the sum of the two frozen TD errors; the critics then update from those same errors. Statistical and design debts select diagnostic episodes before the actor is allowed to act. There is no exact-model deployment branch.

The hard Undertow example below separates three questions: which observations distinguish tasks, how reward/evidence TD terms moved the logits, and how the actor's probabilities changed. The first panel is an exact kernel calculation; the other panels reconstruct saved actor updates and average all three hard-case seeds. The probability panel zooms to the first 12,000 interactions and shows the current actor, not the diagnostic policies executed before it.'''
    critic_findings = f'''**Measured contribution:** at Undertow B=64, the cumulative risky-minus-mean-dead-end logit increment decomposes into **{reward:.2f} from reward TD** and **{evidence:.2f} from evidence TD**, averaged over seeds. Their sum is the actual actor increment. At False Beacon B=64 the analogous evidence increment is **{beacon_evidence:.2f}**. Here reward and task discrimination favor the same risky route. Positive evidence contribution reinforces that route; it is not a measured counterfactual improvement in final return.

**Timing matters:** the first certified replay design precedes or coincides with the first actor episode ({timing_label} in these hard runs, zero-based). Thus initial reference identification is achieved by statistical replay before the learned evidence critic updates. The performance curves support the combined CEAC mechanism; they do not establish that the critic caused that initial identification.

**Replay really uses evidence prices:** for every logged design we verify `priced score = empirical return + beta × empirical likelihood-ratio evidence`. The audit reconstructs all **{updates:,} actor/two-critic updates across {len(audit['cells'])} CEAC runs**, checks every recorded actor-policy identity, and checks that reward/evidence increments sum to the actual update (maximum per-step residual {max_error:.2g}). This uses saved observations, with no new training or alternative policy trajectory.

For the prior-shift family, an opposite-task model with the same q always gives zero cue evidence. Finite-radius design prices can still credit the cue against a different-q alternative; it would be incorrect to claim that CEAC gives every cue zero price. Same-q task-changing constraints still require decision-relevant observations.'''
    table=['| Task 0, d=0 | Reward contribution | Evidence contribution | Final policy loss |',
           '|---|---:|---:|---:|']
    for c in sorted([c for c in audit['cells'] if c['protocol']=='prior_shift' and c['task']==0 and c['dose']==0],key=lambda c:c['seed']):
        last=c['trace'][-1]
        run=next(r for r in prior['runs'] if r['method']=='ceac' and r['task']==0 and r['dose']==0 and r['seed']==c['seed'])
        table.append(f"| Seed {c['seed']} | {last['reward_contrast']['risky_vs_safe']:+.2f} | {last['evidence_contrast']['risky_vs_safe']:+.2f} | {run['summary']['late_gap']:.4f} |")
    limits = '''## 4. When evidence is insufficient

**Correct model identification does not guarantee a well-optimized actor.** The saved task-0/d=0 failure has the correct reference and finite critic values. Its evidence contribution toward risky exceeds the opposing reward contribution; a substantial augmented-objective optimization residual remains. The same settings on all three seeds give the following exact cumulative risky-minus-safe logit decomposition (weighted by true cue-state visitation):

'''+ '\n'.join(table)+'''

These signed contributions explain recorded movement; they do not establish the outcome of removing or retuning the critic. The implementation has not been tuned on these held-out failures. An evidence critic can help direct learning toward informative behavior and still leave finite-budget optimization or over-exploration problems.

**Scope of the comparison:** IDAC/UCB/POMD receive primitive trajectories and public state/action masks; CEAC and IDS additionally receive candidate likelihoods and model-based responses. The offline Graves–Lai oracle additionally knows the realized model and fixes its design from that truth. This is a comparison of exploration mechanisms under explicit information assumptions, not a matched model-free architecture comparison. Three seeds provide descriptive evidence, not a population-level guarantee.'''
    return dict(opening=opening,burden_setup=burden_setup,burden_findings=burden_findings,
        prior_setup=prior_setup,prior_findings=prior_findings,critic_setup=critic_setup,
        critic_findings=critic_findings,limits=limits)


def render_curated(burden, prior, audit, output, stability=None):
    """Portable Markdown, figures and a compact PDF, all from the same inputs."""
    from pathlib import Path
    import json
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from ceac_hardness.report import draw_junctions
    from ceac_hardness.run import digest, atomic_json
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    text=narrative(burden,prior,audit)
    if stability:
        text['opening']+='\n\n**Fresh optimizer confirmation:** the separate stability section adds five new seeds and reveals additional original-CEAC actor failures at maximum Undertow burden. A guarded natural update reduces these errors while preserving both critics and evidence-priced replay. The earlier comparison figures below retain their original optimizer and three-seed slate.'
    figures=[('junctions',draw_junctions()),('burden_fans',draw_failure_fans(burden)),
        ('prior_fans',draw_failure_fans(prior)),('critic_mechanism',draw_critic_mechanism(audit)),
        ('objective_diagnostics',draw_objective_diagnostics(burden,prior))]
    captions={
        'junctions':'Each episode has two physical steps; useful rewards and task information remain fixed as B grows.',
        'burden_fans':'Same algorithm, increasing burden. Mean +/- sample SD over seeds 0, 1, 2; shared linear scales.\nFull-support IDS and oracle also succeed (mean regret <= 0.694); all-method panels remain in the appendix.',
        'prior_fans':'Same algorithm, increasing cue reversal. Mean +/- sample SD; shared scales.\nCorrelated-prior IDS fails at strong reversal; full-support IDS survives. CEAC task-0 failures are retained.',
        'critic_mechanism':'Hard Undertow, B=64, all three seeds. Exact kernel KL and recorded actor-update decomposition.\nNo counterfactual performance claim. The right panel zooms to the first 12,000 interactions.',
        'objective_diagnostics':'Supporting mechanism checks. Left: population objective, not a convergence claim (dots are minimizers).\nMiddle: actual optimistic-Q saturation. Right: analytic cue-only log evidence under the frozen prior.'}
    with PdfPages(output/'report.pdf') as pdf:
        # Text pages avoid a plot-only PDF whose qualifications are easy to lose.
        import textwrap
        def prose(title, paragraphs, footer=None):
            fig=plt.figure(figsize=(11.7,8.3));fig.text(.065,.92,title,fontsize=20,weight='bold')
            y=.84
            for paragraph in paragraphs:
                wrapped=textwrap.fill(paragraph,width=115)
                fig.text(.065,y,wrapped,fontsize=11,va='top',linespacing=1.5)
                y-=.031*(wrapped.count('\n')+1)+.035
            fig.text(.065,.04,footer or 'CEAC: verified saved results | 144 burden + 90 prior cells | 3 held-out seeds | no new training',fontsize=9,color='#555555')
            pdf.savefig(fig);plt.close(fig)
        prose('When exploration follows the wrong signal',[
            'Question: does the same algorithm learn an easy problem, then deteriorate as irrelevant exploration opportunities or misleading cue strength increase? The figures follow STEERING’s within-algorithm variable-hardness visual logic.',
            'Physical burden: Undertow adds zero-reward actions; False Beacon widens a zero-reward chamber. IDAC can favor irrelevant occupancy; optimism pays to resolve a growing domain. At Undertow B=64, POMD stays uniform because its optimistic cost-Q values remain clipped at zero.',
            'Prior misspecification: hold the Bayesian prior fixed while reversing the task–cue correlation. The true joint law leaves support, although both task labels remain possible. Correlated-prior IDS fails at full reversal; product-support IDS survives.',
            'CEAC uses the requested reward critic, price-conditioned evidence critic and evidence-priced replay. In the original three-seed slate it avoids the large burden cost. Its task-0 actor-learning failures remain visible; correct evidence does not guarantee successful finite-budget policy optimization.',
            'Scope: controlled two-step junction variants, not the historical full grids. Every dose is a separate stationary run. Primitive controls observe trajectories; CEAC/IDS additionally know the candidate likelihood laws. The oracle additionally knows the truth. Shared figure scales and all held-out seeds are retained.'])
        for name,fig in figures:
            for suffix in ('png','pdf'):fig.savefig(output/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
            fig.set_size_inches(12.8,8.3)
            fig.subplots_adjust(bottom=.20,top=.87)
            fig.text(.5,.055,captions[name],ha='center',fontsize=10,linespacing=1.5)
            pdf.savefig(fig);plt.close(fig)
        prose('What the evidence critic contributed — and what it did not',[
            text['critic_findings'].split('\n\n')[0].replace('**',''),
            text['critic_findings'].split('\n\n')[1].replace('**',''),
            text['critic_findings'].split('\n\n')[2].replace('**',''),
            'Task-0 failure: saved updates reproduce the correct reference and finite critic values. In the failed d=0/seed-0 run, cumulative risky-minus-safe logit contributions are -7.28 from reward TD and +15.22 from evidence TD. A substantial actor optimization residual remains. All seeds and failures are retained; no held-out tuning or actor-only ablation was performed.',
            'Read report.md / the executable notebook for full numerical tables, replay-score checks, the complete assumptions and links to all-dose/all-method evidence. These diagnostics decompose actual updates; they do not estimate final performance under removal of the critic.'])
        if stability:
            from ceac_stability.report import combined_confirmation,draw_confirmation_learning,family_summary
            comparison=combined_confirmation(stability);summaries=family_summary(comparison)
            fig=plt.figure(figsize=(11.7,8.3))
            fig.text(.065,.92,'Actor stability: speed and final accuracy differ',fontsize=20,weight='bold')
            intro=('All settings retain both critics and evidence-priced replay. Original CEAC won the frozen endpoint-first ranking. '
                   'Guarded natural 0.03 also passed calibration with about 49% lower regret. A secondary comparison of every '
                   'non-reference passer was registered before inspecting fresh-seed outcomes; the primary selection did not change.')
            fig.text(.065,.84,textwrap.fill(intro,115),fontsize=11,va='top',linespacing=1.4)
            table_rows=[]
            for r in summaries:
                name='Cue reversal' if r['protocol']=='prior_shift' else r['kind'].replace('_junction','').replace('_',' ').title()
                table_rows.append([name,str(r['task']),f"{r['reference_cumulative_regret']:,.0f}",
                    f"{r['variant_cumulative_regret']:,.0f}",f"{r['difference_cumulative_regret']:,.0f} +/- {r['sd_difference_cumulative_regret']:,.0f}"])
            ax=fig.add_axes([.065,.33,.87,.34]);ax.axis('off')
            table=ax.table(cellText=table_rows,colLabels=['Family','Task','Original regret','Natural regret','Paired difference +/- SD'],
                cellLoc='center',colWidths=[.22,.08,.20,.20,.30],bbox=[0,0,1,1])
            table.auto_set_font_size(False);table.set_fontsize(10)
            for (r,c),cell in table.get_celld().items():
                cell.set_edgecolor('#cccccc')
                if r==0:cell.set_facecolor('#eeeeee');cell.set_text_props(weight='bold')
            notes=('Differences are natural minus original. Doses are averaged within each of five fresh seeds before computing SD. '
                   'Cue runs use 32,768 episodes; burden runs use 131,072. The floor adds a real cost. Schedules, critic scaling and '
                   'geometry change jointly, so their individual causal contributions are not isolated. External comparators used '
                   'different seeds and retain their original results. All doses and endpoints appear in the stability assessment.')
            fig.text(.065,.24,textwrap.fill(notes,115),fontsize=10.5,va='top',linespacing=1.4)
            fig.text(.065,.04,'Full CEAC optimizer study | 56 calibration + 260 confirmation cells | 5 fresh seeds',fontsize=9,color='#555555')
            pdf.savefig(fig);plt.close(fig)
            fig=draw_confirmation_learning(comparison)
            for ext in ('png','pdf'):fig.savefig(output/f'actor_stability_learning.{ext}',dpi=180,bbox_inches='tight')
            pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    pieces=[text['opening'],text['burden_setup'],'![The two controlled interventions](junctions.png)',
        '![Same algorithm over time as physical burden grows](burden_fans.png)',text['burden_findings'],
        text['prior_setup'],'![Same algorithm over time as the frozen prior becomes wrong](prior_fans.png)',text['prior_findings'],
        text['critic_setup'],'![Kernel evidence, exact critic contributions, actor behavior](critic_mechanism.png)',text['critic_findings'],text['limits'],
        '## Supporting evidence\n\n![Objective, optimism and prior diagnostics](objective_diagnostics.png)\n\n'+
        '[Complete burden results](../hardness_burden/assessment.md) · [Complete prior results](../hardness_prior/assessment.md) · '+
        '[Implementation](../../../docs/hardness_implementation.md) · [Comparator sources](../../../docs/comparator_provenance.md)\n\n'+
        'The executable [notebook](../../Undertow_FalseBeacon.ipynb) uses these same verified exports. '+
        'Regenerate with `python scripts/render_hardness.py --burden runs/ceac_primary_burden_20261111 --prior runs/ceac_primary_prior_20261111 --execute` from the repository root.']
    if stability:
        from ceac_stability.report import notebook_narrative
        pieces.insert(-1,notebook_narrative(stability).replace('evidence/actor_stability/','../actor_stability/')+
                      '\n\n![Registered baseline and maximum-burden learning comparisons](actor_stability_learning.png)')
    (output/'report.md').write_text('\n\n'.join(pieces)+'\n')
    manifest=json.loads((output/'manifest.json').read_text())
    manifest['report_sources']={str(p.relative_to(Path(__file__).resolve().parents[1])):digest(p) for p in (Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'scripts/render_hardness.py')}
    manifest['evidence_inputs']={name:digest(output.parent/name/'manifest.json') for name in ('hardness_burden','hardness_prior')}
    if stability:manifest['evidence_inputs']['actor_stability']=digest(output.parent/'actor_stability/manifest.json')
    manifest['files']={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
    atomic_json(output/'manifest.json',manifest)
    return text
