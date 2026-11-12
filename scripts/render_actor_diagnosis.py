#!/usr/bin/env python3
"""Render the saved-trajectory actor diagnosis; no training or ablation."""
from pathlib import Path
import sys,json,gzip,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ceac_hardness.run import digest,atomic_json


def main():
    output=ROOT/'audits/ceac_actor_failure_20261112'
    manifest=json.loads((output/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if Path(name).name!=name or digest(output/name)!=expected:raise ValueError('Diagnostic artifact mismatch')
    with gzip.open(output/'diagnostics.json.gz','rt') as f:data=json.load(f)
    cells=data['cells'];examples=sorted([r for r in cells if r['task']==0 and r['dose']==0],key=lambda r:r['seed'])
    failed=examples[0];end=failed['trace'][-1];common=end['states'][0]
    crossing=next(r for r in failed['crossings'] if r['state']==1)
    initial=failed['trace'][0];w0=initial['price_sum']
    risky_sd=.3*abs(1-2*w0*np.log(9))
    probe_sd=np.sqrt(.15*.85)*2*w0*np.log(.85/.15)
    bad=[r for r in cells if r['final_augmented_gap']>.01]
    max_good=max(r['final_augmented_gap'] for r in cells if r['final_augmented_gap']<=.01)
    fig,axes=plt.subplots(2,2,figsize=(11.8,7.2))
    colors=['#b3344b','#2378a8','#26806a']
    for r,color in zip(examples,colors):
        x=[t['actor_updates'] for t in r['trace']]
        y=[t['states'][0]['probabilities'][1] for t in r['trace']]
        axes[0,0].plot(x,y,color=color,lw=1.8,label=f"Seed {r['seed']}")
    axes[0,0].set(xlabel='Fresh actor episodes (first 6,000)',ylabel='Risky-action probability at common cue',
        title='A. Early concentration can persist or recover',xlim=(0,6000),ylim=(-.03,1.03))
    axes[0,0].legend(frameon=False,loc='center right')
    x=np.array([t['actor_updates'] for t in failed['trace']])
    for a,label,color in ((0,'Safe','#777777'),(1,'Risky','#b3344b'),(2,'Probe','#26806a')):
        axes[0,1].plot(x,[t['states'][0]['q_augmented'][a] for t in failed['trace']],label=label,color=color,lw=1.8)
    axes[0,1].set(xlabel='Fresh actor episodes',ylabel='Exact expected reward + priced evidence',title='B. Probe is always the better priced action')
    axes[0,1].legend(frameon=False,loc='center right')
    actual=np.array([t['states'][0]['actual_contrast'] for t in failed['trace']])
    expected=np.array([t['states'][0]['expected_contrast'] for t in failed['trace']])
    axes[1,0].plot(x,actual,color='#333333',label='Recorded update sum',lw=1.8)
    axes[1,0].plot(x,expected,color='#2378a8',label='Conditional expected update sum',lw=1.8)
    axes[1,0].plot(x,actual-expected,color='#9d4d93',label='Centered sampling noise',lw=1.6,ls='--')
    axes[1,0].axhline(0,color='#aaaaaa',lw=.6)
    axes[1,0].set(xlabel='Fresh actor episodes (first 4,000)',ylabel='Cumulative probe − risky logit update',
        title='C. Seed 0: early noise suppresses the probe',xlim=(0,4000))
    axes[1,0].legend(frameon=False,fontsize=8,loc='upper right')
    xx=np.arange(3);w=.34
    for dx,key,label,color in ((-w/2,'final_actor_reward_gap','Reward-only gap','#b3344b'),(w/2,'final_augmented_gap','Priced-objective gap','#2378a8')):
        values=[r[key] for r in examples]
        axes[1,1].bar(xx+dx,values,width=w,color=color,label=label)
        for xp,v in zip(xx+dx,values):axes[1,1].text(xp,v+.008,f'{v:.5f}' if v<.001 else f'{v:.3f}',ha='center',fontsize=8)
    axes[1,1].set_xticks(xx,['Seed 0','Seed 1','Seed 2'])
    axes[1,1].set(ylabel='Final current-actor optimality gap',title='D. Objective mismatch ≠ optimization failure',ylim=(0,.44))
    axes[1,1].legend(frameon=False,fontsize=9,loc='upper right')
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);ax.set_axisbelow(True);ax.title.set_fontsize(11)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/f'actor_failure.{suffix}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    all_band=failed['bands']['all_actor'];late=failed['bands']['last_4096_episodes']
    w=failed['final_replay']['replay_price_sum'];exact=failed['final_replay']['exact_kernel_price_sum']
    text=f'''# Why did the CEAC actor fail?

**The large task-0 failures are not caused by a missing or delayed intrinsic reward.** The observed mechanism is early stochastic concentration, weak correction under a nearly deterministic softmax policy, and a separate persistent evidence premium that makes even a well-optimized actor prefer probing to pure reward exploitation.

This is a read-only audit of all **{len(cells)} CEAC prior-shift cells / {sum(r['checked_updates'] for r in cells):,} fresh actor episodes**. Every saved actor-policy identity is reconstructed. No training, actor-only ablation, altered trajectory or post-test tuning is used. The raw studies and numerical kernels are unchanged.

![Saved-policy diagnosis, task 0 at d=0](actor_failure.png)

A shows all three seeds at the cue state visited with probability 0.9; B and C diagnose seed 0. C's conditional expectations are evaluated along the *recorded stochastic policy history*, not the trajectory of a hypothetical deterministic-gradient learner. D compares the final current actor; the published endpoint averages the last 256 behavior episodes, including diagnostics, and is a different statistic.

## 1. Evidence is dense; corrective action samples become rare

The only action choice is terminal, immediately after a forced cue. Risky and probe outcomes each supply a nonzero task likelihood ratio. The safe action supplies zero. There is no long-horizon credit-assignment problem at this decision.

For task 0 / d=0 / seed 0, **{100*all_band['nonzero_fraction']:.3f}% of actor decisions** have nonzero priced evidence. This is already {100*failed['bands']['first_512_actor']['nonzero_fraction']:.1f}% in the first 512 actor episodes and 100% in the final 4,096 episode window. Mean absolute terminal evidence over the run is {all_band['mean_abs_evidence']:.3f}.

Yet the final common-cue policy is:

| Action | Probability | Expected external reward | Expected priced evidence | Combined value |
|---|---:|---:|---:|---:|
'''
    for a,name in enumerate(('Safe','Risky','Probe')):
        text+=f"| {name} | {common['probabilities'][a]:.8f} | {common['q_reward'][a]:.2f} | {common['q_evidence'][a]:.6f} | {common['q_augmented'][a]:.6f} |\n"
    text+=f'''
The probe is clearly better under the current objective, but is sampled with probability only {common['probabilities'][2]:.6f}. In the final 4,096 episode window, the actor visited this cue state **{sum(late['visits'][1]):,} times and sampled the probe {late['visits'][1][2]} times**. There is abundant intrinsic signal from the action it already favors; observations of the corrective action have become rare.

## 2. The current evidence price never makes risky the best action

Let w be the sum of active task-changing evidence prices. Cue evidence is action-independent and can be omitted from the terminal action comparison. The exact task-0 action values are:

```
Q_safe(w)  = 0.50
Q_risky(w) = 0.10 + 1.757780 w
Q_probe(w) = 0.48 + 1.214221 w
```

The coefficients are KL divergences: `KL(Bernoulli(0.1) || Bernoulli(0.9))` and `KL(Bernoulli(0.15) || Bernoulli(0.85))`. Probe beats safe when w > 0.016471, and probe beats risky when w < 0.699096.

The failed run's price stays between **{failed['price_range'][0]:.6f} and {failed['price_range'][1]:.6f}**. Probe is therefore the unique optimal terminal action throughout the actor run; its advantage over risky ranges from {failed['probe_minus_risky_augmented_range'][0]:.6f} to {failed['probe_minus_risky_augmented_range'][1]:.6f}. There is no probe-to-risky target switch to explain the collapse. The same ordering holds in all 15 task-0 cells.

## 3. Early sampling noise, then softmax saturation

The evidence reward is signed and noisy, even though it is dense. At the first actor price, the expected combined returns are {initial['states'][0]['q_augmented'][1]:.3f} for risky and {initial['states'][0]['q_augmented'][2]:.3f} for probe, but their single-observation standard deviations are {risky_sd:.3f} and {probe_sd:.3f}. The mean advantage is about 0.07 against much larger outcome noise. A constant actor step of 0.4 can therefore create large early preference changes.

By fresh actor update **{crossing['actor_updates']}**, seed 0 assigns risky probability {crossing['probabilities'][1]:.6f} at the common cue, while probe remains better by {crossing['probe_advantage']:.6f}. The probe-minus-risky logit change at that point is {crossing['actual_contrast']:.4f}; its conditional expected-update sum along the saved history is {crossing['expected_contrast']:.4f}. The difference, **{crossing['actual_contrast']-crossing['expected_contrast']:.4f}**, is the accumulated centered sampling-noise term. This is an exact decomposition, not a significance test or a counterfactual performance claim.

For this terminal decision, an arbitrary frozen state baseline b cancels from the conditional mean actor update. At temperature 1:

```
E[delta_theta_a | state, history]
    = alpha * pi(a) * (Q_beta(a) - sum_b pi(b) Q_beta(b)).
```

Consequently, two distinct effects occur. A rare good action has a tiny update because of the factor pi(a). Also, when both risky and probe beat safe, suppressing safe can initially increase the more-probable risky action's logit faster than the probe's—even though probe has the highest Q. The negative conditional-drift segment in panel C is compatible with an unbiased ascent direction for the *whole* objective. A higher Q does not guarantee monotonically improving pairwise softmax odds.

At the failed endpoint, probe's expected logit increment is only **{common['expected_step'][2]:.8f} per common-state actor visit**. The expected probe-minus-risky logit increment is {common['expected_contrast_step']:.8f}, against conditional per-step standard deviation {common['conditional_step_sd']:.6f}. The conditional priced-objective gap is still {common['augmented_gap']:.6f}. These are local moments at the recorded endpoint, not an extrapolated recovery-time prediction.

At fixed prices, the objective here is linear in each state's action probabilities. It has no suboptimal interior local maximum: the finite-budget problem is a near-boundary softmax plateau with rare corrective samples. Seed 2 also temporarily exceeds 99% risky probability, then recovers; the plateau is not absorbing or proof of inevitable failure.

## 4. Critic error affects variance here, not mean direction

The terminal reward/evidence critics act only as state baselines. The preceding cue action is forced and has zero policy score. Thus errors in these critic values cannot systematically bias the conditional actor gradient in this particular junction. This conclusion need not extend to multi-step decisions with bootstrapped action-dependent successors.

The failed run's first-512-actor reward/evidence critic RMSEs against exact current-policy values are {failed['bands']['first_512_actor']['reward_critic_rmse']:.3f}/{failed['bands']['first_512_actor']['evidence_critic_rmse']:.3f}. Both critics remain finite. Critic errors and changing prices can alter variance; this audit does not identify their separate causal contribution to collapse. The actor step is 0.4, the reward-critic step 0.12. With a single price w, the evidence-value update has effective step `0.12*w^2` (roughly 0.025–0.039 here), so the configuration does not impose a faster-critic/slower-actor regime. That is a justified parameter-sensitivity hypothesis, not a demonstrated fix.

## 5. The small residual is a different problem: conservative replay pricing

Successful task-0 actors converge close to the probe's priced-objective optimum. The probe pays 0.48 while safe pays 0.50, so roughly 0.02 external-reward loss is expected even with an accurate optimizer under these prices. Calling that entire residual an actor-learning failure would be incorrect.

For seed 0, the final replay-derived price is **{w:.6f}**, versus **{exact:.6f}** from an evaluator-only known-kernel Graves–Lai solve. This is a reference calculation, not a replacement behavior policy. The final bank policies have only **{failed['final_replay']['counts']} compatible episodes each**, despite 29,000 fresh actor episodes. Every changing actor policy has its own label; its data cannot simply be pooled into another frozen policy's confidence stratum.

For the probe policy, the empirical return is 0.48 but its lower confidence bound is about 0.321. Against the active row, empirical trajectory evidence is about 1.303 but its lower bound is about 0.390. The conservative cost/evidence ratio is consequently `(0.5 - 0.321)/0.390 ≈ 0.459`. This directly explains the elevated dual price. More refreshes without more compatible evidence do not automatically shrink those bounds.

## 6. All held-out cells remain in the diagnosis

There are **{len(bad)} cells with a final priced-objective gap above 0.01**; the other {len(cells)-len(bad)} have gaps at most {max_good:.6f}. The threshold only labels this audit table; it excludes no run. The large failures are task 0 / seed 0 at d=0, 0.75 and 1.

| Dose | Task | Seed | Final actor reward gap | Final actor priced-objective gap | Original late behavior gap |
|---|---:|---:|---:|---:|---:|
'''
    rows=[]
    for r in cells:
        row={k:r[k] for k in ('dose','task','seed','final_actor_reward_gap','final_augmented_gap','reported_late_behavior_gap')}
        rows.append(row)
        text+=f"| {r['dose']:g} | {r['task']} | {r['seed']} | {r['final_actor_reward_gap']:.6f} | {r['final_augmented_gap']:.6f} | {r['reported_late_behavior_gap']:.6f} |\n"
    text+='''
## Consequences for the next experiment

1. Test actor/critic step-size separation using independent calibration seeds, preserving both critics and evidence-priced replay. Measure early action concentration, conditional gradient noise, and final priced-objective gap. This targets the large failures.
2. Separately test certified replay precision: compatible sample allocation or a tighter valid confidence construction. Measure price inflation and pure-reward versus priced-objective gaps. This targets the small persistent probing cost.
3. Keep original held-out results intact, fix the revised configuration before a new evaluation, and retain both tasks. A stronger generic intrinsic bonus is not supported by this diagnosis; it can further favor persistent information gathering. Any entropy floor, projection, or altered behavior schedule would be an explicit algorithmic change needing its own specification.

The existing evidence supports these mechanisms but does not establish which proposed change improves performance. No ablation or repair run has been launched.

Reproduce from the repository root with `python scripts/diagnose_actor_failures.py` followed by `python scripts/render_actor_diagnosis.py`. The first script verifies numerical sources, raw hashes and every actor-policy identity. The second verifies the diagnostic export before rendering. Conditional-gradient means were checked against finite differences, and variances against exhaustive action/outcome enumeration.
'''
    (output/'assessment.md').write_text(text)
    with (output/'endpoints.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest['report_source_sha256']=digest(Path(__file__))
    manifest['files']={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}
    atomic_json(output/'manifest.json',manifest)
    print(f'Rendered complete actor diagnosis for {len(cells)} cells at {output}')

if __name__=='__main__':main()
