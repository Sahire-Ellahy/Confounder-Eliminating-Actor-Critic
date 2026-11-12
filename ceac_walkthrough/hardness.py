"""Compact scientific assessment and an analytic support-shift design.

No learner, simulator rollout, or synthetic performance curve is run here.
The cue calculation is a proposed observation-channel experiment, not a
completed Undertow/Frozen Lake result.
"""
from pathlib import Path
import json
import math
import numpy as np
from ceac.visualization import LABELS, ENVIRONMENT_LABELS
from ceac.pathology_report import METHODS, _endpoint_grid


def summarize_current_sweep(bundle, mode=1, tolerance=.01):
    """Classify endpoints using a declared descriptive late-gap tolerance."""
    selected=[r for r in bundle['pathology_statistics'] if r['mode']==mode]
    summaries=[]
    for (kind,context,_),grid in sorted(_endpoint_grid(selected).items()):
        levels=sorted({rho for _,rho,_ in grid})
        seeds=sorted({seed for _,_,seed in grid})
        for method in METHODS:
            if not any(m==method for m,_,_ in grid):continue
            low=[grid[method,levels[0],s] for s in seeds]
            high=[grid[method,levels[-1],s] for s in seeds]
            a=sum(r['last_gap']<=tolerance for r in low)
            b=sum(r['last_gap']<=tolerance for r in high)
            verdict=('passes_both_ends' if a==b==len(seeds) else
                     'fails_both_ends' if a==b==0 else
                     'observed_pass_to_fail' if a==len(seeds) and b==0 else
                     'mixed_seed_outcomes')
            summaries.append(dict(environment=kind,context=context,mode=mode,method=method,
                mild_rho=levels[0],strong_rho=levels[-1],seeds=len(seeds),
                mild_late_gap=float(np.mean([r['last_gap'] for r in low])),
                strong_late_gap=float(np.mean([r['last_gap'] for r in high])),
                mild_passes=a,strong_passes=b,classification=verdict,
                added_regret=float(np.mean([h['regret']-l['regret'] for l,h in zip(low,high)])),
                pomd_zero_fraction=float(np.mean([r['pomd_zero_fraction'] for r in low+high]))))
    return summaries


def bernoulli_kl(p,q):
    if not 0<p<1 or not 0<q<1:raise ValueError('Interior Bernoulli parameters required')
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))


def support_shift_design(levels=(0.,.25,.5,.75,1.), cue_accuracy=.9, diagnostic_accuracy=.85):
    """Exact channel audit for truth theta=1 and a frozen correlated joint prior.

    Frozen prior: mass 1/2 on (theta=0,q=1-a) and (theta=1,q=a).
    Test truth: (theta=1,q=a-(2a-1)*d). Both task modes retain positive
    prior mass; off-correlation JOINT laws are absent from its support.
    A product class includes either task at each registered nuisance q.
    """
    if not .5<cue_accuracy<1 or not .5<diagnostic_accuracy<1:
        raise ValueError('Accuracy must be strictly between .5 and 1')
    levels=tuple(float(d) for d in levels)
    if not levels or len(set(levels))!=len(levels) or any(not 0<=d<=1 for d in levels):
        raise ValueError('Distinct shift doses in [0,1] required')
    a=cue_accuracy; diagnostic_kl=bernoulli_kl(diagnostic_accuracy,1-diagnostic_accuracy)
    log_ratio=math.log(a/(1-a))
    rows=[]
    for d in levels:
        q=a-(2*a-1)*d
        # One independent cue plus one diagnostic is an identifying witness.
        closest_joint_kl=min(bernoulli_kl(q,a),bernoulli_kl(q,1-a)+diagnostic_kl)
        rows.append(dict(shift=d,true_task=1,true_cue_probability=q,
            fixed_prior_true_joint_supported=math.isclose(q,a,abs_tol=1e-12),
            fixed_prior_task1_mass=.5,
            fixed_prior_expected_log_odds_per_cue=(2*q-1)*log_ratio,
            closest_fixed_prior_joint_witness_kl=max(0.,closest_joint_kl),
            product_class_nuisance_kl_against_opposite_task_same_q=0.,
            reliable_diagnostic_kl=diagnostic_kl,
            all_binary_observations_have_positive_probability=True))
    return dict(status='analytic_channel_design_only_no_algorithm_runs',
        levels=list(levels),cue_accuracy=a,diagnostic_accuracy=diagnostic_accuracy,
        frozen_joint_prior=[dict(task=0,q=1-a,mass=.5),dict(task=1,q=a,mass=.5)],
        product_class_models=2*len(levels),rows=rows)


def draw_support_design(plt=None):
    if plt is None:import matplotlib.pyplot as plt
    design=support_shift_design();rows=design['rows']
    x=[r['shift'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].plot(x,[r['true_cue_probability'] for r in rows],'-o',color='#0058c7',lw=2)
    axes[0].axhline(.9,color='#666666',ls='--',label='Frozen prior: task 1 implies q=0.9')
    axes[0].set(xlabel='Cue–task relationship shift d',ylabel='True P(beacon says task 1)',
                title='Same task; increasingly misleading beacon',xticks=x,ylim=(0,1))
    axes[0].legend(loc='lower left',fontsize=9,frameon=False)
    axes[1].plot(x,[r['fixed_prior_expected_log_odds_per_cue'] for r in rows],'-o',color='#c90078',lw=2)
    axes[1].axhline(0,color='#333333',lw=1)
    axes[1].axvline(.5,color='#777777',ls=':',lw=1)
    axes[1].set(xlabel='Cue–task relationship shift d',ylabel='Expected log odds for the true task\nper beacon observation (nats)',
                title='Frozen Bayesian likelihood becomes misleading',xticks=x)
    axes[1].text(.03,.08,'Below zero: repeated cues favor the wrong task',transform=axes[1].transAxes,fontsize=9)
    for axis in axes:axis.grid(alpha=.2);axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Proposed mechanism: correlated prior support breaks under a cue shift',fontsize=15,fontweight='bold')
    fig.text(.5,.02,'Exact observation-channel calculation. No IDS/CEAC performance or easy-to-hard transition has been measured here.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.92))
    return fig


def assessment_markdown(bundle, mode=None):
    if mode is None:
        modes={r['mode'] for r in bundle['pathology_statistics']}
        mode=1 if 1 in modes else min(modes)
    rows=summarize_current_sweep(bundle,mode)
    history=[r for r in bundle['history_audits'] if r['method']=='ceac']
    zero=sum(r['actor_updates']==0 for r in history)
    transitions=sum(r['classification']=='observed_pass_to_fail' for r in rows)
    conclusion=("**No all-seed pass-to-fail learning transition is observed under the declared late-gap criterion.** "
                if transitions==0 else
                f"**{transitions} environment/method pairs show an all-seed pass-to-fail transition under the declared criterion.** ")
    text=['# Is the current pathology parameter a validated hardness axis?', '',
        conclusion + 
        'Rho measures physical diversion probability. It was supplied to the candidate models at every dose; '
        'the full-support Bayesian model remained correctly specified. It must not be relabeled as a measured hardness score.', '',
        '## What the saved runs say', '',
        f'Stationary mode {mode}; seed means below. A descriptive pass means last-32-episode expected value gap ≤0.01. '
        f'{transitions} environment/method pairs pass on every mild-dose seed and fail on every strong-dose seed. '
        'The mild endpoint is not called easy merely because its parameter is small.', '',
        '| Environment | Method | Late gap: mild → strong | Seeds passing: mild → strong |',
        '|---|---|---:|---:|']
    for r in rows:
        text.append(f"| {ENVIRONMENT_LABELS[r['environment']]} | {LABELS[r['method']]} | "
            f"{r['mild_late_gap']:.4f} → {r['strong_late_gap']:.4f} | "
            f"{r['mild_passes']}/{r['seeds']} → {r['strong_passes']}/{r['seeds']} |")
    primitive=[r for r in rows if r['method'] in {'idac','pomd','ucb_q'}]
    saturated=[r for r in rows if r['method']=='pomd' and r['pomd_zero_fraction']==1.]
    text += ['',
        ('Every listed primitive control fails the late-gap criterion at the mild endpoint. '
         'This is not a demonstrated easy baseline.' if primitive and all(r['mild_passes']==0 for r in primitive)
         else 'Use the mild-end pass counts to check easy-case learning; a small dose alone does not certify it.'), '',
        'The excluded-task prior is a fixed support treatment at every rho. Environmental induction of '
        'joint-prior misspecification is a separate, untested mechanism in this sweep.', '',
        f'POMD has all optimistic costs zero in every episode at both endpoints in {len(saturated)} '
        'listed environment/context cases. In those cases its initial uniform policy is unchanged; '
        'that budget regime is not a calibrated easy-case learning baseline.', '',
        f'**Check learned-critic attribution:** {zero}/{len(history)} supplied CEAC histories have zero actor/critic updates. '
        'These runs use the exact-model planning, likelihood and deployment specialization; '
        'they do not isolate benefits of the learned evidence critic or replay estimator.', '',
        '## Two replacement experiments', '',
        '1. **Exploration burden:** hold the useful task, reward gap and diagnostic quality fixed; increase the number/depth '
        'of irrelevant physical opportunities. First demonstrate that the named controls learn on the zero-distraction task. '
        'The prepared chamber study has not passed that calibration: its B=0 case retains the already difficult shoreline.', '',
        '2. **Joint-prior misspecification:** freeze a Bayesian cue–task model that is correct at shift d=0, then change only '
        'the environmental cue relationship. Both task labels keep positive marginal prior mass; the true joint law leaves '
        'the prior support. Compare against product-support Bayesian IDS and a CEAC class containing the shifted nuisance laws.', '',
        'A concrete channel uses prior atoms (task 0,q=0.1) and (task 1,q=0.9), each with mass 1/2. '
        'For true task 1, set q(d)=0.9−0.8d. The fixed model interprets every cue as task evidence, with expected log-odds '
        '(0.8−1.6d) log(9); the direction reverses beyond d=0.5. This is an analytic design calculation, not a measured IDS learning curve.', '',
        'CEAC needs a product/composite model class in which an opposite task can have the same q. Those nuisance cues then have '
        'zero KL against a relevant matched-nuisance alternative, while an independent reliable diagnostic still distinguishes tasks. '
        '**This is an implementation prerequisite:** the current CEAC/IDS planning stack is specialized to two models. '
        'CEAC cannot claim robustness to a truth omitted from its own class, and a full-product-support IDS control may remain robust.', '',
        '## The figure that would answer the question', '',
        'One panel per algorithm; x = training interactions; y = excess policy loss or cumulative regret; '
        'color = low/medium/high validated pathology. Show the same algorithm progressing from successful learning '
        'to delayed or failed learning. Use a second compact endpoint-vs-pathology panel only if it adds information. '
        'Keep support shifts separate from exploration burden; keep full-support IDS and the privileged oracle visible. '
        'Place diagnostics and all additional modes in the linked appendix.', '',
        '## Before the next full sweep', '',
        '- Validate an actually learnable zero-distraction case using calibration seeds; freeze budgets and hyperparameters before evaluation seeds.',
        '- Check optimizer activity and bonus saturation; poor learning at every dose does not qualify as an easy-to-hard result.',
        '- Audit true joint-model membership separately from positive observation probabilities and task-label prior mass.',
        '- Keep informative-route value/KL and reward gap fixed; disclose changes in reachable domain and prescribed bonuses.',
        '- For the learned CEAC mechanism, require active actor/critic learning and matched evidence/replay ablations; retain exact-model CEAC as a separate reference.',
        '- Publish flat curves, CEAC losses and successful full-support IDS results as observed. No ranking is guaranteed by the design.', '']
    return '\n'.join(text).replace('≤',r'$\le$ ').replace('→',r'$\to$').replace('−','-')
