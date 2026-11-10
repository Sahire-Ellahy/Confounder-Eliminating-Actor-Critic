"""Render tables/figures from verified evidence; training is never called."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import hashlib
import numpy as np
from .evidence import atomic_json, digest, encoded, verify_run
from .pathology_report import strength_statistics, report_section
from .visualization import (LABELS, ENVIRONMENT_LABELS, make_figures, export_trace_statistics,
                            band_description, nuisance_sensitivity_rows, summarize_nuisance, RETIRED_METHODS)
from .benchmarks import study_cells


ACCESS = {
    "ceac": "Unknown realized mode; exact planning over declared models; certified reward-only execution",
    "ceac_actor": "Same CEAC design and model family; actor-only free episodes",
    "oracle": "Realized model known; allocation computed offline and fixed",
    "idac": "Primitive trajectories only; upstream tabular softmax actor and two critics",
    "ucb_q": "Primitive trajectories only; Jin et al. Algorithm 1 UCB-Hoeffding Q-learning",
    "pomd": "Primitive trajectories only; optimistic POMD softmax policy and learned empirical model",
    "ids": "Declared model family; full-support Bayesian IDS; full policy response",
    "ids_rare": "Same IDS, with shifted-mode prior 0.001",
    "ids_zero": "Same IDS, with shifted mode explicitly excluded from prior support",
}


def reconstruct_evidence_trace(run, folder):
    """Reconstruct full-history likelihoods; add derived traces only in memory.

    This checks cumulative sufficient statistics and reported policy credits.
    It does not claim that raw trajectories were resampled into critic updates.
    """
    result=run["result"];mode=result["true_mode"];method=result["method"]
    scores=np.zeros(2);counts=defaultdict(int);actor_updates=deployments=0
    max_error=0.;reported_credits=0
    metrics=run["metrics"]
    config=run["manifest"]["configuration"].get("ceac",{})
    threshold=np.log(1/config.get("confidence_delta",.05))
    with (Path(folder)/"trajectories.jsonl").open() as source:
        for index,line in enumerate(source):
            if index>=len(metrics):raise ValueError("Extra saved trajectory")
            episode=json.loads(line);metric=metrics[index]
            if (episode["index"]!=index or metric["episode"]!=index
                    or episode["policy_id"]!=metric["policy_id"]
                    or episode["behavior"]!=metric["behavior"]
                    or episode["context"]!=result["context"]):
                raise ValueError("Trajectory identity/behavior differs from metrics")
            for step in episode["transitions"]:
                likelihood=np.asarray(step["log_likelihood"],dtype=float)
                if likelihood.shape!=(2,) or not np.isfinite(likelihood).all():
                    raise ValueError("Invalid saved two-model likelihood")
                scores+=likelihood
            counts[episode["policy_id"]]+=1
            actor_updates+=int(episode["behavior"]=="actor")
            deployments+=int(episode["behavior"]=="certified_deploy")
            metric["cumulative_observed_llr"]=float(scores[mode]-scores[1-mode])
            metric["saved_episode_count"]=index+1
            if method not in {"ceac","ceac_actor"}:continue
            diagnostic=metric["diagnostics"]
            probability=np.exp(scores-scores.max());probability/=probability.sum()
            recorded=np.asarray(diagnostic["model_mixture"],dtype=float)
            if recorded.shape!=(2,) or not np.isfinite(recorded).all():
                raise ValueError("Invalid recorded CEAC mixture")
            error=float(np.max(np.abs(probability-recorded)))
            max_error=max(max_error,error)
            if error>1e-10:raise ValueError("CEAC evidence disagrees with saved full history")
            plausible=tuple(np.flatnonzero(scores.max()-scores<=threshold))
            if plausible!=tuple(diagnostic["plausible"]):
                raise ValueError("CEAC confidence set disagrees with saved full history")
            if diagnostic["actor_updates"]!=actor_updates or diagnostic["deployment_episodes"]!=deployments:
                raise ValueError("Update/deployment counts disagree with trajectory behavior")
            for identity,entry in diagnostic["ledger"].items():
                if entry["count"]!=counts[identity] or entry["statistical_assigned"]!=counts[identity]:
                    raise ValueError("Reported policy credit disagrees with physical trajectories")
                reported_credits+=1
            metric["replay_evidence_residual"]=error
    if sum(counts.values())!=len(metrics):raise ValueError("Saved history is incomplete")
    return {"benchmark":result.get("benchmark","legacy"),"environment":result["environment"],
            "context":result["context"],"mode":mode,"method":method,"seed":result["seed"],
            "episodes":len(metrics),"saved_policy_strata":len(counts),
            "cumulative_likelihood_reconstructed":True,
            "ceac_all_history_consistent":method in {"ceac","ceac_actor"},
            "maximum_probability_residual":max_error if method in {"ceac","ceac_actor"} else None,
            "reported_credit_entries_checked":reported_credits,
            "actor_updates":actor_updates if method in {"ceac","ceac_actor"} else None,
            "actor_admissions":result.get("diagnostics",{}).get("actor_admissions"),
            "certified_deployment_episodes":deployments}


def load_study(root, select=None):
    """Read verified cells; never train, copy or modify the source study."""
    root=Path(root).resolve()
    study=json.loads((root/"study.json").read_text())
    body=dict(study);identity=body.pop("identity")
    if hashlib.sha256(encoded(body).encode()).hexdigest()!=identity:
        raise ValueError("Study manifest digest mismatch")
    status=json.loads((root/"status.json").read_text())
    if status["status"]!="complete":raise ValueError("Cannot present an incomplete study as complete")
    config=study["configuration"]["study"]
    if "ceac" in config["methods"] and study["configuration"]["ceac"].get("deployment_rule")!="certified":
        raise ValueError("This is an archived actor-only CEAC study. Its original report is preserved; the deployment renderer will not relabel it as certified CEAC.")
    cells=list(study_cells(root,config));index=json.loads((root/"results.json").read_text())
    if len(cells)!=status["completed_cells"] or len(index)!=len(cells):
        raise ValueError("Study completion/index count mismatch")
    runs=[];manifests={};certificates=[]
    for position,cell in enumerate(cells):
        if select is not None and not select(cell):continue
        benchmark,kind,context,mode,seed,method,folder=cell
        run=verify_run(folder)
        if run["manifest"]["sources"]!=study["sources"]:raise ValueError("Mixed runtime sources")
        cfg=run["manifest"]["configuration"]
        if (cfg["environment"],cfg["context"],cfg["true_mode_evaluator_only"],cfg["method"],cfg["seed"],cfg["episodes"])!=(kind,context,mode,method,seed,config["episodes"]):
            raise ValueError("Run identity disagrees with study cell")
        if cfg.get("benchmark",{}).get("name","legacy")!=benchmark:
            raise ValueError("Run benchmark differs from study cell")
        if run["result"]!=index[position]:raise ValueError("Run index mismatch")
        run["history_audit"]=reconstruct_evidence_trace(run,folder)
        runs.append(run);manifests[str(folder.relative_to(root))]=digest(folder/"manifest.json")
        events=json.loads((folder/"audit_events.json").read_text())
        certificates.extend(e["design"] for e in events if e["type"]=="refresh")
    return study,runs,manifests,certificates


def validate_reference(primary, reference, methods):
    """Require matching scientific contracts before reusing any control curve."""
    if primary["sources"]!=reference["sources"] or primary["runtime"]!=reference["runtime"]:
        raise ValueError("Reference runtime/source identity differs")
    if primary["configuration"]["ceac"]!=reference["configuration"]["ceac"]:
        raise ValueError("Reference learner configuration differs")
    target=primary["configuration"]["study"];source=reference["configuration"]["study"]
    if target["episodes"]!=source["episodes"]:raise ValueError("Reference episode budget differs")
    for key in ("environments","contexts","modes","seeds"):
        if not set(target[key]).issubset(source[key]):raise ValueError("Reference is missing "+key)
    if not set(target.get("benchmarks",["legacy"])).issubset(source.get("benchmarks",["legacy"])):
        raise ValueError("Reference is missing benchmark settings")
    if not methods or not set(methods).issubset(source["methods"]):
        raise ValueError("Reference is missing requested methods")
    if set(methods)&set(target["methods"]):raise ValueError("Cannot duplicate a method from the primary study")
    source_specs=reference["configuration"].get("benchmark_specifications",[])
    if any(spec not in source_specs for spec in primary["configuration"].get("benchmark_specifications",[])):
        raise ValueError("Reference environment specifications differ")


def audit_prior_support(runs, model_audits):
    """Check support claims against every saved IDS metric and exact kernel audit."""
    models={(a["benchmark"],a["environment"],a["context"]):a for a in model_audits}
    rows=[]
    for run in runs:
        result=run["result"];method=result["method"]
        if not method.startswith("ids"):continue
        key=(result.get("benchmark","legacy"),result["environment"],result["context"])
        if key not in models:raise ValueError("Prior comparison is missing its exact model audit")
        audit=models[key];mode=result["true_mode"]
        safe_label="exploit" if result["environment"]=="deepsea" else "inner-bridge"
        safe=next(p for p in audit["policies"] if p["label"]==safe_label)
        gap=float(audit["optimal_returns"][mode]-safe["expected_return"][mode])
        prior=run["manifest"]["configuration"]["ids_prior"]
        metrics=run["metrics"]
        posterior=np.array([m["diagnostics"]["posterior"][1] for m in metrics])
        if prior==0:
            if not np.all(posterior==0):raise ValueError("Zero-support posterior was not absorbing")
            if any(m["diagnostics"]["prior_support"]!=[True,False] for m in metrics):
                raise ValueError("Zero-support run changed its declared prior support")
            if any(abs(m["diagnostics"]["regret"])>1e-12 or abs(m["diagnostics"]["information"])>1e-12 for m in metrics):
                raise ValueError("Excluded-support IDS is not subjectively certain")
            if any(abs(m["expected_model_information"])>1e-12 for m in metrics):
                raise ValueError("Selected old route is not observationally confounded")
            expected=gap*np.arange(1,len(metrics)+1)
            if not np.allclose([m["cumulative_pseudo_regret"] for m in metrics],expected,atol=1e-8,rtol=0):
                raise ValueError("Observed support-exclusion regret differs from exact linear prediction")
        elif np.any(posterior<=0) or not np.isfinite(posterior).all():
            raise ValueError("A positive prior was lost numerically")
        rows.append({"benchmark":key[0],"environment":key[1],"context":key[2],"mode":mode,
                     "method":method,"seed":result["seed"],"initial_mode1_prior":prior,
                     "final_mode1_posterior":float(posterior[-1]),
                     "cumulative_pseudo_regret":result["pseudo_regret"],
                     "last_window_expected_return":result["last_window_expected_return"],
                     "diagnostic_visits":result["event_counts"]["diagnostic"],
                     "zero_support_prediction":gap*len(metrics) if prior==0 else None,
                     "support_exclusion_verified":prior==0})
    return rows


def render(root, figures=True, pdf=False, reference=None, reference_methods=("ceac","oracle","ids"), band="sd"):
    root=Path(root).resolve()
    study,runs,manifests,certificates=load_study(root)
    config=dict(study["configuration"]["study"])
    own_count=len(runs);reference_record=None
    if reference is not None:
        reference=Path(reference).resolve()
        reference_identity=json.loads((reference/"study.json").read_text())
        validate_reference(study,reference_identity,reference_methods)
        def select(cell):
            b,k,c,m,s,method,_=cell
            return (b in config.get("benchmarks",["legacy"]) and k in config["environments"]
                    and c in config["contexts"] and m in config["modes"] and s in config["seeds"]
                    and method in reference_methods)
        saved,controls,control_manifests,control_certificates=load_study(reference,select)
        expected=(len(config.get("benchmarks",["legacy"]))*len(config["environments"])*len(config["contexts"])
                  *len(config["modes"])*len(config["seeds"])*len(reference_methods))
        if len(controls)!=expected:raise ValueError("Reference control grid is incomplete")
        runs.extend(controls);certificates.extend(control_certificates)
        config["methods"]=list(reference_methods)+config["methods"]
        reference_record={"root":str(reference),"study_identity":saved["identity"],
                          "study_sha256":digest(reference/"study.json"),
                          "run_manifest_hashes":control_manifests,"reused_runs":len(controls)}
    retired_runs = [r for r in runs if r["result"]["method"] in RETIRED_METHODS]
    runs = [r for r in runs if r["result"]["method"] not in RETIRED_METHODS]
    config["methods"] = [m for m in config["methods"] if m not in RETIRED_METHODS]
    results=[r["result"] for r in runs]
    groups=defaultdict(list)
    for r in results:groups[r.get("benchmark","legacy"),r["environment"],r["context"],r["true_mode"],r["method"]].append(r)
    if band not in {"sd","seed_range"}:raise ValueError("Unknown descriptive band")
    history_rows=[r["history_audit"] for r in runs]
    atomic_json(root/"history_audit.json",{"schema":1,"study_identity":study["identity"],
                "reference_study":reference_record,"rows":history_rows,
                "scope":"cumulative likelihood and reported credits; no learned-critic or replay-resampling ablation"})
    physical_rows=strength_statistics(runs)
    physical=bool(physical_rows)
    sensitivity=nuisance_sensitivity_rows(runs)
    sensitivity_summary=summarize_nuisance(sensitivity,band)
    artifacts=make_figures(root,runs,band=band,sensitivity=sensitivity) if figures else []
    artifacts.extend([root/"history_audit.json",export_trace_statistics(root,runs,band)])
    if physical:
        atomic_json(root/"pathology_statistics.json", dict(schema=1, study_identity=study["identity"],
            independent_unit="training_seed", band=band, paired_differences=True, rows=physical_rows))
        artifacts.append(root/"pathology_statistics.json")
    if sensitivity:
        atomic_json(root/"nuisance_sensitivity.json",{
            "schema":1,"study_identity":study["identity"],"reference_study":reference_record,
            "independent_unit":"training_seed","band":band,"baseline_nuisance_bins":4,
            "paired_subtraction_before_aggregation":True,
            "fixed_non_nuisance_configuration_checked":True,
            "rows":sensitivity,"summaries":sensitivity_summary})
        artifacts.append(root/"nuisance_sensitivity.json")
    certified=[r for r in results if r["method"]=="ceac" and r.get("deployment_certified")]
    oracle_lookup={(r.get("benchmark","legacy"),r["environment"],r["context"],r["true_mode"],r["seed"]):r for r in results if r["method"]=="oracle"}
    matched=[r for r in certified if (r.get("benchmark","legacy"),r["environment"],r["context"],r["true_mode"],r["seed"]) in oracle_lookup
             and abs(r["last_window_expected_return"]-oracle_lookup[r.get("benchmark","legacy"),r["environment"],r["context"],r["true_mode"],r["seed"]]["last_window_expected_return"])<1e-10]

    refined=any(b != "legacy" for b in config.get("benchmarks",["legacy"]))
    prior_comparison=any(m in config["methods"] for m in ("ids_rare","ids_zero"))
    heading="Undertow DeepSea and False Beacon Frozen Lake: "+("physical pathology strength" if physical else "prior support" if prior_comparison else "directed exploration")
    text=["# "+(heading if refined else "CEAC deployment and exploration comparison"),"",
          f"Verified **{len(results)} runs and {sum(r['episodes'] for r in results):,} environment episodes**. The source, configuration, frozen behavior policies, trajectories and artifact hashes are recorded per cell.","",
          "This is a paired, stationary-context finite-model experiment. It tests the deployment correction and exploration behavior; it does not establish an asymptotic coefficient-one theorem or recovery from unannounced temporal changes.","",
          "## Main finding","",
          "The corrected CEAC separates its pricing-policy actor from reward-only deployment. Once the likelihood evidence supports a policy that is reward-optimal for every plausible model, free episodes use that certified policy. Statistical and Graves–Lai debts retain priority.","",
          (f"**CEAC matches the offline oracle’s last-window expected return in {len(matched)} of {len(certified)} certified cells.** Endpoint equality is distinct from cumulative exploration cost." if not refined else "This study measures cumulative regret and diagnostic/nuisance visits under its recorded settings. Preflight verifies confounding and useful diagnostic routes. Sensitivity is assessed only for matched multilevel studies; no algorithm ranking is assumed in advance."),"",
          f"{len(certified)} CEAC cells finish with a valid deployment certificate. This certifies the deployed response on the model-confidence event; it does not mean the stochastic proposal actor has converged.","",
          "The offline oracle knows the realized mode, solves its full-policy Lai–Graves allocation once, and keeps that allocation fixed throughout execution. Its privilege is explicit; it is a benchmark, not an online learner.","",
          "## CEAC, certified deployment, and the offline oracle", "",
          "The plotted CEAC is the maintained **single episode-loop, exact-model specialization**. It retains actor/reward-critic/evidence-critic updates, policy-labeled trajectory history, target/calibration, design refreshes and cumulative execution debt. Exact finite-model evaluation and separation provide its design and deployment responses. The label does not mean that the learned actor alone produced the plotted behavior.", "",
          "| Property | CEAC (exact-model single loop) | Offline Graves–Lai oracle |",
          "|---|---|---|",
          "| Realized mode | Unknown; inferred from real trajectories | Supplied before the first episode |",
          "| Candidate-model laws | Known, with exact evaluation/planning | The same laws, plus the true-mode label |",
          "| Exploration allocation | Refreshed from confidence, calibration and evidence debt | Solved once for the true mode; normalized allocation stays fixed |",
          "| Exploitation | Reward-only response passing a confidence-set certificate | True-mode optimal policy from the outset |",
          "| Physical episodes | Statistical requests, then design debt, then certified deployment or actor | Fixed-design debt, otherwise true-mode optimal deployment |", "",
          "Certified deployment means the selected reward-only policy satisfies", "",
          "$$", r"\max_{m\in\mathcal C_t}\{J_m^\star-J_m(\widehat\pi_t)\}\le\tau,", "$$", "",
          r"where $\mathcal C_t$ is CEAC's trajectory-likelihood confidence set and $\tau=10^{-10}$ in these runs. Conditional on the true model belonging to that set, the same bound holds for its true-model value gap. The set has a per-run anytime guarantee under the declared model family; the certificate is not a guarantee under arbitrary model misspecification.", "",
          "Certification is checked without the true-mode label. It affects free-episode exploitation; statistical and design debts still run first. The oracle also executes scheduled experiments even though it knows the true mode, so its plotted finite-budget cost is the implemented fixed-design benchmark, not a finite-time lower bound every method must exceed.", "",
          "The exact tabular separator and Appendix D's reward-only response are part of the v90 finite-model framework. Connecting that certified response to free-episode execution is the documented implementation refinement. This preserves one environment episode loop while using exact optimization at refreshes; it does not establish actor/critic convergence or a literal empirical-only implementation of every pseudocode estimator.", "",
          "## Comparator access and scope","","| Method | Information and policy access |","|---|---|"]
    for method in config["methods"]:text.append(f"| {LABELS[method]} | {ACCESS[method]} |")
    if reference_record is not None:
        text += ["", f"This report combines **{own_count} new runs with {reference_record['reused_runs']} reused controls**. Controls were read from `{reference_record['root']}` after verifying equal runtime sources, environment definitions, learner settings, seeds and episode budgets. Their saved trajectories were not rerun or copied.", ""]
    if physical:
        environment_audit=json.loads((root/"environment_audit.json").read_text())
        if environment_audit["study_identity"] != study["identity"] or not environment_audit.get("strength_checks"):
            raise ValueError("Physical strength report requires its matched-domain preflight audit")
        artifacts.append(root/"environment_audit.json")
        text += report_section(physical_rows)
    if refined and not physical:
        environment_audit=json.loads((root/"environment_audit.json").read_text())
        if environment_audit["study_identity"] != study["identity"]:
            raise ValueError("Environment audit identity mismatch")
        artifacts.append(root/"environment_audit.json")
        text += ["", "## Controlled benchmark changes", "",
                 "**Undertow DeepSea**: spending effort in a noisy sonar/drift field loses progress without revealing the profitable gate choice. **False Beacon Frozen Lake** retains its shoreline geometry and varies the reed-observation alphabet. This sweep varies observable nuisance states, not the number of physical coves or candidate task modes.", "",
                 "The strong-signal control uses M=4 and route probability separation 0.25 from the midpoint. Weak-signal arms use separation 0.05 and M=4,16,64. Probe signals remain Bernoulli(0.15) versus Bernoulli(0.85). DeepSea uses terminal reward 0.5, signed challenge reward scale 0.5, and probe cost 0.02. Shores uses inner mean success 0.5, outer success 0.502 +/- separation, unit dock rewards, safe-dock reward 0.3 and deep-survey fee 0.02. The 0.002 offset compensates for two additional travel steps.", "",
                 "The refined DeepSea enables a mode-independent nuisance action at each depth and supplies one sonar-sweep seed policy to every bank method. FrozenLake keeps its original six seed routes. Seed-bank size is constant across M; CEAC still pays its one-episode-per-seed initialization. This experiment does not claim an M-fold uniform-route-search penalty, which would require expanding the route bank.", "",
                 f"The shared CEAC/oracle allocation mass bound is {study['configuration']['ceac']['mass_bound']:g}. It is enlarged because inverse weak-route KL exceeds the old bound of 20. Confidence level, statistical gate, debt accounting and actor update remain unchanged. Probe and nuisance events are evaluator annotations; they are never supplied to a learner as labels.", "",
                 "The preflight retains historical episodic finite-policy objective witnesses for audit only. They are not the continuing reset-process objective used by the new IDAC implementation and do not establish IDAC failure. Source-faithful controls require their own measured trajectories and convergence diagnostics.", "",
                 "| Setting | Environment | States | Seed/public bank | Oracle cost (mode 0 / 1) | Risky-route / optimal information cost (mode 0) | Zero-information OIR witness (mode 0) |",
                 "|---|---|---:|---:|---:|---:|---|"]
        for a in environment_audit["models"]:
            cost=a["oracle_design_costs"]
            witness=", ".join(a["zero_information_policies_beating_reward_optimum_in_oir_mode0"]) or "none in audited bank"
            text.append(f"| {a['benchmark']} | {a['environment']} | {a['reachable_states']} | {a['seed_policies']} / {a['public_bank_size']} | {cost[0]:.5f} / {cost[1]:.5f} | {a['risky_to_optimal_information_cost_ratio_mode0']:.1f} | {witness} |")
        text += ["", "These are paired stationary modes, not within-run shifts. There is no hidden change point or nuisance-only switching process. Any prior-support treatment is labeled separately below; full-support IDS remains the Bayesian control. The exact finite-model planning/deployment limitation from the earlier audit continues to apply.", ""]
        if "weak_signal_m64" in config.get("benchmarks",[]):
            text += ["", "## Weak signals and M=64: mode-0 comparison", "",
                     "Mode 0 makes optimistic risky-route trials costly. Mode 1 is the counter-control where those trials are reward-optimal; both are shown in the figures. Entries below are seed means, not significance claims.", "",
                     "| Method | DeepSea regret | Shores regret | DeepSea nuisance visits | Shores nuisance visits |",
                     "|---|---:|---:|---:|---:|"]
            for method in config["methods"]:
                values=[]
                for kind in ("deepsea","frozenlake"):
                    subset=[r for r in results if r.get("benchmark")=="weak_signal_m64" and r["environment"]==kind and r["true_mode"]==0 and r["method"]==method]
                    values.append(subset)
                mean=lambda rows,field: ("n/a" if not rows else f"{np.mean([r['event_counts']['nuisance'] if field=='nuisance' else r[field] for r in rows]):.3f}")
                text.append(f"| {LABELS[method]} | {mean(values[0],'pseudo_regret')} | {mean(values[1],'pseudo_regret')} | {mean(values[0],'nuisance')} | {mean(values[1],'nuisance')} |")
            text += ["", "Raw observed-state coverage includes elapsed time and nuisance symbols. Physical-location coverage is logged separately. A high count of observed states is not itself evidence of resolving the task mode.", "",
                     "Full mathematical specification and implementation adjustments: [benchmark design ledger](../../audits/ceac_undertow_design_20261110.md).", ""]

    if sensitivity:
        text += ["", "## Learning curves by experimental condition", "",
                 "Following the training-time sensitivity layout in [STEERING](https://arxiv.org/pdf/2301.12038), each method has a panel: x is training episode, y is cumulative pseudo-regret, and color identifies nuisance diversity M=4,16,64. These cached doses are symbol counts, not physical undertow strength or false-beacon activation probability. Shared linear axes preserve comparisons; coincident traces are labeled without artificial jitter.", "",
                 "The custom policy-bank UCB, model-confidence, uniform-bank and episodic policy-mixture OIR controls have been retired. Their historical data remain in the original study and source archive but are excluded from active figures and tables. Their results do not establish a failure of published coverage/optimism algorithms. Replacement IDAC, UCB Q-learning and optimistic POMD use distinct method identifiers; only actually recorded methods are plotted.", "",
                 "Matched endpoint summaries below are a supplemental audit. Pseudo-regret is measured against the true-mode optimal policy, not against the executed offline oracle's allocation. The current historical CEAC and full-support IDS controls are stable with nuisance diversity.", "",
                 "| Environment | Method | Regret at M=4 | Regret at M=64 | Paired added regret, mean +/- sample SD |",
                 "|---|---|---:|---:|---:|"]
        for kind,context in sorted({(r["environment"],r["context"]) for r in sensitivity}):
            for method in config["methods"]:
                lookup={(r["nuisance_bins"],r["metric"]):r for r in sensitivity_summary
                        if r["environment"]==kind and r["context"]==context and r["mode"]==0 and r["method"]==method}
                if (4,"cumulative_regret") not in lookup or (64,"cumulative_regret") not in lookup:
                    continue
                low,high=lookup[4,"cumulative_regret"],lookup[64,"cumulative_regret"]
                delta=lookup[64,"added_regret"]
                sd="unavailable" if delta["sample_sd"] is None else f"{delta['sample_sd']:.3f}"
                text.append(f"| {ENVIRONMENT_LABELS[kind]} (context {context}) | {LABELS[method]} | {low['mean']:.3f} | {high['mean']:.3f} | {delta['mean']:.3f} +/- {sd} |")
        text += ["", "Three seeds support descriptive comparisons only. The strong-signal M=4 control is excluded from this axis because it changes the task signal as well. Prior-support exclusion remains a separate assumption treatment; it is not relabeled as increasing nuisance intensity.", ""]

    if refined and any(m in config["methods"] for m in ("ids_rare","ids_zero")):
        prior_rows=audit_prior_support(runs,environment_audit["models"])
        atomic_json(root/"prior_support_audit.json",{"study_identity":study["identity"],
                    "reference_study":reference_record,"rows":prior_rows})
        artifacts.append(root/"prior_support_audit.json")
        text += ["", "## Prior support: delayed discovery versus permanent exclusion", "",
                 "The recorded IDS priors are listed below. The zero arm is an explicit misspecified Bayesian support treatment. CEAC and full-support IDS retain both candidate modes; this is an information-assumption difference, not a claim that CEAC can identify a model outside its own candidate family.", "",
                 "For positive observation likelihoods, Bayesian updating preserves zero prior mass:", "",
                 "$$", r"p_t(1)=\frac{p_0(1)L_t(1)}{p_0(0)L_t(0)+p_0(1)L_t(1)}=0\quad\text{if }p_0(1)=0.", "$$", "",
                 "The selected old-route trajectory law is identical across modes. With mode 1 realized, its positive value gap is Delta=e. IDS assigns this old route zero subjective regret and zero information; under the declared IDS zero/zero convention it never requests a diagnostic. The exact prediction is R_T=T Delta, using each recorded episode budget and kernel value gap. The audit checks the prediction at every episode, and also checks the mode-0 negative control. A small positive prior is different: it can recover, but diagnostic waiting times may exceed this finite budget.", "",
                 "| Setting | Environment | Mode | IDS prior | Seeds | Final posterior for mode 1 | Cumulative regret | Diagnostic visits |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        prior_groups=defaultdict(list)
        for row in prior_rows:
            prior_groups[row["benchmark"],row["environment"],row["mode"],row["initial_mode1_prior"]].append(row)
        for (benchmark,kind,mode,prior),rows in sorted(prior_groups.items()):
            avg=lambda field:float(np.mean([r[field] for r in rows]))
            text.append(f"| {benchmark} | {ENVIRONMENT_LABELS[kind]} | {mode} | {prior:g} | {len(rows)} | {avg('final_mode1_posterior'):.6g} | {avg('cumulative_pseudo_regret'):.4f} | {avg('diagnostic_visits'):.1f} |")
        text += ["", "This is a static out-of-support task mode. It does not yet test an unknown change point, change-aware Bayesian filtering, posterior resets or model-class expansion.", ""]
    ceac_history=[r for r in history_rows if r["method"]=="ceac"]
    if ceac_history:
        zero_updates=sum(r["actor_updates"]==0 for r in ceac_history)
        actor_admissions=sum(r["actor_admissions"] for r in ceac_history)
        error=max(r["maximum_probability_residual"] for r in ceac_history)
        text += ["", "## What these traces establish about evidence and replay", "",
                 f"Saved trajectories reconstruct the cumulative likelihood mixture and confidence sets in all {len(ceac_history)} CEAC cells, with maximum probability residual {error:.3g}. Reported policy-credit counts match physical trajectory counts at every episode. The audit is saved in `history_audit.json`.", "",
                 f"**{zero_updates} of {len(ceac_history)} CEAC runs made zero actor/evidence-critic updates; {actor_admissions} learned actor proposals were admitted.** Actor and evidence-value updates occur only on episodes labeled actor. Exact declared-model evaluation supplies diagnostic-design coefficients and response/deployment certificates. These experiments therefore support cumulative evidence accounting, directed experimental design and certified deployment; they do not isolate a causal performance gain from the learned TD evidence critic or replay resampling.", "",
                 "The replay object retains policy-labeled trajectories and cumulative likelihood sufficient statistics. Its ledger report exposes policies with quotas/statistical requests; it does not expose all internal buffer entries. The reconstruction verifies every saved episode and every reported credit, rather than interpreting missing ledger keys as discarded evidence. No frozen diagnostic/deployment trajectory is counted as an off-policy actor update.", "",
                 "A learned-critic claim needs an active learned-estimator lane and matched ablations of evidence pricing and cumulative-history use. Those experiments are not inferred from these cached curves.", ""]
    primitive_runs = [r for r in runs if r["result"]["method"] in {"idac", "ucb_q", "pomd"}]
    if primitive_runs:
        text += ["", "## Primitive learner update diagnostics", "",
                 "These runs validate the new primitive-action implementations. A short run is not a convergence claim or evidence of systematic failure under increasing pathology strength. In particular, the published POMD bonuses can leave every optimistic cost clipped to zero, so its softmax policy stays uniform.", "",
                 "| Setting | Environment | Mode | Method | Seed | Episodes | Actor / Q updates | Episodes with all POMD costs zero |",
                 "|---|---|---:|---|---:|---:|---:|---:|"]
        for run in primitive_runs:
            r=run["result"];d=r["diagnostics"]
            updates=d.get("actor_updates",d.get("updates",0))
            zero=(sum(m["diagnostics"]["all_optimistic_q_zero"] for m in run["metrics"])
                  if r["method"] == "pomd" else "n/a")
            text.append(f"| {r.get('benchmark','legacy')} | {r['environment']} | {r['true_mode']} | {LABELS[r['method']]} | {r['seed']} | {r['episodes']} | {updates} | {zero} |")
        text += ["", "IDAC updates both its cost critic and shadow critic at each reported actor update. UCB-Q counts updates on the analytically padded horizon; physical simulator steps remain separately charged. All three controls use the public finite domain and reward envelope, while CEAC/IDS additionally use declared candidate-model laws. That information-access difference must accompany any ranking.", ""]
    text += ["", "## Comparator provenance", "",
             f"Excluded {len(retired_runs)} historical custom-control runs from these comparisons. Raw run hashes and source archives remain unchanged. Active primitive-action controls are IDAC (the upstream tabular two-critic/softmax update), UCB Q-learning (Jin et al., 2018, Algorithm 1), and optimistic POMD with a softmax policy (Shani et al., 2020, Algorithm 2). Their exact source, update equations, reward normalization and reset/padding adapter are documented in [the comparator ledger](../../docs/comparator_provenance.md). No cached policy-mixture OIR or policy-bank UCB curve is relabeled as a new algorithm.", "",
             "## Time traces","",
             "Expected policy return is evaluated exactly in the realized candidate model **by the evaluator only**. Pseudo-regret sums its gap to the model-optimal policy. Expected model KL measures decision-relevant evidence, not raw state coverage. The return trace uses a causal 16-episode moving mean within each seed before aggregation; other traces are cumulative. Observed log-likelihood-ratio traces are reconstructed from saved trajectories and oriented toward the realized mode for evaluation only. They differ from expected model KL, which is an exact evaluator quantity.",""]
    text += [band_description(band)+f" There are {len(config['seeds'])} seeds per curve in this report. The episode index is not treated as additional independent replication. These are pointwise descriptive bands, not confidence intervals or simultaneous coverage guarantees. Nonnegative plot domains may clip a negative mean-minus-SD edge; exported values are untruncated.", "",
             "The compressed `trace_statistics.csv.gz` records seed count, mean, sample SD, range, band edges and smoothing for every exported time point.", ""]
    for path in artifacts:
        if path.suffix==".png":text.extend([f"![{path.stem.replace('_',' ')}](figures/{path.name})",""])
    text += ["## Endpoints","","| Setting | Environment | Context | Mode | Method | Seeds | Last-window expected return | Cumulative pseudo-regret | Primitive steps |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for (benchmark,kind,context,mode,method),rows in sorted(groups.items()):
        mean=lambda field:float(np.mean([r[field] for r in rows]))
        steps=float(np.mean([r.get("primitive_steps",float('nan')) for r in rows]))
        text.append(f"| {benchmark} | {kind} | {context} | {mode} | {LABELS[method]} | {len(rows)} | {mean('last_window_expected_return'):.6f} | {mean('pseudo_regret'):.6f} | {steps:.0f} |")
    text += ["","Budgets are matched in episodes. Episode lengths differ; the table also reports primitive environment steps. Planning uses the declared finite models and is not charged as physical self-play. This is not a wall-clock-matched comparison. Terminal success is not interchangeable with return: a confounded route can reach a dock reliably while missing the better objective.","",
             "## Deployment versus proposal actor","",
             "| Setting | Environment | Mode | Method | Seed | Actor episodes | Final actor return | Certified response return | Deployment episodes |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in results:
        if r["method"] in {"ceac","ceac_actor"}:
            text.append(f"| {r.get('benchmark','legacy')} | {r['environment']} | {r['true_mode']} | {LABELS[r['method']]} | {r['seed']} | {r['diagnostics']['actor_updates']} | {r['final_actor_expected_return']:.6f} | {r.get('final_deployment_expected_return',float('nan')):.6f} | {r['diagnostics'].get('deployment_episodes',0)} |")
    text += ["","Certified deployment trajectories remain policy-labeled evidence; they do not become off-policy actor-gradient updates. When included, the actor-only ablation shares initialization, design logic and random-stream identifiers, with histories diverging after policies differ.","",
             "## Mathematical correction","",
             "Given a likelihood-confidence set $\\mathcal C_t$, form a proposal from its confidence-restricted model estimate, then verify its reward gap in every plausible model:","",
             "$$","\\epsilon_t^{\\mathrm{deploy}}=\\max_{m\\in\\mathcal C_t}\\left[J_m^\\star-J_m(\\widehat\\pi_t)\\right].","$$","",
             "CEAC permits certified deployment only when this residual is at most the declared numerical tolerance. On the simultaneous confidence event, the same bound holds for the true-model deployment gap. If the confidence set changes, the response and certificate are refreshed. No true-mode label is used to construct this policy.","",
             "Appendix D defines a deployment response; Section 4 and Appendix G require the reward-only certificate. The Section 3 selector did not explicitly connect that response to free-episode execution. The implementation makes that connection explicit, using exact finite-model planning. It is a documented execution refinement, not evidence that actor SGD suddenly converged.","",
             f"Recorded {len(certificates)} CEAC/ablation refresh certificates. Maximum design primal/dual gap: {max((c['primal_dual_gap'] for c in certificates),default=0):.3g}; maximum full-policy column residual: {max((c['column_residual'] for c in certificates),default=0):.3g}.","",
             "Detailed specifications and remaining theorem-level limits: [correction ledger](../../audits/ceac_v90_corrections_20261110.md). The v90 manuscript and previous experiment artifacts remain unchanged.",""]
    (root/"report.md").write_text("\n".join(text))
    if pdf:
        if not shutil.which("pandoc") or not shutil.which("pdflatex"):
            raise RuntimeError("PDF export needs pandoc and pdflatex; Markdown and figures were already rendered")
        subprocess.run(["pandoc","report.md","--from=markdown+tex_math_dollars","--pdf-engine=pdflatex",
                        "-V","geometry:landscape,margin=0.5in","-V","fontsize=10pt","-V","colorlinks=true",
                        "--output=report.pdf"],cwd=root,check=True,capture_output=True,text=True)
        artifacts.append(root/"report.pdf")
    renderer=Path(__file__).resolve().parent
    bundle=root/"source_bundle.tar.gz"
    if bundle.exists():
        with tarfile.open(bundle,"r:gz") as archive:
            for name,checksum in study["sources"].items():
                content=archive.extractfile("ceac/"+name).read()
                if hashlib.sha256(content).hexdigest()!=checksum:
                    raise ValueError("Runtime source bundle disagrees with study identity")
    else:
        for name,checksum in study["sources"].items():
            if digest(renderer/name)!=checksum:
                raise ValueError("Current runtime differs from study; cannot reconstruct its source bundle")
        with tarfile.open(bundle,"w:gz") as archive:
            for name in study["sources"]:archive.add(renderer/name,arcname="ceac/"+name)
            archive.add(root/"study.json",arcname="study.json")
            archive.add(renderer.parent/"requirements-ceac.txt",arcname="requirements-ceac.txt")
            for test in sorted((renderer.parent/"tests/ceac").glob("*.py")):
                archive.add(test,arcname="tests/ceac/"+test.name)
    artifacts.append(bundle)
    atomic_json(root/"report_audit.json",{"schema":3,"study_identity":study["identity"],
                "reference_study":reference_record,
                "descriptive_band":band,"independent_unit":"training_seed","expected_return_smoothing":16,
                "run_manifest_hashes":manifests,"excluded_retired_runs":len(retired_runs),
                "excluded_method_ids":sorted(RETIRED_METHODS),"report_sha256":digest(root/"report.md"),
                "renderer_sources":{p.name:digest(p) for p in (renderer/"report.py",renderer/"visualization.py",renderer/"pathology_report.py")},
                "artifacts":{str(p.relative_to(root)):digest(p) for p in artifacts}})
    return root/"report.md"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--no-figures",action="store_true")
    parser.add_argument("--pdf",action="store_true")
    parser.add_argument("--band",choices=("sd","seed_range"),default="sd")
    parser.add_argument("--reference",type=Path,help="Reuse compatible completed controls, without rerunning training")
    parser.add_argument("--reference-methods",nargs="+",choices=tuple(LABELS),default=["ceac","oracle","ids"])
    args=parser.parse_args()
    print(render(args.input,figures=not args.no_figures,pdf=args.pdf,reference=args.reference,reference_methods=args.reference_methods,band=args.band))


if __name__=="__main__":main()
