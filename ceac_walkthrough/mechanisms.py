"""Analysis of saved CEAC episodes, without training or changing any policy.

The replay reconstruction verifies every actor-policy identity and decomposes
its *recorded* update. It is not an estimate of an intervention on the critic.
This module deliberately lives outside the frozen numerical-source packages.
"""
from collections import Counter
from pathlib import Path
import gzip
import json
import math
import numpy as np
from scipy.optimize import minimize_scalar
from ceac_hardness.run import ROOT, digest, atomic_json


def occupancy_ratio(p, breadth, kappa=.1):
    """Exact population C/(H+kappa), optimal-useful/uniform-nuisance mixture."""
    p = np.asarray(p, dtype=float)
    if breadth == 0 and np.any(p != 0):
        raise ValueError('Nuisance action is unavailable at B=0')
    if np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid mixture')
    from scipy.special import xlogy
    binary = -xlogy(p, p) - xlogy(1-p, 1-p)
    entropy = math.log(2) + .5*(binary + p*math.log(max(1, breadth)))
    cost = 1.01 - .45*(1-p)
    return cost/(entropy+kappa)


def population_nuisance(breadth):
    if not breadth:
        return 0.
    fit = minimize_scalar(lambda p: occupancy_ratio(p, breadth), bounds=(0, 1), method='bounded')
    candidates = [0., float(fit.x), 1.]
    return min(candidates, key=lambda p: occupancy_ratio(p, breadth))


def decompose_update(actor, episode, policy):
    """Use frozen, pre-update TD errors saved by TwoCriticActor.update."""
    reward = np.zeros_like(actor.logits)
    evidence = np.zeros_like(actor.logits)
    for step, (state, dr, di, _) in zip(episode.transitions, actor.last_update['errors']):
        if step.state != state:
            raise ValueError('Misaligned TD errors')
        score = -policy.probabilities[state].copy()
        score[step.action] += 1
        score *= actor.actor_rate / actor.temperature
        reward[state] += dr*score
        evidence[state] += di*score
    return reward, evidence


def reconstruct_cell(folder):
    """Reconstruct all actual actor updates; no sampler, LP or selector runs."""
    from ceac.contracts import Episode, Transition
    from ceac_pseudocode.learning import TwoCriticActor
    from ceac_hardness.environment import JunctionConfig
    from ceac_hardness.model import JunctionModel
    folder = Path(folder)
    state = json.loads((folder/'status.json').read_text())
    if state['status'] != 'complete':
        raise ValueError('Incomplete reconstruction input')
    for name, checksum in state['artifacts'].items():
        if digest(folder/name) != checksum:
            raise ValueError(f'Reconstruction input mismatch: {name}')
    numerical = json.loads((folder/'manifest.json').read_text())['sources']
    for name, checksum in numerical.items():
        if digest(ROOT/name) != checksum:
            raise ValueError(f'Numerical reconstruction source changed: {name}')
    summary = json.loads((folder/'summary.json').read_text())
    settings = summary['settings']
    model = JunctionModel(JunctionConfig(**settings['environment']))
    cfg = settings['ceac_config']
    actor = TwoCriticActor(model, cfg['actor_rate'], cfg['critic_rate'], cfg['temperature'])
    designs = [e for e in json.loads((folder/'design_history.json').read_text()) if 'design' in e]
    if not designs:
        raise ValueError('No certified replay designs')
    replay_checks = []
    for event in designs:
        d = event['design']; data = d['arrays']
        weights = np.array([d['beta'][i][j] for i, j in d['rows']])
        reward = np.asarray(data['return_mean'])
        evidence = weights @ np.asarray(data['information_mean'])
        np.testing.assert_allclose(reward+evidence, event['priced_replay_scores'], atol=1e-12, rtol=1e-12)
        replay_checks.append(dict(episode=event['episode'], references=d['references'],
            policy_ids=d['policy_ids'], return_mean=reward.tolist(), evidence_bonus=evidence.tolist(),
            priced_scores=event['priced_replay_scores'], beta=d['beta']))
    task = settings['task_evaluator_only']
    dose = model.cfg.burden if model.cfg.protocol == 'burden' else settings['dose']
    q = (.9-.8*dose if task else .1+.8*dose) if model.cfg.protocol == 'prior_shift' else .5
    states = [0] if model.cfg.protocol == 'burden' else [1, 2]
    state_weights = np.array([1.]) if len(states) == 1 else np.array([1-q, q])
    reward_sum = np.zeros_like(actor.logits); evidence_sum = np.zeros_like(actor.logits)
    step_error = 0.; cursor = -1; first_actor = None; behaviors = Counter(); trace = []
    # Deterministic contrasts: action 1 (risky) minus safe/probe/mean nuisance.
    def contrasts(array):
        averages = state_weights @ array[states]
        return dict(risky_vs_safe=float(averages[1]-averages[0]),
                    risky_vs_probe=float(averages[1]-averages[2]),
                    risky_vs_nuisance=float(averages[1]-averages[3:].mean()) if model.actions > 3 else None)
    def capture(episode):
        p = actor.snapshot().probabilities
        average = state_weights @ p[states]
        trace.append(dict(episode=episode, interactions=2*episode, actor_updates=actor.updates,
            actor_probabilities=dict(safe=float(average[0]), risky=float(average[1]),
                                    probe=float(average[2]), nuisance=float(average[3:].sum())),
            reward_contrast=contrasts(reward_sum), evidence_contrast=contrasts(evidence_sum),
            logit_contrast=contrasts(actor.logits), behaviors=dict(behaviors)))
    capture(0)
    with gzip.open(folder/'episodes.jsonl.gz', 'rt') as handle:
        for count, line in enumerate(handle, 1):
            raw = json.loads(line); index = raw['episode']
            if index != count-1:
                raise ValueError('Noncontiguous saved episodes')
            while cursor+1 < len(designs) and designs[cursor+1]['episode'] <= index:
                cursor += 1
            behaviors[raw['behavior']] += 1
            if raw['behavior'] == 'actor':
                if cursor < 0:
                    raise ValueError('Actor precedes certified design')
                if first_actor is None:
                    first_actor = index
                beta = np.array(designs[cursor]['design']['beta'])
                policy = actor.snapshot()
                if policy.identity != raw['policy_id']:
                    raise ValueError(f'Actor policy reconstruction diverged: {folder}, episode {index}')
                episode = Episode(index, 0, policy.identity, 'actor', tuple(Transition(
                    x['state'], x['action'], x['reward'], x['next_state'], x['done'],
                    tuple(x['log_likelihood'])) for x in raw['transitions']))
                before = actor.logits.copy()
                actor.update(episode, policy, beta)
                dr, di = decompose_update(actor, episode, policy)
                error = float(np.max(np.abs(actor.logits-before-dr-di)))
                if error > 1e-11:
                    raise ValueError('Reward/evidence decomposition does not sum to recorded update')
                step_error = max(step_error, error)
                reward_sum += dr; evidence_sum += di
            if count % settings['log_every'] == 0 or count == settings['episodes']:
                capture(count)
    if count != settings['episodes'] or actor.updates != summary['actor_updates'] or actor.priced_updates != summary['priced_actor_updates']:
        raise ValueError('Reconstruction update totals disagree with completed run')
    np.testing.assert_allclose(reward_sum+evidence_sum, actor.logits, atol=1e-9, rtol=1e-9)
    # Explicit evidence identity: same-q opposite tasks have zero cue KL.
    truth = model.true_index(task, q)
    other = model.true_index(1-task, q)
    info_states = [0] if model.cfg.protocol == 'burden' else [1, 2]
    action_kl = model.kl[truth, other, info_states[0], :3].tolist()
    return dict(identity=summary['identity'], folder=str(folder.relative_to(ROOT)),
        protocol=model.cfg.protocol, kind=model.cfg.kind, dose=dose, task=task, seed=settings['seed'],
        input_hashes=state['artifacts'], actor_updates=actor.updates, priced_updates=actor.priced_updates,
        checked_actor_policy_ids=actor.updates, checked_replay_scores=len(designs),
        first_certified_design=designs[0]['episode'], first_actor_episode=first_actor,
        max_update_sum_error=step_error, trace=trace, first_design=replay_checks[0], last_design=replay_checks[-1],
        action_task_kl=action_kl,
        cue_same_q_opposite_task_kl=float(model.kl[truth, other, 0, 0]) if model.cfg.protocol == 'prior_shift' else None)


def export_reconstruction(burden_folder, prior_folder, output, workers=4):
    """All 54 CEAC cells; cache only when raw and analysis identities match."""
    from concurrent.futures import ProcessPoolExecutor
    folders = []
    for study in (burden_folder, prior_folder):
        for entry in json.loads((Path(study)/'results.json').read_text()):
            cell = ROOT/entry['folder']
            if json.loads((cell/'summary.json').read_text())['settings']['method'] == 'ceac':
                folders.append(cell)
    inputs = {str(p.relative_to(ROOT)): digest(p/'status.json') for p in folders}
    source = digest(Path(__file__))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        cached = load_mechanisms(output)
        if cached['source_sha256'] == source and cached['inputs'] == inputs:
            # A status file cannot vouch for mutated raw files by itself.
            for p in folders:
                for name, expected in json.loads((p/'status.json').read_text())['artifacts'].items():
                    if digest(p/name) != expected:
                        raise ValueError('Raw reconstruction input changed')
            print(f'Using {len(cached["cells"])} verified saved-update reconstructions.', flush=True)
            return cached
    cells = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cell in pool.map(reconstruct_cell, folders):
            cells.append(cell)
            print(f'Reconstructed {len(cells)}/{len(folders)}: {cell["kind"]}, dose {cell["dose"]}, task {cell["task"]}, seed {cell["seed"]}', flush=True)
    payload = dict(schema=1, training_launched=False, algorithm_modified=False,
        source_sha256=source, inputs=inputs, cells=cells,
        interpretation='Exact decomposition of recorded updates, not a counterfactual critic-removal experiment.')
    with gzip.open(output/'updates.json.gz', 'wt') as handle:
        json.dump(payload, handle, allow_nan=False, separators=(',', ':'))
    atomic_json(output/'manifest.json', dict(schema=1, files={'updates.json.gz':digest(output/'updates.json.gz')},
        source_sha256=source, cells=len(cells), actor_updates=sum(c['actor_updates'] for c in cells)))
    return payload


def load_mechanisms(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, checksum in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Artifact path leaves export directory')
        if digest(folder/name) != checksum:
            raise ValueError(f'Mechanism evidence mismatch: {name}')
    with gzip.open(folder/'updates.json.gz', 'rt') as handle:
        result = json.load(handle)
    if result['schema'] != 1 or result['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Mechanism schema/source mismatch')
    return result

