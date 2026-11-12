"""Independent kernel, solver, support and artifact checks for the new junctions."""
from itertools import product,combinations
from dataclasses import replace
import json,math,gzip
from pathlib import Path
import numpy as np
import unittest,tempfile
from functools import partial,update_wrapper
from ceac.contracts import Policy,Episode,Transition
from ceac_hardness.environment import JunctionConfig,JunctionEnv
from ceac_hardness.model import JunctionModel
from ceac_hardness.design import PolicyBank,bounded_design,target_calibration,_aec_candidate
from ceac_hardness.ids import BayesianIDS
from ceac_hardness.learning import LikelihoodEvidence,ActorCritic
from ceac_hardness.run import run_cell


def deterministic(model):
    for choices in product(*[np.flatnonzero(mask) for mask in model.mask]):
        p=np.zeros(model.mask.shape);p[np.arange(model.n),choices]=1
        yield Policy(p)


def tree(model,policy,posterior):
    """Enumerate whole trajectories independently of DP, occupancy ratios and KL stages."""
    returns=np.zeros(model.num_models);kl=np.zeros((model.num_models,model.num_models));mi=0.
    def visit(s,weights,reward):
        nonlocal mi
        for a,pa in enumerate(policy.probabilities[s]):
            if pa==0:continue
            laws=model.laws[s,a]
            for key in laws[0]:
                w=weights*pa*np.array([law[key] for law in laws]);total=reward+key[1]
                if key[2]:
                    returns[:]+=w*total
                    kl[:]+=w[:,None]*np.log(w[:,None]/w[None,:])
                    mi+=float(np.sum(posterior*w*np.log(w/(posterior@w))))
                else:visit(model.index[key[0]],w,total)
    visit(model.initial,np.ones(model.num_models),0.)
    return returns,kl,mi

def test_dp_response_and_mi_against_full_tree(protocol,burden):
    m=JunctionModel(JunctionConfig(protocol=protocol,burden=burden));posterior=np.arange(1,m.num_models+1,dtype=float);posterior/=posterior.sum()
    rng=np.random.default_rng(43);rw=rng.normal(size=m.num_models);iw=rng.normal(size=(m.num_models,m.num_models));scores=[]
    for p in deterministic(m):
        returns,kl,mi=tree(m,p,posterior)
        np.testing.assert_allclose(m.returns(p),returns,atol=1e-12)
        np.testing.assert_allclose(m.evaluate(p)[1],kl,atol=1e-12)
        np.testing.assert_allclose(m.evaluate_stage(p,m.mutual_information_stage(posterior)),mi,atol=1e-12)
        scores.append(rw@returns+np.sum(iw*kl))
    p,value=m.response(rw,iw);np.testing.assert_allclose(value,max(scores),atol=1e-11)

def test_burden_changes_only_irrelevant_opportunities(kind):
    base=JunctionModel(JunctionConfig(kind=kind));expected=[base.evaluate(p) for p in base.seed_policies()]
    for burden in (0,4,16):
        m=JunctionModel(JunctionConfig(kind=kind,burden=burden))
        for a,p in enumerate(m.seed_policies()):
            j,k=m.evaluate(p);np.testing.assert_allclose(j,expected[a][0]);np.testing.assert_allclose(k,expected[a][1])
        if burden:
            for a in range(3,m.actions):
                assert np.all(m.kl[:,:,0,a]==0);assert np.all(m.reward[:,0,a]==0)
        for task in (0,1):
            env=JunctionEnv(m.cfg,task);obs,_=env.reset(seed=7)
            for a in range(m.actions):
                env.reset();first=env.step(a);assert not first[2];second=env.step(0);assert second[2]
                assert first[4]['physical_steps']+second[4]['physical_steps']==2
                if a>=3:assert first[1]+second[1]==0


def test_easy_useful_policies_have_equal_occupancy_entropy():
    m=JunctionModel()
    for p in m.seed_policies():
        occupancy=np.zeros(m.n);occupancy[m.initial]=.5
        for a,pa in enumerate(p.probabilities[0]):
            if not pa:continue
            for key,weight in m.laws[0,a][0].items():occupancy[m.index[key[0]]]+=.5*pa*weight
        np.testing.assert_allclose(-np.sum(occupancy[occupancy>0]*np.log(occupancy[occupancy>0])),math.log(2))


def test_fixed_joint_support_is_not_task_or_observation_support():
    m=JunctionModel(JunctionConfig(protocol='prior_shift'));prior=m.prior('correlated');broad=m.prior('product')
    for task in (0,1):
        assert sum(p for p,(t,q) in zip(prior,m.hypotheses) if t==task)==.5
        for dose in (0,.25,.5,.75,1):
            q=(.9-.8*dose) if task else (.1+.8*dose);truth=m.true_index(task,q)
            assert (prior[truth]>0)==(dose==0);assert broad[truth]>0
    for laws in m.laws.values():
        assert all(law.keys()==laws[0].keys() and min(law.values())>0 and math.isclose(sum(law.values()),1.) for law in laws)
    # Cue cannot distinguish policy-changing alternatives with equal nuisance q.
    for i,(t,q) in enumerate(m.hypotheses):
        j=m.true_index(1-t,q);assert m.kl[i,j,0,0]==0


def test_general_binary_port_matches_original_design_ids_and_actor():
    from ceac.design import bounded_design as original_design,PolicyBank as OriginalBank,target_calibration as original_aec
    from ceac.ids import BayesianIDS as OriginalIDS
    from ceac.learning import ActorCritic as OriginalActor
    m=JunctionModel();b=PolicyBank(m,m.seed_policies());old=OriginalBank(m,m.seed_policies())
    cs=[]
    for ref in ((0,),(1,),(0,1)):
        x=bounded_design(m,b,ref);y=original_design(m,old,ref)
        np.testing.assert_allclose(x.cost,y.cost);np.testing.assert_allclose(x.beta,y.beta)
        if len(ref)==1:cs.append(x.cost)
    a=target_calibration(m,b,np.array([.3,.7]),cs);o=original_aec(m,old,np.array([.3,.7]),cs)
    np.testing.assert_allclose(a['value'],o['value']);assert a['good']==o['good']
    for prior in (.2,.7):
        x=BayesianIDS(m,np.random.default_rng(1),prior);y=OriginalIDS(m,np.random.default_rng(1),prior)
        np.testing.assert_allclose(x.distribution(),y.distribution(),atol=1e-12)
    actors=[ActorCritic(m),OriginalActor(m)]
    for actor in actors:
        p=actor.snapshot();e=Episode(0,0,p.identity,'actor',(Transition(0,2,.48,1,False,(-2.,-.1)),Transition(1,0,0.,-1,True,(0.,0.))))
        actor.update(e,p,np.array([[0.,.3],[.7,0.]]))
    np.testing.assert_allclose(actors[0].logits,actors[1].logits);np.testing.assert_allclose(actors[0].row_values,actors[1].row_values)


def test_maximal_feasible_aec_matches_exhaustive_four_model_search():
    m=JunctionModel(JunctionConfig(protocol='prior_shift'),q_values=(.1,.9));base=PolicyBank(m,m.seed_policies())
    c=[bounded_design(m,base,(i,),32).cost for i in range(4)]
    for xi in (np.ones(4)/4,np.array([.03,.07,.3,.6]),np.array([.8,.1,.07,.03])):
        b=PolicyBank(m,m.seed_policies());actual=target_calibration(m,b,xi,c,n_max=32)
        exhaustive=PolicyBank(m,list(deterministic(m)))
        rows=[_aec_candidate(m,exhaustive,xi,good,c,.25,32) for n in range(5) for good in combinations(range(4),n)]
        valid=[r for r in rows if r['feasible']]
        expected=max(r['value'] for r in valid)
        np.testing.assert_allclose(actual['value'],expected,atol=1e-10)
        assert actual['target_gap']<1e-9 and actual['calibration_gap']<1e-9


def test_product_critic_shapes_confidence_and_no_absorbing_numerical_support():
    m=JunctionModel(JunctionConfig(protocol='prior_shift'));evidence=LikelihoodEvidence(m.num_models)
    np.testing.assert_allclose(evidence.threshold,math.log(9/.05))
    actor=ActorCritic(m);assert actor.row_values.shape==(10,10,m.n)
    agent=BayesianIDS(m,np.random.default_rng(1),m.prior('product'))
    agent.log_weights[:]=-1000.;agent.log_weights[0]=0.
    agent.distribution();assert agent.record['regret']<=1e-12;assert agent.initial_support.all();assert np.isfinite(agent.log_weights).all()
    assert agent.posterior_underflow_entries==9
    # Evidence that reverses the accumulated log odds can revive numerical zeros.
    policy=m.seed_policies()[0];ll=np.full(10,0.);ll[0]=-1000.
    episode=Episode(1,0,policy.identity,'test',(Transition(0,0,0.,-1,True,tuple(ll)),))
    agent.observe(episode,policy,np.zeros((10,10)));np.testing.assert_allclose(agent.posterior,.1)


def _check_artifacts(tmp_path):
    args=(JunctionConfig(),'ids_product',12,1,0,16)
    a=run_cell(*args,tmp_path/'a');b=run_cell(*args,tmp_path/'b')
    assert a['late_gap']==b['late_gap'] and a['events']==b['events']
    for name in ('trace.jsonl.gz','episodes.jsonl.gz'):
        with gzip.open(tmp_path/'a'/name,'rt') as x,gzip.open(tmp_path/'b'/name,'rt') as y:assert x.read()==y.read()
    assert run_cell(*args,tmp_path/'a')==json.loads(json.dumps(a))
    with unittest.TestCase().assertRaisesRegex(ValueError,'different identity'):run_cell(*args[:-1],17,tmp_path/'a')
    (tmp_path/'a'/'summary.json').write_text('{}')
    with unittest.TestCase().assertRaisesRegex(ValueError,'artifact mismatch'):run_cell(*args,tmp_path/'a')


def test_reproducible_artifacts_reject_corruption_and_different_settings():
    with tempfile.TemporaryDirectory() as directory:_check_artifacts(Path(directory))


def load_tests(loader,tests,pattern):
    functions=[f for name,f in globals().items() if name.startswith('test_') and name not in
        {'test_dp_response_and_mi_against_full_tree','test_burden_changes_only_irrelevant_opportunities'}]
    functions += [partial(test_dp_response_and_mi_against_full_tree,p,b) for p,b in [('burden',0),('burden',4),('prior_shift',0)]]
    functions += [partial(test_burden_changes_only_irrelevant_opportunities,k) for k in ('undertow_junction','false_beacon_junction')]
    return unittest.TestSuite(unittest.FunctionTestCase(update_wrapper(f,f.func) if isinstance(f,partial) else f) for f in functions)

if __name__=='__main__':unittest.main()
