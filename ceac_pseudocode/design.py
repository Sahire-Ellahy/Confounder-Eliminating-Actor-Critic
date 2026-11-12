"""Restricted master consumes observed replay estimates, never model means."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog

@dataclass
class ReplayDesign:
    policy_ids:tuple
    eta:np.ndarray
    beta:np.ndarray
    mass_price:float
    cost:float
    row_minimum:float
    primal_dual_gap:float
    references:tuple
    rows:tuple
    data:dict
    column_residual:float=0.

    def record(self):
        return dict(eta=self.eta.tolist(),beta=self.beta.tolist(),mass_price=self.mass_price,cost=self.cost,
            row_minimum=self.row_minimum,primal_dual_gap=self.primal_dual_gap,column_residual=self.column_residual,
            references=[int(r) for r in self.references],rows=self.rows,policy_ids=self.policy_ids,
            arrays={k:v.tolist() for k,v in self.data.items()},estimation='observed_policy_labeled_replay')


def solve_replay_master(replay,policies,references,upper_optimum,mass_bound=32.):
    model=replay.model;rows=tuple(r for r in model.alternatives if r[0] in references)
    data=replay.arrays(policies,rows);costs=np.maximum(0.,upper_optimum-data['return_lower']);k=data['information_lower']
    result=linprog(costs,A_ub=np.vstack([-k,np.ones(len(policies))]),
        b_ub=np.r_[-np.ones(len(rows)),mass_bound],bounds=(0,None),method='highs')
    if not result.success:
        if result.status==2:return None # Explicit statistical debt, never exact-model substitution.
        raise RuntimeError('CEAC replay master failed: '+result.message)
    prices=-result.ineqlin.marginals[:-1];mass_price=float(-result.ineqlin.marginals[-1])
    beta=np.zeros((model.num_models,model.num_models))
    for row,price in zip(rows,prices):beta[row]=price
    dual=float(prices.sum()-mass_bound*mass_price);minimum=float(np.min(k@result.x))
    if abs(result.fun-dual)>1e-7 or minimum<1-1e-7:raise RuntimeError('Replay primal/dual certificate failed')
    return ReplayDesign(tuple(p.identity for p in policies),result.x,beta,mass_price,float(result.fun),minimum,
        abs(float(result.fun)-dual),tuple(references),rows,data)
