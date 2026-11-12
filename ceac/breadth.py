"""Separately registered physical-breadth experiment and exact preflight."""
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from . import benchmarks
from .model import FiniteModel, environment as original_environment
from .benchmark_audit import audit_model
from .environments import frozenlake
from .environments.beacon_breadth import BeaconBreadthConfig, BeaconBreadthEnv

BREADTH_PROFILES={f"beacon_breadth_b{b:02d}":b for b in (0,1,2,4,8)}


@dataclass(frozen=True)
class BreadthBenchmark(benchmarks.Benchmark):
    beacon_cells: int = 0

    def overrides(self, kind):
        if kind!="frozenlake":
            raise ValueError("Beacon breadth profiles require --environments frozenlake")
        return dict(super().overrides(kind),beacon_cells=self.beacon_cells)


def register_breadth():
    for name,b in BREADTH_PROFILES.items():
        benchmark=BreadthBenchmark(name,4,.05,beacon_cells=b)
        if name in benchmarks.BENCHMARKS and benchmarks.BENCHMARKS[name]!=benchmark:
            raise ValueError("Conflicting breadth benchmark registration")
        benchmarks.BENCHMARKS[name]=benchmark


def specification(name, kind):
    register_breadth()
    spec=benchmarks.specification(name,kind)
    if name in BREADTH_PROFILES:
        spec=dict(spec,nuisance_mechanism="mode_independent_physical_chamber_breadth",
                  seed_bank_rule="six_original_routes_at_every_breadth",
                  dose_unit="additional_traversable_chamber_cells")
    return spec


def make_environment(kind, modes=(0,1,0), block_length=64, benchmark="legacy"):
    if benchmark not in BREADTH_PROFILES:
        return original_environment(kind,modes,block_length,benchmark)
    override=specification(benchmark,kind)["configuration_overrides"]
    return BeaconBreadthEnv(BeaconBreadthConfig(true_modes=tuple(modes),block_length=block_length,**override))


class BreadthModel(FiniteModel):
    """Use the unchanged finite-model compiler, evaluator and response solvers.

    Only simulator construction differs. This keeps historical kernels and
    their hash-verified notebook usable while registering a new runtime lane.
    """
    def __init__(self, kind="frozenlake", context=0, benchmark="beacon_breadth_b00"):
        if kind!="frozenlake" or benchmark not in BREADTH_PROFILES:
            raise ValueError("Not a registered Frozen Lake breadth profile")
        self.kind,self.context,self.benchmark=kind,int(context),benchmark
        self.specification=specification(benchmark,kind)
        self.module=frozenlake
        self._models=[make_environment(kind,(m,m,m),benchmark=benchmark) for m in (0,1)]
        self.cfg=self._models[0].config
        self.actions=self._models[0].action_space.n
        self.num_models=2
        self.initial_observation=self._models[0].reset(
            seed=0,options={"episode_index":self.context*self.cfg.block_length})[0]
        self._compile()
        self.optimal_policies=[self.response(np.eye(2)[m],np.zeros((2,2)))[0] for m in (0,1)]
        self.optimal_returns=np.array([self.evaluate(p)[0][m] for m,p in enumerate(self.optimal_policies)])
        self.alternatives=tuple((m,j) for m in (0,1) for j in (0,1) if m!=j
            and self.optimal_returns[j]-self.evaluate(self.optimal_policies[m])[0][j]>1e-10)
        self._mi_cache=None


def make_model(kind, context=0, benchmark="legacy"):
    factory=BreadthModel if benchmark in BREADTH_PROFILES else FiniteModel
    return factory(kind,context,benchmark)


def audit_breadth_model(model, mass_bound=128.):
    row=audit_model(model,mass_bound)
    env=model._models[0]
    field=env.field_cells
    observed=set()
    max_kl=0.
    for s,obs in enumerate(model.state_ids):
        state=env.decode(int(obs));here=(state.row,state.col)
        if here in field:
            observed.add(here)
        for action in np.flatnonzero(model.mask[s]):
            law=model.laws[s,int(action)][0]
            if here in field or any((env.decode(o).row,env.decode(o).col) in field for o,_,_ in law):
                max_kl=max(max_kl,float(model.kl[:,:,s,action].max()))
    if observed!=field or max_kl>1e-12:
        raise ValueError("Beacon footprint is unreachable or leaks task information")
    row["breadth"]={"cells":model.cfg.beacon_cells,"physical_cells":sorted(map(list,field)),
                    "field_transition_max_task_kl":max_kl,"horizon":model.cfg.max_steps,
                    "primitive_actions":model.actions}
    return row


def audit_breadth_family(audits):
    groups=defaultdict(list)
    for row in audits:
        if row["benchmark"] in BREADTH_PROFILES:
            groups[row["environment"],row["context"]].append(row)
    checked=[]
    for (kind,context),rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:BREADTH_PROFILES[r["benchmark"]])
        if len({r["benchmark"] for r in rows})!=len(rows):
            raise ValueError("Duplicate breadth preflight")
        ref=rows[0]
        def fixed(row):
            cfg=dict(row["specification"]["configuration_overrides"])
            cfg.pop("beacon_cells")
            return cfg,row["seed_policies"],row["breadth"]["horizon"],row["breadth"]["primitive_actions"]
        ref_policies={p["label"]:p for p in ref["policies"]}
        last_cells=set();last_states=-1
        for row in rows:
            if fixed(row)!=fixed(ref):
                raise ValueError("A non-breadth setting changed")
            cells=set(map(tuple,row["breadth"]["physical_cells"]))
            if len(cells)!=BREADTH_PROFILES[row["benchmark"]] or not last_cells<=cells:
                raise ValueError("Footprints are not the declared nested breadths")
            if row["reachable_states"]<=last_states or row["breadth"]["field_transition_max_task_kl"]>1e-12:
                raise ValueError("Breadth must add reachable states with zero field task KL")
            for key in ("optimal_returns","return_range","oracle_design_costs"):
                np.testing.assert_allclose(row[key],ref[key],atol=1e-10,rtol=0)
            policies={p["label"]:p for p in row["policies"]}
            for label in frozenlake.POLICY_NAMES:
                for metric in ("expected_return","task_kl"):
                    np.testing.assert_allclose(policies[label][metric],ref_policies[label][metric],atol=1e-10,rtol=0)
            last_cells,last_states=cells,row["reachable_states"]
        checked.append(dict(environment=kind,context=context,
            breadths=[BREADTH_PROFILES[r["benchmark"]] for r in rows],
            reachable_states=[r["reachable_states"] for r in rows],fixed_domain=False,
            fixed_horizon_and_action_alphabet=True,fixed_six_seed_routes=True,
            fixed_useful_route_values_and_KL=True,fixed_optimum_and_oracle_cost=True,
            zero_task_KL_in_added_chamber=True))
    return checked
