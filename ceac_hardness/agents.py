# Dimension-general port of ceac/agents.py; see docs/hardness_implementation.md.
"""Shared single-loop CEAC and explicitly privileged Lai--Graves control."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .design import PolicyBank, bounded_design, target_calibration
from .learning import ActorCritic, DebtLedger, LikelihoodEvidence, evidence_scale
from .deployment import certify_deployment


@dataclass(frozen=True)
class CEACConfig:
    confidence_delta: float = .05
    refresh_interval: int = 8
    snapshot_interval: int = 4
    mass_bound: float = 20.
    normalization: float = 20.
    allocation_epsilon: float = .25
    scale_inflation: float = 1.
    temperature: float = 1.
    actor_rate: float = .08
    critic_rate: float = .3
    actor_initialization: str = "model_mixture"
    deployment_rule: str = "certified"
    deployment_tolerance: float = 1e-10

    def __post_init__(self):
        if self.deployment_rule not in {"certified", "actor_only"}:
            raise ValueError("Unknown deployment rule")
        if self.deployment_tolerance < 0:
            raise ValueError("Deployment tolerance must be nonnegative")
        if self.refresh_interval < 1 or self.snapshot_interval < 1:
            raise ValueError("Schedules must have positive intervals")
        if self.normalization < self.mass_bound:
            raise ValueError("Normalization must contain the certified mass-bound designs")
        if self.actor_initialization not in {"model_mixture", "uniform"}:
            raise ValueError("Unknown actor initialization")


class CEAC:
    def __init__(self, model, rng, config=None):
        self.model, self.rng, self.config = model, rng, config or CEACConfig()
        self.bank = PolicyBank(model, model.seed_policies())
        self.replay = LikelihoodEvidence(models=model.num_models, delta=self.config.confidence_delta)
        self.ledger = DebtLedger()
        for policy in self.bank.policies:
            self.ledger.require_seed(policy.identity)
        uniform_information = self.bank.arrays()[1].mean(axis=2)
        minimum = min(uniform_information[row] for row in model.alternatives)
        if minimum <= 0 or self.config.mass_bound < 1/minimum:
            raise ValueError("Declared mass bound fails the seed-mixture certificate")
        self.seed_mass_bound = 1/minimum
        # Public hypothetical-model coefficients, not the realized-model oracle.
        coefficient_bank = PolicyBank(model, model.seed_policies())
        self.oracle_certificates = [bounded_design(model, coefficient_bank, (m,), self.config.mass_bound)
                                    for m in range(model.num_models)]
        self.coefficients = np.array([d.cost for d in self.oracle_certificates])
        initial = (model.response(np.ones(model.num_models)/model.num_models, np.zeros((self.model.num_models, self.model.num_models)), "initial-mixture")[0]
                   if self.config.actor_initialization == "model_mixture" else None)
        self.actor = ActorCritic(model, self.config.temperature, self.config.actor_rate,
                                 self.config.critic_rate, initial)
        self.initial_actor = self.actor.snapshot()
        self.events = []
        self.design = None
        self.deployment = None
        self.deployment_policies = {}
        self.deployment_episodes = 0
        self.aec = None
        self.episodes = 0
        self.actor_episodes = 0
        self.admissions = 0
        self.refresh()

    def refresh(self):
        plausible = self.replay.plausible
        aec = target_calibration(self.model, self.bank, self.replay.mixture, self.coefficients,
                                 self.config.allocation_epsilon, self.config.normalization)
        before = len(self.bank.policies)
        design = bounded_design(self.model, self.bank, plausible, self.config.mass_bound)
        if len(self.bank.policies) != before:
            # Vectors/certificates in both stages refer to the same frozen bank.
            aec = target_calibration(self.model, self.bank, self.replay.mixture, self.coefficients,
                                     self.config.allocation_epsilon, self.config.normalization)
            design = bounded_design(self.model, self.bank, plausible, self.config.mass_bound)
        self.design, self.aec = design, aec
        self.deployment = certify_deployment(self.model, self.replay, self.config.deployment_tolerance)
        self.deployment_policies[self.deployment.policy.identity] = self.deployment.policy
        self.ledger.add_design(design, evidence_scale(max(1, self.episodes), self.config.scale_inflation))
        request = None
        if not set(plausible).issubset(aec["good"]):
            index = int(self.rng.choice(len(aec["mixture"]), p=aec["mixture"]))
            request = aec["policy_ids"][index]
            self.ledger.request_fresh(request)
        self.events.append({"episode": self.episodes, "type": "refresh", "plausible": plausible,
                            "model_mixture": self.replay.mixture.tolist(), "design": design.record(),
                            "deployment": self.deployment.record(),
                            "aec": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in aec.items()},
                            "statistical_request": request, "ledger": self.ledger.record()})

    def select(self, index):
        del index
        if ((self.episodes and self.episodes % self.config.refresh_interval == 0)
                or self.replay.plausible != self.design.references):
            self.refresh()
        identity, behavior = self.ledger.select()
        if identity is None:
            if self.config.deployment_rule == "certified" and self.deployment.valid_for(self.replay.plausible):
                return self.deployment.policy, "certified_deploy", np.zeros((self.model.num_models, self.model.num_models))
            return self.actor.snapshot(), "actor", self.design.beta.copy()
        policy = self.bank.policies[self.bank._ids[identity]]
        return policy, behavior, self.design.beta.copy()

    def observe(self, episode, policy, prices):
        changed = self.actor.update(episode, policy, prices)
        self.actor_episodes += int(changed)
        self.deployment_episodes += int(episode.behavior == "certified_deploy")
        self.replay.append(episode)
        self.ledger.credit(episode)
        self.episodes += 1
        if self.episodes % self.config.snapshot_interval == 0:
            proposal = self.actor.snapshot()
            returns, information = self.model.evaluate(proposal)
            rc = (self.design.constant - self.design.reward_weights @ returns + self.design.mass_price
                  - np.sum(self.design.beta * information))
            admitted = rc < -1e-9 and self.bank.add(proposal)
            self.admissions += int(admitted)
            self.events.append({"episode": self.episodes, "type": "actor_snapshot",
                                "policy_id": proposal.identity, "reduced_cost_upper": float(rc),
                                "admitted": bool(admitted), "certificate": "exact_declared_models"})
            if admitted:
                self.refresh()

    def diagnostics(self):
        return {"plausible": self.replay.plausible, "model_mixture": self.replay.mixture.tolist(),
                "bank_size": len(self.bank.policies), "actor_updates": self.actor.updates,
                "deployment_rule": self.config.deployment_rule,
                "deployment_episodes": self.deployment_episodes, "deployment": self.deployment.record(),
                "actor_admissions": self.admissions, "design": self.design.record(),
                "ledger": self.ledger.record(), "seed_mass_bound": self.seed_mass_bound}


class LaiGravesOracle:
    """Known-mode benchmark; expected design counts, with the same scale.

Only the runner can supply the true mode to this explicitly privileged arm.
"""
    def __init__(self, model, mode, config=None):
        self.model, self.mode, self.config = model, int(mode), config or CEACConfig()
        self.bank = PolicyBank(model, model.seed_policies())
        self.design = bounded_design(model, self.bank, (mode,), self.config.mass_bound)
        self.ledger = DebtLedger()
        self.episodes = 0
        self.events = []

    def select(self, index):
        del index
        self.ledger.add_design(self.design, evidence_scale(max(1, self.episodes), self.config.scale_inflation))
        identity, _ = self.ledger.select()
        if identity is None:
            return self.model.optimal_policies[self.mode], "oracle_deploy", np.zeros((self.model.num_models, self.model.num_models))
        return self.bank.policies[self.bank._ids[identity]], "oracle_design", np.zeros((self.model.num_models, self.model.num_models))

    def observe(self, episode, policy, prices):
        del policy, prices
        self.ledger.credit(episode)
        self.episodes += 1

    def diagnostics(self):
        return {"oracle_mode": self.mode, "design": self.design.record(), "ledger": self.ledger.record()}
