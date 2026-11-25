"""Catalog of persistent ProblemBuilder function classes organized strictly by environment family."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from ceac_bench.core.contracts import ConditionKey, DomainMetadata
from ceac_bench.core.primitives import FrozenFunctionClass, seed_for
from .interfaces import BoundProblem, EnvironmentFamily, FiniteDAGOracleEvaluator, PublicProblemView


@dataclass(frozen=True)
class JunctionProblemBuilder(FrozenFunctionClass):
    """Controlled two-step Undertow & False Beacon junction builder (`burden` or `prior_shift`)."""

    variant_id: str  # "undertow_junction" | "false_beacon_junction"
    protocol: str = "burden"

    @property
    def metadata(self) -> DomainMetadata:
        is_u = self.variant_id == "undertow_junction"
        fam = "undertow" if is_u else "false_beacon"
        short = "Undertow Junction" if is_u else ("False Beacon Cue-Reversal" if self.protocol == "prior_shift" else "False Beacon Junction")
        title = f"Controlled {short} ({self.protocol})"
        sym = "d" if self.protocol == "prior_shift" else "B"
        lbl = "Cue-correlation reversal d" if self.protocol == "prior_shift" else "Nuisance breadth B"
        return DomainMetadata(
            family_id=fam,
            variant_id=self.variant_id,
            title=title,
            short_title=short,
            horizon=2,
            dose_symbol=sym,
            dose_label=lbl,
            easy_dose=0,
            event_names=("safe", "risky", "probe", "nuisance"),
            reward_bounds=(0.0, 1.0),
            rng_namespace=self.protocol,
            restricted_prior_name="correlated" if self.protocol == "prior_shift" else "zero",
            reference_prior_name="product",
        )

    def __call__(self, condition: ConditionKey) -> BoundProblem:
        from ceac_hardness.environment import JunctionConfig, JunctionEnv
        from ceac_hardness.model import JunctionModel

        burden = int(condition.dose) if self.protocol == "burden" else 0
        shift = float(condition.dose) if self.protocol == "prior_shift" else 0.0
        task_i = int(condition.task)
        q = (0.9 - 0.8 * shift if task_i == 1 else 0.1 + 0.8 * shift) if self.protocol == "prior_shift" else 0.5
        cfg = JunctionConfig(kind=self.variant_id, protocol=self.protocol, burden=burden)
        model = JunctionModel(cfg)
        truth = model.true_index(task_i, q)
        env = JunctionEnv(cfg, task=task_i, q=q)
        meta = self.metadata
        pub = PublicProblemView.from_finite_model(meta, model, asdict(cfg))
        ev = FiniteDAGOracleEvaluator.build(self.variant_id, model, truth=truth, realized_nuisance=0)
        return BoundProblem(condition=condition, metadata=meta, env=env, public_view=pub, evaluator=ev)


@dataclass(frozen=True)
class GridPathDepProblemBuilder(FrozenFunctionClass):
    """Multi-step path-dependent Undertow DeepSea (`undertow_pathdep`) & False Beacon (`false_beacon_pathdep`) builder."""

    variant_id: str  # "undertow_pathdep" | "false_beacon_pathdep"

    @property
    def metadata(self) -> DomainMetadata:
        is_u = self.variant_id == "undertow_pathdep"
        return DomainMetadata(
            family_id="undertow" if is_u else "false_beacon",
            variant_id=self.variant_id,
            title=(
                "Path-Dependent Undertow DeepSea (Thermocline Lock, H = 3)"
                if is_u
                else "Path-Dependent False Beacon Shore (Icebreaker Staging, H = 3)"
            ),
            short_title="Path-Dependent Undertow" if is_u else "Path-Dependent False Beacon",
            horizon=3,
            dose_symbol="k",
            dose_label="Path-dependent confounding modules k (6k echo plumes/glints, 3^k − 1 off-path eddies)",
            easy_dose=0,
            event_names=(
                "safe",
                "stage",
                "unstaged",
                "target_route",
                "wrong_route",
                "probe",
                "echo_reporter",
                "passenger_omics",
                "eddy_trap",
            ),
            reward_bounds=(-0.02, 1.02),
            rng_namespace="grid_pathdep",
        )

    def __call__(self, condition: ConditionKey) -> BoundProblem:
        from ceac_pathdep_grids.environments import (
            PathDepFalseBeaconConfig,
            PathDepFalseBeaconEnv,
            PathDepFalseBeaconModel,
            PathDepUndertowConfig,
            PathDepUndertowEnv,
            PathDepUndertowModel,
            relabel_grid_pathdep,
        )

        dose_i = int(condition.dose)
        seed_i = int(condition.seed)
        task_i = int(condition.task)
        if self.variant_id == "undertow_pathdep":
            cfg = relabel_grid_pathdep(PathDepUndertowConfig(complexity=dose_i), seed_i)
            model = PathDepUndertowModel(cfg)
            env_cls = PathDepUndertowEnv
        elif self.variant_id == "false_beacon_pathdep":
            cfg = relabel_grid_pathdep(PathDepFalseBeaconConfig(complexity=dose_i), seed_i)
            model = PathDepFalseBeaconModel(cfg)
            env_cls = PathDepFalseBeaconEnv
        else:
            raise ValueError(f"Unknown grid pathdep variant: {self.variant_id}")

        pas_rng = np.random.default_rng(seed_for("grid_pathdep_pas", self.variant_id, seed_i, dose_i))
        pas = int(pas_rng.integers(cfg.modifier_states)) if cfg.complexity > 0 else 0
        truth = model.true_index(task_i, passenger=pas)
        ch, mod = model.hypotheses[truth]
        env = env_cls(cfg, channel=ch, modifier=mod)
        meta = self.metadata
        pub = PublicProblemView.from_finite_model(meta, model, asdict(cfg))
        ev = FiniteDAGOracleEvaluator.build(self.variant_id, model, truth=truth, realized_nuisance=pas)
        return BoundProblem(condition=condition, metadata=meta, env=env, public_view=pub, evaluator=ev)


@dataclass(frozen=True)
class LockProblemBuilder(FrozenFunctionClass):
    """Sequential and bidirectional combination lock builder (`L in {1, 2, 3, 4}`)."""

    variant_id: str  # "combination_lock" | "bidirectional_lock"

    @property
    def metadata(self) -> DomainMetadata:
        is_bi = self.variant_id == "bidirectional_lock"
        return DomainMetadata(
            family_id="locks",
            variant_id=self.variant_id,
            title="Bidirectional Combination Lock" if is_bi else "Sequential Combination Lock",
            short_title="Bidirectional Lock" if is_bi else "Combination Lock",
            horizon=5,
            dose_symbol="L",
            dose_label="Gated lock depth L (horizon H = L + 1)",
            easy_dose=1,
            event_names=("safe", "target", "probe", "sink", "decoy_unlock"),
            reward_bounds=(-0.02, 1.02),
            rng_namespace="locks",
            restricted_prior_name="decoy",
            reference_prior_name="product",
        )

    def __call__(self, condition: ConditionKey) -> BoundProblem:
        from ceac_pathologies.environments import LockConfig, LockEnv
        from ceac_pathologies.model import LockModel

        cfg = LockConfig(kind=self.variant_id, depth=int(condition.dose))
        model = LockModel(cfg)
        truth = model.true_index(int(condition.task))
        env = LockEnv(cfg, task=int(condition.task))
        base_meta = self.metadata
        meta = DomainMetadata(
            family_id=base_meta.family_id,
            variant_id=base_meta.variant_id,
            title=base_meta.title,
            short_title=base_meta.short_title,
            horizon=int(model.horizon),
            dose_symbol=base_meta.dose_symbol,
            dose_label=base_meta.dose_label,
            easy_dose=base_meta.easy_dose,
            event_names=base_meta.event_names,
            reward_bounds=base_meta.reward_bounds,
            rng_namespace=base_meta.rng_namespace,
            restricted_prior_name=base_meta.restricted_prior_name,
            reference_prior_name=base_meta.reference_prior_name,
        )
        pub = PublicProblemView.from_finite_model(meta, model, asdict(cfg))
        ev = FiniteDAGOracleEvaluator.build(self.variant_id, model, truth=truth, realized_nuisance=0)
        return BoundProblem(condition=condition, metadata=meta, env=env, public_view=pub, evaluator=ev)


@dataclass(frozen=True)
class EpistasisProblemBuilder(FrozenFunctionClass):
    """Single-stage epistatic signalling under off-pathway pleiotropic confounding."""

    variant_id: str = "epistatic_signalling"
    subtypes: int = 128

    @property
    def metadata(self) -> DomainMetadata:
        return DomainMetadata(
            family_id="signalling",
            variant_id=self.variant_id,
            title="Epistatic Signalling under Pleiotropic Confounding",
            short_title="Epistatic Signalling",
            horizon=2,
            dose_symbol="k",
            dose_label="Non-causal pleiotropic passenger modules k",
            easy_dose=0,
            event_names=("safe", "synergy", "probe", "passenger_omics", "pleiotropic"),
            reward_bounds=(0.0, 1.0),
            rng_namespace="epistasis",
            restricted_prior_name="canonical",
            reference_prior_name="reference",
        )

    def __call__(self, condition: ConditionKey) -> BoundProblem:
        from ceac_pathologies.environments import EpistasisConfig, EpistasisEnv
        from ceac_pathologies.model import EpistasisModel

        dose_i = int(condition.dose)
        seed_i = int(condition.seed)
        task_i = int(condition.task)
        cfg = EpistasisConfig(kind=self.variant_id, subtypes=self.subtypes, pleiotropy=dose_i)
        model = EpistasisModel(cfg)
        pas_rng = np.random.default_rng(seed_for("realized_passenger", seed_i, cfg.pleiotropy))
        pas = int(pas_rng.integers(cfg.passenger_states)) if cfg.pleiotropy > 0 else 0
        truth = model.true_index(task_i, passenger=pas)
        subtype = (cfg.subtypes - 1) if task_i == 1 else 0
        env = EpistasisEnv(cfg, subtype=subtype, passenger=pas)
        meta = self.metadata
        pub = PublicProblemView.from_finite_model(meta, model, asdict(cfg))
        ev = FiniteDAGOracleEvaluator.build(self.variant_id, model, truth=truth, realized_nuisance=pas)
        return BoundProblem(condition=condition, metadata=meta, env=env, public_view=pub, evaluator=ev)


@dataclass(frozen=True)
class SignallingRewiringProblemBuilder(FrozenFunctionClass):
    """Multi-step path-dependent oncogenic signalling rewiring (`EGFR/MEK` priming, H = 3)."""

    variant_id: str = "signalling_rewiring"

    @property
    def metadata(self) -> DomainMetadata:
        return DomainMetadata(
            family_id="signalling",
            variant_id=self.variant_id,
            title="Path-Dependent Signalling Rewiring (EGFR/MEK Priming, H = 3)",
            short_title="Signalling Rewiring (H = 3)",
            horizon=3,
            dose_symbol="k",
            dose_label="Path-dependent confounding modules k (6k echo clades, 3^k − 1 stress states)",
            easy_dose=0,
            event_names=(
                "safe",
                "prime",
                "unprimed_cotarget",
                "synergy",
                "wrong_cotarget",
                "probe",
                "echo_reporter",
                "passenger_omics",
                "pleiotropic",
            ),
            reward_bounds=(-0.02, 1.02),
            rng_namespace="pathdep",
            restricted_prior_name="canonical",
            reference_prior_name="reference",
        )

    def __call__(self, condition: ConditionKey) -> BoundProblem:
        from ceac_pathdep.environments import (
            SignallingRewiringConfig,
            SignallingRewiringEnv,
            SignallingRewiringModel,
            relabel_signalling,
        )

        dose_i = int(condition.dose)
        seed_i = int(condition.seed)
        task_i = int(condition.task)
        cfg = relabel_signalling(SignallingRewiringConfig(complexity=dose_i), seed_i)
        model = SignallingRewiringModel(cfg)
        pas_rng = np.random.default_rng(seed_for("pathdep_passenger", self.variant_id, seed_i, dose_i))
        pas = int(pas_rng.integers(cfg.modifier_states)) if cfg.complexity > 0 else 0
        truth = model.true_index(task_i, passenger=pas)
        pw, mod = model.hypotheses[truth]
        env = SignallingRewiringEnv(cfg, pathway=pw, modifier=mod)
        meta = self.metadata
        pub = PublicProblemView.from_finite_model(meta, model, asdict(cfg))
        ev = FiniteDAGOracleEvaluator.build(self.variant_id, model, truth=truth, realized_nuisance=pas)
        return BoundProblem(condition=condition, metadata=meta, env=env, public_view=pub, evaluator=ev)


@dataclass(frozen=True)
class K562GIScreenProblemBuilder(FrozenFunctionClass):
    """Horlbeck 2018 K562 pairwise CRISPRi screen (`gi_arrayed` and `gi_multiplexed`)."""

    variant_id: str = "gi_arrayed"

    @property
    def metadata(self) -> DomainMetadata:
        is_arr = self.variant_id == "gi_arrayed"
        return DomainMetadata(
            family_id="k562_gi",
            variant_id=self.variant_id,
            title="Horlbeck 2018 K562 Arrayed Pair Screen" if is_arr else "Horlbeck 2018 K562 Multiplexed Screen",
            short_title="K562 GI Arrayed" if is_arr else "K562 GI Multiplexed",
            horizon=2,
            dose_symbol="k",
            dose_label="Variables with non-causal main effects k (M = 46 · 2^k)",
            easy_dose=0,
            event_names=("control", "single_ko", "passenger_omics", "pair_test", "sl_pair", "block"),
            reward_bounds=(0.0, 1.0),
            rng_namespace="gi_screen",
            restricted_prior_name="additive",
            reference_prior_name="reference",
        )

    def __call__(self, condition: ConditionKey) -> BoundProblem:
        from ceac_gi.environment import GIScreenEnv, GIScreenModel, build_config, relabel

        dose_i = int(condition.dose)
        seed_i = int(condition.seed)
        task_i = int(condition.task)
        cfg = relabel(build_config(self.variant_id, dose_i), seed_i)
        model = GIScreenModel(cfg)
        pas_rng = np.random.default_rng(seed_for("realized_passenger", seed_i, cfg.nuisance))
        pas = int(pas_rng.integers(cfg.passenger_states)) if cfg.nuisance > 0 else 0
        truth = model.true_index(task_i, passenger=pas)
        env = GIScreenEnv(cfg, pair=model.hypotheses[truth][0], passenger=pas)
        meta = self.metadata
        pub = PublicProblemView.from_finite_model(meta, model, asdict(cfg))
        ev = FiniteDAGOracleEvaluator.build(self.variant_id, model, truth=truth, realized_nuisance=pas)
        return BoundProblem(condition=condition, metadata=meta, env=env, public_view=pub, evaluator=ev)


@dataclass(frozen=True)
class CanalizedGIProblemBuilder(FrozenFunctionClass):
    """Multi-step canalized K562 genetic risk screen (`PPP2R1A` hub de-canalization, H = 3)."""

    variant_id: str = "canalized_gi"

    @property
    def metadata(self) -> DomainMetadata:
        return DomainMetadata(
            family_id="k562_gi",
            variant_id=self.variant_id,
            title="Canalized Genetic Risk Screen (K562 PPP2R1A Hub, H = 3)",
            short_title="Canalized K562 Genetic Risk (H = 3)",
            horizon=3,
            dose_symbol="k",
            dose_label="Path-dependent confounding modules k (6k echo eQTLs, 3^k − 1 off-path states)",
            easy_dose=0,
            event_names=(
                "control",
                "decanalize_hub",
                "single_ko",
                "sl_pair",
                "pair_test",
                "probe",
                "echo_reporter",
                "passenger_omics",
                "pleiotropic",
            ),
            reward_bounds=(-0.02, 1.02),
            rng_namespace="pathdep",
            restricted_prior_name="additive",
            reference_prior_name="reference",
        )

    def __call__(self, condition: ConditionKey) -> BoundProblem:
        from ceac_pathdep.environments import (
            CanalizedGIEnv,
            CanalizedGIModel,
            build_canalized_gi_config,
            relabel_canalized_gi,
        )

        dose_i = int(condition.dose)
        seed_i = int(condition.seed)
        task_i = int(condition.task)
        cfg = relabel_canalized_gi(build_canalized_gi_config(dose_i), seed_i)
        model = CanalizedGIModel(cfg)
        pas_rng = np.random.default_rng(seed_for("pathdep_passenger", self.variant_id, seed_i, dose_i))
        pas = int(pas_rng.integers(cfg.passenger_states)) if cfg.complexity > 0 else 0
        truth = model.true_index(task_i, passenger=pas)
        q, mod = model.hypotheses[truth]
        env = CanalizedGIEnv(cfg, active_modifier=q, passenger=mod)
        meta = self.metadata
        pub = PublicProblemView.from_finite_model(meta, model, asdict(cfg))
        ev = FiniteDAGOracleEvaluator.build(self.variant_id, model, truth=truth, realized_nuisance=pas)
        return BoundProblem(condition=condition, metadata=meta, env=env, public_view=pub, evaluator=ev)


ALL_FAMILIES: tuple[EnvironmentFamily, ...] = (
    EnvironmentFamily(
        family_id="undertow",
        title="Undertow DeepSea Family",
        builders=(
            JunctionProblemBuilder("undertow_junction", protocol="burden"),
            GridPathDepProblemBuilder("undertow_pathdep"),
        ),
    ),
    EnvironmentFamily(
        family_id="false_beacon",
        title="False Beacon Frozen Lake Family",
        builders=(
            JunctionProblemBuilder("false_beacon_junction", protocol="burden"),
            JunctionProblemBuilder("false_beacon_junction", protocol="prior_shift"),
            GridPathDepProblemBuilder("false_beacon_pathdep"),
        ),
    ),
    EnvironmentFamily(
        family_id="locks",
        title="Combination Lock Family",
        builders=(
            LockProblemBuilder("combination_lock"),
            LockProblemBuilder("bidirectional_lock"),
        ),
    ),
    EnvironmentFamily(
        family_id="signalling",
        title="Oncogenic Signalling Pathway Family",
        builders=(
            EpistasisProblemBuilder("epistatic_signalling"),
            SignallingRewiringProblemBuilder("signalling_rewiring"),
        ),
    ),
    EnvironmentFamily(
        family_id="k562_gi",
        title="Human K562 Genetic Interaction Family",
        builders=(
            K562GIScreenProblemBuilder("gi_arrayed"),
            K562GIScreenProblemBuilder("gi_multiplexed"),
            CanalizedGIProblemBuilder("canalized_gi"),
        ),
    ),
)


def get_family(family_id: str) -> EnvironmentFamily:
    for fam in ALL_FAMILIES:
        if fam.family_id == family_id:
            return fam
    raise KeyError(f"Unknown environment family: {family_id}")


def get_problem_builder(variant_id: str, *, protocol: str | None = None) -> ProblemBuilder:
    for fam in ALL_FAMILIES:
        for b in fam.builders:
            if b.metadata.variant_id == variant_id:
                if protocol is None or getattr(b, "protocol", None) == protocol:
                    return b
    raise KeyError(f"Unknown problem variant={variant_id!r}, protocol={protocol!r}")
