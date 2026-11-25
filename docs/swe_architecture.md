# Four-Layer Benchmark Architecture (`ceac_bench`)

`ceac_bench` separates problem construction, sequential-decision execution, metric/Bellman logging & caching, and visual/notebook rendering into four strictly unidirectional layers connected by immutable persistent data structures and picklable persistent function classes (`FrozenFunctionClass`).

## 1. Layer Responsibilities & Import Rules

| Layer | Package | Core Abstractions | Forbidden Dependencies |
|---|---|---|---|
| **Foundation** | `ceac_bench.core` | `FrozenFunctionClass`, `PersistentCallable`, `RNGStreamSet`, `ConditionKey`, `DomainMetadata`, `PolicyView`, `SelectionStep`, `CriticSnapshot`, `HeldOutBatch` | Higher layers, `matplotlib`, `IPython`, `nbformat` |
| **Layer 1: Problem Setup** | `ceac_bench.problems` | `StepEnvironment`, `PublicProblemView` (zero `truth` leak), `OracleEvaluator` (`FiniteDAGOracleEvaluator`), `BoundProblem`, `ProblemBuilder`, `EnvironmentFamily` | `runner`, `metrics`, `store`, `reporting`, UI libraries |
| **Layer 2: Experimental Runners** | `ceac_bench.runner` | `Learner`, `LearnerFactory` (`IDACFactory`, `UCBQFactory`, `POMDFactory`, `BayesianIDSFactory`, `OptimalActionIDSFactory`, `CEACFactory`, `GreedyRankedCEACFactory`), `CellRunner`, `CalibrationGate`, `SpawnStudyExecutor` | `metrics`, `reporting`, UI libraries |
| **Layer 3: Metrics & Caching** | `ceac_bench.metrics`, `ceac_bench.store` | `PerformanceTraceObserver`, `DomainEventObserver`, `LearnerDiagnosticsObserver`, `HeldOutBellmanObserver`, `CellRecord`, `StudyBundle` (schema v2), `CellCacheStore`, `BundleStore`, `LegacyBundleAdapter` | `reporting`, UI libraries |
| **Layer 4: Rendering** | `ceac_bench.reporting` | `VisualTheme` (single source of truth for Cobalt/Periwinkle/Red/Green/Hot-Pink styling), `SingleEnvDoseScalingPlot`, `SingleEnvTimeTracePlot`, `SingleEnvCriticBellmanPlot`, `DoseTransitionMarkdownTable`, `PairedAdvantageMarkdownTable`, `NotebookSpec`, `NotebookComposer` | `problems`, `runner`, `metrics` (consumes `StudyBundle` only) |

Automated AST boundary checks run via:

```bash
.venv/bin/python -m ceac_bench.cli check-boundaries
.venv/bin/python -m unittest discover -s tests/architecture -v
```

## 2. Persistent Function Classes & Content Addressing

Every cross-layer strategy (`ProblemBuilder`, `LearnerFactory`, `MetricObserver`, `CalibrationGate`, `Plotter`, `TableBuilder`) inherits from `FrozenFunctionClass` (`@dataclass(frozen=True)`):
- **Spawn-Worker Safe**: Pure top-level frozen dataclasses with zero closures or lambdas, pickling cleanly across `multiprocessing.get_context("spawn")`.
- **Self-Invalidating Content Hashes**: `fn.digest()` hashes `(type_id, module_source_sha256, canonical_params)`, so `CellCacheStore` automatically invalidates cached cells whenever parameters or implementation code change, while resuming unchanged cells instantaneously.

## 3. Unified CLI Entrypoints

```bash
# List registered suites and their four-layer specification digests
.venv/bin/python -m ceac_bench.cli list-suites

# Inspect any v2 or legacy v1 portable evidence bundle
.venv/bin/python -m ceac_bench.cli inspect-bundle notebooks/evidence/pathdep_undertow_falsebeacon

# Run a registered suite with spawn-pool isolation and automatic StudyBundle v2 + figure export
.venv/bin/python -m ceac_bench.cli run grid_pathdep --output runs/grid_pathdep_v2 --workers 4
```
