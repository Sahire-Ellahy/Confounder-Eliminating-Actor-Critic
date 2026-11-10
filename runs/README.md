# Local experiment cache

Raw immutable studies stay here and are excluded from Git.

Current physical-strength sweep: pathology_strength_20261110.
It contains 336 cells at 2,048 episodes per cell, with method overlays,
strength traces, paired added regret and mechanism diagnostics emitted by the
standard report renderer. The notebook export points to this study.
See the [protocol](../docs/pathology_sweep.md) for the exact controls.

Earlier comparator integration check: `primitive_comparator_validation_20261110`
(48 runs, 128 episodes, two environments, two modes, two seeds; CEAC, offline
oracle, IDS, IDAC, UCB Q-learning and optimistic POMD). Its report includes
update counters and POMD bonus-saturation diagnostics. This is not a convergence
or pathology-strength experiment.

The preceding `source_comparator_smoke_20261110` is the completed five-method
integration check before POMD was added. Its exact source is archived separately.

Historical studies: `ceac_undertow_20261110` (336 original runs) and
`ceac_prior_support_20261110` (48 additional runs). Their raw records are
unchanged; retired custom controls are excluded from current reports/notebooks.
The cached M sweep measures nuisance alphabet diversity, not physical strength.

New runs snapshot their source at launch. Fresh output directories prevent
historical source/config/seed identities from being overwritten. The notebook's
portable export has no dependency on these local raw directories.
