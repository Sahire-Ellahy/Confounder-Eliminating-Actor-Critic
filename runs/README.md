# Local experiment cache

Raw immutable studies stay here and are excluded from Git.

Current results: `ceac_undertow_20261110` (336 runs) and
`ceac_prior_support_20261110` (48 new runs with reused controls).
Historical validation, deployment-comparison and preflight folders retain their
original paths and hashes. They are not silently merged into current evidence.

A fresh GitHub clone uses the verified compact export in `notebooks/evidence/`.
Use `python -m ceac.run --output runs/<new-name> ...` for a new study.
