"""Immutable run identity and streaming evidence, separate from the learner."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import scipy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def encoded(value):
    return json.dumps(jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(encoded(value))
    temporary.replace(path)


def source_identity():
    root = Path(__file__).resolve().parent
    renderers = {"report.py", "visualization.py"}
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.py"))
            if str(p.relative_to(root)) not in renderers}


def manifest(configuration):
    bank_method = configuration.get("method") in {"policy_ucb", "oir_idac", "coverage"}
    body = {"schema": 3, "configuration": configuration, "sources": source_identity(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "algorithm_scope": "v90 finite-model CEAC specialization; exact evaluation and response certificates",
            "policy_class": ("shared enriched public bank" if bank_method else
                             "all mask-respecting finite-horizon Markov policies"),
            "stationarity": "fixed model within each visible context"}
    body["identity"] = hashlib.sha256(encoded(body).encode()).hexdigest()
    return body


def verify_run(folder, expected=None):
    folder = Path(folder)
    recorded = json.loads((folder / "manifest.json").read_text())
    identity = recorded.pop("identity")
    if hashlib.sha256(encoded(recorded).encode()).hexdigest() != identity:
        raise ValueError("Manifest digest mismatch")
    recorded["identity"] = identity
    if expected is not None and recorded != jsonable(expected):
        raise ValueError("Existing run has a different source/config/seed/prior/runtime identity")
    status = json.loads((folder / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError("Existing run is incomplete; preserve it and use a fresh output directory")
    for name, checksum in status["artifacts"].items():
        target = (folder / name).resolve()
        if not target.is_relative_to(folder.resolve()) or digest(target) != checksum:
            raise ValueError("Run artifact mismatch: " + name)
    metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    episodes = [json.loads(line) for line in (folder / "trajectories.jsonl").read_text().splitlines()]
    if len(metrics) != status["episodes"] or len(episodes) != len(metrics):
        raise ValueError("Run completion count mismatch")
    if [r["episode"] for r in metrics] != list(range(len(metrics))):
        raise ValueError("Episode index is not contiguous")
    for episode in episodes:
        if not (folder / "policies" / (episode["policy_id"] + ".npz")).is_file():
            raise ValueError("Missing frozen behavior policy")
    return {"manifest": recorded, "status": status, "metrics": metrics,
            "result": json.loads((folder / "result.json").read_text())}


class RunWriter:
    def __init__(self, folder, identity, model):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        (self.folder / "policies").mkdir()
        atomic_json(self.folder / "manifest.json", identity)
        atomic_json(self.folder / "status.json", {"status": "running"})
        np.savez_compressed(self.folder / "state_index.npz", state_ids=model.state_ids, action_mask=model.mask)
        self.metrics = (self.folder / "metrics.jsonl").open("x")
        self.trajectories = (self.folder / "trajectories.jsonl").open("x")
        self.count = 0

    def policy(self, policy):
        path = self.folder / "policies" / (policy.identity + ".npz")
        if not path.exists():
            np.savez_compressed(path, probabilities=policy.probabilities)

    def episode(self, episode, metric):
        self.trajectories.write(json.dumps(jsonable(asdict(episode)), sort_keys=True, allow_nan=False) + "\n")
        self.metrics.write(json.dumps(jsonable(metric), sort_keys=True, allow_nan=False) + "\n")
        self.trajectories.flush(); self.metrics.flush()
        self.count += 1

    def finish(self, result, events):
        self.metrics.close(); self.trajectories.close()
        atomic_json(self.folder / "result.json", result)
        atomic_json(self.folder / "audit_events.json", events)
        artifacts = {str(p.relative_to(self.folder)): digest(p) for p in sorted(self.folder.rglob("*"))
                     if p.is_file() and p.name != "status.json"}
        atomic_json(self.folder / "status.json", {"status": "complete", "episodes": self.count, "artifacts": artifacts})

    def fail(self, error):
        self.metrics.close(); self.trajectories.close()
        atomic_json(self.folder / "status.json", {"status": "failed", "episodes": self.count,
                                                "error": type(error).__name__ + ": " + str(error)})
