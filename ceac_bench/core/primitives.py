"""Self-describing persistent function classes, content digests, and deterministic RNG streams."""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
_MODULE_SHA_CACHE: dict[str, str] = {}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sanitize_json_value(value: Any) -> Any:
    """Convert dataclasses, tuples, paths, numpy scalars/arrays, and PersistentCallables into JSON-safe values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        py_val = value.item()
        if isinstance(py_val, float) and not np.isfinite(py_val):
            return None
        return py_val
    if isinstance(value, np.ndarray):
        return [sanitize_json_value(v) for v in value.tolist()]
    if isinstance(value, Path):
        p = value.resolve()
        return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    if isinstance(value, FrozenFunctionClass):
        return value.to_spec()
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: sanitize_json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): sanitize_json_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(v) for v in value]
    if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    raise TypeError(f"Unsupported value in persistent spec: {type(value)!r}")


def canonical_json(value: Any, *, indent: int | None = 2) -> str:
    clean = sanitize_json_value(value)
    if indent is None:
        return json.dumps(clean, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return json.dumps(clean, sort_keys=True, indent=indent, allow_nan=False) + "\n"


def atomic_write_json(path: Path | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + ".pending")
    pending.write_text(canonical_json(value, indent=2))
    pending.replace(target)


def seed_for(*parts: Any) -> int:
    """Deterministic 64-bit integer seed compatible with existing suite RNG derivations."""
    payload = json.dumps(list(parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _module_source_sha256(cls: type) -> str:
    mod_name = cls.__module__
    if mod_name in _MODULE_SHA_CACHE:
        return _MODULE_SHA_CACHE[mod_name]
    src_file = inspect.getsourcefile(cls)
    sha = digest_file(src_file) if src_file and Path(src_file).is_file() else "dynamic"
    _MODULE_SHA_CACHE[mod_name] = sha
    return sha


def _mro_module_sha256(cls: type) -> list[str]:
    """Hash every defining module along `cls.__mro__` so base-class edits also invalidate caches."""
    out: list[str] = []
    seen: set[str] = set()
    for base in cls.__mro__:
        if base is object:
            continue
        mod = base.__module__
        if mod in seen:
            continue
        seen.add(mod)
        out.append(f"{mod}:{_module_source_sha256(base)}")
    return out


@runtime_checkable
class PersistentCallable(Protocol):
    """Picklable, self-identifying function object safe for spawn workers and SHA-256 manifests."""

    @property
    def type_id(self) -> str: ...

    def to_spec(self) -> dict[str, Any]: ...

    def digest(self) -> str: ...


@dataclass(frozen=True)
class FrozenFunctionClass:
    """Base mixin for immutable, picklable persistent function classes passed across benchmark layers.

    Subclasses are frozen dataclasses implementing ``__call__`` (or domain methods).
    Their ``digest()`` captures class identity, all serialized configuration fields,
    and the SHA-256 of the defining Python module so caches invalidate automatically
    whenever parameters or implementation code change.
    """

    @property
    def type_id(self) -> str:
        cls = type(self)
        return f"{cls.__module__}.{cls.__qualname__}"

    def to_spec(self) -> dict[str, Any]:
        params = {
            f.name: sanitize_json_value(getattr(self, f.name))
            for f in fields(self)
            if not f.name.startswith("_")
        }
        return {
            "type_id": self.type_id,
            "mro_sha256": _mro_module_sha256(type(self)),
            "params": params,
        }

    def digest(self) -> str:
        return digest_bytes(canonical_json(self.to_spec(), indent=None).encode("utf-8"))


@dataclass(frozen=True)
class RNGStreamSet:
    """Isolated deterministic RNG seeds for environment transitions, learner, action sampling, and held-out audits."""

    env_seed: int
    learner_seed: int
    action_seed: int
    eval_seed_prefix: tuple[Any, ...]

    @classmethod
    def from_parts(
        cls,
        namespace: str,
        seed: int,
        variant_id: str,
        dose: int | float,
        task: int,
        method_id: str,
    ) -> "RNGStreamSet":
        return cls(
            env_seed=seed_for(namespace, seed, variant_id, dose, task, "shared_environment"),
            learner_seed=seed_for(namespace, seed, variant_id, dose, task, method_id, "learner"),
            action_seed=seed_for(namespace, seed, variant_id, dose, task, method_id, "action"),
            eval_seed_prefix=("critic_eval", namespace, variant_id, method_id, int(seed), dose),
        )

    def make_learner_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.learner_seed)

    def make_action_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.action_seed)

    def make_checkpoint_eval_rng(self, checkpoint: int) -> np.random.Generator:
        return np.random.default_rng(seed_for(*self.eval_seed_prefix, int(checkpoint)))
