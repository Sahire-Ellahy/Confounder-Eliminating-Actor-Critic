"""Horlbeck et al. (2018, Cell 174:953) K562 CRISPRi gene-level GI map: loading and panel selection.

Source: Mendeley Data doi:10.17632/rdzk59n6j4.1, file GI_map_treeview.zip
(published SHA-256 d389b3eb...f9115). The archive is downloaded once into
data/external/horlbeck2018/ and verified here; nothing is fetched at run time.

Only the published gene-by-gene GI scores are used. Single-gene growth
phenotypes are not in this archive, so any main-effect nuisance used by the
benchmark is explicitly synthetic and labelled as such.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import hashlib
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/horlbeck2018/GI_map_treeview.zip"
ARCHIVE_SHA256 = "d389b3ebdd2c88f86cc446f272f3ff6526df635d666cd17d67440bf10c0f9115"
MEMBER = "GI_map_treeview/K562 gene-level map/K562_gene.cdt"
SOURCE = "Horlbeck et al. 2018 Cell 174:953-967; Mendeley Data 10.17632/rdzk59n6j4.1; K562 CRISPRi gene-level GI map"


@lru_cache(maxsize=1)
def load_k562_map() -> tuple[tuple[str, ...], np.ndarray]:
    """Return (gene names, symmetric GI matrix) after checksum verification."""
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"{ARCHIVE} missing. Download GI_map_treeview.zip from doi:10.17632/rdzk59n6j4.1 into that path.")
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"GI map archive checksum mismatch: {digest}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        lines = archive.read(MEMBER).decode().splitlines()
    columns = lines[0].split("\t")[4:]
    rows = [line.split("\t") for line in lines[1:] if line.startswith("GENE")]
    names = tuple(r[1] for r in rows)
    if list(names) != columns:
        raise ValueError("Unexpected row/column ordering in the GI map")
    matrix = np.array([[float(v) for v in r[4:]] for r in rows])
    if matrix.shape != (len(names), len(names)) or not np.allclose(matrix, matrix.T) or not np.isfinite(matrix).all():
        raise ValueError("GI map must be a finite symmetric gene-by-gene matrix")
    return names, matrix


def panel_values(genes: tuple[str, ...]) -> tuple[float, ...]:
    names, matrix = load_k562_map()
    index = [names.index(g) for g in genes]
    return tuple(float(matrix[index[i], index[j]]) for i, j in combinations(range(len(genes)), 2))


def select_panel(size: int = 10, strong: float = -5.0, background_floor: float = -2.5, seed: int = 2018,
                 attempts: int = 20000) -> dict:
    """Deterministic public rule: one real strong negative pair plus genes whose
    other pairs all exceed `background_floor`. Pair order is itertools.combinations."""
    names, matrix = load_k562_map()
    rng = np.random.default_rng(seed)
    n = len(names)
    upper = np.triu_indices(n, 1)
    strong_pairs = [(int(i), int(j)) for i, j in zip(*upper) if matrix[i, j] <= strong]
    order = rng.permutation(len(strong_pairs))
    for idx in order:
        a, b = strong_pairs[idx]
        chosen = [a, b]
        candidates = [g for g in rng.permutation(n) if g not in chosen]
        for g in candidates:
            if all(matrix[g, c] > background_floor for c in chosen):
                chosen.append(int(g))
            if len(chosen) == size:
                break
        if len(chosen) == size:
            genes = tuple(names[g] for g in chosen)
            values = panel_values(genes)
            pairs = list(combinations(range(size), 2))
            sl = pairs.index((0, 1))
            assert sum(v <= strong for v in values) == 1 and values[sl] <= strong
            return dict(genes=genes, values=values, sl_pair_index=sl, strong_value=values[sl])
    raise RuntimeError("No panel satisfies the public selection rule")


def panel_base_rate(size: int = 10, strong: float = -5.0, draws: int = 20000, seed: int = 7) -> float:
    """Empirical probability that a random panel of `size` map genes contains >=1 strong negative pair."""
    names, matrix = load_k562_map()
    rng = np.random.default_rng(seed)
    hits = 0
    iu = np.triu_indices(size, 1)
    for _ in range(draws):
        g = rng.choice(len(names), size=size, replace=False)
        hits += bool((matrix[np.ix_(g, g)][iu] <= strong).any())
    return hits / draws
