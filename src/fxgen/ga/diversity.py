"""Phenotypic diversity: fitness sharing on the similarity of trade signals.

Two trees are similar when they hold the same position on the same bars.
Similarity is a directional Jaccard index on sub-sampled decision bars:
``|bars where both are long or both are short| / |bars where either trades|``.
"""

from __future__ import annotations

import numpy as np

PHENOTYPE_STRIDE = 24


def phenotype(actions: list[np.ndarray]) -> np.ndarray:
    """Concatenate sub-sampled per-pair action arrays (int8)."""
    return np.concatenate([a[::PHENOTYPE_STRIDE] for a in actions]).astype(np.int8)


def similarity_matrix(phenotypes: np.ndarray) -> np.ndarray:
    """Pairwise directional Jaccard similarity of a (pop, bars) int8 matrix."""
    long = (phenotypes > 0).astype(np.float32)
    short = (phenotypes < 0).astype(np.float32)
    active = long + short
    same = long @ long.T + short @ short.T
    both_active = active @ active.T
    n_active = active.sum(axis=1)
    union = n_active[:, None] + n_active[None, :] - both_active
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(union > 0, same / union, 1.0)
    return sim


def shared_fitness(raw: np.ndarray, phenotypes: np.ndarray, radius: float) -> np.ndarray:
    """Divide positive fitness by the niche count (similarity above ``radius``)."""
    sim = similarity_matrix(phenotypes)
    sigma = 1.0 - radius
    distance = 1.0 - sim
    share = np.where(distance < sigma, 1.0 - distance / sigma, 0.0) if sigma > 0 else (distance == 0).astype(float)
    niche = share.sum(axis=1)
    return np.where(raw > 0, raw / niche, raw * niche)
