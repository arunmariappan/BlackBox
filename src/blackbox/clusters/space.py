"""The clustering itself: feature vectors, HDBSCAN, centres and radii, live assignment, and stable ids.

A failure's vector is its normalised description embedding next to a weighted one-hot part (ending, category,
metric flags, node of the first problem). HDBSCAN (`min_cluster_size = 3`) needs no number of groups in advance and
leaves odd failures unclustered rather than forcing them into a group.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN

type Array = np.ndarray[Any, np.dtype[np.float32]]


def tokens_of(signature: dict[str, Any], category: str | None) -> list[str]:
    tokens = [f"ending:{signature.get('ending')}"]
    if category:
        tokens.append(f"category:{category}")
    tokens.extend(f"flag:{flag}" for flag in signature.get("flags") or [])
    if signature.get("first_problem_node"):
        tokens.append(f"node:{signature['first_problem_node']}")
    if signature.get("failed_judges"):
        tokens.extend(f"judge:{name}" for name in signature["failed_judges"])
    return tokens


@dataclass
class Space:
    vocabulary: list[str]
    weight: float

    @classmethod
    def build(cls, token_lists: list[list[str]], weight: float) -> Space:
        return cls(sorted({token for tokens in token_lists for token in tokens}), weight)

    def onehot(self, tokens: list[str]) -> Array:
        index = {token: i for i, token in enumerate(self.vocabulary)}
        vector = np.zeros(len(self.vocabulary), dtype=np.float32)
        for token in tokens:
            if token in index:
                vector[index[token]] = 1.0
        norm = float(np.linalg.norm(vector))
        out: Array = (vector / norm if norm else vector) * self.weight
        return out.astype(np.float32)

    def vector(self, embedding: Array, tokens: list[str]) -> Array:
        out: Array = np.concatenate([embedding.astype(np.float32), self.onehot(tokens)])
        return out


def cluster_labels(vectors: Array, min_cluster_size: int = 3) -> list[int]:
    """Cluster labels per vector; -1 means unclustered."""
    if len(vectors) < min_cluster_size:
        return [-1] * len(vectors)
    model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", allow_single_cluster=True, copy=True)
    labels = model.fit_predict(np.asarray(vectors, dtype=np.float64))
    return [int(label) for label in labels]


def centre_and_radius(members: Array) -> tuple[Array, float]:
    """The mean of the members and the 95th percentile of their distances to it."""
    centre: Array = members.mean(axis=0).astype(np.float32)
    distances = np.linalg.norm(members - centre, axis=1)
    radius = float(np.percentile(distances, 95)) if len(distances) else 0.0
    return centre, max(radius, 1e-6)


def nearest(vector: Array, centres: list[tuple[str, Array, float]]) -> tuple[str | None, float]:
    best: tuple[str | None, float] = (None, float("inf"))
    for cluster_id, centre, _ in centres:
        if len(centre) != len(vector):
            continue
        distance = float(np.linalg.norm(vector - centre))
        if distance < best[1]:
            best = (cluster_id, distance)
    return best


def assign(vector: Array, centres: list[tuple[str, Array, float]]) -> str | None:
    """The nearest cluster, if the vector is within its radius; otherwise None (stays unclustered)."""
    cluster_id, distance = nearest(vector, centres)
    if cluster_id is None:
        return None
    radius = next(r for cid, _, r in centres if cid == cluster_id)
    return cluster_id if distance <= radius else None


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def match_ids(old: dict[str, set[str]], new: list[set[str]], threshold: float = 0.5) -> list[str | None]:
    """For each new cluster, the old id it inherits: the old cluster it shares the most members with (Jaccard ≥
    threshold), each old id used once, best matches first. A split keeps the id on its larger part."""
    pairs = sorted(
        (
            (jaccard(members, old_members), len(members), i, old_id)
            for i, members in enumerate(new)
            for old_id, old_members in old.items()
        ),
        key=lambda item: (-item[0], -item[1], item[2], item[3]),
    )
    out: list[str | None] = [None] * len(new)
    used: set[str] = set()
    for score, _, i, old_id in pairs:
        if score < threshold or out[i] is not None or old_id in used:
            continue
        out[i] = old_id
        used.add(old_id)
    return out
