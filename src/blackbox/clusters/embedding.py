"""Embeddings for failure descriptions: `fastembed` (`BAAI/bge-small-en-v1.5`, 384 dimensions) on the CPU, in a
thread so the event loop never blocks. A hashing embedder (no model, no download) stands in for tests and machines
that can't fetch the model; the embedder in use is recorded with every clustering."""

import asyncio
import hashlib
import itertools
import logging
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

type Vectors = np.ndarray[Any, np.dtype[np.float32]]


class Embedder(Protocol):
    name: str
    dimensions: int

    async def embed(self, texts: list[str]) -> Vectors: ...


def _normalise(vectors: Vectors) -> Vectors:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Vectors = (vectors / norms).astype(np.float32)
    return out


class HashingEmbedder:
    """Bag of words and word pairs hashed into a fixed number of dimensions. Deterministic, instant, crude."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.name = f"hashing-{dimensions}"

    def _one(self, text: str) -> np.ndarray[Any, np.dtype[np.float32]]:
        words = re.findall(r"[a-z0-9_\-]+", text.lower())
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in words + [f"{a} {b}" for a, b in itertools.pairwise(words)]:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return vector

    async def embed(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return _normalise(np.stack([self._one(t) for t in texts]))


class FastEmbedder:
    """`fastembed`'s ONNX model, downloaded once into `cache_dir` and kept in memory."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: Path | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.name = f"fastembed:{model_name}"
        self.dimensions = 384
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, cache_dir=str(self.cache_dir) if self.cache_dir else None)
        return self._model

    def _embed_sync(self, texts: list[str]) -> Vectors:
        model = self._load()
        vectors = np.array(list(model.embed(texts)), dtype=np.float32)
        self.dimensions = int(vectors.shape[1]) if len(vectors) else self.dimensions
        return _normalise(vectors)

    async def embed(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return await asyncio.to_thread(self._embed_sync, texts)


def make_embedder(kind: str, model_name: str, cache_dir: Path | None) -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    return FastEmbedder(model_name, cache_dir)


def to_bytes(vector: np.ndarray[Any, Any]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_bytes(data: bytes) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.frombuffer(data, dtype=np.float32).copy()
