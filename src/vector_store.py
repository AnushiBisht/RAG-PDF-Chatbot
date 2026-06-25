"""
vector_store.py
Thin FAISS wrapper. FAISS only stores vectors, so metadata (text, page,
source file, content type) is kept in a parallel Python list and
saved/loaded alongside the index file.
"""

import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # vectors are pre-normalized -> cosine sim
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadatas: List[Dict[str, Any]]):
        if vectors.shape[0] == 0:
            return
        self.index.add(vectors)
        self.metadata.extend(metadatas)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector.reshape(1, -1), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str, dim: int) -> "VectorStore":
        store = cls(dim)
        index_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "metadata.pkl")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            store.index = faiss.read_index(index_path)
            with open(meta_path, "rb") as f:
                store.metadata = pickle.load(f)
        return store

    @property
    def count(self) -> int:
        return self.index.ntotal
