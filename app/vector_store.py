import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    source: str
    similarity: float


class FaissVectorStore:
    def __init__(self, index_dir: Path, embedding_model: str, openai_api_key: str) -> None:
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=openai_api_key)
        self.index: faiss.IndexFlatIP | None = None
        self.metadata: list[dict] = []

    def build(self, chunks: list[str], source: str) -> None:
        embeddings = self._embed_texts(chunks)
        normalized = self._normalize(embeddings)

        dim = normalized.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(normalized)
        self.metadata = [
            {"chunk_id": idx, "text": text, "source": source}
            for idx, text in enumerate(chunks)
        ]
        self._persist()

    def load(self) -> bool:
        index_file = self.index_dir / "index.faiss"
        metadata_file = self.index_dir / "metadata.json"
        if not index_file.exists() or not metadata_file.exists():
            return False

        self.index = faiss.read_index(str(index_file))
        self.metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        return True

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self.index is None or not self.metadata:
            return []

        query_vector = self._embed_texts([query])
        query_vector = self._normalize(query_vector)

        scores, indices = self.index.search(query_vector, top_k)

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=int(meta["chunk_id"]),
                    text=str(meta["text"]),
                    source=str(meta["source"]),
                    similarity=float(score),
                )
            )
        return results

    def _persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            return
        faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        (self.index_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=True, indent=2), encoding="utf-8"
        )

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.embedding_model, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms
