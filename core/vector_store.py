"""Persistent FAISS vector store for HireFlow."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from config import get_settings
from utils import get_logger


logger = get_logger(__name__)


class FAISSVectorStore:
    """Persistent cosine-similarity vector store using FAISS."""

    def __init__(
        self,
        index_dir: str | Path | None = None,
        dimension: int | None = None,
    ) -> None:

        settings = get_settings()

        self.index_dir = Path(
            index_dir or settings.hybrid_index_dir
        )

        self.dimension = (
            dimension or settings.embedding_dimension
        )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path = (
            self.index_dir / "semantic.index"
        )

        self.metadata_path = (
            self.index_dir
            / "semantic_metadata.json"
        )

        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

        if self.index_path.exists():
            self.load()
        else:
            self._create_empty_index()

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _create_empty_index(self) -> None:

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.metadata = []

    def _ensure_index(self) -> faiss.Index:

        if self.index is None:
            self._create_empty_index()

        assert self.index is not None

        return self.index

    # ------------------------------------------------------------------
    # ADD
    # ------------------------------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> None:
        """Add normalized embeddings and matching metadata."""

        if len(embeddings) == 0:
            return

        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        if len(metadata) != len(vectors):
            raise ValueError(
                "Number of metadata records must "
                "match number of embeddings."
            )

        # Prevent duplicate candidate IDs.
        candidate_ids = {
            str(item.get("candidate_id", ""))
            for item in metadata
        }

        existing_ids = {
            str(item.get("candidate_id", ""))
            for item in self.metadata
        }

        keep_indices = [
            index
            for index, item in enumerate(metadata)
            if str(item.get("candidate_id", ""))
            not in existing_ids
        ]

        if not keep_indices:
            logger.info(
                "All supplied candidates already exist "
                "in FAISS index."
            )
            return

        vectors = vectors[keep_indices]

        new_metadata = [
            metadata[index]
            for index in keep_indices
        ]

        # Normalize for cosine similarity via inner product.
        faiss.normalize_L2(vectors)

        index = self._ensure_index()

        index.add(vectors)

        self.metadata.extend(
            new_metadata
        )

        self.save()

        logger.info(
            "Added %s candidate(s) to FAISS.",
            len(new_metadata),
        )

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[dict]:
        """Search using cosine similarity."""

        index = self._ensure_index()

        if index.ntotal == 0:
            return []

        query = np.asarray(
            query_embedding,
            dtype="float32",
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.dimension:
            raise ValueError(
                "Query embedding dimension mismatch."
            )

        faiss.normalize_L2(query)

        k = min(
            top_k,
            index.ntotal,
        )

        scores, indices = index.search(
            query,
            k,
        )

        results = []

        for score, position in zip(
            scores[0],
            indices[0],
        ):

            if position < 0:
                continue

            if position >= len(self.metadata):
                continue

            item = dict(
                self.metadata[position]
            )

            item["score"] = float(score)

            results.append(item)

        return results

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def save(self) -> None:

        index = self._ensure_index()

        faiss.write_index(
            index,
            str(self.index_path),
        )

        self.metadata_path.write_text(
            json.dumps(
                self.metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self) -> None:

        if not self.index_path.exists():
            self._create_empty_index()
            return

        self.index = faiss.read_index(
            str(self.index_path)
        )

        if self.metadata_path.exists():

            self.metadata = json.loads(
                self.metadata_path.read_text(
                    encoding="utf-8"
                )
            )

        else:

            self.metadata = []

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:

        index = self._ensure_index()

        return int(index.ntotal)

    def contains(
        self,
        candidate_id: str,
    ) -> bool:

        return any(
            str(item.get("candidate_id"))
            == candidate_id
            for item in self.metadata
        )