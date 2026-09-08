"""Hybrid semantic + keyword candidate index for HireFlow."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from config import get_settings
from core.parsing import extract_pdf_text
from embeddings import GeminiEmbeddingService
from core.vector_store import FAISSVectorStore
from utils import get_logger


logger = get_logger(__name__)


# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25."""

    return re.findall(
        r"\b[a-zA-Z0-9+#./-]+\b",
        text.lower(),
    )


def min_max_normalize(
    scores: list[float],
) -> list[float]:
    """Normalize scores into [0, 1]."""

    if not scores:
        return []

    values = np.asarray(
        scores,
        dtype=float,
    )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        if maximum == 0:
            return [0.0] * len(values)

        return [1.0] * len(values)

    normalized = (
        values - minimum
    ) / (
        maximum - minimum
    )

    return normalized.tolist()


# ============================================================================
# RESULT
# ============================================================================

class HybridSearchResult:
    """Single hybrid-search result."""

    def __init__(
        self,
        candidate_id: str,
        hybrid_score: float,
        semantic_score: float,
        keyword_score: float,
    ) -> None:

        self.candidate_id = candidate_id
        self.hybrid_score = hybrid_score
        self.semantic_score = semantic_score
        self.keyword_score = keyword_score

    def to_dict(self) -> dict:

        return {
            "candidate_id": self.candidate_id,
            "hybrid_score": self.hybrid_score,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
        }


# ============================================================================
# HYBRID INDEXER
# ============================================================================

class HybridIndexer:
    """Manage FAISS + BM25 candidate retrieval."""

    def __init__(self) -> None:

        settings = get_settings()

        self.settings = settings

        self.vector_store = FAISSVectorStore()

        self.embedding_service = (
            GeminiEmbeddingService()
        )

        self.keyword_index_path = (
            settings.hybrid_index_dir
            / "hybrid_index.json"
        )

        self.documents: list[dict] = []

        self.bm25: BM25Okapi | None = None

        self.load_keyword_index()

    # ------------------------------------------------------------------
    # KEYWORD INDEX
    # ------------------------------------------------------------------

    def build_keyword_index(self) -> int:
        """Build BM25 from all PDFs currently in resume directory."""

        documents = []

        resume_dir = Path(
            self.settings.resume_dir
        )

        resume_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for pdf_path in sorted(
            resume_dir.glob("*.pdf")
        ):

            try:

                text = extract_pdf_text(
                    pdf_path
                )

                documents.append(
                    {
                        "candidate_id": pdf_path.stem,
                        "text": text,
                        "tokens": tokenize(text),
                    }
                )

            except Exception as exc:

                logger.warning(
                    "Could not index %s: %s",
                    pdf_path.name,
                    exc,
                )

        self.documents = documents

        self._rebuild_bm25()

        self._save_keyword_index()

        return len(documents)

    def _rebuild_bm25(self) -> None:

        if not self.documents:

            self.bm25 = None
            return

        tokenized_documents = [
            item["tokens"]
            for item in self.documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    def _save_keyword_index(self) -> None:

        self.keyword_index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = [
            {
                "candidate_id":
                    item["candidate_id"],
                "text":
                    item["text"],
            }
            for item in self.documents
        ]

        self.keyword_index_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load_keyword_index(self) -> None:

        if not self.keyword_index_path.exists():

            self.documents = []
            self.bm25 = None

            return

        try:

            payload = json.loads(
                self.keyword_index_path.read_text(
                    encoding="utf-8"
                )
            )

            self.documents = []

            for item in payload:

                text = str(
                    item.get(
                        "text",
                        "",
                    )
                )

                self.documents.append(
                    {
                        "candidate_id":
                            str(
                                item.get(
                                    "candidate_id",
                                    "",
                                )
                            ),
                        "text": text,
                        "tokens": tokenize(text),
                    }
                )

            self._rebuild_bm25()

        except Exception as exc:

            logger.warning(
                "Could not load BM25 index: %s",
                exc,
            )

            self.documents = []
            self.bm25 = None

    # ------------------------------------------------------------------
    # ADD ONE RESUME
    # ------------------------------------------------------------------

    def add_resume(
        self,
        pdf_path: str | Path,
        candidate_id: str | None = None,
    ) -> None:
        """
        Add one resume to both semantic and keyword indexes.
        """

        pdf_path = Path(pdf_path)

        if candidate_id is None:
            candidate_id = pdf_path.stem

        text = extract_pdf_text(
            pdf_path
        )

        # --------------------------------------------------------------
        # FAISS
        # --------------------------------------------------------------

        if not self.vector_store.contains(
            candidate_id
        ):

            embedding = (
                self.embedding_service.embed_document(
                    text
                )
            )

            vector = np.asarray(
                embedding,
                dtype="float32",
            )

            self.vector_store.add(
                embeddings=vector.reshape(1, -1),
                metadata=[
                    {
                        "candidate_id":
                            candidate_id,
                        "source":
                            str(pdf_path),
                    }
                ],
            )

        # --------------------------------------------------------------
        # BM25 DOCUMENT
        # --------------------------------------------------------------

        existing_index = next(
            (
                item
                for item in self.documents
                if item["candidate_id"]
                == candidate_id
            ),
            None,
        )

        if existing_index is None:

            self.documents.append(
                {
                    "candidate_id":
                        candidate_id,
                    "text":
                        text,
                    "tokens":
                        tokenize(text),
                }
            )

        else:

            existing_index["text"] = text
            existing_index["tokens"] = tokenize(
                text
            )

        self._rebuild_bm25()
        self._save_keyword_index()

        logger.info(
            "Resume indexed successfully: %s",
            candidate_id,
        )

    # ------------------------------------------------------------------
    # ADD DIRECTORY
    # ------------------------------------------------------------------

    def add_all_resumes(self) -> int:
        """Index every PDF in the resume directory."""

        count = 0

        resume_dir = Path(
            self.settings.resume_dir
        )

        for pdf_path in sorted(
            resume_dir.glob("*.pdf")
        ):

            try:

                self.add_resume(
                    pdf_path
                )

                count += 1

            except Exception as exc:

                logger.warning(
                    "Failed to index %s: %s",
                    pdf_path.name,
                    exc,
                )

        return count

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def _semantic_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:

        query_embedding = (
            self.embedding_service.embed_query(
                query
            )
        )

        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict]:

        if self.bm25 is None:
            return []

        query_tokens = tokenize(
            query
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            if index >= len(
                self.documents
            ):
                continue

            results.append(
                {
                    "candidate_id":
                        self.documents[index][
                            "candidate_id"
                        ],
                    "score":
                        float(scores[index]),
                }
            )

        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[HybridSearchResult]:
        """Perform normalized semantic + keyword search."""

        if not query.strip():
            return []

        retrieval_k = max(
            top_k,
            self.settings.semantic_top_k,
            self.settings.keyword_top_k,
        )

        semantic_results = (
            self._semantic_search(
                query,
                retrieval_k,
            )
        )

        keyword_results = (
            self._keyword_search(
                query,
                retrieval_k,
            )
        )

        semantic_raw = [
            float(item["score"])
            for item in semantic_results
        ]

        keyword_raw = [
            float(item["score"])
            for item in keyword_results
        ]

        semantic_normalized = (
            min_max_normalize(
                semantic_raw
            )
        )

        keyword_normalized = (
            min_max_normalize(
                keyword_raw
            )
        )

        semantic_map = {
            item["candidate_id"]:
                semantic_normalized[index]
            for index, item in enumerate(
                semantic_results
            )
        }

        keyword_map = {
            item["candidate_id"]:
                keyword_normalized[index]
            for index, item in enumerate(
                keyword_results
            )
        }

        candidate_ids = (
            set(semantic_map)
            | set(keyword_map)
        )

        results = []

        for candidate_id in candidate_ids:

            semantic_score = float(
                semantic_map.get(
                    candidate_id,
                    0.0,
                )
            )

            keyword_score = float(
                keyword_map.get(
                    candidate_id,
                    0.0,
                )
            )

            hybrid_score = (
                self.settings.semantic_weight
                * semantic_score
                +
                self.settings.keyword_weight
                * keyword_score
            )

            results.append(
                HybridSearchResult(
                    candidate_id=candidate_id,
                    hybrid_score=hybrid_score,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                )
            )

        results.sort(
            key=lambda item:
                item.hybrid_score,
            reverse=True,
        )

        return results[:top_k]

    # ------------------------------------------------------------------
    # INFO
    # ------------------------------------------------------------------

    @property
    def keyword_index_size(self) -> int:

        return len(
            self.documents
        )

    @property
    def semantic_index_size(self) -> int:

        return self.vector_store.size


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    indexer = HybridIndexer()

    count = indexer.add_all_resumes()

    print(
        f"Indexed resumes: {count}"
    )

    print(
        f"FAISS size: "
        f"{indexer.semantic_index_size}"
    )

    print(
        f"BM25 size: "
        f"{indexer.keyword_index_size}"
    )