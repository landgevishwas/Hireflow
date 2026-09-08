"""Gemini embedding utilities for HireFlow."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from utils import get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when text embedding generation fails."""


class GeminiEmbeddingService:
    """Generate normalized text embeddings using Gemini."""

    def __init__(self) -> None:
        self.settings = get_settings(require_api_key=True)

        self.client = genai.Client(
            api_key=self.settings.gemini_api_key
        )

        self.model = self.settings.gemini_embedding_model
        self.dimension = self.settings.embedding_dimension

    def _validate_text(self, text: str) -> str:
        """Validate text before sending it to Gemini."""
        if not isinstance(text, str):
            raise EmbeddingError("Embedding input must be a string.")

        text = text.strip()

        if not text:
            raise EmbeddingError("Cannot create embedding for empty text.")

        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _request_embedding(
        self,
        text: str,
        task_type: str,
    ) -> list[float]:
        """Send one embedding request to Gemini."""

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimension,
            ),
        )

        if not response.embeddings:
            raise EmbeddingError(
                "Gemini returned no embedding."
            )

        values = response.embeddings[0].values

        if not values:
            raise EmbeddingError(
                "Gemini returned an empty embedding."
            )

        if len(values) != self.dimension:
            raise EmbeddingError(
                f"Expected embedding dimension "
                f"{self.dimension}, got {len(values)}."
            )

        return list(values)

    def embed_document(self, text: str) -> np.ndarray:
        """
        Embed resume/document text.

        RETRIEVAL_DOCUMENT tells Gemini that this text
        will be stored and later retrieved.
        """
        text = self._validate_text(text)

        try:
            values = self._request_embedding(
                text=text,
                task_type="RETRIEVAL_DOCUMENT",
            )
        except Exception as exc:
            logger.error("Document embedding failed: %s", exc)
            raise EmbeddingError(
                f"Failed to generate document embedding: {exc}"
            ) from exc

        return self._normalize(values)

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed search query / job description.

        RETRIEVAL_QUERY tells Gemini that this text is
        being used to retrieve relevant documents.
        """
        text = self._validate_text(text)

        try:
            values = self._request_embedding(
                text=text,
                task_type="RETRIEVAL_QUERY",
            )
        except Exception as exc:
            logger.error("Query embedding failed: %s", exc)
            raise EmbeddingError(
                f"Failed to generate query embedding: {exc}"
            ) from exc

        return self._normalize(values)

    def embed_documents(
        self,
        texts: Iterable[str],
    ) -> np.ndarray:
        """Embed multiple resume/document texts."""

        vectors: list[np.ndarray] = []

        for text in texts:
            vectors.append(self.embed_document(text))

        if not vectors:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        return np.vstack(vectors).astype(np.float32)

    @staticmethod
    def _normalize(values: list[float]) -> np.ndarray:
        """L2-normalize an embedding vector."""

        vector = np.asarray(values, dtype=np.float32)

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise EmbeddingError(
                "Cannot normalize a zero embedding vector."
            )

        return vector / norm