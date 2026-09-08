"""Central configuration for HireFlow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """Application-wide immutable configuration."""

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------

    gemini_api_key: str = os.getenv(
        "GEMINI_API_KEY",
        "",
    )

    gemini_llm_model: str = os.getenv(
        "GEMINI_LLM_MODEL",
        "gemini-3.8-flash",
    )

    gemini_llm_fallback_model: str = os.getenv(
        "GEMINI_LLM_FALLBACK_MODEL",
        "gemini-3.6-flash",
    )

    gemini_embedding_model: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "gemini-embedding-2",
    )

    # ------------------------------------------------------------------
    # Hybrid retrieval
    # ------------------------------------------------------------------

    semantic_weight: float = _get_float(
        "SEMANTIC_WEIGHT",
        0.70,
    )

    keyword_weight: float = _get_float(
        "KEYWORD_WEIGHT",
        0.30,
    )

    semantic_top_k: int = _get_int(
        "SEMANTIC_TOP_K",
        20,
    )

    keyword_top_k: int = _get_int(
        "KEYWORD_TOP_K",
        20,
    )

    final_top_k: int = _get_int(
        "FINAL_TOP_K",
        10,
    )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    embedding_dimension: int = _get_int(
        "EMBEDDING_DIMENSION",
        768,
    )

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------

    resume_dir: Path = PROJECT_ROOT / os.getenv(
        "RESUME_DIR",
        "data/resumes",
    )

    hybrid_index_dir: Path = PROJECT_ROOT / os.getenv(
        "HYBRID_INDEX_DIR",
        "data/hybrid_index",
    )

    memory_dir: Path = PROJECT_ROOT / os.getenv(
        "MEMORY_DIR",
        "data/memory",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_level: str = os.getenv(
        "LOG_LEVEL",
        "INFO",
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
        require_api_key: bool = False,
    ) -> None:

        if not 0 <= self.semantic_weight <= 1:
            raise ValueError(
                "SEMANTIC_WEIGHT must be between 0 and 1."
            )

        if not 0 <= self.keyword_weight <= 1:
            raise ValueError(
                "KEYWORD_WEIGHT must be between 0 and 1."
            )

        total_weight = (
            self.semantic_weight
            + self.keyword_weight
        )

        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(
                "SEMANTIC_WEIGHT + KEYWORD_WEIGHT "
                "must equal 1.0."
            )

        positive_values = {
            "SEMANTIC_TOP_K": self.semantic_top_k,
            "KEYWORD_TOP_K": self.keyword_top_k,
            "FINAL_TOP_K": self.final_top_k,
            "EMBEDDING_DIMENSION": self.embedding_dimension,
        }

        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(
                    f"{name} must be greater than 0."
                )

        if not self.gemini_llm_model.strip():
            raise ValueError(
                "GEMINI_LLM_MODEL cannot be empty."
            )

        if require_api_key and not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Copy .env.example to .env and add your API key."
            )


settings = Settings()


def ensure_directories() -> None:
    """Create required application directories."""

    settings.resume_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.hybrid_index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    settings.memory_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def get_settings(
    require_api_key: bool = False,
) -> Settings:
    """Return validated application settings."""

    settings.validate(
        require_api_key=require_api_key,
    )

    return settings