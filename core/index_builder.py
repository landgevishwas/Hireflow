"""Build the semantic FAISS index from resume PDFs."""

from __future__ import annotations

import json
from pathlib import Path

from config import get_settings
from core.ingestion import ingest_resume_directory
from embeddings import GeminiEmbeddingService
from utils import get_logger

logger = get_logger(__name__)


def build_resume_index() -> dict:
    """Embed all valid resumes and build the FAISS semantic index."""

    settings = get_settings(require_api_key=True)

    ingestion_result = ingest_resume_directory(
        settings.resume_dir
    )

    logger.info(
        "Resume ingestion complete: %d successful, %d failed.",
        len(ingestion_result.documents),
        len(ingestion_result.failures),
    )

    if not ingestion_result.documents:
        raise RuntimeError(
            "No valid resumes were found in the resume directory."
        )

    embedding_service = GeminiEmbeddingService()

    # Import here to avoid unnecessary initialization before
    # resume ingestion succeeds.
    from core.vector_store import FAISSVectorStore

    vector_store = FAISSVectorStore()
    vector_store.create()

    embedding_failures: list[dict] = []

    for position, document in enumerate(
        ingestion_result.documents,
        start=1,
    ):
        logger.info(
            "Embedding resume %d/%d: %s",
            position,
            len(ingestion_result.documents),
            document.filename,
        )

        try:
            vector = embedding_service.embed_document(
                document.text
            )

            metadata = {
                "candidate_id": document.candidate_id,
                "filename": document.filename,
                "source_path": document.source_path,
                "word_count": document.word_count,
            }

            vector_store.add(
                vector.reshape(1, -1),
                [metadata],
            )

        except Exception as exc:
            logger.error(
                "Embedding failed for %s: %s",
                document.filename,
                exc,
            )

            embedding_failures.append(
                {
                    "candidate_id": document.candidate_id,
                    "filename": document.filename,
                    "error": str(exc),
                }
            )

    if vector_store.size == 0:
        raise RuntimeError(
            "No resume embeddings were successfully generated."
        )

    vector_store.save()

    report = {
        "total_resumes_discovered": (
            len(ingestion_result.documents)
            + len(ingestion_result.failures)
        ),
        "successfully_ingested": len(
            ingestion_result.documents
        ),
        "ingestion_failures": [
            {
                "filename": failure.filename,
                "source_path": failure.source_path,
                "error": failure.error,
            }
            for failure in ingestion_result.failures
        ],
        "successfully_embedded": vector_store.size,
        "embedding_failures": embedding_failures,
        "index_path": str(vector_store.index_path),
        "metadata_path": str(vector_store.metadata_path),
    }

    report_path = (
        settings.hybrid_index_dir
        / "semantic_build_report.json"
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Semantic index built successfully: %d vectors.",
        vector_store.size,
    )

    return report


if __name__ == "__main__":
    result = build_resume_index()

    print("\n=== HireFlow Semantic Index ===")
    print(
        "Resumes discovered:",
        result["total_resumes_discovered"],
    )
    print(
        "Successfully ingested:",
        result["successfully_ingested"],
    )
    print(
        "Successfully embedded:",
        result["successfully_embedded"],
    )
    print(
        "Ingestion failures:",
        len(result["ingestion_failures"]),
    )
    print(
        "Embedding failures:",
        len(result["embedding_failures"]),
    )
    print("FAISS index:", result["index_path"])
    print("Metadata:", result["metadata_path"])