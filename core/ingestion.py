"""Resume discovery and ingestion pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from core.parsing import (
    ResumeParsingError,
    extract_pdf_text,
)
from utils import discover_files, get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class ResumeDocument:
    """Generic representation of an ingested resume.

    This is intentionally not the final Pydantic Candidate schema.
    The canonical schemas will be introduced in Phase 2.
    """

    candidate_id: str
    source_path: str
    filename: str
    text: str
    word_count: int


@dataclass(frozen=True)
class IngestionFailure:
    """Information about a resume that could not be ingested."""

    source_path: str
    filename: str
    error: str


@dataclass(frozen=True)
class IngestionResult:
    """Result of an ingestion run."""

    documents: tuple[ResumeDocument, ...]
    failures: tuple[IngestionFailure, ...]


def generate_candidate_id(
    pdf_path: Path,
) -> str:
    """Generate a deterministic candidate ID from the filename.

    Important:
    This is only a temporary generic strategy.

    Once the real dataset is inspected, we will prefer an existing
    candidate ID from the dataset if one exists.
    """

    return pdf_path.stem.strip()


def ingest_resume(
    pdf_path: Path,
) -> ResumeDocument:
    """Parse and create an internal representation of one resume."""

    text = extract_pdf_text(pdf_path)

    word_count = len(text.split())

    if word_count == 0:
        raise ResumeParsingError(
            f"Resume contains no usable words: {pdf_path.name}"
        )

    candidate_id = generate_candidate_id(pdf_path)

    if not candidate_id:
        raise ResumeParsingError(
            f"Could not generate candidate ID: {pdf_path.name}"
        )

    return ResumeDocument(
        candidate_id=candidate_id,
        source_path=str(pdf_path.resolve()),
        filename=pdf_path.name,
        text=text,
        word_count=word_count,
    )


def ingest_resume_directory(
    resume_directory: Path,
) -> IngestionResult:
    """Discover and ingest all PDF resumes in a directory.

    Individual failures do not terminate the entire ingestion run.
    """

    pdf_files = discover_files(
        resume_directory,
        extensions={".pdf"},
    )

    logger.info(
        "Discovered %d PDF resume(s).",
        len(pdf_files),
    )

    documents: list[ResumeDocument] = []
    failures: list[IngestionFailure] = []

    seen_candidate_ids: set[str] = set()

    for pdf_path in pdf_files:
        try:
            document = ingest_resume(pdf_path)

            if document.candidate_id in seen_candidate_ids:
                failures.append(
                    IngestionFailure(
                        source_path=str(pdf_path),
                        filename=pdf_path.name,
                        error=(
                            "Duplicate candidate ID: "
                            f"{document.candidate_id}"
                        ),
                    )
                )

                logger.warning(
                    "Duplicate candidate ID detected: %s",
                    document.candidate_id,
                )

                continue

            seen_candidate_ids.add(
                document.candidate_id
            )

            documents.append(document)

            logger.info(
                "Successfully ingested: %s",
                pdf_path.name,
            )

        except ResumeParsingError as exc:
            failure = IngestionFailure(
                source_path=str(pdf_path),
                filename=pdf_path.name,
                error=str(exc),
            )

            failures.append(failure)

            logger.error(
                "Failed to ingest %s: %s",
                pdf_path.name,
                exc,
            )

        except Exception as exc:
            failure = IngestionFailure(
                source_path=str(pdf_path),
                filename=pdf_path.name,
                error=f"Unexpected error: {exc}",
            )

            failures.append(failure)

            logger.exception(
                "Unexpected ingestion error for %s",
                pdf_path.name,
            )

    return IngestionResult(
        documents=tuple(documents),
        failures=tuple(failures),
    )


def ingestion_result_to_dict(
    result: IngestionResult,
) -> dict:
    """Convert ingestion results to a JSON-friendly dictionary."""

    return {
        "documents": [
            asdict(document)
            for document in result.documents
        ],
        "failures": [
            asdict(failure)
            for failure in result.failures
        ],
        "summary": {
            "total_successful": len(result.documents),
            "total_failed": len(result.failures),
        },
    }