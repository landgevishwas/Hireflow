"""PDF parsing and structured Gemini extraction for HireFlow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypeVar

from google import genai
from google.genai import types
from pypdf import PdfReader
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from schemas import Candidate, JobDescription
from utils import get_logger


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# ============================================================================
# ERRORS
# ============================================================================

class ResumeParsingError(Exception):
    """Raised when a resume or document cannot be parsed."""


# ============================================================================
# PDF UTILITIES
# ============================================================================

def validate_pdf_path(pdf_path: str | Path) -> Path:
    """Validate that a PDF exists and has a PDF extension."""

    path = Path(pdf_path)

    if not path.exists():
        raise ResumeParsingError(
            f"PDF file does not exist: {path}"
        )

    if not path.is_file():
        raise ResumeParsingError(
            f"Path is not a file: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ResumeParsingError(
            f"Expected a PDF file: {path}"
        )

    return path


def clean_resume_text(text: str) -> str:
    """Clean extracted PDF text while preserving useful content."""

    if not text:
        return ""

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove null/control characters.
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        " ",
        text,
    )

    # Normalize excessive whitespace inside lines.
    lines: list[str] = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()

        if line:
            lines.append(line)

    # Preserve paragraph/section boundaries.
    cleaned = "\n".join(lines)

    # Avoid huge blank-line sequences.
    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


def extract_pdf_text(
    pdf_path: str | Path,
) -> str:
    """Extract text from all pages of a PDF."""

    path = validate_pdf_path(pdf_path)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ResumeParsingError(
            f"Could not open PDF: {path}"
        ) from exc

    pages: list[str] = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(
                "Could not extract page %s from %s: %s",
                page_number,
                path.name,
                exc,
            )
            continue

        if page_text.strip():
            pages.append(page_text)

    text = clean_resume_text(
        "\n\n".join(pages)
    )

    if not text:
        raise ResumeParsingError(
            f"No extractable text found in PDF: {path}"
        )

    return text


def count_words(text: str) -> int:
    """Return approximate word count."""

    if not text:
        return 0

    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text,
        )
    )


def is_probably_scanned_pdf(
    pdf_path: str | Path,
) -> bool:
    """Return True when a PDF appears to contain little/no text."""

    try:
        text = extract_pdf_text(pdf_path)
    except ResumeParsingError:
        return True

    return count_words(text) < 20


# ============================================================================
# STRUCTURED GEMINI EXTRACTOR
# ============================================================================

class StructuredExtractor:
    """Convert raw resume/JD text into validated Pydantic models."""

    def __init__(self) -> None:

        settings = get_settings(
            require_api_key=True
        )

        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

        self.model = settings.gemini_llm_model

        self.fallback_model = (
            settings.gemini_llm_fallback_model
        )

    # ------------------------------------------------------------------------
    # JSON SCHEMA
    # ------------------------------------------------------------------------

    @staticmethod
    def _clean_json_schema(value):
        """
        Remove JSON Schema fields that may be rejected by Gemini.

        Pydantic can generate additionalProperties=false
        for extra='forbid'. Gemini response_json_schema
        may reject that field, so remove it recursively.
        """

        if isinstance(value, dict):

            cleaned = {}

            for key, item in value.items():

                if key == "additionalProperties":
                    continue

                cleaned[key] = (
                    StructuredExtractor
                    ._clean_json_schema(item)
                )

            return cleaned

        if isinstance(value, list):

            return [
                StructuredExtractor
                ._clean_json_schema(item)
                for item in value
            ]

        return value

    # ------------------------------------------------------------------------
    # GEMINI CALL
    # ------------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=8,
        ),
        reraise=True,
    )
    def _generate_with_model(
        self,
        *,
        model: str,
        prompt: str,
        schema: type[T],
    ) -> T:
        """Generate structured JSON and validate it."""

        gemini_schema = (
            self._clean_json_schema(
                schema.model_json_schema()
            )
        )

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=gemini_schema,
            ),
        )

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        # Preferred path.
        try:
            return schema.model_validate_json(
                response.text
            )
        except Exception:
            pass

        # Fallback JSON parsing.
        try:
            data = json.loads(response.text)

            return schema.model_validate(
                data
            )

        except Exception as exc:

            raise ValueError(
                "Gemini returned invalid structured JSON."
            ) from exc

    def _generate(
        self,
        *,
        prompt: str,
        schema: type[T],
    ) -> T:
        """Use primary model and fallback model."""

        models = [self.model]

        if (
            self.fallback_model
            and self.fallback_model != self.model
        ):
            models.append(
                self.fallback_model
            )

        last_error: Exception | None = None

        for model in models:

            try:

                logger.info(
                    "Structured extraction using model=%s",
                    model,
                )

                return self._generate_with_model(
                    model=model,
                    prompt=prompt,
                    schema=schema,
                )

            except Exception as exc:

                last_error = exc

                logger.warning(
                    "Structured extraction failed "
                    "with model=%s: %s",
                    model,
                    exc,
                )

        raise RuntimeError(
            "All Gemini extraction models failed. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------------
    # CANDIDATE
    # ------------------------------------------------------------------------

    def parse_candidate(
        self,
        text: str,
        candidate_id: str,
    ) -> Candidate:
        """Extract a structured Candidate from resume text."""

        if not text.strip():
            raise ResumeParsingError(
                "Resume text is empty."
            )

        prompt = f"""
You are a recruitment resume extraction engine.

Extract structured candidate information from the
resume below.

STRICT RULES:

1. Use ONLY information explicitly present in the resume.

2. NEVER invent information.

3. If a field is unavailable, use an empty string,
   empty list, or null as appropriate.

4. Preserve names, company names, job titles,
   dates, education, certifications, and skills.

5. Extract both technical and professional skills.

6. Preserve important software and tools such as
   Excel, SAP, Oracle, QuickBooks, SQL, etc.

7. Estimate total_experience_years only when the
   resume contains enough reliable date information.

8. Do not infer experience from education.

9. Ignore corrupted template placeholders or
   unresolved expressions such as:
   {{generate_achievements(...)}}.

10. Do not create achievements that are not explicitly
    present.

11. Keep raw_text equal to the cleaned source text.

12. Do not use protected characteristics.

13. candidate_id must be exactly:
    {candidate_id}

RESUME:
{text}
"""

        candidate = self._generate(
            prompt=prompt,
            schema=Candidate,
        )

        return candidate.model_copy(
            update={
                "candidate_id": candidate_id,
                "raw_text": text,
            }
        )

    # ------------------------------------------------------------------------
    # JOB DESCRIPTION
    # ------------------------------------------------------------------------

    def parse_job_description(
        self,
        text: str,
        job_id: str,
    ) -> JobDescription:
        """Extract a structured JobDescription."""

        if not text.strip():
            raise ValueError(
                "Job description text is empty."
            )

        prompt = f"""
You are a recruitment job-description extraction engine.

Extract a structured JobDescription from the text below.

STRICT RULES:

1. Use ONLY information explicitly present in the
   job description.

2. NEVER invent requirements.

3. Separate required and preferred requirements.

4. Preserve technical tools and software names.

5. Extract responsibilities as separate items.

6. Extract minimum and maximum experience years
   only when explicitly stated or clearly expressed.

7. Preserve education requirements.

8. Preserve certifications and distinguish preferred
   certifications when the text makes that distinction.

9. Do not convert general responsibilities into
   mandatory skills unless the JD clearly requires them.

10. Do not use protected characteristics.

11. job_id must be exactly:
    {job_id}

JOB DESCRIPTION:
{text}
"""

        job = self._generate(
            prompt=prompt,
            schema=JobDescription,
        )

        return job.model_copy(
            update={
                "job_id": job_id,
                "raw_text": text,
            }
        )


# ============================================================================
# PUBLIC CANDIDATE FUNCTIONS
# ============================================================================

def parse_candidate_text(
    text: str,
    candidate_id: str,
) -> Candidate:
    """Parse raw resume text into Candidate."""

    extractor = StructuredExtractor()

    return extractor.parse_candidate(
        text=text,
        candidate_id=candidate_id,
    )


def parse_candidate_pdf(
    pdf_path: str | Path,
    candidate_id: str | None = None,
) -> Candidate:
    """Extract and structure a candidate PDF."""

    path = validate_pdf_path(pdf_path)

    if candidate_id is None:
        candidate_id = path.stem

    text = extract_pdf_text(path)

    return parse_candidate_text(
        text=text,
        candidate_id=candidate_id,
    )


# ============================================================================
# PUBLIC JOB DESCRIPTION FUNCTIONS
# ============================================================================

def parse_job_description(
    jd_text: str,
    job_id: str,
) -> JobDescription:
    """
    Parse raw job-description text into JobDescription.

    This is the public function used by Streamlit.
    """

    if not jd_text.strip():
        raise ValueError(
            "Job description text is empty."
        )

    extractor = StructuredExtractor()

    return extractor.parse_job_description(
        text=jd_text,
        job_id=job_id,
    )


def parse_job_description_pdf(
    pdf_path: str | Path,
    job_id: str | None = None,
) -> JobDescription:
    """Extract and structure a job-description PDF."""

    path = validate_pdf_path(pdf_path)

    if job_id is None:
        job_id = path.stem

    text = extract_pdf_text(path)

    return parse_job_description(
        jd_text=text,
        job_id=job_id,
    )


# ============================================================================
# QUICK MANUAL TEST
# ============================================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="HireFlow document parser"
    )

    parser.add_argument(
        "pdf",
        help="Path to PDF file",
    )

    parser.add_argument(
        "--type",
        choices=["candidate", "job"],
        default="candidate",
    )

    args = parser.parse_args()

    if args.type == "candidate":

        result = parse_candidate_pdf(
            args.pdf
        )

    else:

        result = parse_job_description_pdf(
            args.pdf
        )

    print(
        result.model_dump_json(
            indent=2
        )
    )