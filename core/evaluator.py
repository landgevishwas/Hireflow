"""Gemini evaluation and ranking metrics for HireFlow."""

from __future__ import annotations

import json
import math
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings
from schemas import Candidate, JobDescription
from utils import get_logger


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# ============================================================================
# GEMINI EVALUATION SCHEMA
# ============================================================================

class Evidence(BaseModel):
    """Evidence supporting an evaluation finding."""

    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(default="")
    evidence: str = Field(default="")
    source: str = Field(default="resume")


class CandidateEvaluation(BaseModel):
    """Structured and explainable Gemini evaluation."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., min_length=1)

    overall_score: float = Field(
        ge=0,
        le=100,
    )

    recommendation: str = Field(
        default="Review"
    )

    strengths: list[str] = Field(
        default_factory=list
    )

    concerns: list[str] = Field(
        default_factory=list
    )

    matched_requirements: list[str] = Field(
        default_factory=list
    )

    missing_requirements: list[str] = Field(
        default_factory=list
    )

    evidence: list[Evidence] = Field(
        default_factory=list
    )

    reasoning: str = Field(
        default=""
    )


# ============================================================================
# GEMINI EVALUATOR
# ============================================================================

class CandidateEvaluator:
    """Evaluate candidates against job descriptions using Gemini."""

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

    @staticmethod
    def _clean_json_schema(value):
        """Remove JSON Schema fields unsupported by Gemini."""

        if isinstance(value, dict):
            cleaned = {}

            for key, item in value.items():
                if key == "additionalProperties":
                    continue

                cleaned[key] = (
                    CandidateEvaluator
                    ._clean_json_schema(item)
                )

            return cleaned

        if isinstance(value, list):
            return [
                CandidateEvaluator
                ._clean_json_schema(item)
                for item in value
            ]

        return value

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
    ) -> CandidateEvaluation:

        schema = CandidateEvaluation

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
                "Gemini evaluator returned empty response."
            )

        try:
            return schema.model_validate_json(
                response.text
            )
        except Exception:
            try:
                data = json.loads(response.text)
                return schema.model_validate(data)
            except Exception as exc:
                raise ValueError(
                    f"Invalid evaluator output: {exc}"
                ) from exc

    def _generate(
        self,
        prompt: str,
    ) -> CandidateEvaluation:

        models = [self.model]

        if (
            self.fallback_model
            and self.fallback_model != self.model
        ):
            models.append(self.fallback_model)

        last_error: Exception | None = None

        for model in models:
            try:
                logger.info(
                    "Candidate evaluation using model=%s",
                    model,
                )

                return self._generate_with_model(
                    model=model,
                    prompt=prompt,
                )

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Evaluation failed with model=%s.",
                    model,
                )

        raise RuntimeError(
            "All Gemini evaluator models failed. "
            f"Last error: {last_error}"
        )

    def evaluate(
        self,
        candidate: Candidate,
        job: JobDescription,
    ) -> CandidateEvaluation:
        """Evaluate one candidate against one job."""

        prompt = f"""
You are an explainable recruitment evaluation engine.

Evaluate the candidate against the job description.

RULES:

1. Use ONLY information explicitly present in the
   candidate and job description.

2. NEVER invent skills, experience, achievements,
   education, certifications, or tools.

3. Missing information is not proof that the candidate
   lacks the skill. Use "not demonstrated" when needed.

4. Required requirements are more important than
   preferred requirements.

5. Do not penalize a candidate merely for exceeding
   the job's stated maximum experience.

6. Every important positive or negative conclusion
   should have supporting evidence.

7. Do not use protected characteristics or unrelated
   personal information.

8. Do not consider age, gender, race, religion,
   nationality, marital status, photographs, or similar
   protected characteristics.

9. Recommendation must be exactly one of:
   Strong Match
   Match
   Review
   Weak Match

10. overall_score must be between 0 and 100.

11. Evidence should be specific and traceable to the
    candidate resume.

12. Keep reasoning concise.

CANDIDATE:
{candidate.model_dump_json(indent=2)}

JOB DESCRIPTION:
{job.model_dump_json(indent=2)}
"""

        result = self._generate(prompt)

        return result.model_copy(
            update={
                "candidate_id": candidate.candidate_id
            }
        )

    def evaluate_candidate(
        self,
        candidate: Candidate,
        job: JobDescription,
    ) -> CandidateEvaluation:
        """Compatibility alias used by the Streamlit UI."""
        return self.evaluate(
            candidate=candidate,
            job=job,
        )


# ============================================================================
# RANKING METRICS
# ============================================================================

def precision_at_k(
    ranked_candidate_ids: list[str],
    relevant_candidate_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision@K."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    ranked = ranked_candidate_ids[:k]

    if not ranked:
        return 0.0

    relevant = sum(
        candidate_id in relevant_candidate_ids
        for candidate_id in ranked
    )

    return relevant / len(ranked)


def recall_at_k(
    ranked_candidate_ids: list[str],
    relevant_candidate_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall@K."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_candidate_ids:
        return 0.0

    ranked = ranked_candidate_ids[:k]

    relevant = sum(
        candidate_id in relevant_candidate_ids
        for candidate_id in ranked
    )

    return relevant / len(relevant_candidate_ids)


def mean_reciprocal_rank(
    ranked_candidate_ids: list[str],
    relevant_candidate_ids: set[str],
) -> float:
    """Calculate reciprocal rank of first relevant result."""

    if not relevant_candidate_ids:
        return 0.0

    for rank, candidate_id in enumerate(
        ranked_candidate_ids,
        start=1,
    ):
        if candidate_id in relevant_candidate_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    ranked_candidate_ids: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """Calculate NDCG@K using supplied relevance scores."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    ranked = ranked_candidate_ids[:k]

    if not ranked:
        return 0.0

    def dcg(scores: list[float]) -> float:
        total = 0.0

        for index, score in enumerate(
            scores,
            start=1,
        ):
            total += (
                (2**score - 1)
                / math.log2(index + 1)
            )

        return total

    actual_scores = [
        max(
            0.0,
            float(
                relevance_scores.get(
                    candidate_id,
                    0.0,
                )
            ),
        )
        for candidate_id in ranked
    ]

    ideal_scores = sorted(
        (
            max(0.0, float(score))
            for score in relevance_scores.values()
        ),
        reverse=True,
    )[:k]

    ideal_dcg = dcg(ideal_scores)

    if ideal_dcg == 0:
        return 0.0

    return dcg(actual_scores) / ideal_dcg


# ============================================================================
# EXPLAINABILITY METRICS
# ============================================================================

def evidence_coverage(
    evaluation: CandidateEvaluation,
) -> float:
    """
    Measure how many matched/missing requirements have
    supporting evidence.
    """

    total_requirements = (
        len(evaluation.matched_requirements)
        + len(evaluation.missing_requirements)
    )

    if total_requirements == 0:
        return 0.0

    evidence_requirements = {
        item.requirement.strip().lower()
        for item in evaluation.evidence
        if item.requirement.strip()
    }

    covered = 0

    for requirement in (
        evaluation.matched_requirements
        + evaluation.missing_requirements
    ):
        if requirement.strip().lower() in evidence_requirements:
            covered += 1

    return covered / total_requirements


def requirement_coverage(
    evaluation: CandidateEvaluation,
) -> float:
    """Measure matched requirement coverage."""

    total = (
        len(evaluation.matched_requirements)
        + len(evaluation.missing_requirements)
    )

    if total == 0:
        return 0.0

    return (
        len(evaluation.matched_requirements)
        / total
    )


# ============================================================================
# EVALUATION METRICS REPORT
# ============================================================================

def calculate_ranking_metrics(
    ranked_candidate_ids: list[str],
    relevant_candidate_ids: set[str],
    relevance_scores: dict[str, float],
    k: int = 10,
) -> dict[str, float]:
    """Calculate retrieval/ranking metrics."""

    return {
        "precision_at_k": precision_at_k(
            ranked_candidate_ids,
            relevant_candidate_ids,
            k,
        ),
        "recall_at_k": recall_at_k(
            ranked_candidate_ids,
            relevant_candidate_ids,
            k,
        ),
        "mrr": mean_reciprocal_rank(
            ranked_candidate_ids,
            relevant_candidate_ids,
        ),
        "ndcg_at_k": ndcg_at_k(
            ranked_candidate_ids,
            relevance_scores,
            k,
        ),
    }


# ============================================================================
# PUBLIC EVALUATION FUNCTIONS
# ============================================================================

def evaluate_candidate(
    candidate: Candidate,
    job: JobDescription,
) -> CandidateEvaluation:
    """Evaluate one candidate."""

    evaluator = CandidateEvaluator()

    return evaluator.evaluate(
        candidate=candidate,
        job=job,
    )


def evaluate_candidates(
    candidates: list[Candidate],
    job: JobDescription,
) -> list[CandidateEvaluation]:
    """Evaluate multiple candidates."""

    evaluator = CandidateEvaluator()

    results: list[CandidateEvaluation] = []

    for candidate in candidates:
        try:
            result = evaluator.evaluate(
                candidate=candidate,
                job=job,
            )

            results.append(result)

        except Exception as exc:
            logger.error(
                "Evaluation failed for candidate %s: %s",
                candidate.candidate_id,
                exc,
            )

    results.sort(
        key=lambda item: item.overall_score,
        reverse=True,
    )

    return results