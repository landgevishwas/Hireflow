"""Explainable candidate re-ranking for HireFlow."""

from __future__ import annotations

from dataclasses import dataclass, field

from schemas import Candidate, JobDescription
from utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReRankResult:
    """Explainable ranking result for one candidate."""

    candidate_id: str
    final_score: float

    hybrid_score: float
    experience_score: float
    required_skills_score: float
    preferred_skills_score: float
    education_score: float
    technical_stack_score: float

    matched_required_skills: tuple[str, ...] = field(
        default_factory=tuple
    )
    missing_required_skills: tuple[str, ...] = field(
        default_factory=tuple
    )
    matched_preferred_skills: tuple[str, ...] = field(
        default_factory=tuple
    )

    strengths: tuple[str, ...] = field(
        default_factory=tuple
    )
    concerns: tuple[str, ...] = field(
        default_factory=tuple
    )


def _normalize(value: str) -> str:
    """Normalize text for comparison."""

    return " ".join(
        value.lower().strip().split()
    )


def _skill_match(
    candidate_skill: str,
    requirement: str,
) -> bool:
    """Perform conservative skill matching."""

    candidate = _normalize(candidate_skill)
    required = _normalize(requirement)

    if not candidate or not required:
        return False

    return (
        candidate == required
        or required in candidate
        or candidate in required
    )


def _candidate_skills(
    candidate: Candidate,
) -> list[str]:
    """Return cleaned candidate skills."""

    return [
        skill.strip()
        for skill in candidate.skills
        if skill.strip()
    ]


def _required_skills(
    job: JobDescription,
) -> list[str]:
    """Return required job skills."""

    return [
        requirement.name
        for requirement in job.skill_requirements
        if requirement.requirement_type == "required"
    ]


def _preferred_skills(
    job: JobDescription,
) -> list[str]:
    """Return preferred job skills."""

    return [
        requirement.name
        for requirement in job.skill_requirements
        if requirement.requirement_type == "preferred"
    ]


def _match_skills(
    candidate: Candidate,
    requirements: list[str],
) -> tuple[list[str], list[str]]:
    """Return matched and missing requirements."""

    candidate_skills = _candidate_skills(candidate)

    matched: list[str] = []
    missing: list[str] = []

    for requirement in requirements:

        if any(
            _skill_match(
                skill,
                requirement,
            )
            for skill in candidate_skills
        ):
            matched.append(requirement)
        else:
            missing.append(requirement)

    return matched, missing


def _experience_score(
    candidate: Candidate,
    job: JobDescription,
) -> tuple[float, str | None]:
    """Calculate experience fit score."""

    minimum = job.required_experience_years_min
    maximum = job.required_experience_years_max
    candidate_years = candidate.total_experience_years

    if candidate_years is None:
        return 0.0, "Candidate experience could not be determined."

    if minimum is None:
        return 1.0, None

    if candidate_years < minimum:
        ratio = candidate_years / minimum if minimum else 0.0
        return max(0.0, min(1.0, ratio)), (
            f"Candidate has {candidate_years:g} years "
            f"against required minimum of {minimum:g} years."
        )

    # Experience above the stated maximum is not treated as a failure.
    if maximum is not None and candidate_years > maximum:
        return 1.0, (
            f"Candidate has {candidate_years:g} years; "
            f"job lists {minimum:g}-{maximum:g} years."
        )

    return 1.0, None


def _education_score(
    candidate: Candidate,
    job: JobDescription,
) -> tuple[float, str | None]:
    """Calculate education relevance."""

    if not job.education_requirements:
        return 1.0, None

    if not candidate.education:
        return 0.0, "No education information found."

    candidate_text = " ".join(
        f"{education.degree} "
        f"{education.field_of_study}"
        for education in candidate.education
    ).lower()

    for requirement in job.education_requirements:

        requirement_text = _normalize(
            requirement
        )

        requirement_words = [
            word
            for word in requirement_text.split()
            if len(word) > 2
        ]

        if requirement_text in candidate_text:
            return 1.0, None

        matched_words = sum(
            word in candidate_text
            for word in requirement_words
        )

        if requirement_words and (
            matched_words
            / len(requirement_words)
            >= 0.5
        ):
            return 1.0, None

    return 0.0, (
        "Candidate education does not clearly "
        "match the stated requirement."
    )


def _technical_stack_score(
    candidate: Candidate,
    job: JobDescription,
) -> tuple[float, list[str]]:
    """
    Estimate technical-stack coverage from the candidate's
    skills against job requirements.
    """

    all_requirements = (
        _required_skills(job)
        + _preferred_skills(job)
    )

    if not all_requirements:
        return 1.0, []

    matched, _ = _match_skills(
        candidate,
        all_requirements,
    )

    score = len(matched) / len(
        all_requirements
    )

    return score, matched


def rerank_candidate(
    candidate: Candidate,
    job: JobDescription,
    hybrid_score: float,
) -> ReRankResult:
    """Calculate an explainable final ranking score."""

    required = _required_skills(job)
    preferred = _preferred_skills(job)

    matched_required, missing_required = (
        _match_skills(
            candidate,
            required,
        )
    )

    matched_preferred, _ = _match_skills(
        candidate,
        preferred,
    )

    required_score = (
        len(matched_required)
        / len(required)
        if required
        else 1.0
    )

    preferred_score = (
        len(matched_preferred)
        / len(preferred)
        if preferred
        else 0.0
    )

    experience_score, experience_concern = (
        _experience_score(
            candidate,
            job,
        )
    )

    education_score, education_concern = (
        _education_score(
            candidate,
            job,
        )
    )

    technical_score, technical_matches = (
        _technical_stack_score(
            candidate,
            job,
        )
    )

    # ---------------------------------------------------------------
    # Final explainable score
    #
    # Hybrid retrieval remains important, but explicit job fit
    # receives more influence during re-ranking.
    # ---------------------------------------------------------------

    final_score = (
        0.30 * hybrid_score
        + 0.25 * required_score
        + 0.15 * preferred_score
        + 0.15 * experience_score
        + 0.10 * education_score
        + 0.05 * technical_score
    )

    final_score = max(
        0.0,
        min(1.0, final_score),
    )

    strengths: list[str] = []
    concerns: list[str] = []

    if matched_required:
        strengths.append(
            "Matched required skills: "
            + ", ".join(matched_required)
        )

    if matched_preferred:
        strengths.append(
            "Matched preferred skills: "
            + ", ".join(matched_preferred)
        )

    if technical_matches:
        strengths.append(
            "Technical stack overlap detected."
        )

    if experience_score >= 1.0:
        strengths.append(
            "Experience requirement is satisfied."
        )

    if missing_required:
        concerns.append(
            "Missing required skills: "
            + ", ".join(missing_required)
        )

    if experience_concern:
        concerns.append(
            experience_concern
        )

    if education_concern:
        concerns.append(
            education_concern
        )

    logger.info(
        "Re-ranked candidate %s with score %.4f",
        candidate.candidate_id,
        final_score,
    )

    return ReRankResult(
        candidate_id=candidate.candidate_id,
        final_score=final_score,
        hybrid_score=hybrid_score,
        experience_score=experience_score,
        required_skills_score=required_score,
        preferred_skills_score=preferred_score,
        education_score=education_score,
        technical_stack_score=technical_score,
        matched_required_skills=tuple(
            matched_required
        ),
        missing_required_skills=tuple(
            missing_required
        ),
        matched_preferred_skills=tuple(
            matched_preferred
        ),
        strengths=tuple(strengths),
        concerns=tuple(concerns),
    )


def rerank_candidates(
    candidates: list[Candidate],
    job: JobDescription,
    hybrid_scores: dict[str, float] | None = None,
    retrieval_scores: dict[str, float | dict] | None = None,
) -> list[ReRankResult]:
    """Re-rank multiple candidates using hybrid retrieval scores.

    ``hybrid_scores`` is the preferred interface. ``retrieval_scores`` is
    accepted for compatibility with the Streamlit application, where each
    value may be a result dictionary containing ``hybrid_score``.
    """

    if hybrid_scores is None:
        hybrid_scores = {}

    if retrieval_scores is not None:
        for candidate_id, value in retrieval_scores.items():
            if isinstance(value, dict):
                hybrid_scores[candidate_id] = float(
                    value.get("hybrid_score", 0.0)
                )
            else:
                hybrid_scores[candidate_id] = float(value)

    results: list[ReRankResult] = []

    for candidate in candidates:

        hybrid_score = float(
            hybrid_scores.get(
                candidate.candidate_id,
                0.0,
            )
        )

        result = rerank_candidate(
            candidate=candidate,
            job=job,
            hybrid_score=hybrid_score,
        )

        results.append(result)

    results.sort(
        key=lambda item: item.final_score,
        reverse=True,
    )

    return results