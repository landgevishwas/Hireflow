"""Explicit candidate filtering for HireFlow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from schemas import Candidate, JobDescription
from utils import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class FilterResult:
    """Result of applying eligibility filters to a candidate."""

    candidate_id: str
    eligible: bool
    passed_filters: tuple[str, ...] = field(default_factory=tuple)
    failed_filters: tuple[str, ...] = field(default_factory=tuple)
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _normalize(value: str) -> str:
    """Normalize text for comparisons."""
    return " ".join(value.lower().strip().split())


def _skill_names(candidate: Candidate) -> set[str]:
    """Return normalized candidate skills."""
    return {
        _normalize(skill)
        for skill in candidate.skills
        if skill.strip()
    }


def _has_skill(
    candidate_skills: set[str],
    required_skill: str,
) -> bool:
    """Check exact or phrase-level skill presence."""

    required = _normalize(required_skill)

    if not required:
        return True

    return any(
        required == skill
        or required in skill
        or skill in required
        for skill in candidate_skills
    )


def _check_experience(
    candidate: Candidate,
    job: JobDescription,
) -> tuple[bool, str]:
    """Check minimum experience requirement."""

    minimum = job.required_experience_years_min

    if minimum is None:
        return True, "No minimum experience requirement."

    if candidate.total_experience_years is None:
        return False, (
            f"Required minimum experience: {minimum:g} years; "
            "candidate experience could not be determined."
        )

    if candidate.total_experience_years < minimum:
        return False, (
            f"Required minimum experience: {minimum:g} years; "
            f"candidate has {candidate.total_experience_years:g} years."
        )

    return True, (
        f"Experience requirement met: "
        f"{candidate.total_experience_years:g} years."
    )


def _check_required_skills(
    candidate: Candidate,
    job: JobDescription,
) -> tuple[bool, list[str]]:
    """Check explicitly required skills."""

    candidate_skills = _skill_names(candidate)

    required_skills = [
        requirement.name
        for requirement in job.skill_requirements
        if requirement.requirement_type == "required"
    ]

    missing: list[str] = []

    for skill in required_skills:
        if not _has_skill(candidate_skills, skill):
            missing.append(skill)

    return len(missing) == 0, missing


def _check_education(
    candidate: Candidate,
    job: JobDescription,
) -> tuple[bool, str]:
    """Check whether candidate has relevant education."""

    if not job.education_requirements:
        return True, "No education requirement specified."

    if not candidate.education:
        return False, "No education information found."

    candidate_education = " ".join(
        (
            f"{education.degree} "
            f"{education.field_of_study}"
        )
        for education in candidate.education
    ).lower()

    for requirement in job.education_requirements:
        normalized_requirement = _normalize(requirement)

        # Conservative matching: only claim a match when
        # the requirement text appears in candidate education.
        if normalized_requirement in candidate_education:
            return True, (
                f"Education requirement matched: {requirement}."
            )

    return False, (
        "Candidate education does not clearly match "
        "the specified education requirements."
    )


def apply_filters(
    candidate: Candidate,
    job: JobDescription,
) -> FilterResult:
    """
    Apply explicit eligibility filters.

    Filtering is intentionally independent of semantic or
    keyword retrieval scores.
    """

    passed: list[str] = []
    failed: list[str] = []
    reasons: list[str] = []

    # --------------------------------------------------------------
    # Experience
    # --------------------------------------------------------------

    experience_passed, experience_reason = _check_experience(
        candidate,
        job,
    )

    if experience_passed:
        passed.append("experience")
    else:
        failed.append("experience")

    reasons.append(experience_reason)

    # --------------------------------------------------------------
    # Required skills
    # --------------------------------------------------------------

    skills_passed, missing_skills = _check_required_skills(
        candidate,
        job,
    )

    if skills_passed:
        passed.append("required_skills")
        reasons.append("All explicitly required skills are present.")
    else:
        failed.append("required_skills")
        reasons.append(
            "Missing required skills: "
            + ", ".join(missing_skills)
        )

    # --------------------------------------------------------------
    # Education
    # --------------------------------------------------------------

    education_passed, education_reason = _check_education(
        candidate,
        job,
    )

    if education_passed:
        passed.append("education")
    else:
        failed.append("education")

    reasons.append(education_reason)

    eligible = len(failed) == 0

    return FilterResult(
        candidate_id=candidate.candidate_id,
        eligible=eligible,
        passed_filters=tuple(passed),
        failed_filters=tuple(failed),
        reasons=tuple(reasons),
    )


def filter_candidates(
    candidates: list[Candidate],
    job: JobDescription,
) -> list[FilterResult]:
    """Apply explicit filters to multiple candidates."""

    results: list[FilterResult] = []

    for candidate in candidates:
        result = apply_filters(
            candidate,
            job,
        )

        results.append(result)

        logger.info(
            "Candidate %s: eligible=%s",
            candidate.candidate_id,
            result.eligible,
        )

    return results