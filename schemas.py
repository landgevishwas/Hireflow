"""Pydantic schemas for HireFlow candidates and job descriptions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class WorkExperience(BaseModel):
    """A single professional experience entry."""

    model_config = ConfigDict(extra="forbid")

    company: str = Field(default="", description="Company or organization name.")
    job_title: str = Field(default="", description="Job title or role.")
    start_date: str = Field(
        default="",
        description="Start date as written in the resume.",
    )
    end_date: str = Field(
        default="",
        description="End date as written in the resume. Use Present when applicable.",
    )
    description: str = Field(
        default="",
        description="Responsibilities, achievements, and other role details.",
    )


class Education(BaseModel):
    """A single education entry."""

    model_config = ConfigDict(extra="forbid")

    degree: str = Field(default="", description="Degree or qualification.")
    field_of_study: str = Field(
        default="",
        description="Major, specialization, or field of study.",
    )
    institution: str = Field(
        default="",
        description="School, college, or university.",
    )
    graduation_year: str = Field(
        default="",
        description="Graduation year as written in the resume.",
    )
    grade: str = Field(
        default="",
        description="GPA, percentage, grade, or other academic result.",
    )


class Certification(BaseModel):
    """A professional certification."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="", description="Certification name.")
    issuer: str = Field(default="", description="Certification issuing organization.")
    year: str = Field(default="", description="Certification year if available.")


class Candidate(BaseModel):
    """Structured representation of a job candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(
        ...,
        min_length=1,
        description="Unique internal candidate identifier.",
    )
    name: str = Field(
        default="",
        description="Candidate's full name.",
    )
    email: str = Field(default="", description="Candidate email.")
    phone: str = Field(default="", description="Candidate phone number.")
    location: str = Field(default="", description="Candidate location.")
    linkedin: str = Field(default="", description="LinkedIn profile if available.")

    professional_summary: str = Field(
        default="",
        description="Professional summary or objective.",
    )

    skills: list[str] = Field(
        default_factory=list,
        description="Technical and professional skills.",
    )

    experience: list[WorkExperience] = Field(
        default_factory=list,
        description="Professional work experience.",
    )

    education: list[Education] = Field(
        default_factory=list,
        description="Educational background.",
    )

    certifications: list[Certification] = Field(
        default_factory=list,
        description="Professional certifications.",
    )

    total_experience_years: float | None = Field(
        default=None,
        ge=0,
        description="Estimated total professional experience in years.",
    )

    raw_text: str = Field(
        default="",
        description="Cleaned text extracted from the original resume.",
    )


class SkillRequirement(BaseModel):
    """A job skill with an explicit requirement level."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="Skill or competency name.",
    )

    requirement_type: Literal["required", "preferred"] = Field(
        ...,
        description="Whether the skill is required or preferred.",
    )


class JobDescription(BaseModel):
    """Structured representation of a job description."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(
        ...,
        min_length=1,
        description="Unique internal job identifier.",
    )

    title: str = Field(
        default="",
        description="Job title.",
    )

    company: str = Field(
        default="",
        description="Hiring company.",
    )

    location: str = Field(
        default="",
        description="Job location.",
    )

    employment_type: str = Field(
        default="",
        description="Employment type such as full-time or part-time.",
    )

    work_mode: str = Field(
        default="",
        description="Work arrangement such as remote, hybrid, or onsite.",
    )

    salary_range: str = Field(
        default="",
        description="Salary range if provided.",
    )

    responsibilities: list[str] = Field(
        default_factory=list,
        description="Primary job responsibilities.",
    )

    skill_requirements: list[SkillRequirement] = Field(
        default_factory=list,
        description="Required and preferred skills.",
    )

    required_experience_years_min: float | None = Field(
        default=None,
        ge=0,
        description="Minimum required years of experience.",
    )

    required_experience_years_max: float | None = Field(
        default=None,
        ge=0,
        description="Maximum required years of experience.",
    )

    education_requirements: list[str] = Field(
        default_factory=list,
        description="Required or preferred educational qualifications.",
    )

    certifications: list[str] = Field(
        default_factory=list,
        description="Required or preferred certifications.",
    )

    raw_text: str = Field(
        default="",
        description="Cleaned text of the original job description.",
    )