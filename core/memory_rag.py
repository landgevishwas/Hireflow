"""Recruiter memory for HireFlow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import get_settings
from utils import get_logger


logger = get_logger(__name__)


class MemoryError(Exception):
    """Raised when recruiter memory operations fail."""


@dataclass
class RecruiterMemory:
    """Persisted recruiter preferences and decisions."""

    preferred_skills: list[str] = field(
        default_factory=list
    )

    excluded_skills: list[str] = field(
        default_factory=list
    )

    preferred_titles: list[str] = field(
        default_factory=list
    )

    preferred_industries: list[str] = field(
        default_factory=list
    )

    notes: list[str] = field(
        default_factory=list
    )

    candidate_decisions: dict[str, str] = field(
        default_factory=dict
    )

    updated_at: str = ""


class RecruiterMemoryStore:
    """Simple JSON-backed recruiter memory."""

    def __init__(
        self,
        memory_path: Path | None = None,
    ) -> None:

        settings = get_settings()

        self.memory_path = (
            memory_path
            or settings.memory_dir / "recruiter_memory.json"
        )

        self.memory = RecruiterMemory()

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp() -> str:
        """Return UTC timestamp."""

        return datetime.now(
            timezone.utc
        ).isoformat()

    def _touch(self) -> None:
        self.memory.updated_at = (
            self._timestamp()
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(
            value.lower().strip().split()
        )

    # ------------------------------------------------------------------
    # LOAD / SAVE
    # ------------------------------------------------------------------

    def load(self) -> RecruiterMemory:
        """Load memory from JSON."""

        if not self.memory_path.exists():
            logger.info(
                "Recruiter memory does not exist yet."
            )

            self.memory = RecruiterMemory()

            return self.memory

        try:
            with self.memory_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            self.memory = RecruiterMemory(
                **data
            )

            logger.info(
                "Loaded recruiter memory from %s",
                self.memory_path,
            )

            return self.memory

        except Exception as exc:

            raise MemoryError(
                f"Failed to load recruiter memory: {exc}"
            ) from exc

    def save(self) -> None:
        """Persist memory to JSON."""

        try:
            self.memory_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._touch()

            with self.memory_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    asdict(self.memory),
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            logger.info(
                "Saved recruiter memory to %s",
                self.memory_path,
            )

        except Exception as exc:

            raise MemoryError(
                f"Failed to save recruiter memory: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # SKILLS
    # ------------------------------------------------------------------

    def add_preferred_skill(
        self,
        skill: str,
    ) -> None:
        """Remember a preferred skill."""

        skill = skill.strip()

        if not skill:
            return

        existing = {
            self._normalize(item)
            for item in self.memory.preferred_skills
        }

        if self._normalize(skill) not in existing:
            self.memory.preferred_skills.append(
                skill
            )

            self.save()

    def add_excluded_skill(
        self,
        skill: str,
    ) -> None:
        """Remember an excluded skill."""

        skill = skill.strip()

        if not skill:
            return

        existing = {
            self._normalize(item)
            for item in self.memory.excluded_skills
        }

        if self._normalize(skill) not in existing:
            self.memory.excluded_skills.append(
                skill
            )

            self.save()

    # ------------------------------------------------------------------
    # TITLES / INDUSTRIES
    # ------------------------------------------------------------------

    def add_preferred_title(
        self,
        title: str,
    ) -> None:
        """Remember a preferred job title."""

        title = title.strip()

        if not title:
            return

        existing = {
            self._normalize(item)
            for item in self.memory.preferred_titles
        }

        if self._normalize(title) not in existing:
            self.memory.preferred_titles.append(
                title
            )

            self.save()

    def add_preferred_industry(
        self,
        industry: str,
    ) -> None:
        """Remember a preferred industry."""

        industry = industry.strip()

        if not industry:
            return

        existing = {
            self._normalize(item)
            for item in self.memory.preferred_industries
        }

        if self._normalize(industry) not in existing:
            self.memory.preferred_industries.append(
                industry
            )

            self.save()

    # ------------------------------------------------------------------
    # NOTES
    # ------------------------------------------------------------------

    def add_note(
        self,
        note: str,
    ) -> None:
        """Save a recruiter note."""

        note = note.strip()

        if not note:
            return

        self.memory.notes.append(
            note
        )

        self.save()

    # ------------------------------------------------------------------
    # CANDIDATE DECISIONS
    # ------------------------------------------------------------------

    def record_candidate_decision(
        self,
        candidate_id: str,
        decision: str,
    ) -> None:
        """Persist recruiter decision for a candidate."""

        candidate_id = candidate_id.strip()
        decision = decision.strip()

        if not candidate_id:
            raise MemoryError(
                "candidate_id cannot be empty."
            )

        if not decision:
            raise MemoryError(
                "decision cannot be empty."
            )

        self.memory.candidate_decisions[
            candidate_id
        ] = decision

        self.save()

    def get_candidate_decision(
        self,
        candidate_id: str,
    ) -> str | None:
        """Return previous recruiter decision."""

        return self.memory.candidate_decisions.get(
            candidate_id
        )

    # ------------------------------------------------------------------
    # SEARCH SIGNALS
    # ------------------------------------------------------------------

    def get_ranking_signals(self) -> dict[str, Any]:
        """
        Return memory signals for downstream ranking.

        Memory is advisory only. It must not override explicit
        job requirements or hard filters.
        """

        return {
            "preferred_skills": list(
                self.memory.preferred_skills
            ),
            "excluded_skills": list(
                self.memory.excluded_skills
            ),
            "preferred_titles": list(
                self.memory.preferred_titles
            ),
            "preferred_industries": list(
                self.memory.preferred_industries
            ),
        }

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all recruiter memory."""

        self.memory = RecruiterMemory()

        self.save()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_recruiter_memory() -> RecruiterMemoryStore:
    """Create and load recruiter memory store."""

    store = RecruiterMemoryStore()

    store.load()

    return store