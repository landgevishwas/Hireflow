"""Shared utilities for HireFlow."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable


def configure_logging(level: str = "INFO") -> None:
    """Configure application-wide logging."""
    logging.basicConfig(
        level=getattr(
            logging,
            level.upper(),
            logging.INFO,
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)


def clean_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """Normalize basic text noise."""
    text = text.replace("\x00", " ")

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    lines = [
        clean_whitespace(line)
        for line in text.split("\n")
    ]

    return "\n".join(
        line
        for line in lines
        if line
    )


def discover_files(
    directory: Path,
    extensions: Iterable[str] | None = None,
) -> list[Path]:
    """Recursively discover files.

    Args:
        directory: Directory to search.
        extensions: Optional allowed file extensions.

    Returns:
        Sorted list of matching file paths.
    """

    if not directory.exists():
        return []

    allowed = None

    if extensions:
        allowed = {
            ext.lower()
            if ext.startswith(".")
            else f".{ext.lower()}"
            for ext in extensions
        }

    files = [
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and (
                allowed is None
                or path.suffix.lower() in allowed
            )
        )
    ]

    return sorted(files)


def safe_filename(name: str) -> str:
    """Convert a filename into a filesystem-safe name."""
    cleaned = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        name,
    )

    return cleaned.strip("._") or "file"