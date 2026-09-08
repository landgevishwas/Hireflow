"""HireFlow CLI entry point for Phase 0."""

from config import ensure_directories, settings
from utils import configure_logging


def main() -> None:
    """Run the Phase 0 configuration check."""

    configure_logging(settings.log_level)

    settings.validate(
        require_api_key=False,
    )

    ensure_directories()

    print("HireFlow Phase 0 setup is valid.")
    print(f"Project root: {settings.resume_dir.parent}")
    print(f"Resume directory: {settings.resume_dir}")
    print(
        "Hybrid index directory: "
        f"{settings.hybrid_index_dir}"
    )
    print(
        "Memory directory: "
        f"{settings.memory_dir}"
    )


if __name__ == "__main__":
    main()