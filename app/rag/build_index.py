from app.rag.ancient_books.cli import main


def build_index() -> None:
    """Build the verified production index from an existing local corpus."""

    main(["build-index"])


if __name__ == "__main__":
    build_index()
