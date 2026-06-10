from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAW_DATA_DIR = Path("data/raw")

# 胃胀           -> topic = 胃胀
## 胃胀问诊要点  -> section = 胃胀问诊要点

def parse_markdown_sections(text: str, source: str, filename: str) -> list[Document]:
    """
    将 Markdown 按 # / ## 标题切成 section 文档。
    每个 section 会保留 topic / section metadata。
    """
    documents: list[Document] = []

    current_topic = "通用"
    current_section = "正文"
    buffer: list[str] = []

    def flush():
        nonlocal buffer

        content = "\n".join(buffer).strip()
        if not content:
            buffer = []
            return

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": source,
                    "filename": filename,
                    "topic": current_topic,
                    "section": current_section,
                },
            )
        )

        buffer = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("# "):
            flush()
            current_topic = stripped.replace("# ", "", 1).strip()
            current_section = "概述"
            buffer.append(line)

        elif stripped.startswith("## "):
            flush()
            current_section = stripped.replace("## ", "", 1).strip()
            buffer.append(line)

        else:
            buffer.append(line)

    flush()

    return documents


def load_markdown_documents() -> list[Document]:
    documents: list[Document] = []

    for file_path in RAW_DATA_DIR.glob("*.md"):
        text = file_path.read_text(encoding="utf-8")
        section_docs = parse_markdown_sections(
            text=text,
            source=str(file_path),
            filename=file_path.name,
        )
        documents.extend(section_docs)

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    对 section 文档继续切 chunk。
    metadata 会自动继承。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n### ", "\n## ", "\n# ", "\n\n", "\n", "。", "，", " "],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks