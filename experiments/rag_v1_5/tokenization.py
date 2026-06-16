import jieba


def tokenize_text(text: str) -> list[str]:
    normalized = " ".join(text.split())
    return [
        token.strip()
        for token in jieba.lcut(normalized, HMM=False)
        if token.strip()
    ]
