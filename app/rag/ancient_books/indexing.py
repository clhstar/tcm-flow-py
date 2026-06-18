"""
读取 chunks.jsonl
  ↓
校验 chunk 是否为空、chunk_id 是否重复
  ↓
按 chunk_id 排序，保证顺序稳定
  ↓
写 rows.jsonl，保存 chunk 原始信息
  ↓
用 jieba 对 chunk.text 分词，写 bm25_tokens.jsonl
  ↓
用 embedding 模型编码 chunk.text，得到 dense 向量
  ↓
归一化 dense 向量，保存 dense.npy
  ↓
生成 manifest.json，记录索引文件、模型版本、hash 等信息
"""

import json
from pathlib import Path

import jieba
import numpy as np

from .pipeline import sha256_file, write_json
from .schema import RetrievalChunk


def normalize_vectors(vectors, expected_count: int) -> np.ndarray:
    """
    对 embedding 模型输出的 Dense 向量做校验和归一化。

    参数：
        vectors:
            encoder.encode() 返回的向量。
            可能是 list，也可能是 numpy array。

        expected_count:
            期望的向量数量。
            一般应该等于 chunks 的数量。

    返回：
        归一化后的 numpy.ndarray，类型为 float32。

    作用：
        1. 确保向量形状正确。
        2. 确保没有 NaN / Inf。
        3. 确保没有零向量。
        4. 对向量做 L2 归一化，方便后续用点积计算余弦相似度。
    """

    # 把模型输出转成 numpy 数组，并指定为 float32
    # float32 比 float64 更省内存，也更适合向量检索
    array = np.asarray(vectors, dtype=np.float32)

    # 检查向量必须是二维矩阵
    # 正确形状应该类似：
    #   [chunk数量, 向量维度]
    #
    # 例如：
    #   1000 个 chunk，每个向量 1024 维
    #   array.shape = (1000, 1024)
    if array.ndim != 2 or array.shape[0] != expected_count:
        raise ValueError("Dense 向量形状不符合索引输入")

    # 检查向量中是否存在 NaN 或 Inf
    #
    # NaN: Not a Number
    # Inf: Infinity
    #
    # 如果存在这些值，后续相似度计算会出错。
    if not np.isfinite(array).all():
        raise ValueError("Dense 向量包含 NaN 或 Inf")

    # 计算每一行向量的 L2 范数
    #
    # axis=1 表示按行计算
    # keepdims=True 表示保持二维形状，方便后面广播相除
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    # 如果某个向量的范数是 0，说明它是零向量。
    # 零向量不能做归一化，因为会除以 0。
    if np.any(norms == 0):
        raise ValueError("Dense 向量不能为零向量")
    # 对每个向量做 L2 归一化
    #
    # 归一化后，每个向量长度都是 1。
    # 这样后续两个向量的点积，就等价于余弦相似度。
    return (array / norms).astype(np.float32, copy=False)


def _read_chunks(path: Path) -> list[RetrievalChunk]:
    """
    从 chunks.jsonl 文件中读取 RetrievalChunk 列表。

    chunks.jsonl 是 JSON Lines 格式：
        一行一个 JSON。

    每一行大概长这样：
        {"chunk_id": "...", "parent_id": "...", "text": "...", ...}

    返回：
        RetrievalChunk 对象列表。
    """

    # 逐行读取 JSONL 文件
    # line.strip() 用来跳过空行
    #
    # RetrievalChunk.model_validate_json(line)
    # 说明 RetrievalChunk 很可能是 Pydantic 模型。
    # 它会把 JSON 字符串解析并校验成 RetrievalChunk 对象。
    chunks = [
        RetrievalChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # 生产索引不允许空数据
    if not chunks:
        raise ValueError("生产索引不能使用空 chunks.jsonl")

    # 按 chunk_id 排序
    #
    # 这个非常重要：
    # rows.jsonl、bm25_tokens.jsonl、dense.npy 三个文件的顺序必须一致。
    #
    # 排序后，每次构建索引顺序稳定，方便实验复现。
    chunks.sort(key=lambda item: item.chunk_id)

    # 检查 chunk_id 是否重复
    #
    # chunk_id 是检索块的唯一标识。
    # 如果重复，后续检索结果就无法准确定位 chunk。
    if len({item.chunk_id for item in chunks}) != len(chunks):
        raise ValueError("生产索引存在重复 chunk_id")
    return chunks


def _write_json_rows(path: Path, rows: list[dict]) -> None:
    """
    将若干 dict 写成 JSONL 文件。

    JSONL 格式：
        每一行是一个独立 JSON 对象。

    参数：
        path:
            输出路径。

        rows:
            要写入的字典列表。
    """

    # 确保父目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    # 写 JSONL
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


def build_index(
    *,
    chunks_path: Path,
    corpus_manifest_sha256: str,
    output_dir: Path,
    encoder,
    model_record: dict,
) -> dict:
    """
    构建检索索引。

    输入：
        chunks_path:
            RetrievalChunk 的 JSONL 文件路径。
            通常来自前一步 parent-child 构建结果。

        corpus_manifest_sha256:
            语料库 manifest 的 hash。
            用来记录当前索引是基于哪个语料版本构建的。

        output_dir:
            索引输出目录。

        encoder:
            embedding 模型封装对象。
            需要有 encode() 方法。

        model_record:
            embedding 模型信息。
            例如：
            {
                "model": "BAAI/bge-m3",
                "revision": "5617a9f..."
            }

    输出：
        manifest 字典。
    """

    # 读取并校验 chunks
    chunks = _read_chunks(chunks_path)

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_path = output_dir / "rows.jsonl"
    tokens_path = output_dir / "bm25_tokens.jsonl"
    dense_path = output_dir / "dense.npy"

    # ------------------------------------------------------------
    # 1. 写 rows.jsonl
    # ------------------------------------------------------------
    #
    # rows.jsonl 保存 chunk 的完整元数据。
    #
    # 例如：
    #   chunk_id
    #   parent_id
    #   text
    #   source_type
    #   symptom_tags
    #   evidence_role
    #
    # 后续检索命中某一行 dense 向量时，
    # 可以根据相同顺序回到 rows.jsonl 找到 chunk 信息。
    _write_json_rows(
        rows_path,
        [item.model_dump(mode="json") for item in chunks],
    )

    # ------------------------------------------------------------
    # 2. 写 bm25_tokens.jsonl
    # ------------------------------------------------------------
    #
    # BM25 是关键词检索，需要先把文本分词。
    #
    # jieba.lcut(item.text, HMM=False)
    # 表示使用 jieba 对中文文本分词。
    #
    # HMM=False:
    #   不启用隐马尔可夫新词发现。
    #   好处是结果更稳定。
    #   缺点是对未登录词、新词、古籍词可能不够灵活。
    _write_json_rows(
        tokens_path,
        [
            {
                "chunk_id": item.chunk_id,
                "tokens": jieba.lcut(item.text, HMM=False),
            }
            for item in chunks
        ],
    )

    # ------------------------------------------------------------
    # 3. 生成 Dense 向量
    # ------------------------------------------------------------
    #
    # 对每个 child chunk 的 text 做 embedding。
    #
    # encoder.encode([...]) 返回形状类似：
    #   [chunk数量, 向量维度]
    #
    # 然后 normalize_vectors() 会做：
    #   - 形状校验
    #   - NaN / Inf 校验
    #   - 零向量校验
    #   - L2 归一化
    vectors = normalize_vectors(
        encoder.encode([item.text for item in chunks]),
        len(chunks),
    )
    
    # 保存 Dense 向量矩阵
    #
    # dense.npy 是 numpy 的二进制格式。
    # allow_pickle=False 是安全设置，避免 pickle 反序列化风险。
    np.save(dense_path, vectors, allow_pickle=False)

    paths = {
        "rows": rows_path,
        "bm25_tokens": tokens_path,
        "dense": dense_path,
    }
    
    # ------------------------------------------------------------
    # 4. 生成 manifest.json
    # ------------------------------------------------------------
    #
    # manifest 是索引说明文件。
    #
    # 它记录：
    #   - 当前索引版本
    #   - 索引状态
    #   - chunk 数量
    #   - 向量维度
    #   - 语料 hash
    #   - embedding 模型信息
    #   - 每个索引文件的大小和 hash
    #
    # 这样可以保证后续实验可追溯。
    manifest = {
        "version": "v1.0.0",
        "status": "ready",
        "row_count": len(chunks),
        "vector_dimension": int(vectors.shape[1]),
        "corpus_manifest_sha256": corpus_manifest_sha256.upper(),
        "embedding_model": {
            "model": model_record["model"],
            "revision": model_record["revision"],
        },
        "files": {
            name: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in paths.items()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
