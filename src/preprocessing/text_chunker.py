"""
TextChunker — 文本文档分块。

使用 LangChain 的 RecursiveCharacterTextSplitter 进行语义边界感知的分块。
"""

import re
import uuid
from typing import Dict, List, Optional


class TextChunker:
    """文本文档分块器。

    使用递归字符分割, 优先在段落/句子边界处分块,
    保持语义完整性的同时控制块大小。

    Usage:
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk("长文本内容...", metadata={"source": "doc.txt"})
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",
            "\n",
            "。",
            ". ",
            "? ",
            "! ",
            "；",
            "; ",
            "，",
            ", ",
            " ",
            "",
        ]

    def chunk(
        self,
        text: str,
        metadata: Optional[dict] = None,
        source_file: Optional[str] = None,
    ) -> List[dict]:
        """将文本分割为块。

        Args:
            text: 输入文本。
            metadata: 附加元数据 (合并到每个 chunk)。
            source_file: 来源文件路径。

        Returns:
            chunk 字典列表, 每个包含:
                - chunk_id: 唯一 ID
                - text: 块文本
                - metadata: 元数据 (含 source_file, chunk_index 等)
                - modality: "text"
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError(
                "需要 langchain-text-splitters。请安装: pip install langchain-text-splitters"
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        docs = splitter.create_documents(
            texts=[text],
            metadatas=[metadata or {}],
        )

        chunks = []
        for i, doc in enumerate(docs):
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": doc.page_content.strip(),
                "metadata": {
                    "source_file": source_file or metadata.get("source", "unknown"),
                    "chunk_index": i,
                    "chunk_count": len(docs),
                    "modality": "text",
                    "content_type": "text_chunk",
                    **(metadata or {}),
                },
                "modality": "text",
            })

        return chunks

    def chunk_with_token_limit(
        self,
        text: str,
        max_tokens: int = 512,
        metadata: Optional[dict] = None,
        source_file: Optional[str] = None,
    ) -> List[dict]:
        """使用 token 计数的分块 (需要 tiktoken)。

        Args:
            text: 输入文本。
            max_tokens: 每块最大 token 数。
            metadata: 附加元数据。
            source_file: 来源文件路径。

        Returns:
            chunk 字典列表。
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError("需要 tiktoken。请安装: pip install tiktoken")

        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")

        sentences = self._split_sentences(text)
        chunks = []
        current_chunk: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = len(enc.encode(sentence))

            if current_tokens + sent_tokens > max_tokens and current_chunk:
                chunk_text = "".join(current_chunk)
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "text": chunk_text.strip(),
                    "metadata": {
                        "source_file": source_file or (metadata or {}).get("source", "unknown"),
                        "chunk_index": len(chunks),
                        "modality": "text",
                        "content_type": "text_chunk",
                        "token_count": current_tokens,
                        **(metadata or {}),
                    },
                    "modality": "text",
                })
                current_chunk = []
                current_tokens = 0

                # 重叠: 保留最后一句
                if self.chunk_overlap > 0:
                    overlap_sentences = self._estimate_overlap_sentences(
                        chunk_text, enc, self.chunk_overlap
                    )
                    current_chunk = overlap_sentences
                    current_tokens = sum(len(enc.encode(s)) for s in current_chunk)

            current_chunk.append(sentence)
            current_tokens += sent_tokens

        # 最后一块
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text.strip(),
                "metadata": {
                    "source_file": source_file or (metadata or {}).get("source", "unknown"),
                    "chunk_index": len(chunks),
                    "modality": "text",
                    "content_type": "text_chunk",
                    "token_count": current_tokens,
                    **(metadata or {}),
                },
                "modality": "text",
            })

        # 更新 chunk_count
        for c in chunks:
            c["metadata"]["chunk_count"] = len(chunks)

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """简单的句子分割, 兼顾中英文。"""
        # 按中英文标点分割但保留标点
        pattern = r"(?<=[.!?。!?；;])\s*"
        sentences = re.split(pattern, text)
        return [s for s in sentences if s.strip()]

    @staticmethod
    def _estimate_overlap_sentences(
        chunk_text: str, enc, target_overlap_tokens: int
    ) -> List[str]:
        """估计重叠所需的句子数。"""
        sentences = TextChunker._split_sentences(chunk_text)
        overlap = []
        tokens = 0
        for s in reversed(sentences):
            s_tokens = len(enc.encode(s))
            if tokens + s_tokens > target_overlap_tokens and overlap:
                break
            overlap.insert(0, s)
            tokens += s_tokens
        return overlap
