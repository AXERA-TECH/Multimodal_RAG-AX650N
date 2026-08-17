"""
数据模型定义 — MediaChunk, SearchResult, Modality 枚举等。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class ContentType(str, Enum):
    TEXT_CHUNK = "text_chunk"
    IMAGE = "image"
    AUDIO_SEGMENT = "audio_segment"
    VIDEO_FRAME = "video_frame"


@dataclass
class MediaChunk:
    """向量库中存储的多模态数据块。

    每个 MediaChunk 代表一个已嵌入的媒体片段:
      - 文本块: 来自文档的 ~512 token 文本
      - 图片: 单张图片 (可含缩略图)
      - 音频片段: ~25s 的音频
      - 视频帧: 视频的关键帧图片
    """

    chunk_id: str
    modality: Modality
    content_type: str
    source_file: str
    source_file_name: str = ""
    content_preview: str = ""  # 人类可读预览

    # 文本内容 (仅 TEXT 模态)
    text_content: Optional[str] = None

    # 媒体文件缓存路径 (相对路径，指向 media_cache/ 下的文件)
    # IMAGE/VIDEO_FRAME: JPEG 图片，AUDIO: WAV 音频片段
    media_path: Optional[str] = None

    # 缩略图 base64 (小尺寸，可存入 metadata)
    thumbnail_base64: Optional[str] = None

    # 时间偏移 (AUDIO / VIDEO)
    start_offset: Optional[float] = None
    end_offset: Optional[float] = None

    # 帧信息 (VIDEO)
    frame_index: Optional[int] = None
    timestamp_sec: Optional[float] = None

    # 嵌入维度
    embedding_dim: int = 1024

    # 原始文件内容路径 (供 LLM 引用)
    original_content_path: Optional[str] = None

    # 创建时间
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # 附加元数据
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """转为 ChromaDB 兼容的元数据字典。"""
        return {
            "chunk_id": self.chunk_id,
            "modality": self.modality.value,
            "content_type": self.content_type,
            "source_file": self.source_file,
            "source_file_name": self.source_file_name,
            "content_preview": self.content_preview or "",
            "media_path": self.media_path or "",
            "start_offset": self.start_offset or 0.0,
            "end_offset": self.end_offset or 0.0,
            "frame_index": self.frame_index if self.frame_index is not None else -1,
            "timestamp_sec": self.timestamp_sec or 0.0,
            "embedding_dim": self.embedding_dim,
            "created_at": self.created_at,
            "thumbnail_base64": self.thumbnail_base64 or "",
        }

    @classmethod
    def from_chroma_result(
        cls,
        chroma_id: str,
        metadata: Dict[str, Any],
        document: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> "MediaChunk":
        """从 ChromaDB 查询结果构造 MediaChunk。"""
        return cls(
            chunk_id=metadata.get("chunk_id", chroma_id),
            modality=Modality(metadata.get("modality", "text")),
            content_type=metadata.get("content_type", "text_chunk"),
            source_file=metadata.get("source_file", ""),
            source_file_name=metadata.get("source_file_name", ""),
            content_preview=metadata.get("content_preview", document or ""),
            text_content=document if metadata.get("modality") == "text" else None,
            media_path=metadata.get("media_path") or None,
            thumbnail_base64=metadata.get("thumbnail_base64"),
            start_offset=metadata.get("start_offset"),
            end_offset=metadata.get("end_offset"),
            frame_index=metadata.get("frame_index"),
            timestamp_sec=metadata.get("timestamp_sec"),
            embedding_dim=metadata.get("embedding_dim", 1024),
            created_at=metadata.get("created_at", ""),
        )


@dataclass
class SearchResult:
    """检索结果。"""

    chunk: MediaChunk
    score: float
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "modality": self.chunk.modality.value,
            "content_type": self.chunk.content_type,
            "source_file": self.chunk.source_file,
            "source_file_name": self.chunk.source_file_name,
            "content_preview": self.chunk.content_preview,
            "text_content": self.chunk.text_content,
            "media_url": f"/api/media/{self.chunk.chunk_id}" if self.chunk.media_path else None,
            "thumbnail_base64": self.chunk.thumbnail_base64,
            "start_offset": self.chunk.start_offset,
            "end_offset": self.chunk.end_offset,
            "timestamp_sec": self.chunk.timestamp_sec,
            "score": round(self.score, 4),
            "rank": self.rank,
        }


@dataclass
class IngestionStats:
    """入库统计。"""

    files_processed: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    chunks_by_modality: Dict[str, int] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "chunks_created": self.chunks_created,
            "chunks_by_modality": self.chunks_by_modality,
            "total_latency_sec": round(self.total_latency_ms / 1000, 2),
            "errors": self.errors,
        }


@dataclass
class QueryResponse:
    """查询响应。"""

    query: str
    answer: str
    sources: List[Dict[str, Any]]
    latency_breakdown: Dict[str, float]  # {embed_ms, retrieve_ms, generate_ms, total_ms}
    cross_modal: bool = False
    modality_breakdown: Dict[str, int] = field(default_factory=dict)
