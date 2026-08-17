"""
MultiModalRetriever — 多模态检索编排。

支持:
  1. 稠密检索 (纯嵌入相似度)
  2. 跨模态检索 (文本查询 → 图片/音频/视频)
  3. 结果融合与排序
"""

import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import settings
from src.embeddings.jina_embedder import EmbeddingTask, JinaEmbedder, Modality
from src.storage.schemas import SearchResult
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)

# 媒体缓存目录 (与 ingestion.py 保持一致)
_MEDIA_CACHE = Path(__file__).resolve().parent.parent.parent / "media_cache"


def _read_media_base64(media_path: Optional[str]) -> Optional[str]:
    """从媒体缓存文件中读取并返回 base64 编码。"""
    if not media_path:
        return None
    fpath = _MEDIA_CACHE / Path(media_path).name
    if fpath.exists():
        return base64.b64encode(fpath.read_bytes()).decode("utf-8")
    return None


def _extract_video_frames_base64(
    media_path: Optional[str],
    max_frames: int = 4,
) -> List[str]:
    """从预处理后的视频中提取帧，返回 base64 JPEG 字符串列表。

    用于 Q&A 流程中将视频内容送入 LLM。
    视频文件来自 media_cache 中已预处理的 MP4。
    """
    if not media_path:
        return []
    fpath = _MEDIA_CACHE / Path(media_path).name
    if not fpath.exists():
        return []

    # 获取视频时长
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(fpath)],
            capture_output=True, text=True, timeout=10,
        )
        info = json.loads(probe.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
    except Exception:
        duration = 0

    if duration <= 0:
        return []

    # 计算帧间隔 (秒), 均匀提取
    interval = max(duration / max_frames, 0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(fpath),
            "-vf", f"fps=1/{interval}",
            f"{tmpdir}/frame_%02d.jpg",
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
        except Exception as e:
            logger.warning(f"视频帧提取失败: {e}")
            return []

        frames = []
        for f in sorted(Path(tmpdir).glob("frame_*.jpg")):
            try:
                b64 = base64.b64encode(f.read_bytes()).decode("utf-8")
                # 验证是否为有效图片
                from src.generation.prompt_templates import _validate_base64_image
                if _validate_base64_image(b64):
                    frames.append(b64)
            except Exception:
                continue

        return frames[:max_frames]


class MultiModalRetriever:
    """多模态检索器。

    编排嵌入和向量搜索, 支持:
      - 单模态检索 (指定过滤某模态)
      - 跨模态检索 (对所有模态分别检索)
      - 结果融合 (加权合并各模态结果)

    Usage:
        retriever = MultiModalRetriever(vector_store, embedder)
        results = retriever.retrieve("日落是什么颜色的?", top_k=10)
        cross_modal = retriever.cross_modal_retrieve("ocean waves")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: JinaEmbedder,
    ):
        self.store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        modality_filter: Optional[Modality] = None,
        source_file_filter: Optional[str] = None,
        fusion_weights: Optional[Dict[str, float]] = None,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """主检索方法。

        Args:
            query: 查询文本。
            top_k: 返回结果数。
            modality_filter: 按模态过滤 (None = 所有模态)。
            source_file_filter: 按来源文件过滤。
            fusion_weights: 模态融合权重, 如 {"text": 1.0, "image": 0.8}。

        Returns:
            排序后的 SearchResult 列表。
        """
        # 1. 嵌入查询
        query_embedding = self.embedder.embed_query(query)

        # 2. 加权融合检索
        if fusion_weights and not modality_filter:
            return self._weighted_fusion_retrieve(
                query_embedding, top_k, fusion_weights, min_similarity
            )

        # 3. 标准检索 (指定了模态过滤或来源过滤, 直接查)
        if modality_filter or source_file_filter:
            return self.store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                modality_filter=modality_filter,
                source_file_filter=source_file_filter,
                min_similarity=min_similarity,
            )

        # 4. 无过滤: 全局检索, 按原始相似度排序
        return self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
        )

    def cross_modal_retrieve(
        self,
        query: str,
        top_k_per_modality: int = 5,
        modalities: Optional[List[Modality]] = None,
        min_similarity: float = 0.0,
    ) -> Dict[str, List[SearchResult]]:
        """跨模态检索: 一次查询, 从每种模态分别返回结果。

        这是展示 jina-embeddings-v5-omni-small 统一嵌入空间
        能力的核心功能 —— 文本查询可以直接检索到语义相关的
        图片、音频和视频。

        Args:
            query: 查询文本。
            top_k_per_modality: 每种模态返回的结果数。
            modalities: 要检索的模态, 默认全部。

        Returns:
            {modality_value: [SearchResult, ...]} 字典。
        """
        query_embedding = self.embedder.embed_query(query)

        return self.store.search_cross_modal(
            query_embedding=query_embedding,
            top_k_per_modality=top_k_per_modality,
            modalities=modalities,
            min_similarity=min_similarity,
        )

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 10,
        include_modalities: Optional[List[Modality]] = None,
    ) -> Dict[str, Any]:
        """检索并组织上下文, 供 LLM 使用。

        Args:
            query: 查询文本。
            top_k: 总返回结果数。
            include_modalities: 要包含的模态, 默认全部。

        Returns:
            包含以下字段的字典:
                - results: 所有检索结果
                - text_contexts: 文本上下文列表
                - image_contexts: 图片上下文列表 (含 base64)
                - audio_contexts: 音频上下文列表
                - video_contexts: 视频帧上下文列表
        """
        if include_modalities:
            # 对每种模态分别检索
            all_results = []
            per_modality = max(top_k // len(include_modalities), 3)
            for mod in include_modalities:
                mod_results = self.retrieve(
                    query, top_k=per_modality, modality_filter=mod
                )
                all_results.extend(mod_results)
            # 按原始相似度排序
            all_results.sort(key=lambda r: r.score, reverse=True)
            all_results = all_results[:top_k]
        else:
            all_results = self.retrieve(query, top_k=top_k)

        # 按模态分组
        text_contexts = []
        image_contexts = []
        audio_contexts = []
        video_contexts = []

        for r in all_results:
            chunk = r.chunk
            if chunk.modality == Modality.TEXT:
                text_contexts.append({
                    "text": chunk.text_content or chunk.content_preview,
                    "source": chunk.source_file_name,
                    "score": r.score,
                })
            elif chunk.modality == Modality.IMAGE:
                image_contexts.append({
                    "base64": _read_media_base64(chunk.media_path),
                    "thumbnail": chunk.thumbnail_base64,
                    "source": chunk.source_file_name,
                    "score": r.score,
                    "preview": chunk.content_preview,
                })
            elif chunk.modality == Modality.AUDIO:
                audio_contexts.append({
                    "base64": _read_media_base64(chunk.media_path),
                    "source": chunk.source_file_name,
                    "start_sec": chunk.start_offset,
                    "end_sec": chunk.end_offset,
                    "score": r.score,
                    "preview": chunk.content_preview,
                })
            elif chunk.modality == Modality.VIDEO:
                video_contexts.append({
                    "frames_base64": _extract_video_frames_base64(
                        chunk.media_path, max_frames=settings.video_qa_frames
                    ),
                    "thumbnail": chunk.thumbnail_base64,
                    "source": chunk.source_file_name,
                    "timestamp_sec": chunk.timestamp_sec,
                    "score": r.score,
                    "preview": chunk.content_preview,
                })

        return {
            "results": all_results,
            "text_contexts": text_contexts,
            "image_contexts": image_contexts,
            "audio_contexts": audio_contexts,
            "video_contexts": video_contexts,
        }

    @staticmethod
    def _calibrate_scores_by_modality(results: List[SearchResult]) -> List[SearchResult]:
        """组内归一化: 消除文本查询对文本模态的自然偏向。

        对每种模态分别做最大归一化 (score / max_score_in_group)，
        使不同模态的 top 结果可以公平比较。
        """
        if not results:
            return results

        groups: Dict[str, List[SearchResult]] = {}
        for r in results:
            mod = r.chunk.modality.value
            groups.setdefault(mod, []).append(r)

        calibrated = []
        for mod_items in groups.values():
            max_score = max(r.score for r in mod_items)
            if max_score > 0:
                for r in mod_items:
                    r.score = r.score / max_score
            calibrated.extend(mod_items)

        calibrated.sort(key=lambda r: r.score, reverse=True)
        return calibrated

    def _weighted_fusion_retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        weights: Dict[str, float],
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """加权融合检索: 对各模态分别检索, 加权合并结果。"""
        all_results: List[SearchResult] = []

        for mod_str, weight in weights.items():
            if weight <= 0:
                continue
            try:
                modality = Modality(mod_str)
            except ValueError:
                continue

            mod_top_k = max(
                int(top_k * weight / sum(weights.values())), 2
            )
            mod_results = self.store.search(
                query_embedding=query_embedding,
                top_k=mod_top_k,
                modality_filter=modality,
                min_similarity=min_similarity,
            )
            # 应用权重
            for r in mod_results:
                r.score *= weight
            all_results.extend(mod_results)

        # 重排
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]
