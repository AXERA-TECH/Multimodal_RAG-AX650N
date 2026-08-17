"""
VectorStore — ChromaDB 多模态向量存储。

单 collection 存储所有模态的嵌入向量,
通过 metadata 过滤实现跨模态检索。
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from config.settings import settings
from src.storage.schemas import MediaChunk, Modality, SearchResult

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB 多模态向量存储。

    在单个 ChromaDB collection 中管理所有模态的嵌入向量,
    使用 modality 元数据字段进行过滤, 支持:
      - 按模态过滤检索
      - 跨模态并行检索 (一次查询, 各模态分别返回 top-k)
      - 按来源文件过滤
      - 增量入库 (跳过已索引文件)

    Usage:
        store = VectorStore()
        store.add(embeddings, chunks)
        results = store.search(query_embedding, top_k=10, modality_filter=Modality.IMAGE)
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        self.embedding_dim = embedding_dim or settings.embedding_dimensions

        # 确保持久化目录存在
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        self._client = None
        self._collection = None
        self._init_client()

    def _init_client(self):
        """初始化 ChromaDB 客户端和 collection。"""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            # 获取或创建 collection
            self._collection = self._get_or_create_collection()
            logger.info(
                f"已加载 collection: {self.collection_name} "
                f"(count={self._collection.count()})"
            )

        except ImportError:
            raise ImportError("需要 chromadb。请安装: pip install chromadb")

    def _get_or_create_collection(self):
        """获取或创建 collection（容错处理：被其他进程删除后自动重建）。"""
        try:
            return self._client.get_collection(name=self.collection_name)
        except Exception:
            logger.warning(f"Collection 不存在，正在创建: {self.collection_name}")
            return self._client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Multimodal RAG collection",
                    "embedding_model": settings.embedding_model_name,
                    "embedding_dim": self.embedding_dim,
                },
            )

    def _ensure_collection(self):
        """确保 collection 可用，如已被外部删除则自动重建。"""
        try:
            self._collection.count()
        except Exception:
            logger.warning("Collection 引用失效，正在重新连接...")
            self._collection = self._get_or_create_collection()

    # ============================================================
    # CRUD 操作
    # ============================================================

    def add(
        self,
        embeddings: np.ndarray,
        chunks: List[MediaChunk],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """批量添加嵌入向量及其元数据。

        Args:
            embeddings: 嵌入矩阵, shape (n, dim)。
            chunks: MediaChunk 列表, 长度与 embeddings 一致。
            ids: 自定义 ID 列表, 默认使用 chunk.chunk_id。

        Returns:
            添加的 chunk ID 列表。
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"embeddings 和 chunks 长度不一致: {len(embeddings)} vs {len(chunks)}"
            )

        if len(embeddings) == 0:
            return []

        ids = ids or [c.chunk_id for c in chunks]
        metadatas = [c.to_chroma_metadata() for c in chunks]

        # 文档内容: 对于文本存储原文, 对于媒体存储预览文本
        documents = []
        for c in chunks:
            if c.modality == Modality.TEXT and c.text_content:
                documents.append(c.text_content)
            else:
                documents.append(c.content_preview or "")

        try:
            self._ensure_collection()
            self._collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=documents,
            )
            logger.debug(f"已添加 {len(ids)} 个向量到 collection")
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            raise

        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        modality_filter: Optional[Modality] = None,
        source_file_filter: Optional[str] = None,
        where_clause: Optional[dict] = None,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """相似度搜索。

        Args:
            query_embedting: 查询向量, shape (dim,)。
            top_k: 返回结果数。
            modality_filter: 按模态过滤。
            source_file_filter: 按来源文件过滤。
            where_clause: 自定义 ChromaDB where 过滤条件。
            min_similarity: 最低相似度阈值 (0.0~1.0), 低于此值的结果被过滤。

        Returns:
            SearchResult 列表, 按相似度降序排列。
        """
        where = self._build_where_clause(
            modality_filter, source_file_filter, where_clause
        )

        try:
            self._ensure_collection()
            chroma_results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

        results = []
        if chroma_results["ids"] and chroma_results["ids"][0]:
            for i, chunk_id in enumerate(chroma_results["ids"][0]):
                metadata = (
                    chroma_results["metadatas"][0][i]
                    if chroma_results["metadatas"] and chroma_results["metadatas"][0]
                    else {}
                )
                document = (
                    chroma_results["documents"][0][i]
                    if chroma_results["documents"] and chroma_results["documents"][0]
                    else None
                )
                distance = (
                    chroma_results["distances"][0][i]
                    if chroma_results["distances"] and chroma_results["distances"][0]
                    else 0.0
                )
                similarity = 1.0 - (distance / 2.0) if distance else 1.0

                # 相似度阈值过滤
                if similarity < min_similarity:
                    continue

                chunk = MediaChunk.from_chroma_result(
                    chroma_id=chunk_id,
                    metadata=metadata,
                    document=document,
                )

                results.append(
                    SearchResult(chunk=chunk, score=similarity, rank=i + 1)
                )

        return results

    def search_cross_modal(
        self,
        query_embedding: np.ndarray,
        top_k_per_modality: int = 5,
        modalities: Optional[List[Modality]] = None,
        min_similarity: float = 0.0,
    ) -> Dict[str, List[SearchResult]]:
        """跨模态搜索: 对每种模态分别检索。"""
        if modalities is None:
            modalities = list(Modality)

        results_by_modality: Dict[str, List[SearchResult]] = {}

        for modality in modalities:
            mod_results = self.search(
                query_embedding=query_embedding,
                top_k=top_k_per_modality,
                modality_filter=modality,
                min_similarity=min_similarity,
            )
            if mod_results:
                results_by_modality[modality.value] = mod_results

        return results_by_modality

    def delete_by_source(self, source_file: str) -> int:
        """删除来自指定文件的所有 chunks。

        Args:
            source_file: 来源文件路径。

        Returns:
            删除的 chunk 数量。
        """
        try:
            self._ensure_collection()
            # 先查询 IDs
            results = self._collection.get(
                where={"source_file": source_file},
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                self._collection.delete(ids=ids)
                logger.info(f"已删除 {len(ids)} 个 chunks (source: {source_file})")
            return len(ids)
        except Exception as e:
            logger.error(f"删除失败: {e}")
            return 0

    def delete_by_ids(self, ids: List[str]) -> int:
        """按 ID 列表删除 chunks。"""
        try:
            self._ensure_collection()
            self._collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            logger.error(f"删除失败: {e}")
            return 0

    def clear(self) -> int:
        """清空整个 collection。"""
        count = self.count()
        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Multimodal RAG collection",
                    "embedding_model": settings.embedding_model_name,
                },
            )
            logger.info(f"已清空 collection, 删除了 {count} 个 chunks")
        except Exception as e:
            logger.error(f"清空失败: {e}")
        return count

    # ============================================================
    # 查询操作
    # ============================================================

    def count(self) -> int:
        """返回 collection 中的 chunk 总数。"""
        try:
            self._ensure_collection()
            return self._collection.count()
        except Exception:
            return 0

    def list_sources(self) -> List[Dict[str, Any]]:
        """列出所有已索引的来源文件及其统计。"""
        try:
            self._ensure_collection()
            results = self._collection.get(include=["metadatas"])
            sources: Dict[str, Dict[str, Any]] = {}

            if results["metadatas"]:
                for m in results["metadatas"]:
                    src = m.get("source_file", "unknown")
                    mod = m.get("modality", "text")
                    if src not in sources:
                        sources[src] = {
                            "source_file": src,
                            "source_file_name": m.get("source_file_name", ""),
                            "count": 0,
                            "modalities": {},
                        }
                    sources[src]["count"] += 1
                    sources[src]["modalities"][mod] = (
                        sources[src]["modalities"].get(mod, 0) + 1
                    )

            return list(sources.values())
        except Exception as e:
            logger.error(f"列出 sources 失败: {e}")
            return []

    def get_chunks_by_source(self, source_file: str) -> List[MediaChunk]:
        """获取指定来源的所有 chunks。"""
        try:
            self._ensure_collection()
            results = self._collection.get(
                where={"source_file": source_file},
                include=["metadatas", "documents"],
            )
            chunks = []
            if results["ids"]:
                for i, cid in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    doc = results["documents"][i] if results["documents"] else None
                    chunks.append(
                        MediaChunk.from_chroma_result(cid, meta, doc)
                    )
            return chunks
        except Exception as e:
            logger.error(f"获取 chunks 失败: {e}")
            return []

    def is_source_indexed(self, source_file: str) -> bool:
        """检查来源文件是否已索引。"""
        try:
            results = self._collection.get(
                where={"source_file": source_file},
                include=[],
            )
            return len(results.get("ids", [])) > 0
        except Exception:
            return False

    # ============================================================
    # 内部辅助
    # ============================================================

    def _build_where_clause(
        self,
        modality_filter: Optional[Modality] = None,
        source_file_filter: Optional[str] = None,
        custom_where: Optional[dict] = None,
    ) -> Optional[dict]:
        """构建 ChromaDB where 过滤条件。"""
        conditions = []

        if modality_filter:
            conditions.append({"modality": modality_filter.value})

        if source_file_filter:
            conditions.append({"source_file": source_file_filter})

        if custom_where:
            conditions.append(custom_where)

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def get_stats(self) -> Dict[str, Any]:
        """获取 collection 统计信息。"""
        try:
            results = self._collection.get(include=["metadatas"])
            total = len(results.get("ids", []))

            modality_counts = {"text": 0, "image": 0, "audio": 0, "video": 0}
            if results["metadatas"]:
                for m in results["metadatas"]:
                    mod = m.get("modality", "text")
                    modality_counts[mod] = modality_counts.get(mod, 0) + 1

            return {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "total_chunks": total,
                "embedding_dim": self.embedding_dim,
                "modality_breakdown": modality_counts,
                "unique_sources": len(
                    set(
                        m.get("source_file", "")
                        for m in (results.get("metadatas") or [])
                    )
                ),
            }
        except Exception as e:
            logger.error(f"获取统计失败: {e}")
            return {"error": str(e)}
