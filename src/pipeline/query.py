"""
QueryPipeline — 端到端查询流水线。

编排: 问题 → 嵌入 → 检索 → 上下文组装 → LLM 生成 → 结构化响应。
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings
from src.embeddings.jina_embedder import JinaEmbedder, Modality
from src.generation.llm_backend import LLMBackend
from src.generation.prompt_templates import PromptTemplate
from src.retrieval.retriever import MultiModalRetriever
from src.storage.schemas import QueryResponse
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class QueryPipeline:
    """端到端查询流水线。

    用法:
        pipeline = QueryPipeline()
        response = pipeline.query("这张图片展示了什么?")
        print(response.answer)
        for src in response.sources:
            print(f"  [{src['modality']}] {src['content_preview']}")
    """

    def __init__(
        self,
        embedder: Optional[JinaEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[MultiModalRetriever] = None,
        llm_backend: Optional[LLMBackend] = None,
        prompt_template: Optional[PromptTemplate] = None,
        top_k: int = 10,
        cross_modal_top_k: int = 5,
    ):
        self.embedder = embedder or JinaEmbedder()
        self.store = vector_store or VectorStore()
        self.retriever = retriever or MultiModalRetriever(self.store, self.embedder)
        self.llm = llm_backend or LLMBackend()
        self.prompt_template = prompt_template or PromptTemplate()
        self.top_k = top_k
        self.cross_modal_top_k = cross_modal_top_k

    # ============================================================
    # 公共 API
    # ============================================================

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        modality_filter: Optional[Modality] = None,
        cross_modal: bool = False,
        include_modalities: Optional[List[Modality]] = None,
        system_prompt: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> QueryResponse:
        """执行完整的 RAG 查询。

        Args:
            question: 用户问题。
            top_k: 检索结果数。
            modality_filter: 按模态过滤检索。
            cross_modal: 是否启用跨模态检索。
            include_modalities: 跨模态时包含的模态列表。
            system_prompt: 自定义系统提示词。

        Returns:
            QueryResponse 包含回答、来源、延迟分解。
        """
        top_k = top_k or self.top_k
        t_total_start = time.perf_counter()

        # 1. 嵌入查询
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_query(question)
        embed_ms = (time.perf_counter() - t0) * 1000

        # 2. 检索
        t0 = time.perf_counter()

        if cross_modal:
            # 跨模态检索
            cross_results = self.retriever.cross_modal_retrieve(
                query=question,
                top_k_per_modality=self.cross_modal_top_k,
                min_similarity=min_similarity,
                modalities=include_modalities,
            )

            # 展平结果并按原始相似度排序
            all_results = []
            for mod_results in cross_results.values():
                all_results.extend(mod_results)
            all_results.sort(key=lambda r: r.score, reverse=True)
            all_results = all_results[:top_k]

            modality_breakdown = {
                mod: len(results) for mod, results in cross_results.items()
            }

            # 构建上下文
            contexts = self.retriever.retrieve_with_context(
                query=question,
                top_k=top_k,
                include_modalities=include_modalities or list(Modality),
            )
        elif modality_filter or include_modalities:
            all_results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                modality_filter=modality_filter,
                min_similarity=min_similarity,
            )
            modality_breakdown = {}
            for r in all_results:
                m = r.chunk.modality.value
                modality_breakdown[m] = modality_breakdown.get(m, 0) + 1

            contexts = self.retriever.retrieve_with_context(
                query=question,
                top_k=top_k,
                include_modalities=[modality_filter] if modality_filter else None,
            )
        else:
            all_results = self.retriever.retrieve(
                query=question,
                top_k=top_k,
                min_similarity=min_similarity,
            )
            modality_breakdown = {}
            for r in all_results:
                m = r.chunk.modality.value
                modality_breakdown[m] = modality_breakdown.get(m, 0) + 1

            contexts = self.retriever.retrieve_with_context(
                query=question,
                top_k=top_k,
            )

        retrieve_ms = (time.perf_counter() - t0) * 1000

        # 3. 生成回答
        t0 = time.perf_counter()
        gen_result = self.llm.generate_with_context(
            query=question,
            retrieved_contexts=contexts,
            prompt_template=self.prompt_template,
            system_prompt=system_prompt,
        )
        generate_ms = gen_result.get("latency_ms", 0)

        total_ms = (time.perf_counter() - t_total_start) * 1000

        # 4. 构建响应
        sources = [r.to_dict() for r in all_results[:top_k]]

        return QueryResponse(
            query=question,
            answer=gen_result["answer"],
            sources=sources,
            latency_breakdown={
                "embed_ms": round(embed_ms, 1),
                "retrieve_ms": round(retrieve_ms, 1),
                "generate_ms": round(generate_ms, 1),
                "total_ms": round(total_ms, 1),
            },
            cross_modal=cross_modal,
            modality_breakdown=modality_breakdown,
        )

    def cross_modal_query(
        self,
        question: str,
        top_k_per_modality: int = 5,
        modalities: Optional[List[Modality]] = None,
    ) -> QueryResponse:
        """跨模态查询 (便捷方法)。

        文本查询 → 从各模态分别检索 → 生成综合回答。
        这是展示 jina-embeddings-v5-omni-small 能力的核心 API。
        """
        return self.query(
            question=question,
            top_k=top_k_per_modality * (len(modalities) if modalities else 4),
            cross_modal=True,
            include_modalities=modalities,
        )

    def retrieve_only(
        self,
        query: str,
        top_k: int = 10,
        modality_filter: Optional[Modality] = None,
    ) -> Dict[str, Any]:
        """仅检索 (不生成回答), 用于调试和展示。"""
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_query(query)
        embed_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            modality_filter=modality_filter,
        )
        retrieve_ms = (time.perf_counter() - t0) * 1000

        modality_breakdown = {}
        for r in results:
            m = r.chunk.modality.value
            modality_breakdown[m] = modality_breakdown.get(m, 0) + 1

        return {
            "query": query,
            "results": [r.to_dict() for r in results],
            "latency_breakdown": {
                "embed_ms": round(embed_ms, 1),
                "retrieve_ms": round(retrieve_ms, 1),
            },
            "modality_breakdown": modality_breakdown,
            "total_results": len(results),
        }

    def retrieve_all_modalities(
        self,
        query: str,
        top_k_per_modality: int = 5,
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """检索并列出所有模态的结果 (用于 UI 展示)。"""
        cross_results = self.retriever.cross_modal_retrieve(
            query=query,
            top_k_per_modality=top_k_per_modality,
            min_similarity=min_similarity,
        )

        output: Dict[str, Any] = {
            "query": query,
            "query_mode": "text",
            "modalities": {},
            "total_results": 0,
        }

        for mod_str, results in cross_results.items():
            output["modalities"][mod_str] = {
                "count": len(results),
                "results": [r.to_dict() for r in results],
            }
            output["total_results"] += len(results)

        return output

    def retrieve_all_modalities_by_file(
        self,
        file_bytes: bytes,
        file_modality: Modality,
        top_k_per_modality: int = 5,
        file_name: str = "",
        min_similarity: float = 0.0,
    ) -> Dict[str, Any]:
        """接受图片/音频/视频文件作为查询，执行跨模态检索。

        Args:
            file_bytes: 文件字节数据。
            file_modality: 文件模态 (IMAGE / AUDIO / VIDEO)。
            top_k_per_modality: 每种模态返回的结果数。
            file_name: 显示用的文件名。

        Returns:
            与 retrieve_all_modalities 相同结构的字典，额外包含 query_file 信息。
        """
        import tempfile
        from src.embeddings.jina_embedder import EmbeddingInput

        suffix = Path(file_name).suffix if file_name else ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            embed_input = EmbeddingInput(
                modality=file_modality,
                content=str(Path(tmp_path).resolve()),
                source_path=file_name,
            )
            result = self.embedder.embed([embed_input])
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        query_embedding = result.embeddings[0]

        cross_results = self.retriever.store.search_cross_modal(
            query_embedding=query_embedding,
            top_k_per_modality=top_k_per_modality,
            min_similarity=min_similarity,
        )

        output: Dict[str, Any] = {
            "query": f"[{file_modality.value.upper()}] {file_name}",
            "query_mode": file_modality.value,
            "query_file": file_name,
            "modalities": {},
            "total_results": 0,
        }

        for mod_str, results in cross_results.items():
            output["modalities"][mod_str] = {
                "count": len(results),
                "results": [r.to_dict() for r in results],
            }
            output["total_results"] += len(results)

        return output

    # ============================================================
    # 便捷方法
    # ============================================================

    def ask_about_file(
        self,
        question: str,
        source_file: str,
        top_k: int = 10,
    ) -> QueryResponse:
        """针对特定来源文件的查询。"""
        t_total_start = time.perf_counter()

        # 检索 (按来源过滤)
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_query(question)
        results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            source_file_filter=source_file,
        )
        retrieve_ms = (time.perf_counter() - t0) * 1000

        # 构建上下文
        from src.embeddings.jina_embedder import EmbeddingInput
        from src.retrieval.retriever import _read_media_base64
        contexts = {
            "results": results,
            "text_contexts": [],
            "image_contexts": [],
            "audio_contexts": [],
            "video_contexts": [],
        }
        for r in results:
            c = r.chunk
            if c.modality == Modality.TEXT:
                contexts["text_contexts"].append({
                    "text": c.text_content or c.content_preview,
                    "source": c.source_file_name,
                    "score": r.score,
                })
            elif c.modality == Modality.IMAGE:
                contexts["image_contexts"].append({
                    "base64": _read_media_base64(c.media_path),
                    "source": c.source_file_name,
                    "score": r.score,
                    "preview": c.content_preview,
                })
            elif c.modality == Modality.AUDIO:
                contexts["audio_contexts"].append({
                    "base64": _read_media_base64(c.media_path),
                    "source": c.source_file_name,
                    "start_sec": c.start_offset,
                    "end_sec": c.end_offset,
                    "score": r.score,
                    "preview": c.content_preview,
                })
            elif c.modality == Modality.VIDEO:
                from src.retrieval.retriever import _extract_video_frames_base64
                contexts["video_contexts"].append({
                    "frames_base64": _extract_video_frames_base64(
                        c.media_path, max_frames=settings.video_qa_frames
                    ),
                    "source": c.source_file_name,
                    "timestamp_sec": c.timestamp_sec,
                    "score": r.score,
                    "preview": c.content_preview,
                })

        # 生成
        t0 = time.perf_counter()
        gen_result = self.llm.generate_with_context(
            query=question,
            retrieved_contexts=contexts,
            prompt_template=self.prompt_template,
        )
        generate_ms = gen_result.get("latency_ms", 0)

        total_ms = (time.perf_counter() - t_total_start) * 1000

        return QueryResponse(
            query=question,
            answer=gen_result["answer"],
            sources=[r.to_dict() for r in results],
            latency_breakdown={
                "embed_ms": 0,  # included in retrieve
                "retrieve_ms": round(retrieve_ms, 1),
                "generate_ms": round(generate_ms, 1),
                "total_ms": round(total_ms, 1),
            },
        )
