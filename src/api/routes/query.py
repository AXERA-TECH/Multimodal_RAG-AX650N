"""
查询 API 路由 — RAG 问答、检索、跨模态搜索。
"""

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.embeddings.jina_embedder import Modality
from src.pipeline.query import QueryPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])

_pipeline: Optional[QueryPipeline] = None


def get_pipeline() -> QueryPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = QueryPipeline()
    return _pipeline


class QueryRequest(BaseModel):
    question: str
    top_k: int = 10
    modality_filter: Optional[str] = None
    cross_modal: bool = False
    include_modalities: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    min_similarity: float = 0.0


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 10
    modality_filter: Optional[str] = None
    cross_modal: bool = False
    top_k_per_modality: int = 5


@router.post("")
async def ask_question(req: QueryRequest):
    """完整 RAG 查询: 检索 + LLM 生成回答。"""
    pipeline = get_pipeline()

    modality_filter = None
    if req.modality_filter:
        try:
            modality_filter = Modality(req.modality_filter)
        except ValueError:
            return JSONResponse(
                {"error": f"无效的模态: {req.modality_filter}"}, status_code=400
            )

    include_modalities = None
    if req.include_modalities:
        try:
            include_modalities = [Modality(m) for m in req.include_modalities]
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    try:
        response = pipeline.query(
            question=req.question,
            top_k=req.top_k,
            modality_filter=modality_filter,
            cross_modal=req.cross_modal,
            include_modalities=include_modalities,
            system_prompt=req.system_prompt,
            min_similarity=req.min_similarity,
        )

        return JSONResponse({
            "query": response.query,
            "answer": response.answer,
            "sources": response.sources,
            "latency_breakdown": response.latency_breakdown,
            "cross_modal": response.cross_modal,
            "modality_breakdown": response.modality_breakdown,
        })
    except Exception as e:
        logger.error(f"查询失败: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/retrieve")
async def retrieve_only(req: RetrieveRequest):
    """仅检索 (不生成回答), 用于调试和 UI 展示。"""
    pipeline = get_pipeline()

    modality_filter = None
    if req.modality_filter:
        try:
            modality_filter = Modality(req.modality_filter)
        except ValueError:
            return JSONResponse(
                {"error": f"无效的模态: {req.modality_filter}"}, status_code=400
            )

    try:
        if req.cross_modal:
            result = pipeline.retrieve_all_modalities(
                query=req.query,
                top_k_per_modality=req.top_k_per_modality,
            )
        else:
            result = pipeline.retrieve_only(
                query=req.query,
                top_k=req.top_k,
                modality_filter=modality_filter,
            )

        return JSONResponse(result)
    except Exception as e:
        logger.error(f"检索失败: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/cross-modal")
async def cross_modal_search(
    query: Optional[str] = Form(default=None),
    top_k_per_modality: int = Form(default=5),
    modalities: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    min_similarity: float = Form(default=0.0),
):
    """跨模态搜索: 文本或文件查询检索所有模态的结果。

    支持两种模式:
      - 文本查询: 传 query 参数
      - 文件查询: 传 file 参数 (图片/音频/视频直接用 Jina 多模态嵌入)
      同时传时优先使用文件。
    """
    pipeline = get_pipeline()

    mod_list = None
    if modalities:
        try:
            mod_list = [Modality(m.strip()) for m in modalities.split(",")]
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

    try:
        if file is not None:
            content = await file.read()
            mod = _detect_file_modality(file.filename or "unknown", content)
            result = pipeline.retrieve_all_modalities_by_file(
                file_bytes=content,
                file_modality=mod,
                top_k_per_modality=top_k_per_modality,
                file_name=file.filename or "upload",
                min_similarity=min_similarity,
            )
        elif query and query.strip():
            result = pipeline.retrieve_all_modalities(
                query=query.strip(),
                top_k_per_modality=top_k_per_modality,
                min_similarity=min_similarity,
            )
        else:
            return JSONResponse(
                {"error": "请提供 query 文本或上传文件"}, status_code=400
            )

        return JSONResponse(result)
    except Exception as e:
        logger.error(f"跨模态搜索失败: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


def _detect_file_modality(filename: str, content: bytes) -> Modality:
    """通过文件扩展名和内容魔数检测模态。"""
    suffix = Path(filename).suffix.lower()

    # 图片扩展名
    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}:
        return Modality.IMAGE
    # 音频扩展名
    if suffix in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma"}:
        return Modality.AUDIO
    # 视频扩展名
    if suffix in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"}:
        return Modality.VIDEO

    # 回退: 检查文件头魔数
    if content[:4] == b"\xff\xd8\xff" or content[:8] == b"\x89PNG\r\n\x1a\n":
        return Modality.IMAGE
    if content[:4] == b"RIFF" or content[:3] == b"ID3":
        return Modality.AUDIO

    raise ValueError(f"无法识别文件模态: {filename}")


@router.post("/ask-about-file")
async def ask_about_file(
    question: str = Form(...),
    source_file: str = Form(...),
    top_k: int = Form(default=10),
):
    """针对特定已索引文件的问答。"""
    pipeline = get_pipeline()

    try:
        response = pipeline.ask_about_file(
            question=question,
            source_file=source_file,
            top_k=top_k,
        )
        return JSONResponse({
            "query": response.query,
            "answer": response.answer,
            "sources": response.sources,
            "latency_breakdown": response.latency_breakdown,
        })
    except Exception as e:
        logger.error(f"查询失败: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)
