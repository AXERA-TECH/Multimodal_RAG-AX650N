"""
管理 API 路由 — 向量库统计、来源管理、清理操作、模型配置。
"""

import logging
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.settings import settings
from src.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["management"])

_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


@router.get("/stats")
async def get_stats():
    """获取向量库统计信息。"""
    store = get_store()
    try:
        stats = store.get_stats()
        return JSONResponse(stats)
    except Exception as e:
        logger.error(f"获取统计失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/sources")
async def list_sources():
    """列出所有已索引的来源文件。"""
    store = get_store()
    try:
        sources = store.list_sources()
        return JSONResponse({"sources": sources, "total": len(sources)})
    except Exception as e:
        logger.error(f"列出 sources 失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.delete("/sources")
async def delete_source(source_file: str = Form(...)):
    """删除指定来源文件的所有 chunks。"""
    store = get_store()
    try:
        count = store.delete_by_source(source_file)
        return JSONResponse({
            "deleted": count,
            "source_file": source_file,
            "message": f"已删除 {count} 个 chunks",
        })
    except Exception as e:
        logger.error(f"删除 source 失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/sources/check")
async def check_source(source_file: str = Form(...)):
    """检查来源文件是否已索引。"""
    store = get_store()
    try:
        indexed = store.is_source_indexed(source_file)
        return JSONResponse({"source_file": source_file, "indexed": indexed})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/sources/chunks")
async def get_source_chunks(source_file: str):
    """获取指定来源文件的所有 chunks 详情。"""
    store = get_store()
    try:
        chunks = store.get_chunks_by_source(source_file)
        return JSONResponse({
            "source_file": source_file,
            "total": len(chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "modality": c.modality.value,
                    "content_type": c.content_type,
                    "content_preview": c.content_preview,
                    "text_content": c.text_content,
                    "media_url": f"/api/media/{c.chunk_id}" if c.media_path else None,
                    "thumbnail_base64": c.thumbnail_base64,
                    "start_offset": c.start_offset,
                    "end_offset": c.end_offset,
                    "timestamp_sec": c.timestamp_sec,
                    "frame_index": c.frame_index,
                    "embedding_dim": c.embedding_dim,
                    "created_at": c.created_at,
                }
                for c in chunks
            ],
        })
    except Exception as e:
        logger.error(f"获取 chunks 失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.delete("/collection")
async def clear_collection():
    """清空整个向量库 collection。"""
    store = get_store()
    try:
        count = store.clear()
        return JSONResponse({
            "deleted": count,
            "message": f"已清空 collection，删除了 {count} 个 chunks",
        })
    except Exception as e:
        logger.error(f"清空失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# ============================================================
# 模型配置
# ============================================================

ENV_PATH = Path(__file__).resolve().parent.parent.parent.parent / ".env"


class ModelConfigUpdate(BaseModel):
    llm_api_base: Optional[str] = None
    llm_model_name: Optional[str] = None
    embedding_api_base: Optional[str] = None
    embedding_model_name: Optional[str] = None
    # 预处理参数
    text_chunk_size: Optional[int] = None
    text_chunk_overlap: Optional[int] = None
    image_target_size: Optional[int] = None
    image_quality: Optional[int] = None
    audio_max_duration_sec: Optional[float] = None
    audio_overlap_sec: Optional[float] = None
    video_target_size: Optional[int] = None
    video_max_frames: Optional[int] = None
    video_qa_frames: Optional[int] = None


@router.get("/config")
async def get_config():
    """获取当前配置。"""
    return JSONResponse({
        "llm_api_base": settings.llm_api_base,
        "llm_model_name": settings.llm_model_name,
        "llm_max_tokens": settings.llm_max_tokens,
        "llm_temperature": settings.llm_temperature,
        "embedding_api_base": settings.embedding_api_base,
        "embedding_model_name": settings.embedding_model_name,
        "embedding_dimensions": settings.embedding_dimensions,
        "preprocessing": {
            "text_chunk_size": settings.text_chunk_size,
            "text_chunk_overlap": settings.text_chunk_overlap,
            "image_target_size": settings.image_target_size,
            "image_quality": settings.image_quality,
            "audio_max_duration_sec": settings.audio_max_duration_sec,
            "audio_overlap_sec": settings.audio_overlap_sec,
            "video_target_size": settings.video_target_size,
            "video_max_frames": settings.video_max_frames,
            "video_qa_frames": settings.video_qa_frames,
        },
    })


@router.post("/config")
async def update_config(req: ModelConfigUpdate):
    """更新配置 (写入 .env 文件, 需重启生效)。"""
    try:
        content = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""

        updates = {}
        if req.llm_api_base is not None:
            updates["LLM_API_BASE"] = req.llm_api_base
        if req.llm_model_name is not None:
            updates["LLM_MODEL_NAME"] = req.llm_model_name
        if req.embedding_api_base is not None:
            updates["EMBEDDING_API_BASE"] = req.embedding_api_base
        if req.embedding_model_name is not None:
            updates["EMBEDDING_MODEL_NAME"] = req.embedding_model_name
        # 预处理参数
        if req.text_chunk_size is not None:
            updates["TEXT_CHUNK_SIZE"] = str(req.text_chunk_size)
        if req.text_chunk_overlap is not None:
            updates["TEXT_CHUNK_OVERLAP"] = str(req.text_chunk_overlap)
        if req.image_target_size is not None:
            updates["IMAGE_TARGET_SIZE"] = str(req.image_target_size)
        if req.image_quality is not None:
            updates["IMAGE_QUALITY"] = str(req.image_quality)
        if req.audio_max_duration_sec is not None:
            updates["AUDIO_MAX_DURATION_SEC"] = str(req.audio_max_duration_sec)
        if req.audio_overlap_sec is not None:
            updates["AUDIO_OVERLAP_SEC"] = str(req.audio_overlap_sec)
        if req.video_target_size is not None:
            updates["VIDEO_TARGET_SIZE"] = str(req.video_target_size)
        if req.video_max_frames is not None:
            updates["VIDEO_MAX_FRAMES"] = str(req.video_max_frames)
        if req.video_qa_frames is not None:
            updates["VIDEO_QA_FRAMES"] = str(req.video_qa_frames)

        for key, value in updates.items():
            if re.search(f"^{key}=", content, re.MULTILINE):
                content = re.sub(f"^{key}=.*$", f"{key}={value}", content, flags=re.MULTILINE)
            else:
                content += f"\n{key}={value}\n"

        ENV_PATH.write_text(content, encoding="utf-8")
        return JSONResponse({"status": "ok", "updated": updates, "message": "配置已保存，重启服务后生效"})
    except Exception as e:
        logger.error(f"更新配置失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
