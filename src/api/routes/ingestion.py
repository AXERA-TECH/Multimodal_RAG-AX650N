"""
数据入库 API 路由。
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from config.settings import settings
from src.pipeline.ingestion import IngestionPipeline, EXTENSION_MAP

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])

_pipeline: Optional[IngestionPipeline] = None


def get_pipeline(incremental: bool = True) -> IngestionPipeline:
    """获取入库流水线。每次调用时刷新预处理参数以反映 .env 变更。"""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline(incremental=incremental)
    else:
        _pipeline.incremental = incremental
        # 热更新预处理参数（避免重启服务）
        _pipeline.image_processor.target_size = settings.image_target_size
        _pipeline.image_processor.quality = settings.image_quality
        _pipeline.audio_processor.max_duration_sec = settings.audio_max_duration_sec
        _pipeline.audio_processor.overlap_sec = settings.audio_overlap_sec
        _pipeline.video_processor.target_size = settings.video_target_size
        _pipeline.video_processor.max_frames = settings.video_max_frames
    return _pipeline


@router.post("/files")
async def ingest_files(
    files: List[UploadFile] = File(...),
    incremental: bool = Form(default=True),
):
    """上传并入库多个文件。

    支持所有配置的文件格式 (文本/图片/音频/视频)。
    """
    pipeline = get_pipeline(incremental=incremental)

    results = []
    total_stats = {
        "files_processed": 0,
        "files_failed": 0,
        "chunks_created": 0,
        "chunks_by_modality": {},
        "errors": [],
    }

    for file in files:
        # 保存上传文件到临时目录
        suffix = Path(file.filename or "upload").suffix
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="rag_upload_"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            stats = pipeline.ingest_file(tmp_path, display_name=file.filename)
            results.append({
                "filename": file.filename,
                "status": "success" if stats.files_processed > 0 else "skipped",
                "chunks": stats.chunks_created,
                "modalities": stats.chunks_by_modality,
                "latency_ms": round(stats.total_latency_ms, 1),
            })

            total_stats["files_processed"] += stats.files_processed
            total_stats["files_failed"] += stats.files_failed
            total_stats["chunks_created"] += stats.chunks_created
            for m, c in stats.chunks_by_modality.items():
                total_stats["chunks_by_modality"][m] = (
                    total_stats["chunks_by_modality"].get(m, 0) + c
                )
            total_stats["errors"].extend(stats.errors)

        except Exception as e:
            logger.error(f"入库失败 {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e),
            })
            total_stats["files_failed"] += 1
            total_stats["errors"].append(f"{file.filename}: {str(e)}")

        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return JSONResponse({
        "results": results,
        "summary": total_stats,
    })


@router.post("/directory")
async def ingest_directory(
    directory: str = Form(...),
    recursive: bool = Form(default=True),
    incremental: bool = Form(default=True),
):
    """入库指定目录中的所有支持文件。"""
    pipeline = get_pipeline(incremental=incremental)

    try:
        stats = pipeline.ingest_directory(directory, recursive=recursive)
        return JSONResponse(stats.to_dict())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@router.get("/extensions")
async def get_supported_extensions():
    """获取支持的文件扩展名列表。"""
    by_modality = {}
    for ext, mod in sorted(EXTENSION_MAP.items()):
        mod_str = mod.value
        if mod_str not in by_modality:
            by_modality[mod_str] = []
        by_modality[mod_str].append(ext)

    return JSONResponse({
        "extensions": by_modality,
        "mime_hint": {
            "text": "text/*, application/pdf, application/json",
            "image": "image/*",
            "audio": "audio/*",
            "video": "video/*",
        },
    })
