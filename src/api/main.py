"""
FastAPI 应用入口 — Multimodal RAG REST API。
启动: uvicorn src.api.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import ingestion, query, management

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期。"""
    logger.info("Multimodal RAG API 启动")
    if STATIC_DIR.exists():
        logger.info(f"静态文件目录: {STATIC_DIR}")
    yield
    logger.info("Multimodal RAG API 关闭")


app = FastAPI(
    title="Multimodal RAG API",
    description="多模态检索增强生成系统的 REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由 (必须在 static mount 之前注册)
app.include_router(ingestion.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(management.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


# 媒体文件服务 (图片/音频/视频帧)
MEDIA_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "media_cache"


@app.get("/api/media/{chunk_id}")
async def serve_media(chunk_id: str):
    """提供媒体文件 (图片、音频、视频帧)。"""
    from fastapi.responses import FileResponse

    # 查找匹配 chunk_id 的文件 (可能有不同后缀)
    for suffix in (".mp4", ".jpg", ".wav", ".mp3", ".png"):
        fpath = MEDIA_CACHE_DIR / f"{chunk_id}{suffix}"
        if fpath.exists():
            media_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".wav": "audio/wav",
                ".mp3": "audio/mpeg",
                ".mp4": "video/mp4",
            }.get(suffix, "application/octet-stream")
            return FileResponse(fpath, media_type=media_type)

    return JSONResponse({"error": "媒体文件不存在"}, status_code=404)


# 静态文件 (SPA 前端)
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
else:
    logger.warning(f"静态文件目录不存在: {STATIC_DIR}")
