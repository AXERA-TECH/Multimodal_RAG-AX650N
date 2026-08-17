"""
全局配置 — 使用 Pydantic Settings 从 .env 和环境变量加载。
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """多模态 RAG 系统全局配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 项目路径 ---
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path("data")
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "multimodal_rag"

    # --- LLM (OpenAI 兼容) ---
    llm_api_key: Optional[str] = None
    llm_api_base: str = "https://api.openai.com/v1"
    llm_model_name: str = "Qwen/Qwen3.6-27B-FP8"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.3

    # --- Embedding (OpenAI 兼容) ---
    embedding_api_key: Optional[str] = None
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_model_name: str = "text-embedding-v4"
    embedding_dimensions: int = 1024

    # --- 预处理参数 ---
    # 文本
    text_chunk_size: int = 512
    text_chunk_overlap: int = 64

    # 图片
    image_target_size: int = 256        # 处理后图片的边长 (px), 居中裁剪为正方形
    image_quality: int = 85             # JPEG 编码质量 (1-100)

    # 音频
    audio_max_duration_sec: float = 25.0   # 单个音频片段最大时长 (秒)
    audio_overlap_sec: float = 1.0         # 片段间重叠时长 (秒)

    # 视频 (单视频单向量)
    video_target_size: int = 256          # 处理后视频分辨率短边 (px), 居中填充正方形
    video_max_frames: int = 32            # 处理后视频最大帧数
    video_qa_frames: int = 4              # Q&A 时从视频提取的帧数 (送入 LLM 的图片数量)

    # --- 检索参数 ---
    retrieval_top_k: int = 10
    cross_modal_top_k: int = 5


# 全局单例
settings = Settings()
