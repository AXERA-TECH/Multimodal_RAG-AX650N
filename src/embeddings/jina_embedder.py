"""
JinaEmbedder — 统一多模态嵌入接口。

支持 text / image / audio / video 四种模态,
通过 Jina AI API 或本地 sentence-transformers 模型生成嵌入向量。
"""

import base64
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from PIL import Image

from config.settings import settings

logger = logging.getLogger(__name__)


# ============================================================
# 枚举 & 数据结构
# ============================================================


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class EmbeddingTask(str, Enum):
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_PASSAGE = "retrieval.passage"
    TEXT_MATCHING = "text-matching"
    CLASSIFICATION = "classification"
    SEPARATION = "separation"


@dataclass
class EmbeddingInput:
    """统一的多模态嵌入输入。"""

    modality: Modality
    content: Union[str, bytes]  # text 字符串, 或 image/audio/video 的 bytes
    source_path: Optional[str] = None

    def to_api_dict(self) -> Dict[str, Any]:
        """转为 Jina API 要求的输入格式。"""
        if self.modality == Modality.TEXT:
            if isinstance(self.content, bytes):
                text = self.content.decode("utf-8")
            else:
                text = self.content
            return {"text": text}
        elif self.modality == Modality.IMAGE:
            if isinstance(self.content, str):
                # 已经是 base64 字符串
                return {"image": self.content}
            else:
                b64 = base64.b64encode(self.content).decode("utf-8")
                return {"image": b64}
        elif self.modality == Modality.AUDIO:
            if isinstance(self.content, str):
                return {"audio": self.content}
            else:
                b64 = base64.b64encode(self.content).decode("utf-8")
                return {"audio": b64}
        elif self.modality == Modality.VIDEO:
            if isinstance(self.content, str):
                return {"video": self.content}
            else:
                b64 = base64.b64encode(self.content).decode("utf-8")
                return {"video": b64}
        else:
            raise ValueError(f"Unknown modality: {self.modality}")


@dataclass
class EmbeddingResult:
    """嵌入结果。"""

    embeddings: np.ndarray  # shape (n_inputs, dimensions)
    dimensions: int
    model_name: str
    latency_ms: float
    input_count: int


# ============================================================
# JinaEmbedder
# ============================================================


class JinaEmbedder:
    """多模态嵌入模型封装。

    两种后端:
      - 'openai': OpenAI 兼容 Embeddings API (/v1/embeddings), 默认
      - 'local':  本地 sentence-transformers 模型

    Usage:
        # OpenAI 兼容 API
        embedder = JinaEmbedder(
            backend="openai",
            api_key="sk-xxx",
            api_base_url="https://your-proxy.com/v1",
            model_name="text-embedding-v4",
        )

        result = embedder.embed([
            EmbeddingInput(Modality.TEXT, "Hello world"),
            EmbeddingInput(Modality.IMAGE, image_bytes),
        ])
    """

    def __init__(
        self,
        backend: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        task: EmbeddingTask = EmbeddingTask.RETRIEVAL_PASSAGE,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        self.backend = backend
        self.dimensions = dimensions or settings.embedding_dimensions
        self.task = task
        self.max_retries = max_retries
        self.timeout = timeout

        self.model_name = model_name or settings.embedding_model_name
        self.api_key = api_key or settings.embedding_api_key
        self.api_base_url = api_base_url or settings.embedding_api_base

        self._local_model = None

        if backend == "openai" and not self.api_key:
            raise ValueError(
                "OpenAI embedding backend 需要 EMBEDDING_API_KEY。"
                "请在 .env 文件中设置, 或传入 api_key 参数。"
            )

        if self.backend == "local":
            self._init_local_model()

    def _init_local_model(self):
        """加载本地 sentence-transformers 模型。"""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"加载本地模型: {self.model_name}")
            self._local_model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                model_kwargs={"default_task": "retrieval"},
            )
            logger.info("本地模型加载完成")
        except ImportError:
            raise ImportError(
                "本地后端需要 sentence-transformers。请安装: pip install sentence-transformers"
            )

    # ============================================================
    # 公共 API
    # ============================================================

    def embed(
        self,
        inputs: List[EmbeddingInput],
        task: Optional[EmbeddingTask] = None,
    ) -> EmbeddingResult:
        """批量多模态嵌入。

        Args:
            inputs: 嵌入输入列表, 可混合不同模态。
            task: 嵌入任务类型 (影响 embedding 偏向)。

        Returns:
            EmbeddingResult 包含嵌入矩阵和元数据。
        """
        if not inputs:
            raise ValueError("inputs 不能为空")

        task = task or self.task
        t0 = time.perf_counter()

        if self.backend == "openai":
            embeddings = self._embed_via_openai(inputs, task)
        else:
            embeddings = self._embed_local(inputs, task)

        latency = (time.perf_counter() - t0) * 1000

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=embeddings.shape[1],
            model_name=self.model_name,
            latency_ms=latency,
            input_count=len(inputs),
        )

    def embed_query(self, text: str) -> np.ndarray:
        """便捷方法: 嵌入查询文本。

        使用 retrieval.query 任务标记, 对查询意图做优化。
        返回 shape (dimensions,) 的向量。
        """
        inp = EmbeddingInput(Modality.TEXT, text)
        result = self.embed([inp], task=EmbeddingTask.RETRIEVAL_QUERY)
        return result.embeddings[0]

    def embed_document(self, inp: EmbeddingInput) -> np.ndarray:
        """便捷方法: 嵌入文档内容。

        使用 retrieval.passage 任务标记。
        返回 shape (dimensions,) 的向量。
        """
        result = self.embed([inp], task=EmbeddingTask.RETRIEVAL_PASSAGE)
        return result.embeddings[0]

    def embed_batch(
        self,
        inputs: List[EmbeddingInput],
        batch_size: int = 16,
        task: Optional[EmbeddingTask] = None,
    ) -> EmbeddingResult:
        """分批嵌入, 避免单次请求过大。

        Args:
            inputs: 嵌入输入列表。
            batch_size: 每批处理的输入数量。
            task: 嵌入任务类型。
        """
        task = task or self.task
        all_embeddings = []
        t0 = time.perf_counter()

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            logger.debug(f"嵌入批次 {i // batch_size + 1}, 大小 {len(batch)}")
            result = self.embed(batch, task=task)
            all_embeddings.append(result.embeddings)

        embeddings = np.concatenate(all_embeddings, axis=0)
        latency = (time.perf_counter() - t0) * 1000

        return EmbeddingResult(
            embeddings=embeddings,
            dimensions=embeddings.shape[1],
            model_name=self.model_name,
            latency_ms=latency,
            input_count=len(inputs),
        )

    # ============================================================
    # 便捷工厂方法
    # ============================================================

    def embed_text(self, text: str) -> np.ndarray:
        """嵌入单段文本 (文档模式)。"""
        return self.embed_document(EmbeddingInput(Modality.TEXT, text))

    def embed_image_path(self, image_path: str) -> np.ndarray:
        """嵌入图片文件。"""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return self.embed_document(
            EmbeddingInput(Modality.IMAGE, image_bytes, source_path=image_path)
        )

    def embed_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """嵌入图片字节。"""
        return self.embed_document(EmbeddingInput(Modality.IMAGE, image_bytes))

    def embed_audio_path(self, audio_path: str) -> np.ndarray:
        """嵌入音频文件。"""
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        return self.embed_document(
            EmbeddingInput(Modality.AUDIO, audio_bytes, source_path=audio_path)
        )

    def embed_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """嵌入音频字节。"""
        return self.embed_document(EmbeddingInput(Modality.AUDIO, audio_bytes))

    # ============================================================
    # 后端实现
    # ============================================================

    def _embed_via_openai(self, inputs: List[EmbeddingInput], task: EmbeddingTask) -> np.ndarray:
        """通过 axllm / OpenAI 兼容 Embeddings API 生成嵌入。

        TEXT / IMAGE / AUDIO: 内嵌 base64 data URI
        VIDEO: 直接传本地文件绝对路径 (str 类型 content = 文件路径)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        prompt_name = "query" if task == EmbeddingTask.RETRIEVAL_QUERY else "document"

        # 统一使用 messages 格式, 媒体传本地文件绝对路径 (参照 axllm demo)
        content_blocks = []
        for inp in inputs:
            if inp.modality == Modality.TEXT:
                text = inp.content if isinstance(inp.content, str) else inp.content.decode("utf-8")
                content_blocks.append({"type": "text", "text": text})
            elif inp.modality in (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO):
                fpath = inp.content if isinstance(inp.content, str) else inp.content.decode("utf-8")
                type_map = {
                    Modality.IMAGE: "image_url",
                    Modality.AUDIO: "audio_url",
                    Modality.VIDEO: "video_url",
                }
                content_blocks.append({
                    "type": type_map[inp.modality],
                    type_map[inp.modality]: {"url": str(Path(fpath).resolve())},
                })

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt_name": prompt_name,
            "encoding_format": "float",
            "messages": [
                {"role": "user", "content": content_blocks}
            ],
        }

        if self.dimensions:
            payload["dimensions"] = self.dimensions

        url = f"{self.api_base_url.rstrip('/')}/embeddings"

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    url, json=payload, headers=headers, timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                try:
                    # 兼容两种顶层格式:
                    # 1. {"data": [...], "model": ...}  — OpenAI 标准
                    # 2. [{"embedding": [...], "index": 0}, ...]  — AX650 文件路径模式
                    if isinstance(data, list):
                        raw = data
                    elif isinstance(data, dict):
                        raw = data["data"]
                    else:
                        raise RuntimeError(f"未知响应类型: {type(data).__name__}")

                    if isinstance(raw, list):
                        if not raw:
                            raise RuntimeError("空响应: data 列表为空")
                        first = raw[0]
                        if isinstance(first, dict):
                            embeddings = [item["embedding"] for item in raw]
                        elif isinstance(first, (list, tuple)):
                            embeddings = [list(item) for item in raw]
                        elif isinstance(first, (int, float)):
                            embeddings = [list(raw)]
                        else:
                            raise RuntimeError(f"未知 data 元素类型: {type(first)}")
                    elif isinstance(raw, dict):
                        if "embedding" in raw:
                            embeddings = [raw["embedding"]]
                        else:
                            embeddings = [
                                raw[k]["embedding"] if isinstance(raw[k], dict) else list(raw[k])
                                for k in sorted(raw.keys(),
                                key=lambda x: int(x) if str(x).isdigit() else 0)
                            ]
                    else:
                        raise RuntimeError(f"无法解析 data 类型: {type(raw)}")
                    return np.array(embeddings, dtype=np.float32)
                except Exception as parse_err:
                    logger.error(f"响应解析失败. HTTP {resp.status_code}, "
                               f"顶层类型: {type(data).__name__}, "
                               f"内容预览: {str(data)[:200]}")
                    raise

            except requests.exceptions.RequestException as e:
                logger.warning(f"API 请求失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Embedding API 调用失败, 已重试 {self.max_retries} 次: {e}")

        raise RuntimeError("Unexpected error in _embed_via_openai")
    def _embed_local(
        self, inputs: List[EmbeddingInput], task: EmbeddingTask
    ) -> np.ndarray:
        """通过本地 sentence-transformers 模型生成嵌入。"""
        if self._local_model is None:
            self._init_local_model()

        # sentence-transformers 的多模态处理方式:
        # 对于纯文本, 直接传字符串列表
        # 对于图片, 传 PIL Image 列表
        # 对于混合模态, 需要特殊处理

        has_non_text = any(inp.modality != Modality.TEXT for inp in inputs)

        if not has_non_text:
            texts = [
                inp.content if isinstance(inp.content, str) else inp.content.decode("utf-8")
                for inp in inputs
            ]
            embeddings = self._local_model.encode(
                texts,
                prompt_name=task.value,
                normalize_embeddings=True,
            )
            return np.array(embeddings, dtype=np.float32)

        # 混合模态: 逐个处理
        embeddings_list = []
        for inp in inputs:
            if inp.modality == Modality.TEXT:
                text = inp.content if isinstance(inp.content, str) else inp.content.decode("utf-8")
                emb = self._local_model.encode(
                    text,
                    prompt_name=task.value,
                    normalize_embeddings=True,
                )
            elif inp.modality == Modality.IMAGE:
                image = Image.open(BytesIO(inp.content))
                emb = self._local_model.encode(
                    image,
                    prompt_name=task.value,
                    normalize_embeddings=True,
                )
            else:
                # audio/video: 尝试作为原始字节传入
                # 本地模型可能不支持, 回退到使用 bytes 的 hash 作为简陋替代
                logger.warning(
                    f"本地模型可能不完全支持 {inp.modality} 模态, 使用尽力而为模式"
                )
                try:
                    emb = self._local_model.encode(
                        inp.content,
                        prompt_name=task.value,
                        normalize_embeddings=True,
                    )
                except Exception:
                    # 最终回退: 使用全零向量 + 警告
                    logger.error(f"无法嵌入 {inp.modality} 输入, 使用零向量")
                    emb = np.zeros(self.dimensions, dtype=np.float32)

            embeddings_list.append(emb)

        return np.array(embeddings_list, dtype=np.float32)


# ============================================================
# 辅助函数
# ============================================================


def image_to_base64(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    target_size: int = 256,
    quality: int = 85,
) -> str:
    """将图片转为 base64 字符串 (JPEG 格式, 居中裁剪到 target_size)。

    Args:
        image_path: 图片文件路径。
        image_bytes: 图片字节数据。与 image_path 二选一。
        target_size: 目标尺寸 (正方形)。
        quality: JPEG 质量 (1-100)。

    Returns:
        base64 编码的图片字符串。
    """
    if image_path:
        img = Image.open(image_path)
    elif image_bytes:
        img = Image.open(BytesIO(image_bytes))
    else:
        raise ValueError("image_path 或 image_bytes 必须提供一个")

    # 转换为 RGB (处理 RGBA / P 模式)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # 居中裁剪到 target_size
    w, h = img.size
    if w < h:
        new_w, new_h = target_size, int(h * (target_size / w))
    else:
        new_h, new_w = target_size, int(w * (target_size / h))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))

    # 编码为 JPEG
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
