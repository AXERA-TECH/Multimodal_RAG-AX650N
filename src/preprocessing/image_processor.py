"""
ImageProcessor — 图片预处理。

加载、验证、缩放图片, 并支持转为 base64 编码。
"""

import base64
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from config.settings import settings

logger = logging.getLogger(__name__)

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif",
}


class ImageProcessor:
    """图片预处理器。

    功能:
      - 加载并验证图片
      - 等比缩放 (保持宽高比)
      - 转为 JPEG 字节或 base64 字符串
      - 生成用于媒体展示的缩略图

    Usage:
        proc = ImageProcessor(target_size=256)
        result = proc.process("photo.jpg")  # -> {"image_bytes": ..., "base64": ..., "metadata": ...}
    """

    def __init__(
        self,
        target_size: int = 256,
        quality: int = 85,
        thumbnail_dim: int = 256,
    ):
        self.target_size = target_size
        self.quality = quality
        self.thumbnail_dim = thumbnail_dim

    def process(self, image_path: str) -> dict:
        """处理单张图片。

        Args:
            image_path: 图片文件路径。

        Returns:
            包含以下字段的字典:
                - chunk_id: 唯一 ID
                - image_bytes: JPEG 编码的图片字节
                - base64: base64 字符串
                - metadata: 元数据 (尺寸、格式、来源等)
                - thumbnail_base64: 缩略图 base64
                - modality: "image"

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 文件格式不支持或内容无效。
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"不支持的图片格式: {suffix}。支持的格式: {SUPPORTED_IMAGE_FORMATS}"
            )

        # 预检文件头魔数，防止非图片文件以图片扩展名上传
        self._validate_image_bytes(path)

        try:
            img = Image.open(image_path)
            img.load()  # 强制加载像素数据，触发真实解码
        except Exception as e:
            raise ValueError(
                f"无法解析图片文件，文件可能已损坏或格式与扩展名不匹配: {path.name}"
            ) from e

        original_size = img.size

        # 转 RGB
        if img.mode in ("RGBA", "P", "LA", "PA"):
            img = img.convert("RGB")

        # 缩放到目标尺寸 (居中裁剪)
        img_resized = self._resize_to_target(img, self.target_size)

        # 缩略图
        thumbnail = self._resize_to_target(img.copy(), self.thumbnail_dim)

        # 编码为 JPEG
        main_bytes = self._encode_jpeg(img_resized)
        thumbnail_bytes = self._encode_jpeg(thumbnail, quality=75)

        chunk_id = str(uuid.uuid4())

        return {
            "chunk_id": chunk_id,
            "image_bytes": main_bytes,
            "base64": base64.b64encode(main_bytes).decode("utf-8"),
            "thumbnail_base64": base64.b64encode(thumbnail_bytes).decode("utf-8"),
            "metadata": {
                "chunk_id": chunk_id,
                "source_file": str(path.resolve()),
                "source_file_name": path.name,
                "original_size": original_size,
                "resized_size": (self.target_size, self.target_size),
                "format": suffix.lstrip("."),
                "modality": "image",
                "content_type": "image",
            },
            "modality": "image",
        }

    def process_batch(self, image_paths: List[str]) -> List[dict]:
        """批量处理图片。"""
        results = []
        for p in image_paths:
            try:
                results.append(self.process(p))
            except Exception as e:
                logger.error(f"处理图片失败 {p}: {e}")
        return results

    def _resize_to_target(self, img: Image.Image, size: int) -> Image.Image:
        """居中裁剪并缩放到指定尺寸 (size x size)。"""
        w, h = img.size

        # 等比缩放使短边 = size
        if w < h:
            new_w = size
            new_h = int(h * (size / w))
        else:
            new_h = size
            new_w = int(w * (size / h))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # 居中裁剪
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        return img.crop((left, top, left + size, top + size))

    def _encode_jpeg(self, img: Image.Image, quality: Optional[int] = None) -> bytes:
        """编码为 JPEG 字节。"""
        quality = quality or self.quality
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    @staticmethod
    def _validate_image_bytes(path: Path):
        """通过文件头魔数验证是否为有效图片。"""
        MAGIC_BYTES = {
            b"\xff\xd8\xff": "JPEG",
            b"\x89PNG\r\n\x1a\n": "PNG",
            b"GIF87a": "GIF",
            b"GIF89a": "GIF",
            b"RIFF": "WEBP",
            b"BM": "BMP",
            b"II*\x00": "TIFF",
            b"MM\x00*": "TIFF",
        }

        try:
            with open(path, "rb") as f:
                header = f.read(12)
        except OSError as e:
            raise ValueError(f"无法读取文件: {e}")

        if len(header) < 4:
            raise ValueError(f"文件过小，不是有效图片: {path.name}")

        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return

        for magic, fmt in MAGIC_BYTES.items():
            if header[:len(magic)] == magic:
                return

        raise ValueError(
            f"文件头不匹配任何已知图片格式，文件可能不是有效图片: {path.name}"
        )

    @staticmethod
    def get_image_base64(image_path: str, target_size: int = 256) -> str:
        """快速获取图片 base64 (不生成缩略图)。"""
        proc = ImageProcessor(target_size=target_size)
        result = proc.process(image_path)
        return result["base64"]

    @staticmethod
    def bytes_to_base64(image_bytes: bytes, target_size: int = 256) -> str:
        """将图片 bytes 转为 base64。"""
        proc = ImageProcessor(target_size=target_size)
        img = Image.open(BytesIO(image_bytes))
        if img.mode in ("RGBA", "P", "LA", "PA"):
            img = img.convert("RGB")
        img_resized = proc._resize_to_target(img, target_size)
        jpeg_bytes = proc._encode_jpeg(img_resized)
        return base64.b64encode(jpeg_bytes).decode("utf-8")
