"""
VideoProcessor — 视频预处理。

将视频缩放到指定分辨率、控制帧率，输出单个处理后视频文件用于嵌入。
"""

import logging
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_FORMATS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv",
}


class VideoProcessor:
    """视频预处理器 — 单视频单向量。

    将视频缩放到指定分辨率、控制帧率，输出一个 MP4 文件，
    整个视频作为一个 VIDEO 向量嵌入。

    Usage:
        proc = VideoProcessor(target_size=256, max_frames=32)
        result = proc.process("video.mp4")
        # result["video_bytes"] → 处理后的视频数据
    """

    def __init__(
        self,
        target_size: int = 256,
        max_frames: int = 32,
    ):
        self.target_size = target_size   # 输出视频分辨率 (短边)
        self.max_frames = max_frames     # 输出视频最大帧数

    def process(self, video_path: str) -> dict:
        """处理视频文件。

        缩放 + 帧率控制 → 单个 MP4 文件。

        Returns:
            {chunk_id, video_bytes, metadata}
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_VIDEO_FORMATS:
            raise ValueError(f"不支持的视频格式: {suffix}")

        # 获取元数据
        metadata = self._get_video_metadata(video_path)
        duration = metadata.get("duration", 0)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = tmp.name

        try:
            # 计算目标帧率: 总帧数 / 时长
            target_fps = min(self.max_frames / duration, 30) if duration > 0 else 10

            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", (
                    f"scale='min({self.target_size},iw)':min'({self.target_size},ih)':force_original_aspect_ratio=1,"
                    f"fps={target_fps},"
                    f"pad={self.target_size}:{self.target_size}:(ow-iw)/2:(oh-ih)/2:black"
                ),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                "-an",                    # 去掉音频
                "-movflags", "+faststart",
                out_path,
            ]
            subprocess.run(cmd, capture_output=True, timeout=300)

            out_file = Path(out_path)
            if out_file.exists() and out_file.stat().st_size > 0:
                video_bytes = out_file.read_bytes()
                chunk_id = str(uuid.uuid4())
                return {
                    "chunk_id": chunk_id,
                    "video_bytes": video_bytes,
                    "text": f"[视频: {path.name}, {duration:.0f}s, {target_fps:.1f}fps]",
                    "metadata": {
                        "source_file": str(path.resolve()),
                        "source_file_name": path.name,
                        "duration_sec": round(duration, 2),
                        "target_size": self.target_size,
                        "target_fps": round(target_fps, 1),
                        "max_frames": self.max_frames,
                        "file_size_mb": round(out_file.stat().st_size / (1024 * 1024), 2),
                    },
                }
            else:
                raise RuntimeError("ffmpeg 输出文件为空")

        finally:
            Path(out_path).unlink(missing_ok=True)

    def _get_video_metadata(self, video_path: str) -> dict:
        """获取视频元数据 (ffprobe)。"""
        try:
            import json
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", video_path],
                capture_output=True, text=True, timeout=30,
            )
            data = json.loads(result.stdout)
            fmt = data.get("format", {})
            video_stream = None
            for s in data.get("streams", []):
                if s.get("codec_type") == "video":
                    video_stream = s
                    break
            duration = float(fmt.get("duration", 0))
            return {
                "duration": round(duration, 2),
                "fps": self._parse_fps(video_stream),
                "resolution": (
                    f"{video_stream.get('width','?')}x{video_stream.get('height','?')}"
                    if video_stream else "unknown"
                ),
                "codec": video_stream.get("codec_name", "unknown") if video_stream else "unknown",
            }
        except Exception:
            return {"duration": 0, "fps": 0, "resolution": "unknown", "codec": "unknown"}

    @staticmethod
    def _parse_fps(video_stream: Optional[dict]) -> float:
        if not video_stream:
            return 0
        r = video_stream.get("r_frame_rate", "0/1")
        if "/" in r:
            parts = r.split("/")
            return round(float(parts[0]) / float(parts[1]), 2) if float(parts[1]) > 0 else 0
        try:
            return float(r)
        except (ValueError, TypeError):
            return 0
