"""
AudioProcessor — 音频预处理。

加载音频文件, 切分为适配模型限制的片段 (≤25s)。
"""

import io
import logging
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac", ".wma",
}


class AudioProcessor:
    """音频预处理器。

    将音频文件加载并切分为固定时长的片段,
    适配 jina-embeddings-v5-omni-small 的音频输入限制 (~30s)。

    支持两种后端:
      - librosa (主要, 功能更丰富)
      - ffmpeg (回退, 格式支持更广)

    Usage:
        proc = AudioProcessor(max_duration_sec=25.0)
        chunks = proc.process("recording.mp3")
        for c in chunks:
            print(c["metadata"]["start_sec"], c["metadata"]["end_sec"])
    """

    def __init__(
        self,
        max_duration_sec: float = 25.0,
        overlap_sec: float = 1.0,
        sample_rate: int = 16000,
    ):
        self.max_duration_sec = max_duration_sec
        self.overlap_sec = overlap_sec
        self.sample_rate = sample_rate

    def process(self, audio_path: str) -> List[dict]:
        """处理音频文件, 返回片段列表。

        Args:
            audio_path: 音频文件路径。

        Returns:
            片段字典列表, 每个包含:
                - chunk_id: 唯一 ID
                - audio_bytes: 音频片段字节 (WAV 格式)
                - base64: base64 编码
                - text: 占位描述文本 (可后续扩展为 Whisper 转录)
                - metadata: 元数据
                - modality: "audio"
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(
                f"不支持的音频格式: {suffix}。支持的格式: {SUPPORTED_AUDIO_FORMATS}"
            )

        # 获取音频时长
        duration = self._get_duration(audio_path)

        # 加载音频
        try:
            audio_data, sr = self._load_audio(audio_path)
        except Exception as e:
            logger.warning(f"librosa 加载失败, 回退到 ffmpeg: {e}")
            return self._process_with_ffmpeg(audio_path, duration)

        # 切分
        return self._split_audio(audio_data, sr, duration, audio_path)

    def process_batch(self, audio_paths: List[str]) -> List[dict]:
        """批量处理音频。"""
        results = []
        for p in audio_paths:
            try:
                results.extend(self.process(p))
            except Exception as e:
                logger.error(f"处理音频失败 {p}: {e}")
        return results

    # ============================================================
    # 内部方法
    # ============================================================

    def _get_duration(self, audio_path: str) -> float:
        """获取音频时长 (秒)。"""
        try:
            import librosa

            return librosa.get_duration(path=audio_path)
        except Exception:
            # 回退: ffprobe
            try:
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "quiet", "-show_entries",
                        "format=duration", "-of", "csv=p=0", audio_path,
                    ],
                    capture_output=True, text=True, timeout=30,
                )
                return float(result.stdout.strip())
            except Exception:
                logger.warning("无法获取音频时长, 假设 60s")
                return 60.0

    def _load_audio(self, audio_path: str):
        """使用 librosa 加载音频。"""
        import librosa

        audio, sr = librosa.load(
            audio_path,
            sr=self.sample_rate,
            mono=True,
        )
        return audio, sr

    def _split_audio(
        self,
        audio_data: np.ndarray,
        sr: int,
        total_duration: float,
        source_path: str,
    ) -> List[dict]:
        """按固定时长切分音频。"""
        import soundfile as sf

        max_samples = int(self.max_duration_sec * sr)
        overlap_samples = int(self.overlap_sec * sr)
        step = max_samples - overlap_samples

        chunks = []
        start_sample = 0
        path = Path(source_path)

        while start_sample < len(audio_data):
            end_sample = min(start_sample + max_samples, len(audio_data))
            segment = audio_data[start_sample:end_sample]

            start_sec = start_sample / sr
            end_sec = end_sample / sr

            # 编码为 WAV 字节
            buf = io.BytesIO()
            sf.write(buf, segment, sr, format="WAV")
            audio_bytes = buf.getvalue()

            import base64
            chunk_id = str(uuid.uuid4())

            chunks.append({
                "chunk_id": chunk_id,
                "audio_bytes": audio_bytes,
                "base64": base64.b64encode(audio_bytes).decode("utf-8"),
                "text": f"[音频片段: {path.name} @ {start_sec:.1f}s - {end_sec:.1f}s]",
                "metadata": {
                    "chunk_id": chunk_id,
                    "source_file": str(path.resolve()),
                    "source_file_name": path.name,
                    "start_sec": round(start_sec, 2),
                    "end_sec": round(end_sec, 2),
                    "duration_sec": round(end_sec - start_sec, 2),
                    "sample_rate": sr,
                    "total_duration_sec": round(total_duration, 2),
                    "modality": "audio",
                    "content_type": "audio_segment",
                },
                "modality": "audio",
            })

            # 下一段
            start_sample += step
            if start_sample >= len(audio_data):
                break

        return chunks

    def _process_with_ffmpeg(
        self, audio_path: str, total_duration: float
    ) -> List[dict]:
        """使用 ffmpeg 切分音频 (回退方案)。"""
        import base64

        path = Path(audio_path)
        chunks = []
        segment_idx = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            output_pattern = f"{tmpdir}/segment_%03d.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-f", "segment",
                    "-segment_time", str(self.max_duration_sec),
                    "-ac", "1",              # 单声道
                    "-ar", "16000",          # 16kHz 采样率 (AX650 要求)
                    "-c:a", "pcm_s16le",     # PCM WAV 编码
                    output_pattern,
                ],
                capture_output=True,
                timeout=300,
            )

            tmp_path = Path(tmpdir)
            for seg_path in sorted(tmp_path.glob("segment_*.wav")):
                seg_duration = self._get_duration(str(seg_path))
                with open(seg_path, "rb") as f:
                    seg_bytes = f.read()

                chunk_id = str(uuid.uuid4())
                start_sec = segment_idx * self.max_duration_sec
                end_sec = min(start_sec + seg_duration, total_duration)

                chunks.append({
                    "chunk_id": chunk_id,
                    "audio_bytes": seg_bytes,
                    "base64": base64.b64encode(seg_bytes).decode("utf-8"),
                    "text": f"[音频片段: {path.name} @ {start_sec:.1f}s - {end_sec:.1f}s]",
                    "metadata": {
                        "chunk_id": chunk_id,
                        "source_file": str(path.resolve()),
                        "source_file_name": path.name,
                        "start_sec": round(start_sec, 2),
                        "end_sec": round(end_sec, 2),
                        "duration_sec": round(seg_duration, 2),
                        "total_duration_sec": round(total_duration, 2),
                        "modality": "audio",
                        "content_type": "audio_segment",
                    },
                    "modality": "audio",
                })
                segment_idx += 1

        return chunks
