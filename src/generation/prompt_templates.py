"""
PromptTemplate — 多模态上下文组装。

将检索到的多模态结果组装为结构化的 LLM 提示词。
使用 OpenAI Chat Completions 兼容格式。
"""

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _validate_base64_image(b64: str) -> bool:
    """验证 base64 字符串是否为有效图片。"""
    try:
        from PIL import Image
        data = base64.b64decode(b64)
        if len(data) < 100:  # 太小不可能是有效图片
            return False
        img = Image.open(BytesIO(data))
        img.verify()
        return True
    except Exception:
        return False


class PromptTemplate:
    """多模态 RAG 提示词模板。

    将检索到的文本/图片/音频/视频上下文拼装成 LLM 可理解的格式。
    使用 OpenAI Chat Completions 兼容的消息格式。

    Usage:
        template = PromptTemplate()
        messages = template.build_messages(
            query="图中有什么?",
            retrieved_contexts={...},
        )
    """

    SYSTEM_PROMPT = """你是一个多模态智能问答助手。你可以基于提供的上下文回答用户的问题,
上下文可能包括文本段落、图片、音频描述和视频帧描述。

回答规则:
1. 仔细分析所有提供的多模态上下文 (文本、图片、音频、视频帧)
2. 基于上下文中的具体信息回答问题, 不要编造信息
3. 引用来源时使用格式: [来源: 文件名]
4. 如果上下文中包含图片或视频帧, 请描述你看到的视觉内容
5. 如果上下文中包含音频信息, 请参考音频内容
6. 如果上下文不足以回答问题, 请明确说明
7. 用中文回答, 保持清晰、准确、有条理
8. 直接输出最终答案，不要展示思考过程、推理步骤或中间分析"""

    USER_TEMPLATE = """## 用户问题

{query}

## 检索到的上下文

{context_sections}

请基于以上上下文回答用户的问题。"""

    def build_messages(
        self,
        query: str,
        retrieved_contexts: Dict[str, Any],
        max_images: int = 10,
    ) -> List[Dict[str, Any]]:
        """构建 LLM 消息列表。

        Args:
            query: 用户查询文本。
            retrieved_contexts: 来自 MultiModalRetriever.retrieve_with_context() 的结果。
            max_images: 最多包含的图片/视频帧数量 (控制 token 消耗)。

        Returns:
            OpenAI Chat Completions 兼容的消息列表。
        """
        return self._build_openai_messages(query, retrieved_contexts, max_images)

    def build_system_prompt(self) -> str:
        """获取系统提示词。"""
        return self.SYSTEM_PROMPT

    def build_context_text(
        self,
        retrieved_contexts: Dict[str, Any],
        include_images_as_text: bool = True,
    ) -> str:
        """构建纯文本上下文 (用于不支持多模态的 LLM)。"""
        sections = []

        text_contexts = retrieved_contexts.get("text_contexts", [])
        if text_contexts:
            sections.append("### 文本内容\n")
            for i, ctx in enumerate(text_contexts, 1):
                sections.append(
                    f"**[文本 {i}]** (来源: {ctx['source']}, 相关度: {ctx['score']:.2f})\n"
                    f"{ctx['text']}\n"
                )

        image_contexts = retrieved_contexts.get("image_contexts", [])
        if image_contexts and include_images_as_text:
            sections.append("### 图片内容\n")
            for i, ctx in enumerate(image_contexts, 1):
                sections.append(
                    f"**[图片 {i}]** (来源: {ctx['source']}, 相关度: {ctx['score']:.2f})\n"
                    f"[图片描述: {ctx.get('preview', '未提供描述')}]\n"
                )

        audio_contexts = retrieved_contexts.get("audio_contexts", [])
        if audio_contexts:
            sections.append("### 音频内容\n")
            for i, ctx in enumerate(audio_contexts, 1):
                start = ctx.get("start_sec", 0)
                end = ctx.get("end_sec", 0)
                sections.append(
                    f"**[音频 {i}]** (来源: {ctx['source']}, "
                    f"时间: {start:.1f}s-{end:.1f}s, 相关度: {ctx['score']:.2f})\n"
                    f"[音频描述: {ctx.get('preview', '未提供转录')}]\n"
                )

        video_contexts = retrieved_contexts.get("video_contexts", [])
        if video_contexts:
            sections.append("### 视频帧内容\n")
            for i, ctx in enumerate(video_contexts, 1):
                ts = ctx.get("timestamp_sec", 0)
                sections.append(
                    f"**[视频帧 {i}]** (来源: {ctx['source']}, "
                    f"时间戳: {ts:.1f}s, 相关度: {ctx['score']:.2f})\n"
                )

        return "\n".join(sections) if sections else "未找到相关内容。"

    def _build_openai_messages(
        self,
        query: str,
        contexts: Dict[str, Any],
        max_images: int,
    ) -> List[Dict[str, Any]]:
        """构建 OpenAI Chat Completions API 格式的消息。

        GPT-4V/GPT-4o 支持 image_url content 类型。
        """
        user_content: List[Dict[str, Any]] = []

        # 文本上下文
        context_text = self.build_context_text(contexts, include_images_as_text=True)
        user_content.append({
            "type": "text",
            "text": self.USER_TEMPLATE.format(
                query=query,
                context_sections=context_text,
            ),
        })

        # 图片 (GPT-4V 支持 image_url 和 base64)
        image_count = 0

        # 处理图片上下文
        for ctx in contexts.get("image_contexts", []):
            if image_count >= max_images:
                break
            b64 = ctx.get("base64")
            if b64 and _validate_base64_image(b64):
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "auto",
                    },
                })
                image_count += 1

        # 处理视频上下文 — 每帧作为独立图片发送
        for ctx in contexts.get("video_contexts", []):
            if image_count >= max_images:
                break
            frames = ctx.get("frames_base64", [])
            for b64 in frames:
                if image_count >= max_images:
                    break
                if b64 and _validate_base64_image(b64):
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "auto",
                        },
                    })
                    image_count += 1

        return [
            {"role": "user", "content": user_content},
        ]


# ============================================================
# 预置模板
# ============================================================


class CompactPromptTemplate(PromptTemplate):
    """紧凑版提示词模板 — 适用于 token 预算有限时。"""

    SYSTEM_PROMPT = """你是多模态问答助手。严格基于提供的上下文回答问题, 不要编造信息。
引用来源用 [来源: 文件名] 格式。直接输出最终答案，不要展示思考过程。用中文回答。"""

    USER_TEMPLATE = """问题: {query}

上下文:
{context_sections}

回答:"""


class DetailedPromptTemplate(PromptTemplate):
    """详细版提示词模板 — 适用于需要深度分析时。"""

    SYSTEM_PROMPT = """你是一位精通多模态内容分析的专家助手。你需要:

1. 仔细审视所有提供的文本、图片、音频和视频帧
2. 综合不同模态的信息, 发现它们之间的关联
3. 对视觉内容 (图片/视频帧) 给出详细的描述
4. 引用具体的来源和时间戳
5. 如果信息之间存在矛盾, 请指出
6. 区分你确信的事实和你推断的内容

直接输出最终分析结果，不要展示思考过程。用中文回答。"""

    USER_TEMPLATE = """## 分析任务

{query}

## 可用资料

{context_sections}

## 请提供

1. **直接回答**: 针对问题给出明确答案
2. **证据来源**: 列出支撑答案的每条证据及其来源
3. **多模态分析**: 如果有多种模态的信息, 分析它们如何相互补充
4. **置信度评估**: 对你的回答的确定程度做出评估"""
