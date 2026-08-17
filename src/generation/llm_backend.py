"""
LLMBackend — OpenAI 兼容的 LLM 后端接口。

支持所有兼容 OpenAI Chat Completions API 的服务 (vLLM, Ollama, DashScope 等)。
"""

import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class LLMBackend:
    """OpenAI 兼容 LLM 后端接口。

    支持所有兼容 OpenAI Chat Completions API 的服务。

    Usage:
        llm = LLMBackend()
        answer = llm.generate(messages=[{"role": "user", "content": "Hello"}])

        # 带多模态上下文
        answer = llm.generate_with_context(
            query="图中有什么?",
            retrieved_contexts=contexts,
            prompt_template=template,
        )
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: int = 120,
    ):
        self.model_name = model_name or settings.llm_model_name
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.temperature = temperature or settings.llm_temperature
        self.timeout = timeout

        self.api_key = api_key or settings.llm_api_key
        self.base_url = base_url or settings.llm_api_base
        self._client = None

    def _get_client(self):
        """延迟初始化 OpenAI 客户端。"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("需要 openai SDK。请安装: pip install openai")
        return self._client

    # ============================================================
    # 公共 API
    # ============================================================

    def generate(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """生成回复。

        Args:
            messages: 消息列表。
            system_prompt: 系统提示词。
            max_tokens: 最大生成 token 数。
            temperature: 生成温度。

        Returns:
            生成的文本回复。
        """
        client = self._get_client()

        api_messages: List[Dict[str, Any]] = []

        # 合并所有 system 消息到一条 (部分 LLM 要求 system 必须在最前面且只能有一条)
        system_contents = []
        if system_prompt:
            system_contents.append(system_prompt)

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_contents.append(content)
                continue

            if isinstance(content, list):
                api_messages.append({"role": role, "content": content})
            else:
                api_messages.append({"role": role, "content": str(content)})

        if system_contents:
            api_messages.insert(0, {"role": "system", "content": "\n".join(system_contents)})

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"LLM API 调用失败: {e}")
            raise

    def generate_with_context(
        self,
        query: str,
        retrieved_contexts: Dict[str, Any],
        prompt_template=None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """基于检索上下文生成回答 (高层 API)。

        Args:
            query: 用户查询。
            retrieved_contexts: 来自 Retriever.retrieve_with_context() 的结果。
            prompt_template: PromptTemplate 实例。
            system_prompt: 自定义系统提示词。

        Returns:
            包含 answer, model_name, latency_ms 的字典。
        """
        if prompt_template is None:
            from src.generation.prompt_templates import PromptTemplate
            prompt_template = PromptTemplate()

        messages = prompt_template.build_messages(
            query=query,
            retrieved_contexts=retrieved_contexts,
        )

        sp = system_prompt or prompt_template.build_system_prompt()

        t0 = time.perf_counter()
        answer = self.generate(messages, system_prompt=sp)
        latency = (time.perf_counter() - t0) * 1000

        return {
            "answer": answer,
            "model_name": self.model_name,
            "latency_ms": latency,
        }

    # ============================================================
    # 便捷方法
    # ============================================================

    def simple_ask(self, question: str, context: str = "") -> str:
        """简单问答 (不涉及多模态上下文)。"""
        user_content = question
        if context:
            user_content = f"上下文:\n{context}\n\n问题: {question}"

        return self.generate(
            messages=[{"role": "user", "content": user_content}],
            system_prompt="你是一个有帮助的AI助手。直接输出答案，不要展示思考过程。用中文回答。",
        )

    def ask_with_image(
        self,
        question: str,
        image_base64: str,
        image_media_type: str = "image/jpeg",
    ) -> str:
        """带图片的问答。

        Args:
            question: 问题文本。
            image_base64: 图片的 base64 编码。
            image_media_type: 图片 MIME 类型。
        """
        content = [
            {"type": "image_url", "image_url": {
                "url": f"data:{image_media_type};base64,{image_base64}"
            }},
            {"type": "text", "text": question},
        ]
        return self.generate(
            messages=[{"role": "user", "content": content}],
            system_prompt="你是一个有帮助的AI助手。请描述和分析图片内容。用中文回答。",
        )
