"""
Multimodal RAG — 多模态检索增强生成系统。

基于 jina-embeddings-v5-omni-small 的统一嵌入空间,
支持文本、图片、音频、视频的智能检索与问答。

核心组件:
    - JinaEmbedder: 多模态嵌入
    - VectorStore: ChromaDB 向量存储
    - MultiModalRetriever: 跨模态检索
    - LLMBackend: 多 LLM 生成
    - IngestionPipeline: 数据入库
    - QueryPipeline: 智能查询
"""

from src.embeddings.jina_embedder import (
    EmbeddingInput,
    EmbeddingResult,
    EmbeddingTask,
    JinaEmbedder,
    Modality,
)
from src.storage.schemas import (
    IngestionStats,
    MediaChunk,
    QueryResponse,
    SearchResult,
)
from src.storage.vector_store import VectorStore
from src.retrieval.retriever import MultiModalRetriever
from src.generation.llm_backend import LLMBackend
from src.generation.prompt_templates import PromptTemplate
from src.pipeline.ingestion import IngestionPipeline
from src.pipeline.query import QueryPipeline
