#!/usr/bin/env python3
"""
Multimodal RAG CLI — 命令行交互工具。

用法:
    python -m ui.cli ingest --path ./data/
    python -m ui.cli query "图中有什么?"
    python -m ui.cli cross-modal "ocean waves"
    python -m ui.cli status
    python -m ui.cli clear
"""

import sys
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.pipeline.ingestion import IngestionPipeline
from src.pipeline.query import QueryPipeline
from src.storage.vector_store import VectorStore
from src.embeddings.jina_embedder import Modality
from config.settings import settings

app = typer.Typer(
    name="multimodal-rag",
    help="多模态 RAG 智能问答系统 CLI",
    add_completion=False,
)

console = Console()

# 延迟初始化的组件
_ingestion_pipeline: Optional[IngestionPipeline] = None
_query_pipeline: Optional[QueryPipeline] = None
_vector_store: Optional[VectorStore] = None


def get_ingestion_pipeline() -> IngestionPipeline:
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline()
    return _ingestion_pipeline


def get_query_pipeline() -> QueryPipeline:
    global _query_pipeline
    if _query_pipeline is None:
        _query_pipeline = QueryPipeline()
    return _query_pipeline


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


# ============================================================
# ingest — 入库数据
# ============================================================


@app.command()
def ingest(
    path: str = typer.Argument(..., help="文件或目录路径"),
    recursive: bool = typer.Option(True, help="递归处理子目录"),
    incremental: bool = typer.Option(True, help="跳过已索引的文件"),
):
    """入库文件或目录到向量库。"""
    console.print(f"[bold blue]入库路径:[/] {path}")

    pipeline = get_ingestion_pipeline()
    pipeline.incremental = incremental

    p = Path(path)
    if p.is_file():
        stats = pipeline.ingest_file(str(p))
    elif p.is_dir():
        stats = pipeline.ingest_directory(str(p), recursive=recursive)
    else:
        console.print(f"[bold red]路径不存在:[/] {path}")
        raise typer.Exit(code=1)

    # 输出统计
    table = Table(title="入库统计")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    table.add_row("处理文件数", str(stats.files_processed))
    table.add_row("失败文件数", str(stats.files_failed))
    table.add_row("生成 Chunks", str(stats.chunks_created))
    for mod, count in stats.chunks_by_modality.items():
        table.add_row(f"  └ {mod}", str(count))
    table.add_row("总耗时", f"{stats.total_latency_ms/1000:.1f}s")

    if stats.errors:
        table.add_row("错误", "\n".join(stats.errors[:5]))

    console.print(table)


# ============================================================
# query — 问答
# ============================================================


@app.command()
def query(
    question: str = typer.Argument(..., help="要查询的问题"),
    top_k: int = typer.Option(10, help="检索结果数"),
    modality: Optional[str] = typer.Option(None, help="按模态过滤: text/image/audio/video"),
    cross_modal: bool = typer.Option(False, "--cross-modal", "-x", help="启用跨模态检索"),
    no_generate: bool = typer.Option(False, "--no-generate", help="仅检索, 不生成回答"),
):
    """查询向量库并生成回答。"""
    pipeline = get_query_pipeline()

    modality_filter = None
    if modality:
        try:
            modality_filter = Modality(modality)
        except ValueError:
            console.print(f"[bold red]无效的模态:[/] {modality}")
            console.print("可用: text, image, audio, video")
            raise typer.Exit(code=1)

    with console.status("[bold green]处理中..."):
        if no_generate:
            result = pipeline.retrieve_only(
                query=question,
                top_k=top_k,
                modality_filter=modality_filter,
            )
            _display_retrieve_results(result)
        else:
            response = pipeline.query(
                question=question,
                top_k=top_k,
                modality_filter=modality_filter,
                cross_modal=cross_modal,
            )
            _display_query_response(response)


def _display_query_response(response):
    """显示查询响应。"""
    # 回答
    console.print()
    console.print(Panel(
        Markdown(response.answer),
        title="[bold green]回答[/]",
        border_style="green",
    ))

    # 延迟
    lb = response.latency_breakdown
    console.print(
        f"[dim]⏱ 总耗时: {lb['total_ms']:.0f}ms "
        f"(Embed: {lb['embed_ms']:.0f}ms | "
        f"检索: {lb['retrieve_ms']:.0f}ms | "
        f"生成: {lb['generate_ms']:.0f}ms)[/]"
    )

    # 来源
    if response.sources:
        console.print()
        console.print(f"[bold]📎 来源 ({len(response.sources)} 条):[/]")
        table = Table(show_header=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("模态", width=8)
        table.add_column("来源", style="cyan")
        table.add_column("预览", style="white")
        table.add_column("相关度", justify="right")

        for i, src in enumerate(response.sources[:10], 1):
            mod = src["modality"]
            mod_emoji = {"text": "📝", "image": "🖼", "audio": "🎵", "video": "🎬"}.get(mod, "❓")
            preview = src.get("content_preview", "")[:80]
            table.add_row(
                str(i),
                f"{mod_emoji} {mod}",
                src.get("source_file_name", "")[:30],
                preview,
                f"{src['score']:.3f}",
            )

        console.print(table)


def _display_retrieve_results(result):
    """显示仅检索结果。"""
    console.print()
    console.print(f"[bold]查询:[/] {result['query']}")
    console.print(f"[bold]找到 {result['total_results']} 条结果[/]")
    console.print(f"[dim]模态分布: {result['modality_breakdown']}[/]")

    table = Table(title="检索结果")
    table.add_column("#", style="dim", width=4)
    table.add_column("模态", width=8)
    table.add_column("来源", style="cyan")
    table.add_column("预览", style="white")
    table.add_column("相关度", justify="right")

    for i, src in enumerate(result["results"][:20], 1):
        mod = src["modality"]
        mod_emoji = {"text": "📝", "image": "🖼", "audio": "🎵", "video": "🎬"}.get(mod, "❓")
        preview = src.get("content_preview", "")[:80]
        table.add_row(
            str(i),
            f"{mod_emoji} {mod}",
            src.get("source_file_name", "")[:30],
            preview,
            f"{src['score']:.3f}",
        )

    console.print(table)


# ============================================================
# cross-modal — 跨模态搜索
# ============================================================


@app.command(name="cross-modal")
def cross_modal(
    query_text: str = typer.Argument(..., help="搜索查询文本"),
    top_k: int = typer.Option(5, help="每种模态的结果数"),
    modalities: Optional[str] = typer.Option(None, help="模态列表 (逗号分隔): text,image,audio,video"),
):
    """跨模态搜索: 文本查询检索所有模态的内容。"""
    pipeline = get_query_pipeline()

    mod_list = None
    if modalities:
        mod_list = [Modality(m.strip()) for m in modalities.split(",")]

    with console.status("[bold green]跨模态搜索中..."):
        result = pipeline.retrieve_all_modalities(
            query=query_text,
            top_k_per_modality=top_k,
        )
        # 手动扩展以支持模态过滤
        if mod_list:
            filtered = {}
            for mod_str, data in result["modalities"].items():
                if Modality(mod_str) in mod_list:
                    filtered[mod_str] = data
            result["modalities"] = filtered

    console.print()
    console.print(f"[bold]跨模态搜索:[/] {query_text}")

    for mod_str in ["text", "image", "audio", "video"]:
        if mod_str not in result["modalities"]:
            continue
        data = result["modalities"][mod_str]
        mod_emoji = {"text": "📝", "image": "🖼", "audio": "🎵", "video": "🎬"}.get(mod_str, "❓")

        console.print()
        console.print(f"[bold]{mod_emoji} {mod_str.upper()} ({data['count']} 条)[/]")

        for i, r in enumerate(data["results"][:5], 1):
            preview = r.get("content_preview", "")[:100]
            score = r.get("score", 0)
            source = r.get("source_file_name", "")
            console.print(f"  {i}. [{score:.3f}] {source}: {preview}")


# ============================================================
# status — 状态查询
# ============================================================


@app.command()
def status():
    """查看向量库状态和统计。"""
    store = get_vector_store()
    stats = store.get_stats()

    console.print()
    console.print("[bold blue]📊 向量库状态[/]")
    console.print()

    table = Table()
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")

    table.add_row("Collection", stats.get("collection_name", "N/A"))
    table.add_row("存储路径", stats.get("persist_directory", "N/A"))
    table.add_row("总 Chunks", str(stats.get("total_chunks", 0)))
    table.add_row("嵌入维度", str(stats.get("embedding_dim", "N/A")))
    table.add_row("唯一来源数", str(stats.get("unique_sources", 0)))

    breakdown = stats.get("modality_breakdown", {})
    for mod, count in breakdown.items():
        emoji = {"text": "📝", "image": "🖼", "audio": "🎵", "video": "🎬"}.get(mod, "❓")
        table.add_row(f"  {emoji} {mod}", str(count))

    console.print(table)

    # 来源列表
    sources = store.list_sources()
    if sources:
        console.print()
        console.print("[bold]📁 已索引来源:[/]")
        for s in sources[:20]:
            mods = ", ".join(f"{m}:{c}" for m, c in s.get("modalities", {}).items())
            console.print(f"  • {s['source_file_name']} ({s['count']} chunks: {mods})")


# ============================================================
# clear — 清空 / 删除
# ============================================================


@app.command()
def clear(
    source: Optional[str] = typer.Option(None, help="按来源文件删除"),
    all_: bool = typer.Option(False, "--all", help="清空整个向量库"),
):
    """删除索引数据。"""
    store = get_vector_store()

    if all_:
        confirm = typer.confirm("⚠️ 确定要清空整个向量库吗? 此操作不可逆!")
        if confirm:
            count = store.clear()
            console.print(f"[green]✓ 已清空, 删除了 {count} 个 chunks[/]")
        else:
            console.print("已取消")
    elif source:
        count = store.delete_by_source(source)
        console.print(f"[green]✓ 已删除 {count} 个 chunks (来源: {source})[/]")
    else:
        console.print("[yellow]请指定 --source 或 --all[/]")


# ============================================================
# list-sources — 列出来源
# ============================================================


@app.command(name="list-sources")
def list_sources():
    """列出所有已索引的来源文件。"""
    store = get_vector_store()
    sources = store.list_sources()

    if not sources:
        console.print("[yellow]暂无已索引的来源[/]")
        return

    console.print()
    console.print(f"[bold]📁 已索引来源 ({len(sources)} 个):[/]")
    console.print()

    table = Table()
    table.add_column("文件", style="cyan")
    table.add_column("Chunks", justify="right")
    table.add_column("模态分布", style="green")

    for s in sources:
        mods = ", ".join(f"{m}:{c}" for m, c in s.get("modalities", {}).items())
        table.add_row(
            s.get("source_file_name", "?"),
            str(s.get("count", 0)),
            mods,
        )

    console.print(table)


# ============================================================
# main
# ============================================================


def main():
    """CLI 入口。"""
    app()


if __name__ == "__main__":
    main()
