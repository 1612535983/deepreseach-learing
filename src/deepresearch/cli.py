"""Command-line entry point for the minimal agent."""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence
from typing import Any

from deepresearch.agent import (
    resume_question,
    run_demo,
    run_question,
    stream_question,
    stream_resume_question,
)
from deepresearch.checkpointing import get_checkpoint_tuple, open_sqlite_checkpointer
from deepresearch.events import ResearchEvent
from deepresearch.reporting import (
    format_governance_summary,
    format_research_event,
    format_trace,
    save_markdown_report,
)


def _print_event(event: ResearchEvent) -> None:
    print(format_research_event(event), flush=True)


def _checkpoint_summary(thread_id: str) -> str:
    """Format the latest saved metadata without calling the model."""

    with open_sqlite_checkpointer() as checkpointer:
        checkpoint = get_checkpoint_tuple(checkpointer, thread_id)
    state: dict[str, Any] = dict(
        checkpoint.checkpoint.get("channel_values", {})
    )
    plan = state.get("plan") or {}
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    completed_steps = sum(
        1
        for step in steps
        if isinstance(step, dict) and step.get("status") == "completed"
    )
    metadata = checkpoint.metadata
    lines = [
        f"任务 ID：{thread_id}",
        f"研究问题：{state.get('research_question') or '未记录'}",
        f"Checkpoint 时间：{checkpoint.checkpoint.get('ts') or '未知'}",
        f"Graph 步骤：{metadata.get('step', '未知')}",
        f"计划进度：{completed_steps}/{len(steps)}",
        f"搜索次数：{len(state.get('search_records', []))}",
        f"网页读取次数：{len(state.get('page_records', []))}",
        f"来源数量：{len(state.get('sources', []))}",
        f"证据数量：{len(state.get('observations', []))}",
        f"最终报告：{'已生成' if state.get('final_report') else '未生成'}",
        "",
        *format_governance_summary(state).splitlines(),
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepresearch",
        description="从 Poirot 核心执行链开始复现 Deep Research",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="离线验证最小 Agent 链路")
    demo.add_argument(
        "question",
        nargs="?",
        default="这个最小 Agent 的执行链是否已经跑通？",
    )

    run = subparsers.add_parser("run", help="使用 .env 中的真实模型回答问题")
    run.add_argument("question", help="要交给 Agent 的问题")
    run.add_argument(
        "--show-trace",
        action="store_true",
        help="显示由程序统计的研究记录和上下文治理指标",
    )
    run.add_argument(
        "--output",
        metavar="REPORT.md",
        help="把 final_report 保存为新的 Markdown 文件（不会覆盖已有文件）",
    )
    run.add_argument(
        "--stream",
        action="store_true",
        help="实时显示计划、Tool、证据和反思进度",
    )
    run.add_argument(
        "--thread-id",
        help="指定持久化任务 ID；省略时自动生成",
    )

    resume = subparsers.add_parser(
        "resume",
        help="从 SQLite 中的最新 Checkpoint 继续任务",
    )
    resume.add_argument("thread_id", help="要恢复的任务 ID")
    resume.add_argument(
        "--show-trace",
        action="store_true",
        help="显示恢复后的研究记录和上下文治理指标",
    )
    resume.add_argument(
        "--output",
        metavar="REPORT.md",
        help="把恢复后的 final_report 保存为新的 Markdown 文件",
    )
    resume.add_argument(
        "--stream",
        action="store_true",
        help="实时显示恢复后的执行进度",
    )

    inspect = subparsers.add_parser(
        "inspect",
        help="查看持久化任务摘要，不调用模型",
    )
    inspect.add_argument("thread_id", help="要查看的任务 ID")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = None

    try:
        if args.command == "inspect":
            print(_checkpoint_summary(args.thread_id))
            return 0
        if args.command == "demo":
            result = run_demo(args.question)
        elif args.command == "resume":
            if args.stream:
                result = stream_resume_question(args.thread_id, _print_event)
            else:
                result = resume_question(args.thread_id)
        elif args.stream:
            result = stream_question(args.question, _print_event, thread_id=args.thread_id)
        else:
            result = run_question(args.question, thread_id=args.thread_id)
        if args.command in {"run", "resume"} and args.output:
            output_path = save_markdown_report(result.state, args.output)
    except (ValueError, RuntimeError, OSError, sqlite3.DatabaseError) as exc:
        print(f"错误：{exc}")
        return 1

    if getattr(args, "stream", False):
        print()
    print(result.answer)
    if result.thread_id is not None:
        print()
        print(f"任务 ID：{result.thread_id}（已持久化到 SQLite）")
    if output_path is not None:
        print()
        print(f"报告已保存：{output_path}")
    if getattr(args, "show_trace", False):
        print()
        print(format_trace(result.state))
    return 0
