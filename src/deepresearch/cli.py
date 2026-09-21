"""Command-line entry point for the minimal agent."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from deepresearch.agent import run_demo, run_question, stream_question
from deepresearch.events import ResearchEvent
from deepresearch.reporting import (
    format_research_event,
    format_trace,
    save_markdown_report,
)


def _print_event(event: ResearchEvent) -> None:
    print(format_research_event(event), flush=True)


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
        help="显示由程序统计的搜索、网页读取和来源记录",
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
        help="为当前进程内的 InMemory Checkpoint 指定任务 ID",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "demo":
            result = run_demo(args.question)
        elif args.stream:
            result = stream_question(
                args.question,
                _print_event,
                thread_id=args.thread_id,
            )
        else:
            result = run_question(args.question, thread_id=args.thread_id)
        output_path = None
        if args.command == "run" and args.output:
            output_path = save_markdown_report(result.state, args.output)
    except (ValueError, RuntimeError) as exc:
        print(f"错误：{exc}")
        return 1

    if getattr(args, "stream", False):
        print()
    print(result.answer)
    if result.thread_id is not None:
        print()
        print(f"任务 ID：{result.thread_id}（仅当前进程内有效）")
    if output_path is not None:
        print()
        print(f"报告已保存：{output_path}")
    if getattr(args, "show_trace", False):
        print()
        print(format_trace(result.state))
    return 0
