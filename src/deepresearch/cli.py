"""Command-line entry point for the minimal agent."""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from deepresearch.agent import (
    resume_question,
    run_demo,
    run_question,
    stream_question,
    stream_resume_question,
)
from deepresearch.checkpointing import get_checkpoint_tuple, open_sqlite_checkpointer
from deepresearch.events import ResearchEvent
from deepresearch.memory.bootstrap import get_memory_provider
from deepresearch.memory.config import MemoryConfig
from deepresearch.memory.schema import MemoryType
from deepresearch.memory.types import MemoryFilter
from deepresearch.skill.config import SkillConfig
from deepresearch.skill.manager import BUILTIN_SKILLS_DIR, SkillManager
from deepresearch.skill.parser import discover_skills
from deepresearch.skill.types import SkillRecord
from deepresearch.reporting import (
    format_governance_summary,
    format_memory_summary,
    format_research_event,
    format_skill_summary,
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
        "",
        *format_memory_summary(state).splitlines(),
        "",
        *format_skill_summary(state).splitlines(),
        "",
        *_memory_store_summary().splitlines(),
    ]
    return "\n".join(lines)


def _memory_store_summary() -> str:
    """Inspect the external truth store without requiring a model API key."""

    load_dotenv()
    config = MemoryConfig.from_env()
    provider = get_memory_provider(config)
    if provider is None:
        return "长期记忆库：未启用"
    traces = provider.store().list_by_filter(
        MemoryFilter(namespace=config.namespace, include_forgotten=True)
    )
    active = [trace for trace in traces if not trace.metadata.get("forgotten")]
    episodic = sum(trace.type == MemoryType.EPISODIC for trace in active)
    semantic = sum(trace.type == MemoryType.SEMANTIC for trace in active)
    procedural = sum(trace.type == MemoryType.PROCEDURAL for trace in active)
    forgotten = len(traces) - len(active)
    return "\n".join(
        [
            "长期记忆库：",
            f"存储路径：{config.storage_path}",
            f"Namespace：{config.namespace}",
            f"记忆总数：{len(traces)}",
            f"有效记忆：{len(active)}",
            f"Episodic：{episodic}",
            f"Semantic：{semantic}",
            f"Procedural：{procedural}",
            f"Forgotten：{forgotten}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepresearch",
        description="可恢复、可审计的 Deep Research Agent",
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
    run.add_argument(
        "--skill",
        action="append",
        default=[],
        metavar="NAME",
        help="按名称强制选择 Skill；可重复传入",
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

    skills = subparsers.add_parser(
        "skills",
        help="验证、查看和管理本地 Skill 目录",
    )
    skill_commands = skills.add_subparsers(dest="skill_command", required=True)
    skill_commands.add_parser("list", help="列出当前激活的 Skill 版本")
    show = skill_commands.add_parser("show", help="查看 Skill 元数据与正文")
    show.add_argument("identifier", help="Skill 名称或 skill_id")
    history = skill_commands.add_parser("history", help="查看 Skill 版本历史")
    history.add_argument("name", help="Skill 名称")
    validate = skill_commands.add_parser("validate", help="只校验，不写入仓库")
    validate.add_argument("paths", nargs="*", help="待扫描目录；默认读取配置")
    for action in ("enable", "disable"):
        command = skill_commands.add_parser(action, help=f"{action} Skill 版本")
        command.add_argument("identifier", help="Skill 名称或 skill_id")
    rollback = skill_commands.add_parser("rollback", help="切换激活版本")
    rollback.add_argument("skill_id", help="目标历史版本的 skill_id")
    return parser


def _cli_skill_config() -> SkillConfig:
    load_dotenv()
    config = SkillConfig.from_env()
    return config if config.enabled else replace(config, use="default")


def _resolve_skill(manager: SkillManager, identifier: str) -> SkillRecord:
    record = manager.store.get(identifier) or manager.store.get_active(identifier)
    if record is None:
        raise ValueError(f"找不到 Skill：{identifier}")
    return record


def _skill_record_line(record: SkillRecord) -> str:
    status = "enabled" if record.enabled else "disabled"
    return (
        f"{record.name} [{record.skill_id}] v{record.version} {status} "
        f"origin={record.lineage.origin} selections={record.total_selections} "
        f"injections={record.total_injections} "
        f"completion={record.completion_rate:.1%}"
    )


def _validate_skills(args: argparse.Namespace) -> int:
    config = _cli_skill_config()
    requested = tuple(Path(path) for path in args.paths)
    sources: list[tuple[tuple[str | Path, ...], str]] = []
    if requested:
        sources.append((requested, "IMPORTED"))
    else:
        if config.include_builtin and BUILTIN_SKILLS_DIR.exists():
            sources.append(((BUILTIN_SKILLS_DIR,), "BUILTIN"))
        sources.append((config.skill_dirs, "IMPORTED"))
    valid = 0
    invalid = 0
    for directories, origin in sources:
        result = discover_skills(
            directories,
            origin=origin,  # type: ignore[arg-type]
            max_file_chars=config.max_file_chars,
        )
        for parsed in result.skills:
            print(f"[有效] {parsed.record.name}：{parsed.record.source_path}")
            valid += 1
        for error in result.errors:
            print(f"[无效] {error.path}：{error.error}")
            invalid += 1
    print(f"校验完成：{valid} 个有效，{invalid} 个无效")
    return 1 if invalid else 0


def _manage_skills(args: argparse.Namespace) -> int:
    if args.skill_command == "validate":
        return _validate_skills(args)
    manager = SkillManager(_cli_skill_config())
    try:
        manager.load_startup()
        if args.skill_command == "list":
            records = manager.list_skills()
            if not records:
                print("没有发现 Skill。")
            for record in records:
                print(_skill_record_line(record))
            return 0
        if args.skill_command == "history":
            records = manager.store.get_versions(args.name)
            if not records:
                raise ValueError(f"找不到 Skill：{args.name}")
            for record in records:
                marker = " * active" if record.is_active else ""
                print(f"{_skill_record_line(record)}{marker}")
            return 0
        if args.skill_command == "rollback":
            manager.store.rollback(args.skill_id)
            print(f"已切换激活版本：{args.skill_id}")
            return 0
        record = _resolve_skill(manager, args.identifier)
        if args.skill_command == "show":
            print(_skill_record_line(record))
            print(f"description: {record.description}")
            print(f"tags: {', '.join(record.tags) or '-'}")
            print(f"tools: {', '.join(record.allowed_tools) or '-'}")
            print(f"source: {record.source_path}")
            print()
            print(manager.store.read_body(record.skill_id))
            return 0
        enabled = args.skill_command == "enable"
        manager.store.set_enabled(record.skill_id, enabled)
        print(f"已{'启用' if enabled else '停用'}：{record.name} [{record.skill_id}]")
        return 0
    finally:
        manager.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = None

    try:
        if args.command == "skills":
            return _manage_skills(args)
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
            result = stream_question(
                args.question,
                _print_event,
                thread_id=args.thread_id,
                skill_overrides=tuple(args.skill),
            )
        else:
            result = run_question(
                args.question,
                thread_id=args.thread_id,
                skill_overrides=tuple(args.skill),
            )
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
