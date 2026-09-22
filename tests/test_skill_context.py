from __future__ import annotations

from pathlib import Path

from langgraph.runtime import Runtime

from deepresearch.context.tagged import ContextAssembler
from deepresearch.middlewares.skill_injection import SkillInjectionMiddleware
from deepresearch.skill.config import SkillConfig
from deepresearch.skill.context import SkillContextRenderer
from deepresearch.skill.manager import SkillManager
from deepresearch.state import create_initial_state, merge_skill_runtime


def _runtime(tmp_path: Path, body: str = "Read primary sources."):
    root = tmp_path / "skills"
    directory = root / "verify"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: source-verification",
                "description: Verify important claims",
                "tags: [research]",
                "tools: [read_page]",
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )
    manager = SkillManager(
        SkillConfig(
            use="default",
            db_path=tmp_path / "skills.db",
            storage_path=tmp_path / "objects",
            skill_dirs=(root,),
            include_builtin=False,
            min_relevance=0.0,
            token_budget=500,
        )
    )
    record = manager.load_startup()[0]
    state = create_initial_state("verify sources")
    state["skills"]["selected"] = [
        {
            "skill_id": record.skill_id,
            "name": record.name,
            "content_hash": record.content_hash,
            "score": 0.9,
            "reason": "test",
            "forced": False,
            "allowed_tools": list(record.allowed_tools),
        }
    ]
    return manager, state


def test_renderer_loads_body_without_copying_it_into_state(tmp_path: Path) -> None:
    manager, state = _runtime(tmp_path)
    renderer = SkillContextRenderer(manager.store, token_budget=500)

    rendered = renderer.render(state)

    assert "<skill_context>" in rendered.text
    assert "Read primary sources." in rendered.text
    assert rendered.included[0]["name"] == "source-verification"
    assert "Read primary sources." not in repr(state["skills"])
    assert rendered.token_count > 0
    manager.close()


def test_renderer_compacts_at_p4_and_drops_non_terminal_at_p5(tmp_path: Path) -> None:
    manager, state = _runtime(tmp_path, body="FULL BODY THAT SHOULD DISAPPEAR")
    renderer = SkillContextRenderer(manager.store, token_budget=500)
    state["governance"]["context"]["pending_stages"] = ["P4"]

    compact = renderer.render(state)
    assert "Verify important claims" in compact.text
    assert "FULL BODY" not in compact.text

    state["governance"]["context"]["pending_stages"] = ["P5"]
    terminal = renderer.render(state)
    assert terminal.text == ""
    assert terminal.dropped[0]["drop_reason"] == "p5_non_terminal"
    manager.close()


def test_renderer_drops_content_that_cannot_fit_budget(tmp_path: Path) -> None:
    manager, state = _runtime(tmp_path, body="x " * 1_000)
    renderer = SkillContextRenderer(manager.store, token_budget=2)

    rendered = renderer.render(state)

    assert rendered.text == ""
    assert rendered.dropped[0]["drop_reason"] == "token_budget"
    manager.close()


def test_injection_middleware_records_once_per_render_signature(tmp_path: Path) -> None:
    manager, state = _runtime(tmp_path)
    renderer = SkillContextRenderer(manager.store, token_budget=500)
    middleware = SkillInjectionMiddleware(manager, renderer)

    first = middleware.before_model(state, Runtime())
    assert first is not None
    state["skills"] = merge_skill_runtime(state["skills"], first["skills"])

    assert state["skills"]["injection_count"] == 1
    assert state["skills"]["injected_tokens"] > 0
    assert middleware.before_model(state, Runtime()) is None
    skill_id = state["skills"]["selected"][0]["skill_id"]
    assert manager.store.get_metrics(skill_id).injections == 1  # type: ignore[union-attr]
    manager.close()


def test_context_assembler_projects_skills_without_mutating_messages(tmp_path: Path) -> None:
    manager, state = _runtime(tmp_path)
    renderer = SkillContextRenderer(manager.store, token_budget=500)
    assembler = ContextAssembler(skill_context_provider=renderer)
    original_messages = list(state["messages"])

    assembled = assembler.assemble(
        state,
        state["messages"],
        None,
        include_research_context=False,
    )

    assert "<skill_context>" in str(assembled.system_message.content)
    assert state["messages"] == original_messages
    assert all(message.type != "system" for message in state["messages"])
    manager.close()
