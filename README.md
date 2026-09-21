# DeepResearch 最小复现

这个项目不是直接复制 Poirot，而是沿着 Poirot 的核心执行链逐层重建：先让最小 Agent 跑通，再按 Git 阶段加入 Tool、Middleware、State、Skill 和其他基础设施。

当前版本是 **阶段 0：最小 Agent**。它只包含：

```text
命令行问题
    ↓
Settings 读取模型配置
    ↓
ChatOpenAI（兼容 OpenAI 协议的模型）
    ↓
create_agent() 编译 Agent Graph
    ↓
graph.invoke()
    ↓
ResearchResult(question, answer)
```

## 输入和输出

- 输入：一个非空的自然语言问题，例如“什么是 ReAct？”
- 输出：`ResearchResult`，其中 `question` 是原始问题，`answer` 是模型最终回答。
- 当前解决的问题：验证 CLI、配置、模型、Agent Graph 和结果封装这一条最小主链。
- 当前不解决的问题：联网搜索、引用核验、研究规划、长程状态、Skill、记忆和多 Agent。

## 环境准备

项目要求 Python 3.12+，推荐使用 `uv`：

```bash
uv sync --extra dev
```

先运行完全离线的 Demo，不需要 API Key：

```bash
uv run deepresearch demo
```

预期看到：

```text
最小链路已跑通：命令行输入已经进入 LangChain Agent Graph，模型响应也已被封装为 ResearchResult。
```

运行测试：

```bash
uv run pytest -q
```

## 接入真实模型

复制环境变量模板：

```bash
cp .env.example .env
```

填写 `.env` 后运行：

```bash
uv run deepresearch run "请解释 ReAct Agent 的基本工作方式"
```

`.env` 已被 `.gitignore` 排除，真实 API Key 不会进入 Git。

## 名词解释

- **LLM / 大模型**：接收消息并生成回答的模型，本阶段使用 `ChatOpenAI` 适配兼容 OpenAI 协议的服务。
- **Agent**：让模型在一个执行框架中运行的程序。未来它可以判断何时调用工具、何时继续研究、何时结束。
- **Graph**：Agent 的执行流程图。`create_agent()` 会为我们生成最基本的模型调用循环。
- **ReAct**：Reason + Act，即“思考下一步并执行动作”。没有 Tool 时，本阶段只走一次模型回答；加入 Tool 后才形成真正有动作的循环。
- **Tool**：可执行能力，例如网页搜索、网页读取和文件操作。
- **Skill**：告诉 Agent“应该怎样完成某类任务”的过程知识，通常是提示词，不等于 Tool。
- **Middleware**：插在 Agent 生命周期中的横切逻辑，例如模型调用前注入 Skill、工具调用后收集证据。
- **State**：Graph 节点之间共享的数据，例如消息、来源、研究计划和最终报告。

## 分阶段复现路线

每个阶段都应该满足“代码可运行、测试通过、单独 Git 提交”后，再进入下一阶段。

1. **阶段 0（当前）— 最小 Agent**：CLI → 模型 → `create_agent` → 回答。
2. **阶段 1 — 第一个 Tool**：增加 `web_search`，让模型产生 tool call，并把搜索结果返回模型。
3. **阶段 2 — 研究 State**：加入 `sources`、`observations`、`research_question` 和 reducer。
4. **阶段 3 — 第一个 Middleware**：在工具调用后把结果整理为证据；先只做 Evidence Middleware。
5. **阶段 4 — 研究闭环**：加入 Todo/Plan、证据充分性检查和最终报告生成。
6. **阶段 5 — Skill**：把“如何检索、如何核验来源、如何写报告”做成可选择和注入的过程知识。
7. **阶段 6 — 工程能力**：持久化 checkpointer、流式输出、日志、错误恢复和测试评估。
8. **阶段 7 — 高级能力**：MCP、记忆、Sandbox、多 Agent；这些不要提前加入最小主链。

## Git 管理建议

推荐让 `main` 始终保持可运行，每一层能力使用一条短分支：

```text
main
 ├── feat/web-search-tool
 ├── feat/research-state
 ├── feat/evidence-middleware
 └── feat/skill-system
```

日常流程：

```bash
git switch -c feat/web-search-tool
# 修改代码并测试
git add .
git commit -m "feat: add web search tool"
git switch main
git merge --no-ff feat/web-search-tool
```

一次提交只完成一个可解释的变化。不要提交 `.env`、`.venv`、缓存、运行日志和本地数据库。

## 与 Poirot 的对应关系

| 本项目当前文件 | Poirot 中对应职责 | 作用 |
|---|---|---|
| `src/deepresearch/cli.py` | `backend/app/cli/main.py` | 接收用户输入 |
| `src/deepresearch/config.py` | `backend/agents/config/` | 读取模型配置 |
| `build_agent()` | `agents/leader/factory.py` | 编译 Agent Graph |
| `run_with_model()` | `agents/leader/agent.py` | 执行 Graph 并整理输出 |
| `ResearchResult` | `AgentRunResult` | 稳定的输出边界 |

先掌握这 5 个位置，再扩展功能，会比一开始搬运 Poirot 的全部模块更容易定位问题。
