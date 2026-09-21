# DeepResearch 最小复现

这个项目不是直接复制 Poirot，而是沿着 Poirot 的核心执行链逐层重建：先让最小 Agent 跑通，再按 Git 阶段加入 Tool、Middleware、State、Skill 和其他基础设施。

当前版本是 **阶段 6A：带受限反思闭环的研究 Agent**：

```text
命令行问题
    ↓
Settings 读取模型配置
    ↓
ChatOpenAI（兼容 OpenAI 协议的模型）
    ↓
create_agent() 编译 Agent Graph
    ├── messages
    ├── research_question
    ├── search_records
    ├── page_records
    ├── plan / current_step_id
    ├── reflection_attempts / research_gaps
    ├── sources / observations
    └── final_report
    ↓
PlanContextMiddleware 选择性注入计划进度
    ↓
模型首次调用 write_research_plan
    ↓
模型判断是否调用 web_search
    ├── 不调用 → 直接回答
    └── 调用 → DuckDuckGo 搜索 → ToolMessage
                                ↓
                         EvidenceMiddleware
                         ├── search_records
                         ├── sources（按 URL 去重）
                         └── observations（关联 step_id）
                                ↓
                         再次调用模型
                                ↓
                         选择关键 URL 调用 read_page
                                ↓
                         EvidenceMiddleware
                         ├── page_records
                         └── 正文 observations（关联 step_id）
                                ↓
                         update_plan_step 推进计划
    ↓
模型准备输出最终回答
    ↓
ReflectionMiddleware 程序化检查
    ├── 计划和证据足够 → 允许结束
    └── 存在缺口 → 写入 research_gaps → 回到模型（最多 2 次）
                                                    ↓
                                      超过上限后保留缺口并允许结束
    ↓
ResearchResult(question, answer, state)
```

## 输入和输出

- 输入：一个非空的自然语言问题，例如“什么是 ReAct？”
- 输出：`ResearchResult`，其中包含原始问题、模型最终回答和完整 `ResearchState`。
- 当前解决的问题：先创建结构化计划，按步骤搜索、读取网页，把证据关联到当前步骤，并在模型准备结束时检查研究缺口。
- 当前不解决的问题：判断证据内容在语义上是否可靠、正式报告生成、长期记忆、Skill 和多 Agent。

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

显示由程序根据 State 计算的真实搜索记录和来源：

```bash
uv run deepresearch run "研究 LangChain Agent" --show-trace
```

`--show-trace` 中的统计不调用 LLM；它直接读取 `search_records`、`page_records`、
`sources` 和 `observations`，用于区分真实执行记录与模型生成的自然语言说明。

`run` 模式会把 `web_search` 和 `read_page` 注册给真实模型；模型可以先搜索，
再选择重要来源读取正文。`write_research_plan` 和 `update_plan_step` 负责创建并推进计划。
`PlanContextMiddleware` 每轮只向模型展示计划进度和聚合计数，不会把完整 State 全量注入。
`ReflectionMiddleware` 在模型不再调用 Tool、准备结束时读取 State，检查计划是否完成、
来源和 Observation 是否达到最小数量、是否成功读取过网页正文。检查不调用额外 LLM；
如果不满足，会把缺口交给下一轮模型，但最多回跳 2 次，避免无限循环。
`demo` 模式仍使用无工具的 Fake Model，保证在没有网络和 API Key 时也能验证基础链路。

`.env` 已被 `.gitignore` 排除，真实 API Key 不会进入 Git。

## 名词解释

- **LLM / 大模型**：接收消息并生成回答的模型，本阶段使用 `ChatOpenAI` 适配兼容 OpenAI 协议的服务。
- **Agent**：让模型在一个执行框架中运行的程序；当前模型可以判断是直接回答，还是先调用搜索工具。
- **Graph**：Agent 的执行流程图。`create_agent()` 会为我们生成最基本的模型调用循环。
- **ReAct**：Reason + Act，即“判断下一步并执行动作”。当前动作包括 `web_search` 和 `read_page`，工具结果会返回模型形成下一轮判断。
- **Tool**：可执行能力，例如网页搜索、网页读取和文件操作。
- **Skill**：告诉 Agent“应该怎样完成某类任务”的过程知识，通常是提示词，不等于 Tool。
- **Middleware**：插在 Agent 生命周期中的横切逻辑，例如模型调用前注入 Skill、工具调用后收集证据。
- **State**：Graph 节点之间共享的数据，例如消息、来源、研究计划和最终报告。
- **上下文投影**：从完整 State 中筛选当前模型真正需要的字段；本项目只投影计划和聚合进度，不投影全部内部数据。
- **反思（Reflection）**：模型准备结束时进行质量检查。当前版本是可测试的程序规则，不是再调用一次 LLM 自我评价。
- **回跳（jump_to）**：Middleware 返回的图路由指令。`jump_to="model"` 表示当前不结束，重新执行模型节点。

## 分阶段复现路线

每个阶段都应该满足“代码可运行、测试通过、单独 Git 提交”后，再进入下一阶段。

1. **阶段 0（已完成）— 最小 Agent**：CLI → 模型 → `create_agent` → 回答。
2. **阶段 1（已完成）— 第一个 Tool**：增加 `web_search`，让模型产生 tool call，并把搜索结果返回模型。
3. **阶段 2（已完成）— 研究 State**：加入 `search_records`、`sources`、`observations`、`research_question` 和 reducer。
4. **阶段 3（已完成）— 第一个 Middleware**：由 Evidence Middleware 把工具结果整理为证据，并提供真实执行 Trace。
5. **阶段 4（已完成）— 网页正文读取**：增加安全 URL 校验和 `read_page`，把关键网页正文沉淀为证据。
6. **阶段 5（已完成）— 结构化研究计划**：模型创建和推进 Plan，Middleware 选择性注入进度，证据关联步骤。
7. **阶段 6A（当前）— 受限研究反思**：程序检查计划和证据，通过 `jump_to` 最多继续研究 2 次。
8. **阶段 6B — 最终报告**：根据 State 中的证据生成有引用的正式报告，并保存到 `final_report`。
9. **阶段 7 — Skill**：把“如何检索、如何核验来源、如何写报告”做成可选择和注入的过程知识。
10. **阶段 8 — 工程能力**：持久化 checkpointer、流式输出、日志、错误恢复和测试评估。
11. **阶段 9 — 高级能力**：MCP、记忆、Sandbox、多 Agent；这些不要提前加入最小主链。

## Git 管理建议

当前项目直接在 `main` 上按阶段开发；每次提交前必须先跑完测试，让 `main` 始终保持可运行：

```text
main: minimal-agent → web-search-tool → research-state → middleware → skill
```

日常流程：

```bash
# 修改代码并测试
uv run pytest -q
git add .
git commit -m "feat: add web search tool"
```

一次提交只完成一个可解释的变化。不要提交 `.env`、`.venv`、缓存、运行日志和本地数据库。

## 与 Poirot 的对应关系

| 本项目当前文件 | Poirot 中对应职责 | 作用 |
|---|---|---|
| `src/deepresearch/cli.py` | `backend/app/cli/main.py` | 接收用户输入 |
| `src/deepresearch/config.py` | `backend/agents/config/` | 读取模型配置 |
| `src/deepresearch/state.py` | `agents/state/types.py` + `reducers.py` | 定义共享研究状态和合并规则 |
| `src/deepresearch/tools/web_search.py` | `agents/agent_tools/builtin/ddg_search.py` | 执行网页搜索 |
| `src/deepresearch/tools/read_page.py` | 网页读取类 Tool | 安全下载并提取网页正文 |
| `src/deepresearch/tools/research_plan.py` | Todo/计划类状态 Tool | 创建计划并推进步骤状态 |
| `src/deepresearch/middlewares/evidence.py` | `agents/middlewares/evidence_middleware.py` | 把搜索和正文结果沉淀为结构化证据 |
| `src/deepresearch/middlewares/plan_context.py` | 计划上下文 Middleware | 选择性向模型暴露计划进度 |
| `src/deepresearch/middlewares/reflection.py` | 反思/质量控制 Middleware | 在结束前检查缺口，并有限次回到模型 |
| `build_agent()` | `agents/leader/factory.py` | 编译 Agent Graph |
| `run_with_model()` | `agents/leader/agent.py` | 执行 Graph 并整理输出 |
| `ResearchResult` | `AgentRunResult` | 稳定的输出边界 |

先掌握这 5 个位置，再扩展功能，会比一开始搬运 Poirot 的全部模块更容易定位问题。
