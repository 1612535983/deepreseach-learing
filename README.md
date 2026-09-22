# DeepResearch 最小复现

这个项目不是直接复制 Poirot，而是沿着 Poirot 的核心执行链逐层重建：先让最小 Agent 跑通，再按 Git 阶段加入 Tool、Middleware、State、Skill 和其他基础设施。

当前版本是 **阶段 9B：具备 Tagged Context 投影能力的研究 Agent**。原始消息继续保存在
State 中，模型调用前会额外组装一份带语义标签的请求视图；暂时不会删除、外化或压缩消息：

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
    ├── final_report
    ├── governance.context
    │   ├── 当前 Token / 上下文窗口 / 占用比例
    │   ├── 累计 Token / 模型调用次数
    │   └── pending_stages / hard_limit_reached
    ├── tagged_context（最后一次模型可见上下文的审计快照）
    └── SQLite Checkpointer（按 thread_id 持久化快照）
    ↓
TaggedContextMiddleware 组装 request-scoped 模型视图
    ├── <system> / <goal> / <date>
    ├── <research_plan> / <research_progress> / <research_gaps>
    └── <turn> / <answer> / <toolcall> / <toolresult>
    ↓
SequentialToolCallMiddleware 禁止并行 Tool Call
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
模型调用 write_final_report
    ↓
校验研究进度和引用 URL
    ↓
Markdown 报告写入 State.final_report
    ↓
模型准备结束
    ↓
ReflectionMiddleware 程序化检查
    ├── 计划、证据、报告齐全 → 允许结束
    └── 存在缺口 → 写入 research_gaps → 回到模型（最多 2 次）
                                                    ↓
                                      超过上限后保留缺口并允许结束
    ↓
ResearchResult(question, answer, state)
    ↓ 可选 --output
save_markdown_report() 创建 .md 文件
```

流式模式不会重新组装另一套 Agent。`create_agent()` 仍然只负责生成同一个 Graph，
执行层可以选择 `invoke()` 一次性返回，或者选择 `stream()/astream()` 逐步接收结果：

```text
Compiled Agent Graph
    ├── invoke()  → 最后一次性返回 State
    └── stream()  → updates 转 ResearchEvent，values 保存最终 State
                                      ↓
                                CLI 实时显示
```

Checkpointer 在 `create_agent()` 时装入 Graph；运行时通过 config 提供 `thread_id`：

```text
SqliteSaver → create_agent(checkpointer=...)
                         ↓
graph.stream(..., config={"configurable": {"thread_id": "research-001"}})
                         ↓
LangGraph 在每个执行步骤后自动把 State 保存到 SQLite
                         ↓
下一个 CLI 进程用同一 thread_id 调用 graph.stream(None, config=...)
                         ↓
从最近的 Graph 节点继续执行
```

## 输入和输出

- 输入：一个非空的自然语言问题，例如“什么是 ReAct？”
- 输出：`ResearchResult`，其中包含原始问题、模型最终回答和完整 `ResearchState`。
- 当前解决的问题：创建计划、搜索和读取网页、记录证据、检查研究缺口、校验报告引用，并输出 Markdown 研究报告。
- 当前不解决的问题：判断证据内容在语义上是否可靠、长期记忆、Skill 和多 Agent。

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

把最终报告保存为新的 Markdown 文件：

```bash
uv run deepresearch run "研究 LangChain Agent" \
  --output reports/langchain-agent.md
```

`--output` 只负责把 `State.final_report` 写入磁盘，不会调用 LLM，也不会覆盖已有文件。
LLM 负责生成 Markdown 内容；`write_final_report` Tool 负责校验引用并更新 State；
`save_markdown_report()` 才是真正创建 `.md` 文件的代码。

实时显示计划、Tool、证据、反思和报告进度：

```bash
uv run deepresearch run "研究 LangChain Agent" \
  --stream \
  --show-trace \
  --output reports/langchain-agent.md
```

`--stream` 使用程序根据真实 Graph 更新生成的 `ResearchEvent`，不是让 LLM 描述自己做了什么。
流式执行结束后仍会返回完整 `ResearchResult`，因此 Trace 和 Markdown 导出可以继续使用。

为当前运行指定 Checkpoint 任务 ID：

```bash
uv run deepresearch run "研究 LangChain Agent" \
  --thread-id research-001 \
  --stream
```

省略 `--thread-id` 时会自动生成 ID。Checkpoint 默认写入
`.deepresearch/checkpoints.sqlite`，该目录已被 Git 忽略。新任务不能复用已有 ID，以免把新问题
合并进旧 State；如果任务中断，应改用：

```bash
uv run deepresearch resume research-001 --stream
```

`resume` 不会重新调用 `create_initial_state()`。它把 `None` 作为 Graph 输入，并用相同的
`thread_id` 读取最近快照；已完成任务会直接返回保存结果，中断任务会从待执行节点继续。
恢复后的报告也可以配合 `--show-trace` 和 `--output reports/result.md` 使用。

不调用模型、只查看任务的最新持久化摘要：

```bash
uv run deepresearch inspect research-001
```

`inspect` 会显示原问题、Checkpoint 时间、Graph 步骤、计划进度、研究统计，以及保存在
Checkpoint State 中的模型名称、上下文窗口、当前 Token、占用比例、累计 Token、模型调用
次数和 `pending_stages`。同步 Graph 使用 `SqliteSaver`，异步 Graph 使用
`AsyncSqliteSaver`。不要让两个进程同时用同一个 `thread_id` 执行；不同任务应使用不同 ID。

`--show-trace` 中的统计不调用 LLM；它直接读取 `search_records`、`page_records`、
`sources`、`observations` 和 `governance.context`，用于区分真实执行记录与模型生成的
自然语言说明。

`run` 模式会把 `web_search` 和 `read_page` 注册给真实模型；模型可以先搜索，
再选择重要来源读取正文。`write_research_plan` 和 `update_plan_step` 负责创建并推进计划。
研究完成后，模型必须调用 `write_final_report`，并提交 Markdown 内容和实际引用的 URL。
Tool 会确认研究已经完成、至少引用两个收集过的来源，而且声明的 URL 确实出现在报告正文中。
`TaggedContextMiddleware` 每轮从 State 投影系统规则、研究目标、计划进度和聚合计数，并为
对话、回答、Tool Call 和 Tool 结果建立语义标签。投影只修改当次 `ModelRequest`，不会覆盖
State 中的原始消息；最后一次投影另存到 `tagged_context`，用于 Checkpoint 和审计。
`SequentialToolCallMiddleware` 会向模型绑定层传入 `parallel_tool_calls=False`，确保会更新
Plan 和 `current_step_id` 的 Tool 逐轮执行，避免多个 `Command` 同时写入同一个 State 字段。
`ReflectionMiddleware` 在模型不再调用 Tool、准备结束时读取 State，检查计划是否完成、
来源和 Observation 是否达到最小数量、是否成功读取过网页正文。检查不调用额外 LLM；
如果不满足，会把缺口交给下一轮模型，但最多回跳 2 次，避免无限循环。
`ContextGovernanceMiddleware` 在每次模型调用后统计当前消息 Token、识别模型上下文窗口、
累计供应商返回的 Token usage，并按 P1～P5 阈值写入 `governance.context`。当前阶段仅记录
`pending_stages`，不会执行外化、压缩或强制收尾。
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
- **Markdown**：一种纯文本格式。LLM 生成 Markdown 字符串，Python 再把字符串写入 `.md` 文件。
- **ResearchEvent**：由 LangGraph 原始更新转换出的稳定应用事件，用于 CLI，未来也可以用于 SSE 或 WebSocket。
- **Checkpointer**：在 Graph 执行步骤结束后自动保存 State 快照；`thread_id` 用来区分任务。
- **Checkpoint**：某一执行时刻的 State 和 Graph 运行位置；恢复时不需要应用代码手动重放每个 Tool。
- **SQLite**：单文件关系型数据库。本项目用它持久化 Checkpoint，不用额外启动数据库服务。
- **Token**：模型处理文本时使用的计量单位，不等同于字符数；当前优先使用模型计数器，不可用时采用字符估算。
- **上下文窗口**：一次模型请求可容纳的 Token 上限。当前按显式配置、模型属性、名称映射、默认值的顺序识别，并记录识别来源。
- **pending_stages**：当前占用比例已经触发但尚未真正执行的 P1～P5 治理阶段。

## 分阶段复现路线

每个阶段都应该满足“代码可运行、测试通过、单独 Git 提交”后，再进入下一阶段。

1. **阶段 0（已完成）— 最小 Agent**：CLI → 模型 → `create_agent` → 回答。
2. **阶段 1（已完成）— 第一个 Tool**：增加 `web_search`，让模型产生 tool call，并把搜索结果返回模型。
3. **阶段 2（已完成）— 研究 State**：加入 `search_records`、`sources`、`observations`、`research_question` 和 reducer。
4. **阶段 3（已完成）— 第一个 Middleware**：由 Evidence Middleware 把工具结果整理为证据，并提供真实执行 Trace。
5. **阶段 4（已完成）— 网页正文读取**：增加安全 URL 校验和 `read_page`，把关键网页正文沉淀为证据。
6. **阶段 5（已完成）— 结构化研究计划**：模型创建和推进 Plan，Middleware 选择性注入进度，证据关联步骤。
7. **阶段 6A（已完成）— 受限研究反思**：程序检查计划和证据，通过 `jump_to` 最多继续研究 2 次。
8. **阶段 6B（已完成）— 最终报告**：校验报告引用，保存到 `final_report`，并由 CLI 导出 `.md` 文件。
9. **阶段 7（已完成）— 流式输出**：把 `stream()/astream()` 更新转换成统一事件，并在 CLI 实时显示。
10. **阶段 8A（已完成）— 内存 Checkpointer**：把 Checkpointer 装入 Graph，通过 config 的 `thread_id` 自动保存和隔离 State。
11. **阶段 8B（已完成）— SQLite Checkpointer**：让 Checkpoint 跨进程持久化，并增加 `resume` 和 `inspect` 命令。
12. **阶段 9A（已完成）— Context Governance 观测层**：增加治理 State、Token 统计、模型窗口识别、阈值判断、Middleware 记录，以及 Trace/inspect 展示；暂不修改消息。
13. **阶段 9B（当前）— Tagged Context**：保留原始 State，同时组装并审计带语义标签的模型请求视图。
14. **阶段 9C — P1 Tool 结果外化**：把大体积 Tool 结果保存到消息之外，只保留引用和摘要。
15. **阶段 9D — P4 Snapshot 与摘要压缩**：保存压缩前快照，并把较旧上下文替换成可恢复摘要。
16. **阶段 9E — P5 强制收尾**：高占用时停止继续调用 Tool，引导模型生成最终产物。
17. **阶段 10 — 记忆系统**：区分任务短期记忆和可跨任务检索的长期记忆。
18. **阶段 11 — Skill**：在上下文预算内加载“如何检索、核验来源、写报告”的过程知识。
19. **后续阶段**：错误恢复与预算、评估、MCP、Sandbox、API/前端，最后再考虑多 Agent。

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
| `src/deepresearch/checkpointing.py` | Checkpoint 配置层 | 创建 Saver、生成 `thread_id` 和运行 config |
| `src/deepresearch/events.py` | Agent 事件协议 | 把 LangGraph 原始更新转换成稳定的 `ResearchEvent` |
| `src/deepresearch/state.py` | `agents/state/types.py` + `reducers.py` | 定义共享研究状态和合并规则 |
| `src/deepresearch/context/` | 上下文治理策略层 | 定义治理类型、Token 统计、窗口识别和 P1～P5 阈值判断 |
| `src/deepresearch/context/tagged.py` | Tagged Context 组装层 | 把选定 State 和消息转换成 request-scoped 标签化视图 |
| `src/deepresearch/tools/web_search.py` | `agents/agent_tools/builtin/ddg_search.py` | 执行网页搜索 |
| `src/deepresearch/tools/read_page.py` | 网页读取类 Tool | 安全下载并提取网页正文 |
| `src/deepresearch/tools/research_plan.py` | Todo/计划类状态 Tool | 创建计划并推进步骤状态 |
| `src/deepresearch/tools/final_report.py` | 最终产物 Tool | 校验报告与来源，并把 Markdown 保存到 State |
| `src/deepresearch/middlewares/evidence.py` | `agents/middlewares/evidence_middleware.py` | 把搜索和正文结果沉淀为结构化证据 |
| `src/deepresearch/middlewares/plan_context.py` | 计划上下文兼容层 | 提供计划投影格式，实际 Agent 由 Tagged Context 统一组装 |
| `src/deepresearch/middlewares/sequential_tools.py` | Tool 调度 Middleware | 禁止并行 Tool Call，避免 State 并发写冲突 |
| `src/deepresearch/middlewares/reflection.py` | 反思/质量控制 Middleware | 在结束前检查缺口，并有限次回到模型 |
| `src/deepresearch/middlewares/context_governance.py` | 上下文治理 Middleware | 每次模型调用后把测量和阈值判断结果写入 State |
| `src/deepresearch/middlewares/tagged_context.py` | Tagged Context Middleware | 请求前保存审计快照，并只对本次模型请求应用标签化投影 |
| `src/deepresearch/reporting.py` | 输出层 | 显示 Trace，并把 `final_report` 写成 `.md` 文件 |
| `build_agent()` | `agents/leader/factory.py` | 编译 Agent Graph |
| `run_with_model()` | `agents/leader/agent.py` | 执行 Graph 并整理输出 |
| `ResearchResult` | `AgentRunResult` | 稳定的输出边界 |

先掌握这 5 个位置，再扩展功能，会比一开始搬运 Poirot 的全部模块更容易定位问题。
