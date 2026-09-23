# LangGraph Agent 核心工作机制研究报告

## 一、概述

LangGraph 是构建在 LangChain 之上的一个库，用于创建**有环图（cyclical graphs）**，以支持 Agent 运行时。LangChain 官方博客指出，LangChain 的链本质上是**有向无环图（DAG）**，而复杂 LLM 应用常见的模式是引入"循环"——用 LLM 推理下一步该做什么，本质上相当于"在 for 循环中运行 LLM"，这类系统通常被称为 Agent。LangGraph 通过将这类"状态机"以图的形式表达出来，让开发者既能获得循环带来的灵活性，又能保留对流程的人工控制。

> 官方定义："At its core, LangGraph models agent workflows as graphs."（LangGraph 的核心是把 Agent 工作流建模为图。）

## 二、三大核心构件：State、Nodes、Edges

根据官方 Graph API 文档，LangGraph 用三个关键组件定义 Agent 行为：

| 组件 | 作用 |
|------|------|
| **State（状态）** | 表示应用当前快照的共享数据结构，通常用共享状态 schema 定义 |
| **Nodes（节点）** | 编码 Agent 逻辑的函数。接收当前状态作为输入，执行计算或副作用，返回更新后的状态 |
| **Edges（边）** | 根据当前状态决定下一个执行哪个节点的函数，可以是条件分支或固定转移 |

官方文档用一句话概括：**"nodes do the work, edges tell what to do next."（节点干活，边决定下一步。）**

值得注意的是，节点和边本质上都只是函数——它们可以包含 LLM，也可以只是普通代码。

## 三、底层执行模型：消息传递与 Super-step

LangGraph 的底层图算法采用**消息传递（message passing）**来定义通用程序，灵感来自 Google 的 **Pregel** 系统：

- 当某个节点完成操作后，会沿一条或多条边向其他节点发送消息；
- 接收节点执行其函数，再把结果消息传给下一批节点，如此往复；
- 程序以离散的 **"super-step"（超级步）**推进。一个 super-step 可视为对图节点的一次迭代：**并行运行的节点属于同一个 super-step，顺序运行的节点属于不同的 super-step**；
- 执行开始时所有节点处于非活跃状态；节点在任一入边（channel）收到新消息时变为活跃；每个 super-step 结束时，没有收到消息的节点通过标记自己为非活跃来"投票停止"；
- 当所有节点都非活跃且没有消息在传输时，图执行终止。

## 四、StateGraph 与状态管理

### 4.1 StateGraph 与编译

`StateGraph` 是主要使用的图类，由用户定义的 State 对象参数化。构建流程为：**定义状态 → 添加节点和边 → 编译（compile）**。

编译步骤会做基本的结构检查（如无孤立节点），并可在此指定运行时参数（如 checkpointer、breakpoints）。**必须先编译才能使用图**。

### 4.2 Schema 与 Reducer

State 由**图的 schema** 和 **reducer 函数**组成。schema 是所有节点和边的输入 schema，可以是 `TypedDict` 或 Pydantic 模型。所有节点发出对 State 的更新，再通过指定的 reducer 函数应用。

Reducer 是理解状态更新的关键，每个 key 有独立的 reducer：

- **默认 reducer**：忽略左参数，直接用右参数（节点更新值）覆盖状态值；
- **自定义 reducer**：合并左右参数，适合累积值（如向列表追加）。例如用 `Annotated[list[str], operator.add]` 让 `bar` 键通过列表相加来更新；
- **Overwrite**：用于绕过 reducer 直接覆盖状态值（例如清空一个使用合并 reducer 的错误缓冲区）；
- **UntrackedValue**：用于图执行期间存在但**永不 checkpoint** 的状态字段（如不可序列化的数据库连接、临时缓存）。

### 4.3 多 schema

除了单一 schema，LangGraph 还支持：
- **私有 schema（PrivateState）**：节点可写入私有状态通道用于内部通信；
- **显式输入/输出 schema**：约束图的输入与输出（`StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)`）。

节点可以写入图状态中的任意通道（图状态是初始化时定义的所有通道的并集）。

## 五、循环与条件路由

LangGraph 相对传统 DAG 链的核心价值在于**支持循环**。官方博客描述了典型 Agent 循环的两个步骤：

1. 调用 LLM 决定 (a) 采取什么行动，或 (b) 给用户什么回复；
2. 执行给定行动，然后回到步骤 1。

重复直到生成最终回复。边分为几类：
- **起始边**：连接图起点到某个节点（`set_entry_point`）；
- **普通边**：一个节点之后**总是**调用另一个节点（`add_edge("tools", "model")`）；
- **条件边**：用一个函数（常由 LLM 驱动）决定下一个节点，需传入上游节点、判断函数和映射表（`add_conditional_edge("model", should_continue, {"end": END, "continue": "tools"})`）。

此外还有特殊的 `END` 节点表示图的结束——循环必须能够终止。

## 六、持久化：Checkpointer 与 Store

官方 Persistence 文档指出，持久化让 LangGraph 应用在单次运行之外保留有用信息，适用于继续对话、中断后恢复、故障恢复或跨交互记忆。LangGraph 提供两套互补的持久化系统：

| | **Checkpointer** | **Store** |
|---|---|---|
| 持久化内容 | 图的**状态快照** | 应用定义的键值数据 |
| 作用域 | 单个 thread | 跨 thread |
| 记忆类型 | 短期、线程级记忆 | 长期、跨线程记忆 |
| 用途 | 对话连续性、人机协同、时间旅行、容错 | 用户偏好、事实、共享知识 |
| 访问方式 | 在 config 中传 `thread_id` | 从节点或应用代码读写 |

大多数应用可同时使用两者：checkpointer 跟踪当前 thread，store 跟踪跨 thread 的持久信息。编译时通过 `builder.compile(checkpointer=checkpointer, store=store)` 挂载。

**常见问题**：`MemorySaver`/`InMemorySaver` 把 checkpoint 存在内存中，进程重启即丢失，生产环境应使用 `PostgresSaver` 或 `SqliteSaver`；长对话中 checkpoint 会无限增长，需定期清理或设置保留策略。

## 七、人机协同（Human-in-the-Loop）与 Interrupts

Interrupts 允许在图的特定点暂停执行，等待外部输入后再继续，从而实现人机协同模式。触发中断时，LangGraph 用持久化层保存图状态，并无限期等待直到恢复执行。

工作机制：
- 在节点中任意位置调用 `interrupt()` 函数，接受任意 JSON 可序列化的值并呈现给调用方；
- 需要 **checkpointer** 持久化状态，以及 config 中的 **thread ID** 让运行时知道从哪个状态恢复；
- 恢复时用 `Command(resume=...)` 重新调用图，该值成为节点内 `interrupt()` 调用的返回值；
- 与静态 breakpoint（在特定节点前后暂停）不同，interrupts 是**动态的**，可放在代码任意位置并基于应用逻辑条件触发。

**关键点**：恢复时必须使用与中断时相同的 thread ID；节点恢复时会从调用 interrupt 的节点开头重新执行，因此 interrupt 之前的代码会再次运行。常见模式包括审批工作流、审查与编辑、中断工具调用、验证人工输入等。

## 八、第三方视角与交叉验证

Mem0 的技术文章从第三方角度印证了上述机制，并补充了实践视角：

- LangGraph 通过**节点、边、状态管理**三大核心概念，把 AI 从简单应答者转变为能思考、行动、观察并动态决定下一步的问题解决者；
- 与 LangChain 的对比：LangChain 擅长**线性链**（状态管理有限、决策最少、工具选择预定），LangGraph 擅长**有环图**（状态管理先进、决策复杂、工具选择自适应）；
- 二者互补而非竞争——许多生产系统在 LangGraph 工作流中使用 LangChain 组件；
- **重要局限**：LangGraph 的状态管理在单个会话内表现优秀，但**记忆并非一等抽象**。跨会话的持久化必须由开发者显式设计、存储和检索（通过 threads 和 checkpointers），否则 Agent 默认只有会话级上下文，会话结束即遗忘用户偏好。

## 九、结论

LangGraph Agent 的核心工作机制可归纳为：

1. **图建模**：把 Agent 工作流建模为图，由 State、Nodes、Edges 三要素构成；
2. **消息传递执行**：基于 Pregel 风格的消息传递，以离散 super-step 推进，支持并行与顺序执行；
3. **状态驱动**：共享状态 + 每键独立的 reducer 函数管理状态更新，支持覆盖、累积、私有通道与多 schema；
4. **循环与条件路由**：通过条件边实现"LLM 在循环中推理"的 Agent 行为，同时保留人工控制；
5. **持久化**：Checkpointer（线程级短期记忆）与 Store（跨线程长期记忆）两套互补系统；
6. **人机协同**：通过动态 interrupts 暂停/恢复执行，实现审批、审查、编辑等模式。

这套机制使 LangGraph 能够构建比传统线性链更灵活、更可控、更可靠的 Agent 系统。

## 参考来源

1. LangChain 官方文档 — Graph API overview：https://docs.langchain.com/oss/python/langgraph/graph-api
2. LangChain 官方文档 — Persistence：https://docs.langchain.com/oss/python/langgraph/persistence
3. LangChain 官方文档 — Interrupts (Human-in-the-loop)：https://docs.langchain.com/oss/python/langgraph/interrupts
4. LangChain 官方博客 — LangGraph：https://www.langchain.com/blog/langgraph
5. Mem0 技术博客 — LangGraph Tutorial: Build AI Agents with Memory：https://mem0.ai/blog/langgraph-tutorial-build-advanced-ai-agents

---

## Jev 报告质量评估

> 本节由程序在报告生成后追加，不属于研究正文。

- 状态：completed
- 模式：Shadow（仅观察，不自动修改报告）
- Provider / 模型：jev / jev-1.13.0
- 综合质量分：**72.9%**
- 建议动作：修订报告
- 实际动作：仅记录结果

### 分项结果

| 指标 | 结果 | 置信度 |
| --- | ---: | ---: |
| 回答相关概率 | 95.0% | — |
| 证据支持概率 | 57.0% | — |
| 引用充分概率 | 64.0% | — |
| 证据足够概率 | 73.0% | — |
| 继续研究概率 | 26.0% | — |
| 来源质量评分 | 2.82 / 3 | 82.0% |

### 调用统计

- 延迟：912 ms
- Token：输入 6106 / 输出 116
- 成本：Provider 未提供
- 评估时间：2026-09-23T02:22:51.478524+00:00
