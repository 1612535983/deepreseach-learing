# 示例报告：LangChain Agent 如何工作？

> 这是一个经过压缩的展示样例，用来说明 DeepResearch Agent 的输出结构。
> 实际运行时，报告内容、来源和长度由研究问题及模型决定。

## 结论摘要

LangChain Agent 可以理解为一个由模型驱动的工具调用循环：模型先读取当前消息和状态，
判断是否需要调用工具；工具执行结果作为 `ToolMessage` 返回后，模型继续判断下一步，
直到任务完成。LangGraph 为这一循环补充了状态管理、持久化、流式输出和可恢复执行能力。

## 核心执行链

1. 用户提交自然语言问题。
2. Agent 创建结构化研究计划。
3. 模型选择搜索或网页读取工具。
4. Middleware 把工具结果整理为来源和证据。
5. 质量规则检查计划、来源、正文读取和报告是否完整。
6. Agent 生成带来源链接的 Markdown 报告。

## 关键组件

| 组件 | 解决的问题 |
|---|---|
| Model | 判断下一步动作并生成内容 |
| Tools | 搜索网页、读取正文、更新计划和提交报告 |
| State | 保存消息、计划、来源、证据和最终报告 |
| Middleware | 在模型和工具生命周期中执行证据收集、治理和检查 |
| Checkpointer | 保存 Graph 的运行位置和 State，使中断任务可以恢复 |

## 参考来源

- [LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)

---

生成同类报告：

```bash
uv run deepresearch run "LangChain Agent 如何工作？" \
  --stream \
  --show-trace \
  --output reports/langchain-agent.md
```
