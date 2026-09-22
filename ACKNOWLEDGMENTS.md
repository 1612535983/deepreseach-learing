# Acknowledgments

DeepResearch Agent is an independent, learning-oriented engineering project.
Its project-specific state model, tools, middleware composition, P1/P4/P5
governance pipeline, and Markdown/BM25 memory implementation live in this
repository and are covered by its tests.

## Runtime foundations

- [LangChain](https://github.com/langchain-ai/langchain) — model and tool
  interfaces, `create_agent`, middleware APIs, and integrations (MIT License).
- [LangGraph](https://github.com/langchain-ai/langgraph) — stateful execution,
  streaming, and checkpointing primitives (MIT License).

## Architecture references

The following public projects were studied to compare design choices rather
than treated as one-to-one source templates:

- [DeerFlow](https://github.com/bytedance/deer-flow) — long-horizon harness
  boundaries, context engineering, external working memory, and persistent
  memory as a runtime capability (MIT License).
- [Open Deep Research](https://github.com/langchain-ai/open_deep_research) —
  plan-and-research workflows, report generation, and evaluation structure
  (MIT License).
- [Poirot](https://github.com/HezaoHezao/poirot) — one reference among several
  for middleware-oriented agent organization, context governance, and memory
  layering (MIT License).

Reference does not imply affiliation, endorsement, or source compatibility.
All third-party names and trademarks belong to their respective owners. See
each dependency or upstream repository for its complete license terms.
