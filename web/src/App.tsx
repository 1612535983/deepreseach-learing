import { FormEvent, useEffect, useMemo, useState } from "react";

import { createRun, listSkills, Skill } from "./api";

const EXAMPLE_QUESTIONS = [
  "解释 JEV 如何评估研究报告质量，并说明 Gate 为什么不会无限回跳",
  "比较 ReAct Agent 与基于计划的研究 Agent，它们分别适合什么任务？",
  "研究长期记忆与 Skill 在 Agent 系统中的职责区别",
];

function App() {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [question, setQuestion] = useState("");
  const [selectedSkills, setSelectedSkills] = useState<string[]>([]);
  const [loadingSkills, setLoadingSkills] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeThread = useMemo(
    () => new URLSearchParams(window.location.search).get("run"),
    [],
  );

  useEffect(() => {
    listSkills()
      .then(setSkills)
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "Skill 加载失败");
      })
      .finally(() => setLoadingSkills(false));
  }, []);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!question.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const run = await createRun(question, selectedSkills);
      window.location.assign(`/?run=${encodeURIComponent(run.thread_id)}`);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "创建任务失败");
      setSubmitting(false);
    }
  }

  function toggleSkill(name: string) {
    setSelectedSkills((current) =>
      current.includes(name)
        ? current.filter((item) => item !== name)
        : [...current, name],
    );
  }

  if (activeThread) {
    return (
      <Shell>
        <main className="placeholder-view">
          <p className="eyebrow">RESEARCH RUN</p>
          <h1>研究任务已创建</h1>
          <p className="thread-code">{activeThread}</p>
          <p className="muted">实时运行视图正在载入。</p>
          <a className="text-link" href="/">
            ← 发起新的研究
          </a>
        </main>
      </Shell>
    );
  }

  return (
    <Shell>
      <main className="home-grid">
        <section className="hero-copy">
          <p className="eyebrow">AUDITABLE RESEARCH AGENT</p>
          <h1>
            把一个问题，
            <br />
            变成可核验的研究。
          </h1>
          <p className="lede">
            自动规划、搜索和阅读网页，持续整理证据，并生成带来源与 JEV
            质量评估的 Markdown 报告。
          </p>
          <div className="capability-row" aria-label="核心能力">
            <span>计划</span>
            <span>证据</span>
            <span>Checkpoint</span>
            <span>JEV</span>
          </div>
        </section>

        <section className="research-card">
          <form onSubmit={submit}>
            <div className="field-heading">
              <div>
                <p className="step-number">01</p>
                <h2>你想研究什么？</h2>
              </div>
              <span className="required-label">必填</span>
            </div>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="输入一个需要搜索、核验和综合分析的开放问题……"
              maxLength={10_000}
              rows={7}
              autoFocus
            />

            <div className="example-list">
              {EXAMPLE_QUESTIONS.map((example) => (
                <button
                  type="button"
                  className="example-button"
                  key={example}
                  onClick={() => setQuestion(example)}
                >
                  {example}
                </button>
              ))}
            </div>

            <div className="field-heading skill-heading">
              <div>
                <p className="step-number">02</p>
                <h2>指定研究方法</h2>
              </div>
              <span className="optional-label">可选</span>
            </div>
            <p className="field-help">
              不选择时由系统自动匹配；手动选择会强制使用对应 Skill。
            </p>
            <div className="skill-list">
              {loadingSkills && <span className="muted">正在读取 Skill…</span>}
              {!loadingSkills && skills.length === 0 && (
                <span className="muted">Skill 系统未启用，将使用默认研究流程。</span>
              )}
              {skills.map((skill) => {
                const selected = selectedSkills.includes(skill.name);
                return (
                  <button
                    type="button"
                    className={`skill-chip${selected ? " selected" : ""}`}
                    aria-pressed={selected}
                    key={skill.skill_id}
                    onClick={() => toggleSkill(skill.name)}
                    title={skill.description}
                  >
                    <span>{skill.name}</span>
                    <small>v{skill.version}</small>
                  </button>
                );
              })}
            </div>

            {error && <p className="error-message">{error}</p>}
            <button
              className="primary-button"
              type="submit"
              disabled={!question.trim() || submitting}
            >
              <span>{submitting ? "正在创建任务…" : "开始深度研究"}</span>
              <span aria-hidden="true">↗</span>
            </button>
          </form>
        </section>
      </main>
    </Shell>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-mark">DR</span>
          <span>
            <strong>DeepResearch</strong>
            <small>WORKBENCH</small>
          </span>
        </a>
        <div className="system-state">
          <span className="status-dot" />
          本地研究系统
        </div>
      </header>
      {children}
      <footer>
        <span>DeepResearch Agent</span>
        <span>可恢复 · 可审计 · 有边界</span>
      </footer>
    </div>
  );
}

export default App;
