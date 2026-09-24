import { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

import {
  getRun,
  PlanStep,
  ResearchEvent,
  resumeRun,
  RunDetail,
  subscribeToRun,
} from "./api";

const STATUS_LABELS: Record<string, string> = {
  queued: "等待开始",
  running: "研究进行中",
  completed: "研究已完成",
  failed: "运行失败",
  interrupted: "等待恢复",
};

const EVENT_LABELS: Record<string, string> = {
  run_started: "启动",
  tool_requested: "工具",
  tool_completed: "工具",
  plan_updated: "计划",
  search_completed: "搜索",
  page_read_completed: "阅读",
  reflection: "反思",
  skill_selected: "Skill",
  skill_injected: "Skill",
  skill_metrics: "Skill",
  skill_evaluated: "评估",
  report_created: "报告",
  report_evaluated: "JEV",
  finalization: "收尾",
  run_completed: "完成",
  run_failed: "失败",
};

function RunView({ threadId }: { threadId: string }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [events, setEvents] = useState<ResearchEvent[]>([]);
  const [livePlan, setLivePlan] = useState<PlanStep[]>([]);
  const [connection, setConnection] = useState<"connecting" | "live" | "closed">(
    "connecting",
  );
  const [error, setError] = useState<string | null>(null);
  const [resuming, setResuming] = useState(false);
  const [copied, setCopied] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const detail = await getRun(threadId);
      setRun(detail);
      if (detail.plan) setLivePlan(detail.plan.steps);
      setError(null);
      return detail;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务加载失败");
      return null;
    }
  }, [threadId]);

  useEffect(() => {
    let closed = false;
    let unsubscribe: (() => void) | null = null;

    void refresh().then((detail) => {
      if (
        closed ||
        !detail ||
        ["completed", "failed", "interrupted"].includes(detail.status)
      ) {
        setConnection("closed");
        return;
      }
      unsubscribe = subscribeToRun(
        threadId,
        (event) => {
          setConnection("live");
          setEvents((current) =>
            current.some((item) => item.id === event.id)
              ? current
              : [...current, event],
          );
          if (event.event_type === "plan_updated") {
            const steps = event.data.steps;
            if (Array.isArray(steps)) setLivePlan(steps as PlanStep[]);
          }
          if (event.event_type === "run_started") {
            setRun((current) =>
              current ? { ...current, status: "running" } : current,
            );
          }
          if (["run_completed", "run_failed"].includes(event.event_type)) {
            setConnection("closed");
            unsubscribe?.();
            window.setTimeout(() => void refresh(), 80);
          }
        },
        () => setConnection("connecting"),
      );
    });

    return () => {
      closed = true;
      unsubscribe?.();
    };
  }, [refresh, threadId]);

  useEffect(() => {
    if (!run || !["queued", "running"].includes(run.status)) return;
    const timer = window.setInterval(() => void refresh(), 2500);
    return () => window.clearInterval(timer);
  }, [refresh, run?.status]);

  const liveCounts = useMemo(
    () => ({
      searches: events.filter((event) => event.event_type === "search_completed")
        .length,
      pages: events.filter((event) => event.event_type === "page_read_completed")
        .length,
      reflections: events.filter((event) => event.event_type === "reflection")
        .length,
    }),
    [events],
  );

  const evaluationEvent = [...events]
    .reverse()
    .find((event) => event.event_type === "report_evaluated");
  const liveScore = evaluationEvent?.data.composite_score;
  const score =
    run?.evaluation.composite_score ??
    (typeof liveScore === "number" ? liveScore : null);

  async function resume() {
    setResuming(true);
    setError(null);
    try {
      await resumeRun(threadId);
      window.location.reload();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "恢复任务失败");
      setResuming(false);
    }
  }

  async function copyReport() {
    if (!run?.final_report) return;
    await navigator.clipboard.writeText(run.final_report);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  function downloadReport() {
    if (!run?.final_report) return;
    const blob = new Blob([run.final_report], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${threadId}.md`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  if (!run && !error) {
    return (
      <main className="run-loading">
        <span className="loader" />
        <p>正在连接研究任务…</p>
      </main>
    );
  }

  if (!run) {
    return (
      <main className="run-loading">
        <p className="error-message">{error}</p>
        <a className="text-link" href="/">
          ← 返回新建研究
        </a>
      </main>
    );
  }

  const active = ["queued", "running"].includes(run.status);
  const plan = livePlan.length ? livePlan : run.plan?.steps ?? [];
  const selectedSkillNames =
    run.selected_skills.map((skill) => skill.name).length > 0
      ? run.selected_skills.map((skill) => skill.name)
      : (([...events]
          .reverse()
          .find((event) => event.event_type === "skill_selected")?.data.names as
          | string[]
          | undefined) ?? []);

  return (
    <main className="run-page">
      <nav className="run-nav">
        <a className="text-link" href="/">
          ← 新建研究
        </a>
        <span className="thread-code">{threadId}</span>
      </nav>

      <section className="run-heading">
        <div>
          <div className="run-state-row">
            <span className={`run-state ${run.status}`}>
              {active && <span className="pulse" />}
              {STATUS_LABELS[run.status]}
            </span>
            {active && (
              <span className="connection-state">
                {connection === "live" ? "实时连接" : "正在连接"}
              </span>
            )}
          </div>
          <h1>{run.question}</h1>
        </div>
        {(["failed", "interrupted"] as string[]).includes(run.status) && (
          <button className="resume-button" onClick={resume} disabled={resuming}>
            {resuming ? "正在恢复…" : "从 Checkpoint 恢复"}
          </button>
        )}
      </section>

      {error && <p className="error-message run-error">{error}</p>}
      {run.error && <p className="error-message run-error">{run.error}</p>}

      <section className="metric-strip">
        <Metric
          label="搜索"
          value={Math.max(run.progress.search_count, liveCounts.searches)}
        />
        <Metric
          label="网页阅读"
          value={Math.max(run.progress.page_read_count, liveCounts.pages)}
        />
        <Metric label="来源" value={run.progress.source_count} />
        <Metric label="证据" value={run.progress.evidence_count} />
        <Metric
          label="JEV 质量"
          value={score === null ? "—" : `${Math.round(score * 100)}%`}
          accent={score !== null}
        />
      </section>

      <div className="run-layout">
        <div className="run-main">
          {run.final_report ? (
            <section className="report-panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">FINAL REPORT</p>
                  <h2>研究报告</h2>
                </div>
                <div className="report-actions">
                  <button onClick={copyReport}>{copied ? "已复制" : "复制"}</button>
                  <button onClick={downloadReport}>下载 .md</button>
                </div>
              </div>
              <article className="markdown-body">
                <ReactMarkdown
                  components={{
                    a: ({ children, ...props }) => (
                      <a {...props} target="_blank" rel="noreferrer">
                        {children}
                      </a>
                    ),
                  }}
                >
                  {run.final_report}
                </ReactMarkdown>
              </article>
            </section>
          ) : (
            <section className="timeline-panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">LIVE TRACE</p>
                  <h2>研究正在发生</h2>
                </div>
                {active && <span className="live-badge">LIVE</span>}
              </div>
              {events.length === 0 ? (
                <div className="empty-timeline">
                  <span className="loader" />
                  <p>Agent 正在准备研究环境…</p>
                </div>
              ) : (
                <ol className="event-list">
                  {events.map((event) => (
                    <li key={event.id} className={event.event_type}>
                      <div className="event-index">{String(event.id).padStart(2, "0")}</div>
                      <div className="event-copy">
                        <span>{EVENT_LABELS[event.event_type] ?? "事件"}</span>
                        <p>{event.message}</p>
                      </div>
                      <time>
                        {new Date(event.created_at).toLocaleTimeString("zh-CN", {
                          hour: "2-digit",
                          minute: "2-digit",
                          second: "2-digit",
                        })}
                      </time>
                    </li>
                  ))}
                </ol>
              )}
            </section>
          )}

          {run.sources.length > 0 && (
            <section className="sources-panel">
              <div className="panel-header">
                <div>
                  <p className="eyebrow">SOURCES</p>
                  <h2>引用来源</h2>
                </div>
                <span className="source-count">{run.sources.length}</span>
              </div>
              <div className="source-grid">
                {run.sources.map((source, index) => (
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                    className="source-card"
                    key={source.url}
                  >
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <h3>{source.title || source.url}</h3>
                    {source.snippet && <p>{source.snippet}</p>}
                    <small>{new URL(source.url).hostname}</small>
                  </a>
                ))}
              </div>
            </section>
          )}
        </div>

        <aside className="run-sidebar">
          <section className="side-panel plan-panel">
            <p className="side-label">研究计划</p>
            {plan.length === 0 ? (
              <p className="muted side-empty">等待 Agent 创建计划…</p>
            ) : (
              <ol>
                {plan.map((step, index) => (
                  <li className={step.status} key={step.step_id}>
                    <span>{step.status === "completed" ? "✓" : index + 1}</span>
                    <div>
                      <p>{step.title}</p>
                      <small>{step.status.replace("_", " ")}</small>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </section>

          <section className="side-panel">
            <p className="side-label">运行方法</p>
            {selectedSkillNames.length ? (
              <div className="selected-skills">
                {selectedSkillNames.map((name) => (
                  <span key={name}>{name}</span>
                ))}
              </div>
            ) : (
              <p className="muted side-empty">自动选择中</p>
            )}
          </section>

          <section className="side-panel jev-panel">
            <div className="side-title-row">
              <p className="side-label">JEV 报告评估</p>
              <span>{run.evaluation.mode.toUpperCase()}</span>
            </div>
            {score === null ? (
              <p className="muted side-empty">报告生成后执行概率评估</p>
            ) : (
              <>
                <div className="score-row">
                  <strong>{Math.round(score * 100)}</strong>
                  <span>/ 100</span>
                </div>
                <div className="score-track">
                  <span style={{ width: `${score * 100}%` }} />
                </div>
                <dl>
                  <div>
                    <dt>建议动作</dt>
                    <dd>{run.evaluation.recommended_action ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>运行决策</dt>
                    <dd>{run.evaluation.runtime_action ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>Gate 回跳</dt>
                    <dd>{run.evaluation.gate_attempts}</dd>
                  </div>
                </dl>
              </>
            )}
          </section>

          <section className="side-panel diagnostics-panel">
            <p className="side-label">运行诊断</p>
            <dl>
              <div>
                <dt>模型调用</dt>
                <dd>{run.governance.model_call_count}</dd>
              </div>
              <div>
                <dt>输入 Token</dt>
                <dd>{run.governance.input_tokens.toLocaleString()}</dd>
              </div>
              <div>
                <dt>输出 Token</dt>
                <dd>{run.governance.output_tokens.toLocaleString()}</dd>
              </div>
              <div>
                <dt>上下文占用</dt>
                <dd>{Math.round(run.governance.utilization_ratio * 100)}%</dd>
              </div>
            </dl>
          </section>
        </aside>
      </div>
    </main>
  );
}

function Metric({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string | number;
  accent?: boolean;
}) {
  return (
    <div className={accent ? "metric accent" : "metric"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default RunView;
