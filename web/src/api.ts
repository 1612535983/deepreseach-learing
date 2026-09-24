export type RunStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "interrupted";

export interface RunAccepted {
  thread_id: string;
  status: RunStatus;
}

export interface Skill {
  skill_id: string;
  name: string;
  description: string;
  version: number;
  origin: string;
  tags: string[];
  allowed_tools: string[];
  total_selections: number;
  completion_rate: number;
}

export interface PlanStep {
  step_id: string;
  title: string;
  status: string;
}

export interface ResearchPlan {
  goal: string;
  steps: PlanStep[];
}

export interface Source {
  title: string;
  url: string;
  snippet: string;
  query: string;
}

export interface SelectedSkill {
  skill_id: string;
  name: string;
  score: number;
  reason: string;
  forced: boolean;
}

export interface Evaluation {
  status: string;
  mode: string;
  composite_score: number | null;
  recommended_action: string | null;
  runtime_action: string | null;
  gate_attempts: number;
  provider: string | null;
  model: string | null;
  latency_ms: number;
  cost_usd: number | null;
  last_error: string | null;
  notes: string[];
  answers: Record<string, Record<string, unknown>>;
}

export interface Governance {
  current_tokens: number;
  window_tokens: number;
  utilization_ratio: number;
  model_call_count: number;
  input_tokens: number;
  output_tokens: number;
  finalization_active: boolean;
  finalization_reason: string | null;
}

export interface RunDetail {
  thread_id: string;
  question: string;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  error: string | null;
  current_step_id: string | null;
  research_gaps: string[];
  plan: ResearchPlan | null;
  progress: {
    search_count: number;
    page_read_count: number;
    source_count: number;
    evidence_count: number;
    reflection_attempts: number;
  };
  sources: Source[];
  selected_skills: SelectedSkill[];
  evaluation: Evaluation;
  governance: Governance;
  final_report: string | null;
  final_report_source_urls: string[];
}

export interface ResearchEvent {
  id: number;
  event_type: string;
  message: string;
  data: Record<string, unknown>;
  node: string | null;
  created_at: string;
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = `请求失败（${response.status}）`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      // Keep the HTTP status fallback when the response is not JSON.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function listSkills(): Promise<Skill[]> {
  return apiRequest<Skill[]>("/api/skills");
}

export function createRun(
  question: string,
  skillOverrides: string[],
): Promise<RunAccepted> {
  return apiRequest<RunAccepted>("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      skill_overrides: skillOverrides,
    }),
  });
}

export function getRun(threadId: string): Promise<RunDetail> {
  return apiRequest<RunDetail>(`/api/runs/${encodeURIComponent(threadId)}`);
}

export function resumeRun(threadId: string): Promise<RunAccepted> {
  return apiRequest<RunAccepted>(
    `/api/runs/${encodeURIComponent(threadId)}/resume`,
    { method: "POST" },
  );
}

const EVENT_TYPES = [
  "run_started",
  "tool_requested",
  "tool_completed",
  "plan_updated",
  "search_completed",
  "page_read_completed",
  "reflection",
  "skill_selected",
  "skill_injected",
  "skill_metrics",
  "skill_evaluated",
  "report_created",
  "report_evaluated",
  "finalization",
  "run_completed",
  "run_failed",
];

export function subscribeToRun(
  threadId: string,
  onEvent: (event: ResearchEvent) => void,
  onConnectionError: () => void,
): () => void {
  const source = new EventSource(
    `/api/runs/${encodeURIComponent(threadId)}/events`,
  );
  const listener = (message: MessageEvent<string>) => {
    try {
      onEvent(JSON.parse(message.data) as ResearchEvent);
    } catch {
      onConnectionError();
    }
  };
  EVENT_TYPES.forEach((type) => source.addEventListener(type, listener));
  source.onerror = onConnectionError;
  return () => {
    EVENT_TYPES.forEach((type) => source.removeEventListener(type, listener));
    source.close();
  };
}
