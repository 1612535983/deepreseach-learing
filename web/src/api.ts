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
