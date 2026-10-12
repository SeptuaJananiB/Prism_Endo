export type RunSummary = {
  name: string; frames: number; fps: number | null; modified: number; source: string; surgery?: string | null;
  hi_pct?: number; high_pct?: number; bleeding_frames?: number | null; cvs_pct?: number | null;
};
export type Series = Record<string, (number | string | boolean | null)[]>;
export type RunDetail = {
  name: string;
  summary: {
    frames_processed: number;
    keypoint_source: string;
    tissue_model: boolean;
    device: string;
    runtime?: { processing_fps?: number };
    warnings?: string[];
    input?: string;
    source_fps?: number;
    stride?: number;
    risk_mode?: string;
    tissue_checkpoint?: string;
    bleeding_detector?: boolean;
    cvs_model?: boolean;
    instrument_model?: boolean;
    proximity_targets?: Record<string, { reliability?: string; note?: string }>;
  };
  series: Series;
  videos: { original: string | null; annotated: string | null };
};
export type TissueClass = { name: string; reliability: string; dice: number | null };
export type Meta = {
  surgery_types: string[];
  inputs: string[];
  tissues?: Record<string, TissueClass[]>;
  defaults?: { targets: string[]; show_all_tissue: boolean; mask_alpha: number };
};
export type Job = {
  id: string;
  status: "queued" | "loading" | "running" | "finished" | "failed" | "cancelled";
  cancel?: boolean;
  phase?: string;
  live?: boolean;
  end?: boolean;
  live_fps?: number;
  view_fps?: number;
  risk?: string;
  warning?: string;
  bleeding?: boolean;
  done: number;
  total: number | null;
  run?: string;
  source?: string;
  error?: string;
};

export type LiveTip = { dist: number | null; speed: number | null; approach: number | null; ttc: number | null; target: string | null };
export type LiveSample = { t: number; frame: number; risk: string; score: number | null; warning: string; bleeding: boolean; tips: Record<string, LiveTip> };
export type LiveData = {
  series: LiveSample[]; events: { t: number; kind: string; label: string }[]; risk: string; fps: number;
  info: { risk_score?: number | null; bleeding?: boolean; cvs?: boolean; explanation?: string; cvs_c?: (number | null)[] };
};

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      msg = (await r.json()).detail ?? msg;
    } catch {
      /* not JSON */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

export const api = {
  meta: () => j<Meta>("/api/meta"),
  runs: () => j<RunSummary[]>("/api/runs"),
  run: (name: string) => j<RunDetail>(`/api/runs/${encodeURIComponent(name)}`),
  frameUrl: (name: string, i: number) => `/api/runs/${encodeURIComponent(name)}/frame/${i}`,
  start: (body: object) =>
    j<Job>("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  job: (id: string) => j<Job>(`/api/jobs/${id}`),
  llmStatus: () => j<{ configured: boolean; model: string }>("/api/llm/status"),
  explain: (name: string, refresh = false) =>
    j<{ text: string; model: string; created: number; cached: boolean }>(
      `/api/runs/${encodeURIComponent(name)}/explain${refresh ? "?refresh=true" : ""}`, { method: "POST" }),
  active: () => j<Job | Record<string, never>>("/api/active"),
  finish: (id: string) => j<Job>(`/api/jobs/${id}/finish`, { method: "POST" }),
  cameras: () => j<{ cameras: number[] }>("/api/cameras"),
  liveUrl: (id: string) => `/api/live/${id}.mjpg`,
  liveData: (id: string, since = 0) => j<LiveData>(`/api/live/${id}/data?since=${since}`),
  cancel: (id: string) => j<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  flags: (name: string) => j<{ t: number; frame: number; kind: "agree" | "wrong" | "important"; note: string }[]>(`/api/runs/${encodeURIComponent(name)}/flags`),
  saveFlags: (name: string, flags: object[]) =>
    j<{ saved: number }>(`/api/runs/${encodeURIComponent(name)}/flags`, { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(flags) }),
  upload: (file: File, onProgress: (p: number) => void) =>
    new Promise<{ source: string }>((resolve, reject) => {
      const x = new XMLHttpRequest();
      const fd = new FormData();
      fd.append("file", file);
      x.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      x.onload = () => (x.status < 300 ? resolve(JSON.parse(x.responseText)) : reject(new Error(JSON.parse(x.responseText || "{}").detail ?? x.statusText)));
      x.onerror = () => reject(new Error("upload failed"));
      x.open("POST", "/api/upload");
      x.send(fd);
    }),
};
