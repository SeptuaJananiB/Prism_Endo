import { useEffect, useMemo, useState } from "react";
import { api, type LiveData, type LiveSample } from "../lib/api";
import { LineChart, RISK_VAR } from "./Charts";

type View = "distance" | "speed" | "ttc" | "risk";
const VIEWS: { id: View; label: string }[] = [
  { id: "risk", label: "Risk" }, { id: "distance", label: "Distance" }, { id: "speed", label: "Tip speed" }, { id: "ttc", label: "Time to contact" }];
const tipName = (n: string) => n.replace(/(lhook|grasper)_tip(_2)?/g, (_m, k, two) => `${k === "lhook" ? "L-hook" : "Grasper"} tip${two ? " 2" : ""}`);
const nice = (s?: string | null) => (s ?? "").replace(/_/g, " ");
const fmt = (v: number | null | undefined, d = 0) => (v == null ? "–" : v.toFixed(d));

/** Live session: per-tip readouts, charts of the whole session, per-second table and event log, one sample per tracked frame, polled 4x a second. */
export function LiveInsights({ jobId }: { jobId: string }) {
  const [samples, setSamples] = useState<LiveSample[]>([]);
  const [data, setData] = useState<LiveData | null>(null);
  const [view, setView] = useState<View>("risk");

  useEffect(() => {
    let alive = true, since = 0, timer = 0;
    const tick = async () => {
      try {
        const d = await api.liveData(jobId, since);
        if (!alive) return;
        if (d.series.length) since = d.series[d.series.length - 1].frame;
        setData(d);
        if (d.series.length) setSamples((prev) => prev.concat(d.series));
      } catch { /* session not started yet or just ended */ }
      if (alive) timer = window.setTimeout(tick, 250);
    };
    tick();
    return () => { alive = false; window.clearTimeout(timer); };
  }, [jobId]);

  // charts draw at most ~2000 points (every frame is kept; long sessions are thinned only for drawing)
  const drawn = useMemo(() => { const k = Math.ceil(samples.length / 2000) || 1; return k === 1 ? samples : samples.filter((_, i) => i % k === 0 || i === samples.length - 1); }, [samples]);
  const t = useMemo(() => drawn.map((s) => s.t), [drawn]);
  const names = useMemo(() => Array.from(new Set(samples.flatMap((s) => Object.keys(s.tips)))), [samples]);
  const lines = useMemo(() => {
    if (view === "risk") return [{ name: "Risk score", values: bridge(drawn, drawn.map((s) => s.score)) }];
    const key = view === "distance" ? "dist" : view === "speed" ? "speed" : "ttc";
    return names.map((n) => ({ name: tipName(n), values: bridge(drawn, drawn.map((s) => s.tips[n]?.[key] ?? null)) }));
  }, [drawn, names, view]);
  const seconds = useMemo(() => perSecond(samples, names), [samples, names]);
  const unit = view === "distance" ? " px" : view === "speed" ? " px/s" : view === "ttc" ? " s" : "";
  const last = samples[samples.length - 1];
  const info = data?.info ?? {};
  const events = [...(data?.events ?? [])].reverse();

  return (
    <div className="grid border-t border-line @4xl:grid-cols-[minmax(0,1fr)_320px]">
      <div className="min-w-0 border-line p-4 @4xl:border-r">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h3 className="m-0 text-[13px] font-semibold">Live signals <span className="font-normal text-muted tabular">· every frame · whole session · {(last?.t ?? 0).toFixed(1)} s · frame {last?.frame ?? 0}</span></h3>
          <div role="tablist" aria-label="Live signal" className="flex gap-1 rounded-lg bg-[var(--surface-2)] p-0.5">
            {VIEWS.map((v) => (
              <button key={v.id} role="tab" aria-selected={view === v.id} onClick={() => setView(v.id)}
                className={`h-7 rounded-md px-2.5 text-xs ${view === v.id ? "bg-[var(--surface)] font-medium shadow-sm" : "text-muted"}`}>{v.label}</button>
            ))}
          </div>
        </div>
        {samples.length > 1 && !lines.some((l) => l.values.some((v) => v != null))
          ? <div className="grid h-[200px] place-items-center px-6 text-center text-[13px] text-muted">
              No instrument tip detected yet — distance, speed, contact time and risk are measured from the tip.
              Bring an instrument (grasper or L-hook) into view.</div>
          : samples.length > 1
          ? <LineChart key={view} t={t} lines={lines} unit={unit} height={200} colorBy={view === "risk" ? drawn.map((s) => s.risk) : undefined} />
          : <div className="grid h-[200px] place-items-center text-[13px] text-muted">Waiting for the first tracked frames…</div>}
        <div className="mt-4 grid grid-cols-1 gap-3 @xl:grid-cols-2">
          {names.length === 0 && <p className="m-0 text-[13px] text-muted">No instrument tip in view.</p>}
          {names.map((n) => {
            const s = last?.tips[n];
            const warn = s?.ttc != null && s.ttc <= 1;
            return (
              <div key={n} className={`rounded-lg border p-3 ${warn ? "border-err bg-[var(--err-bg)]" : "border-line"}`}>
                <div className="flex items-center justify-between text-[13px] font-medium"><span>{tipName(n)}</span>
                  <span className="text-xs text-muted">{s ? `near ${nice(s.target) || "–"}` : "not tracked"}</span></div>
                <dl className="m-0 mt-2 grid grid-cols-3 gap-2 text-xs">
                  <div><dt className="text-muted">Distance</dt><dd className="m-0 text-[15px] font-semibold tabular">{fmt(s?.dist)} <span className="text-xs font-normal text-muted">px</span></dd></div>
                  <div><dt className="text-muted">Speed</dt><dd className="m-0 text-[15px] font-semibold tabular">{fmt(s?.speed)} <span className="text-xs font-normal text-muted">px/s</span></dd></div>
                  <div><dt className="text-muted">Contact in</dt><dd className={`m-0 text-[15px] font-semibold tabular ${warn ? "text-err" : ""}`}>{s?.ttc == null ? "–" : `${s.ttc.toFixed(1)} s`}</dd></div>
                </dl>
              </div>
            );
          })}
        </div>
      </div>
      <div className="min-w-0 border-t border-line p-4 @4xl:col-span-2">
        <h3 className="m-0 mb-2 text-[13px] font-semibold">Every second <span className="font-normal text-muted tabular">· {seconds.length} s · newest first</span></h3>
        <div className="max-h-[320px] overflow-auto rounded-lg border border-line">
          <table className="w-full border-collapse text-left text-xs tabular">
            <thead className="sticky top-0 bg-[var(--surface-2)]"><tr>
              <th className="px-2 py-1.5 font-medium">Second</th><th className="px-2 py-1.5 font-medium">Frames</th><th className="px-2 py-1.5 font-medium">Risk (max)</th>
              {names.map((n) => <th key={n} className="px-2 py-1.5 font-medium">{tipName(n)}: min dist · max speed</th>)}
              <th className="px-2 py-1.5 font-medium">Min contact in</th><th className="px-2 py-1.5 font-medium">Bleeding</th><th className="px-2 py-1.5 font-medium">Warning</th>
            </tr></thead>
            <tbody>
              {seconds.length === 0 && <tr><td className="px-2 py-2 text-muted" colSpan={6 + names.length}>Waiting for the first second…</td></tr>}
              {seconds.map((r) => (
                <tr key={r.sec} className="border-t border-line">
                  <td className="px-2 py-1">{r.sec}–{r.sec + 1} s</td><td className="px-2 py-1">{r.frames}</td>
                  <td className="px-2 py-1"><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: RISK_VAR[r.risk] }} aria-hidden />{r.risk}</td>
                  {names.map((n) => <td key={n} className="px-2 py-1">{fmt(r.tips[n]?.dist)} px · {fmt(r.tips[n]?.speed)} px/s{r.tips[n]?.target ? ` · ${nice(r.tips[n].target)}` : ""}</td>)}
                  <td className={`px-2 py-1 ${r.ttc != null && r.ttc <= 1 ? "text-err" : ""}`}>{r.ttc == null ? "–" : `${r.ttc.toFixed(1)} s`}</td>
                  <td className="px-2 py-1">{r.bleeding ? <span className="text-err">yes</span> : "no"}</td>
                  <td className="px-2 py-1 text-err">{r.warning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <aside className="flex min-h-0 flex-col border-t border-line @4xl:border-t-0" aria-label="Live status and events">
        <dl className="m-0 grid grid-cols-2 gap-3 border-b border-line p-4 text-xs">
          <div><dt className="text-muted">Risk</dt><dd className="m-0 mt-1 flex items-center gap-1.5 text-[14px] font-semibold">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: RISK_VAR[data?.risk ?? "Unavailable"] }} aria-hidden />{data?.risk ?? "–"}
            {info.risk_score != null && <span className="font-normal text-muted tabular">{info.risk_score.toFixed(2)}</span>}</dd></div>
          <div><dt className="text-muted">Bleeding</dt><dd className="m-0 mt-1 text-[14px] font-semibold">{info.bleeding ? <span className="text-err">Flagged</span> : "Not flagged"}</dd></div>
          <div className="col-span-2"><dt className="text-muted">Critical View of Safety</dt><dd className="m-0 mt-1 text-[13px]">
            {info.cvs ? <span className="font-semibold">Reached</span> : "Not reached"}
            {info.cvs_c?.some((v) => v != null) && <span className="text-muted tabular"> · C1 {fmt(info.cvs_c[0], 2)} C2 {fmt(info.cvs_c[1], 2)} C3 {fmt(info.cvs_c[2], 2)}</span>}</dd></div>
          {info.explanation && <div className="col-span-2"><dt className="text-muted">Why</dt><dd className="m-0 mt-1 line-clamp-3 text-[13px]">{tipName(String(info.explanation))}</dd></div>}
        </dl>
        <h3 className="m-0 px-4 pt-3 text-[13px] font-semibold">Events <span className="font-normal text-muted tabular">· {events.length}</span></h3>
        <ol className="m-0 max-h-[260px] list-none overflow-y-auto p-2">
          {events.length === 0 && <li className="px-2 py-2 text-[13px] text-muted">No events yet.</li>}
          {events.map((e, k) => (
            <li key={`${e.t}-${k}`} className="flex items-baseline gap-2 rounded-md px-2 py-1.5 text-[13px]">
              <span className="w-12 shrink-0 text-xs text-muted tabular">{e.t.toFixed(1)} s</span>
              <span className={e.kind === "Contact warning" || e.kind === "Bleeding" ? "text-err" : ""}>{e.label}</span>
            </li>
          ))}
        </ol>
      </aside>
    </div>
  );
}

const RANK: Record<string, number> = { Unavailable: 0, Minimal: 1, Low: 2, Medium: 3, High: 4 };
type SecRow = { sec: number; frames: number; risk: string; ttc: number | null; bleeding: boolean; warning: string;
  tips: Record<string, { dist: number | null; speed: number | null; target: string | null }> };

/** One row per second of the session (newest first): worst risk, closest distance, fastest speed, soonest contact. */
function perSecond(samples: LiveSample[], names: string[]): SecRow[] {
  const rows = new Map<number, SecRow>();
  for (const s of samples) {
    const sec = Math.floor(s.t);
    let r = rows.get(sec);
    if (!r) { r = { sec, frames: 0, risk: "Unavailable", ttc: null, bleeding: false, warning: "", tips: {} }; rows.set(sec, r); }
    r.frames += 1;
    if ((RANK[s.risk] ?? 0) > (RANK[r.risk] ?? 0)) r.risk = s.risk;
    r.bleeding ||= s.bleeding;
    if (s.warning && !r.warning) r.warning = s.warning;
    for (const n of names) {
      const v = s.tips[n];
      if (!v) continue;
      const c = (r.tips[n] ??= { dist: null, speed: null, target: null });
      if (v.dist != null && (c.dist == null || v.dist < c.dist)) { c.dist = v.dist; c.target = v.target; }
      if (v.speed != null && (c.speed == null || v.speed > c.speed)) c.speed = v.speed;
      if (v.ttc != null && (r.ttc == null || v.ttc < r.ttc)) r.ttc = v.ttc;
    }
  }
  return [...rows.values()].reverse();
}

/** Tips drop out for single frames; hold the last value across gaps under 0.5 s so the line stays continuous. */
function bridge(samples: LiveSample[], values: (number | null)[]): (number | null)[] {
  let last: number | null = null, lastT = -Infinity;
  return values.map((v, i) => {
    if (v != null) { last = v; lastT = samples[i].t; return v; }
    return last != null && samples[i].t - lastT < 0.5 ? last : null;
  });
}
