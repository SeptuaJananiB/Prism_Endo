import { useEffect, useMemo, useState } from "react";
import { api, type LiveData, type LiveSample } from "../lib/api";
import { LineChart, RISK_VAR } from "./Charts";

type View = "distance" | "speed" | "ttc" | "risk";
const VIEWS: { id: View; label: string }[] = [
  { id: "distance", label: "Distance" }, { id: "speed", label: "Tip speed" }, { id: "ttc", label: "Time to contact" }, { id: "risk", label: "Risk" }];
const tipName = (n: string) => n.replace(/(lhook|grasper)_tip(_2)?/g, (_m, k, two) => `${k === "lhook" ? "L-hook" : "Grasper"} tip${two ? " 2" : ""}`);
const nice = (s?: string | null) => (s ?? "").replace(/_/g, " ");
const fmt = (v: number | null | undefined, d = 0) => (v == null ? "–" : v.toFixed(d));

/** Live session: per-tip readouts, rolling charts (last 60 s) and event log, updated twice a second. */
export function LiveInsights({ jobId }: { jobId: string }) {
  const [samples, setSamples] = useState<LiveSample[]>([]);
  const [data, setData] = useState<LiveData | null>(null);
  const [view, setView] = useState<View>("distance");

  useEffect(() => {
    let alive = true, since = -1, timer = 0;
    const tick = async () => {
      try {
        const d = await api.liveData(jobId, since);
        if (!alive) return;
        if (d.series.length) since = d.series[d.series.length - 1].t;
        setData(d);
        if (d.series.length) setSamples((prev) => [...prev, ...d.series].slice(-600));
      } catch { /* session not started yet or just ended */ }
      if (alive) timer = window.setTimeout(tick, 500);
    };
    tick();
    return () => { alive = false; window.clearTimeout(timer); };
  }, [jobId]);

  const t = useMemo(() => samples.map((s) => s.t), [samples]);
  const names = useMemo(() => Array.from(new Set(samples.flatMap((s) => Object.keys(s.tips)))), [samples]);
  const lines = useMemo(() => {
    if (view === "risk") return [{ name: "Risk score", values: samples.map((s) => s.score) }];
    const key = view === "distance" ? "dist" : view === "speed" ? "speed" : "ttc";
    return names.map((n) => ({ name: tipName(n), values: samples.map((s) => s.tips[n]?.[key] ?? null) }));
  }, [samples, names, view]);
  const unit = view === "distance" ? " px" : view === "speed" ? " px/s" : view === "ttc" ? " s" : "";
  const last = samples[samples.length - 1];
  const info = data?.info ?? {};
  const events = [...(data?.events ?? [])].reverse();

  return (
    <div className="grid border-t border-line @4xl:grid-cols-[minmax(0,1fr)_320px]">
      <div className="min-w-0 border-line p-4 @4xl:border-r">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h3 className="m-0 text-[13px] font-semibold">Live signals <span className="font-normal text-muted">· last 60 s</span></h3>
          <div role="tablist" aria-label="Live signal" className="flex gap-1 rounded-lg bg-[var(--surface-2)] p-0.5">
            {VIEWS.map((v) => (
              <button key={v.id} role="tab" aria-selected={view === v.id} onClick={() => setView(v.id)}
                className={`h-7 rounded-md px-2.5 text-xs ${view === v.id ? "bg-[var(--surface)] font-medium shadow-sm" : "text-muted"}`}>{v.label}</button>
            ))}
          </div>
        </div>
        {samples.length > 1
          ? <LineChart key={view} t={t} lines={lines} unit={unit} height={200} colorBy={view === "risk" ? samples.map((s) => s.risk) : undefined} />
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
