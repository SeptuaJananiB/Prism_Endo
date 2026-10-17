import { useEffect, useMemo, useRef, useState } from "react";
import { api, type LiveData, type LiveSample } from "../lib/api";
import { LineChart, RISK_VAR } from "./Charts";
import { playBurst, unlockAudio, type Priority } from "../lib/alarm";

type View = "distance" | "speed" | "ttc" | "risk";
const VIEWS: { id: View; label: string }[] = [
  { id: "risk", label: "Risk" }, { id: "distance", label: "Distance" }, { id: "speed", label: "Tip speed" }, { id: "ttc", label: "Time to contact" }];
const tipName = (n: string) => n.replace(/(lhook|grasper)_tip(_2)?/g, (_m, k, two) => `${k === "lhook" ? "L-hook" : "Grasper"} tip${two ? " 2" : ""}`);
const nice = (s?: string | null) => (s ?? "").replace(/_/g, " ");
const fmt = (v: number | null | undefined, d = 0) => (v == null ? "–" : v.toFixed(d));

/** Live session: per-tip readouts, charts of the last 60 s (scroll, zoom and pan back through the whole session), per-second table and event log, one sample per tracked frame, polled 4x a second. */
export function LiveInsights({ jobId }: { jobId: string }) {
  const [samples, setSamples] = useState<LiveSample[]>([]);
  const [data, setData] = useState<LiveData | null>(null);
  const [view, setView] = useState<View>("risk");
  // chart window: last `span` seconds; `end` null = follow live, a number = scrolled back to that time
  const [span, setSpan] = useState(60);
  const [end, setEnd] = useState<number | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const drag = useRef<{ x: number; end: number } | null>(null);
  // audible alarm (IEC 60601-1-8 style): high = contact warning / tip in No-Go zone, medium = risk High
  const [sound, setSound] = useState(() => { try { return localStorage.getItem("prism-alarm") !== "off"; } catch { return true; } });
  const busyUntil = useRef(0);
  const lastMedium = useRef(0);

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

  const latest = samples[samples.length - 1];
  const alarm: Priority | null = !latest ? null
    : latest.warning || Object.values(latest.tips).some((v) => v.zone === "no_go") ? "high"
    : latest.risk === "High" ? "medium" : null;
  useEffect(() => {
    if (!sound || !alarm) return;
    const now = performance.now();
    if (now < busyUntil.current) return;                                   // a burst is still playing
    if (alarm === "medium" && now - lastMedium.current < 8000) return;     // medium repeats every 8 s
    const len = playBurst(alarm);
    busyUntil.current = now + len * 1000 + (alarm === "high" ? 400 : 0);  // high repeats back to back
    if (alarm === "medium") lastMedium.current = now;
  }, [alarm, sound, latest?.frame]);
  const toggleSound = () => {
    const on = !sound;
    if (on) unlockAudio();
    setSound(on);
    try { localStorage.setItem("prism-alarm", on ? "on" : "off"); } catch { /* private mode */ }
  };
  const total = samples.length ? samples[samples.length - 1].t : 0;
  const t0 = samples.length ? samples[0].t : 0;
  const winEnd = end == null ? total : Math.min(end, total);
  const winStart = Math.max(t0, winEnd - span);
  const clampEnd = (e: number) => (e >= total - 0.25 ? null : Math.max(t0 + Math.min(span, total - t0), e));
  // samples inside the window; wide windows are grouped into ~500 points keeping the WORST value of each group
  // (highest risk / speed, closest distance, soonest contact) so the chart stays readable and spikes are not lost
  const drawn = useMemo(() => {
    const lo = lowerBound(samples, winStart), hi = lowerBound(samples, winEnd + 1e-6);
    return worstPerBucket(samples.slice(lo, hi), 500);
  }, [samples, winStart, winEnd]);
  // wheel = zoom (5 s .. whole session), kept non-passive so the page does not scroll while zooming
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;  // plain wheel scrolls the page; Ctrl + wheel zooms the chart
      e.preventDefault();
      // proportional: a trackpad pinch sends many small events, a mouse wheel notch one large one
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      const f = Math.exp(Math.max(-60, Math.min(60, dy)) * 0.004);
      setSpan((sp) => Math.max(5, Math.min(Math.max(5, total - t0), sp * f)));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [total, t0]);
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
          <h3 className="m-0 text-[13px] font-semibold">Live signals <span className="font-normal text-muted tabular">· every frame · {end == null ? `last ${span < 60 ? span.toFixed(0) + " s" : (span / 60).toFixed(1) + " min"}` : "scrolled back"} · {(last?.t ?? 0).toFixed(1)} s · frame {last?.frame ?? 0}</span></h3>
          <button className={`btn h-7 px-2.5 text-xs ${sound && alarm ? "border-err text-err" : ""}`} onClick={toggleSound} aria-pressed={sound}
            title={sound ? "Alarm sound on: beeps for contact warnings, No-Go zone and High risk" : "Alarm sound off"}>
            {sound ? "🔔 Sound on" : "🔕 Sound off"}</button>
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
          ? <div ref={box} className="cursor-grab touch-none select-none active:cursor-grabbing" title="Drag to pan · Ctrl + scroll wheel to zoom"
              onPointerDown={(e) => { drag.current = { x: e.clientX, end: winEnd }; (e.target as Element).setPointerCapture?.(e.pointerId); }}
              onPointerMove={(e) => { const d = drag.current; if (!d || !box.current) return; setEnd(clampEnd(d.end - ((e.clientX - d.x) / box.current.clientWidth) * span)); }}
              onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
              {drawn.length > 1
                ? <LineChart key={view} t={t} lines={lines} unit={unit} height={200} colorBy={view === "risk" ? drawn.map((s) => s.risk) : undefined} />
                : <div className="grid h-[200px] place-items-center text-[13px] text-muted">No frames in this window.</div>}
            </div>
          : <div className="grid h-[200px] place-items-center text-[13px] text-muted">Waiting for the first tracked frames…</div>}
        {samples.length > 1 && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
            <button className="btn h-7 w-7 p-0" onClick={() => setSpan((sp) => Math.max(5, sp * 0.5))} aria-label="Zoom in" title="Zoom in">+</button>
            <button className="btn h-7 w-7 p-0" onClick={() => setSpan((sp) => Math.min(Math.max(5, total - t0), sp * 2))} aria-label="Zoom out" title="Zoom out">−</button>
            <span className="w-24 text-muted tabular">window {span < 60 ? `${span.toFixed(0)} s` : `${(span / 60).toFixed(1)} min`}</span>
            <label htmlFor="live-scroll" className="sr-only">Scroll through the session</label>
            <input id="live-scroll" type="range" className="min-w-[120px] flex-1" min={t0 + Math.min(span, total - t0)} max={total} step={0.1} value={winEnd}
              disabled={total - t0 <= span} onChange={(e) => setEnd(clampEnd(Number(e.target.value)))}
              aria-valuetext={`${winStart.toFixed(0)} to ${winEnd.toFixed(0)} seconds`} />
            <span className="text-muted tabular">{winStart.toFixed(0)}–{winEnd.toFixed(0)} s</span>
            <button className={`btn h-7 px-2.5 ${end == null ? "border-err text-err" : ""}`} onClick={() => { setEnd(null); setSpan(60); }}
              aria-pressed={end == null} aria-label="Follow live (last 60 s)" title="Follow the live feed (last 60 s)">● Live</button>
          </div>
        )}
        <div className="mt-4 grid grid-cols-1 gap-3 @xl:grid-cols-2">
          {names.length === 0 && <p className="m-0 text-[13px] text-muted">No instrument tip in view.</p>}
          {names.map((n) => {
            const s = last?.tips[n];
            const warn = s?.ttc != null && s.ttc <= 1;
            return (
              <div key={n} className={`rounded-lg border p-3 ${warn ? "border-err bg-[var(--err-bg)]" : "border-line"}`}>
                <div className="flex items-center justify-between text-[13px] font-medium"><span>{tipName(n)}</span>
                  <span className="flex items-center gap-1.5 text-xs text-muted">
                    {s?.zone === "no_go" && <span className="chip chip-err h-5 px-1.5 text-[11px]">No-Go zone</span>}
                    {s?.zone === "go" && <span className="chip h-5 px-1.5 text-[11px]" style={{ color: "var(--risk-minimal)", background: "var(--surface-2)" }}>Go zone</span>}
                    {s ? `near ${nice(s.target) || "–"}` : "not tracked"}</span></div>
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
          <div className="col-span-2"><dt className="text-muted">Suggested dissection zones</dt><dd className="m-0 mt-1 text-[13px]">
            {info.zones ? <><span style={{ color: "var(--risk-minimal)" }}>■</span> Go {fmt((info.go_frac ?? 0) * 100, 1)}% · <span className="text-err">■</span> No-Go {fmt((info.nogo_frac ?? 0) * 100, 1)}%
              <span className="text-muted"> · estimate</span></> : <span className="text-muted">Shown when the gallbladder is visible (gallbladder model)</span>}</dd></div>
          {info.explanation && <div className="col-span-2"><dt className="text-muted">Why</dt><dd className="m-0 mt-1 line-clamp-3 text-[13px]">{tipName(String(info.explanation))}</dd></div>}
        </dl>
        <h3 className="m-0 px-4 pt-3 text-[13px] font-semibold">Events <span className="font-normal text-muted tabular">· {events.length}</span></h3>
        <ol className="m-0 max-h-[260px] list-none overflow-y-auto p-2">
          {events.length === 0 && <li className="px-2 py-2 text-[13px] text-muted">No events yet.</li>}
          {events.map((e, k) => (
            <li key={`${e.t}-${k}`} className="flex items-baseline gap-2 rounded-md px-2 py-1.5 text-[13px]">
              <span className="w-12 shrink-0 text-xs text-muted tabular">{e.t.toFixed(1)} s</span>
              <span className={e.kind === "Contact warning" || e.kind === "Bleeding" || e.kind === "No-Go zone" ? "text-err" : ""}>{e.label}</span>
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

/** First index whose time is >= t (samples are in time order). */
function lowerBound(samples: LiveSample[], t: number): number {
  let lo = 0, hi = samples.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (samples[m].t < t) lo = m + 1; else hi = m; }
  return lo;
}

const minN = (a: number | null, b: number | null) => (a == null ? b : b == null ? a : Math.min(a, b));
const maxN = (a: number | null, b: number | null) => (a == null ? b : b == null ? a : Math.max(a, b));

/** Group samples into at most n points, keeping the worst value of each group. */
function worstPerBucket(win: LiveSample[], n: number): LiveSample[] {
  if (win.length <= n) return win;
  const k = Math.ceil(win.length / n), out: LiveSample[] = [];
  for (let i = 0; i < win.length; i += k) {
    const g = win.slice(i, i + k);
    const tips: Record<string, LiveSample["tips"][string]> = {};
    let risk = g[0].risk, score: number | null = null;
    for (const s of g) {
      if ((RANK[s.risk] ?? 0) > (RANK[risk] ?? 0)) risk = s.risk;
      score = maxN(score, s.score);
      for (const [name, v] of Object.entries(s.tips)) {
        const c = tips[name];
        tips[name] = c ? { ...c, dist: minN(c.dist, v.dist), speed: maxN(c.speed, v.speed), ttc: minN(c.ttc, v.ttc),
                           target: v.dist != null && (c.dist == null || v.dist < c.dist) ? v.target : c.target } : { ...v };
      }
    }
    out.push({ ...g[g.length - 1], risk, score, tips, warning: g.find((s) => s.warning)?.warning ?? "" });
  }
  return out;
}
