import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { api, type Job, type Meta, type RunDetail, type RunSummary } from "./lib/api";
import { useTheme } from "./lib/hooks";
import { episodes, milestones as getMilestones, summaryBullets, timeIndex, tracks as getTracks } from "./lib/derive";
import { Sidebar, type SidebarHandle } from "./components/Sidebar";
import { Kpi } from "./components/Kpi";
import { LineChart, RISK_VAR } from "./components/Charts";
import { StackedTimeline, type FlagT } from "./components/StackedTimeline";
import { Card, DownloadMenu, EmptyState, EventsPanel, FlagsPanel, ReportPanel, SettingsPanel, ShortcutsDialog, Toasts, type Toast } from "./components/Panels";
import { Compare } from "./components/Compare";
import { Landing, Splash } from "./components/Landing";
import { AiExplain } from "./components/AiExplain";
import { SidePanel } from "./components/SidePanel";
import { LiveInsights } from "./components/LiveInsights";

const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const label = (c: string) => c.replace(/_(prox_distance_px|speed_px_s)$/, "").replace(/_/g, " ");
type TabId = "risk" | "distance" | "speed" | "cvs" | "bleeding";

function Tabs<T extends string>({ value, options, onChange, label: aria }: { value: T; options: { id: T; label: string }[]; onChange: (v: T) => void; label: string }) {
  const refs = useRef<Record<string, HTMLButtonElement | null>>({});
  const [thumb, setThumb] = useState({ left: 3, width: 0 });
  useLayoutEffect(() => { const el = refs.current[value]; if (el) setThumb({ left: el.offsetLeft, width: el.offsetWidth }); }, [value, options.length]);
  return (
    <div className="seg" role="tablist" aria-label={aria} onKeyDown={(e) => {
      const k = options.findIndex((o) => o.id === value);
      const n = e.key === "ArrowRight" ? k + 1 : e.key === "ArrowLeft" ? k - 1 : -2;
      if (n === -2) return;
      const o = options[(n + options.length) % options.length];
      onChange(o.id); refs.current[o.id]?.focus();
    }}>
      <span className="thumb" style={thumb} aria-hidden />
      {options.map((o) => <button key={o.id} ref={(el) => { refs.current[o.id] = el; }} role="tab" aria-selected={o.id === value}
        tabIndex={o.id === value ? 0 : -1} onClick={() => onChange(o.id)}>{o.label}</button>)}
    </div>
  );
}

const Icon = {
  menu: <path d="M4 7h16M4 12h16M4 17h16" />,
  moon: <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />,
  sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>,
  compare: <path d="M8 3v18M16 3v18M3 8h5M16 16h5" />,
  kbd: <><rect x="2" y="6" width="20" height="12" rx="2" /><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10" /></>,
};
const Svg = ({ d }: { d: React.ReactNode }) => <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round" aria-hidden>{d}</svg>;

export default function App() {
  const [theme, setTheme] = useTheme();
  const [meta, setMeta] = useState<Meta | null>(null);
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [current, setCurrent] = useState<string | null>(null);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [tab, setTab] = useState<TabId>("risk");
  const [frame, setFrame] = useState(0);
  const [imgReady, setImgReady] = useState(false);
  const [imgErr, setImgErr] = useState(false);
  const [flags, setFlags] = useState<FlagT[]>([]);
  const [saving, setSaving] = useState(false);
  const [compare, setCompare] = useState(false);
  const [help, setHelpState] = useState(false);
  const [helpReq, setHelpReq] = useState(0);
  const setHelp = useCallback((v: boolean) => { setHelpState(v); if (v) setHelpReq((n) => n + 1); }, []);
  const [menu, setMenu] = useState(false);
  // AI panel: docked on wide screens (open unless closed before); an overlay on narrow ones (start closed)
  const [aiOpen, setAiOpen] = useState(() => {
    if (window.innerWidth < 1280) return false;
    try { return localStorage.getItem("pe-ai-open") !== "0"; } catch { return true; }
  });
  useEffect(() => { try { localStorage.setItem("pe-ai-open", aiOpen ? "1" : "0"); } catch { /* storage blocked */ } }, [aiOpen]);
  // landing page first (until the user opens the workspace); splash while the first data loads
  const [view, setView] = useState<"landing" | "app">(() => (location.hash === "#app" ? "app" : "landing"));
  const [booted, setBooted] = useState(false);
  useEffect(() => { const h = window.setTimeout(() => setBooted(true), 700); return () => window.clearTimeout(h); }, []);
  const go = useCallback((v: "landing" | "app") => {
    setView(v);
    history.replaceState(null, "", v === "app" ? "#app" : location.pathname);
    window.scrollTo(0, 0);
  }, []);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [fresh, setFresh] = useState(false); // highlight results that just finished
  const vMain = useRef<HTMLVideoElement>(null);
  const vOrig = useRef<HTMLVideoElement>(null);
  const side = useRef<SidebarHandle>(null);

  const toast = useCallback((t: Omit<Toast, "id">) => {
    const id = Date.now() + Math.random();
    setToasts((x) => [...x, { ...t, id }]);
    setTimeout(() => setToasts((x) => x.filter((y) => y.id !== id)), 7000);
  }, []);
  const loadRuns = useCallback(() => api.runs().then(setRuns).catch((e) => setErr(String(e.message ?? e))), []);
  useEffect(() => { api.meta().then(setMeta).catch(() => {}); loadRuns(); }, [loadRuns]);
  // pick up an analysis started elsewhere (another tab, the API) so it is always visible and stoppable
  useEffect(() => {
    const check = () => api.active().then((a) => {
      if ("id" in a && a.id) setJob((cur) => (cur && cur.id === a.id ? cur : (a as Job)));
    }).catch(() => {});
    check();
    const h = setInterval(check, 3000);
    return () => clearInterval(h);
  }, []);
  useEffect(() => { if (!current && runs?.length) setCurrent(runs[0].name); }, [runs, current]);
  const [reload, setReload] = useState(0);
  useEffect(() => {
    if (!current) return;
    setLoading(true); setErr(null); setCompare(false);
    Promise.all([api.run(current), api.flags(current).catch(() => [])])
      .then(([d, f]) => { setDetail(d); setFlags(f as FlagT[]); setFrame(0); })
      .catch((e) => setErr(e.message)).finally(() => setLoading(false));
  }, [current, reload]);

  useEffect(() => {
    if (!job || !["queued", "loading", "running"].includes(job.status)) return;
    const h = setInterval(() => api.job(job.id).then((j) => {
      setJob(j);
      if (j.status === "finished" && j.run) {
        loadRuns();
        // open the finished analysis straight away and highlight it
        setDetail(null); setCurrent(j.run); setReload((r) => r + 1); setFresh(true);
        document.getElementById("main")?.scrollTo({ top: 0, behavior: "smooth" });
        window.setTimeout(() => setFresh(false), 3200);
        toast({ tone: "ok", text: `Analysis finished: ${j.run}. Showing the results.` });
      }
      if (j.status === "failed") toast({ tone: "err", text: `Analysis failed: ${j.error}` });
      if (j.status === "cancelled") toast({ tone: "info", text: "Analysis stopped. No results were saved for it." });
    }).catch(() => {}), 1000);
    return () => clearInterval(h);
  }, [job, loadRuns, toast]);

  const s = useMemo(() => detail?.series ?? {}, [detail]);
  const t = useMemo(() => (s.timestamp_s ?? []).map((v) => Number(v)), [s]);
  const cat = useMemo(() => (s.risk_category ?? []).map((v) => String(v ?? "Unavailable")), [s]);
  const n = t.length;
  const target = Object.keys(detail?.summary.proximity_targets ?? {})[0] ?? "target";
  const trks = useMemo(() => getTracks(s, n, 60, target), [s, n, target]);
  const evs = useMemo(() => episodes(t, s), [t, s]);
  const ms = useMemo(() => getMilestones(t, s), [t, s]);
  const bullets = useMemo(() => summaryBullets(t, s, ms), [t, s, ms]);

  // video <-> frame sync: the annotated video drives the playhead; the original follows it
  useEffect(() => {
    const v = vMain.current;
    if (!v || !n) return;
    // playhead at ~10 Hz while playing (60 Hz full re-renders froze the page)
    let timer = 0;
    const sync = () => {
      setFrame(timeIndex(t, v.currentTime));
      const o = vOrig.current;
      if (o && Math.abs(o.currentTime - v.currentTime) > 0.25) o.currentTime = v.currentTime;
    };
    const step = () => { window.clearInterval(timer); sync(); };
    const play = () => { vOrig.current?.play().catch(() => {}); window.clearInterval(timer); timer = window.setInterval(sync, 100); };
    const pause = () => { vOrig.current?.pause(); step(); };
    v.addEventListener("play", play); v.addEventListener("pause", pause); v.addEventListener("seeked", step);
    return () => { window.clearInterval(timer); v.removeEventListener("play", play); v.removeEventListener("pause", pause); v.removeEventListener("seeked", step); };
  }, [detail, t, n, compare]);

  const seek = useCallback((i: number) => {
    const k = Math.max(0, Math.min(n - 1, i));
    setFrame(k);
    if (vMain.current && t[k] != null) vMain.current.currentTime = t[k];
  }, [n, t]);
  // inspector image follows the playhead once it settles (one server decode, not 25 per second)
  const [shownFrame, setShownFrame] = useState(0);
  useEffect(() => { const h = window.setTimeout(() => setShownFrame(frame), 250); return () => window.clearTimeout(h); }, [frame]);
  useEffect(() => { setImgReady(false); setImgErr(false); }, [shownFrame, current]);

  const saveFlags = useCallback((next: FlagT[]) => {
    setFlags(next);
    if (!current) return;
    setSaving(true);
    api.saveFlags(current, next).catch((e) => toast({ tone: "err", text: `Could not save flag: ${e.message}` })).finally(() => setSaving(false));
  }, [current, toast]);
  const addFlag = useCallback((f: FlagT) => {
    saveFlags([...flags, f].sort((a, b) => a.t - b.t));
    toast({ tone: "info", text: `Flag "${f.kind === "wrong" ? "AI wrong" : f.kind}" added at ${f.t.toFixed(1)} s` });
  }, [flags, saveFlags, toast]);

  // keyboard-first review
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const el = e.target instanceof HTMLElement ? e.target : document.body;
      if (el.closest("input, textarea, select, [contenteditable]") || e.ctrlKey || e.metaKey || e.altKey) return;
      const v = vMain.current;
      const step = e.shiftKey ? Math.round((detail?.summary.source_fps ?? 25) / (detail?.summary.stride ?? 1)) : 1;
      const key = e.key.toLowerCase();
      if (e.key === "?") { setHelp(true); e.preventDefault(); }
      else if (key === " " && v && !el.closest("button, a, video")) { v.paused ? v.play() : v.pause(); e.preventDefault(); }
      else if (key === "d") seek(frame - step);
      else if (key === "f") seek(frame + step);
      else if (key === "j") { const p = [...evs].reverse().find((x) => x.i < frame); if (p) seek(p.i); }
      else if (key === "k") { const q = evs.find((x) => x.i > frame); if (q) seek(q.i); }
      else if (key === "t") setTheme(theme === "light" ? "dark" : "light");
      else if (["1", "2", "3"].includes(key) && n) addFlag({ t: t[frame], frame, kind: (["agree", "wrong", "important"] as const)[Number(key) - 1], note: "" });
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [frame, evs, seek, theme, setTheme, n, t, addFlag, detail]);

  const stop = async () => {
    if (!job) return;
    try { setJob(await api.cancel(job.id)); } catch (e) { toast({ tone: "err", text: (e as Error).message }); }
  };
  const finish = async () => {
    if (!job) return;
    try { setJob(await api.finish(job.id)); toast({ tone: "info", text: "Finishing live session — saving the recording…" }); }
    catch (e) { toast({ tone: "err", text: (e as Error).message }); }
  };
  const liveJob = job && job.live && ["queued", "loading", "running"].includes(job.status) ? job : null;
  const start = async (body: object) => {
    try { setJob(await api.start(body)); toast({ tone: "info", text: "Analysis started" }); }
    catch (e) {
      const a = await api.active().catch(() => ({}));
      if ("id" in a && a.id) setJob(a as Job);
      toast({ tone: "err", text: (e as Error).message });
    }
  };

  const share = (pred: (i: number) => boolean) => (n ? (100 * t.filter((_, i) => pred(i)).length) / n : 0);
  const hiPct = share((i) => cat[i] === "Medium" || cat[i] === "High");
  const bleed = (s.bleeding_detected ?? []).filter((v) => v === true).length;
  const cvsPct = s.cvs_achieved ? share((i) => s.cvs_achieved[i] === true) : null;
  const views = useMemo<Record<TabId, { lines: { name: string; values: (number | null)[] }[]; unit: string }>>(() => {
  const pick = (suffix: string) => Object.keys(s).filter((c) => c.endsWith(suffix));
  return {
    risk: { lines: [{ name: "Risk score", values: (s.risk_score ?? []).map(num) }], unit: "" },
    distance: { lines: pick("_prox_distance_px").map((c) => ({ name: label(c), values: s[c].map(num) })), unit: " px" },
    speed: { lines: pick("_speed_px_s").map((c) => ({ name: label(c), values: s[c].map(num) })), unit: " px/s" },
    cvs: { lines: ["cvs_c1", "cvs_c2", "cvs_c3"].filter((c) => s[c]).map((c) => ({ name: c.replace("cvs_", "").toUpperCase(), values: s[c].map(num) })), unit: "" },
    bleeding: { lines: s.bleeding_frac ? [{ name: "Bleeding area", values: s.bleeding_frac.map((v) => (num(v) ?? 0) * 100) }] : [], unit: " %" },
  };
  }, [s]);
  const sm = detail?.summary;
  const device = sm ? (/privateuse|cuda|dml/.test(sm.device) ? "GPU" : sm.device.toUpperCase()) : "";
  const why = s.risk_explanation?.[frame] || Object.keys(s).filter((c) => c.endsWith("_risk_explanation")).map((c) => s[c][frame]).find((v) => v);
  const settingsRows: [string, React.ReactNode][] = sm ? [
    ["Video", (sm.input ?? "").split(/[\\/]/).pop()],
    ["Surgery / tissue model", sm.tissue_checkpoint ? (sm.tissue_checkpoint.includes("dsad") ? "Other (DSAD)" : "Gallbladder") : (sm.tissue_model ? "on" : "off")],
    ["Tip source", sm.keypoint_source],
    ["Bleeding detector", sm.bleeding_detector === false ? "off" : "on"],
    ["CVS model", sm.cvs_model === false ? "off" : "on"],
    ["Risk", sm.risk_mode ?? "rule_based"],
    ["Proximity target", Object.entries(sm.proximity_targets ?? {}).map(([k, v]) => `${k} (${v.reliability ?? "?"})`).join(", ") || "none"],
    ["Frame rate / stride", `${sm.source_fps ?? "?"} fps / ${sm.stride ?? 1}`],
    ["Device", device],
  ] : [];

  if (!booted || (runs === null && !err)) return <Splash />;
  if (view === "landing")
    return <Landing runs={runs} onEnter={() => go("app")} onOpenRun={(nm) => { setCurrent(nm); go("app"); }} />;

  return (
    <div className="flex h-dvh min-h-full overflow-hidden">
      <a href="#main" className="skip">Skip to results</a>
      <Sidebar ref={side} meta={meta} runs={runs} current={current} onOpen={(nm) => setCurrent(nm)} onStart={start} onStop={stop} onFinish={finish} job={job}
        onUploaded={(src) => { api.meta().then(setMeta); toast({ tone: "ok", text: `Uploaded ${src.split(/[\\/]/).pop()}` }); }}
        open={menu} onClose={() => setMenu(false)} />

      <main id="main" className="min-w-0 flex-1 overflow-y-auto" tabIndex={-1}>
        <header className="no-print sticky top-0 z-20 border-b border-line bg-bg/85 backdrop-blur-md">
          <div className="mx-auto flex h-16 max-w-[1280px] items-center justify-between gap-3 px-4 sm:px-8">
            <div className="flex min-w-0 items-center gap-3">
              <button className="btn h-9 w-9 p-0 lg:hidden" onClick={() => setMenu(true)} aria-label="Open menu"><Svg d={Icon.menu} /></button>
              <div className="min-w-0">
                <nav aria-label="Breadcrumb" className="text-xs text-muted">
                  <button className="hover:text-ink hover:underline" onClick={() => go("landing")}>Home</button> / Library
                </nav>
                <h1 className="m-0 truncate text-[17px] font-semibold tracking-tight">{liveJob ? "Live session" : current ?? "No analysis selected"}</h1>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {detail && !liveJob && <button className={`btn h-9 px-3 text-[13px] ${compare ? "border-ink" : ""}`} aria-pressed={compare} onClick={() => setCompare((c) => !c)} title="Compare with another analysis"><Svg d={Icon.compare} /><span className="hidden md:inline">Compare</span></button>}
              {detail && !liveJob && <DownloadMenu run={detail.name} />}
              {detail && !liveJob && <button className={`btn h-9 px-3 text-[13px] ${aiOpen ? "border-ink" : ""}`} aria-pressed={aiOpen} aria-controls="ai-panel"
                onClick={() => setAiOpen((o) => !o)} title="AI explanation panel">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M12 3v3m0 12v3M3 12h3m12 0h3M6 6l2 2m8 8 2 2m0-12-2 2M8 16l-2 2" /></svg>
                <span className="hidden md:inline">AI</span></button>}
              <button className="btn h-9 w-9 p-0" onClick={() => setHelp(true)} aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)"><Svg d={Icon.kbd} /></button>
              <button className="btn h-9 w-9 p-0" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`} title="Theme (T)">
                <Svg d={theme === "light" ? Icon.moon : Icon.sun} />
              </button>
            </div>
          </div>
        </header>

        <div className="@container mx-auto max-w-[1280px] space-y-6 px-4 py-6 sm:px-8 sm:py-8">
          {err && <p role="alert" className="chip chip-err m-0 h-auto py-2">{err}</p>}
          {liveJob && (
            <section className="card rise overflow-hidden" aria-label="Live analysis">
              <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
                <div className="flex items-center gap-2.5">
                  <span className="relative flex h-2.5 w-2.5" aria-hidden><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-err opacity-60" /><span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-err" /></span>
                  <h2 className="m-0 text-[14px] font-semibold">Live analysis</h2>
                  <span className="text-xs text-muted tabular">{liveJob.status === "running" ? `view ${liveJob.view_fps ?? "–"} fps · analysis ${liveJob.live_fps ?? "–"} fps · ${liveJob.done} analysed` : liveJob.phase ?? "Starting camera…"}</span>
                </div>
                <div className="flex items-center gap-2">
                  {liveJob.risk && <span className="chip" style={{ color: "var(--ink)", background: "var(--surface-2)" }}><span className="h-2 w-2 rounded-full" style={{ background: RISK_VAR[liveJob.risk] }} aria-hidden />Risk: {liveJob.risk}</span>}
                  {liveJob.bleeding && <span className="chip chip-err">Bleeding</span>}
                  <button className="btn btn-primary h-9 px-3 text-[13px]" onClick={finish} disabled={liveJob.cancel || liveJob.end}>Finish &amp; save</button>
                  <button className="btn h-9 px-3 text-[13px] text-err" onClick={stop} disabled={liveJob.cancel}>Discard</button>
                </div>
              </header>
              {liveJob.warning && (
                <div role="alert" className="flex items-center gap-2 bg-[var(--err-bg)] px-4 py-2 text-[13px] font-medium text-err">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden><path d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" /></svg>
                  Contact warning: {liveJob.warning}
                </div>
              )}
              <div className="grid aspect-video place-items-center bg-black">
                {liveJob.status === "running"
                  ? <img src={api.liveUrl(liveJob.id)} alt="Live annotated camera view" className="h-full w-full object-contain" />
                  : <span className="text-[13px] text-white/70">{liveJob.phase ?? "Starting camera…"}</span>}
              </div>
              {liveJob.status === "running" && <div className="@container"><LiveInsights jobId={liveJob.id} /></div>}
            </section>
          )}
          {runs && runs.length === 0 && <EmptyState onFocusForm={() => { setMenu(true); setTimeout(() => side.current?.focusForm(), 320); }} />}

          {/* while a live session runs, show only the live view (not the previously opened saved analysis) */}
          {liveJob ? null : (loading || (!detail && runs?.length !== 0)) && !err ? (
            <div className="space-y-6" aria-busy="true" aria-label="Loading analysis">
              <div className="grid grid-cols-2 gap-4 @2xl:grid-cols-3 @6xl:grid-cols-6">{Array.from({ length: 6 }, (_, k) => <div key={k} className="skeleton h-[104px]" />)}</div>
              <div className="grid grid-cols-1 gap-4 @4xl:grid-cols-2"><div className="skeleton aspect-video" /><div className="skeleton aspect-video" /></div>
              <div className="skeleton h-56" />
            </div>
          ) : detail && sm && (
            <div key={detail.name} className={`space-y-6 ${fresh ? "fresh" : ""}`}>
              <section aria-label="Key figures" className="grid grid-cols-2 gap-4 @2xl:grid-cols-3 @6xl:grid-cols-6">
                <Kpi i={0} label="Frames" value={sm.frames_processed} caption={n ? `${t[n - 1].toFixed(0)} s of video` : ""} />
                <Kpi i={1} label="Speed" value={sm.runtime?.processing_fps ?? 0} decimals={1} suffix=" fps" caption={`on ${device}`} />
                <Kpi i={2} label="Tip source" value={sm.keypoint_source} caption="instrument keypoints" />
                <Kpi i={3} label="Medium / high risk" value={hiPct} suffix="%" caption="of frames" tone={hiPct > 25 ? "warn" : undefined} chip={hiPct > 25 ? "Elevated" : undefined} />
                <Kpi i={4} label="Bleeding" value={bleed} caption={bleed ? "frames flagged" : "no frames flagged"} tone={bleed ? "err" : "ok"} chip={bleed ? "Review" : "Clear"} />
                <Kpi i={5} label="CVS met" value={cvsPct ?? 0} suffix="%" caption="frames, all 3 criteria" />
              </section>

              {!!sm.warnings?.length && (
                <details className="card rise group p-0 no-print" style={{ ["--i" as string]: 2 }}>
                  <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-[13px]">
                    <span className="flex items-center gap-2"><span className="chip chip-warn">{sm.warnings.length}</span> Reliability notes for this analysis</span>
                    <svg className="transition-transform duration-200 group-open:rotate-180" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden><path d="m6 9 6 6 6-6" /></svg>
                  </summary>
                  <ul className="m-0 space-y-2 border-t border-line px-6 py-3 text-[13px] text-ink-2">{sm.warnings.map((w, k) => <li key={k}>{w}</li>)}</ul>
                </details>
              )}

              {compare && runs ? <Compare a={detail} runs={runs} onClose={() => setCompare(false)} /> : (
                <div className="grid grid-cols-1 gap-4 @4xl:grid-cols-[1.35fr_1fr] no-print">
                  <figure className="card rise m-0 overflow-hidden" style={{ ["--i" as string]: 3 }}>
                    <figcaption className="flex h-10 items-center justify-between border-b border-line px-4">
                      <span className="text-[13px] font-medium">Annotated</span>
                      <span className="flex items-center gap-2 text-xs text-muted tabular">
                        <span className="h-2 w-2 rounded-full" style={{ background: RISK_VAR[cat[frame]] }} aria-hidden />{cat[frame]} · {(t[frame] ?? 0).toFixed(1)} s
                      </span>
                    </figcaption>
                    {detail.videos.annotated ? <video ref={vMain} src={detail.videos.annotated} controls preload="metadata" className="block aspect-video w-full bg-black" aria-label="Annotated video. Space plays or pauses." />
                      : <div className="grid aspect-video place-items-center text-muted">Video not available</div>}
                  </figure>
                  <div className="grid grid-rows-[auto_1fr] gap-4">
                    <figure className="card rise m-0 overflow-hidden" style={{ ["--i" as string]: 4 }}>
                      <figcaption className="flex h-10 items-center justify-between border-b border-line px-4">
                        <span className="text-[13px] font-medium">Original</span><span className="text-xs text-muted">follows the annotated video</span>
                      </figcaption>
                      {detail.videos.original ? <video ref={vOrig} src={detail.videos.original} muted preload="metadata" className="block aspect-video w-full bg-black" aria-label="Original video, synced" />
                        : <div className="grid aspect-video place-items-center text-muted">Video not available</div>}
                    </figure>
                    <Card i={5} title="Why this risk">
                      {/* fixed two-line box: the text changes every frame and must not resize the layout */}
                      <p className="m-0 h-[calc(2*1.6em+2rem)] overflow-hidden p-4 text-[13px] leading-[1.6] text-ink-2 [display:-webkit-box] [-webkit-box-orient:vertical] [-webkit-line-clamp:2]"
                        title={why ? String(why) : undefined} aria-live="off">
                        {why ? String(why).replace(/(lhook|grasper)_tip(_2)?/g, (_m, k, two) => `${k === "lhook" ? "L-hook" : "Grasper"} tip${two ? " 2" : ""}`) : "No explanation for this frame (no valid tip or target)."}
                      </p>
                    </Card>
                  </div>
                </div>
              )}

              <Card i={6} title="Timeline" aside={<span className="text-xs text-muted">Synced with the video · rule-based risk</span>}>
                <div className="p-4 sm:p-5"><StackedTimeline t={t} tracks={trks} milestones={ms} flags={flags} frame={frame} onSeek={seek} run={detail.name} /></div>
              </Card>

              <div className="grid grid-cols-1 gap-4 @5xl:grid-cols-3">
                <Card i={7} title="Summary" className="print-full"><ReportPanel bullets={bullets} onSeek={seek} onPrint={() => window.print()} /></Card>
                <Card i={8} title="Events" aside={<span className="text-xs text-muted tabular">{evs.length} · J / K</span>} className="no-print"><EventsPanel events={evs} onSeek={seek} current={frame} /></Card>
                <Card i={9} title="Reviewer flags" aside={<span className="text-xs text-muted">1 / 2 / 3</span>} className="no-print">
                  <FlagsPanel flags={flags} t={t[frame] ?? 0} frame={frame} saving={saving} onSeek={seek} onAdd={addFlag}
                    onRemove={(k) => saveFlags(flags.filter((_, j) => j !== k))} />
                </Card>
              </div>

              <Card i={10} title="Signals" aside={<Tabs label="Signal" value={tab} onChange={setTab} options={[
                { id: "risk", label: "Risk" }, { id: "distance", label: "Distance" }, { id: "speed", label: "Tip speed" }, { id: "cvs", label: "CVS" }, { id: "bleeding", label: "Bleeding" }]} />}>
                <div className="p-4 sm:p-5" role="tabpanel" aria-label={tab}>
                  <LineChart key={tab} t={t} lines={views[tab].lines} unit={views[tab].unit} onSeek={seek} colorBy={tab === "risk" ? cat : undefined} />
                </div>
              </Card>

              <div className="grid grid-cols-1 gap-4 @6xl:grid-cols-[1fr_340px]">
                <Card i={11} title="Frame inspector" aside={<span className="text-xs text-muted tabular">Frame {frame + 1} of {n} · D / F</span>}>
                  <div id="inspector" className="grid grid-cols-1 @xl:grid-cols-[1fr_220px]">
                    <div className="relative aspect-video bg-black">
                      {!imgReady && !imgErr && <div className="skeleton absolute inset-0 rounded-none" aria-hidden />}
                      {imgErr && <div className="absolute inset-0 grid place-items-center text-[13px] text-muted">Frame image not available for this analysis</div>}
                      <img src={api.frameUrl(detail.name, shownFrame)} alt={`Annotated frame ${shownFrame + 1}`} onLoad={() => setImgReady(true)} onError={() => setImgErr(true)}
                        className={`h-full w-full object-contain transition-opacity duration-300 ${imgReady ? "opacity-100" : "opacity-0"}`} />
                    </div>
                    <div className="flex flex-col gap-4 border-t border-line p-4 @xl:border-l @xl:border-t-0">
                      <div><div className="label">Time</div><div className="mt-1 text-[20px] font-semibold tabular">{(t[frame] ?? 0).toFixed(2)} s</div></div>
                      <div><div className="label">Risk</div>
                        <div className="mt-1.5 flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full" style={{ background: RISK_VAR[cat[frame]] }} aria-hidden />
                          <span className="font-medium">{cat[frame]}</span>{num(s.risk_score?.[frame]) != null && <span className="text-muted tabular">{(num(s.risk_score[frame]) as number).toFixed(2)}</span>}</div></div>
                      <div><div className="label">Bleeding</div><div className="mt-1.5">{s.bleeding_detected?.[frame] === true ? <span className="chip chip-err">Flagged</span> : <span className="text-[13px] text-muted">Not flagged</span>}</div></div>
                      <div className="mt-auto space-y-2">
                        <label htmlFor="frame" className="label block">Scrub</label>
                        <input id="frame" type="range" min={0} max={Math.max(0, n - 1)} value={frame} onChange={(e) => seek(Number(e.target.value))} aria-valuetext={`${(t[frame] ?? 0).toFixed(1)} seconds`} />
                        <div className="grid grid-cols-4 gap-1.5">
                          {([["−1s", -25, "Back one second"], ["‹", -1, "Previous frame"], ["›", 1, "Next frame"], ["+1s", 25, "Forward one second"]] as const).map(([l, d, a]) => (
                            <button key={l} className="btn h-8 px-0 text-[13px]" onClick={() => seek(frame + d)} aria-label={a}>{l}</button>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </Card>
                <Card i={12} title="Analysis settings"><SettingsPanel rows={settingsRows} /></Card>
              </div>

              <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-5 text-xs text-muted">
                <span>Measurements are image-based estimates; not validated for clinical use.</span>
                <button className="underline-offset-2 hover:text-ink hover:underline" onClick={() => setHelp(true)} aria-label="Show keyboard shortcuts list">Press ? for keyboard shortcuts</button>
              </footer>
            </div>
          )}
        </div>
      </main>

      {detail && !liveJob && (
        <div id="ai-panel" className="contents">
          <SidePanel open={aiOpen} onClose={() => setAiOpen(false)} title="AI explanation" badge={<span className="chip chip-info">gpt-oss</span>}>
            <AiExplain run={detail.name} />
          </SidePanel>
        </div>
      )}

      <ShortcutsDialog open={help} req={helpReq} onClose={() => setHelp(false)} />
      <Toasts items={toasts} dismiss={(id) => setToasts((x) => x.filter((y) => y.id !== id))} />
    </div>
  );
}
