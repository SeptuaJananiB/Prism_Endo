import { memo, useEffect, useMemo, useRef, useState } from "react";
import { api, type RunDetail, type RunSummary } from "../lib/api";
import { RISK_VAR } from "./Charts";
import { Select } from "./Select";

/** Side-by-side comparison of two analyses: synced annotated videos and a risk diff track. */
export const Compare = memo(function Compare({ a, runs, onClose }: { a: RunDetail; runs: RunSummary[]; onClose: () => void }) {
  const others = runs.filter((r) => r.name !== a.name);
  const [bName, setB] = useState(others.find((r) => r.source === runs.find((x) => x.name === a.name)?.source)?.name ?? others[0]?.name ?? "");
  const [b, setBD] = useState<RunDetail | null>(null);
  const va = useRef<HTMLVideoElement>(null), vb = useRef<HTMLVideoElement>(null);
  useEffect(() => { if (bName) api.run(bName).then(setBD).catch(() => setBD(null)); }, [bName]);
  // keep B in step with A
  useEffect(() => {
    const A = va.current, B = vb.current;
    if (!A || !B) return;
    const sync = () => { if (Math.abs(B.currentTime - A.currentTime) > 0.15) B.currentTime = A.currentTime; };
    const play = () => B.play().catch(() => {});
    const pause = () => B.pause();
    A.addEventListener("timeupdate", sync); A.addEventListener("seeked", sync); A.addEventListener("play", play); A.addEventListener("pause", pause);
    return () => { A.removeEventListener("timeupdate", sync); A.removeEventListener("seeked", sync); A.removeEventListener("play", play); A.removeEventListener("pause", pause); };
  }, [b]);
  const ca = (a.series.risk_category ?? []).map(String), cb = (b?.series.risk_category ?? []).map(String);
  const n = Math.min(ca.length, cb.length);
  const same = useMemo(() => { let k = 0; for (let i = 0; i < n; i++) if (ca[i] === cb[i]) k++; return n ? Math.round((100 * k) / n) : 0; }, [ca, cb, n]);
  const pct = (c: string[], k: string) => (c.length ? Math.round((100 * c.filter((x) => x === k).length) / c.length) : 0);

  return (
    <section className="card rise overflow-hidden" aria-label="Compare analyses">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-4 py-3">
        <h2 className="label m-0">Compare analyses</h2>
        <div className="flex items-center gap-2">
          <label htmlFor="cmp" className="text-[13px] text-muted">Compare with</label>
          <Select id="cmp" className="w-72" value={bName} onChange={setB}
            options={others.map((r) => ({ value: r.name, label: r.name, hint: r.surgery ?? undefined }))} />
          <button className="btn h-9 px-3 text-[13px]" onClick={onClose}>Close</button>
        </div>
      </header>
      <div className="grid grid-cols-1 gap-px bg-line md:grid-cols-2">
        {[{ d: a, ref: va, c: ca, main: true }, { d: b, ref: vb, c: cb, main: false }].map(({ d, ref, c, main }, k) => (
          <figure key={k} className="m-0 bg-surface">
            <figcaption className="flex h-10 items-center justify-between px-4 text-[13px]">
              <span className="truncate font-medium">{d?.name ?? "…"}</span>
              <span className="text-xs text-muted tabular">Medium+ {pct(c, "Medium") + pct(c, "High")}% · High {pct(c, "High")}%</span>
            </figcaption>
            {d?.videos.annotated ? (
              <video ref={ref} src={d.videos.annotated} controls={main} muted={!main} preload="metadata" className="block aspect-video w-full bg-black" />
            ) : <div className="skeleton aspect-video rounded-none" />}
          </figure>
        ))}
      </div>
      <div className="space-y-2 p-4">
        <div className="flex items-center justify-between text-xs text-muted">
          <span>Risk category per frame: top = {a.name}, bottom = {b?.name ?? "…"}</span>
          <span className="tabular">{same}% of frames agree</span>
        </div>
        {[ca, cb].map((c, k) => (
          <div key={k} className="flex h-4 overflow-hidden rounded" aria-hidden>
            {c.slice(0, n || c.length).map((x, i) => <span key={i} className="h-full flex-1" style={{ background: RISK_VAR[x] ?? RISK_VAR.Unavailable }} />)}
          </div>
        ))}
        <div className="flex h-2 overflow-hidden rounded" aria-hidden title="Differences">
          {Array.from({ length: n }, (_, i) => <span key={i} className="h-full flex-1" style={{ background: ca[i] === cb[i] ? "transparent" : "var(--ink)" }} />)}
        </div>
        <p className="m-0 text-xs text-muted">Dark ticks mark frames where the two analyses disagree. Video B follows the controls of video A.</p>
      </div>
    </section>
  );
});
