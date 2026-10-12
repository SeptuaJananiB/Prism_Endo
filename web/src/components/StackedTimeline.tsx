import { memo, useMemo, useRef, useState } from "react";
import type { Milestone, Track } from "../lib/derive";
import { timeIndex } from "../lib/derive";
import { RISK, RISK_VAR } from "./Charts";
import { api } from "../lib/api";

export type FlagT = { t: number; frame: number; kind: "agree" | "wrong" | "important"; note: string };
export const FLAG_TONE: Record<FlagT["kind"], string> = { agree: "var(--ok)", wrong: "var(--err)", important: "var(--info)" };

type Seg = { a: number; b: number; v: string | boolean; unsure: boolean; i: number };

function segments(t: number[], tr: Track, w0: number, w1: number): Seg[] {
  const out: Seg[] = [];
  let st = -1;
  for (let i = 0; i <= t.length; i++) {
    const inWin = i < t.length && t[i] >= w0 && t[i] <= w1;
    const same = st >= 0 && i < t.length && inWin && tr.values[i] === tr.values[st] && !!tr.unsure?.[i] === !!tr.unsure?.[st];
    if (st >= 0 && !same) {
      out.push({ a: t[st], b: i < t.length ? Math.min(t[i], w1) : t[t.length - 1], v: tr.values[st], unsure: !!tr.unsure?.[st], i: st });
      st = -1;
    }
    if (st < 0 && inWin) st = i;
  }
  return out;
}

export const StackedTimeline = memo(function StackedTimeline({ t, tracks, milestones, flags, frame, onSeek, run }: {
  t: number[]; tracks: Track[]; milestones: Milestone[]; flags: FlagT[]; frame: number; onSeek: (i: number) => void; run: string;
}) {
  const t0 = t[0] ?? 0, tN = t[t.length - 1] ?? 1;
  const [win, setWin] = useState<[number, number]>([t0, tN]);
  const [drag, setDrag] = useState<[number, number] | null>(null);
  const dragStart = useRef<number | null>(null); // ref: correct even before React re-renders mid-drag
  const area = useRef<HTMLDivElement>(null);
  const [w0, w1] = win[1] > win[0] ? win : [t0, tN];
  const pct = (v: number) => `${(100 * (v - w0)) / (w1 - w0 || 1)}%`;
  const toT = (clientX: number) => {
    const r = area.current!.getBoundingClientRect();
    return w0 + Math.min(1, Math.max(0, (clientX - r.left) / r.width)) * (w1 - w0);
  };
  const segs = useMemo(() => tracks.map((tr) => segments(t, tr, w0, w1)), [t, tracks, w0, w1]);
  const zoomed = w0 > t0 + 1e-6 || w1 < tN - 1e-6;
  const thumbs = Array.from({ length: 10 }, (_, k) => w0 + ((k + 0.5) / 10) * (w1 - w0));
  const ph = t[frame] ?? t0;

  return (
    <div className="select-none">
      <div className="mb-3 flex items-center justify-between gap-3 text-xs text-muted">
        <span>Drag across the tracks to zoom · double-click to reset · click to jump</span>
        <span className="flex items-center gap-2 tabular">
          {zoomed && <button className="btn h-7 px-2.5 text-xs" onClick={() => setWin([t0, tN])}>Reset zoom</button>}
          {w0.toFixed(1)}–{w1.toFixed(1)} s
        </span>
      </div>

      <div className="grid grid-cols-[112px_1fr] gap-x-3">
        <div aria-hidden>
          <div className="label flex h-6 items-center">Markers</div>
          {tracks.map((tr) => (
            <div key={tr.id} className="flex h-7 items-center border-t border-line text-[12px] font-medium text-ink-2" title={tr.hint}>
              <span className="truncate">{tr.label}</span>
            </div>
          ))}
          <div className="label flex h-[54px] items-center border-t border-line">Frames</div>
        </div>

        <div className="min-w-0">
          <div ref={area} className="relative cursor-crosshair"
            onMouseDown={(e) => {
              if (e.button !== 0 || (e.target as HTMLElement).closest("button")) return;
              const v = toT(e.clientX); dragStart.current = v; setDrag([v, v]);
            }}
            onMouseMove={(e) => { if (dragStart.current != null) setDrag([dragStart.current, toT(e.clientX)]); }}
            onMouseUp={(e) => {
              const s0 = dragStart.current;
              dragStart.current = null; setDrag(null);
              if (s0 == null) return;
              const a = Math.min(s0, toT(e.clientX)), b = Math.max(s0, toT(e.clientX));
              if (b - a > (w1 - w0) / 60) setWin([a, b]); else onSeek(timeIndex(t, a));
            }}
            onMouseLeave={() => { dragStart.current = null; setDrag(null); }}
            onDoubleClick={() => setWin([t0, tN])}>
            <div className="relative h-6">
              {milestones.filter((m) => m.t >= w0 && m.t <= w1).map((m, k) => (
                <button key={`m${k}`} title={`${m.label} · ${m.t.toFixed(1)} s`} aria-label={`${m.label} at ${m.t.toFixed(1)} seconds`}
                  onClick={() => onSeek(m.i)}
                  className="absolute top-1/2 z-10 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rotate-45 rounded-[2px] border-2 border-surface transition-transform duration-150 hover:scale-125"
                  style={{ left: pct(m.t), background: `var(--${m.tone})` }} />
              ))}
              {flags.filter((f) => f.t >= w0 && f.t <= w1).map((f, k) => (
                <button key={`f${k}`} title={`Reviewer: ${f.kind}${f.note ? ` - ${f.note}` : ""}`} aria-label={`Reviewer flag ${f.kind} at ${f.t.toFixed(1)} seconds`}
                  onClick={() => onSeek(f.frame)} className="absolute top-0 z-10 -translate-x-1/2 transition-transform hover:-translate-y-0.5"
                  style={{ left: pct(f.t) }}>
                  <svg width="12" height="16" viewBox="0 0 12 16" aria-hidden><path d="M2 1v14" stroke={FLAG_TONE[f.kind]} strokeWidth="1.6" /><path d="M2.5 1.5h8l-2 3 2 3h-8z" fill={FLAG_TONE[f.kind]} /></svg>
                </button>
              ))}
            </div>
            {tracks.map((tr, k) => (
              <div key={tr.id} className="relative h-7 border-t border-line" role="img" aria-label={`${tr.label} track`}>
                <div className="absolute inset-x-0 top-1.5 bottom-1.5 overflow-hidden rounded-[4px] bg-surface-2">
                  {segs[k].map((g, j) => {
                    if (tr.id !== "risk" && g.v !== true) return null;
                    const bg = tr.id === "risk" ? RISK_VAR[String(g.v)] ?? RISK_VAR.Unavailable : tr.color;
                    return (
                      <div key={j} className="absolute inset-y-0" title={`${tr.label}${tr.id === "risk" ? `: ${g.v}` : ""} · ${g.a.toFixed(1)}-${g.b.toFixed(1)} s${g.unsure ? " · low confidence" : ""}`}
                        style={{
                          left: pct(g.a), width: `max(1px, calc(${pct(g.b)} - ${pct(g.a)}))`, background: bg,
                          backgroundImage: g.unsure ? "repeating-linear-gradient(135deg, rgba(255,255,255,.55) 0 2px, transparent 2px 5px)" : undefined,
                        }} />
                    );
                  })}
                </div>
              </div>
            ))}
            {drag && <div className="pointer-events-none absolute inset-y-0 rounded bg-info/15 ring-1 ring-info/40"
              style={{ left: pct(Math.min(...drag)), width: `calc(${pct(Math.max(...drag))} - ${pct(Math.min(...drag))})` }} />}
            {ph >= w0 && ph <= w1 && (
              <div className="pointer-events-none absolute -bottom-1 top-0 w-px bg-ink" style={{ left: pct(ph) }}>
                <span className="absolute -top-1 left-1/2 h-2 w-2 -translate-x-1/2 rounded-full bg-ink" />
              </div>
            )}
          </div>

          <div className="grid h-[54px] grid-cols-10 gap-1 border-t border-line pt-1.5">
            {thumbs.map((tv, k) => {
              const i = timeIndex(t, tv);
              return (
                <button key={k} onClick={() => onSeek(i)} aria-label={`Jump to ${tv.toFixed(1)} seconds`}
                  className="group relative overflow-hidden rounded-[4px] border border-line bg-black">
                  <img loading="lazy" src={`${api.frameUrl(run, i)}?w=160&kind=original`} alt="" onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} className="h-full w-full object-cover opacity-90 transition duration-200 group-hover:scale-105 group-hover:opacity-100" />
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <ul className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted" aria-label="Legend">
        {RISK.map((r) => <li key={r} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: RISK_VAR[r] }} />{r}</li>)}
        <li className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm" style={{ background: "var(--risk-low)", backgroundImage: "repeating-linear-gradient(135deg, rgba(255,255,255,.6) 0 2px, transparent 2px 5px)" }} />Low-confidence tip</li>
        <li className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rotate-45 rounded-[2px] bg-warn" />Milestone</li>
      </ul>
    </div>
  );
});
