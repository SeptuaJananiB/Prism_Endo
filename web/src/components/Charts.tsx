import { memo, useId, useMemo, useRef, useState } from "react";

export const RISK = ["Unavailable", "Minimal", "Low", "Medium", "High"] as const;
export const RISK_VAR: Record<string, string> = {
  Unavailable: "var(--risk-unavailable)",
  Minimal: "var(--risk-minimal)",
  Low: "var(--risk-low)",
  Medium: "var(--risk-medium)",
  High: "var(--risk-high)",
};
const LINE_COLORS = ["var(--ink)", "var(--info)", "var(--warn)", "var(--ok)", "var(--err)"];

/** Risk category over time as one strip of merged segments; segments are buttons for keyboard + screen readers. */
export const RiskTimeline = memo(function RiskTimeline({ t, cat, onSeek }: { t: number[]; cat: string[]; onSeek: (i: number) => void }) {
  const segs = useMemo(() => {
    const out: { s: number; e: number; c: string; i: number }[] = [];
    let start = 0;
    for (let i = 1; i <= t.length; i++) {
      if (i === t.length || cat[i] !== cat[start]) {
        out.push({ s: t[start], e: i < t.length ? t[i] : t[t.length - 1], c: cat[start] || "Unavailable", i: start });
        start = i;
      }
    }
    return out;
  }, [t, cat]);
  if (!t.length) return null;
  const total = Math.max(t[t.length - 1] - t[0], 1e-6);
  const share = RISK.map((r) => ({ r, n: cat.filter((c) => (c || "Unavailable") === r).length }));
  return (
    <div>
      <div className="grow flex h-7 w-full overflow-hidden rounded-lg border border-line" role="list" aria-label="Risk over time">
        {segs.map((g, k) => (
          <button
            key={k}
            role="listitem"
            title={`${g.c}: ${g.s.toFixed(1)}–${g.e.toFixed(1)} s`}
            aria-label={`${g.c} from ${g.s.toFixed(1)} to ${g.e.toFixed(1)} seconds`}
            onClick={() => onSeek(g.i)}
            className="h-full border-0 p-0 transition-[filter,opacity] duration-150 hover:brightness-110 focus-visible:relative focus-visible:z-10"
            style={{ flex: Math.max(g.e - g.s, total / 1500), background: RISK_VAR[g.c] ?? RISK_VAR.Unavailable }}
          />
        ))}
      </div>
      <div className="mt-2 flex items-center justify-between text-xs text-muted tabular">
        <span>{t[0].toFixed(0)} s</span>
        <ul className="flex flex-wrap items-center gap-x-4 gap-y-1" aria-label="Legend">
          {share.map(({ r, n }) => (
            <li key={r} className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: RISK_VAR[r] }} aria-hidden />
              {r} <span className="text-muted/80">{Math.round((100 * n) / cat.length)}%</span>
            </li>
          ))}
        </ul>
        <span>{t[t.length - 1].toFixed(0)} s</span>
      </div>
    </div>
  );
});

type Line = { name: string; values: (number | null)[] };

/** Minimal SVG line chart: hover/keyboard crosshair, animated draw, accessible summary. */
export const LineChart = memo(function LineChart({ t, lines, unit = "", height = 220, onSeek, colorBy }: {
  t: number[]; lines: Line[]; unit?: string; height?: number; onSeek?: (i: number) => void;
  /** per-sample risk category: colours the first line and a faded area under it by risk level */
  colorBy?: string[];
}) {
  const uid = useId().replace(/:/g, "");
  const W = 1000, H = height, P = { l: 44, r: 12, t: 12, b: 26 };
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const all = lines.flatMap((l) => l.values.filter((v): v is number => v != null && Number.isFinite(v)));
  const max = all.length ? Math.max(...all) : 1;
  const min = Math.min(0, all.length ? Math.min(...all) : 0);
  const span = max - min || 1;
  const t0 = t[0] ?? 0, t1 = t[t.length - 1] ?? 1;
  const x = (v: number) => P.l + ((v - t0) / (t1 - t0 || 1)) * (W - P.l - P.r);
  const y = (v: number) => P.t + (1 - (v - min) / span) * (H - P.t - P.b);
  const paths = lines.map((l) => {
    let d = "", pen = false;
    l.values.forEach((v, i) => {
      if (v == null || !Number.isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(t[i]).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  });
  // risk-coloured horizontal gradient (hard stops at category changes) + area under the first line
  const stops = useMemo(() => {
    if (!colorBy?.length || t.length < 2) return [];
    const span_t = t[t.length - 1] - t[0] || 1;
    const out: { o: number; c: string }[] = [];
    let start = 0;
    for (let i = 1; i <= t.length; i++) {
      if (i === t.length || colorBy[i] !== colorBy[start]) {
        const c = RISK_VAR[colorBy[start]] ?? RISK_VAR.Unavailable;
        out.push({ o: (t[start] - t[0]) / span_t, c }, { o: ((i < t.length ? t[i] : t[t.length - 1]) - t[0]) / span_t, c });
        start = i;
      }
    }
    return out;
  }, [colorBy, t]);
  const area = useMemo(() => {
    if (!stops.length || !lines[0]) return "";
    const base = y(Math.max(min, 0));
    let d = "", run: [number, number][] = [];
    const flush = () => {
      if (run.length > 1) d += `M${run[0][0].toFixed(1)},${base}` + run.map(([a, b]) => `L${a.toFixed(1)},${b.toFixed(1)}`).join("") + `L${run[run.length - 1][0].toFixed(1)},${base}Z`;
      run = [];
    };
    lines[0].values.forEach((v, i) => (v == null || !Number.isFinite(v) ? flush() : run.push([x(t[i]), y(v)])));
    flush();
    return d;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stops, lines, t, min, span]);
  const strokeFor = (k: number) => (k === 0 && stops.length ? `url(#h${uid})` : LINE_COLORS[k % LINE_COLORS.length]);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((k) => min + k * span);
  const xticks = Array.from({ length: 6 }, (_, k) => t0 + (k / 5) * (t1 - t0));
  const pick = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r || !t.length) return;
    const tv = t0 + ((clientX - r.left) / r.width * W - P.l) / (W - P.l - P.r) * (t1 - t0);
    let best = 0;
    for (let i = 0; i < t.length; i++) if (Math.abs(t[i] - tv) < Math.abs(t[best] - tv)) best = i;
    setHover(best);
  };
  if (!all.length) return <p className="py-10 text-center text-muted">No data for this view.</p>;
  const flat = max === min;
  return (
    <figure className="m-0">
      {flat && (
        <p className="flat-note m-0 mb-2 flex items-center gap-2 text-[13px] text-muted">
          <span className="chip chip-mute">Flat</span>All values are {max.toFixed(0)}{unit} throughout this analysis.
        </p>
      )}
      <svg
        ref={ref}
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full cursor-crosshair select-none"
        role="img"
        aria-label={`${lines.map((l) => l.name).join(", ")} over time, range ${min.toFixed(1)} to ${max.toFixed(1)}${unit}`}
        tabIndex={0}
        onMouseMove={(e) => pick(e.clientX)}
        onMouseLeave={() => setHover(null)}
        onClick={() => hover != null && onSeek?.(hover)}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") setHover((h) => Math.min(t.length - 1, (h ?? -1) + Math.max(1, Math.round(t.length / 100))));
          if (e.key === "ArrowLeft") setHover((h) => Math.max(0, (h ?? 1) - Math.max(1, Math.round(t.length / 100))));
          if (e.key === "Enter" && hover != null) onSeek?.(hover);
        }}
      >
        {ticks.map((v, k) => (
          <g key={k}>
            <line x1={P.l} x2={W - P.r} y1={y(v)} y2={y(v)} stroke="var(--line)" strokeDasharray={k ? "3 4" : undefined} />
            <text x={P.l - 8} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--muted)" className="tabular">
              {v.toFixed(span < 5 ? 1 : 0)}
            </text>
          </g>
        ))}
        {xticks.map((v, k) => (
          <text key={k} x={x(v)} y={H - 6} textAnchor="middle" fontSize="11" fill="var(--muted)" className="tabular">
            {v.toFixed(0)} s
          </text>
        ))}
        {stops.length > 0 && (
          <defs>
            <linearGradient id={`h${uid}`} gradientUnits="userSpaceOnUse" x1={P.l} x2={W - P.r} y1="0" y2="0">
              {stops.map((g, k) => <stop key={k} offset={g.o} stopColor={g.c} />)}
            </linearGradient>
            <linearGradient id={`v${uid}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0" stopColor="#fff" stopOpacity="0.42" />
              <stop offset="0.55" stopColor="#fff" stopOpacity="0.14" />
              <stop offset="1" stopColor="#fff" stopOpacity="0" />
            </linearGradient>
            <mask id={`m${uid}`} maskContentUnits="userSpaceOnUse">
              <rect x="0" y={P.t} width={W} height={H - P.t - P.b} fill={`url(#v${uid})`} />
            </mask>
          </defs>
        )}
        {area && <path d={area} fill={`url(#h${uid})`} mask={`url(#m${uid})`} className="fade" />}
        {paths.map((d, k) => (
          <path key={`${k}-${d.length}`} d={d} fill="none" stroke={strokeFor(k)} strokeWidth="1.6"
            strokeLinejoin="round" pathLength={1} className="draw" style={{ ["--len" as string]: 1 }} />
        ))}
        {hover != null && (
          <g className="fade" pointerEvents="none">
            <line x1={x(t[hover])} x2={x(t[hover])} y1={P.t} y2={H - P.b} stroke="var(--line-strong)" />
            {lines.map((l, k) => l.values[hover] != null && (
              <circle key={k} cx={x(t[hover])} cy={y(l.values[hover] as number)} r="4" fill="var(--surface)"
                stroke={k === 0 && colorBy ? RISK_VAR[colorBy[hover]] ?? RISK_VAR.Unavailable : LINE_COLORS[k % LINE_COLORS.length]} strokeWidth="2" />
            ))}
          </g>
        )}
      </svg>
      <figcaption className="mt-2 flex min-h-6 flex-wrap items-center gap-x-5 gap-y-1 text-xs text-muted" aria-live="polite">
        {lines.map((l, k) => (
          <span key={l.name} className="flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-3 rounded" style={{ background: LINE_COLORS[k % LINE_COLORS.length] }} aria-hidden />
            {l.name}
            {hover != null && l.values[hover] != null && (
              <b className="font-medium text-ink tabular">{(l.values[hover] as number).toFixed(2)}{unit}</b>
            )}
          </span>
        ))}
        {hover != null && <span className="ml-auto tabular">t = {t[hover].toFixed(2)} s · click to inspect</span>}
      </figcaption>
    </figure>
  );
});
