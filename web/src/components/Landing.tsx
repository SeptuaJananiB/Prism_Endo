import type { RunSummary } from "../lib/api";

export function Logo({ size = 34 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
      <rect width="32" height="32" rx="8" fill="var(--ink)" />
      <circle cx="16" cy="16" r="7" fill="none" stroke="var(--bg)" strokeWidth="2.5" />
      <circle cx="16" cy="16" r="2.5" fill="var(--bg)" />
    </svg>
  );
}

/** Full-screen loading state shown while the app fetches its first data. */
export function Splash() {
  return (
    <div className="fixed inset-0 z-[60] grid place-items-center bg-bg" role="status" aria-live="polite" aria-label="Loading PRISM-Endo">
      <div className="flex flex-col items-center gap-5">
        <div className="relative grid place-items-center">
          <span className="absolute h-20 w-20 animate-ping rounded-2xl bg-ink/10 [animation-duration:1.6s]" aria-hidden />
          <Logo size={56} />
        </div>
        <div className="text-center">
          <div className="text-[18px] font-semibold tracking-tight">PRISM-Endo</div>
          <div className="mt-1 text-[13px] text-muted">Loading analyses and models…</div>
        </div>
        <div className="h-1 w-48 overflow-hidden rounded-full bg-line" aria-hidden>
          <div className="h-full w-1/3 rounded-full bg-ink animate-[slide_1.1s_var(--ease)_infinite]" />
        </div>
      </div>
    </div>
  );
}

const FEATURES: [string, string, React.ReactNode][] = [
  ["Instrument tracking", "Grasper and L-hook masks and tips in every frame, with motion and speed.",
    <path key="i" d="M4 20 14 10m0 0 2-6 4 4-6 2Z" />],
  ["Tissue proximity", "Distance from each tip to the tissues you choose, with tested / not-tested flags.",
    <><circle key="a" cx="12" cy="12" r="8" /><circle key="b" cx="12" cy="12" r="3" /></>],
  ["Explainable risk", "A risk timeline with a plain-language reason for every moment.",
    <path key="r" d="M3 17 9 11l4 4 8-8M15 7h6v6" />],
  ["Bleeding & safety", "Learned bleeding detection and Critical View of Safety estimates.",
    <path key="s" d="M12 3s6 6.5 6 11a6 6 0 0 1-12 0c0-4.5 6-11 6-11Z" />],
  ["AI explanation", "gpt-oss turns the numbers into a readable summary of what happened in the video.",
    <path key="x" d="M12 3v3m0 12v3M3 12h3m12 0h3M6 6l2 2m8 8 2 2m0-12-2 2M8 16l-2 2" />],
  ["Review tools", "Synced video and timeline, events, reviewer flags, compare, export and print.",
    <path key="v" d="M4 6h16M4 12h10M4 18h7" />],
];

export function Landing({ runs, onEnter, onOpenRun }: { runs: RunSummary[] | null; onEnter: () => void; onOpenRun: (n: string) => void }) {
  const recent = (runs ?? []).slice(0, 3);
  const frames = (runs ?? []).reduce((a, r) => a + (r.frames ?? 0), 0);
  return (
    <div className="min-h-dvh overflow-y-auto bg-bg">
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(60%_50%_at_70%_0%,color-mix(in_srgb,var(--info)_9%,transparent),transparent)]" aria-hidden />
      <header className="relative mx-auto flex h-20 max-w-[1180px] items-center justify-between px-6">
        <div className="flex items-center gap-3"><Logo /><span className="text-[16px] font-semibold tracking-tight">PRISM-Endo</span></div>
        <button className="btn h-10 px-4" onClick={onEnter}>Open workspace</button>
      </header>

      <main className="relative mx-auto max-w-[1180px] px-6 pb-24">
        <section className="grid items-center gap-12 pt-10 lg:grid-cols-[1.1fr_1fr] lg:pt-16">
          <div className="rise">
            <span className="chip chip-info">Surgical video analysis</span>
            <h1 className="mt-5 text-[44px] font-semibold leading-[1.05] tracking-[-0.03em] sm:text-[56px]">
              See every instrument,<br />every risk moment.
            </h1>
            <p className="mt-5 max-w-[540px] text-[17px] leading-relaxed text-ink-2">
              PRISM-Endo analyses laparoscopic video frame by frame — instruments, tissue proximity, bleeding and the
              Critical View of Safety — and explains what it found in plain language.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <button className="btn btn-primary h-12 px-6 text-[15px]" onClick={onEnter}>
                Start analysing
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden><path d="M5 12h14m-6-6 6 6-6 6" /></svg>
              </button>
              {recent[0] && <button className="btn h-12 px-6 text-[15px]" onClick={() => onOpenRun(recent[0].name)}>Open latest analysis</button>}
            </div>
            <dl className="mt-10 grid max-w-[480px] grid-cols-3 gap-6">
              {[["Analyses", (runs?.length ?? 0).toLocaleString()], ["Frames analysed", frames.toLocaleString()], ["Signals per frame", "8"]].map(([k, v]) => (
                <div key={k}><dt className="label">{k}</dt><dd className="m-0 mt-1 text-[24px] font-semibold tabular">{v}</dd></div>
              ))}
            </dl>
          </div>

          <div className="rise card overflow-hidden p-0" style={{ ["--i" as string]: 2 }} aria-hidden>
            <div className="flex h-9 items-center gap-1.5 border-b border-line px-4">{[0, 1, 2].map((k) => <i key={k} className="h-2.5 w-2.5 rounded-full bg-line-strong" />)}</div>
            <div className="space-y-4 p-5">
              <div className="grid grid-cols-3 gap-3">
                {[["Medium+ risk", "49%", "var(--warn)"], ["Bleeding", "0", "var(--ok)"], ["CVS met", "0%", "var(--ink)"]].map(([k, v, c]) => (
                  <div key={k} className="rounded-lg border border-line p-3"><div className="label">{k}</div><div className="mt-1 text-[20px] font-semibold" style={{ color: c }}>{v}</div></div>
                ))}
              </div>
              <div className="flex h-6 overflow-hidden rounded-md">
                {["minimal", "low", "medium", "medium", "low", "high", "medium", "low", "minimal", "medium", "low", "medium"].map((r, k) => (
                  <span key={k} className="grow" style={{ flex: 1 + (k % 3), background: `var(--risk-${r})` }} />
                ))}
              </div>
              <svg viewBox="0 0 300 80" className="h-20 w-full">
                <defs><linearGradient id="lg" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="var(--risk-medium)" stopOpacity=".35" /><stop offset="1" stopColor="var(--risk-medium)" stopOpacity="0" /></linearGradient></defs>
                <path d="M0 60 L20 40 L35 50 L55 22 L70 35 L90 30 L110 55 L130 45 L150 15 L170 30 L190 26 L210 50 L230 38 L250 44 L270 20 L300 34 L300 80 L0 80Z" fill="url(#lg)" />
                <path d="M0 60 L20 40 L35 50 L55 22 L70 35 L90 30 L110 55 L130 45 L150 15 L170 30 L190 26 L210 50 L230 38 L250 44 L270 20 L300 34" fill="none" stroke="var(--risk-medium)" strokeWidth="2" pathLength={1} className="draw" style={{ ["--len" as string]: 1 }} />
              </svg>
              <div className="rounded-lg bg-surface-2 p-3 text-[13px] leading-relaxed text-ink-2">
                <b className="text-ink">AI explanation</b> — The grasper worked close to the liver for most of the clip; risk peaked at 0:27.9 when the tip moved fast towards it…
              </div>
            </div>
          </div>
        </section>

        <section className="mt-24" aria-labelledby="features-h">
          <h2 id="features-h" className="label">What it does</h2>
          <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map(([t, d, icon], k) => (
              <div key={t} className="card card-hover rise p-6" style={{ ["--i" as string]: k }}>
                <span className="grid h-10 w-10 place-items-center rounded-lg bg-surface-2 text-ink">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden>{icon}</svg>
                </span>
                <h3 className="mt-4 text-[15px] font-semibold">{t}</h3>
                <p className="mt-1.5 text-[14px] leading-relaxed text-muted">{d}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="mt-24 grid gap-4 lg:grid-cols-3" aria-labelledby="how-h">
          <h2 id="how-h" className="label lg:col-span-3">How it works</h2>
          {[["1", "Add a video", "Upload a laparoscopic video or pick one from the library, then choose the surgery type."],
            ["2", "Analyse", "Every frame runs through segmentation, tip tracking, proximity, bleeding and safety models."],
            ["3", "Review", "Explore the synced timeline and charts, read the AI explanation, flag moments, export."]].map(([n, t, d]) => (
            <div key={n} className="flex gap-4">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-line-strong text-[14px] font-semibold tabular">{n}</span>
              <div><h3 className="m-0 text-[15px] font-semibold">{t}</h3><p className="mt-1 text-[14px] leading-relaxed text-muted">{d}</p></div>
            </div>
          ))}
        </section>

        {recent.length > 0 && (
          <section className="mt-24" aria-labelledby="recent-h">
            <div className="flex items-end justify-between"><h2 id="recent-h" className="label m-0">Recent analyses</h2>
              <button className="text-[13px] font-medium text-info hover:underline" onClick={onEnter}>See all</button></div>
            <div className="mt-4 grid gap-4 md:grid-cols-3">
              {recent.map((r) => (
                <button key={r.name} onClick={() => onOpenRun(r.name)} className="card card-hover group overflow-hidden p-0 text-left">
                  <div className="aspect-video overflow-hidden bg-black">
                    <img src={`/api/runs/${encodeURIComponent(r.name)}/frame/${Math.floor((r.frames ?? 2) / 2)}?w=480&kind=original`} alt=""
                      onError={(e) => { e.currentTarget.style.visibility = "hidden"; }}
                      className="h-full w-full object-cover transition-transform duration-500 group-hover:scale-[1.03]" />
                  </div>
                  <div className="p-4">
                    <div className="truncate text-[14px] font-semibold">{r.name}</div>
                    <div className="mt-1 text-[12px] text-muted tabular">{r.frames?.toLocaleString()} frames{r.hi_pct != null ? ` · ${r.hi_pct}% medium+ risk` : ""}</div>
                  </div>
                </button>
              ))}
            </div>
          </section>
        )}

        <footer className="mt-24 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-6 text-[13px] text-muted">
          <span className="flex items-center gap-2"><Logo size={20} />PRISM-Endo</span>
          <span>Measurements are image-based estimates; not validated for clinical use.</span>
        </footer>
      </main>
    </div>
  );
}
