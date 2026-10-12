import { useEffect, useRef, useState } from "react";
import type { Episode } from "../lib/derive";
import { FLAG_TONE, type FlagT } from "./StackedTimeline";

export function Card({ title, aside, children, className = "", i = 0 }: {
  title: string; aside?: React.ReactNode; children: React.ReactNode; className?: string; i?: number;
}) {
  return (
    <section className={`card rise flex min-w-0 flex-col ${className}`} style={{ ["--i" as string]: i }} aria-label={title}>
      <header className="flex h-11 shrink-0 items-center justify-between gap-3 border-b border-line px-4">
        <h2 className="label m-0">{title}</h2>
        {aside}
      </header>
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  );
}

const fmt = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

export function EventsPanel({ events, onSeek, current }: { events: Episode[]; onSeek: (i: number) => void; current: number }) {
  const [filter, setFilter] = useState<string>("All");
  const kinds = ["All", ...Array.from(new Set(events.map((e) => e.kind)))];
  const shown = events.filter((e) => filter === "All" || e.kind === filter);
  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap gap-1.5 border-b border-line px-4 py-2.5" role="group" aria-label="Filter events">
        {kinds.map((k) => (
          <button key={k} onClick={() => setFilter(k)} aria-pressed={filter === k}
            className={`chip cursor-pointer transition-colors ${filter === k ? "border-ink bg-ink text-bg" : "chip-mute hover:border-line-strong"}`}>{k}</button>
        ))}
      </div>
      {shown.length === 0 ? (
        <p className="m-0 px-4 py-8 text-center text-[13px] text-muted">No events of this type.</p>
      ) : (
        <ol className="m-0 max-h-[300px] list-none overflow-y-auto p-1.5">
          {shown.map((e, k) => {
            const active = current >= e.i && e.end >= 0;
            return (
              <li key={k}>
                <button onClick={() => onSeek(e.i)}
                  className="group flex w-full items-center gap-3 rounded-md px-2.5 py-2 text-left transition-colors hover:bg-surface-2">
                  <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: `var(--${e.tone})` }} aria-hidden />
                  <span className="flex-1 text-[13px] font-medium">{e.kind}</span>
                  <span className="text-xs text-muted tabular">{fmt(e.start)} · {(e.end - e.start).toFixed(1)} s</span>
                  <svg className={`text-muted transition-transform duration-150 group-hover:translate-x-0.5 ${active ? "" : ""}`} width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden><path d="m9 6 6 6-6 6" /></svg>
                </button>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

export function ReportPanel({ bullets, onSeek, onPrint }: { bullets: { text: string; i?: number }[]; onSeek: (i: number) => void; onPrint: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <ul className="m-0 flex-1 list-none space-y-2.5 p-4">
        {bullets.map((b, k) => (
          <li key={k} className="rise flex gap-2.5 text-[13px] leading-relaxed text-ink-2" style={{ ["--i" as string]: k }}>
            <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-muted" aria-hidden />
            <span>
              {b.text}{" "}
              {b.i != null && <button className="text-info underline-offset-2 hover:underline" onClick={() => onSeek(b.i!)}>View</button>}
            </span>
          </li>
        ))}
      </ul>
      <div className="flex items-center justify-between gap-3 border-t border-line px-4 py-3">
        <span className="min-w-0 text-xs leading-snug text-muted">Automatic estimate, not a clinical report.</span>
        <button className="btn h-8 shrink-0 whitespace-nowrap px-3 text-[13px]" onClick={onPrint}>Print / PDF</button>
      </div>
    </div>
  );
}

export function FlagsPanel({ flags, t, frame, onAdd, onRemove, onSeek, saving }: {
  flags: FlagT[]; t: number; frame: number; onAdd: (f: FlagT) => void; onRemove: (k: number) => void; onSeek: (i: number) => void; saving: boolean;
}) {
  const [note, setNote] = useState("");
  const add = (kind: FlagT["kind"]) => { onAdd({ t, frame, kind, note: note.trim() }); setNote(""); };
  return (
    <div className="flex h-full flex-col">
      <div className="space-y-2.5 border-b border-line p-4">
        <label htmlFor="flag-note" className="block text-[13px] font-medium text-ink-2">Flag the current moment <span className="font-normal text-muted tabular">({t.toFixed(1)} s)</span></label>
        <input id="flag-note" className="field h-9" placeholder="Optional note, e.g. tip label on wrong instrument" value={note}
          onChange={(e) => setNote(e.target.value)} onKeyDown={(e) => e.key === "Enter" && add("important")} />
        <div className="grid grid-cols-3 gap-2">
          <button className="btn h-8 px-2 text-[13px]" onClick={() => add("agree")}><span className="h-2 w-2 rounded-full" style={{ background: FLAG_TONE.agree }} />Agree</button>
          <button className="btn h-8 px-2 text-[13px]" onClick={() => add("wrong")}><span className="h-2 w-2 rounded-full" style={{ background: FLAG_TONE.wrong }} />AI wrong</button>
          <button className="btn h-8 px-2 text-[13px]" onClick={() => add("important")}><span className="h-2 w-2 rounded-full" style={{ background: FLAG_TONE.important }} />Important</button>
        </div>
      </div>
      <ol className="m-0 max-h-[220px] flex-1 list-none overflow-y-auto p-1.5" aria-live="polite">
        {flags.length === 0 && <li className="px-3 py-6 text-center text-[13px] text-muted">No flags yet. Your flags are saved with this analysis and build the human-labelled set.</li>}
        {flags.map((f, k) => (
          <li key={k} className="fade group flex items-center gap-2 rounded-md px-2.5 py-1.5 hover:bg-surface-2">
            <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: FLAG_TONE[f.kind] }} aria-hidden />
            <button className="min-w-0 flex-1 truncate text-left text-[13px]" onClick={() => onSeek(f.frame)}>
              <span className="font-medium capitalize">{f.kind === "wrong" ? "AI wrong" : f.kind}</span>
              <span className="text-muted tabular"> · {f.t.toFixed(1)} s</span>{f.note && <span className="text-muted"> · {f.note}</span>}
            </button>
            <button className="rounded p-1 text-muted opacity-0 transition-opacity hover:text-err group-hover:opacity-100 focus-visible:opacity-100"
              onClick={() => onRemove(k)} aria-label={`Remove flag at ${f.t.toFixed(1)} seconds`}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden><path d="M18 6 6 18M6 6l12 12" /></svg>
            </button>
          </li>
        ))}
      </ol>
      <div className="border-t border-line px-4 py-2 text-xs text-muted" aria-live="polite">{saving ? "Saving…" : `${flags.length} saved`}</div>
    </div>
  );
}

export function SettingsPanel({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-6 gap-y-2.5 p-4 text-[13px]">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-muted">{k}</dt>
          <dd className="m-0 min-w-0 truncate text-right font-medium tabular">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function DownloadMenu({ run }: { run: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const h = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);
  const items: [string, string, string][] = [
    ["Annotated video", "MP4", `/api/runs/${encodeURIComponent(run)}/file/video`],
    ["Per-frame results", "CSV", `/api/runs/${encodeURIComponent(run)}/file/csv`],
    ["Run summary", "JSON", `/api/runs/${encodeURIComponent(run)}/file/summary`],
  ];
  return (
    <div className="relative" ref={ref} onKeyDown={(e) => e.key === "Escape" && setOpen(false)}>
      <button className="btn h-9 px-3 text-[13px]" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden><path d="M12 3v12m0 0-4-4m4 4 4-4M4 19h16" /></svg>
        Download
      </button>
      {open && (
        <div role="menu" className="card rise absolute right-0 top-11 z-30 w-60 p-1.5 shadow-[var(--shadow-hover)]">
          {items.map(([l, ext, href]) => (
            <a key={l} role="menuitem" href={href} download onClick={() => setOpen(false)}
              className="flex items-center justify-between rounded-md px-3 py-2 text-[13px] text-ink no-underline hover:bg-surface-2">
              {l}<span className="chip chip-mute">{ext}</span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}

export const SHORTCUTS: [string, string][] = [
  ["Space", "Play / pause"], ["D / F", "Previous / next frame"], ["Shift + D / F", "Back / forward 1 s"],
  ["J / K", "Previous / next event"], ["1 / 2 / 3", "Flag: agree / AI wrong / important"],
  ["T", "Toggle light / dark"], ["?", "Show this sheet"], ["Esc", "Close dialogs"],
];

export function ShortcutsDialog({ open, req = 0, onClose }: { open: boolean; req?: number; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open, req]); // req: a fresh "open" request even if React merged close+open into no change
  // keep React state in sync however the dialog closes (Esc, backdrop, button)
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    const h = () => { if (!d.open) onClose(); }; // a late "close" after a quick reopen must not close it again
    d.addEventListener("close", h);
    return () => d.removeEventListener("close", h);
  }, [onClose]);
  return (
    <dialog ref={ref} onClick={(e) => e.target === ref.current && onClose()}
      className="card m-auto w-[420px] max-w-[92vw] p-0 text-ink backdrop:bg-black/30 backdrop:backdrop-blur-[2px] open:animate-[rise_.35s_var(--ease)]"
      aria-labelledby="sc-title">
      <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
        <h2 id="sc-title" className="m-0 text-[15px] font-semibold">Keyboard shortcuts</h2>
        <button className="btn h-8 w-8 p-0" onClick={onClose} aria-label="Close">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden><path d="M18 6 6 18M6 6l12 12" /></svg>
        </button>
      </div>
      <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-6 gap-y-2.5 p-5 text-[13px]">
        {SHORTCUTS.map(([k, v]) => (
          <div key={k} className="contents">
            <dt>{k.split(" / ").map((x, i) => <span key={i}>{i > 0 && <span className="mx-1 text-muted">/</span>}<kbd className="rounded border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-xs">{x}</kbd></span>)}</dt>
            <dd className="m-0 text-ink-2">{v}</dd>
          </div>
        ))}
      </dl>
    </dialog>
  );
}

export type Toast = { id: number; tone: "ok" | "err" | "info"; text: string; action?: { label: string; run: () => void } };

export function Toasts({ items, dismiss }: { items: Toast[]; dismiss: (id: number) => void }) {
  return (
    <div className="pointer-events-none fixed bottom-5 right-5 z-50 flex w-[340px] max-w-[calc(100vw-40px)] flex-col gap-2" aria-live="polite" aria-relevant="additions">
      {items.map((t) => (
        <div key={t.id} role="status" className="card rise pointer-events-auto flex items-start gap-3 p-3.5 shadow-[var(--shadow-hover)]">
          <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full" style={{ background: `var(--${t.tone})` }} aria-hidden />
          <p className="m-0 flex-1 text-[13px] leading-snug">{t.text}</p>
          {t.action && <button className="btn h-7 px-2.5 text-xs" onClick={() => { t.action!.run(); dismiss(t.id); }}>{t.action.label}</button>}
          <button className="text-muted hover:text-ink" onClick={() => dismiss(t.id)} aria-label="Dismiss">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden><path d="M18 6 6 18M6 6l12 12" /></svg>
          </button>
        </div>
      ))}
    </div>
  );
}

export function EmptyState({ onFocusForm }: { onFocusForm: () => void }) {
  return (
    <div className="rise mx-auto flex max-w-md flex-col items-center py-24 text-center">
      <svg width="120" height="88" viewBox="0 0 120 88" fill="none" aria-hidden>
        <rect x="8" y="8" width="104" height="64" rx="10" stroke="var(--line-strong)" strokeWidth="2" />
        <circle cx="60" cy="40" r="16" fill="var(--info-bg)" stroke="var(--info)" strokeWidth="2" />
        <path d="m55 33 12 7-12 7z" fill="var(--info)" />
        <path d="M20 80h80" stroke="var(--line)" strokeWidth="6" strokeLinecap="round" />
        <path d="M20 80h34" stroke="var(--risk-medium)" strokeWidth="6" strokeLinecap="round" />
      </svg>
      <h2 className="mt-6 text-[18px] font-semibold tracking-tight">No analyses yet</h2>
      <p className="mt-2 text-[14px] leading-relaxed text-muted">Upload a laparoscopic video or pick one from the list, choose the surgery type, and start an analysis. Results appear here.</p>
      <button className="btn btn-primary mt-6" onClick={onFocusForm}>Start your first analysis</button>
    </div>
  );
}
