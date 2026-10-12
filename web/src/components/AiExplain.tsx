import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Speak } from "./Speak";

/** Minimal, safe Markdown -> React (headings, bullets, bold, italics, inline code). No raw HTML is ever injected. */
function inline(s: string, key: string) {
  const parts = s.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g).filter(Boolean);
  return parts.map((p, i) =>
    p.startsWith("**") ? <strong key={`${key}-${i}`} className="font-semibold text-ink">{p.slice(2, -2)}</strong>
      : p.startsWith("`") ? <code key={`${key}-${i}`} className="rounded bg-surface-2 px-1 text-[12px]">{p.slice(1, -1)}</code>
        : p.startsWith("*") ? <em key={`${key}-${i}`}>{p.slice(1, -1)}</em> : <span key={`${key}-${i}`}>{p}</span>);
}

export function Markdown({ text }: { text: string }) {
  const out: React.ReactNode[] = [];
  let list: string[] = [];
  const flush = (k: number) => {
    if (list.length) out.push(<ul key={`u${k}`} className="m-0 mb-3 list-none space-y-1.5 p-0">{list.map((l, i) => (
      <li key={i} className="flex gap-2.5"><span className="mt-[8px] h-1.5 w-1.5 shrink-0 rounded-full bg-muted" aria-hidden /><span>{inline(l, `l${k}${i}`)}</span></li>))}</ul>);
    list = [];
  };
  text.split("\n").forEach((raw, k) => {
    const l = raw.trimEnd();
    const m = l.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)$/);
    if (m) { list.push(m[1]); return; }
    flush(k);
    const h = l.replace(/^\*\*\s*/, "").replace(/\s*\*\*\s*$/, "");
    if (/^#{1,4}\s/.test(h)) out.push(<h3 key={k} className="mb-2 mt-4 text-[13px] font-semibold uppercase tracking-[0.06em] text-muted first:mt-0">{h.replace(/^#+\s*/, "")}</h3>);
    else if (l.trim()) out.push(<p key={k} className="m-0 mb-3">{inline(l, `p${k}`)}</p>);
  });
  flush(-1);
  return <div className="text-[14px] leading-relaxed text-ink-2">{out}</div>;
}

export function AiExplain({ run }: { run: string }) {
  const [status, setStatus] = useState<{ configured: boolean; model: string } | null>(null);
  const [res, setRes] = useState<{ text: string; model: string; created: number; cached: boolean } | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.llmStatus().then(setStatus).catch(() => setStatus({ configured: false, model: "" })); }, []);
  useEffect(() => { setRes(null); setErr(null); }, [run]);
  const go = async (refresh = false) => {
    setBusy(true); setErr(null);
    try { setRes(await api.explain(run, refresh)); } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  const model = (res?.model ?? status?.model ?? "").replace(/^openai\//, "");

  return (
    <div className="flex min-h-full flex-col">
      <div className="min-h-[120px] flex-1 p-5" aria-live="polite" aria-busy={busy}>
        {status && !status.configured && (
          <div className="space-y-2 text-[13px] leading-relaxed text-ink-2">
            <p className="m-0">Add a free Groq API key to turn this analysis into a written explanation with <b>gpt-oss</b>:</p>
            <ol className="m-0 list-decimal space-y-1 pl-5">
              <li>Create a key at <span className="font-medium">console.groq.com/keys</span></li>
              <li>Copy <code className="rounded bg-surface-2 px-1">.env.example</code> to <code className="rounded bg-surface-2 px-1">.env</code> in the project folder and paste the key after <code className="rounded bg-surface-2 px-1">GROQ_API_KEY=</code></li>
              <li>Restart the app</li>
            </ol>
          </div>
        )}
        {status?.configured && !res && !busy && !err && (
          <div className="flex h-full flex-col items-start justify-center gap-3">
            <p className="m-0 text-[14px] leading-relaxed text-ink-2">Get a plain-language explanation of this analysis: what happened in the video, key moments, risk and safety, and how far to trust it.</p>
            <button className="btn btn-primary" onClick={() => go()}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M12 3v3m0 12v3M3 12h3m12 0h3M6 6l2 2m8 8 2 2m0-12-2 2M8 16l-2 2" /></svg>
              Explain this analysis
            </button>
          </div>
        )}
        {busy && (
          <div className="space-y-3" aria-label="Writing explanation">
            <div className="flex items-center gap-2 text-[13px] text-muted"><span className="live-dot" aria-hidden />{model} is writing the explanation…</div>
            {[92, 80, 86, 60, 74].map((w, k) => <div key={k} className="skeleton h-3.5" style={{ width: `${w}%` }} />)}
          </div>
        )}
        {err && <p role="alert" className="m-0 rounded-lg border border-[var(--err-line)] bg-[var(--err-bg)] px-3 py-2 text-[13px] text-err">{err}</p>}
        {res && !busy && <div className="fade space-y-4"><Speak text={res.text} /><Markdown text={res.text} /></div>}
      </div>
      {(res || err) && (
        <div className="sticky bottom-0 flex items-center justify-between gap-3 border-t border-line bg-surface px-5 py-3 text-xs text-muted">
          <span>{res ? `Written by ${model} from the analysis numbers only (no video sent)${res.cached ? " · saved copy" : ""}.` : ""} AI text can be wrong.</span>
          <button className="btn h-8 shrink-0 px-3 text-[13px]" onClick={() => go(true)} disabled={busy}>Regenerate</button>
        </div>
      )}
    </div>
  );
}
