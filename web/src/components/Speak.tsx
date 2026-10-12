import { useEffect, useMemo, useRef, useState } from "react";
import { Select } from "./Select";

/** "0","06.4" -> "6.4 seconds"; "1","12" -> "1 minute 12 seconds". */
function spokenTime(min: string, sec: string, unit = true): string {
  const m = Number(min), s = Number(sec);
  const secs = `${s}${unit ? (s === 1 ? " second" : " seconds") : ""}`;
  if (m === 0) return secs;
  return `${m} ${m === 1 ? "minute" : "minutes"}${s ? ` ${secs}` : ""}`;
}

/** Markdown -> sentences that sound natural when read aloud. */
export function speechSentences(md: string): string[] {
  const txt = md
    .split("\n")
    .map((l) => l.replace(/^\s*\*\*\s*|\s*\*\*\s*$/g, "").trim())
    .map((l) => (/^#{1,4}\s/.test(l) ? `${l.replace(/^#+\s*/, "")}.` : l))
    .map((l) => l.replace(/^\s*(?:[-*•]|\d+\.)\s+/, ""))
    .join(" ")
    .replace(/[*_`#]/g, "")
    .replace(/(\d+):(\d{2}(?:\.\d)?)\s*[-–‑]\s*(\d+):(\d{2}(?:\.\d)?)/g, (_m, a, b, c, d) =>
      a === "0" && c === "0" ? `${spokenTime(a, b, false)} to ${spokenTime(c, d)}` : `${spokenTime(a, b)} to ${spokenTime(c, d)}`)
    .replace(/\b(\d+):(\d{2}(?:\.\d)?)\b/g, (_m, a, b) => spokenTime(a, b))
    .replace(/(\d)\s?px\b/g, "$1 pixels")
    .replace(/(\d)\s?%/g, "$1 percent")
    .replace(/\bCVS\b/g, "Critical View of Safety")
    .replace(/[≈~]/g, "about ")
    .replace(/\s+/g, " ")
    .trim();
  return txt.split(/(?<=[.!?])\s+(?=[A-Z0-9])/).map((s) => s.trim()).filter(Boolean);
}

const RATES = [0.8, 1, 1.15, 1.3, 1.5];

/** Read text aloud with the browser's built-in speech engine (offline, nothing is sent anywhere). */
export function Speak({ text }: { text: string }) {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window;
  const sentences = useMemo(() => speechSentences(text), [text]);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [voice, setVoice] = useState<string>(() => { try { return localStorage.getItem("pe-voice") ?? ""; } catch { return ""; } });
  const [rate, setRate] = useState(1);
  const [state, setState] = useState<"idle" | "playing" | "paused">("idle");
  const [idx, setIdx] = useState(-1);
  const stopped = useRef(false);

  useEffect(() => {
    if (!supported) return;
    const load = () => setVoices(speechSynthesis.getVoices().filter((v) => v.lang.toLowerCase().startsWith("en")));
    load();
    speechSynthesis.addEventListener("voiceschanged", load);
    return () => { speechSynthesis.removeEventListener("voiceschanged", load); speechSynthesis.cancel(); };
  }, [supported]);
  useEffect(() => { if (supported) { stopped.current = true; speechSynthesis.cancel(); setState("idle"); setIdx(-1); } }, [text, supported]);
  useEffect(() => { try { if (voice) localStorage.setItem("pe-voice", voice); } catch { /* storage blocked */ } }, [voice]);

  // one utterance per sentence: avoids Chrome cutting long utterances and lets us highlight progress
  const speakFrom = (i: number) => {
    if (i >= sentences.length || stopped.current) { setState("idle"); setIdx(-1); return; }
    const u = new SpeechSynthesisUtterance(sentences[i]);
    const v = voices.find((x) => x.name === voice) ?? voices.find((x) => /natural|online|google/i.test(x.name)) ?? voices[0];
    if (v) u.voice = v;
    u.rate = rate;
    u.onstart = () => setIdx(i);
    u.onend = () => speakFrom(i + 1);
    u.onerror = (e) => { if (e.error !== "interrupted" && e.error !== "canceled") { setState("idle"); setIdx(-1); } };
    speechSynthesis.speak(u);
  };
  const play = () => {
    if (state === "paused") { speechSynthesis.resume(); setState("playing"); return; }
    stopped.current = false;
    speechSynthesis.cancel();
    setState("playing");
    speakFrom(0);
  };
  const pause = () => { speechSynthesis.pause(); setState("paused"); };
  const stop = () => { stopped.current = true; speechSynthesis.cancel(); setState("idle"); setIdx(-1); };

  if (!supported) return <p className="m-0 text-xs text-muted">Read-aloud is not supported in this browser.</p>;
  const pct = idx >= 0 ? Math.round((100 * (idx + 1)) / sentences.length) : 0;

  return (
    <div className="rounded-xl border border-line bg-surface-2 p-3" role="group" aria-label="Read aloud">
      <div className="flex flex-wrap items-center gap-2">
        {state === "playing"
          ? <button className="btn h-9 px-3 text-[13px]" onClick={pause} aria-label="Pause reading">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>Pause</button>
          : <button className="btn btn-primary h-9 px-3 text-[13px]" onClick={play} aria-label={state === "paused" ? "Resume reading" : "Read explanation aloud"}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden><path d="M11 5 6 9H3v6h3l5 4V5Z" /><path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14" /></svg>
              {state === "paused" ? "Resume" : "Listen"}</button>}
        {state !== "idle" && <button className="btn h-9 px-3 text-[13px]" onClick={stop} aria-label="Stop reading">
          <svg width="12" height="12" viewBox="0 0 24 24" aria-hidden><rect x="5" y="5" width="14" height="14" rx="2.5" fill="currentColor" /></svg>Stop</button>}
      </div>
      <div className="mt-2.5 flex flex-wrap items-center gap-2">
        <div className="min-w-[150px] flex-1">
          <label htmlFor="tts-voice" className="sr-only">Voice</label>
          <Select id="tts-voice" className="w-full" value={voice || voices[0]?.name || ""} onChange={setVoice}
            placeholder="Default voice" options={voices.map((v) => ({ value: v.name, label: v.name.replace(/^Microsoft\s+|\s*-\s*English.*$/g, ""), hint: v.lang }))} />
        </div>
          <div className="seg shrink-0 [&_button]:px-2 [&_button]:text-[12px]" role="radiogroup" aria-label="Reading speed">
            {RATES.map((r) => (
              <button key={r} role="radio" aria-checked={rate === r} aria-selected={rate === r} onClick={() => setRate(r)}
                className={rate === r ? "!text-ink !bg-surface rounded-[7px]" : ""}>{r}×</button>
            ))}
          </div>
      </div>
      {state !== "idle" && (
        <div className="mt-3 space-y-2" aria-live="polite">
          <p className="m-0 rounded-lg bg-surface px-3 py-2 text-[13px] leading-relaxed text-ink">
            <span className="sr-only">Now reading: </span>{sentences[idx] ?? "…"}
          </p>
          <div className="flex items-center gap-2 text-xs text-muted">
            <div className="h-1 flex-1 overflow-hidden rounded-full bg-line"><div className="h-full rounded-full bg-ink transition-[width] duration-300" style={{ width: `${pct}%` }} /></div>
            <span className="tabular">{Math.max(0, idx + 1)} / {sentences.length}</span>
          </div>
        </div>
      )}
    </div>
  );
}
