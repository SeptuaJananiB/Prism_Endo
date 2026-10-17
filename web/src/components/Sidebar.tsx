import { forwardRef, useImperativeHandle, useRef, useState } from "react";
import { api, type Job, type Meta, type RunSummary } from "../lib/api";
import { Select } from "./Select";
import { Grip, useResizable } from "../lib/resize";

export type SidebarHandle = { focusForm: () => void };

const REL: Record<string, { tone: string; label: string }> = {
  evaluated: { tone: "var(--ok)", label: "tested on unseen patients" },
  insufficient_test_data: { tone: "var(--warn)", label: "barely tested" },
  unevaluated: { tone: "var(--muted)", label: "not tested" },
  failed_on_heldout: { tone: "var(--err)", label: "failed its test" },
  untrained: { tone: "var(--muted)", label: "not trained" },
  unknown: { tone: "var(--muted)", label: "reliability unknown" },
};
const pretty = (s: string) => s.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

function Group({ title, children, aside }: { title: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return (
    <section className="space-y-3.5" aria-label={title}>
      <div className="flex items-center justify-between">
        <h2 className="label m-0">{title}</h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

function Field({ id, label, help, children }: { id: string; label: string; help?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-[13px] font-medium text-ink-2">{label}</label>
      {children}
      {help && <p id={`${id}-help`} className="m-0 text-xs leading-snug text-muted">{help}</p>}
    </div>
  );
}

export const Sidebar = forwardRef<SidebarHandle, {
  meta: Meta | null; runs: RunSummary[] | null; current: string | null; onOpen: (n: string) => void;
  onStart: (b: object) => void; onStop: () => void; onFinish: () => void; job: Job | null; onUploaded: (src: string) => void; open: boolean; onClose: () => void;
}>(function Sidebar({ meta, runs, current, onOpen, onStart, onStop, onFinish, job, onUploaded, open, onClose }, ref) {
  const [source, setSource] = useState("");
  const [asLive, setAsLive] = useState(false);   // replay the chosen video file through the live pipeline
  const [surgery, setSurgery] = useState("");
  const [kp, setKp] = useState("hybrid");
  const [maxFrames, setMaxFrames] = useState(0);
  const [targets, setTargets] = useState<string[] | null>(null);
  const [drag, setDrag] = useState(false);
  const [up, setUp] = useState<{ name: string; p: number } | null>(null);
  const [cams, setCams] = useState<number[]>([]);
  const [camBusy, setCamBusy] = useState(false);
  const [camMsg, setCamMsg] = useState("");
  const [upErr, setUpErr] = useState<string | null>(null);
  const firstField = useRef<HTMLDivElement>(null);
  const fileIn = useRef<HTMLInputElement>(null);
  useImperativeHandle(ref, () => ({ focusForm: () => firstField.current?.querySelector("button")?.focus() }));

  const busy = !!job && ["queued", "loading", "running"].includes(job.status);
  const src = source || meta?.inputs[0] || "";
  const surg = surgery || meta?.surgery_types[0] || "";
  const tissues = meta?.tissues?.[surg] ?? [];
  const tg = targets ?? tissues.map((x) => x.name); // all tissues by default
  const { width, dragging, handleProps } = useResizable("pe-left-width", 320, 272, 480, "right");
  const pct = job?.total ? Math.round((100 * job.done) / job.total) : 0;
  const isFile = !!src && !src.startsWith("camera:");
  const isLive = src.startsWith("camera:") || (isFile && asLive);

  const upload = async (f: File | undefined) => {
    if (!f) return;
    setUpErr(null);
    setUp({ name: f.name, p: 0 });
    try {
      const r = await api.upload(f, (p) => setUp({ name: f.name, p }));
      setSource(r.source);
      onUploaded(r.source);
    } catch (e) {
      setUpErr((e as Error).message);
    } finally {
      setUp(null);
    }
  };
  const toggle = (t: string) => setTargets(tg.includes(t) ? tg.filter((x) => x !== t) : [...tg, t]);

  return (
    <>
      <div className={`fixed inset-0 z-30 bg-black/30 backdrop-blur-[1px] transition-opacity lg:hidden ${open ? "opacity-100" : "pointer-events-none opacity-0"}`}
        onClick={onClose} aria-hidden />
      <aside className={`sidebar fixed inset-y-0 left-0 z-40 flex max-w-[88vw] shrink-0 flex-col border-r border-line lg:relative lg:translate-x-0 ${dragging ? "" : "transition-transform duration-300"} ${open ? "translate-x-0" : "-translate-x-full"}`}
        style={{ width }} aria-label="Controls">
        <div {...handleProps} aria-label="Resize sidebar" className="group absolute inset-y-0 -right-1.5 z-10 hidden w-3 cursor-col-resize lg:block">
          <Grip dragging={dragging} />
        </div>
        <div className="flex h-16 shrink-0 items-center gap-3 border-b border-line px-6">
          <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden>
            <rect width="32" height="32" rx="8" fill="var(--ink)" />
            <circle cx="16" cy="16" r="7" fill="none" stroke="var(--bg)" strokeWidth="2.5" />
            <circle cx="16" cy="16" r="2.5" fill="var(--bg)" />
          </svg>
          <div className="flex-1">
            <div className="text-[15px] font-semibold leading-tight tracking-tight">PRISM-Endo</div>
            <div className="text-xs text-muted">Surgical video analysis</div>
          </div>
          <button className="btn h-8 w-8 p-0 lg:hidden" onClick={onClose} aria-label="Close menu">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden><path d="M18 6 6 18M6 6l12 12" /></svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          <form className="space-y-7 px-6 py-6" aria-label="New analysis"
            onSubmit={(e) => {
              e.preventDefault();
              onStart({ source: isFile && asLive ? `video:${src}` : src, surgery: surg, keypoint_source: kp, max_frames: maxFrames, targets: tg, show_all_tissue: true });
            }}>
            <Group title="1 · Video">
              <div
                onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => { e.preventDefault(); setDrag(false); upload(e.dataTransfer.files[0]); }}
                className={`rounded-xl border border-dashed px-4 py-4 text-center transition-colors duration-150 ${drag ? "border-info bg-info/5" : "border-line-strong bg-surface"}`}>
                {up ? (
                  <div className="space-y-1.5 text-left" aria-live="polite">
                    <div className="truncate text-xs text-muted">Uploading {up.name}</div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-line"><div className="h-full rounded-full bg-info transition-[width]" style={{ width: `${Math.round(up.p * 100)}%` }} /></div>
                  </div>
                ) : (
                  <>
                    <svg className="mx-auto mb-1.5 text-muted" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M12 16V4m0 0-4 4m4-4 4 4M4 20h16" /></svg>
                    <p className="m-0 text-[13px] text-ink-2">Drop a video or <button type="button" className="font-medium text-info underline-offset-2 hover:underline" onClick={() => fileIn.current?.click()}>browse</button></p>
                    <p className="m-0 mt-0.5 text-xs text-muted">MP4, MOV, AVI, MKV, WEBM</p>
                  </>
                )}
                <input ref={fileIn} type="file" accept="video/*" className="sr-only" tabIndex={-1} aria-label="Upload video" onChange={(e) => upload(e.target.files?.[0])} />
                {upErr && <p role="alert" className="m-0 mt-2 text-xs text-err">{upErr}</p>}
              </div>
              <Field id="src" label="Video to analyse">
                <div ref={firstField}>
                  <Select id="src" value={src} onChange={setSource} disabled={busy} placeholder="No videos found"
                    options={[
                      ...cams.map((c) => ({ value: `camera:${c}`, label: `Live camera ${c}`, hint: "real time" })),
                      ...(meta?.inputs ?? []).map((p) => ({ value: p, label: p.split(/[\\/]/).pop() ?? p, hint: p.includes("uploads") ? "uploaded" : undefined }))]} />
                </div>
              </Field>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                <button type="button" className="font-medium text-info hover:underline disabled:opacity-50" disabled={busy || camBusy}
                  onClick={async () => {
                    setCamBusy(true); setCamMsg("");
                    try { const r = await api.cameras(); setCams(r.cameras); setCamMsg(r.cameras.length ? `${r.cameras.length} camera(s) found` : "No camera found");
                      if (r.cameras.length) setSource(`camera:${r.cameras[0]}`); }
                    catch (e) { setCamMsg((e as Error).message); } finally { setCamBusy(false); }
                  }}>{camBusy ? "Searching…" : "Find live cameras"}</button>
                {camMsg && <span className="text-muted" aria-live="polite">{camMsg}</span>}
              </div>
              {isFile && (
                <label className="flex items-start gap-2 text-xs leading-snug text-ink-2">
                  <input type="checkbox" className="mt-0.5" checked={asLive} disabled={busy} onChange={(e) => setAsLive(e.target.checked)} />
                  <span><b>Play as live camera</b> · replays this video in real time through live mode, without a webcam in between
                    (filming a screen blurs and discolours the image and weakens detection)</span>
                </label>
              )}
              {isLive && <p className="m-0 text-xs leading-snug text-muted">Live mode analyses the camera in real time (faster settings). Use <b>Finish &amp; save</b> to end the session.</p>}
            </Group>

            <Group title="2 · Model">
              <Field id="surg" label="Surgery type" help={'Chooses the tissue model. "Other" reduces false liver on appendix and bowel.'}>
                <Select id="surg" value={surg} onChange={(v) => { setSurgery(v); setTargets(null); }} disabled={busy} describedBy="surg-help"
                  options={(meta?.surgery_types ?? []).map((t) => ({ value: t, label: t }))} />
              </Field>
              <div className="grid grid-cols-[1fr_96px] gap-3">
                <Field id="kp" label="Tip source">
                  <Select id="kp" value={kp} onChange={setKp} disabled={busy} options={[
                    { value: "hybrid", label: "Hybrid", hint: "recommended" },
                    { value: "dlc", label: "DLC only" },
                    { value: "mask_tip", label: "Geometry only" }]} />
                </Field>
                <Field id="mf" label="Max frames">
                  <input id="mf" className="field tabular" type="number" min={0} step={50} value={maxFrames}
                    onChange={(e) => setMaxFrames(Math.max(0, Number(e.target.value) || 0))} disabled={busy} aria-describedby="mf-help" />
                </Field>
              </div>
              <p id="mf-help" className="-mt-1.5 m-0 text-xs text-muted">Max frames 0 analyses the whole video.</p>
            </Group>

            <Group title="3 · Tissue annotation" aside={
              <button type="button" className="text-xs font-medium text-info hover:underline disabled:opacity-50" disabled={busy}
                onClick={() => setTargets(tg.length === tissues.length ? [] : tissues.map((x) => x.name))}>
                {tg.length === tissues.length ? "Clear all" : `Select all (${tg.length}/${tissues.length})`}
              </button>}>
              <fieldset className="m-0 space-y-2.5 border-0 p-0" disabled={busy}>
                <legend className="mb-2 p-0 text-[13px] font-medium text-ink-2">Measure distance to</legend>
                <div className="flex flex-wrap gap-1.5">
                  {tissues.map((t) => {
                    const on = tg.includes(t.name);
                    const r = REL[t.reliability] ?? REL.unknown;
                    return (
                      <button key={t.name} type="button" onClick={() => toggle(t.name)} aria-pressed={on}
                        title={`${pretty(t.name)}: ${r.label}${t.dice ? ` (Dice ${t.dice.toFixed(2)})` : ""}`}
                        className={`inline-flex h-8 items-center gap-1.5 rounded-full border px-3 text-[12px] font-medium transition-all duration-150 active:scale-[.97] disabled:opacity-50 ${
                          on ? "border-ink bg-ink text-bg" : "border-line bg-surface text-ink-2 hover:border-line-strong"}`}>
                        <span className="h-1.5 w-1.5 rounded-full" style={{ background: on ? "var(--bg)" : r.tone }} aria-hidden />
                        {pretty(t.name)}
                        <span className="sr-only">, {r.label}</span>
                      </button>
                    );
                  })}
                  {tissues.length === 0 && <span className="text-xs text-muted">No tissue model for this surgery type.</span>}
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-1 pt-1.5 pb-1 text-[11px] text-muted" aria-hidden>
                  <span className="flex items-center gap-1"><i className="h-1.5 w-1.5 rounded-full bg-ok" />tested</span>
                  <span className="flex items-center gap-1"><i className="h-1.5 w-1.5 rounded-full bg-warn" />barely tested</span>
                  <span className="flex items-center gap-1"><i className="h-1.5 w-1.5 rounded-full bg-muted" />not tested</span>
                </div>

              </fieldset>

            </Group>

            <div className="space-y-3">
              <button type="submit" className="btn btn-primary h-11 w-full text-[14px]" disabled={busy || !src}>{busy ? "Analysing…" : "Analyse video"}</button>
              <div aria-live="polite">
                {job && busy && (
                  <div className="rise space-y-2.5 rounded-xl border border-line bg-surface p-3.5">
                    <div className="flex items-center justify-between gap-2 text-xs text-muted">
                      <span className="flex min-w-0 items-center gap-2"><span className="live-dot shrink-0" aria-hidden />
                        <span className="truncate">{job.cancel ? "Stopping…" : job.phase && job.phase !== "Processing frames" ? job.phase : job.status === "running" ? `Processing ${(job.source ?? "").split(/[\\/]/).pop()}` : "Loading models"}</span></span>
                      <span className="shrink-0 tabular">{job.total ? `${job.done} / ${job.total}` : ""}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-line" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct} aria-label="Analysis progress">
                        <div className={`h-full rounded-full transition-[width] duration-500 ${job.cancel ? "bg-muted" : "bg-ink"}`} style={{ width: `${pct}%` }} />
                      </div>
                      <span className="w-9 text-right text-xs text-muted tabular">{pct}%</span>
                    </div>
                    {job.live && (
                      <button type="button" onClick={onFinish} disabled={job.cancel}
                        className="btn btn-primary h-9 w-full text-[13px]" aria-label="Finish and save live analysis">Finish &amp; save</button>
                    )}
                    <button type="button" onClick={onStop} disabled={job.cancel}
                      className="btn h-8 w-full text-[13px] text-err hover:border-[var(--err-line)] hover:bg-[var(--err-bg)]" aria-label="Stop analysis">
                      {job.cancel ? <>Stopping…</> : <><svg width="12" height="12" viewBox="0 0 24 24" aria-hidden><rect x="5" y="5" width="14" height="14" rx="2.5" fill="currentColor" /></svg>{job.live ? "Discard" : "Stop analysis"}</>}
                    </button>
                    <p className="m-0 text-xs leading-snug text-muted">{job.live ? "Finish & save keeps the recording; Discard throws it away." : "Stopping discards this run; earlier results stay unchanged."}</p>
                  </div>
                )}
              </div>
            </div>
          </form>

          <div className="border-t border-line px-6 py-6">
            <Group title="Saved analyses" aside={<span className="text-xs text-muted tabular">{runs?.length ?? ""}</span>}>
              {!runs ? <div className="skeleton h-[38px]" /> : (
                <Field id="run-pick" label="Open analysis" help="Newest first">
                  <Select id="run-pick" value={current ?? ""} onChange={(v) => { onOpen(v); onClose(); }} disabled={runs.length === 0}
                    describedBy="run-pick-help" placeholder="No analyses yet"
                    options={runs.map((r) => ({ value: r.name, label: r.name,
                      hint: `${r.frames?.toLocaleString()} fr${r.surgery ? ` · ${r.surgery}` : ""}${(r.bleeding_frames ?? 0) > 0 ? " · bleeding" : ""}` }))} />
                </Field>
              )}
            </Group>
          </div>
        </div>
      </aside>
    </>
  );
});
