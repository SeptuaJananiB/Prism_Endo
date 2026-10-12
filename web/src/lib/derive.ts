import type { Series } from "./api";

export type Track = {
  id: string;
  label: string;
  /** per-frame value: category string for risk, boolean otherwise */
  values: (string | boolean)[];
  /** per-frame "uncertain" flag (rendered hatched) */
  unsure?: boolean[];
  color: string;
  hint: string;
};
export type Episode = { start: number; end: number; i: number; kind: string; tone: "err" | "warn" | "ok" | "info" };
export type Milestone = { i: number; t: number; label: string; tone: "err" | "warn" | "ok" | "info" };

const truthy = (v: unknown) => v === true || v === "True" || v === "true" || v === 1;
const numv = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function timeIndex(t: number[], sec: number) {
  let lo = 0, hi = t.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (t[mid] < sec) lo = mid + 1; else hi = mid;
  }
  return lo > 0 && Math.abs(t[lo - 1] - sec) < Math.abs(t[lo] - sec) ? lo - 1 : lo;
}

export function tracks(s: Series, n: number, nearPx = 60, target = "target"): Track[] {
  const out: Track[] = [];
  const cat = (s.risk_category ?? []).map((v) => String(v ?? "Unavailable"));
  const kps = Object.keys(s).filter((c) => c.endsWith("_kp_status")).map((c) => c.replace("_kp_status", ""));
  const unsureAt = (i: number) => kps.some((k) => s[`${k}_kp_status`]?.[i] === "low_confidence");
  if (cat.length) out.push({ id: "risk", label: "Risk", values: cat, color: "", hint: "Rule-based estimate",
    unsure: Array.from({ length: n }, (_, i) => unsureAt(i)) });
  if (s.bleeding_detected) out.push({ id: "bleeding", label: "Bleeding", values: s.bleeding_detected.map(truthy),
    color: "var(--risk-high)", hint: "SurgBlood detector, F1 0.937" });
  if (s.cvs_achieved) out.push({ id: "cvs", label: "CVS met", values: s.cvs_achieved.map(truthy),
    color: "var(--risk-minimal)", hint: "All 3 criteria, mAP 0.539" });
  const vis = (frac: string, kp: string) =>
    s[frac] ? s[frac].map((v) => (numv(v) ?? 0) > 0.002) : (s[`${kp}_kp_status`] ?? []).map((v) => v === "ok" || v === "low_confidence");
  if (s.inst_grasper_frac || s.grasper_tip_kp_status) out.push({ id: "grasper", label: "Grasper", values: vis("inst_grasper_frac", "grasper_tip"),
    color: "var(--info)", hint: "Instrument mask, Dice 0.85" });
  if (s.inst_l_hook_frac || s.lhook_tip_kp_status) out.push({ id: "lhook", label: "L-hook", values: vis("inst_l_hook_frac", "lhook_tip"),
    color: "#7a5af8", hint: "Instrument mask, Dice 0.84" });
  const dist = Object.keys(s).filter((c) => c.endsWith("_prox_distance_px"));
  if (dist.length) {
    out.push({ id: "near", label: `Near ${target}`, color: "var(--risk-medium)", hint: `Tip within ${nearPx} px (image space)`,
      values: Array.from({ length: n }, (_, i) => dist.some((c) => { const d = numv(s[c][i]); return d != null && d <= nearPx; })) });
  }
  return out;
}

/** Runs where flag holds, bridging gaps shorter than gapS (signals flicker frame to frame). */
function runs(t: number[], flag: (i: number) => boolean, minS: number, gapS = 0.4) {
  const out: { start: number; end: number; i: number }[] = [];
  let st = -1, last = -1;
  for (let i = 0; i <= t.length; i++) {
    const f = i < t.length && flag(i);
    if (f) {
      if (st < 0) st = i;
      last = i;
    } else if (st >= 0 && (i === t.length || t[i] - t[last] > gapS)) {
      if (t[last] - t[st] >= minS) out.push({ start: t[st], end: t[last], i: st });
      st = -1;
    }
  }
  return out;
}

export function episodes(t: number[], s: Series, minS = 0.5): Episode[] {
  const cat = (s.risk_category ?? []).map((v) => String(v ?? ""));
  const ev: Episode[] = [
    ...runs(t, (i) => cat[i] === "High", 0.1).map((r) => ({ ...r, kind: "High risk", tone: "err" as const })),
    ...runs(t, (i) => cat[i] === "Medium" || cat[i] === "High", 2).map((r) => ({ ...r, kind: "Sustained medium+ risk", tone: "warn" as const })),
  ];
  if (s.bleeding_detected) ev.push(...runs(t, (i) => truthy(s.bleeding_detected[i]), minS).map((r) => ({ ...r, kind: "Bleeding", tone: "err" as const })));
  if (s.cvs_achieved) ev.push(...runs(t, (i) => truthy(s.cvs_achieved[i]), minS).map((r) => ({ ...r, kind: "CVS met", tone: "ok" as const })));
  return ev.sort((a, b) => a.start - b.start);
}

export function milestones(t: number[], s: Series): Milestone[] {
  const out: Milestone[] = [];
  const first = (pred: (i: number) => boolean) => { for (let i = 0; i < t.length; i++) if (pred(i)) return i; return -1; };
  const b = s.bleeding_detected ? first((i) => truthy(s.bleeding_detected[i])) : -1;
  if (b >= 0) out.push({ i: b, t: t[b], label: "First bleeding", tone: "err" });
  const c = s.cvs_achieved ? first((i) => truthy(s.cvs_achieved[i])) : -1;
  if (c >= 0) out.push({ i: c, t: t[c], label: "CVS first met", tone: "ok" });
  const h = first((i) => s.risk_category?.[i] === "High");
  if (h >= 0) out.push({ i: h, t: t[h], label: "First high risk", tone: "err" });
  const dist = Object.keys(s).filter((k) => k.endsWith("_prox_distance_px"));
  let best = -1, bd = Infinity;
  for (let i = 0; i < t.length; i++) for (const k of dist) { const d = numv(s[k][i]); if (d != null && d < bd) { bd = d; best = i; } }
  if (best >= 0) out.push({ i: best, t: t[best], label: `Closest approach (${bd.toFixed(0)} px)`, tone: "warn" });
  let peak = -1, pv = -Infinity;
  (s.risk_score ?? []).forEach((v, i) => { const x = numv(v); if (x != null && x > pv) { pv = x; peak = i; } });
  if (peak >= 0) out.push({ i: peak, t: t[peak], label: `Peak risk (${pv.toFixed(2)})`, tone: "warn" });
  return out.sort((a, b) => a.t - b.t);
}

export function summaryBullets(t: number[], s: Series, ms: Milestone[]) {
  const n = t.length || 1;
  const cat = (s.risk_category ?? []).map((v) => String(v ?? "Unavailable"));
  const pct = (k: string) => Math.round((100 * cat.filter((c) => c === k).length) / n);
  const bl = (s.bleeding_detected ?? []).filter(truthy).length;
  const cvs = s.cvs_achieved ? Math.round((100 * s.cvs_achieved.filter(truthy).length) / n) : null;
  const out: { text: string; i?: number }[] = [
    { text: `${(t[t.length - 1] ?? 0).toFixed(0)} s analysed: ${pct("Medium")} % of frames medium risk, ${pct("High")} % high, ${pct("Low") + pct("Minimal")} % low or minimal.` },
  ];
  const peak = ms.find((m) => m.label.startsWith("Peak risk"));
  if (peak) out.push({ text: `${peak.label} at ${peak.t.toFixed(1)} s.`, i: peak.i });
  const close = ms.find((m) => m.label.startsWith("Closest"));
  if (close) out.push({ text: `${close.label} at ${close.t.toFixed(1)} s.`, i: close.i });
  out.push(bl ? { text: `Bleeding flagged in ${bl} frames; first at ${ms.find((m) => m.label === "First bleeding")?.t.toFixed(1)} s.`, i: ms.find((m) => m.label === "First bleeding")?.i }
    : { text: "No bleeding detected." });
  if (cvs != null) out.push(cvs ? { text: `Critical View of Safety estimated as met in ${cvs} % of frames.`, i: ms.find((m) => m.label === "CVS first met")?.i }
    : { text: "Critical View of Safety not reached (estimate)." });
  return out;
}
