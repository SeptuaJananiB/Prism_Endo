/**
 * Operating-theatre monitor beeps (like an ECG / pulse-oximeter monitor), generated with the Web Audio API.
 * One short, pure tone; the beep gets FASTER and slightly HIGHER as the situation gets worse:
 *   lowest (risk Low)      1 beep every 2 s
 *   low    (risk Medium)   1 beep every 1 s
 *   medium (risk High)     2 beeps per second
 *   high   (contact warning / tip in No-Go zone)  rapid triple beeps, continuous
 */
export type Priority = "high" | "medium" | "low" | "lowest";

export const BEEP: Record<Priority, { every: number; freq: number; count: number; gap: number }> = {
  lowest: { every: 2000, freq: 880, count: 1, gap: 0 },
  low: { every: 1000, freq: 940, count: 1, gap: 0 },
  medium: { every: 500, freq: 1000, count: 1, gap: 0 },
  high: { every: 700, freq: 1100, count: 3, gap: 0.12 },
};

let ctx: AudioContext | null = null;

/** Create / resume the audio context. Call from a click handler: browsers only allow audio after a user gesture. */
export function unlockAudio(): AudioContext | null {
  try {
    ctx ??= new AudioContext();
    if (ctx.state === "suspended") void ctx.resume();
    return ctx;
  } catch {
    return null;
  }
}

/** One monitor beep: pure sine, ~90 ms, quick attack and release so it sounds clean, not clicky. */
function beep(ac: AudioContext, start: number, freq: number, volume: number) {
  const o = ac.createOscillator();
  const g = ac.createGain();
  o.type = "sine";
  o.frequency.value = freq;
  g.gain.setValueAtTime(0, start);
  g.gain.linearRampToValueAtTime(volume, start + 0.005);
  g.gain.setValueAtTime(volume, start + 0.07);
  g.gain.exponentialRampToValueAtTime(0.0001, start + 0.09);
  o.connect(g).connect(ac.destination);
  o.start(start);
  o.stop(start + 0.1);
}

/** Play the beep(s) for one tick of this level. */
export function playBurst(p: Priority, volume = 0.3): void {
  const ac = unlockAudio();
  if (!ac || ac.state !== "running") return;
  const b = BEEP[p];
  const t0 = ac.currentTime + 0.01;
  for (let i = 0; i < b.count; i++) beep(ac, t0 + i * b.gap, b.freq, volume);
}
