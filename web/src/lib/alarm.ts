/**
 * Operating-theatre monitor beep, using a real EKG monitor recording (public/sounds/ekg-beep.mp3: one 196 ms,
 * 2 kHz beep cut from "EKG" by freesound_community, Pixabay Content License). The beep gets FASTER as risk rises:
 *   lowest (risk Low)      1 beep every 2 s
 *   low    (risk Medium)   1 beep every 1.2 s (the recording's own rhythm)
 *   medium (risk High)     2 beeps per second
 *   high   (contact warning / tip in No-Go zone)  rapid triple beeps, continuous
 * If the file cannot be loaded, a synthesised tone of the same pitch is used instead.
 */
export type Priority = "high" | "medium" | "low" | "lowest";

export const BEEP: Record<Priority, { every: number; count: number; gap: number }> = {
  lowest: { every: 2000, count: 1, gap: 0 },
  low: { every: 1200, count: 1, gap: 0 },
  medium: { every: 500, count: 1, gap: 0 },
  high: { every: 900, count: 3, gap: 0.22 },
};

let ctx: AudioContext | null = null;
let sample: AudioBuffer | null = null;
let loading = false;

/** Create / resume the audio context. Call from a click handler: browsers only allow audio after a user gesture. */
export function unlockAudio(): AudioContext | null {
  try {
    ctx ??= new AudioContext();
    if (ctx.state === "suspended") void ctx.resume();
    if (!sample && !loading) {
      loading = true;
      const ac = ctx;
      fetch("/sounds/ekg-beep.mp3")
        .then((r) => r.arrayBuffer())
        .then((b) => ac.decodeAudioData(b))
        .then((buf) => { sample = buf; })
        .catch(() => { /* keep the synthesised fallback */ })
        .finally(() => { loading = false; });
    }
    return ctx;
  } catch {
    return null;
  }
}

function beep(ac: AudioContext, start: number, volume: number) {
  const g = ac.createGain();
  g.gain.value = volume;
  g.connect(ac.destination);
  if (sample) {
    const s = ac.createBufferSource();
    s.buffer = sample;
    s.connect(g);
    s.start(start);
    return;
  }
  const o = ac.createOscillator();                 // fallback: same pitch and length as the recording
  o.type = "sine";
  o.frequency.value = 2050;
  g.gain.setValueAtTime(0, start);
  g.gain.linearRampToValueAtTime(volume * 0.5, start + 0.01);
  g.gain.setValueAtTime(volume * 0.5, start + 0.18);
  g.gain.linearRampToValueAtTime(0, start + 0.2);
  o.connect(g);
  o.start(start);
  o.stop(start + 0.21);
}

/** Play the beep(s) for one tick of this level. */
export function playBurst(p: Priority, volume = 0.8): void {
  const ac = unlockAudio();
  if (!ac || ac.state !== "running") return;
  const b = BEEP[p];
  const t0 = ac.currentTime + 0.01;
  for (let i = 0; i < b.count; i++) beep(ac, t0 + i * b.gap, volume);
}
