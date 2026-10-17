/**
 * Audible alarms modelled on IEC 60601-1-8 (medical alarm systems), generated with the Web Audio API.
 *  - high priority:   10 fast pulses per burst (3 + 2, pause, 3 + 2)      -> contact warning, tip in No-Go zone
 *  - medium priority: 3 slower pulses per burst                           -> risk High
 * Tone: 440 Hz fundamental plus 3 harmonics (the standard asks for >= 4 spectral peaks within 15 dB, 150-1000 Hz
 * fundamental), with short ramps so pulses do not click.
 */
export type Priority = "high" | "medium";

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

function pulse(ac: AudioContext, start: number, dur: number, f0: number, gain: number) {
  const out = ac.createGain();
  out.gain.setValueAtTime(0, start);
  out.gain.linearRampToValueAtTime(gain, start + 0.02);          // rise
  out.gain.setValueAtTime(gain, start + dur - 0.03);
  out.gain.linearRampToValueAtTime(0, start + dur);               // fall
  out.connect(ac.destination);
  [1, 2, 3, 4].forEach((h, i) => {                               // fundamental + 3 harmonics
    const o = ac.createOscillator();
    const g = ac.createGain();
    o.type = "sine";
    o.frequency.value = f0 * h;
    g.gain.value = [1, 0.6, 0.45, 0.35][i];
    o.connect(g).connect(out);
    o.start(start);
    o.stop(start + dur + 0.01);
  });
}

/** Play one alarm burst. Returns its length in seconds. */
export function playBurst(p: Priority, volume = 0.25): number {
  const ac = unlockAudio();
  if (!ac || ac.state !== "running") return 0;
  const t0 = ac.currentTime + 0.02;
  if (p === "high") {
    // 3 + 2 pulses, pause, 3 + 2 pulses (150 ms pulses, 100 ms apart)
    const at = [0, 0.25, 0.5, 1.0, 1.25, 2.2, 2.45, 2.7, 3.2, 3.45];
    at.forEach((s) => pulse(ac, t0 + s, 0.15, 523, volume));
    return 3.6;
  }
  // medium: 3 pulses, 200 ms long, 300 ms apart
  [0, 0.3, 0.6].forEach((s) => pulse(ac, t0 + s, 0.2, 440, volume * 0.8));
  return 0.8;
}
