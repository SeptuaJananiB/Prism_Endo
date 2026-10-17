/**
 * Operating-theatre monitor sound: the reference EKG monitor recording (public/sounds/ekg.mp3, "EKG" by
 * freesound_community, Pixabay Content License) played exactly as recorded, looped while any risk or warning is
 * active and stopped when the risk is Minimal / no instrument is in view.
 */
export type Priority = "high" | "medium" | "low" | "lowest";

let ctx: AudioContext | null = null;
let buffer: AudioBuffer | null = null;
let loading: Promise<void> | null = null;
let source: AudioBufferSourceNode | null = null;
let wanted = false;

/** Create / resume the audio context and load the recording. Call from a click: browsers need a user gesture. */
export function unlockAudio(): AudioContext | null {
  try {
    ctx ??= new AudioContext();
    if (ctx.state === "suspended") void ctx.resume();
    if (!buffer && !loading) {
      const ac = ctx;
      loading = fetch("/sounds/ekg.mp3")
        .then((r) => r.arrayBuffer())
        .then((b) => ac.decodeAudioData(b))
        .then((buf) => { buffer = buf; if (wanted) startMonitor(); })
        .catch(() => { loading = null; });
    }
    return ctx;
  } catch {
    return null;
  }
}

/** Start looping the recording (no-op if it is already playing). */
export function startMonitor(): void {
  wanted = true;
  const ac = unlockAudio();
  if (!ac || !buffer || source) return;
  const s = ac.createBufferSource();
  s.buffer = buffer;
  s.loop = true;
  s.connect(ac.destination);
  s.start();
  source = s;
}

/** Stop the recording. */
export function stopMonitor(): void {
  wanted = false;
  if (source) {
    try { source.stop(); } catch { /* already stopped */ }
    source.disconnect();
    source = null;
  }
}
