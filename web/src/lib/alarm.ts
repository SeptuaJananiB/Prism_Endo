/**
 * Operating-theatre monitor sounds cut from the reference EKG monitor recording ("EKG" by freesound_community,
 * Pixabay Content License), one loop per situation:
 *   normal  public/sounds/ekg-normal.mp3  section 1, heartbeat beep (2 kHz, every 1.2 s)   -> risk Low / Medium
 *   high    public/sounds/ekg-high.mp3    section 2, alarm (500 Hz triples + 3.5 kHz)    -> risk High
 *   touch   public/sounds/ekg-flatline.mp3   section 3, continuous flat-line tone (1 kHz)  -> tip touching tissue (distance 0)
 */
export type Sound = "normal" | "high" | "touch";

const FILES: Record<Sound, string> = {
  normal: "/sounds/ekg-normal.mp3",
  high: "/sounds/ekg-high.mp3",
  touch: "/sounds/ekg-flatline.mp3",
};

let ctx: AudioContext | null = null;
const buffers: Partial<Record<Sound, AudioBuffer>> = {};
let loading = false;
let current: { sound: Sound; node: AudioBufferSourceNode } | null = null;
let wanted: Sound | null = null;

/** Create / resume the audio context and load the three loops. Call from a click: browsers need a user gesture. */
export function unlockAudio(): AudioContext | null {
  try {
    ctx ??= new AudioContext();
    if (ctx.state === "suspended") void ctx.resume();
    if (!loading && Object.keys(buffers).length < 3) {
      loading = true;
      const ac = ctx;
      Promise.all((Object.keys(FILES) as Sound[]).map((k) =>
        fetch(FILES[k]).then((r) => r.arrayBuffer()).then((b) => ac.decodeAudioData(b)).then((buf) => { buffers[k] = buf; })))
        .then(() => { if (wanted) play(wanted); })
        .catch(() => { /* stay silent if the files cannot be loaded */ })
        .finally(() => { loading = false; });
    }
    return ctx;
  } catch {
    return null;
  }
}

/** Loop this sound (switching from any other); null stops all sound. */
export function play(sound: Sound | null): void {
  wanted = sound;
  if (current && current.sound === sound) return;
  if (current) {
    try { current.node.stop(); } catch { /* already stopped */ }
    current.node.disconnect();
    current = null;
  }
  if (!sound) return;
  const ac = unlockAudio();
  const buf = buffers[sound];
  if (!ac || !buf) return;                 // starts as soon as the files have loaded
  const node = ac.createBufferSource();
  node.buffer = buf;
  node.loop = true;
  node.connect(ac.destination);
  node.start();
  current = { sound, node };
}
