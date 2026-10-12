import { useEffect, useRef, useState } from "react";

const reduced = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/** Animates a number from its previous value to `target` (ease-out), respecting reduced motion. */
export function useCountUp(target: number, ms = 900) {
  const [v, setV] = useState(0);
  const from = useRef(0);
  useEffect(() => {
    if (reduced()) {
      setV(target);
      from.current = target;
      return;
    }
    const start = performance.now();
    const a = from.current;
    let raf = 0;
    const tick = (t: number) => {
      const k = Math.min(1, (t - start) / ms);
      const e = 1 - Math.pow(1 - k, 3);
      setV(a + (target - a) * e);
      if (k < 1) raf = requestAnimationFrame(tick);
      else from.current = target;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

export function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    try {
      return (localStorage.getItem("pe-theme") as "light" | "dark") || "light";
    } catch {
      return "light";
    }
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("pe-theme", theme);
    } catch {
      /* storage blocked */
    }
  }, [theme]);
  return [theme, setTheme] as const;
}
