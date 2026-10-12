import { useCallback, useEffect, useRef, useState } from "react";

/** Width state for a panel resized by dragging one edge; remembered per browser. */
export function useResizable(key: string, def: number, min: number, max: number, edge: "left" | "right") {
  const clamp = useCallback((w: number) => Math.max(min, Math.min(max, Math.round(w))), [min, max]);
  const [width, setWidth] = useState(() => { try { return clamp(Number(localStorage.getItem(key)) || def); } catch { return def; } });
  const [dragging, setDragging] = useState(false);
  const start = useRef<{ x: number; w: number } | null>(null);
  useEffect(() => { try { localStorage.setItem(key, String(width)); } catch { /* storage blocked */ } }, [key, width]);

  const onMove = useCallback((e: PointerEvent) => {
    if (!start.current) return;
    const dx = e.clientX - start.current.x;
    setWidth(clamp(start.current.w + (edge === "right" ? dx : -dx)));
  }, [clamp, edge]);
  const onUp = useCallback(() => {
    start.current = null; setDragging(false);
    window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp);
    document.body.style.cursor = ""; document.body.style.userSelect = "";
  }, [onMove]);
  useEffect(() => () => onUp(), [onUp]);

  const handleProps = {
    role: "separator" as const, "aria-orientation": "vertical" as const, tabIndex: 0,
    "aria-valuemin": min, "aria-valuemax": max, "aria-valuenow": width,
    title: "Drag to resize · double-click to reset",
    onPointerDown: (e: React.PointerEvent) => {
      start.current = { x: e.clientX, w: width }; setDragging(true);
      window.addEventListener("pointermove", onMove); window.addEventListener("pointerup", onUp);
      document.body.style.cursor = "col-resize"; document.body.style.userSelect = "none";
    },
    onDoubleClick: () => setWidth(def),
    onKeyDown: (e: React.KeyboardEvent) => {
      const grow = edge === "right" ? "ArrowRight" : "ArrowLeft", shrink = edge === "right" ? "ArrowLeft" : "ArrowRight";
      if (e.key === grow) { e.preventDefault(); setWidth((w) => clamp(w + 24)); }
      if (e.key === shrink) { e.preventDefault(); setWidth((w) => clamp(w - 24)); }
      if (e.key === "Home") setWidth(max);
      if (e.key === "End") setWidth(min);
    },
  };
  return { width, dragging, handleProps };
}

/** The visible grip for a resize handle (thin line + pill). Place inside a positioned edge element. */
export function Grip({ dragging }: { dragging: boolean }) {
  return (
    <>
      <span className={`absolute inset-y-0 left-1/2 w-px -translate-x-1/2 transition-colors ${dragging ? "bg-info" : "bg-transparent group-hover:bg-line-strong group-focus-visible:bg-info"}`} />
      <span className={`absolute left-1/2 top-1/2 h-10 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full transition-colors ${dragging ? "bg-info" : "bg-line-strong group-hover:bg-muted"}`} />
    </>
  );
}
