import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";

export type Option = { value: string; label: string; hint?: string };

/** Themed, accessible single-select (WAI-ARIA combobox + listbox). Replaces the native <select>, whose popup
 *  ignores the dark theme on Windows. Keyboard: Enter/Space/ArrowDown open, arrows move, Home/End, type to jump,
 *  Enter selects, Esc closes. */
export function Select({ id, value, options, onChange, disabled, placeholder = "Select…", describedBy, className = "" }: {
  id?: string; value: string; options: Option[]; onChange: (v: string) => void; disabled?: boolean;
  placeholder?: string; describedBy?: string; className?: string;
}) {
  const auto = useId().replace(/:/g, "");
  const btnId = id ?? `sel-${auto}`;
  const listId = `${btnId}-list`;
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [up, setUp] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const typed = useRef({ s: "", t: 0 });
  const idx = Math.max(0, options.findIndex((o) => o.value === value));
  const cur = options.find((o) => o.value === value);

  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => { if (!root.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [open]);
  useLayoutEffect(() => {
    if (!open) return;
    const r = root.current!.getBoundingClientRect();
    setUp(window.innerHeight - r.bottom < 260 && r.top > window.innerHeight - r.bottom);
    list.current?.querySelector<HTMLElement>(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [open, active]);

  const choose = (i: number) => { const o = options[i]; if (o) onChange(o.value); setOpen(false); root.current?.querySelector("button")?.focus(); };
  const openAt = (i: number) => { setActive(i); setOpen(true); };

  const onKey = (e: React.KeyboardEvent) => {
    if (disabled) return;
    const n = options.length;
    if (!open) {
      if (["Enter", " ", "ArrowDown", "ArrowUp"].includes(e.key)) { e.preventDefault(); openAt(idx); }
      return;
    }
    if (e.key === "Escape") { e.preventDefault(); setOpen(false); }
    else if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(n - 1, a + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
    else if (e.key === "Home") { e.preventDefault(); setActive(0); }
    else if (e.key === "End") { e.preventDefault(); setActive(n - 1); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(active); }
    else if (e.key === "Tab") setOpen(false);
    else if (e.key.length === 1) {
      const now = Date.now();
      typed.current = { s: (now - typed.current.t < 600 ? typed.current.s : "") + e.key.toLowerCase(), t: now };
      const k = options.findIndex((o) => o.label.toLowerCase().startsWith(typed.current.s));
      if (k >= 0) setActive(k);
    }
  };

  return (
    <div ref={root} className={`relative ${className}`} onKeyDown={onKey}>
      <button id={btnId} type="button" role="combobox" aria-haspopup="listbox" aria-expanded={open} aria-controls={listId}
        aria-activedescendant={open ? `${listId}-${active}` : undefined} aria-describedby={describedBy} disabled={disabled}
        onClick={() => (open ? setOpen(false) : openAt(idx))}
        className={`field flex items-center justify-between gap-2 text-left disabled:cursor-not-allowed disabled:opacity-50 ${open ? "border-[var(--focus)]" : ""}`}>
        <span className={`truncate ${cur ? "" : "text-muted"}`}>{cur?.label ?? placeholder}</span>
        <svg className={`shrink-0 text-muted transition-transform duration-200 ${open ? "rotate-180" : ""}`} width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" aria-hidden><path d="m6 9 6 6 6-6" /></svg>
      </button>
      {open && (
        <ul ref={list} id={listId} role="listbox" aria-labelledby={btnId} tabIndex={-1}
          className={`card absolute left-0 right-0 z-50 m-0 max-h-[260px] min-w-full list-none overflow-y-auto p-1 shadow-[var(--shadow-hover)] ${up ? "bottom-[calc(100%+6px)] origin-bottom" : "top-[calc(100%+6px)] origin-top"} animate-[pop_.16s_var(--ease)]`}>
          {options.length === 0 && <li className="px-3 py-2 text-[13px] text-muted">No options</li>}
          {options.map((o, i) => {
            const sel = o.value === value;
            return (
              <li key={o.value} id={`${listId}-${i}`} data-i={i} role="option" aria-selected={sel}
                onMouseEnter={() => setActive(i)} onMouseDown={(e) => e.preventDefault()} onClick={() => choose(i)}
                className={`flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-2 text-[13px] transition-colors ${i === active ? "bg-surface-2" : ""}`}>
                <svg className={`shrink-0 ${sel ? "opacity-100" : "opacity-0"}`} width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.8" strokeLinecap="round" aria-hidden><path d="m5 12 5 5 9-10" /></svg>
                <span className="min-w-0 flex-1 truncate">{o.label}</span>
                {o.hint && <span className="shrink-0 text-xs text-muted">{o.hint}</span>}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
