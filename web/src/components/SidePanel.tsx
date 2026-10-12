import { Grip, useResizable } from "../lib/resize";

export function SidePanel({ open, onClose, title, badge, children }: {
  open: boolean; onClose: () => void; title: string; badge?: React.ReactNode; children: React.ReactNode;
}) {
  const { width, dragging, handleProps } = useResizable("pe-ai-width", 420, 300, 760, "left");
  return (
    <>
      {/* small screens: overlay backdrop */}
      <div className={`fixed inset-0 z-30 bg-black/30 transition-opacity xl:hidden ${open ? "opacity-100" : "pointer-events-none opacity-0"}`} onClick={onClose} aria-hidden />
      <aside aria-label={title}
        className={`fixed inset-y-0 right-0 z-40 flex max-w-[92vw] flex-col border-l border-line bg-surface shadow-[var(--shadow-hover)] xl:static xl:z-auto xl:max-w-none xl:shadow-none
          ${open ? "translate-x-0" : "pointer-events-none translate-x-full xl:hidden"} ${dragging ? "" : "transition-transform duration-300"}`}
        style={{ width }}>
        <div {...handleProps} aria-label="Resize panel" className="group absolute inset-y-0 -left-1.5 z-10 hidden w-3 cursor-col-resize xl:block">
          <Grip dragging={dragging} />
        </div>
        <header className="flex h-16 shrink-0 items-center justify-between gap-3 border-b border-line px-5">
          <div className="flex items-center gap-2.5">
            <span className="grid h-8 w-8 place-items-center rounded-lg bg-ink text-bg" aria-hidden>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M12 3v3m0 12v3M3 12h3m12 0h3M6 6l2 2m8 8 2 2m0-12-2 2M8 16l-2 2" /></svg>
            </span>
            <h2 className="m-0 text-[15px] font-semibold">{title}</h2>
            {badge}
          </div>
          <button className="btn h-8 w-8 p-0" onClick={onClose} aria-label={`Close ${title}`}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" aria-hidden><path d="M18 6 6 18M6 6l12 12" /></svg>
          </button>
        </header>
        <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
      </aside>
    </>
  );
}
