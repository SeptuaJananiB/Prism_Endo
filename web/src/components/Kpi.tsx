import { useCountUp } from "../lib/hooks";

type Tone = "ok" | "warn" | "err" | "info" | undefined;

export function Kpi({ label, value, decimals = 0, suffix = "", caption, tone, chip, i = 0 }: {
  label: string; value: number | string; decimals?: number; suffix?: string; caption?: string;
  tone?: Tone; chip?: string; i?: number;
}) {
  const num = typeof value === "number" ? value : NaN;
  const v = useCountUp(Number.isFinite(num) ? num : 0);
  const shown = Number.isFinite(num)
    ? v.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
    : value;
  return (
    <div className="card card-hover rise flex min-h-[104px] flex-col justify-between p-4" style={{ ["--i" as string]: i }}>
      <div className="label">{label}</div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className="text-[22px] font-semibold leading-none tracking-tight tabular">{shown}{suffix}</span>
        {chip && <span className={`chip chip-${tone ?? "mute"}`}>{chip}</span>}
      </div>
      <div className="mt-2 truncate text-xs text-muted">{caption ?? " "}</div>
    </div>
  );
}
