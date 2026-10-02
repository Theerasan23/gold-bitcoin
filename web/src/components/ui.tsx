import type { ReactNode } from "react";

export function Card({ title, aside, children, className = "" }: {
  title?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-xl border border-line bg-surface p-4 ${className}`}>
      {(title || aside) && (
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          {title && <h2 className="text-sm font-semibold text-ink">{title}</h2>}
          {aside && <div className="text-xs text-muted">{aside}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, tone }: {
  label: string; value: ReactNode; sub?: ReactNode; tone?: "good" | "bad" | "warn";
}) {
  const toneCls = tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : "text-ink";
  return (
    <div className="rounded-xl border border-line bg-surface p-4">
      <div className="text-xs text-ink-2">{label}</div>
      <div className={`mt-1 text-2xl font-semibold ${toneCls}`}>{value}</div>
      {sub && <div className="mt-1 text-xs text-muted">{sub}</div>}
    </div>
  );
}

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-line bg-surface p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={`rounded-md px-3 py-1.5 text-sm ${o.value === value ? "bg-surface-2 font-medium text-ink" : "text-ink-2 hover:text-ink"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** สถานะ ดี / กลาง / แย่ : ไอคอน + ข้อความเสมอ ไม่ใช้สีอย่างเดียว */
export function Badge({ tone, children }: { tone: "good" | "bad" | "neutral"; children: ReactNode }) {
  const icon = tone === "good" ? "✓" : tone === "bad" ? "✕" : "–";
  const cls = tone === "good" ? "text-good" : tone === "bad" ? "text-bad" : "text-ink-2";
  return (
    <span className={`inline-flex items-center gap-1 ${cls}`}>
      <span aria-hidden>{icon}</span>
      {children}
    </span>
  );
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div role="alert" className="rounded-xl border border-line bg-surface p-4 text-sm text-bad">
      โหลดข้อมูลไม่สำเร็จ: {error}
    </div>
  );
}
