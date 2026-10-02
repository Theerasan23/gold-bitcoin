"use client";

import {
  createChart,
  CrosshairMode,
  LineSeries,
  PriceScaleMode,
  type IChartApi,
  type ISeriesApi,
  type MouseEventParams,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef, useState } from "react";
import { fmtDate, fmtNum, tickFormatter } from "@/lib/format";
import { useChartColors, type ChartColors } from "@/lib/theme";

export type EquityLine = {
  key: string;
  label: string;
  colorVar: keyof ChartColors;   // สีตามลำดับ slot คงที่ (series1, series2, …) — เส้นเทียบใช้ bench
  width: 1 | 2 | 3;
  time: number[];
  value: number[];
};

type Tip = { left: number; top: number; time: number; rows: { label: string; color: string; value: number }[] } | null;

export default function EquityChart({ lines, log }: { lines: EquityLine[]; log: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<Map<string, ISeriesApi<"Line">>>(new Map());
  const colors = useChartColors();
  const [tip, setTip] = useState<Tip>(null);

  useEffect(() => {
    if (!box.current) return;
    const c = createChart(box.current, {
      autoSize: true,
      crosshair: { mode: CrosshairMode.Magnet, horzLine: { visible: false, labelVisible: false } },
      rightPriceScale: { borderVisible: false },
      // ข้อมูลรายวันหลายปี (2,000+ จุด) : ต้องให้แต่ละจุดแคบกว่า 0.5px ที่เป็นค่าเริ่มต้น ไม่งั้นแสดงไม่ครบช่วง
      timeScale: { borderVisible: false, minBarSpacing: 0.05, tickMarkFormatter: tickFormatter },
      localization: { timeFormatter: (t: Time) => fmtDate(t as number), priceFormatter: (p: number) => fmtNum(p, 0) },
      handleScroll: false,
      handleScale: false,
    });
    chart.current = c;
    const map = series.current;
    // ตอนสร้างกล่องอาจยังกว้าง 0 -> จัดให้เห็นทั้งช่วงทุกครั้งที่ขนาดเปลี่ยน
    const ro = new ResizeObserver(() => c.timeScale().fitContent());
    ro.observe(box.current);
    return () => {
      ro.disconnect();
      c.remove();
      chart.current = null;
      map.clear();
    };
  }, []);

  useEffect(() => {
    chart.current?.priceScale("right").applyOptions({ mode: log ? PriceScaleMode.Logarithmic : PriceScaleMode.Normal });
  }, [log]);

  useEffect(() => {
    const c = chart.current;
    if (!c || !colors) return;
    c.applyOptions({
      layout: { background: { color: colors.surface }, textColor: colors.muted, fontFamily: "system-ui, sans-serif" },
      grid: { vertLines: { visible: false }, horzLines: { color: colors.grid } },
      crosshair: { vertLine: { color: colors.muted, labelBackgroundColor: colors.ink2 } },
    });
    const map = series.current;
    for (const [k, s] of map) {
      if (!lines.find((l) => l.key === k)) {
        c.removeSeries(s);
        map.delete(k);
      }
    }
    for (const l of lines) {
      let s = map.get(l.key);
      if (!s) {
        s = c.addSeries(LineSeries, { priceLineVisible: false, crosshairMarkerRadius: 4 });
        map.set(l.key, s);
      }
      // ป้ายท้ายเส้น (title + ค่าล่าสุด) = direct label
      s.applyOptions({ color: colors[l.colorVar], lineWidth: l.width, title: l.label, lastValueVisible: true });
      s.setData(l.time.map((t, i) => ({ time: t as UTCTimestamp, value: l.value[i] })));
    }
    c.timeScale().fitContent();
  }, [lines, colors]);

  useEffect(() => {
    const c = chart.current;
    if (!c || !colors) return;
    const handler = (p: MouseEventParams<Time>) => {
      if (p.time == null || !p.point) return setTip(null);
      const rows = lines.flatMap((l) => {
        const s = series.current.get(l.key);
        const d = s ? (p.seriesData.get(s) as { value?: number } | undefined) : undefined;
        return d?.value != null ? [{ label: l.label, color: colors[l.colorVar], value: d.value }] : [];
      }).sort((a, b) => b.value - a.value);
      const w = box.current?.clientWidth ?? 600;
      setTip({ left: Math.min(p.point.x + 16, w - 200), top: Math.max(8, p.point.y - 40), time: p.time as number, rows });
    };
    c.subscribeCrosshairMove(handler);
    return () => c.unsubscribeCrosshairMove(handler);
  }, [lines, colors]);

  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2" aria-label="คำอธิบายเส้น">
        {lines.map((l) => (
          <span key={l.key} className="inline-flex items-center gap-1.5">
            <span className="inline-block w-4" style={{ height: l.width, background: `var(--${l.colorVar.replace(/(\d)/, "-$1")})` }} />
            {l.label}
          </span>
        ))}
      </div>
      <div className="relative">
        <div ref={box} className="h-[380px] w-full" />
        {tip && tip.rows.length > 0 && (
          <div
            className="tnum pointer-events-none absolute z-10 min-w-44 rounded-lg border border-line bg-surface p-2.5 text-xs shadow-lg"
            style={{ left: tip.left, top: tip.top }}
          >
            <div className="mb-1.5 text-ink-2">{fmtDate(tip.time)}</div>
            {tip.rows.map((r) => (
              <div key={r.label} className="flex items-center justify-between gap-3 py-0.5">
                <span className="inline-flex items-center gap-1.5 text-ink-2">
                  <span className="inline-block h-0.5 w-3" style={{ background: r.color }} />
                  {r.label}
                </span>
                <b className="font-semibold text-ink">{fmtNum(r.value, 1)}</b>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
