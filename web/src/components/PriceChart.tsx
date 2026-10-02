"use client";

import {
  AreaSeries,
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  CrosshairMode,
  LineSeries,
  LineType,
  LineStyle,
  type IChartApi,
  type IPriceLine,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Candle, PaperPosition, PaperTrade, Strategy } from "@/lib/api";
import { fmtPrice, fmtR, fmtTime, tickFormatter } from "@/lib/format";
import { BoxesPrimitive, type ChartBox } from "@/lib/boxes";
import { alpha, useChartColors } from "@/lib/theme";

type Readout = { time: number; o: number; h: number; l: number; c: number; stop: number | null; brk: number | null; regime: number };

const ts = (t: number) => t as UTCTimestamp;

export type DemoOverlay = { symbol: string; trades: PaperTrade[]; positions: PaperPosition[] };

/** เวลา ms -> เวลาเปิดของแท่งที่มีเวลานั้นอยู่ (วินาที) : จุดเข้า/ออกของเดโมเกิดกลางแท่งได้ */
const toBar = (ms: number, tfSec: number) => Math.floor(ms / 1000 / tfSec) * tfSec;

export default function PriceChart({ candles, live, strategy, demo, tfSec = 14_400 }: {
  candles: Candle[];
  live: Candle | null;
  strategy: Strategy | null;
  /** ถ้าส่งมา : แสดงไม้ของบัญชีเดโมแทนไม้ของ backtest */
  demo?: DemoOverlay;
  tfSec?: number;
}) {
  const box = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const s = useRef<{
    candle: ISeriesApi<"Candlestick">;
    regime: ISeriesApi<"Area">;
    brk: ISeriesApi<"Line">;
    stop: ISeriesApi<"Line">;
    markers: ISeriesMarkersPluginApi<Time>;
    boxes: BoxesPrimitive;
  } | null>(null);
  const colors = useChartColors();
  const [hover, setHover] = useState<Readout | null>(null);
  const [showProfile, setShowProfile] = useState(true);
  const [showSideways, setShowSideways] = useState(true);
  const lines = useRef<IPriceLine[]>([]);
  const demoPos = demo?.positions.find((p) => p.symbol === demo.symbol) ?? null;
  const isDemo = demo != null;

  // ค่าทุกแท่ง (ไว้แสดงตอนเลื่อนเมาส์) : เวลา -> ระดับ SL / breakout / ทิศ
  const lookup = useMemo(() => {
    const m = new Map<number, { stop: number | null; brk: number | null; regime: number }>();
    if (strategy) {
      const se = strategy.series;
      se.time.forEach((t, i) => m.set(t, { stop: se.stop[i], brk: se.breakout[i], regime: se.regime[i] }));
    }
    return m;
  }, [strategy]);

  // สร้างกราฟครั้งเดียว
  useEffect(() => {
    if (!box.current) return;
    const c = createChart(box.current, {
      autoSize: true,
      crosshair: { mode: CrosshairMode.Normal },
      localization: { timeFormatter: (t: Time) => fmtTime(t as number), priceFormatter: (p: number) => fmtPrice(p) },
      timeScale: { timeVisible: true, secondsVisible: false, tickMarkFormatter: tickFormatter, rightOffset: 6 },
      rightPriceScale: { borderVisible: false },
      layout: { attributionLogo: true },
    });
    // แถบพื้นหลังทิศเทรนด์ : area เต็มความสูงบน scale แยก (ต่อเนื่อง ไม่เป็นริ้วแบบ histogram)
    const regime = c.addSeries(AreaSeries, {
      priceScaleId: "regime", lastValueVisible: false, priceLineVisible: false, lineVisible: false,
      crosshairMarkerVisible: false, autoscaleInfoProvider: () => ({ priceRange: { minValue: 0, maxValue: 1 } }),
    });
    c.priceScale("regime").applyOptions({ visible: false, scaleMargins: { top: 0, bottom: 0 } });
    const candle = c.addSeries(CandlestickSeries, { borderVisible: false, priceLineVisible: true });
    const brk = c.addSeries(LineSeries, {
      lineWidth: 1, lineType: LineType.WithSteps, title: "breakout", priceLineVisible: false, crosshairMarkerVisible: false,
    });
    const stop = c.addSeries(LineSeries, {
      lineWidth: 2, lineType: LineType.WithSteps, title: "SL", priceLineVisible: false, crosshairMarkerVisible: false,
    });
    const markers = createSeriesMarkers(candle, []);
    const boxes = new BoxesPrimitive();
    candle.attachPrimitive(boxes);
    chart.current = c;
    s.current = { candle, regime, brk, stop, markers, boxes };
    return () => {
      c.remove();
      chart.current = null;
      s.current = null;
    };
  }, []);

  // สีตามธีม
  useEffect(() => {
    if (!colors || !chart.current || !s.current) return;
    chart.current.applyOptions({
      layout: { background: { color: colors.surface }, textColor: colors.muted, fontFamily: "system-ui, sans-serif" },
      grid: { vertLines: { visible: false }, horzLines: { color: colors.grid } },
      timeScale: { borderColor: colors.axis },
      crosshair: { vertLine: { color: colors.muted, labelBackgroundColor: colors.ink2 }, horzLine: { color: colors.muted, labelBackgroundColor: colors.ink2 } },
    });
    s.current.candle.applyOptions({
      upColor: colors.up, downColor: colors.down, wickUpColor: colors.up, wickDownColor: colors.down,
    });
    s.current.brk.applyOptions({ color: colors.muted });
    s.current.stop.applyOptions({ color: colors.critical });
  }, [colors]);

  // ข้อมูล
  useEffect(() => {
    const se = s.current;
    if (!se || !colors) return;
    se.candle.setData(candles.map((k) => ({ time: ts(k.time), open: k.open, high: k.high, low: k.low, close: k.close })));
    if (!strategy) return;
    const st = strategy.series;
    const first = candles.length ? candles[0].time : 0;
    se.regime.setData(st.time.map((t, i) => {
      const col = st.regime[i] === 1 ? alpha(colors.up, 0.08) : st.regime[i] === -1 ? alpha(colors.down, 0.08) : "rgba(0,0,0,0)";
      return { time: ts(t), value: 1, topColor: col, bottomColor: col, lineColor: col };
    }));
    se.brk.setData(st.time.map((t, i) => (st.breakout[i] == null ? { time: ts(t) } : { time: ts(t), value: st.breakout[i]! })));
    for (const l of lines.current) se.candle.removePriceLine(l);
    lines.current = [];
    const mk: SeriesMarker<Time>[] = [];

    // กรอบ sideway (ADX < 20) + Value Area / POC — กรอบที่เริ่มก่อนข้อมูลที่โหลดไว้ ตัดให้เริ่มที่แท่งแรก
    const bx: ChartBox[] = [];
    if (showSideways) {
      for (const b of strategy.sideways) {
        if (b.to < first) continue;
        bx.push({
          from: Math.max(b.from, first), to: b.to, top: b.top, bottom: b.bottom, extendRight: b.active,
          fill: alpha(colors.muted, 0.1), border: alpha(colors.muted, 0.55), label: "sideway", labelColor: colors.ink2,
        });
      }
    }
    const pf = strategy.profile;
    if (showProfile && pf) {
      bx.push({
        from: Math.max(pf.from, first), to: candles.length ? candles[candles.length - 1].time : pf.from,
        top: pf.vah, bottom: pf.val, extendRight: true,
        fill: alpha(colors.poc, 0.1), border: alpha(colors.poc, 0.6),
        midLine: { price: pf.poc, color: colors.poc }, label: "Value Area 70%", labelColor: colors.poc,
      });
      lines.current.push(se.candle.createPriceLine({
        price: pf.poc, color: colors.poc, lineVisible: false, axisLabelVisible: true, title: "POC",
      }));
    }
    se.boxes.setBoxes(bx);

    if (!demo) {
      // ไม้ที่ระบบคำนวณย้อนหลัง (backtest)
      se.stop.setData(st.time.map((t, i) => (st.stop[i] == null ? { time: ts(t) } : { time: ts(t), value: st.stop[i]! })));
      // ป้าย SL ที่ขอบขวาแสดงเฉพาะตอนถือไม้ (ไม่งั้นจะโชว์ SL ของไม้ที่ปิดไปแล้ว)
      se.stop.applyOptions({ lastValueVisible: strategy.status.in_position, title: strategy.status.in_position ? "SL" : "" });
      for (const tr of strategy.trades) {
        if (tr.entry_time >= first) {
          mk.push({ time: ts(tr.entry_time), position: "belowBar", shape: "arrowUp", color: colors.series1, text: "เข้า" });
        }
        if (tr.exit_time != null && tr.exit_time >= first) {
          mk.push({
            time: ts(tr.exit_time), position: "aboveBar", shape: "arrowDown",
            color: (tr.r ?? 0) >= 0 ? colors.good : colors.critical, text: fmtR(tr.r),
          });
        }
      }
    } else {
      // ไม้ของบัญชีเดโม : จุดเข้า/ออกจริง + เส้นราคาเข้า/SL ของไม้ที่ถืออยู่
      se.stop.setData([]);
      for (const tr of demo.trades.filter((x) => x.symbol === demo.symbol)) {
        mk.push({ time: ts(toBar(tr.entry_time, tfSec)), position: "belowBar", shape: "arrowUp", color: colors.series1,
          text: `เข้า ${fmtPrice(tr.entry_price)}` });
        mk.push({ time: ts(toBar(tr.exit_time, tfSec)), position: "aboveBar", shape: "arrowDown",
          color: tr.r >= 0 ? colors.good : colors.critical, text: `ออก ${fmtPrice(tr.exit_price)} · ${fmtR(tr.r)}` });
      }
      if (demoPos) {
        mk.push({ time: ts(toBar(demoPos.entry_time, tfSec)), position: "belowBar", shape: "arrowUp", color: colors.series1,
          text: `เข้า ${fmtPrice(demoPos.entry_price)}` });
        lines.current.push(se.candle.createPriceLine({
          price: demoPos.entry_price, color: colors.series1, lineWidth: 1, lineStyle: LineStyle.Dashed,
          axisLabelVisible: true, title: "เข้า (เดโม)",
        }));
        lines.current.push(se.candle.createPriceLine({
          price: demoPos.stop, color: colors.critical, lineWidth: 2, lineStyle: LineStyle.Solid,
          axisLabelVisible: true, title: "SL (เดโม)",
        }));
      }
    }
    mk.sort((a, b) => (a.time as number) - (b.time as number));
    se.markers.setMarkers(mk);
  }, [candles, strategy, colors, demo, demoPos, tfSec, showProfile, showSideways]);

  // แท่งที่กำลังวิ่ง (จาก WebSocket)
  useEffect(() => {
    if (live && s.current && candles.length && live.time >= candles[candles.length - 1].time) {
      s.current.candle.update({ time: ts(live.time), open: live.open, high: live.high, low: live.low, close: live.close });
    }
  }, [live, candles]);

  // readout ตามตำแหน่งเมาส์
  useEffect(() => {
    const c = chart.current;
    const se = s.current;
    if (!c || !se) return;
    const handler = (p: MouseEventParams<Time>) => {
      const bar = p.time != null ? (p.seriesData.get(se.candle) as { open: number; high: number; low: number; close: number } | undefined) : undefined;
      if (!bar || p.time == null) return setHover(null);
      const x = lookup.get(p.time as number);
      // โหมดเดโม : เส้น SL ย้อนหลังของ backtest ไม่เกี่ยวกับบัญชีเดโม -> ไม่แสดง
      setHover({ time: p.time as number, o: bar.open, h: bar.high, l: bar.low, c: bar.close,
        stop: isDemo ? null : x?.stop ?? null, brk: x?.brk ?? null, regime: x?.regime ?? 0 });
    };
    c.subscribeCrosshairMove(handler);
    return () => c.unsubscribeCrosshairMove(handler);
  }, [lookup, isDemo]);

  const last = candles.length ? candles[candles.length - 1] : null;
  const shown: Readout | null = hover ?? (last && strategy ? {
    time: (live ?? last).time, o: (live ?? last).open, h: (live ?? last).high, l: (live ?? last).low, c: (live ?? last).close,
    stop: demo ? demoPos?.stop ?? null : strategy.status.stop, brk: strategy.status.breakout_level,
    regime: strategy.status.regime === "up" ? 1 : strategy.status.regime === "down" ? -1 : 0,
  } : null);

  return (
    <div>
      <div className="tnum mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2" aria-live="polite">
        {shown ? (
          <>
            <span className="text-ink">{fmtTime(shown.time)}</span>
            <span>O <b className="font-medium text-ink">{fmtPrice(shown.o)}</b></span>
            <span>H <b className="font-medium text-ink">{fmtPrice(shown.h)}</b></span>
            <span>L <b className="font-medium text-ink">{fmtPrice(shown.l)}</b></span>
            <span>C <b className="font-medium text-ink">{fmtPrice(shown.c)}</b></span>
            <span>breakout <b className="font-medium text-ink">{fmtPrice(shown.brk)}</b></span>
            <span>SL <b className="font-medium text-ink">{fmtPrice(shown.stop)}</b></span>
            <span>ทิศ <b className="font-medium text-ink">{shown.regime === 1 ? "ขาขึ้น" : shown.regime === -1 ? "ขาลง" : "ไม่ชัด"}</b></span>
          </>
        ) : (
          <span>กำลังโหลด…</span>
        )}
      </div>
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
        <label className="inline-flex items-center gap-1.5">
          <input type="checkbox" checked={showProfile} onChange={(e) => setShowProfile(e.target.checked)} />
          <span className="inline-block h-0.5 w-4" style={{ background: "var(--poc)" }} />
          POC / Value Area {strategy?.profile ? `(${strategy.profile.bars} แท่งล่าสุด · POC ${fmtPrice(strategy.profile.poc)})` : ""}
        </label>
        <label className="inline-flex items-center gap-1.5">
          <input type="checkbox" checked={showSideways} onChange={(e) => setShowSideways(e.target.checked)} />
          <span className="inline-block h-3 w-3 rounded-sm border" style={{ background: "color-mix(in srgb, var(--muted) 15%, transparent)", borderColor: "var(--muted)" }} />
          กรอบ sideway (ADX &lt; 20 ต่อเนื่อง ≥ 12 แท่ง)
        </label>
      </div>
      <div ref={box} className="h-[460px] w-full" />
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
        <Key color="var(--candle-up)" kind="box">พื้นหลังเขียว = TF ใหญ่ขาขึ้น (ระบบเข้าได้)</Key>
        <Key color="var(--muted)" kind="line">breakout = High สูงสุด 20 แท่ง (ปิดเหนือเส้น = สัญญาณเข้า)</Key>
        {demo ? (
          <>
            <Key color="var(--series-1)" kind="arrow">▲ เข้า · ▼ ออก ของบัญชีเดโม (ราคาที่ได้จริง)</Key>
            <Key color="var(--critical)" kind="line">เส้นราคาเข้า / SL ของไม้เดโมที่ถืออยู่</Key>
          </>
        ) : (
          <>
            <Key color="var(--critical)" kind="line">SL / trailing stop ของไม้ที่ถือ</Key>
            <Key color="var(--series-1)" kind="arrow">▲ เข้า · ▼ ออก = ไม้ที่ระบบคำนวณย้อนหลัง (backtest) — ไม้ของบัญชีเดโมดูที่หน้า &ldquo;บัญชีเดโม&rdquo;</Key>
          </>
        )}
      </div>
    </div>
  );
}

function Key({ color, kind, children }: { color: string; kind: "line" | "box" | "arrow"; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {kind === "line" && <span className="inline-block h-0.5 w-4" style={{ background: color }} />}
      {kind === "box" && <span className="inline-block h-3 w-3 rounded-sm opacity-40" style={{ background: color }} />}
      {kind === "arrow" && <span style={{ color }}>▲</span>}
      {children}
    </span>
  );
}
