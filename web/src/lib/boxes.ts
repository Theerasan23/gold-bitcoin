"use client";

import type {
  IChartApiBase,
  IPrimitivePaneRenderer,
  IPrimitivePaneView,
  ISeriesApi,
  ISeriesPrimitive,
  SeriesAttachedParameter,
  SeriesType,
  Time,
  UTCTimestamp,
} from "lightweight-charts";

export type ChartBox = {
  from: number;          // เวลาแท่งแรก (วินาที)
  to: number;            // เวลาแท่งสุดท้าย
  top: number;
  bottom: number;
  fill: string;
  border: string;
  extendRight?: boolean; // ยาวถึงขอบขวาของกราฟ (กรอบที่ยังไม่จบ)
  midLine?: { price: number; color: string };   // เส้นกลางกรอบ เช่น POC
  label?: string;
  labelColor?: string;
};

type Target = Parameters<IPrimitivePaneRenderer["draw"]>[0];

/** วาดสี่เหลี่ยมบนกราฟ (ใต้แท่งเทียน) — Lightweight Charts ไม่มี box ในตัว จึงทำเป็น series primitive */
export class BoxesPrimitive implements ISeriesPrimitive<Time> {
  private boxes: ChartBox[] = [];
  private chart: IChartApiBase<Time> | null = null;
  private series: ISeriesApi<SeriesType, Time> | null = null;
  private requestUpdate: (() => void) | null = null;

  private readonly view: IPrimitivePaneView = {
    zOrder: () => "bottom",
    renderer: () => ({ draw: (t: Target) => this.draw(t) }),
  };

  attached(p: SeriesAttachedParameter<Time>) {
    this.chart = p.chart;
    this.series = p.series;
    this.requestUpdate = p.requestUpdate;
  }

  detached() {
    this.chart = null;
    this.series = null;
    this.requestUpdate = null;
  }

  setBoxes(boxes: ChartBox[]) {
    this.boxes = boxes;
    this.requestUpdate?.();
  }

  paneViews() {
    return [this.view];
  }

  private draw(target: Target) {
    const ts = this.chart?.timeScale();
    const se = this.series;
    if (!ts || !se || this.boxes.length === 0) return;
    const half = (ts.options().barSpacing ?? 6) / 2;
    target.useBitmapCoordinateSpace(({ context: ctx, horizontalPixelRatio: hr, verticalPixelRatio: vr, mediaSize }) => {
      for (const b of this.boxes) {
        const xa = ts.timeToCoordinate(b.from as UTCTimestamp);
        const xb = b.extendRight ? mediaSize.width : ts.timeToCoordinate(b.to as UTCTimestamp);
        const ya = se.priceToCoordinate(b.top);
        const yb = se.priceToCoordinate(b.bottom);
        if (xa == null || xb == null || ya == null || yb == null) continue;
        const x1 = Math.round((xa - half) * hr);
        const x2 = Math.round((b.extendRight ? xb : xb + half) * hr);
        const y1 = Math.round(Math.min(ya, yb) * vr);
        const y2 = Math.round(Math.max(ya, yb) * vr);
        ctx.fillStyle = b.fill;
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
        ctx.strokeStyle = b.border;
        ctx.lineWidth = Math.max(1, Math.round(hr));
        ctx.strokeRect(x1 + 0.5, y1 + 0.5, x2 - x1 - 1, y2 - y1 - 1);
        if (b.midLine) {
          const ym = se.priceToCoordinate(b.midLine.price);
          if (ym != null) {
            ctx.strokeStyle = b.midLine.color;
            ctx.lineWidth = Math.max(2, Math.round(2 * hr));
            ctx.beginPath();
            ctx.moveTo(x1, Math.round(ym * vr));
            ctx.lineTo(x2, Math.round(ym * vr));
            ctx.stroke();
          }
        }
        if (b.label) {
          ctx.fillStyle = b.labelColor ?? b.border;
          ctx.font = `${Math.round(11 * vr)}px system-ui, sans-serif`;
          ctx.textBaseline = "top";
          ctx.fillText(b.label, x1 + Math.round(4 * hr), y1 + Math.round(3 * vr));
        }
      }
    });
  }
}
