"use client";

import { useEffect, useState } from "react";

export type ChartColors = {
  surface: string;
  ink: string;
  ink2: string;
  muted: string;
  grid: string;
  axis: string;
  series1: string;
  series2: string;
  series3: string;
  bench: string;
  up: string;
  down: string;
  good: string;
  critical: string;
  poc: string;
};

function read(): ChartColors {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    surface: v("--surface"),
    ink: v("--ink"),
    ink2: v("--ink-2"),
    muted: v("--muted"),
    grid: v("--grid"),
    axis: v("--axis"),
    series1: v("--series-1"),
    series2: v("--series-2"),
    series3: v("--series-3"),
    bench: v("--bench"),
    up: v("--candle-up"),
    down: v("--candle-down"),
    good: v("--good"),
    critical: v("--critical"),
    poc: v("--poc"),
  };
}

/** สีจาก CSS token ปัจจุบัน — อัปเดตเมื่อผู้ใช้สลับธีมหรือระบบเปลี่ยนโหมด */
export function useChartColors(): ChartColors | null {
  const [colors, setColors] = useState<ChartColors | null>(null);
  useEffect(() => {
    const update = () => setColors(read());
    update();
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", update);
    const mo = new MutationObserver(update);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      mq.removeEventListener("change", update);
      mo.disconnect();
    };
  }, []);
  return colors;
}

/** hex -> rgba (ใช้ทำแถบพื้นหลังจาง ๆ) */
export function alpha(hex: string, a: number): string {
  const h = hex.replace("#", "");
  if (h.length !== 6) return hex;
  const n = parseInt(h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}
