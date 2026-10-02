"use client";

import { useEffect, useRef, useState } from "react";
import type { Candle } from "@/lib/api";

/** ราคาสดจาก Binance (แท่งที่กำลังวิ่ง) — เชื่อมต่อใหม่อัตโนมัติ */
export function useLiveKline(symbol: string, tf: string, onClose: () => void) {
  const key = `${symbol}|${tf}`;
  const [live, setLive] = useState<{ key: string; candle: Candle } | null>(null);
  const [conn, setConn] = useState<{ key: string; state: "live" | "offline" } | null>(null);
  const closeRef = useRef(onClose);
  useEffect(() => {
    closeRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let retry = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let stopped = false;
    const connect = () => {
      ws = new WebSocket(`wss://stream.binance.com:9443/ws/${symbol.toLowerCase()}@kline_${tf}`);
      ws.onopen = () => {
        retry = 0;
        setConn({ key, state: "live" });
      };
      ws.onmessage = (ev) => {
        const k = JSON.parse(ev.data).k;
        if (!k) return;
        setLive({ key, candle: { time: k.t / 1000, open: +k.o, high: +k.h, low: +k.l, close: +k.c, volume: +k.v } });
        if (k.x) closeRef.current();   // แท่งปิด -> ให้ระบบคำนวณใหม่
      };
      ws.onclose = () => {
        if (stopped) return;
        setConn({ key, state: "offline" });
        timer = setTimeout(connect, Math.min(30_000, 1000 * 2 ** retry++));
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => {
      stopped = true;
      clearTimeout(timer);
      ws?.close();
    };
  }, [symbol, tf, key]);
  // ค่าของเหรียญ/TF ก่อนหน้า ไม่นับ
  return {
    live: live?.key === key ? live.candle : null,
    state: conn?.key === key ? conn.state : ("connecting" as const),
  };
}

