"use client";

import { useCallback, useEffect, useState } from "react";
import PriceChart from "@/components/PriceChart";
import { Badge, Card, ErrorBox, Segmented, Stat } from "@/components/ui";
import { getJSON, type Candle, type Strategy } from "@/lib/api";
import { fmtNum, fmtPct, fmtPrice, fmtR, fmtTime, SYMBOL_LABEL, TF_SEC } from "@/lib/format";
import { useLiveKline } from "@/lib/useLiveKline";

const SYMBOLS = [
  { value: "BTCUSDT", label: "Bitcoin" },
  { value: "PAXGUSDT", label: "ทองคำ (PAXG)" },
];
const TFS = [
  { value: "15m", label: "15m" },
  { value: "1h", label: "1h" },
  { value: "4h", label: "4h" },
  { value: "1d", label: "1D" },
];
const BARS = 500;
const REASON: Record<string, string> = {
  stop: "โดน SL", trailing: "trailing stop", trend_flip: "เทรนด์ TF ใหญ่กลับ", end: "สิ้นสุดข้อมูล",
};

// strategy ก่อน : API จะดึงแท่งใหม่จาก Binance ถ้าข้อมูลเก่า แล้ว candles จึงได้ข้อมูลล่าสุด
async function fetchAll(symbol: string, tf: string, signal: AbortSignal) {
  const st = await getJSON<Strategy>(`strategy/${symbol}/${tf}?bars=${BARS}`, signal);
  const cs = await getJSON<Candle[]>(`candles/${symbol}/${tf}?limit=${BARS}`, signal);
  return { st, cs };
}

export default function Dashboard() {
  const [symbol, setSymbol] = useState("BTCUSDT");
  const [tf, setTf] = useState("4h");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [strategy, setStrategy] = useState<Strategy | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const loading = loadedKey !== `${symbol}|${tf}`;
  const [capital, setCapital] = useState(10_000);
  const [riskPct, setRiskPct] = useState(1);

  const [nonce, setNonce] = useState(0);   // เพิ่มเมื่อแท่งปิด -> โหลดใหม่

  useEffect(() => {
    const ac = new AbortController();
    const key = `${symbol}|${tf}`;
    const run = () =>
      fetchAll(symbol, tf, ac.signal).then(
        ({ st, cs }) => {
          setStrategy(st);
          setCandles(cs);
          setError(null);
          setLoadedKey(key);
        },
        (e: Error) => e.name !== "AbortError" && setError(e.message),
      );
    run();
    const id = setInterval(run, 5 * 60_000);
    return () => {
      ac.abort();
      clearInterval(id);
    };
  }, [symbol, tf, nonce]);

  const onBarClose = useCallback(() => {
    setTimeout(() => setNonce((n) => n + 1), 5_000);   // รอ Binance ปิดแท่งให้เรียบร้อยก่อนดึง
  }, []);
  const { live, state } = useLiveKline(symbol, tf, onBarClose);

  const st = strategy?.status;
  const price = live?.close ?? st?.close ?? null;
  const prevClose = candles.length ? candles[candles.length - 1].close : null;
  const chg = price != null && prevClose ? ((price - prevClose) / prevClose) * 100 : null;

  // ขนาดไม้ : เสี่ยง riskPct % ของทุน · SL = 2 ATR · spot ซื้อได้ไม่เกินทุน
  const riskMoney = (capital * riskPct) / 100;
  const qtyRisk = st && st.risk_per_unit > 0 ? riskMoney / st.risk_per_unit : 0;
  const qtyCap = price ? capital / price : 0;
  const qty = Math.min(qtyRisk, qtyCap);
  const capped = qtyRisk > qtyCap;

  const recent = strategy ? [...strategy.trades].reverse().slice(0, 12) : [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Segmented label="เหรียญ" value={symbol} options={SYMBOLS} onChange={setSymbol} />
        <Segmented label="Timeframe" value={tf} options={TFS} onChange={setTf} />
        <span className="text-xs text-muted">
          ผลแต่ละ timeframe ดูที่หน้า &ldquo;ผลทดสอบ&rdquo; · ราคาสด:{" "}
          <Badge tone={state === "live" ? "good" : state === "offline" ? "bad" : "neutral"}>
            {state === "live" ? "เชื่อมต่อ Binance" : state === "offline" ? "หลุด กำลังต่อใหม่" : "กำลังเชื่อมต่อ"}
          </Badge>
        </span>
      </div>

      {error && <ErrorBox error={error} />}

      <div className={`grid grid-cols-2 gap-3 lg:grid-cols-4 ${loading && strategy ? "opacity-70" : ""}`}>
        <Stat
          label={`ราคา ${SYMBOL_LABEL[symbol]}`}
          value={fmtPrice(price)}
          sub={chg != null ? `${fmtPct(chg, 2, true)} จากแท่งก่อน` : undefined}
          tone={chg == null ? undefined : chg >= 0 ? "good" : "bad"}
        />
        <Stat
          label="ทิศเทรนด์ (TF ใหญ่)"
          value={st ? (st.regime === "up" ? "▲ ขาขึ้น" : st.regime === "down" ? "▼ ขาลง" : "– ไม่ชัด") : "–"}
          sub={st ? (st.regime === "up" ? "ระบบเปิด long ได้" : "ระบบไม่เข้า (ถือเงินสด)") : undefined}
          tone={st?.regime === "up" ? "good" : st?.regime === "down" ? "bad" : undefined}
        />
        <Stat
          label="สถานะไม้"
          value={st ? (st.in_position ? "ถือ Long" : "ไม่มีไม้") : "–"}
          sub={st ? (st.in_position
            ? `เข้า ${fmtPrice(st.entry_price)} · กำไร ${fmtPct(price && st.entry_price ? ((price - st.entry_price) / st.entry_price) * 100 : null, 2, true)}`
            : st.regime === "up" ? `รอปิดเหนือ ${fmtPrice(st.breakout_level)}` : "รอเทรนด์ขาขึ้น") : undefined}
        />
        <Stat
          label={st?.in_position ? "SL ปัจจุบัน (แท่งถัดไป)" : "จุด breakout ถัดไป"}
          value={fmtPrice(st ? (st.in_position ? st.stop : st.breakout_level) : null)}
          sub={st && price ? (st.in_position && st.stop
            ? `ห่าง ${fmtPct(((price - st.stop) / price) * 100, 2)}`
            : `ห่าง ${fmtPct(((st.breakout_level - price) / price) * 100, 2)}`) : undefined}
        />
      </div>

      <Card
        title={`${SYMBOL_LABEL[symbol]} · ${tf}`}
        aside={st ? `ข้อมูลถึง ${fmtTime(st.time)} · อัปเดตทุกครั้งที่แท่งปิด` : undefined}
      >
        <PriceChart candles={candles} live={live} strategy={strategy} tfSec={TF_SEC[tf]} />
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card title="คำนวณขนาดไม้ (ถ้าเข้าตอนนี้)">
          <div className="space-y-3 text-sm">
            <label className="block">
              <span className="text-ink-2">เงินทุน (USDT)</span>
              <input
                type="number" min={0} step={100} value={capital}
                onChange={(e) => setCapital(Math.max(0, +e.target.value))}
                className="tnum mt-1 w-full rounded-md border border-line bg-surface-2 px-3 py-1.5 text-ink"
              />
            </label>
            <label className="block">
              <span className="text-ink-2">เสี่ยงต่อไม้ (%)</span>
              <input
                type="number" min={0.1} max={5} step={0.1} value={riskPct}
                onChange={(e) => setRiskPct(Math.min(5, Math.max(0.1, +e.target.value)))}
                className="tnum mt-1 w-full rounded-md border border-line bg-surface-2 px-3 py-1.5 text-ink"
              />
            </label>
            <dl className="tnum grid grid-cols-2 gap-y-1.5">
              <dt className="text-ink-2">ระยะ SL (2 ATR)</dt>
              <dd className="text-right text-ink">{fmtPrice(st?.risk_per_unit)}</dd>
              <dt className="text-ink-2">SL ที่ราคา</dt>
              <dd className="text-right text-ink">{fmtPrice(price && st ? price - st.risk_per_unit : null)}</dd>
              <dt className="text-ink-2">ยอมเสียได้</dt>
              <dd className="text-right text-ink">{fmtNum(capped && price && st ? qty * st.risk_per_unit : riskMoney)} USDT</dd>
              <dt className="text-ink-2">ซื้อได้</dt>
              <dd className="text-right font-semibold text-ink">{fmtNum(qty, qty >= 1 ? 3 : 5)} {symbol.replace("USDT", "")}</dd>
              <dt className="text-ink-2">มูลค่า</dt>
              <dd className="text-right text-ink">{fmtNum(price ? qty * price : null)} USDT</dd>
            </dl>
            {capped && (
              <p className="text-xs text-ink-2">
                <Badge tone="neutral">ขนาดถูกจำกัดที่เงินทุน</Badge> (spot ไม่กู้) — ความเสี่ยงจริงต่ำกว่าที่ตั้ง
              </p>
            )}
            <p className="text-xs text-muted">ตัวช่วยคำนวณเท่านั้น ไม่ใช่คำแนะนำการลงทุน</p>
          </div>
        </Card>

        <Card
          className="lg:col-span-2"
          title="ไม้ล่าสุดของระบบ"
          aside={strategy ? `ตั้งแต่ ${fmtTime(strategy.summary.since, false)} · ${strategy.summary.trades} ไม้ · รวม ${fmtR(strategy.summary.total_r)} · ชนะ ${fmtPct(strategy.summary.win_rate, 0)}` : undefined}
        >
          <div className="overflow-x-auto">
            <table className="tnum w-full text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs text-ink-2">
                  <th className="py-1.5 pr-3 font-medium">เข้า</th>
                  <th className="py-1.5 pr-3 text-right font-medium">ราคาเข้า</th>
                  <th className="py-1.5 pr-3 font-medium">ออก</th>
                  <th className="py-1.5 pr-3 text-right font-medium">ราคาออก</th>
                  <th className="py-1.5 pr-3 text-right font-medium">ผล</th>
                  <th className="py-1.5 font-medium">เหตุผลที่ออก</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((t) => (
                  <tr key={t.entry_time} className="border-b border-line last:border-0">
                    <td className="py-1.5 pr-3 text-ink">{fmtTime(t.entry_time)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtPrice(t.entry_price)}</td>
                    <td className="py-1.5 pr-3 text-ink">{t.open ? "ยังถืออยู่" : fmtTime(t.exit_time!)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtPrice(t.exit_price)}</td>
                    <td className="py-1.5 pr-3 text-right">
                      {t.r == null ? "–" : <Badge tone={t.r >= 0 ? "good" : "bad"}>{fmtR(t.r)}</Badge>}
                    </td>
                    <td className="py-1.5 text-ink-2">{t.exit_reason ? REASON[t.exit_reason] ?? t.exit_reason : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
