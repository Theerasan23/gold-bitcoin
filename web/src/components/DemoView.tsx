"use client";

import { useEffect, useMemo, useState } from "react";
import EquityChart, { type EquityLine } from "@/components/EquityChart";
import PriceChart from "@/components/PriceChart";
import { Badge, Card, ErrorBox, Segmented, Stat } from "@/components/ui";
import {
  getJSON, type Candle, type PaperCompare, type PaperEquity, type PaperEvent, type PaperRunInfo, type PaperStatus,
  type PaperTrade, type Strategy,
} from "@/lib/api";
import { useLiveKline } from "@/lib/useLiveKline";
import { fmtNum, fmtPct, fmtPrice, fmtR, fmtTime, SYMBOL_LABEL, TF_LABEL, TF_SEC } from "@/lib/format";

const EVENT: Record<string, string> = {
  new_run: "เริ่มบัญชีใหม่", signal: "สัญญาณเข้า", entry: "เข้าไม้", stop_moved: "เลื่อน SL",
  signal_exit: "สัญญาณออก (ทิศกลับ)", exit: "ออกไม้", skip: "ข้ามสัญญาณ", error: "ข้อผิดพลาด",
};
const REASON: Record<string, string> = { stop: "โดน SL", trailing: "trailing stop", trend_flip: "ทิศกลับ" };
const sec = (ms: number) => Math.floor(ms / 1000);
const TF_ORDER = ["15m", "1h", "4h", "1d"];
const short = (s: string) => s.replace("USDT", "");

function eventDetail(e: PaperEvent): string {
  const n = (k: string) => (typeof e[k] === "number" ? (e[k] as number) : null);
  switch (e.type) {
    case "new_run": return `ทุน ${fmtNum(n("capital"), 0)} · เสี่ยง ${n("risk_pct")}%/ไม้`;
    case "signal": return `ปิด ${fmtPrice(n("close"))} เหนือ breakout -> เข้าที่แท่งถัดไป`;
    case "entry": return `ราคา ${fmtPrice(n("price"))} (backtest สมมติ ${fmtPrice(n("ref_price"))}) · ${fmtNum(n("qty"), 5)} · SL ${fmtPrice(n("stop"))}`;
    case "stop_moved": return `${fmtPrice(n("old"))} -> ${fmtPrice(n("new"))}`;
    case "exit": return `ราคา ${fmtPrice(n("price"))} · ${REASON[String(e.reason)] ?? e.reason} · ${fmtR(n("r"))}`;
    default: return String(e.message ?? e.reason ?? "");
  }
}

/** เส้นเวลาของ chart ต้องเรียงและไม่ซ้ำ */
function series(points: { time: number; equity: number }[]) {
  const m = new Map<number, number>();
  for (const p of points) m.set(sec(p.time), p.equity);
  const t = [...m.keys()].sort((a, b) => a - b);
  return { time: t, value: t.map((x) => m.get(x)!) };
}

/** กราฟ 4h ของเหรียญที่เลือก + จุดเข้า/ออกของบัญชีเดโม */
function DemoChart({ status, trades }: { status: PaperStatus; trades: PaperTrade[] }) {
  const [symbol, setSymbol] = useState(status.config.symbols[0]);
  const [data, setData] = useState<{ key: string; candles: Candle[]; strategy: Strategy } | null>(null);
  const [nonce, setNonce] = useState(0);
  const tf = status.config.interval;

  useEffect(() => {
    const ac = new AbortController();
    const key = symbol;
    const run = () =>
      Promise.all([
        getJSON<Strategy>(`strategy/${symbol}/${tf}?bars=300`, ac.signal),
        getJSON<Candle[]>(`candles/${symbol}/${tf}?limit=300`, ac.signal),
      ]).then(([strategy, candles]) => setData({ key, candles, strategy }), () => {});
    run();
    const id = setInterval(run, 5 * 60_000);
    return () => {
      ac.abort();
      clearInterval(id);
    };
  }, [symbol, tf, nonce]);
  const { live } = useLiveKline(symbol, tf, () => setTimeout(() => setNonce((n) => n + 1), 5_000));

  const cur = data?.key === symbol ? data : null;
  const pos = status.positions.find((p) => p.symbol === symbol);
  const st = cur?.strategy.status;
  const mine = trades.filter((t) => t.symbol === symbol);
  return (
    <Card
      title="จุดเข้า / ออก ของบัญชีเดโม"
      aside={<Segmented label="เหรียญ" value={symbol} onChange={setSymbol}
        options={status.config.symbols.map((s) => ({ value: s, label: SYMBOL_LABEL[s] ?? s }))} />}
    >
      <p className="mb-2 text-sm text-ink-2">
        {pos
          ? `ถือ ${short(symbol)} อยู่ · เข้า ${fmtPrice(pos.entry_price)} · SL ${fmtPrice(pos.stop)} (เลื่อนขึ้นตามราคาทุกครั้งที่แท่ง ${tf} ปิด)`
          : mine.length === 0
            ? st
              ? st.regime === "up"
                ? `ยังไม่มีไม้ ${short(symbol)} — จะเข้าเมื่อแท่ง ${tf} ปิดเหนือ ${fmtPrice(st.breakout_level)} (เส้นสีเทา)`
                : `ยังไม่มีไม้ ${short(symbol)} — เทรนด์ TF ใหญ่ไม่ใช่ขาขึ้น ระบบไม่เข้า (ถือเงินสด)`
              : "กำลังโหลด…"
            : `${short(symbol)} ปิดไปแล้ว ${mine.length} ไม้ · ตอนนี้ไม่มีไม้`}
      </p>
      <PriceChart candles={cur?.candles ?? []} live={live} strategy={cur?.strategy ?? null} tfSec={TF_SEC[tf]}
        demo={{ symbol, trades, positions: status.positions }} />
    </Card>
  );
}

export default function DemoView() {
  const [status, setStatus] = useState<PaperStatus | null>(null);
  const [events, setEvents] = useState<PaperEvent[]>([]);
  const [equity, setEquity] = useState<PaperEquity[]>([]);
  const [trades, setTrades] = useState<PaperTrade[]>([]);
  const [cmp, setCmp] = useState<PaperCompare | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState<number | null>(null);
  const [reload, setReload] = useState(0);
  const [capital, setCapital] = useState(10_000);
  const [riskPct, setRiskPct] = useState(1);
  const [newTf, setNewTf] = useState("4h");
  const [busy, setBusy] = useState(false);
  const [runs, setRuns] = useState<PaperRunInfo[]>([]);
  const [picked, setPicked] = useState<string | null>(null);

  // บัญชีที่เดินอยู่ (timeframe ละ 1 บัญชี)
  useEffect(() => {
    const ac = new AbortController();
    getJSON<PaperRunInfo[]>("paper/runs", ac.signal).then(setRuns, () => {});
    return () => ac.abort();
  }, [reload]);
  const active = runs.filter((r) => r.active).sort((a, b) => TF_ORDER.indexOf(a.interval) - TF_ORDER.indexOf(b.interval));
  const run = picked && runs.some((r) => r.run === picked) ? picked : (active[0]?.run ?? null);
  const q = run ? `run=${encodeURIComponent(run)}` : "";

  // สถานะ + เหตุการณ์ ทุก 15 วินาที (worker อัปเดตทุก 15 วินาที)
  useEffect(() => {
    const ac = new AbortController();
    const tick = () => {
      setNow(Date.now());
      Promise.all([
        getJSON<PaperStatus>(`paper/status?${q}`, ac.signal),
        getJSON<PaperEvent[]>(`paper/events?limit=60&${q}`, ac.signal),
        getJSON<PaperTrade[]>(`paper/trades?${q}`, ac.signal),
      ]).then(([s, e, t]) => {
        setStatus(s);
        setEvents(e);
        setTrades(t);
        setError(null);
      }, (e: Error) => e.name !== "AbortError" && setError(e.message));
    };
    tick();
    const id = setInterval(tick, 15_000);
    return () => {
      ac.abort();
      clearInterval(id);
    };
  }, [reload, q]);

  // เทียบ backtest + equity ทุก 2 นาที
  useEffect(() => {
    const ac = new AbortController();
    const tick = () =>
      Promise.all([
        getJSON<PaperCompare>(`paper/compare?${q}`, ac.signal),
        getJSON<PaperEquity[]>(`paper/equity?${q}`, ac.signal),
      ]).then(([c, q]) => {
        setCmp(c);
        setEquity(q);
      }, () => {});
    tick();
    const id = setInterval(tick, 120_000);
    return () => {
      ac.abort();
      clearInterval(id);
    };
  }, [reload, q]);

  const lines: EquityLine[] = useMemo(() => {
    const out: EquityLine[] = [];
    if (equity.length) out.push({ key: "paper", label: "บัญชีเดโม", colorVar: "series1", width: 2, ...series(equity) });
    if (cmp?.ready && cmp.equity_bt.length) {
      out.push({ key: "bt", label: "backtest ช่วงเดียวกัน", colorVar: "bench", width: 1, ...series(cmp.equity_bt) });
    }
    return out;
  }, [equity, cmp]);

  async function startNew() {
    const old = active.find((r) => r.interval === newTf);
    const ok = window.confirm(
      `เริ่มบัญชีเดโม ${TF_LABEL[newTf]} ทุน ${fmtNum(capital, 0)} USDT เสี่ยง ${riskPct}% ต่อไม้?\n\n` +
      (old ? `บัญชี ${TF_LABEL[newTf]} เดิม (${old.run}) จะหยุดเดินแต่เก็บข้อมูลไว้ — timeframe อื่นเดินต่อ`
           : "timeframe อื่นที่เดินอยู่จะเดินต่อตามปกติ"),
    );
    if (!ok) return;
    setBusy(true);
    try {
      const res = await getJSON<{ run: string }>("paper/new", undefined, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ capital, risk_pct: riskPct, interval: newTf }),
      });
      setPicked(res.run);
      setReload((n) => n + 1);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  if (error && !status) return <ErrorBox error={error} />;
  if (!status) return <p className="text-sm text-muted">กำลังโหลด…</p>;

  const cap = status.config.capital;
  const alive = status.heartbeat_ms != null && now != null && now - status.heartbeat_ms < 90_000;
  const pnlPct = (status.equity / cap - 1) * 100;
  const tf = status.config.interval ?? "4h";
  const nextClose = status.started_bar
    ? Math.max(...Object.values(status.last_bar)) + 2 * TF_SEC[tf] * 1000
    : null;

  return (
    <div className="space-y-4">
      {active.length > 0 && (
        <div className="flex flex-wrap items-center gap-3">
          <Segmented label="บัญชีเดโมตาม timeframe" value={run ?? ""} onChange={setPicked}
            options={active.map((r) => ({ value: r.run, label: `${TF_LABEL[r.interval] ?? r.interval} · ${fmtPct(r.return_pct, 1, true)}` }))} />
          {!status.active && <Badge tone="neutral">บัญชีนี้หยุดเดินแล้ว (ดูย้อนหลัง)</Badge>}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <Badge tone={alive ? "good" : "bad"}>{alive ? "worker ทำงานอยู่" : "worker หยุด"}</Badge>
        <span className="text-ink-2">
          รอบ {status.run} · เริ่ม {fmtTime(sec(status.config.created_ms))} · ระบบ trend {TF_LABEL[tf] ?? tf} · BTC + ทอง ·
          เสี่ยง {status.config.risk_pct}%/ไม้
        </span>
        {nextClose && <span className="text-muted">แท่งถัดไปปิด {fmtTime(sec(nextClose))}</span>}
        {!alive && <span className="text-xs text-bad">เปิดด้วย: docker compose up -d paper</span>}
      </div>
      {error && <ErrorBox error={error} />}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="มูลค่าบัญชีเดโม (USDT)" value={fmtNum(status.equity, 2)} sub={`ทุนเริ่มต้น ${fmtNum(cap, 0)}`} />
        <Stat label="กำไร / ขาดทุน" value={fmtPct(pnlPct, 2, true)} tone={pnlPct > 0 ? "good" : pnlPct < 0 ? "bad" : undefined}
          sub={`${fmtNum(status.equity - cap, 2)} USDT`} />
        <Stat label="ไม้ที่ถือ" value={String(status.positions.length)}
          sub={Object.keys(status.pending).length ? "มีคำสั่งรอเข้า/ออก" : "ไม่มีคำสั่งค้าง"} />
        <Stat label="ไม้ที่ปิดแล้ว" value={String(status.closed.trades)}
          sub={status.closed.trades ? `ชนะ ${status.closed.wins} · รวม ${fmtR(status.closed.total_r)}` : "ยังไม่มี"} />
      </div>

      <DemoChart key={status.run} status={status} trades={trades} />

      <Card title="ไม้ที่ถืออยู่" aside="ราคาอัปเดตทุก 15 วินาทีจาก worker">
        {status.positions.length === 0 ? (
          <p className="text-sm text-ink-2">
            ยังไม่มีไม้ — ระบบรอสัญญาณ (ปิดเหนือ High 20 แท่ง ขณะเทรนด์ TF ใหญ่ขาขึ้น) ตรวจทุกครั้งที่แท่ง 4h ปิด
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="tnum w-full text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs text-ink-2">
                  <th className="py-1.5 pr-3 font-medium">เหรียญ</th>
                  <th className="py-1.5 pr-3 font-medium">เข้าเมื่อ</th>
                  <th className="py-1.5 pr-3 text-right font-medium">ราคาเข้า</th>
                  <th className="py-1.5 pr-3 text-right font-medium">จำนวน</th>
                  <th className="py-1.5 pr-3 text-right font-medium">ราคาตอนนี้</th>
                  <th className="py-1.5 pr-3 text-right font-medium">SL</th>
                  <th className="py-1.5 text-right font-medium">กำไร</th>
                </tr>
              </thead>
              <tbody>
                {status.positions.map((p) => (
                  <tr key={p.symbol} className="border-b border-line last:border-0">
                    <td className="py-1.5 pr-3 text-ink">{SYMBOL_LABEL[p.symbol] ?? p.symbol}</td>
                    <td className="py-1.5 pr-3 text-ink">{fmtTime(sec(p.entry_time))}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtPrice(p.entry_price)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtNum(p.qty, 5)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtPrice(p.price)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtPrice(p.stop)}</td>
                    <td className="py-1.5 text-right">
                      <Badge tone={(p.pnl ?? 0) >= 0 ? "good" : "bad"}>{fmtNum(p.pnl, 2)} · {fmtR(p.r_now)}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <Card
        title="เดโม เทียบ backtest ช่วงเวลาเดียวกัน"
        aside={cmp?.ready ? `${fmtTime(sec(cmp.from))} – ${fmtTime(sec(cmp.to))}` : undefined}
      >
        {!cmp || !cmp.ready ? (
          <p className="text-sm text-ink-2">{cmp && !cmp.ready ? cmp.message : "กำลังคำนวณ…"}</p>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
              <div><div className="text-xs text-ink-2">เดโม</div><div className="tnum text-lg font-semibold">{fmtPct(cmp.summary.paper_return_pct, 2, true)}</div></div>
              <div><div className="text-xs text-ink-2">backtest</div><div className="tnum text-lg font-semibold">{fmtPct(cmp.summary.bt_return_pct, 2, true)}</div></div>
              <div><div className="text-xs text-ink-2">ไม้ที่ตรงกัน</div><div className="tnum text-lg font-semibold">{cmp.summary.matched} / {Math.max(cmp.summary.trades_paper, cmp.summary.trades_bt)}</div></div>
              <div><div className="text-xs text-ink-2">ราคาเข้าต่างเฉลี่ย</div><div className="tnum text-lg font-semibold">{fmtPct(cmp.summary.avg_entry_diff_pct, 3, true)}</div></div>
            </div>
            {lines.length > 0 && <EquityChart lines={lines} log={false} />}
            {cmp.trades.length > 0 && (
              <div className="overflow-x-auto">
                <table className="tnum w-full text-sm">
                  <thead>
                    <tr className="border-b border-line text-left text-xs text-ink-2">
                      <th className="py-1.5 pr-3 font-medium">เหรียญ</th>
                      <th className="py-1.5 pr-3 font-medium">แท่งที่เข้า</th>
                      <th className="py-1.5 pr-3 font-medium">ผลเทียบ</th>
                      <th className="py-1.5 pr-3 text-right font-medium">ราคาเข้า เดโม / backtest</th>
                      <th className="py-1.5 pr-3 text-right font-medium">ต่าง</th>
                      <th className="py-1.5 text-right font-medium">R เดโม / backtest</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cmp.trades.map((r) => (
                      <tr key={`${r.symbol}-${r.entry_bar}`} className="border-b border-line last:border-0">
                        <td className="py-1.5 pr-3 text-ink">{short(r.symbol)}</td>
                        <td className="py-1.5 pr-3 text-ink">{fmtTime(sec(r.entry_bar))}</td>
                        <td className="py-1.5 pr-3"><Badge tone={r.status === "ตรงกัน" ? "good" : "bad"}>{r.status}</Badge></td>
                        <td className="py-1.5 pr-3 text-right text-ink">{fmtPrice(r.paper_entry)} / {fmtPrice(r.bt_entry)}</td>
                        <td className="py-1.5 pr-3 text-right text-ink">{fmtPct(r.entry_diff_pct, 3, true)}</td>
                        <td className="py-1.5 text-right text-ink">{fmtR(r.paper_r)} / {fmtR(r.bt_r)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2" title="บันทึกเหตุการณ์" aside="ล่าสุดอยู่บน">
          <ul className="tnum max-h-96 space-y-1.5 overflow-y-auto text-sm">
            {events.map((e, i) => (
              <li key={`${e.time}-${i}`} className="flex flex-wrap gap-x-3 border-b border-line pb-1.5 last:border-0">
                <span className="w-36 shrink-0 text-ink-2">{fmtTime(sec(e.time))}</span>
                <span className={`w-36 shrink-0 font-medium ${e.type === "error" ? "text-bad" : "text-ink"}`}>
                  {EVENT[e.type] ?? e.type}{e.symbol ? ` · ${short(e.symbol)}` : ""}
                </span>
                <span className="text-ink-2">{eventDetail(e)}</span>
              </li>
            ))}
          </ul>
        </Card>

        <Card title="เริ่มบัญชีเดโมใหม่">
          <div className="space-y-3 text-sm">
            <div>
              <span className="text-ink-2">Timeframe</span>
              <div className="mt-1">
                <Segmented label="Timeframe ของบัญชีใหม่" value={newTf} onChange={setNewTf}
                  options={TF_ORDER.map((t) => ({ value: t, label: TF_LABEL[t] }))} />
              </div>
            </div>
            <label className="block">
              <span className="text-ink-2">ทุนจำลอง (USDT)</span>
              <input type="number" min={100} step={100} value={capital}
                onChange={(e) => setCapital(Math.max(100, +e.target.value))}
                className="tnum mt-1 w-full rounded-md border border-line bg-surface-2 px-3 py-1.5 text-ink" />
            </label>
            <label className="block">
              <span className="text-ink-2">เสี่ยงต่อไม้ (%)</span>
              <input type="number" min={0.1} max={5} step={0.1} value={riskPct}
                onChange={(e) => setRiskPct(Math.min(5, Math.max(0.1, +e.target.value)))}
                className="tnum mt-1 w-full rounded-md border border-line bg-surface-2 px-3 py-1.5 text-ink" />
            </label>
            <button type="button" onClick={startNew} disabled={busy}
              className="w-full rounded-md border border-line bg-surface-2 px-3 py-2 font-medium text-ink hover:bg-page disabled:opacity-50">
              {busy ? "กำลังเริ่ม…" : "เริ่มรอบใหม่"}
            </button>
            <p className="text-xs text-muted">
              เดินพร้อมกันได้ timeframe ละ 1 บัญชี — เริ่มใหม่ใน timeframe เดิมจะแทนบัญชีเดิม (เก็บข้อมูลเก่าไว้)
            </p>
            <p className="text-xs text-muted">
              เงินและคำสั่งทั้งหมดเป็นของจำลองในระบบเรา ไม่ได้เชื่อมบัญชีใด ๆ — ใช้ราคาจริงจาก Binance (อ่านอย่างเดียว)
              เพื่อเก็บข้อมูลไปเทียบกับ backtest
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}
