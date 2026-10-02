"use client";

import { useEffect, useMemo, useState } from "react";
import { Badge, Card, ErrorBox, Segmented } from "@/components/ui";
import { getJSON, type PaperRunInfo, type Research } from "@/lib/api";
import { fmtNum, fmtPct, fmtR, fmtTime, TF_LABEL } from "@/lib/format";

type RuleRow = {
  symbol: string; interval: string; rule: string; trades: number; sqn: number;
  beats_random_pct: number; avg_r_is: number | null; avg_r_oos: number | null;
};
type RangeRow = {
  symbol: string; interval: string; entry: string; regime: string; trades: number; sqn: number;
  beats_random_pct: number; avg_r_is: number | null; avg_r_oos: number | null;
};
type WFRow = {
  symbol: string; interval: string; oos_from: string; oos_to: string; combos: number;
  wf_trades: number; wf_total_r: number; wf_sqn: number; fixed_trades: number; fixed_total_r: number;
  fixed_sqn: number; wf_beats_random_pct: number;
};

type TFRow = {
  interval: string; scope: string; period: string; trades: number; trades_per_month: number; win_rate: number;
  avg_r: number; sqn: number; return_pct: number; cagr_pct: number; max_dd_pct: number; sharpe: number;
  bh_5050_pct: number; bh_5050_dd_pct: number; bh_self_pct?: number | null;
};

const TABS = [
  { value: "timeframes", label: "แยกตาม timeframe" },
  { value: "rules", label: "กฎจากบทความ" },
  { value: "walkforward", label: "Walk-forward" },
  { value: "range", label: "ระบบไซด์เวย์" },
] as const;
type Tab = (typeof TABS)[number]["value"];

const DS_ORDER = ["BTC 1h", "BTC 4h", "BTC 1d", "PAXG 1h", "PAXG 4h", "PAXG 1d"];
const ds = (r: { symbol: string; interval: string }) => `${r.symbol.replace("USDT", "")} ${r.interval}`;

/** ผ่าน = ชนะการเข้าแบบสุ่ม ≥ 95% และ SQN ≥ 1.5 · แย่ = SQN ติดลบ หรือแพ้การสุ่มเกินครึ่ง */
function tone(r: { trades: number; sqn: number; beats_random_pct: number }): "good" | "bad" | "neutral" {
  if (r.trades < 10) return "neutral";
  if (r.beats_random_pct >= 95 && r.sqn >= 1.5) return "good";
  if (r.sqn < 0 || r.beats_random_pct < 50) return "bad";
  return "neutral";
}

function Cell({ r }: { r?: { trades: number; sqn: number; beats_random_pct: number } }) {
  if (!r) return <td className="px-2 py-1.5 text-center text-muted">–</td>;
  if (r.trades < 10) return <td className="px-2 py-1.5 text-center text-xs text-muted">ไม้น้อย ({r.trades})</td>;
  return (
    <td className="px-2 py-1.5 text-center">
      <Badge tone={tone(r)}>{fmtNum(r.sqn, 1)}</Badge>
      <div className="text-[11px] text-muted">สุ่ม {Math.round(r.beats_random_pct)}% · {r.trades}</div>
    </td>
  );
}

function Pivot<T extends { symbol: string; interval: string; trades: number; sqn: number; beats_random_pct: number }>(
  { rows, rowKey, rowLabel }: { rows: T[]; rowKey: (r: T) => string; rowLabel: (k: string) => string },
) {
  const keys = useMemo(() => [...new Set(rows.map(rowKey))], [rows, rowKey]);
  const by = useMemo(() => new Map(rows.map((r) => [`${rowKey(r)}|${ds(r)}`, r])), [rows, rowKey]);
  return (
    <div className="overflow-x-auto">
      <table className="tnum w-full text-sm">
        <thead>
          <tr className="border-b border-line text-xs text-ink-2">
            <th className="py-1.5 pr-3 text-left font-medium">กฎ</th>
            {DS_ORDER.map((d) => <th key={d} className="px-2 py-1.5 font-medium">{d}</th>)}
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k} className="border-b border-line last:border-0">
              <td className="py-1.5 pr-3 text-ink">{rowLabel(k)}</td>
              {DS_ORDER.map((d) => <Cell key={d} r={by.get(`${k}|${d}`)} />)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const SCOPE: Record<string, string> = { BTCUSDT: "BTC", PAXGUSDT: "ทอง", portfolio: "พอร์ต BTC + ทอง" };
const TF_ORDER = ["15m", "1h", "4h", "1d"];
const PERIODS = [
  { value: "all", label: "ทั้งช่วง" },
  { value: "in_sample", label: "ก่อน 2023" },
  { value: "out_sample", label: "ตั้งแต่ 2023" },
] as const;

function Timeframes({ rows, run }: { rows: TFRow[]; run: string }) {
  const [period, setPeriod] = useState<(typeof PERIODS)[number]["value"]>("all");
  const [demo, setDemo] = useState<PaperRunInfo[] | null>(null);
  useEffect(() => {
    const ac = new AbortController();
    getJSON<PaperRunInfo[]>("paper/summary", ac.signal).then(setDemo, () => setDemo([]));
    return () => ac.abort();
  }, []);
  const part = rows.filter((r) => r.period === period)
    .sort((a, b) => TF_ORDER.indexOf(a.interval) - TF_ORDER.indexOf(b.interval)
      || ["BTCUSDT", "PAXGUSDT", "portfolio"].indexOf(a.scope) - ["BTCUSDT", "PAXGUSDT", "portfolio"].indexOf(b.scope));
  const ports = part.filter((r) => r.scope === "portfolio");
  const best = ports.reduce<TFRow | null>((m, r) => (r.trades >= 10 && (!m || r.sqn > m.sqn) ? r : m), null);

  return (
    <div className="space-y-4">
      <Card title="Backtest ระบบเดียวกันบนแต่ละ timeframe" aside={`ผลจาก ${run} · ช่วงเวลาเดียวกันทุก TF · เสี่ยง 1%/ไม้ · ค่าธรรมเนียม 0.1%/ขา`}>
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <Segmented label="ช่วงเวลา" value={period} options={[...PERIODS]} onChange={setPeriod} />
          {best && (
            <span className="text-sm text-ink-2">
              <Badge tone="good">ดีที่สุด {TF_LABEL[best.interval]}</Badge> SQN {fmtNum(best.sqn, 2)} · ผลตอบแทน {fmtPct(best.return_pct, 1, true)} · DD {fmtPct(best.max_dd_pct, 1)}
            </span>
          )}
        </div>
        <div className="overflow-x-auto">
          <table className="tnum w-full text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs text-ink-2">
                <th className="py-1.5 pr-3 font-medium">TF</th>
                <th className="py-1.5 pr-3 font-medium">ชุด</th>
                <th className="py-1.5 pr-3 text-right font-medium">ไม้/เดือน</th>
                <th className="py-1.5 pr-3 text-right font-medium">ชนะ</th>
                <th className="py-1.5 pr-3 text-right font-medium">R เฉลี่ย</th>
                <th className="py-1.5 pr-3 text-right font-medium">SQN</th>
                <th className="py-1.5 pr-3 text-right font-medium">ผลตอบแทน</th>
                <th className="py-1.5 pr-3 text-right font-medium">CAGR</th>
                <th className="py-1.5 pr-3 text-right font-medium">Max DD</th>
                <th className="py-1.5 text-right font-medium">Sharpe</th>
              </tr>
            </thead>
            <tbody>
              {part.map((r) => {
                const port = r.scope === "portfolio";
                return (
                  <tr key={`${r.interval}-${r.scope}`}
                    className={`border-b border-line last:border-0 ${port ? "font-semibold" : ""} ${port && r.interval !== "1d" ? "border-b-2" : ""}`}>
                    <td className="py-1.5 pr-3 text-ink">{TF_LABEL[r.interval] ?? r.interval}</td>
                    <td className="py-1.5 pr-3 text-ink">{SCOPE[r.scope] ?? r.scope}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtNum(r.trades_per_month, 1)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtPct(r.win_rate, 0)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtR(r.avg_r)}</td>
                    <td className="py-1.5 pr-3 text-right">
                      {r.trades < 10 ? <span className="text-xs text-muted">ไม้น้อย ({r.trades})</span>
                        : <Badge tone={r.sqn >= 2 ? "good" : r.sqn < 0 ? "bad" : "neutral"}>{fmtNum(r.sqn, 2)}</Badge>}
                    </td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtPct(r.return_pct, 1, true)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtPct(r.cagr_pct, 1)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtPct(r.max_dd_pct, 1)}</td>
                    <td className="py-1.5 text-right text-ink">{fmtNum(r.sharpe, 2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-muted">
          ค่าพารามิเตอร์เป็นจำนวนแท่งเท่ากันทุก TF (breakout 20 แท่ง, ATR 14 แท่ง) · TF เล็กเทรดถี่ ค่าธรรมเนียมกินกำไรมาก ·
          ซื้อแล้วถือ 50/50 ช่วงเดียวกัน: {ports[0] ? `${fmtPct(ports[0].bh_5050_pct, 1, true)} (DD ${fmtPct(ports[0].bh_5050_dd_pct, 1)})` : "–"}
        </p>
      </Card>

      <Card title="บัญชีเดโมแยกตาม timeframe" aside="เทียบกับ backtest ช่วงเวลาเดียวกับที่บัญชีเดิน">
        {demo == null ? <p className="text-sm text-muted">กำลังโหลด…</p> : demo.length === 0 ? (
          <p className="text-sm text-ink-2">ยังไม่มีบัญชีเดโม — เริ่มได้ที่หน้า &ldquo;บัญชีเดโม&rdquo;</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="tnum w-full text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs text-ink-2">
                  <th className="py-1.5 pr-3 font-medium">TF</th>
                  <th className="py-1.5 pr-3 font-medium">เริ่ม</th>
                  <th className="py-1.5 pr-3 text-right font-medium">ทุน</th>
                  <th className="py-1.5 pr-3 text-right font-medium">มูลค่าตอนนี้</th>
                  <th className="py-1.5 pr-3 text-right font-medium">เดโม</th>
                  <th className="py-1.5 pr-3 text-right font-medium">backtest ช่วงเดียวกัน</th>
                  <th className="py-1.5 pr-3 text-right font-medium">ไม้ (ชนะ)</th>
                  <th className="py-1.5 pr-3 text-right font-medium">R รวม</th>
                  <th className="py-1.5 text-right font-medium">ไม้ตรงกับ backtest</th>
                </tr>
              </thead>
              <tbody>
                {demo.map((d) => (
                  <tr key={d.run} className="border-b border-line last:border-0">
                    <td className="py-1.5 pr-3 font-semibold text-ink">{TF_LABEL[d.interval] ?? d.interval}</td>
                    <td className="py-1.5 pr-3 text-ink-2">{fmtTime(Math.floor(d.created_ms / 1000))}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtNum(d.capital, 0)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtNum(d.equity, 2)}</td>
                    <td className="py-1.5 pr-3 text-right"><Badge tone={d.return_pct > 0 ? "good" : d.return_pct < 0 ? "bad" : "neutral"}>{fmtPct(d.return_pct, 2, true)}</Badge></td>
                    <td className="py-1.5 pr-3 text-right text-ink">{d.compare ? fmtPct(d.compare.bt_return_pct, 2, true) : "รอแท่งแรก"}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{d.trades} ({d.wins})</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtR(d.total_r)}</td>
                    <td className="py-1.5 text-right text-ink">{d.compare ? `${d.compare.matched} / ${Math.max(d.compare.trades_paper, d.compare.trades_bt)}` : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

export default function ResearchView() {
  const [tab, setTab] = useState<Tab>("timeframes");
  const [data, setData] = useState<Record<string, Research<unknown>>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (data[tab]) return;
    const ac = new AbortController();
    getJSON<Research<unknown>>(`research/${tab}`, ac.signal)
      .then((d) => setData((x) => ({ ...x, [tab]: d })))
      .catch((e) => e.name !== "AbortError" && setError(e.message));
    return () => ac.abort();
  }, [tab, data]);

  const cur = data[tab];
  return (
    <div className="space-y-4">
      <Segmented label="ชุดผลทดสอบ" value={tab} options={[...TABS]} onChange={(v) => { setError(null); setTab(v); }} />
      <p className="text-xs text-ink-2">
        ตัวเลขในช่อง = SQN (เกิน 2 = น่าเชื่อ) · บรรทัดล่าง = ชนะการเข้าแบบสุ่มกี่ % · จำนวนไม้ ·{" "}
        <Badge tone="good">ผ่าน</Badge> ชนะสุ่ม ≥ 95% และ SQN ≥ 1.5 · <Badge tone="bad">แย่</Badge> SQN ติดลบ หรือแพ้สุ่ม
      </p>
      {error && <ErrorBox error={error} />}
      {!cur && !error && <p className="text-sm text-muted">กำลังโหลด…</p>}

      {cur && tab === "timeframes" && <Timeframes rows={cur.rows as TFRow[]} run={cur.run} />}

      {cur && tab === "rules" && (
        <Card title="ทดสอบกฎทีละข้อ" aside={`ผลจาก ${cur.run} · เปลี่ยนทีละอย่างจาก breakout 20 + เทรนด์ TF ใหญ่ + trailing 3 ATR`}>
          <Pivot rows={cur.rows as RuleRow[]} rowKey={(r) => r.rule} rowLabel={(k) => k} />
          <p className="mt-3 text-xs text-muted">
            ทดสอบ 32 กฎ × 6 ชุด = 192 ครั้ง ผ่านโดยบังเอิญได้ราว 10 ครั้ง — เชื่อเฉพาะผลที่ไปทางเดียวกันหลายชุดข้อมูล
          </p>
        </Card>
      )}

      {cur && tab === "range" && (
        <Card title="ระบบตลาดไซด์เวย์ (ซื้อสวน รอราคากลับเส้นกลาง)" aside={`ผลจาก ${cur.run}`}>
          <Pivot rows={cur.rows as RangeRow[]} rowKey={(r) => `${r.entry} · ${r.regime}`} rowLabel={(k) => k} />
          <p className="mt-3 text-xs text-muted">ขาดทุนทุกแบบทุกชุด — ตลาดเหล่านี้มีแรงส่ง (momentum) การซื้อสวนจึงแพ้</p>
        </Card>
      )}

      {cur && tab === "walkforward" && (
        <Card title="Walk-forward : ให้ระบบเลือกพารามิเตอร์เอง vs ค่าตายตัว" aside={`ผลจาก ${cur.run}`}>
          <div className="overflow-x-auto">
            <table className="tnum w-full text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs text-ink-2">
                  <th className="py-1.5 pr-3 font-medium">ชุดข้อมูล</th>
                  <th className="py-1.5 pr-3 font-medium">ช่วงทดสอบ</th>
                  <th className="py-1.5 pr-3 text-right font-medium">ให้เลือกเอง (R · SQN)</th>
                  <th className="py-1.5 pr-3 text-right font-medium">ค่าตายตัว (R · SQN)</th>
                  <th className="py-1.5 text-right font-medium">ตัวไหนดีกว่า</th>
                </tr>
              </thead>
              <tbody>
                {(cur.rows as WFRow[]).map((r) => (
                  <tr key={ds(r)} className="border-b border-line last:border-0">
                    <td className="py-1.5 pr-3 text-ink">{ds(r)}</td>
                    <td className="py-1.5 pr-3 text-ink-2">{r.oos_from} – {r.oos_to}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtNum(r.wf_total_r, 1)}R · {fmtNum(r.wf_sqn, 2)}</td>
                    <td className="py-1.5 pr-3 text-right text-ink">{fmtNum(r.fixed_total_r, 1)}R · {fmtNum(r.fixed_sqn, 2)}</td>
                    <td className="py-1.5 text-right">
                      {r.fixed_total_r >= r.wf_total_r ? <Badge tone="good">ค่าตายตัว</Badge> : <Badge tone="neutral">ให้เลือกเอง</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted">
            ยิ่งให้ระบบเลือกจากหลายชุด ยิ่งได้ชุดที่บังเอิญเข้ากับอดีต (overfit) — จึงใช้ค่าตายตัวในระบบจริง
          </p>
        </Card>
      )}
    </div>
  );
}
