"use client";

import { useEffect, useMemo, useState } from "react";
import EquityChart, { type EquityLine } from "@/components/EquityChart";
import { Badge, Card, ErrorBox, Segmented, Stat } from "@/components/ui";
import { getJSON, type Portfolio, type PortfolioMetric } from "@/lib/api";
import { fmtDate, fmtNum, fmtPct } from "@/lib/format";

const NAMES: Record<string, string> = {
  "รวม trend": "BTC + ทอง (trend)",
  "BTC-trend": "BTC trend อย่างเดียว",
  "PAXG-trend": "ทอง trend อย่างเดียว",
  "BTC-range": "BTC ไซด์เวย์",
  "PAXG-range": "ทอง ไซด์เวย์",
  "รวม range": "BTC + ทอง (ไซด์เวย์)",
  "รวมทั้งหมด": "รวมทุกระบบ",
  "BTC trend+range": "BTC trend + ไซด์เวย์",
  "PAXG trend+range": "ทอง trend + ไซด์เวย์",
};
const PERIODS = [
  { value: "all", label: "ทั้งช่วง" },
  { value: "in_sample", label: "ก่อน 2023" },
  { value: "out_sample", label: "ตั้งแต่ 2023" },
] as const;
type Period = (typeof PERIODS)[number]["value"];

export default function PortfolioView() {
  const [data, setData] = useState<Portfolio | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [period, setPeriod] = useState<Period>("all");
  const [log, setLog] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    getJSON<Portfolio>("portfolio/latest", ac.signal)
      .then(setData)
      .catch((e) => e.name !== "AbortError" && setError(e.message));
    return () => ac.abort();
  }, []);

  const lines: EquityLine[] = useMemo(() => {
    if (!data) return [];
    const pick = (key: string, label: string, colorVar: EquityLine["colorVar"], width: EquityLine["width"]) =>
      data.series[key] ? [{ key, label, colorVar, width, ...data.series[key] }] : [];
    return [
      ...pick("รวม trend", "BTC + ทอง (trend)", "series1", 3),
      ...pick("BTC-trend", "BTC trend", "series2", 2),
      ...pick("PAXG-trend", "ทอง trend", "series3", 2),
      ...pick("bh_5050", "ซื้อแล้วถือ 50/50", "bench", 1),
    ];
  }, [data]);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <p className="text-sm text-muted">กำลังโหลด…</p>;

  const rows = data.metrics.filter((m) => m.period === period);
  const main = rows.find((m) => m.portfolio === "รวม trend");
  const first = Object.values(data.series)[0];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Segmented label="ช่วงเวลา" value={period} options={[...PERIODS]} onChange={setPeriod} />
        <span className="text-xs text-muted">
          เสี่ยง {data.config.risk_pct ?? 1}% ต่อไม้ · ถือรวมไม่เกิน {data.config.max_lev ?? 1} เท่าของพอร์ต (spot) · {data.config.interval ?? "4h"}
        </span>
      </div>

      {main && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat label="ผลตอบแทน (BTC + ทอง)" value={fmtPct(main.return_pct, 1, true)}
            sub={`ซื้อแล้วถือ 50/50: ${fmtPct(main.bh_5050_pct, 1, true)}`} tone={main.return_pct >= 0 ? "good" : "bad"} />
          <Stat label="CAGR (ต่อปี)" value={fmtPct(main.cagr_pct, 1)} sub={`Sharpe ${fmtNum(main.sharpe, 2)}`} />
          <Stat label="Max drawdown" value={fmtPct(main.max_dd_pct, 1)}
            sub={`ซื้อแล้วถือ 50/50: ${fmtPct(main.bh_5050_dd_pct, 1)}`} />
          <Stat label="SQN (ความน่าเชื่อถือ)" value={fmtNum(main.sqn, 2)}
            sub={`${main.trades} ไม้ · เฉลี่ย ${fmtNum(main.avg_r, 2)}R ต่อไม้ · เกิน 2 = น่าเชื่อ`} />
        </div>
      )}

      <Card
        title="มูลค่าพอร์ต (เริ่มที่ 100)"
        aside={
          <span className="inline-flex items-center gap-3">
            {first && <span>{fmtDate(first.time[0])} – {fmtDate(first.time[first.time.length - 1])}</span>}
            <label className="inline-flex items-center gap-1.5 text-ink-2">
              <input type="checkbox" checked={log} onChange={(e) => setLog(e.target.checked)} />
              สเกล log
            </label>
          </span>
        }
      >
        <EquityChart lines={lines} log={log} />
      </Card>

      <Card title="ทุกชุดที่ทดสอบ" aside={`ผลจาก ${data.run}`}>
        <MetricsTable rows={rows} />
        <p className="mt-3 text-xs text-muted">
          ระบบไซด์เวย์ (mean reversion) ขาดทุนทุกชุด จึงไม่ใช้ในพอร์ตจริง — แสดงไว้เพื่อเทียบ
        </p>
      </Card>
    </div>
  );
}

function MetricsTable({ rows }: { rows: PortfolioMetric[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="tnum w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-ink-2">
            <th className="py-1.5 pr-3 font-medium">ชุด</th>
            <th className="py-1.5 pr-3 text-right font-medium">ไม้</th>
            <th className="py-1.5 pr-3 text-right font-medium">R เฉลี่ย</th>
            <th className="py-1.5 pr-3 text-right font-medium">SQN</th>
            <th className="py-1.5 pr-3 text-right font-medium">ผลตอบแทน</th>
            <th className="py-1.5 pr-3 text-right font-medium">CAGR</th>
            <th className="py-1.5 pr-3 text-right font-medium">Max DD</th>
            <th className="py-1.5 pr-3 text-right font-medium">Sharpe</th>
            <th className="py-1.5 text-right font-medium">ถือเฉลี่ย</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((m) => (
            <tr key={m.portfolio} className={`border-b border-line last:border-0 ${m.portfolio === "รวม trend" ? "font-semibold" : ""}`}>
              <td className="py-1.5 pr-3 text-ink">{NAMES[m.portfolio] ?? m.portfolio}</td>
              <td className="py-1.5 pr-3 text-right text-ink">{m.trades}</td>
              <td className="py-1.5 pr-3 text-right text-ink">{fmtNum(m.avg_r, 2)}</td>
              <td className="py-1.5 pr-3 text-right">
                <Badge tone={m.sqn >= 2 ? "good" : m.sqn < 0 ? "bad" : "neutral"}>{fmtNum(m.sqn, 2)}</Badge>
              </td>
              <td className="py-1.5 pr-3 text-right text-ink">{fmtPct(m.return_pct, 1, true)}</td>
              <td className="py-1.5 pr-3 text-right text-ink">{fmtPct(m.cagr_pct, 1)}</td>
              <td className="py-1.5 pr-3 text-right text-ink">{fmtPct(m.max_dd_pct, 1)}</td>
              <td className="py-1.5 pr-3 text-right text-ink">{fmtNum(m.sharpe, 2)}</td>
              <td className="py-1.5 text-right text-ink-2">{fmtPct(m.avg_exposure_pct, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
