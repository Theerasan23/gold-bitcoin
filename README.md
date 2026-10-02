# gold-bitcoin

- `gold_bitcoin.pine` — กลยุทธ์ใน TradingView
- `docs/masterthecrypto-notes.md` — กฎจากบทความ + ผลทดสอบ
- `engine/` — Python: ดึงข้อมูล Binance → Parquet, พอร์ตตรรกะจาก Pine, backtest (Numba), API (FastAPI)
- `web/` — Next.js 16 + Lightweight Charts: สถานะระบบแบบเรียลไทม์, พอร์ต, ผลทดสอบ
- `data/` — ไฟล์ข้อมูล (สร้างอัตโนมัติ)
  - `candles/<SYMBOL>/<interval>.parquet`
  - `results/<run_id>/` — `metrics.parquet`, `trades.parquet`, `equity.parquet`
  - `results/ablation-* · rules-* · walkforward-* · range-* · portfolio-*` — ผลทดสอบแต่ละแบบ (CSV)

ทองใช้ `PAXGUSDT` (โทเคนที่มีทองคำจริงหนุน ราคาตาม XAU) เพราะ Binance ให้ข้อมูลฟรี
ต่างจาก XAUUSD ตรงที่เทรด 24/7 ไม่มี gap วันหยุด

## ใช้งานด้วย Docker

```bash
docker compose build        # ถ้าโหลดแพ็กเกจช้า ใส่ PIP_INDEX_URL=<mirror> ใน .env
docker compose run --rm engine fetch                                   # BTCUSDT + PAXGUSDT · 1h 4h 1d
docker compose run --rm engine backtest --symbol BTCUSDT --interval 4h --trades 10
docker compose run --rm engine backtest --symbol BTCUSDT --interval 4h --set min_score=70 tp_mode=fix --save
docker compose run --rm engine ablate                                  # ตัดทีละส่วนเทียบกัน (กลยุทธ์ Pine)
docker compose run --rm engine core --symbol PAXGUSDT --interval 4h --set exit=chandelier trail_atr=5
docker compose run --rm engine rules                                   # ทดสอบกฎจากบทความทีละข้อ เทียบการเข้าแบบสุ่ม
docker compose run --rm engine walkforward                             # เลือกพารามิเตอร์จากอดีต -> ทดสอบอนาคต
docker compose run --rm engine range                                   # ระบบตลาดไซด์เวย์ (mean reversion) เทียบการสุ่ม
docker compose run --rm engine portfolio --risk-pct 1                  # พอร์ตรวม BTC + ทอง ใช้เงินก้อนเดียว
docker compose up -d                                                   # api :8000 + บัญชีเดโม + เว็บ http://localhost:3100
docker compose run --rm engine paper status                            # สถานะบัญชีเดโม
docker compose run --rm engine paper compare                           # เทียบเดโมกับ backtest ช่วงเดียวกัน
docker compose run --rm engine paper new --capital 20000               # เริ่มบัญชีเดโมใหม่ (รอบเก่าเก็บไว้)
docker compose run --rm --entrypoint pytest engine -q tests
```

ชื่อพารามิเตอร์สำหรับ `--set` อยู่ใน `engine/goldbtc/config.py` (ค่าเริ่มต้นตรงกับ Pine)

## บัญชีเดโม (paper trading)

เงินและคำสั่งเป็นของจำลองทั้งหมด ไม่เชื่อมบัญชี exchange ไม่มี API key ไม่ส่งแจ้งเตือนออกไปไหน
ราคาใช้ของจริงจาก Binance แบบอ่านอย่างเดียว (ข้อมูลตลาดสาธารณะ) เพราะต้องเอาไปเทียบกับ backtest

- **worker (service `paper`):**
  - เช็คทุก 15 วินาที
  - แท่ง 4h ปิดเมื่อไรก็ตัดสินใจด้วยกฎเดียวกับ backtest พอร์ต
  - เข้าไม้ที่ราคาจริงตอนนั้น
  - เช็ค SL ด้วยแท่ง 1 นาที
- **กฎตรงกับ backtest:** ตัวที่ใช้ตัดสินใจเป็นตัวเดียวกับ backtest มีเทสต์ยืนยันว่าป้อนข้อมูลเดียวกันแล้วได้ไม้และ equity ตรงกับ backtest ทุกแท่ง
  - ความต่างที่เห็นตอนเทียบจึงมาจาก "ราคาที่ได้จริง" อย่างเดียว เช่น ช้าไปกี่วินาทีหลังแท่งปิด หรือ SL ทะลุระหว่างนาที
- **worker เคยหยุด:** พอเปิดใหม่จะไล่ประมวลผลแท่งที่พลาดไปให้เอง
- **เก็บข้อมูลที่** `data/paper/runs/<run_id>/`:
  - `trades.jsonl`: ทุกไม้ ทั้งราคาที่ได้และราคาที่ backtest สมมติ
  - `events.jsonl`: ทุกเหตุการณ์ เช่น สัญญาณ, เข้า, เลื่อน SL, ออก, error
  - `equity.jsonl`: มูลค่าบัญชีทุกแท่งปิด และทุก 15 นาที
  - `state.json`: สถานะล่าสุด
- **เริ่มรอบใหม่:** ทำได้จากหน้าเว็บ `/demo` หรือ `paper new` รอบเก่าเก็บไว้ไม่ลบ

## หน้าเว็บ (`web/`)

| หน้า | แสดงอะไร |
|---|---|
| `/` สถานะระบบ | ราคาสด (WebSocket Binance), ทิศเทรนด์, ถือไม้อยู่ไหม, SL / จุด breakout ถัดไป, กราฟพร้อมจุดเข้า-ออก, คำนวณขนาดไม้ |
| `/demo` บัญชีเดโม | มูลค่าบัญชี, ไม้ที่ถือ, เทียบเดโมกับ backtest ช่วงเดียวกัน (กราฟ + รายไม้), บันทึกเหตุการณ์, เริ่มรอบใหม่ |
| `/portfolio` | มูลค่าพอร์ต BTC + ทอง เทียบซื้อแล้วถือ, ตาราง metric แยกช่วงเวลา |
| `/research` | ผลทดสอบกฎจากบทความ, walk-forward, ระบบไซด์เวย์ |

- **การเรียก API:** เว็บเรียก API ผ่าน `/engine/*` (route handler ฝั่ง server) จึงไม่ต้องเปิด CORS ส่วนที่อยู่ API ตั้งด้วย `API_URL` ตอนรัน
- **ราคาสด:** browser ต่อ WebSocket ของ Binance ตรง ๆ พอแท่งปิด หน้าเว็บจะขอให้ API ดึงแท่งใหม่และคำนวณระบบใหม่เอง
- **พอร์ต:** ตั้งด้วย `WEB_PORT` ใน `.env` ได้ ค่าเริ่มต้น 3100 เพราะ 3000 ชนกับโปรเจกต์อื่นในเครื่อง
- **รันแบบ dev:** `cd web && npm run dev` (ต้องเปิด API ที่ :8000 ไว้)

## API

| Method | Path | ใช้ทำอะไร |
|---|---|---|
| GET | `/api/candles/{symbol}/{interval}?limit=1500` | แท่งเทียน (รูปแบบ Lightweight Charts) |
| GET | `/api/analysis/{symbol}/{interval}` | สถานะล่าสุดทุกตัววิเคราะห์ + คะแนนย้อนหลัง |
| POST | `/api/backtest` | `{symbol, interval, params: {...}}` รันและบันทึกผล |
| GET | `/api/runs` | รายการผล backtest ที่บันทึกไว้ |
| GET | `/api/runs/{id}` · `/trades` · `/equity` | รายละเอียดผลแต่ละรัน |
| GET | `/api/strategy/{symbol}/{interval}?bars=500` | ระบบ trend ที่ใช้ในพอร์ต: สถานะตอนนี้ + ไม้ทั้งหมด + เส้น SL/breakout/ทิศ (ดึงแท่งใหม่จาก Binance ให้เองถ้าข้อมูลเก่า) |
| GET | `/api/portfolio/latest` | equity พอร์ตล่าสุด (เริ่มที่ 100) + metric |
| GET | `/api/research/{rules\|walkforward\|range\|portfolio}` | ผลทดสอบล่าสุดแต่ละแบบ |
| GET | `/api/paper/status` · `/trades` · `/events` · `/equity` · `/compare` · `/runs` | บัญชีเดโม |
| POST | `/api/paper/new` | `{capital, risk_pct}` เริ่มบัญชีเดโมใหม่ |

## ความต่างจาก TradingView Strategy Tester

- **มี SL ตั้งแต่แท่งที่เข้า:** ใน Pine คำสั่ง exit ถูกตั้งตอนแท่งปิด แท่งแรกที่เข้าจึงไม่มี SL ปิดได้ด้วย `protect_fill_bar=false`
- **ไม่ได้พอร์ตตัวกรอง D1/1W และตารางแผนหลาย TF:** สองส่วนนี้ปิดไว้เป็นค่าเริ่มต้นใน Pine อยู่แล้ว
- **ตัวเลขอาจไม่ตรงกับ TradingView เป๊ะ:** วิธีเริ่มคำนวณ EMA และการนับ pivot ที่ราคาเท่ากันต่างกันเล็กน้อย แต่ทิศทางผลควรเหมือนกัน
