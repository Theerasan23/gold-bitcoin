"""ดึงแท่งเทียนจาก Binance แล้วเก็บเป็น Parquet (อัปเดตแบบต่อท้าย ไม่โหลดซ้ำ)"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import numpy as np
import polars as pl

from .config import DATA_DIR

BINANCE = "https://api.binance.com"
KLINE_LIMIT = 1000

SCHEMA = {
    "open_time": pl.Datetime("ms", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


def candle_path(symbol: str, interval: str, data_dir: Path = DATA_DIR) -> Path:
    # "1M" กับ "1m" ชนกันบนระบบไฟล์ที่ไม่แยกตัวพิมพ์ (macOS) -> ใช้ "1mo"
    name = "1mo" if interval == "1M" else interval
    return data_dir / "candles" / symbol.upper() / f"{name}.parquet"


def _get(client: httpx.Client, path: str, params: dict) -> list | dict:
    for attempt in range(6):
        try:
            r = client.get(path, params=params)
        except httpx.TransportError:
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (418, 429) or r.status_code >= 500:
            time.sleep(int(r.headers.get("Retry-After", 2 ** attempt)))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Binance request failed: {path} {params}")


def fetch_klines(symbol: str, interval: str, start_ms: int = 0) -> pl.DataFrame:
    """ดึงแท่งที่ปิดแล้วทั้งหมดตั้งแต่ start_ms (0 = ตั้งแต่เหรียญเข้าตลาด)"""
    rows: list[list] = []
    now_ms = int(time.time() * 1000)
    with httpx.Client(base_url=BINANCE, timeout=30) as client:
        while True:
            batch = _get(client, "/api/v3/klines",
                         {"symbol": symbol.upper(), "interval": interval,
                          "startTime": start_ms, "limit": KLINE_LIMIT})
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < KLINE_LIMIT:
                break
            start_ms = batch[-1][0] + 1
    rows = [k for k in rows if k[6] < now_ms]  # ตัดแท่งที่ยังไม่ปิด
    if not rows:
        return pl.DataFrame(schema=SCHEMA)
    arr = np.array([[k[0], k[1], k[2], k[3], k[4], k[5]] for k in rows], dtype=object)
    return pl.DataFrame({
        "open_time": pl.Series(arr[:, 0].astype(np.int64)).cast(pl.Datetime("ms")).dt.replace_time_zone("UTC"),
        "open": arr[:, 1].astype(np.float64),
        "high": arr[:, 2].astype(np.float64),
        "low": arr[:, 3].astype(np.float64),
        "close": arr[:, 4].astype(np.float64),
        "volume": arr[:, 5].astype(np.float64),
    })


def fetch_tick_size(symbol: str) -> float:
    with httpx.Client(base_url=BINANCE, timeout=30) as client:
        info = _get(client, "/api/v3/exchangeInfo", {"symbol": symbol.upper()})
    for f in info["symbols"][0]["filters"]:
        if f["filterType"] == "PRICE_FILTER":
            return float(f["tickSize"])
    return 0.01


def update(symbol: str, interval: str, data_dir: Path = DATA_DIR) -> tuple[Path, int]:
    """โหลดเพิ่มเฉพาะแท่งใหม่ต่อจากไฟล์เดิม คืนค่า (path, จำนวนแท่งใหม่)"""
    path = candle_path(symbol, interval, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pl.read_parquet(path) if path.exists() else None
    start = 0
    if old is not None and old.height:
        start = int(old["open_time"].max().timestamp() * 1000) + 1
    new = fetch_klines(symbol, interval, start)
    df = new if old is None else pl.concat([old, new])
    df = df.unique("open_time", keep="last").sort("open_time")
    # เขียนไฟล์ชั่วคราวแล้วสลับทีเดียว : API กับ worker เดโมอัปเดตพร้อมกันได้โดยไฟล์ไม่เสีย
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)

    meta = path.parent / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"symbol": symbol.upper(), "tick_size": fetch_tick_size(symbol)}))
    return path, new.height


def load(symbol: str, interval: str, data_dir: Path = DATA_DIR) -> pl.DataFrame:
    path = candle_path(symbol, interval, data_dir)
    if not path.exists():
        raise FileNotFoundError(f"ไม่มีข้อมูล {symbol} {interval} — รัน: fetch --symbols {symbol} --intervals {interval}")
    return pl.read_parquet(path)


def tick_size(symbol: str, data_dir: Path = DATA_DIR) -> float:
    meta = data_dir / "candles" / symbol.upper() / "meta.json"
    if meta.exists():
        return float(json.loads(meta.read_text())["tick_size"])
    return 0.01
