from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo
except ImportError:
    raise SystemExit("Python 3.9+ required for zoneinfo.")

# =========================================================
# CONFIG
# =========================================================
SYMBOL = "BTCUSDT"
BINANCE_BASE = "https://api.binance.com"
BINANCE_KLINES = "/api/v3/klines"

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES_PREFIX = "KXBTCD"

NY = ZoneInfo("America/New_York")

# Define your historical window here.
# This is interpreted in New York time.
START_DATE_NY = "2026-03-01 00:00"
END_DATE_NY = None  # e.g. "2026-04-20 00:00"; if None, uses current hour

# Must match Data_Sourcing/Kalshi_Pricing_Fetch/Kalshi_Contract_Price_Fetch.py
# so settlement rows describe the same contracts as the price rows.
REFERENCE_PRICE_MODE = "prev_close"  # "open" or "prev_close"

BINANCE_SLEEP_S = 0.05
KALSHI_SLEEP_S = 0.05

OUTPUT_FILENAME = "kalshi_btc_atm_settlements.csv"


# =========================================================
# TIME HELPERS
# =========================================================
def parse_ny_datetime(s: str) -> datetime:
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=NY)


def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def to_utc_millis(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def build_hour_starts(start_ny: datetime, end_ny: datetime) -> List[datetime]:
    """
    Returns NY-aware datetimes for each hour start in [start_ny, end_ny).
    """
    out: List[datetime] = []
    cur = floor_to_hour(start_ny)
    end_ny = floor_to_hour(end_ny)
    while cur < end_ny:
        out.append(cur)
        cur += timedelta(hours=1)
    return out


def build_kalshi_event_ticker(event_end_ny: datetime) -> str:
    """
    Kalshi BTC hourly events are labeled by the event settlement/end hour:
    KXBTCD-YYMONDDHH.
    """
    yy = event_end_ny.strftime("%y")
    mon = event_end_ny.strftime("%b").upper()
    dd = event_end_ny.strftime("%d")
    hh = event_end_ny.strftime("%H")
    return f"{KALSHI_SERIES_PREFIX}-{yy}{mon}{dd}{hh}"


# =========================================================
# BINANCE
# =========================================================
def fetch_binance_klines(
    symbol: str,
    start_utc_ms: int,
    end_utc_ms: int,
    interval: str = "1h",
    limit: int = 1000,
    sleep_s: float = 0.05,
) -> pd.DataFrame:
    rows: List[List[Any]] = []
    cur = start_utc_ms

    while cur < end_utc_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_utc_ms,
            "limit": limit,
        }
        r = requests.get(BINANCE_BASE + BINANCE_KLINES, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        rows.extend(data)

        last_open_time = int(data[-1][0])
        next_cur = last_open_time + 60 * 60 * 1000  # next hour in ms
        if next_cur <= cur:
            break
        cur = next_cur

        time.sleep(sleep_s)
        if len(data) < limit:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["open_time_ny"] = df["open_time_utc"].dt.tz_convert(NY)

    return df.sort_values("open_time_utc").reset_index(drop=True)


# =========================================================
# KALSHI
# =========================================================
def kalshi_request_json(
    session: requests.Session,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url = f"{KALSHI_BASE_URL}{path}"
    r = session.get(url, params=params, timeout=(5, 30))
    if r.status_code != 200:
        raise RuntimeError(f"GET {r.url} failed: {r.status_code} {r.text[:300]}")
    time.sleep(KALSHI_SLEEP_S)
    return r.json()


def kalshi_get_event_markets(session: requests.Session, event_ticker: str) -> List[Dict[str, Any]]:
    data = kalshi_request_json(
        session,
        f"/events/{event_ticker}",
        params={"with_nested_markets": "true"},
    )
    return data.get("markets") or (data.get("event", {}) or {}).get("markets") or []


def extract_strike_ladder(markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ladder: List[Dict[str, Any]] = []

    for m in markets:
        floor_strike = m.get("floor_strike")
        if floor_strike is None:
            continue

        try:
            strike = float(floor_strike)
        except (TypeError, ValueError):
            continue

        ladder.append(
            {
                "market_ticker": m.get("ticker"),
                "strike": strike,
                "result": m.get("result"),  # typically yes / no once finalized
                "status": m.get("status"),
                "expiration_value": m.get("expiration_value"),
            }
        )

    ladder.sort(key=lambda x: x["strike"])
    return ladder


def normalize_result_to_outcome(result: Optional[Any]) -> Optional[int]:
    if result is None or pd.isna(result):
        return None
    text = str(result).strip().lower()
    if text in {"yes", "y", "true", "1", "win", "won"}:
        return 1
    if text in {"no", "n", "false", "0", "lose", "lost"}:
        return 0
    return None


def pick_offsets_from_atm(
    ladder: List[Dict[str, Any]],
    reference_price: float,
    offsets: List[int] = [0, 1, -1, 2, -2],
) -> Dict[int, Optional[Dict[str, Any]]]:
    if not ladder:
        return {off: None for off in offsets}

    strikes = [m["strike"] for m in ladder]
    center_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - reference_price))

    out: Dict[int, Optional[Dict[str, Any]]] = {}
    for off in offsets:
        idx = center_idx + off
        if 0 <= idx < len(ladder):
            out[off] = ladder[idx]
        else:
            out[off] = None
    return out


# =========================================================
# MAIN BUILD
# =========================================================
def main():
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / OUTPUT_FILENAME

    start_ny = parse_ny_datetime(START_DATE_NY)
    if END_DATE_NY is None:
        end_ny = floor_to_hour(datetime.now(NY))
    else:
        end_ny = parse_ny_datetime(END_DATE_NY)

    if end_ny <= start_ny:
        raise ValueError("END_DATE_NY must be after START_DATE_NY.")

    # Pull one extra prior hour when prev_close is used for ATM/OTM selection.
    binance_start_ny = start_ny - timedelta(hours=1) if REFERENCE_PRICE_MODE == "prev_close" else start_ny
    df_binance = fetch_binance_klines(
        symbol=SYMBOL,
        start_utc_ms=to_utc_millis(binance_start_ny),
        end_utc_ms=to_utc_millis(end_ny),
        interval="1h",
        sleep_s=BINANCE_SLEEP_S,
    )

    if df_binance.empty:
        raise RuntimeError("No Binance hourly data returned for the requested window.")

    session = requests.Session()
    session.headers.update({"User-Agent": "kalshi-btc-historic-settlement-builder/1.0"})

    rows: List[Dict[str, Any]] = []

    target_rows = df_binance[(df_binance["open_time_ny"] >= start_ny) & (df_binance["open_time_ny"] < end_ny)].copy()
    if target_rows.empty:
        raise RuntimeError("No target Binance hourly rows found for the requested window.")

    total = len(target_rows)
    for i, row in target_rows.reset_index(drop=True).iterrows():
        hour_start_ny = row["open_time_ny"]
        event_end_ny = hour_start_ny + timedelta(hours=1)
        event_ticker = build_kalshi_event_ticker(event_end_ny)

        if REFERENCE_PRICE_MODE == "open":
            reference_price = float(row["open"])
        elif REFERENCE_PRICE_MODE == "prev_close":
            prev_row = df_binance.loc[df_binance["open_time_ny"] == (hour_start_ny - timedelta(hours=1))]
            if prev_row.empty:
                print(f"[{i+1}/{total}] SKIP {event_ticker}: missing previous-hour Binance close")
                continue
            reference_price = float(prev_row.iloc[0]["close"])
        else:
            raise ValueError("REFERENCE_PRICE_MODE must be 'open' or 'prev_close'.")

        # Diagnostic only: this is not the authoritative contract truth.
        # It approximates the event cutoff with the last Binance 1h close before event_end_ny.
        binance_audit_price = float(row["close"])

        try:
            markets = kalshi_get_event_markets(session, event_ticker)
            ladder = extract_strike_ladder(markets)

            if not ladder:
                print(f"[{i+1}/{total}] SKIP {event_ticker}: no usable strike ladder")
                continue

            picked = pick_offsets_from_atm(ladder, reference_price, offsets=[0, 1, -1, 2, -2])

            def get_strike(off: int) -> Optional[float]:
                m = picked.get(off)
                return None if m is None else float(m["strike"])

            def get_result(off: int) -> Optional[str]:
                m = picked.get(off)
                return None if m is None else m.get("result")

            def get_outcome(off: int) -> Optional[int]:
                return normalize_result_to_outcome(get_result(off))

            def get_market_ticker(off: int) -> Optional[str]:
                m = picked.get(off)
                return None if m is None else m.get("market_ticker")

            rows.append(
                {
                    "event_datetime": event_end_ny.isoformat(),
                    "forecast_hour_start_datetime": hour_start_ny.isoformat(),
                    "event_ticker": event_ticker,
                    "binance_reference_price": reference_price,
                    "binance_audit_price": binance_audit_price,
                    "binance_audit_price_definition": "Binance 1h close for the forecast hour; diagnostic only.",

                    "ATM_market_ticker": get_market_ticker(0),
                    "ATM_strike": get_strike(0),
                    "ATM_result": get_result(0),
                    "ATM_outcome": get_outcome(0),

                    "OTM+1_market_ticker": get_market_ticker(1),
                    "OTM+1_strike": get_strike(1),
                    "OTM+1_result": get_result(1),
                    "OTM+1_outcome": get_outcome(1),

                    "OTM-1_market_ticker": get_market_ticker(-1),
                    "OTM-1_strike": get_strike(-1),
                    "OTM-1_result": get_result(-1),
                    "OTM-1_outcome": get_outcome(-1),

                    "OTM+2_market_ticker": get_market_ticker(2),
                    "OTM+2_strike": get_strike(2),
                    "OTM+2_result": get_result(2),
                    "OTM+2_outcome": get_outcome(2),

                    "OTM-2_market_ticker": get_market_ticker(-2),
                    "OTM-2_strike": get_strike(-2),
                    "OTM-2_result": get_result(-2),
                    "OTM-2_outcome": get_outcome(-2),
                }
            )

            print(
                f"[{i+1}/{total}] OK {event_ticker} | "
                f"ref={reference_price:.2f} | audit={binance_audit_price:.2f} | ATM={get_strike(0)}"
            )

        except Exception as e:
            print(f"[{i+1}/{total}] ERROR {event_ticker}: {e}")

    if not rows:
        raise RuntimeError("No rows were built. Check your date range and API responses.")

    out_df = pd.DataFrame(rows).sort_values("event_datetime").reset_index(drop=True)
    out_df.to_csv(output_path, index=False)

    print("\nDone.")
    print(f"Saved: {output_path}")
    print(f"Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
