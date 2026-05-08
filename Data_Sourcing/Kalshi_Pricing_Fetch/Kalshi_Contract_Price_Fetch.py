from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    raise SystemExit("Python 3.9+ required for zoneinfo.")

# =========================================================
# CONFIG
# =========================================================
SYMBOL = "BTCUSDT"

# Start datetime in UTC. Example: "2026-04-18 00:00:00"
START_DATETIME_UTC = "2026-03-01 05:00:00"

# Number of hourly events to download
NUM_HOURS = 1300

# Strike selection reference price:
#   "open"       -> use the hour's Binance open
#   "prev_close" -> use previous hour's Binance close (often more realistic)
REFERENCE_PRICE_MODE = "prev_close"   # "open" or "prev_close"

# Which Kalshi price to write into CSV:
#   "yes_mid", "yes_ask", "yes_bid", "no_mid", "no_ask", "no_bid"
KALSHI_PRICE_FIELD = "yes_mid"

# Output folder
OUTPUT_FOLDER_NAME = "hourly_events_price_data"

# Networking
BINANCE_BASE = "https://api.binance.com"
BINANCE_KLINES = "/api/v3/klines"

KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_SERIES_PREFIX = "KXBTCD"
KALSHI_CANDLE_PERIOD_INTERVAL_MINUTES = 1
KALSHI_SLEEP_S = 0.10

NY = ZoneInfo("America/New_York")


# =========================================================
# HELPERS
# =========================================================
def parse_utc(dt_str: str) -> datetime:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc)


def to_utc_millis(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware.")
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def build_kalshi_event_ticker(dt_et: datetime) -> str:
    """
    Kalshi BTC hourly events are labeled by ET hour in the format:
    KXBTCD-YYMONDDHH
    """
    yy = dt_et.strftime("%y")
    mon = dt_et.strftime("%b").upper()
    dd = dt_et.strftime("%d")
    hh = dt_et.strftime("%H")
    return f"{KALSHI_SERIES_PREFIX}-{yy}{mon}{dd}{hh}"


def kalshi_request_json(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r = session.get(url, params=params, timeout=(5, 30))
    if r.status_code != 200:
        raise RuntimeError(f"GET {r.url} failed: {r.status_code} {r.text[:300]}")
    time.sleep(KALSHI_SLEEP_S)
    return r.json()


def kalshi_get_event_markets(session: requests.Session, event_ticker: str) -> List[Dict[str, Any]]:
    data = kalshi_request_json(
        session,
        f"{KALSHI_BASE_URL}/events/{event_ticker}",
        params={"with_nested_markets": "true"},
    )
    return data.get("markets") or (data.get("event", {}) or {}).get("markets") or []


def kalshi_get_market_candlesticks(
    session: requests.Session,
    *,
    market_ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int,
) -> List[Dict[str, Any]]:
    url = f"{KALSHI_BASE_URL}/series/{KALSHI_SERIES_PREFIX}/markets/{market_ticker}/candlesticks"
    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval,
    }
    data = kalshi_request_json(session, url, params=params)
    return data.get("candlesticks", []) or []


def _maybe_close01(candle: Dict[str, Any], side_key: str) -> Optional[float]:
    d = candle.get(side_key)
    if not isinstance(d, dict):
        return None

    # NEW Kalshi format
    if "close_dollars" in d:
        try:
            return float(d["close_dollars"])
        except Exception:
            return None

    # OLD fallback format
    if "close" in d:
        try:
            return float(d["close"]) / 100.0
        except Exception:
            return None

    return None

def kalshi_download_candle_series_for_hour(
    session: requests.Session,
    *,
    market_ticker: str,
    hour_start_utc: datetime,
    hour_end_utc: datetime,
) -> pd.DataFrame:
    start_ts = int(hour_start_utc.timestamp())
    end_ts = int(hour_end_utc.timestamp())

    candles = kalshi_get_market_candlesticks(
        session,
        market_ticker=market_ticker,
        start_ts=start_ts,
        end_ts=end_ts,
        period_interval=KALSHI_CANDLE_PERIOD_INTERVAL_MINUTES,
    )

    rows = []
    for c in candles:
        end_period_ts = c.get("end_period_ts")
        if not isinstance(end_period_ts, (int, float)):
            continue

        t_end = int(end_period_ts)
        if not (start_ts < t_end <= end_ts):
            continue

        # Candle ending at t_end represents minute [t_end-60, t_end)
        t_start_utc = datetime.fromtimestamp(t_end - 60, tz=timezone.utc).replace(second=0, microsecond=0)

        yes_ask = _maybe_close01(c, "yes_ask")
        yes_bid = _maybe_close01(c, "yes_bid")
        no_ask = _maybe_close01(c, "no_ask")
        no_bid = _maybe_close01(c, "no_bid")

        rows.append(
            {
                "datetime": t_start_utc,
                "yes_ask": yes_ask,
                "yes_bid": yes_bid,
                "yes_mid": (yes_ask + yes_bid) / 2.0 if yes_ask is not None and yes_bid is not None else None,
                "no_ask": no_ask,
                "no_bid": no_bid,
                "no_mid": (no_ask + no_bid) / 2.0 if no_ask is not None and no_bid is not None else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["datetime", "yes_ask", "yes_bid", "yes_mid", "no_ask", "no_bid", "no_mid"])

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def fetch_binance_klines(
    symbol: str,
    interval: str,
    start_utc_ms: int,
    end_utc_ms: int,
    limit: int = 1000,
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
        next_cur = last_open_time + (60 * 60 * 1000 if interval == "1h" else 60_000)
        if next_cur <= cur:
            break
        cur = next_cur

        time.sleep(0.05)
        if len(data) < limit:
            break

    if not rows:
        raise RuntimeError(f"No Binance {interval} klines returned.")

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

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def choose_atm_plus_minus_one(markets: List[Dict[str, Any]], reference_price: float) -> Dict[str, Dict[str, Any]]:
    """
    Choose ATM as the market whose floor_strike is closest to the reference price.
    Then choose adjacent strikes by sorted strike order:
      OTM+1 = one strike above ATM
      OTM-1 = one strike below ATM
    """
    usable: List[Tuple[float, Dict[str, Any]]] = []
    for m in markets:
        fs = m.get("floor_strike")
        if fs is None:
            continue
        try:
            usable.append((float(fs), m))
        except Exception:
            continue

    if not usable:
        raise RuntimeError("No markets with usable floor_strike found.")

    usable.sort(key=lambda x: x[0])
    strikes = [x[0] for x in usable]

    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - float(reference_price)))

    result: Dict[str, Dict[str, Any]] = {}

    # ATM
    result["ATM"] = usable[atm_idx][1]

    # OTM+1 = strike above ATM
    if atm_idx + 1 < len(usable):
        result["OTM+1"] = usable[atm_idx + 1][1]

    # OTM-1 = strike below ATM
    if atm_idx - 1 >= 0:
        result["OTM-1"] = usable[atm_idx - 1][1]

    return result


def safe_price_lookup(df: pd.DataFrame, dt: datetime, price_field: str) -> Optional[float]:
    if df.empty:
        return None
    hit = df.loc[df["datetime"] == dt]
    if hit.empty:
        return None
    val = hit.iloc[0].get(price_field)
    if pd.isna(val):
        return None
    return float(val)


def build_output_rows(
    event_ticker: str,
    hour_start_utc: datetime,
    atm_market: Optional[Dict[str, Any]],
    plus_market: Optional[Dict[str, Any]],
    minus_market: Optional[Dict[str, Any]],
    atm_df: pd.DataFrame,
    plus_df: pd.DataFrame,
    minus_df: pd.DataFrame,
    price_field: str,
) -> pd.DataFrame:
    rows = []

    atm_strike = float(atm_market["floor_strike"]) if atm_market is not None and atm_market.get("floor_strike") is not None else None
    plus_strike = float(plus_market["floor_strike"]) if plus_market is not None and plus_market.get("floor_strike") is not None else None
    minus_strike = float(minus_market["floor_strike"]) if minus_market is not None and minus_market.get("floor_strike") is not None else None
    atm_ticker = atm_market.get("ticker") if atm_market is not None else None
    plus_ticker = plus_market.get("ticker") if plus_market is not None else None
    minus_ticker = minus_market.get("ticker") if minus_market is not None else None

    for minute_number in range(60):
        dt = hour_start_utc + timedelta(minutes=minute_number)

        rows.append(
            {
                "Event": event_ticker,
                "datetime": dt.isoformat(),
                "minute_number": minute_number,
                "ATM_market_ticker": atm_ticker,
                "ATM_strike": atm_strike,
                "ATM_price": safe_price_lookup(atm_df, dt, price_field),
                "OTM+1_market_ticker": plus_ticker,
                "OTM+1_strike": plus_strike,
                "OTM+1_price": safe_price_lookup(plus_df, dt, price_field),
                "OTM-1_market_ticker": minus_ticker,
                "OTM-1_strike": minus_strike,
                "OTM-1_price": safe_price_lookup(minus_df, dt, price_field),
            }
        )

    return pd.DataFrame(rows)


def event_filename(event_ticker: str, hour_start_utc: datetime) -> str:
    return f"{hour_start_utc.strftime('%Y%m%d_%H00UTC')}__{event_ticker}.csv"


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    start_dt_utc = parse_utc(START_DATETIME_UTC)
    end_dt_utc = start_dt_utc + timedelta(hours=NUM_HOURS)

    output_dir = Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Binance hourly data...")
    # Need one extra prior hour if using prev_close
    binance_fetch_start = start_dt_utc - timedelta(hours=1) if REFERENCE_PRICE_MODE == "prev_close" else start_dt_utc

    df_1h = fetch_binance_klines(
        symbol=SYMBOL,
        interval="1h",
        start_utc_ms=to_utc_millis(binance_fetch_start),
        end_utc_ms=to_utc_millis(end_dt_utc),
    )

    df_1h = df_1h[["open_time", "open", "close"]].copy()
    df_1h = df_1h.sort_values("open_time").reset_index(drop=True)

    # Keep only target hours for processing
    target_hours = []
    for i in range(NUM_HOURS):
        hour_start = start_dt_utc + timedelta(hours=i)
        row = df_1h.loc[df_1h["open_time"] == hour_start]
        if row.empty:
            print(f"Skipping {hour_start.isoformat()} - missing Binance hourly candle.")
            continue
        target_hours.append(hour_start)

    if not target_hours:
        raise RuntimeError("No hourly Binance candles found for requested range.")

    session = requests.Session()
    session.headers.update({"User-Agent": "hourly-kalshi-price-fetch/1.0"})

    for hour_start_utc in target_hours:
        hour_end_utc = hour_start_utc + timedelta(hours=1)

        row = df_1h.loc[df_1h["open_time"] == hour_start_utc]
        if row.empty:
            print(f"Skipping {hour_start_utc.isoformat()} - no Binance row.")
            continue

        row = row.iloc[0]

        if REFERENCE_PRICE_MODE == "open":
            reference_price = float(row["open"])
        elif REFERENCE_PRICE_MODE == "prev_close":
            prev_row = df_1h.loc[df_1h["open_time"] == (hour_start_utc - timedelta(hours=1))]
            if prev_row.empty:
                print(f"Skipping {hour_start_utc.isoformat()} - previous hour close unavailable.")
                continue
            reference_price = float(prev_row.iloc[0]["close"])
        else:
            raise ValueError("REFERENCE_PRICE_MODE must be 'open' or 'prev_close'.")

        # Same logic your original script used: build event ticker using hour_end in New York time
        event_ticker = build_kalshi_event_ticker(hour_end_utc.astimezone(NY))

        print(f"\nProcessing hour {hour_start_utc.isoformat()} -> event {event_ticker} | ref_price={reference_price:,.2f}")

        try:
            markets = kalshi_get_event_markets(session, event_ticker)
            if not markets:
                print("  No Kalshi markets found. Skipping.")
                continue

            picked = choose_atm_plus_minus_one(markets, reference_price=reference_price)

            atm_market = picked.get("ATM")
            plus_market = picked.get("OTM+1")
            minus_market = picked.get("OTM-1")

            if atm_market is None:
                print("  Could not determine ATM market. Skipping.")
                continue

            print(
                "  Selected strikes:",
                {
                    "ATM": atm_market.get("floor_strike"),
                    "OTM+1": plus_market.get("floor_strike") if plus_market else None,
                    "OTM-1": minus_market.get("floor_strike") if minus_market else None,
                },
            )

            atm_df = kalshi_download_candle_series_for_hour(
                session=session,
                market_ticker=atm_market["ticker"],
                hour_start_utc=hour_start_utc,
                hour_end_utc=hour_end_utc,
            )

            plus_df = pd.DataFrame(columns=["datetime", KALSHI_PRICE_FIELD])
            if plus_market is not None:
                plus_df = kalshi_download_candle_series_for_hour(
                    session=session,
                    market_ticker=plus_market["ticker"],
                    hour_start_utc=hour_start_utc,
                    hour_end_utc=hour_end_utc,
                )

            minus_df = pd.DataFrame(columns=["datetime", KALSHI_PRICE_FIELD])
            if minus_market is not None:
                minus_df = kalshi_download_candle_series_for_hour(
                    session=session,
                    market_ticker=minus_market["ticker"],
                    hour_start_utc=hour_start_utc,
                    hour_end_utc=hour_end_utc,
                )

            out_df = build_output_rows(
                event_ticker=event_ticker,
                hour_start_utc=hour_start_utc,
                atm_market=atm_market,
                plus_market=plus_market,
                minus_market=minus_market,
                atm_df=atm_df,
                plus_df=plus_df,
                minus_df=minus_df,
                price_field=KALSHI_PRICE_FIELD,
            )

            out_path = output_dir / event_filename(event_ticker, hour_start_utc)
            out_df.to_csv(out_path, index=False)
            print(f"  Wrote: {out_path}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
