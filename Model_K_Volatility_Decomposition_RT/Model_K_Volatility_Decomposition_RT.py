from __future__ import annotations

import argparse
import html
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model_K.Model_K import (  # noqa: E402
    ASSUMPTIONS,
    attach_outcomes,
    build_brier_decomposition,
    build_kalshi_reality_outcomes,
    build_metrics_summary,
    build_outcome_join_coverage,
    build_resolution_mismatches,
    build_sharpness,
    build_summary_html,
    build_time_bucket_outputs,
    build_time_bucket_summary_html,
    default_settlement_csv,
    expanded_calibration_error,
    load_kalshi_price_outputs,
    load_settlements,
    write_individual_metric_files,
)


SYMBOL = "BTCUSDT"
BINANCE_BASE = "https://api.binance.com"
BINANCE_KLINES = "/api/v3/klines"
ROLLING_WINDOW_HOURS = 72
ROLLING_WINDOW_MINUTES = 4_320

SEGMENTS: List[Dict[str, str]] = [
    {
        "name": "Low Volatility",
        "flag": "is_low_volatility",
        "rule": "Hourly realized volatility at or below the trailing 72-hour rolling 25th percentile.",
    },
    {
        "name": "Standard Volatility",
        "flag": "is_standard_volatility",
        "rule": "Hourly realized volatility strictly between the trailing 72-hour rolling 25th and 75th percentiles.",
    },
    {
        "name": "High Volatility",
        "flag": "is_high_volatility",
        "rule": "Hourly realized volatility at or above the trailing 72-hour rolling 75th percentile.",
    },
    {
        "name": "Low Volatility Extreme",
        "flag": "is_low_volatility_extreme",
        "rule": "Hourly realized volatility at or below the trailing 72-hour rolling 10th percentile.",
    },
    {
        "name": "High Volatility Extreme",
        "flag": "is_high_volatility_extreme",
        "rule": "Hourly realized volatility at or above the trailing 72-hour rolling 90th percentile.",
    },
]


def repo_root_from_script() -> Path:
    return REPO_ROOT


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Run Model K outputs separately across hourly Kalshi BTC market states "
            "defined by real-time trailing Binance realized volatility."
        )
    )
    parser.add_argument(
        "--kalshi-price-dir",
        type=Path,
        default=root / "Data_Sourcing" / "Kalshi_Pricing_Fetch" / "hourly_events_price_data",
        help="Folder containing Kalshi hourly price CSV outputs.",
    )
    parser.add_argument(
        "--settlement-csv",
        type=Path,
        default=default_settlement_csv(root),
        help="Settlement CSV with official Kalshi outcomes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Model_K_Volatility_Decomposition_RT_outputs",
        help="Root directory for real-time volatility decomposition outputs.",
    )
    parser.add_argument(
        "--binance-minute-cache-csv",
        type=Path,
        default=None,
        help=(
            "Optional cache path for Binance 1-minute BTCUSDT klines. "
            "If the cache exists and covers the full required interval, it is reused."
        ),
    )
    parser.add_argument(
        "--refresh-binance-cache",
        action="store_true",
        help="Force a fresh Binance 1-minute download even if the local cache already exists.",
    )
    parser.add_argument(
        "--skip-binance-audit",
        action="store_true",
        help="Skip the existing diagnostic comparison between official Kalshi outcomes and Binance audit prices.",
    )
    parser.add_argument("--calibration-bins", type=int, default=10, help="Number of calibration bins.")
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Threshold used for binary classification accuracy in the metric summary.",
    )
    return parser.parse_args()


def fetch_binance_klines(
    *,
    symbol: str,
    interval: str,
    start_utc_ms: int,
    end_utc_ms: int,
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
        response = requests.get(BINANCE_BASE + BINANCE_KLINES, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data:
            break

        rows.extend(data)

        last_open_time = int(data[-1][0])
        step_ms = 60_000 if interval == "1m" else 60 * 60 * 1000
        next_cur = last_open_time + step_ms
        if next_cur <= cur:
            break
        cur = next_cur

        time.sleep(sleep_s)
        if len(data) < limit:
            break

    if not rows:
        raise RuntimeError(f"No Binance {interval} klines returned for {symbol}.")

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
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["open_time_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True)
    return df.sort_values("open_time_utc").drop_duplicates(subset=["open_time_utc"]).reset_index(drop=True)


def resolve_binance_cache_path(output_dir: Path, cache_arg: Path | None) -> Path:
    if cache_arg is None:
        return output_dir / "binance_1m_klines.csv"
    return cache_arg if cache_arg.is_absolute() else (repo_root_from_script() / cache_arg)


def load_or_fetch_binance_minutes(
    *,
    cache_path: Path,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    refresh_cache: bool,
) -> pd.DataFrame:
    required_start = start_utc.floor("min")
    required_end = end_utc.ceil("min")

    if cache_path.exists() and not refresh_cache:
        cached = pd.read_csv(cache_path)
        if "open_time_utc" not in cached.columns:
            raise ValueError(f"Cache file {cache_path} is missing open_time_utc.")
        cached["open_time_utc"] = pd.to_datetime(cached["open_time_utc"], utc=True)
        if "close_time_utc" in cached.columns:
            cached["close_time_utc"] = pd.to_datetime(cached["close_time_utc"], utc=True)
        coverage_start = cached["open_time_utc"].min()
        coverage_end = cached["open_time_utc"].max() + pd.Timedelta(minutes=1)
        if coverage_start <= required_start and coverage_end >= required_end:
            filtered = cached[
                (cached["open_time_utc"] >= required_start) & (cached["open_time_utc"] < required_end)
            ].copy()
            if not filtered.empty:
                return filtered.reset_index(drop=True)

    fetched = fetch_binance_klines(
        symbol=SYMBOL,
        interval="1m",
        start_utc_ms=int(required_start.timestamp() * 1000),
        end_utc_ms=int(required_end.timestamp() * 1000),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fetched.to_csv(cache_path, index=False)
    return fetched.reset_index(drop=True)


def compute_hourly_realized_volatility(binance_minutes: pd.DataFrame) -> pd.DataFrame:
    work = binance_minutes.copy()
    work = work.dropna(subset=["open_time_utc", "open", "close"])
    work = work[(work["open"] > 0) & (work["close"] > 0)].copy()
    work["hour_start_utc"] = work["open_time_utc"].dt.floor("h")
    work["minute_log_return"] = np.log(work["close"] / work["open"])

    hourly = (
        work.groupby("hour_start_utc", as_index=False)
        .agg(
            realized_variance=("minute_log_return", lambda values: float(np.square(values).sum())),
            realized_volatility=("minute_log_return", lambda values: float(np.sqrt(np.square(values).sum()))),
            minute_bars=("minute_log_return", "size"),
            hour_open=("open", "first"),
            hour_close=("close", "last"),
            hour_high=("high", "max"),
            hour_low=("low", "min"),
        )
        .sort_values("hour_start_utc")
        .reset_index(drop=True)
    )
    return hourly


def add_realtime_window_thresholds(hourly_volatility: pd.DataFrame) -> pd.DataFrame:
    work = hourly_volatility.sort_values("hour_start_utc").reset_index(drop=True).copy()
    rolling = work["realized_volatility"].rolling(
        window=ROLLING_WINDOW_HOURS,
        min_periods=ROLLING_WINDOW_HOURS,
    )
    work["rolling_window_hours"] = ROLLING_WINDOW_HOURS
    work["rolling_window_minutes"] = ROLLING_WINDOW_MINUTES
    work["rolling_window_observations"] = rolling.count()
    work["q10_realized_volatility"] = rolling.quantile(0.10)
    work["q25_realized_volatility"] = rolling.quantile(0.25)
    work["q75_realized_volatility"] = rolling.quantile(0.75)
    work["q90_realized_volatility"] = rolling.quantile(0.90)
    work["rolling_percentile_rank"] = work["realized_volatility"].rolling(
        window=ROLLING_WINDOW_HOURS,
        min_periods=ROLLING_WINDOW_HOURS,
    ).apply(lambda values: float(np.mean(values <= values[-1])), raw=True)
    return work


def build_hourly_market_state_table(raw: pd.DataFrame, hourly_volatility: pd.DataFrame) -> pd.DataFrame:
    events = raw[["event_ticker", "event_datetime_utc"]].drop_duplicates().copy()
    events["forecast_hour_start_utc"] = events["event_datetime_utc"] - pd.Timedelta(hours=1)

    merged = events.merge(
        hourly_volatility,
        left_on="forecast_hour_start_utc",
        right_on="hour_start_utc",
        how="left",
    )

    missing = merged[merged["realized_volatility"].isna()].copy()
    if not missing.empty:
        sample = ", ".join(missing["event_ticker"].head(5).astype(str).tolist())
        raise ValueError(
            "Missing Binance realized volatility for one or more scored Kalshi hours. "
            f"Example event_ticker(s): {sample}"
        )

    missing_window = merged[merged["q25_realized_volatility"].isna()].copy()
    if not missing_window.empty:
        sample = ", ".join(missing_window["event_ticker"].head(5).astype(str).tolist())
        raise ValueError(
            "Insufficient Binance history to compute the trailing 72-hour real-time volatility window "
            f"for one or more scored Kalshi hours. Example event_ticker(s): {sample}"
        )

    merged["is_low_volatility"] = merged["realized_volatility"] <= merged["q25_realized_volatility"]
    merged["is_standard_volatility"] = (
        (merged["realized_volatility"] > merged["q25_realized_volatility"])
        & (merged["realized_volatility"] < merged["q75_realized_volatility"])
    )
    merged["is_high_volatility"] = merged["realized_volatility"] >= merged["q75_realized_volatility"]
    merged["is_low_volatility_extreme"] = merged["realized_volatility"] <= merged["q10_realized_volatility"]
    merged["is_high_volatility_extreme"] = merged["realized_volatility"] >= merged["q90_realized_volatility"]
    merged["primary_volatility_band"] = np.select(
        [
            merged["is_low_volatility"],
            merged["is_high_volatility"],
        ],
        [
            "Low Volatility",
            "High Volatility",
        ],
        default="Standard Volatility",
    )
    return merged.sort_values("forecast_hour_start_utc").reset_index(drop=True)


def thresholds_table(hourly_market_states: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "threshold_name": "Low Volatility Extreme",
            "percentile": 0.10,
            "window_hours": ROLLING_WINDOW_HOURS,
            "window_minutes": ROLLING_WINDOW_MINUTES,
            "latest_realized_volatility_cutoff": float(hourly_market_states["q10_realized_volatility"].iloc[-1]),
            "min_realized_volatility_cutoff": float(hourly_market_states["q10_realized_volatility"].min()),
            "median_realized_volatility_cutoff": float(hourly_market_states["q10_realized_volatility"].median()),
            "max_realized_volatility_cutoff": float(hourly_market_states["q10_realized_volatility"].max()),
            "rule": "realized_volatility <= trailing_72h_q10",
        },
        {
            "threshold_name": "Low Volatility",
            "percentile": 0.25,
            "window_hours": ROLLING_WINDOW_HOURS,
            "window_minutes": ROLLING_WINDOW_MINUTES,
            "latest_realized_volatility_cutoff": float(hourly_market_states["q25_realized_volatility"].iloc[-1]),
            "min_realized_volatility_cutoff": float(hourly_market_states["q25_realized_volatility"].min()),
            "median_realized_volatility_cutoff": float(hourly_market_states["q25_realized_volatility"].median()),
            "max_realized_volatility_cutoff": float(hourly_market_states["q25_realized_volatility"].max()),
            "rule": "realized_volatility <= trailing_72h_q25",
        },
        {
            "threshold_name": "High Volatility",
            "percentile": 0.75,
            "window_hours": ROLLING_WINDOW_HOURS,
            "window_minutes": ROLLING_WINDOW_MINUTES,
            "latest_realized_volatility_cutoff": float(hourly_market_states["q75_realized_volatility"].iloc[-1]),
            "min_realized_volatility_cutoff": float(hourly_market_states["q75_realized_volatility"].min()),
            "median_realized_volatility_cutoff": float(hourly_market_states["q75_realized_volatility"].median()),
            "max_realized_volatility_cutoff": float(hourly_market_states["q75_realized_volatility"].max()),
            "rule": "realized_volatility >= trailing_72h_q75",
        },
        {
            "threshold_name": "High Volatility Extreme",
            "percentile": 0.90,
            "window_hours": ROLLING_WINDOW_HOURS,
            "window_minutes": ROLLING_WINDOW_MINUTES,
            "latest_realized_volatility_cutoff": float(hourly_market_states["q90_realized_volatility"].iloc[-1]),
            "min_realized_volatility_cutoff": float(hourly_market_states["q90_realized_volatility"].min()),
            "median_realized_volatility_cutoff": float(hourly_market_states["q90_realized_volatility"].median()),
            "max_realized_volatility_cutoff": float(hourly_market_states["q90_realized_volatility"].max()),
            "rule": "realized_volatility >= trailing_72h_q90",
        },
    ]
    return pd.DataFrame(rows)


def enrich_rows_with_market_state(frame: pd.DataFrame, hourly_market_states: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    state_columns = [
        "event_ticker",
        "forecast_hour_start_utc",
        "realized_variance",
        "realized_volatility",
        "minute_bars",
        "hour_open",
        "hour_close",
        "hour_high",
        "hour_low",
        "rolling_window_hours",
        "rolling_window_minutes",
        "rolling_window_observations",
        "rolling_percentile_rank",
        "primary_volatility_band",
        "is_low_volatility",
        "is_standard_volatility",
        "is_high_volatility",
        "is_low_volatility_extreme",
        "is_high_volatility_extreme",
        "q10_realized_volatility",
        "q25_realized_volatility",
        "q75_realized_volatility",
        "q90_realized_volatility",
    ]
    return frame.merge(
        hourly_market_states[state_columns],
        on="event_ticker",
        how="left",
    )


def format_threshold_range(thresholds: pd.DataFrame, threshold_name: str) -> str:
    row = thresholds.loc[thresholds["threshold_name"] == threshold_name].iloc[0]
    return (
        f"latest={float(row['latest_realized_volatility_cutoff']):.8f}, "
        f"median={float(row['median_realized_volatility_cutoff']):.8f}, "
        f"range=[{float(row['min_realized_volatility_cutoff']):.8f}, "
        f"{float(row['max_realized_volatility_cutoff']):.8f}]"
    )


def build_segment_messages(
    *,
    segment_name: str,
    segment_rule: str,
    coverage: pd.DataFrame,
    resolution_mismatches: pd.DataFrame,
    segment_event_count: int,
    total_event_count: int,
    thresholds: pd.DataFrame,
) -> List[str]:
    overall_coverage = coverage.loc[coverage["scope"] == "overall"].iloc[0]

    messages = [
        "Loaded official Kalshi contract outcomes from kalshi_btc_atm_settlements.csv.",
        (
            f"Matched {int(overall_coverage['matched_rows'])} of "
            f"{int(overall_coverage['total_forecast_rows'])} forecast rows to official Kalshi outcomes "
            f"inside the {segment_name} segment."
        ),
        (
            f"{segment_name} contains {segment_event_count} of {total_event_count} scored hourly Kalshi markets. "
            f"{segment_rule}"
        ),
        (
            "Hourly realized volatility is computed from Binance BTCUSDT 1-minute candles as "
            "sqrt(sum(log(close/open)^2)) over the 60 one-minute bars in each forecast hour."
        ),
        (
            f"Each scored hour is classified against a trailing {ROLLING_WINDOW_HOURS}-hour "
            f"({ROLLING_WINDOW_MINUTES}-minute) rolling window using only historical Binance volatility available at that hour."
        ),
        (
            "Rolling cutoff summaries for this run: "
            f"q10 {format_threshold_range(thresholds, 'Low Volatility Extreme')}; "
            f"q25 {format_threshold_range(thresholds, 'Low Volatility')}; "
            f"q75 {format_threshold_range(thresholds, 'High Volatility')}; "
            f"q90 {format_threshold_range(thresholds, 'High Volatility Extreme')}."
        ),
    ]
    if len(resolution_mismatches):
        messages.append(
            f"Found {len(resolution_mismatches)} Binance audit mismatch row(s); see resolution_mismatches.csv."
        )
    return messages


def volatility_assumptions(thresholds: pd.DataFrame) -> List[str]:
    return [
        "Volatility decomposition uses Binance BTCUSDT 1-minute candles aligned to each scored Kalshi forecast hour.",
        "Hourly realized volatility is defined as sqrt(sum(log(close/open)^2)) across the 60 one-minute Binance bars inside the forecast hour.",
        (
            f"Real-time classification uses the trailing {ROLLING_WINDOW_HOURS}-hour "
            f"({ROLLING_WINDOW_MINUTES}-minute) Binance window available at each scored hour."
        ),
        (
            "Rolling cutoff summaries in this run: "
            f"q10 {format_threshold_range(thresholds, 'Low Volatility Extreme')}; "
            f"q25 {format_threshold_range(thresholds, 'Low Volatility')}; "
            f"q75 {format_threshold_range(thresholds, 'High Volatility')}; "
            f"q90 {format_threshold_range(thresholds, 'High Volatility Extreme')}."
        ),
        "Low Volatility includes hours with realized volatility <= trailing q25; Standard Volatility includes trailing q25 < realized volatility < trailing q75; High Volatility includes realized volatility >= trailing q75.",
    ]


def dark_mode_style_block() -> str:
    return """<style>
    :root {
      color-scheme: dark;
      --bg: #071018;
      --panel: #0d1722;
      --panel-2: #122130;
      --ink: #e6eef8;
      --muted: #9db0c3;
      --line: #223548;
      --soft: #102030;
      --blue: #6cb6ff;
      --teal: #4fd1c5;
      --amber: #f6ad55;
      --link: #8cc8ff;
    }
    html {
      background: var(--bg);
    }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top, rgba(76, 132, 196, 0.18), transparent 32%),
        linear-gradient(180deg, #08111a 0%, #071018 100%);
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 30px;
      letter-spacing: 0;
      color: var(--ink);
    }
    h2 {
      margin: 34px 0 12px;
      font-size: 18px;
      letter-spacing: 0;
      color: var(--ink);
    }
    p, li {
      color: var(--muted);
      line-height: 1.45;
    }
    a {
      color: var(--link);
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 24px 0;
    }
    .metric-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 14px 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)), var(--panel);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .metric-card span {
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }
    .metric-card strong {
      display: block;
      font-size: 24px;
      letter-spacing: 0;
      color: var(--ink);
    }
    .chart-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }
    .chart {
      width: 100%;
      min-height: 280px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
    }
    .chart-title {
      font-size: 16px;
      font-weight: 650;
      fill: var(--ink);
    }
    .axis {
      stroke: #35516c;
      stroke-width: 1;
    }
    .diag {
      fill: none;
      stroke: #6682a0;
      stroke-width: 1.5;
      stroke-dasharray: 5 5;
    }
    .series {
      fill: none;
      stroke: var(--blue);
      stroke-width: 3;
    }
    circle {
      fill: var(--blue);
    }
    rect {
      fill: var(--teal);
    }
    .axis-label, .tick {
      fill: var(--muted);
      font-size: 12px;
    }
    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
      background: transparent;
    }
    .data-table th, .data-table td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
    }
    .data-table td {
      color: var(--ink);
    }
    .data-table th {
      background: var(--panel-2);
      color: var(--ink);
      font-weight: 650;
    }
    .note {
      color: var(--muted);
      font-size: 13px;
    }
    code {
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 2px 5px;
      color: var(--ink);
    }
  </style>"""


def apply_dark_mode_html(document: str) -> str:
    updated = re.sub(r"<style>.*?</style>", dark_mode_style_block(), document, count=1, flags=re.S)
    replacements = {
        '"font":{"color":"#2a3f5f"}': '"font":{"color":"#e6eef8"}',
        '"paper_bgcolor":"white"': '"paper_bgcolor":"#0d1722"',
        '"plot_bgcolor":"white"': '"plot_bgcolor":"#0d1722"',
        '"bgcolor":"white"': '"bgcolor":"#0d1722"',
        '"lakecolor":"white"': '"lakecolor":"#0d1722"',
        '"landcolor":"white"': '"landcolor":"#0d1722"',
        '"backgroundcolor":"white"': '"backgroundcolor":"#0d1722"',
        '"line":{"color":"white","width":0.5}': '"line":{"color":"#223548","width":0.5}',
        '"fill":{"color":"#EBF0F8"}': '"fill":{"color":"#102030"}',
        '"fill":{"color":"#C8D4E3"}': '"fill":{"color":"#122130"}',
        '"gridcolor":"#EBF0F8"': '"gridcolor":"#223548"',
        '"gridcolor":"#DFE8F3"': '"gridcolor":"#223548"',
        '"gridcolor":"#C8D4E3"': '"gridcolor":"#223548"',
        '"linecolor":"#EBF0F8"': '"linecolor":"#35516c"',
        '"linecolor":"#C8D4E3"': '"linecolor":"#35516c"',
        '"linecolor":"#A2B1C6"': '"linecolor":"#35516c"',
        '"zerolinecolor":"#EBF0F8"': '"zerolinecolor":"#35516c"',
        '"subunitcolor":"#C8D4E3"': '"subunitcolor":"#35516c"',
    }
    for src, dst in replacements.items():
        updated = updated.replace(src, dst)
    return updated


def retitle_summary_html(document: str, *, heading: str, output_folder_name: str) -> str:
    updated = document.replace("<title>Model K Summary</title>", f"<title>{html.escape(heading)}</title>")
    updated = updated.replace(
        "<h1>Model K: Kalshi Probability Evaluation</h1>",
        f"<h1>{html.escape(heading)}</h1>",
    )
    updated = updated.replace(
        "Individual CSV outputs are saved next to this report in <code>Model_K_outputs</code>:",
        f"Individual CSV outputs are saved next to this report in <code>{html.escape(output_folder_name)}</code>:",
    )
    return apply_dark_mode_html(updated)


def retitle_time_bucket_html(document: str, *, heading: str) -> str:
    updated = document.replace(
        "<title>Model K Market Minute Bucket Summary</title>",
        f"<title>{html.escape(heading)}</title>",
    )
    updated = updated.replace(
        "<h1>Model K: Market Minute Buckets</h1>",
        f"<h1>{html.escape(heading)}</h1>",
    )
    return apply_dark_mode_html(updated)


def segment_output_dir(output_root: Path, segment_name: str) -> Path:
    return output_root / segment_name


def build_segment_outputs(
    *,
    segment: Dict[str, str],
    output_root: Path,
    forecasts: pd.DataFrame,
    outcomes: pd.DataFrame,
    raw: pd.DataFrame,
    unmatched: pd.DataFrame,
    hourly_market_states: pd.DataFrame,
    thresholds: pd.DataFrame,
    calibration_bins: int,
    classification_threshold: float,
    skip_binance_audit: bool,
) -> Dict[str, Any]:
    flag = segment["flag"]
    segment_name = segment["name"]
    segment_rule = segment["rule"]
    event_ids = set(hourly_market_states.loc[hourly_market_states[flag], "event_ticker"].astype(str))

    forecast_segment = enrich_rows_with_market_state(
        forecasts[forecasts["event_ticker"].astype(str).isin(event_ids)].copy(),
        hourly_market_states,
    )
    raw_segment = enrich_rows_with_market_state(
        raw[raw["event_ticker"].astype(str).isin(event_ids)].copy(),
        hourly_market_states,
    )
    unmatched_segment = enrich_rows_with_market_state(
        unmatched[unmatched["event_ticker"].astype(str).isin(event_ids)].copy(),
        hourly_market_states,
    )

    coverage = build_outcome_join_coverage(forecast_segment, raw_segment, unmatched_segment)
    resolution_mismatches = pd.DataFrame() if skip_binance_audit else build_resolution_mismatches(raw_segment)
    metrics = build_metrics_summary(raw_segment, threshold=classification_threshold)
    decomposition = build_brier_decomposition(raw_segment, bins=calibration_bins)
    calibration, expanded_summary = expanded_calibration_error(raw_segment, bins=calibration_bins)
    sharpness = build_sharpness(raw_segment)
    time_bucket_metrics, time_bucket_accuracy, time_bucket_brier, time_bucket_calibration = build_time_bucket_outputs(
        raw_segment,
        calibration_bins=calibration_bins,
        threshold=classification_threshold,
    )

    segment_dir = segment_output_dir(output_root, segment_name)
    segment_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(segment_dir / "kalshi_official_outcomes_long.csv", index=False)

    write_individual_metric_files(
        output_dir=segment_dir,
        raw=raw_segment,
        metrics=metrics,
        decomposition=decomposition,
        calibration=calibration,
        expanded_error=calibration,
        expanded_error_summary=expanded_summary,
        sharpness=sharpness,
        coverage=coverage,
        unmatched=unmatched_segment,
        resolution_mismatches=resolution_mismatches,
        time_bucket_metrics=time_bucket_metrics,
        time_bucket_accuracy=time_bucket_accuracy,
        time_bucket_brier=time_bucket_brier,
        time_bucket_calibration=time_bucket_calibration,
    )

    all_assumptions = ASSUMPTIONS + tuple(volatility_assumptions(thresholds))
    (segment_dir / "model_k_assumptions.txt").write_text("\n".join(all_assumptions) + "\n", encoding="utf-8")

    messages = build_segment_messages(
        segment_name=segment_name,
        segment_rule=segment_rule,
        coverage=coverage,
        resolution_mismatches=resolution_mismatches,
        segment_event_count=len(event_ids),
        total_event_count=int(hourly_market_states["event_ticker"].nunique()),
        thresholds=thresholds,
    )

    summary_heading = f"Model K RT: {segment_name}"
    summary_html = build_summary_html(
        raw=raw_segment,
        metrics=metrics,
        decomposition=decomposition,
        calibration=calibration,
        expanded_summary=expanded_summary,
        sharpness=sharpness,
        coverage=coverage,
        resolution_mismatches=resolution_mismatches,
        messages=messages,
        include_coverage=False,
    )
    (segment_dir / "model_k_summary.html").write_text(
        retitle_summary_html(summary_html, heading=summary_heading, output_folder_name=segment_dir.name),
        encoding="utf-8",
    )

    summary_with_coverage_html = build_summary_html(
        raw=raw_segment,
        metrics=metrics,
        decomposition=decomposition,
        calibration=calibration,
        expanded_summary=expanded_summary,
        sharpness=sharpness,
        coverage=coverage,
        resolution_mismatches=resolution_mismatches,
        messages=messages,
        include_coverage=True,
    )
    (segment_dir / "model_k_summary_with_coverage.html").write_text(
        retitle_summary_html(
            summary_with_coverage_html,
            heading=f"Model K RT: {segment_name} With Coverage",
            output_folder_name=segment_dir.name,
        ),
        encoding="utf-8",
    )

    time_bucket_html = build_time_bucket_summary_html(
        time_bucket_metrics=time_bucket_metrics,
        time_bucket_accuracy=time_bucket_accuracy,
        time_bucket_brier=time_bucket_brier,
        time_bucket_calibration=time_bucket_calibration,
    )
    (segment_dir / "time_bucket_summary.html").write_text(
        retitle_time_bucket_html(
            time_bucket_html,
            heading=f"Model K RT: {segment_name} Market Minute Buckets",
        ),
        encoding="utf-8",
    )

    overall_metrics = metrics.loc[metrics["segment"] == "overall"].iloc[0]
    return {
        "segment_name": segment_name,
        "segment_rule": segment_rule,
        "segment_dir": segment_dir,
        "n_hourly_markets": len(event_ids),
        "n_forecasts": int(overall_metrics["n_forecasts"]),
        "n_event_contracts": int(overall_metrics["n_event_contracts"]),
        "brier_score": overall_metrics["brier_score"],
        "log_loss": overall_metrics["log_loss"],
    }


def build_index_html(
    *,
    thresholds: pd.DataFrame,
    segment_rows: pd.DataFrame,
) -> str:
    threshold_table = thresholds.to_html(index=False, border=0, classes="data-table", escape=True, justify="left")
    link_items = []
    for row in segment_rows.to_dict(orient="records"):
        segment_name = str(row["segment_name"])
        summary_href = f"{segment_name}/model_k_summary.html"
        summary_with_coverage_href = f"{segment_name}/model_k_summary_with_coverage.html"
        time_bucket_href = f"{segment_name}/time_bucket_summary.html"
        link_items.append(
            "<li>"
            f"<strong>{html.escape(segment_name)}</strong>: "
            f"<a href='{html.escape(summary_href)}'>summary</a>, "
            f"<a href='{html.escape(summary_with_coverage_href)}'>summary with coverage</a>, "
            f"<a href='{html.escape(time_bucket_href)}'>time bucket summary</a>"
            "</li>"
        )
    segment_table = segment_rows.drop(columns=["segment_dir"]).to_html(
        index=False,
        border=0,
        classes="data-table",
        escape=True,
        justify="left",
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model K Real-Time Volatility Decomposition</title>
  {dark_mode_style_block()}
</head>
<body>
<main>
  <h1>Model K Real-Time Volatility Decomposition</h1>
  <p>
    This directory contains five segment-specific copies of the usual Model K outputs, split by
    hourly BTC realized volatility measured from Binance 1-minute candles and classified with a
    trailing 72-hour real-time window.
  </p>

  <h2>Real-Time Volatility Definition</h2>
  <p>
    Hourly realized volatility is <code>sqrt(sum(log(close/open)^2))</code> over the 60 Binance
    1-minute bars in the forecast hour. Each scored hour is then compared against the trailing
    <code>{ROLLING_WINDOW_HOURS}</code> Binance hours (<code>{ROLLING_WINDOW_MINUTES}</code> minutes)
    available at that time, so the segment cutoffs move through time instead of using one
    full-sample threshold for the entire run.
  </p>

  <h2>Threshold Summaries</h2>
  <div class="table-wrap">{threshold_table}</div>

  <h2>Segment Output Folders</h2>
  <ul>
    {''.join(link_items)}
  </ul>
  <div class="table-wrap">{segment_table}</div>

  <h2>Shared Root Files</h2>
  <ul>
    <li><code>binance_1m_klines.csv</code>: cached Binance minute data used for the decomposition.</li>
    <li><code>binance_hourly_realized_volatility.csv</code>: realized volatility per Binance hour plus rolling real-time thresholds.</li>
    <li><code>hourly_market_volatility_segments.csv</code>: each scored Kalshi hour with its realized volatility, rolling thresholds, and segment flags.</li>
    <li><code>segment_thresholds.csv</code>: summary statistics for the rolling percentile cutoffs used in this run.</li>
    <li><code>all_scored_rows_with_volatility.csv</code>: scored Model K rows with attached hourly volatility metadata.</li>
    <li><code>kalshi_official_outcomes_long.csv</code>: official contract outcomes used for all segment runs.</li>
  </ul>
</main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    forecasts = load_kalshi_price_outputs(args.kalshi_price_dir.resolve())
    settlements = load_settlements(args.settlement_csv.resolve())
    outcomes, _messages = build_kalshi_reality_outcomes(
        forecasts=forecasts,
        settlements=settlements,
        output_dir=output_root,
    )
    raw, unmatched = attach_outcomes(forecasts, outcomes)

    required_start = raw["event_datetime_utc"].min() - pd.Timedelta(hours=1 + ROLLING_WINDOW_HOURS)
    required_end = raw["event_datetime_utc"].max()
    cache_path = resolve_binance_cache_path(output_root, args.binance_minute_cache_csv)
    binance_minutes = load_or_fetch_binance_minutes(
        cache_path=cache_path,
        start_utc=required_start,
        end_utc=required_end,
        refresh_cache=args.refresh_binance_cache,
    )
    hourly_volatility = compute_hourly_realized_volatility(binance_minutes)
    hourly_volatility = add_realtime_window_thresholds(hourly_volatility)
    hourly_market_states = build_hourly_market_state_table(raw, hourly_volatility)
    thresholds = thresholds_table(hourly_market_states)

    raw_with_state = enrich_rows_with_market_state(raw, hourly_market_states)
    unmatched_with_state = enrich_rows_with_market_state(unmatched, hourly_market_states)
    forecasts_with_state = enrich_rows_with_market_state(forecasts, hourly_market_states)

    hourly_volatility.to_csv(output_root / "binance_hourly_realized_volatility.csv", index=False)
    hourly_market_states.to_csv(output_root / "hourly_market_volatility_segments.csv", index=False)
    thresholds.to_csv(output_root / "segment_thresholds.csv", index=False)
    raw_with_state.to_csv(output_root / "all_scored_rows_with_volatility.csv", index=False)
    unmatched_with_state.to_csv(output_root / "all_unmatched_rows_with_volatility.csv", index=False)
    forecasts_with_state.to_csv(output_root / "all_forecast_rows_with_volatility.csv", index=False)

    segment_summaries: List[Dict[str, Any]] = []
    for segment in SEGMENTS:
        segment_summaries.append(
            build_segment_outputs(
                segment=segment,
                output_root=output_root,
                forecasts=forecasts,
                outcomes=outcomes,
                raw=raw,
                unmatched=unmatched,
                hourly_market_states=hourly_market_states,
                thresholds=thresholds,
                calibration_bins=args.calibration_bins,
                classification_threshold=args.classification_threshold,
                skip_binance_audit=args.skip_binance_audit,
            )
        )

    segment_rows = pd.DataFrame(segment_summaries)
    segment_rows["model_k_summary_html"] = segment_rows["segment_name"].map(
        lambda value: f"{value}/model_k_summary.html"
    )
    segment_rows["model_k_summary_with_coverage_html"] = segment_rows["segment_name"].map(
        lambda value: f"{value}/model_k_summary_with_coverage.html"
    )
    segment_rows["time_bucket_summary_html"] = segment_rows["segment_name"].map(
        lambda value: f"{value}/time_bucket_summary.html"
    )
    segment_rows.to_csv(output_root / "segment_output_summary.csv", index=False)

    index_html = build_index_html(
        thresholds=thresholds,
        segment_rows=segment_rows,
    )
    (output_root / "index.html").write_text(index_html, encoding="utf-8")

    print(f"Model K real-time volatility decomposition outputs saved to: {output_root}")
    print(f"Index report: {output_root / 'index.html'}")
    for row in segment_summaries:
        print(
            f"{row['segment_name']}: hourly_markets={row['n_hourly_markets']}, "
            f"forecasts={row['n_forecasts']}, brier_score={row['brier_score']:.6f}"
        )


if __name__ == "__main__":
    main()
