from __future__ import annotations

import argparse
import html
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model_K.Model_K import (  # noqa: E402
    ASSUMPTIONS,
    build_brier_decomposition,
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
    attach_outcomes,
    build_kalshi_reality_outcomes,
    build_metrics_summary,
    write_individual_metric_files,
)


SYMBOL = "BTCUSDT"
BINANCE_BASE = "https://api.binance.com"
BINANCE_KLINES = "/api/v3/klines"

SEGMENTS: List[Dict[str, str]] = [
    {
        "name": "Low Volatility",
        "flag": "is_low_volatility",
        "rule": "Hourly realized volatility at or below the 25th percentile of scored hourly markets.",
    },
    {
        "name": "Standard Volatility",
        "flag": "is_standard_volatility",
        "rule": "Hourly realized volatility strictly between the 25th and 75th percentiles of scored hourly markets.",
    },
    {
        "name": "High Volatility",
        "flag": "is_high_volatility",
        "rule": "Hourly realized volatility at or above the 75th percentile of scored hourly markets.",
    },
    {
        "name": "Low Volatility Extreme",
        "flag": "is_low_volatility_extreme",
        "rule": "Hourly realized volatility at or below the 10th percentile of scored hourly markets.",
    },
    {
        "name": "High Volatility Extreme",
        "flag": "is_high_volatility_extreme",
        "rule": "Hourly realized volatility at or above the 90th percentile of scored hourly markets.",
    },
]


def repo_root_from_script() -> Path:
    return REPO_ROOT


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Run Model K outputs separately across hourly Kalshi BTC market states "
            "defined by Binance realized volatility."
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
        default=Path(__file__).resolve().parent / "Model_K_Volatility_Decomposition_outputs",
        help="Root directory for volatility decomposition outputs.",
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

    quantiles = merged["realized_volatility"].quantile([0.10, 0.25, 0.75, 0.90])
    q10 = float(quantiles.loc[0.10])
    q25 = float(quantiles.loc[0.25])
    q75 = float(quantiles.loc[0.75])
    q90 = float(quantiles.loc[0.90])

    merged["is_low_volatility"] = merged["realized_volatility"] <= q25
    merged["is_standard_volatility"] = (merged["realized_volatility"] > q25) & (merged["realized_volatility"] < q75)
    merged["is_high_volatility"] = merged["realized_volatility"] >= q75
    merged["is_low_volatility_extreme"] = merged["realized_volatility"] <= q10
    merged["is_high_volatility_extreme"] = merged["realized_volatility"] >= q90
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
    merged["q10_realized_volatility"] = q10
    merged["q25_realized_volatility"] = q25
    merged["q75_realized_volatility"] = q75
    merged["q90_realized_volatility"] = q90
    return merged.sort_values("forecast_hour_start_utc").reset_index(drop=True)


def thresholds_table(hourly_market_states: pd.DataFrame) -> pd.DataFrame:
    first = hourly_market_states.iloc[0]
    rows = [
        {
            "threshold_name": "Low Volatility Extreme",
            "percentile": 0.10,
            "realized_volatility_cutoff": float(first["q10_realized_volatility"]),
            "rule": "realized_volatility <= q10",
        },
        {
            "threshold_name": "Low Volatility",
            "percentile": 0.25,
            "realized_volatility_cutoff": float(first["q25_realized_volatility"]),
            "rule": "realized_volatility <= q25",
        },
        {
            "threshold_name": "High Volatility",
            "percentile": 0.75,
            "realized_volatility_cutoff": float(first["q75_realized_volatility"]),
            "rule": "realized_volatility >= q75",
        },
        {
            "threshold_name": "High Volatility Extreme",
            "percentile": 0.90,
            "realized_volatility_cutoff": float(first["q90_realized_volatility"]),
            "rule": "realized_volatility >= q90",
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
    q10 = float(thresholds.loc[thresholds["threshold_name"] == "Low Volatility Extreme", "realized_volatility_cutoff"].iloc[0])
    q25 = float(thresholds.loc[thresholds["threshold_name"] == "Low Volatility", "realized_volatility_cutoff"].iloc[0])
    q75 = float(thresholds.loc[thresholds["threshold_name"] == "High Volatility", "realized_volatility_cutoff"].iloc[0])
    q90 = float(thresholds.loc[thresholds["threshold_name"] == "High Volatility Extreme", "realized_volatility_cutoff"].iloc[0])

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
            f"Scored-hour cutoffs for this run: q10={q10:.8f}, q25={q25:.8f}, "
            f"q75={q75:.8f}, q90={q90:.8f}."
        ),
        (
            "High Volatility is interpreted as the top quartile of scored hours "
            "(realized volatility >= q75) because the original request's 'highest 75%' "
            "conflicts with the 25%-75% Standard Volatility band."
        ),
    ]
    if len(resolution_mismatches):
        messages.append(
            f"Found {len(resolution_mismatches)} Binance audit mismatch row(s); see resolution_mismatches.csv."
        )
    return messages


def volatility_assumptions(thresholds: pd.DataFrame) -> List[str]:
    q10 = float(thresholds.loc[thresholds["threshold_name"] == "Low Volatility Extreme", "realized_volatility_cutoff"].iloc[0])
    q25 = float(thresholds.loc[thresholds["threshold_name"] == "Low Volatility", "realized_volatility_cutoff"].iloc[0])
    q75 = float(thresholds.loc[thresholds["threshold_name"] == "High Volatility", "realized_volatility_cutoff"].iloc[0])
    q90 = float(thresholds.loc[thresholds["threshold_name"] == "High Volatility Extreme", "realized_volatility_cutoff"].iloc[0])
    return [
        "Volatility decomposition uses Binance BTCUSDT 1-minute candles aligned to each scored Kalshi forecast hour.",
        "Hourly realized volatility is defined as sqrt(sum(log(close/open)^2)) across the 60 one-minute Binance bars inside the forecast hour.",
        f"Low/standard/high cutoffs in this run are q25={q25:.8f} and q75={q75:.8f}; extreme cutoffs are q10={q10:.8f} and q90={q90:.8f}.",
        "Low Volatility includes hours with realized volatility <= q25; Standard Volatility includes q25 < realized volatility < q75; High Volatility includes realized volatility >= q75.",
        "High Volatility is interpreted as the top quartile rather than the literal 'highest 75%' because the latter conflicts with the requested Standard Volatility range.",
    ]


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
    return updated


def retitle_time_bucket_html(document: str, *, heading: str) -> str:
    updated = document.replace(
        "<title>Model K Market Minute Bucket Summary</title>",
        f"<title>{html.escape(heading)}</title>",
    )
    updated = updated.replace(
        "<h1>Model K: Market Minute Buckets</h1>",
        f"<h1>{html.escape(heading)}</h1>",
    )
    return updated


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

    summary_heading = f"Model K: {segment_name}"
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
            heading=f"Model K: {segment_name} With Coverage",
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
            heading=f"Model K: {segment_name} Market Minute Buckets",
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
  <title>Model K Volatility Decomposition</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #657085;
      --line: #d7dde8;
      --soft: #f5f7fb;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
    }}
    main {{
      max-width: 1160px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
    }}
    h2 {{
      margin: 34px 0 12px;
      font-size: 18px;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.45;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 18px;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
    }}
    .data-table th {{
      background: var(--soft);
      color: #334155;
      font-weight: 650;
    }}
    code {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 2px 5px;
    }}
  </style>
</head>
<body>
<main>
  <h1>Model K Volatility Decomposition</h1>
  <p>
    This directory contains five segment-specific copies of the usual Model K outputs, split by
    hourly BTC realized volatility measured from Binance 1-minute candles aligned to each scored
    Kalshi forecast hour.
  </p>

  <h2>Realized Volatility Definition</h2>
  <p>
    Hourly realized volatility is <code>sqrt(sum(log(close/open)^2))</code> over the 60 Binance
    1-minute bars in the forecast hour. Segment thresholds are computed on unique scored hourly
    Kalshi markets, not on individual minute-level forecast rows.
  </p>

  <h2>Thresholds</h2>
  <div class="table-wrap">{threshold_table}</div>

  <h2>Segment Output Folders</h2>
  <ul>
    {''.join(link_items)}
  </ul>
  <div class="table-wrap">{segment_table}</div>

  <h2>Shared Root Files</h2>
  <ul>
    <li><code>binance_1m_klines.csv</code>: cached Binance minute data used for the decomposition.</li>
    <li><code>binance_hourly_realized_volatility.csv</code>: realized volatility per Binance hour.</li>
    <li><code>hourly_market_volatility_segments.csv</code>: each scored Kalshi hour with its realized volatility and segment flags.</li>
    <li><code>segment_thresholds.csv</code>: the quantile cutoffs used in this run.</li>
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

    required_start = raw["event_datetime_utc"].min() - pd.Timedelta(hours=1)
    required_end = raw["event_datetime_utc"].max()
    cache_path = resolve_binance_cache_path(output_root, args.binance_minute_cache_csv)
    binance_minutes = load_or_fetch_binance_minutes(
        cache_path=cache_path,
        start_utc=required_start,
        end_utc=required_end,
        refresh_cache=args.refresh_binance_cache,
    )
    hourly_volatility = compute_hourly_realized_volatility(binance_minutes)
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

    print(f"Model K volatility decomposition outputs saved to: {output_root}")
    print(f"Index report: {output_root / 'index.html'}")
    for row in segment_summaries:
        print(
            f"{row['segment_name']}: hourly_markets={row['n_hourly_markets']}, "
            f"forecasts={row['n_forecasts']}, brier_score={row['brier_score']:.6f}"
        )


if __name__ == "__main__":
    main()
