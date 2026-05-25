# helpers/features.py
from __future__ import annotations

import numpy as np
import pandas as pd


def timeframe_to_horizon(timeframe: str) -> int:
    mapping = {
        "Daily": 1,
        "Weekly": 5,
        "Monthly": 21,
    }
    return mapping.get(timeframe, 1)


def build_ml_dataset(price_df: pd.DataFrame, timeframe: str):
    horizon = timeframe_to_horizon(timeframe)
    df = price_df.copy().sort_values("Date").reset_index(drop=True)

    # 只使用當前與過去資料，不使用未來資料做特徵
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_3"] = df["Close"].pct_change(3)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    df["mom_20"] = df["Close"] / df["SMA_20"] - 1
    df["mom_50"] = df["Close"] / df["SMA_50"] - 1
    df["mom_200"] = df["Close"] / df["SMA_200"] - 1

    df["vol_ratio"] = df["Volume"] / df["Volume_MA_20"]
    df["rsi_14"] = df["RSI_14"] / 100.0
    df["vol_5"] = df["Volatility_5"]
    df["vol_20"] = df["Volatility_20"]
    df["hl_spread"] = df["HL_Spread"]

    regime_map = {"Bull": 1, "Sideways": 0, "Bear": -1}
    df["regime_code"] = df["Regime"].map(regime_map)

    # 目標變數：未來 horizon 天報酬是否為正
    df["future_return"] = df["Close"].shift(-horizon) / df["Close"] - 1
    df["target"] = np.where(
        df["future_return"].notna(),
        (df["future_return"] > 0).astype(int),
        np.nan,
    )

    feature_cols = [
        "ret_1",
        "ret_3",
        "ret_5",
        "ret_10",
        "mom_20",
        "mom_50",
        "mom_200",
        "vol_ratio",
        "rsi_14",
        "vol_5",
        "vol_20",
        "hl_spread",
        "regime_code",
    ]

    latest_pool = df.dropna(subset=feature_cols).copy()
    if latest_pool.empty:
        raise ValueError("No valid latest feature row available.")

    latest_snapshot = latest_pool.iloc[-1]
    latest_features = latest_pool.iloc[[-1]][feature_cols]

    labeled = df.dropna(subset=feature_cols + ["future_return", "target"]).copy()
    labeled["target"] = labeled["target"].astype(int)

    return labeled, latest_features, latest_snapshot, feature_cols
