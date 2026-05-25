# helpers/data.py
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data.rename_axis("Date").reset_index()

    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    data["Date"] = pd.to_datetime(data["Date"])
    return data


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def detect_regime(df: pd.DataFrame) -> pd.Series:
    bull = (df["SMA_50"] > df["SMA_200"]) & (df["Close"] > df["SMA_50"])
    bear = (df["SMA_50"] < df["SMA_200"]) & (df["Close"] < df["SMA_50"])

    regime = np.where(bull, "Bull", np.where(bear, "Bear", "Sideways"))
    return pd.Series(regime, index=df.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("Date").reset_index(drop=True)

    out["Return_1"] = out["Close"].pct_change(1)
    out["Return_5"] = out["Close"].pct_change(5)
    out["Return_10"] = out["Close"].pct_change(10)

    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["SMA_200"] = out["Close"].rolling(200).mean()

    out["Volatility_5"] = out["Return_1"].rolling(5).std() * np.sqrt(252)
    out["Volatility_20"] = out["Return_1"].rolling(20).std() * np.sqrt(252)

    out["Volume_MA_20"] = out["Volume"].rolling(20).mean()
    out["RSI_14"] = compute_rsi(out["Close"], 14)
    out["HL_Spread"] = (out["High"] - out["Low"]) / out["Close"].replace(0, np.nan)

    out["Regime"] = detect_regime(out)
    return out


def volatility_label(vol: float) -> str:
    if vol < 0.20:
        return "Low"
    if vol < 0.35:
        return "Moderate"
    if vol < 0.55:
        return "High"
    return "Very High"
