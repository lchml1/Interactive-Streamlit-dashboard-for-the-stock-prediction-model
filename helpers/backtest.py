# helpers/backtest.py
from __future__ import annotations

import numpy as np
import pandas as pd


def build_backtest(
    dataset: pd.DataFrame,
    proba_up,
    confidence_threshold: float = 0.60,
    volatility_threshold: float = 0.40,
):
    bt = dataset.copy().reset_index(drop=True)

    bt["proba_up"] = proba_up
    bt = bt.dropna(subset=["proba_up"]).copy()

    bt["confidence"] = np.where(
        bt["proba_up"] >= 0.5,
        bt["proba_up"],
        1 - bt["proba_up"],
    )

    # Long-or-cash 策略：只在看漲且信心、波動率都過關時進場
    bt["signal"] = (
        (bt["proba_up"] >= 0.5)
        & (bt["confidence"] >= confidence_threshold)
        & (bt["Volatility_20"] <= volatility_threshold)
    ).astype(int)

    bt["strategy_return"] = bt["signal"] * bt["future_return"]
    bt["buy_hold_return"] = bt["future_return"]

    bt["cum_strategy"] = (1 + bt["strategy_return"].fillna(0)).cumprod()
    bt["cum_buy_hold"] = (1 + bt["buy_hold_return"].fillna(0)).cumprod()

    bt["strategy_drawdown"] = bt["cum_strategy"] / bt["cum_strategy"].cummax() - 1
    bt["buy_hold_drawdown"] = bt["cum_buy_hold"] / bt["cum_buy_hold"].cummax() - 1

    trades = bt[bt["signal"] == 1].copy()
    trade_win_rate = (trades["strategy_return"] > 0).mean() if not trades.empty else 0.0

    stats = {
        "trade_count": int(trades.shape[0]),
        "trade_win_rate": float(trade_win_rate),
        "total_strategy_return": float(bt["cum_strategy"].iloc[-1] - 1),
        "total_buy_hold_return": float(bt["cum_buy_hold"].iloc[-1] - 1),
        "max_drawdown": float(bt["strategy_drawdown"].min()),
    }

    return bt, stats


def build_paper_trading_summary(backtest_df: pd.DataFrame, initial_capital: float = 10_000):
    final_capital = float(initial_capital * backtest_df["cum_strategy"].iloc[-1])
    net_profit = final_capital - initial_capital
    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "net_profit": net_profit,
        "return_pct": net_profit / initial_capital,
    }
