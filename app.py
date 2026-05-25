# app.py
import io
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots

from helpers.data import fetch_stock_data, add_indicators, volatility_label
from helpers.features import build_ml_dataset, timeframe_to_horizon
from helpers.modeling import train_and_evaluate_models
from helpers.backtest import build_backtest, build_paper_trading_summary

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Quant Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Dark UI ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
        color: #e5e7eb;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        max-width: 1500px;
    }
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 0.9rem 1rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    .small-note {
        color: #94a3b8;
        font-size: 0.92rem;
    }
    .panel {
        background: rgba(17, 24, 39, 0.66);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 1rem 1rem 0.6rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Quant Stock Prediction Dashboard")
st.caption("Time-series classification + backtesting + risk filters + trading signals")

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("控制面板")

    ticker = st.text_input(
        "股票代號",
        value="AAPL",
        help="例如：AAPL、TSLA、NVDA、MSFT",
    ).upper().strip()

    timeframe = st.selectbox(
        "預測周期",
        options=["Daily", "Weekly", "Monthly"],
        index=0,
    )

    history_years = st.slider(
        "回看歷史年數",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
    )

    confidence_threshold = st.slider(
        "信心門檻",
        min_value=0.50,
        max_value=0.95,
        value=0.60,
        step=0.01,
    )

    volatility_threshold = st.slider(
        "年化波動率上限",
        min_value=0.15,
        max_value=0.80,
        value=0.40,
        step=0.01,
        help="若市場波動超過此值，策略會避免進場。",
    )

    show_last_n = st.slider(
        "圖表顯示最近 K 線數",
        min_value=90,
        max_value=400,
        value=220,
        step=10,
    )

    retrain = st.button("重新訓練模型")

@st.cache_data(show_spinner=False)
def load_data(ticker_symbol: str, years: int) -> pd.DataFrame:
    return fetch_stock_data(ticker_symbol, period=f"{years}y")

def make_price_chart(df: pd.DataFrame, ticker_symbol: str, last_n: int = 220):
    view = df.tail(last_n).copy()
    colors = np.where(view["Close"].diff().fillna(0) >= 0, "#14b8a6", "#f43f5e")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.58, 0.18, 0.24],
    )

    fig.add_trace(
        go.Candlestick(
            x=view["Date"],
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            name="Candlestick",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    for col, color in [("SMA_20", "#60a5fa"), ("SMA_50", "#f59e0b"), ("SMA_200", "#c084fc")]:
        fig.add_trace(
            go.Scatter(
                x=view["Date"],
                y=view[col],
                mode="lines",
                name=col,
                line=dict(width=1.8, color=color),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=view["Date"],
            y=view["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.6,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=view["Date"],
            y=view["RSI_14"],
            mode="lines",
            name="RSI(14)",
            line=dict(width=1.8, color="#38bdf8"),
        ),
        row=3,
        col=1,
    )

    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=860,
        margin=dict(l=20, r=20, t=40, b=20),
        title=f"{ticker_symbol} Price / Volume / RSI",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])

    return fig

def make_confusion_matrix_figure(cm: np.ndarray):
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        x=["Pred Down", "Pred Up"],
        y=["Actual Down", "Actual Up"],
        aspect="auto",
    )
    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def make_feature_importance_figure(fi_df: pd.DataFrame):
    top = fi_df.head(12).sort_values("importance", ascending=True)
    fig = px.bar(
        top,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Tealgrn",
    )
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=30, b=20))
    return fig

def make_backtest_figure(bt: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt["Date"],
            y=bt["cum_strategy"],
            mode="lines",
            name="Strategy",
            line=dict(color="#2dd4bf", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt["Date"],
            y=bt["cum_buy_hold"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#f59e0b", width=2.2),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Cumulative Returns",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Growth of $1",
    )
    return fig

def make_drawdown_figure(bt: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bt["Date"],
            y=bt["strategy_drawdown"],
            mode="lines",
            name="Strategy Drawdown",
            fill="tozeroy",
            line=dict(color="#f43f5e", width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt["Date"],
            y=bt["buy_hold_drawdown"],
            mode="lines",
            name="Buy & Hold Drawdown",
            fill="tozeroy",
            line=dict(color="#60a5fa", width=1.4),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        title="Drawdown",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_tickformat=".1%",
    )
    return fig

# ---------- Main Pipeline ----------
try:
    raw_df = load_data(ticker, history_years)
    price_df = add_indicators(raw_df)

    dataset, latest_features, latest_snapshot, feature_cols = build_ml_dataset(
        price_df=price_df,
        timeframe=timeframe,
    )

    if len(dataset) < 160:
        st.error("資料量太少，建議把歷史年數拉高，至少讓模型有足夠時間序列樣本。")
        st.stop()

    model_bundle = train_and_evaluate_models(
        dataset=dataset,
        feature_cols=feature_cols,
    )

    # 最新一期預測：只用最後一筆可用特徵做推論
    latest_proba_up = float(model_bundle["model"].predict_proba(latest_features)[0][1])
    latest_direction = "UP" if latest_proba_up >= 0.5 else "DOWN"
    latest_confidence = max(latest_proba_up, 1 - latest_proba_up)
    latest_vol = float(latest_snapshot["Volatility_20"])
    latest_close = float(latest_snapshot["Close"])
    latest_regime = str(latest_snapshot["Regime"])

    backtest_df, backtest_stats = build_backtest(
        dataset=dataset,
        proba_up=model_bundle["best_oof_proba"],
        confidence_threshold=confidence_threshold,
        volatility_threshold=volatility_threshold,
    )

    paper_stats = build_paper_trading_summary(
        backtest_df=backtest_df,
        initial_capital=10_000,
    )

except Exception as e:
    st.error(f"資料或模型處理失敗：{e}")
    st.stop()

# ---------- Prediction Panel ----------
st.subheader("Prediction Panel")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Prediction", latest_direction)
m2.metric("Confidence", f"{latest_confidence * 100:.2f}%")
m3.metric("Prob(UP)", f"{latest_proba_up * 100:.2f}%")
m4.metric("Latest Price", f"${latest_close:,.2f}")
m5.metric("Volatility", volatility_label(latest_vol))

a1, a2, a3, a4 = st.columns(4)
a1.metric("Best Model", model_bundle["best_model_name"])
a2.metric("Accuracy", f"{model_bundle['best_metrics']['accuracy'] * 100:.2f}%")
a3.metric("Historical Success", f"{backtest_stats['trade_win_rate'] * 100:.2f}%")
a4.metric("Market Regime", latest_regime)

if latest_vol > volatility_threshold:
    st.warning(
        f"⚠️ 市場波動偏高。當前年化波動率約 {latest_vol:.2f}，已超過你設定的 {volatility_threshold:.2f}，策略會傾向避免進場。"
    )

st.markdown(
    f"""
    <div class="small-note">
    預測周期：<b>{timeframe}</b>（約 {timeframe_to_horizon(timeframe)} 個交易日）｜
    使用 TimeSeriesSplit，避免資料洩漏｜
    特徵全部來自過去與當前資料，不偷看未來。
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Live Charts", "Model Performance", "Backtesting", "Signals Dashboard"]
)

with tab1:
    st.plotly_chart(make_price_chart(price_df, ticker, show_last_n), use_container_width=True)

with tab2:
    c1, c2 = st.columns([0.9, 1.1])

    with c1:
        st.markdown("### Model Comparison")
        st.dataframe(
            model_bundle["comparison_df"].style.format(
                {
                    "cv_accuracy": "{:.2%}",
                    "precision": "{:.2%}",
                    "recall": "{:.2%}",
                    "f1": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("### Confusion Matrix")
        st.plotly_chart(
            make_confusion_matrix_figure(model_bundle["best_metrics"]["confusion_matrix"]),
            use_container_width=True,
        )

    with c2:
        st.markdown("### Feature Importance")
        st.plotly_chart(
            make_feature_importance_figure(model_bundle["feature_importance"]),
            use_container_width=True,
        )

        st.markdown("### Performance Snapshot")
        perf = pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                "Value": [
                    model_bundle["best_metrics"]["accuracy"],
                    model_bundle["best_metrics"]["precision"],
                    model_bundle["best_metrics"]["recall"],
                    model_bundle["best_metrics"]["f1"],
                ],
            }
        )
        st.dataframe(perf.style.format({"Value": "{:.2%}"}), use_container_width=True)

with tab3:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Strategy Return", f"{backtest_stats['total_strategy_return'] * 100:.2f}%")
    b2.metric("Buy & Hold", f"{backtest_stats['total_buy_hold_return'] * 100:.2f}%")
    b3.metric("Max Drawdown", f"{backtest_stats['max_drawdown'] * 100:.2f}%")
    b4.metric("Trades Taken", f"{backtest_stats['trade_count']}")

    st.plotly_chart(make_backtest_figure(backtest_df), use_container_width=True)
    st.plotly_chart(make_drawdown_figure(backtest_df), use_container_width=True)

    st.markdown("### Paper Trading Simulation")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Initial Capital", f"${paper_stats['initial_capital']:,.0f}")
    p2.metric("Final Capital", f"${paper_stats['final_capital']:,.0f}")
    p3.metric("Net Profit", f"${paper_stats['net_profit']:,.0f}")
    p4.metric("Return", f"{paper_stats['return_pct'] * 100:.2f}%")

with tab4:
    st.markdown("### Latest Trading Signals")
    recent = backtest_df.tail(20).copy()
    recent["Date"] = recent["Date"].dt.date
    recent["Prob_UP"] = recent["proba_up"]
    recent["Confidence"] = recent["confidence"]

    show_cols = [
        "Date",
        "Close",
        "Regime",
        "Volatility_20",
        "Prob_UP",
        "Confidence",
        "signal",
        "future_return",
    ]
    st.dataframe(
        recent[show_cols].style.format(
            {
                "Close": "${:,.2f}",
                "Volatility_20": "{:.2f}",
                "Prob_UP": "{:.2%}",
                "Confidence": "{:.2%}",
                "future_return": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

# ---------- Model Save / Download ----------
st.subheader("Model Export")

model_artifact = {
    "ticker": ticker,
    "timeframe": timeframe,
    "feature_cols": feature_cols,
    "best_model_name": model_bundle["best_model_name"],
    "model": model_bundle["model"],
    "metadata": {
        "accuracy": model_bundle["best_metrics"]["accuracy"],
        "confidence_threshold": confidence_threshold,
        "volatility_threshold": volatility_threshold,
    },
}

binary_model = pickle.dumps(model_artifact)

st.download_button(
    label="Download trained model (.pkl)",
    data=binary_model,
    file_name=f"{ticker.lower()}_{timeframe.lower()}_model.pkl",
    mime="application/octet-stream",
)

st.caption("This dashboard is for education and research only, not financial advice.")
